"""
Process Pool - Concurrent CLI execution with configurable pool size.

Enables parallel execution of multiple Claude CLI queries with:
- Configurable pool size (max concurrent processes)
- Process lifecycle management
- Queue-based scheduling
- Graceful shutdown
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import structlog

from ccflow.executor import CLIExecutor
from ccflow.reliability import get_correlation_id, set_correlation_id
from ccflow.types import CLIAgentOptions

logger = structlog.get_logger(__name__)


class TaskStatus(str, Enum):
    """Status of a pool task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PoolTask:
    """A task submitted to the process pool."""

    id: str
    prompt: str
    options: CLIAgentOptions
    timeout: float = 300.0
    cwd: Path | str | None = None
    correlation_id: str | None = None

    # Runtime state
    status: TaskStatus = field(default=TaskStatus.PENDING)
    result: list[dict] | None = field(default=None)
    error: Exception | None = field(default=None)
    started_at: float | None = field(default=None)
    completed_at: float | None = field(default=None)

    @classmethod
    def create(
        cls,
        prompt: str,
        options: CLIAgentOptions | None = None,
        timeout: float = 300.0,
        cwd: Path | str | None = None,
        correlation_id: str | None = None,
    ) -> PoolTask:
        """Create a new pool task."""
        return cls(
            id=uuid4().hex[:12],
            prompt=prompt,
            options=options or CLIAgentOptions(),
            timeout=timeout,
            cwd=cwd,
            correlation_id=correlation_id or get_correlation_id(),
        )


@dataclass
class PoolStats:
    """Statistics about the process pool."""

    pool_size: int
    active_tasks: int
    pending_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_tasks: int

    @property
    def utilization(self) -> float:
        """Current utilization as a percentage."""
        if self.pool_size == 0:
            return 0.0
        return (self.active_tasks / self.pool_size) * 100


@dataclass
class PoolConfig:
    """Configuration for the process pool."""

    # Pool size
    max_workers: int = 4

    # Queue settings
    max_queue_size: int = 100

    # Timeouts
    shutdown_timeout: float = 30.0
    task_timeout: float = 300.0

    # Retry on failure
    retry_failed: bool = False
    max_retries: int = 2

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be at least 1")


class ProcessPool:
    """Pool of CLI executor processes for concurrent execution.

    Manages a pool of worker tasks that consume from a shared queue,
    enabling parallel execution of multiple Claude queries.

    Example:
        >>> pool = ProcessPool(config=PoolConfig(max_workers=4))
        >>> await pool.start()
        >>>
        >>> # Submit tasks
        >>> task1 = await pool.submit("Analyze code", options1)
        >>> task2 = await pool.submit("Review PR", options2)
        >>>
        >>> # Wait for results
        >>> result1 = await pool.wait(task1.id)
        >>> result2 = await pool.wait(task2.id)
        >>>
        >>> await pool.shutdown()

    With context manager:
        >>> async with ProcessPool() as pool:
        ...     tasks = await pool.map(prompts, options)
        ...     results = await pool.gather(*[t.id for t in tasks])
    """

    def __init__(
        self,
        config: PoolConfig | None = None,
        executor: CLIExecutor | None = None,
    ) -> None:
        """Initialize process pool.

        Args:
            config: Pool configuration
            executor: Custom executor (creates new one if None)
        """
        self.config = config or PoolConfig()
        self._executor = executor

        # Task management
        self._queue: asyncio.Queue[PoolTask] = asyncio.Queue(
            maxsize=self.config.max_queue_size
        )
        self._tasks: dict[str, PoolTask] = {}
        self._task_events: dict[str, asyncio.Event] = {}

        # Worker management
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._shutdown_event = asyncio.Event()

        # Statistics
        self._completed_count = 0
        self._failed_count = 0

    @property
    def executor(self) -> CLIExecutor:
        """Get or create the CLI executor."""
        if self._executor is None:
            self._executor = CLIExecutor()
        return self._executor

    @property
    def is_running(self) -> bool:
        """Check if pool is running."""
        return self._running

    async def start(self) -> None:
        """Start the process pool workers."""
        if self._running:
            return

        self._running = True
        self._shutdown_event.clear()

        # Start workers
        for i in range(self.config.max_workers):
            worker = asyncio.create_task(
                self._worker_loop(i),
                name=f"pool-worker-{i}",
            )
            self._workers.append(worker)

        logger.info(
            "process_pool_started",
            max_workers=self.config.max_workers,
            max_queue_size=self.config.max_queue_size,
        )

    async def shutdown(self, wait: bool = True) -> None:
        """Shutdown the process pool.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        if not self._running:
            return

        self._running = False
        self._shutdown_event.set()

        if wait and self._tasks:
            # Wait for pending tasks with timeout
            pending = [
                task_id
                for task_id, task in self._tasks.items()
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
            ]

            if pending:
                logger.info(
                    "pool_shutdown_waiting",
                    pending_tasks=len(pending),
                    timeout=self.config.shutdown_timeout,
                )

                try:
                    await asyncio.wait_for(
                        asyncio.gather(
                            *[self._task_events[tid].wait() for tid in pending],
                            return_exceptions=True,
                        ),
                        timeout=self.config.shutdown_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "pool_shutdown_timeout",
                        remaining_tasks=len(
                            [
                                t
                                for t in self._tasks.values()
                                if t.status
                                in (TaskStatus.PENDING, TaskStatus.RUNNING)
                            ]
                        ),
                    )

        # Cancel workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info(
            "process_pool_shutdown",
            completed=self._completed_count,
            failed=self._failed_count,
        )

    async def submit(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
        timeout: float | None = None,
        cwd: Path | str | None = None,
    ) -> PoolTask:
        """Submit a task to the pool.

        Args:
            prompt: The prompt to execute
            options: CLI options
            timeout: Task timeout (uses pool default if None)
            cwd: Working directory

        Returns:
            The submitted task

        Raises:
            RuntimeError: If pool is not running
            asyncio.QueueFull: If queue is full
        """
        if not self._running:
            raise RuntimeError("Pool is not running. Call start() first.")

        task = PoolTask.create(
            prompt=prompt,
            options=options,
            timeout=timeout or self.config.task_timeout,
            cwd=cwd,
        )

        self._tasks[task.id] = task
        self._task_events[task.id] = asyncio.Event()

        try:
            self._queue.put_nowait(task)
        except asyncio.QueueFull:
            del self._tasks[task.id]
            del self._task_events[task.id]
            raise

        logger.debug(
            "task_submitted",
            task_id=task.id,
            queue_size=self._queue.qsize(),
        )

        return task

    async def submit_nowait(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
        timeout: float | None = None,
        cwd: Path | str | None = None,
    ) -> PoolTask | None:
        """Submit a task without waiting if queue is full.

        Returns:
            The submitted task, or None if queue is full
        """
        try:
            return await self.submit(prompt, options, timeout, cwd)
        except asyncio.QueueFull:
            return None

    async def wait(self, task_id: str, timeout: float | None = None) -> PoolTask:
        """Wait for a task to complete.

        Args:
            task_id: ID of the task to wait for
            timeout: Maximum time to wait

        Returns:
            The completed task

        Raises:
            KeyError: If task not found
            asyncio.TimeoutError: If timeout exceeded
        """
        if task_id not in self._tasks:
            raise KeyError(f"Task {task_id} not found")

        event = self._task_events[task_id]

        if timeout:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        else:
            await event.wait()

        return self._tasks[task_id]

    async def map(
        self,
        prompts: list[str],
        options: CLIAgentOptions | None = None,
        timeout: float | None = None,
    ) -> list[PoolTask]:
        """Submit multiple tasks.

        Args:
            prompts: List of prompts to execute
            options: Shared options for all tasks
            timeout: Task timeout

        Returns:
            List of submitted tasks
        """
        tasks = []
        for prompt in prompts:
            task = await self.submit(prompt, options, timeout)
            tasks.append(task)
        return tasks

    async def gather(
        self,
        *task_ids: str,
        return_exceptions: bool = False,
    ) -> list[PoolTask | Exception]:
        """Wait for multiple tasks to complete.

        Args:
            task_ids: IDs of tasks to wait for
            return_exceptions: If True, return exceptions instead of raising

        Returns:
            List of completed tasks or exceptions
        """
        results: list[PoolTask | Exception] = []

        for task_id in task_ids:
            try:
                task = await self.wait(task_id)
                if task.error and not return_exceptions:
                    raise task.error
                results.append(task if not task.error else task.error)
            except Exception as e:
                if return_exceptions:
                    results.append(e)
                else:
                    raise

        return results

    def get_task(self, task_id: str) -> PoolTask | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if cancelled, False if not found or already running
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False

        if task.status != TaskStatus.PENDING:
            return False

        task.status = TaskStatus.CANCELLED
        self._task_events[task_id].set()

        logger.debug("task_cancelled", task_id=task_id)
        return True

    def stats(self) -> PoolStats:
        """Get current pool statistics."""
        active = sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING
        )
        pending = sum(
            1 for t in self._tasks.values() if t.status == TaskStatus.PENDING
        )

        return PoolStats(
            pool_size=self.config.max_workers,
            active_tasks=active,
            pending_tasks=pending,
            completed_tasks=self._completed_count,
            failed_tasks=self._failed_count,
            total_tasks=len(self._tasks),
        )

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop that processes tasks from the queue."""
        logger.debug("worker_started", worker_id=worker_id)

        while self._running or not self._queue.empty():
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    if self._shutdown_event.is_set():
                        break
                    continue

                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    self._queue.task_done()
                    continue

                # Execute task
                await self._execute_task(task, worker_id)
                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(
                    "worker_error",
                    worker_id=worker_id,
                    error=str(e),
                )

        logger.debug("worker_stopped", worker_id=worker_id)

    async def _execute_task(self, task: PoolTask, worker_id: int) -> None:
        """Execute a single task."""
        import time

        task.status = TaskStatus.RUNNING
        task.started_at = time.time()

        # Set correlation ID for this task
        set_correlation_id(task.correlation_id)

        logger.debug(
            "task_started",
            task_id=task.id,
            worker_id=worker_id,
        )

        try:
            # Build flags and execute
            flags = self.executor.build_flags(task.options)
            events: list[dict] = []

            async for event in self.executor.execute(
                prompt=task.prompt,
                flags=flags,
                timeout=task.timeout,
                cwd=task.cwd,
            ):
                events.append(event)

            task.result = events
            task.status = TaskStatus.COMPLETED
            self._completed_count += 1

            logger.debug(
                "task_completed",
                task_id=task.id,
                worker_id=worker_id,
                events_count=len(events),
            )

        except Exception as e:
            task.error = e
            task.status = TaskStatus.FAILED
            self._failed_count += 1

            logger.warning(
                "task_failed",
                task_id=task.id,
                worker_id=worker_id,
                error=str(e),
            )

        finally:
            task.completed_at = time.time()
            self._task_events[task.id].set()

    async def __aenter__(self) -> ProcessPool:
        """Enter async context."""
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit async context."""
        await self.shutdown()


# =============================================================================
# Streaming Pool
# =============================================================================


class StreamingTask:
    """A task that yields events as they arrive."""

    def __init__(
        self,
        task_id: str,
        prompt: str,
        options: CLIAgentOptions,
        timeout: float,
        cwd: Path | str | None,
    ) -> None:
        self.id = task_id
        self.prompt = prompt
        self.options = options
        self.timeout = timeout
        self.cwd = cwd

        self._queue: asyncio.Queue[dict | None] = asyncio.Queue()
        self._error: Exception | None = None
        self._done = False

    async def put_event(self, event: dict) -> None:
        """Add an event to the stream."""
        await self._queue.put(event)

    async def finish(self, error: Exception | None = None) -> None:
        """Mark the stream as finished."""
        self._error = error
        self._done = True
        await self._queue.put(None)

    async def __aiter__(self) -> AsyncIterator[dict]:
        """Iterate over events as they arrive."""
        while True:
            event = await self._queue.get()
            if event is None:
                if self._error:
                    raise self._error
                break
            yield event


class StreamingPool:
    """Pool that yields events as they stream from CLI.

    Unlike ProcessPool which waits for complete results,
    StreamingPool yields events in real-time.

    Example:
        >>> async with StreamingPool() as pool:
        ...     task = await pool.submit("Analyze this code")
        ...     async for event in task:
        ...         print(event)
    """

    def __init__(
        self,
        config: PoolConfig | None = None,
        executor: CLIExecutor | None = None,
    ) -> None:
        self.config = config or PoolConfig()
        self._executor = executor

        self._tasks: dict[str, StreamingTask] = {}
        self._workers: list[asyncio.Task] = []
        self._running = False
        self._semaphore: asyncio.Semaphore | None = None

    @property
    def executor(self) -> CLIExecutor:
        """Get or create the CLI executor."""
        if self._executor is None:
            self._executor = CLIExecutor()
        return self._executor

    async def start(self) -> None:
        """Start the streaming pool."""
        if self._running:
            return

        self._running = True
        self._semaphore = asyncio.Semaphore(self.config.max_workers)

        logger.info(
            "streaming_pool_started",
            max_workers=self.config.max_workers,
        )

    async def shutdown(self) -> None:
        """Shutdown the streaming pool."""
        if not self._running:
            return

        self._running = False

        # Cancel all running workers
        for worker in self._workers:
            worker.cancel()

        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

        logger.info("streaming_pool_shutdown")

    async def submit(
        self,
        prompt: str,
        options: CLIAgentOptions | None = None,
        timeout: float | None = None,
        cwd: Path | str | None = None,
    ) -> StreamingTask:
        """Submit a streaming task.

        Returns a StreamingTask that can be iterated for events.
        """
        if not self._running:
            raise RuntimeError("Pool is not running. Call start() first.")

        task = StreamingTask(
            task_id=uuid4().hex[:12],
            prompt=prompt,
            options=options or CLIAgentOptions(),
            timeout=timeout or self.config.task_timeout,
            cwd=cwd,
        )

        self._tasks[task.id] = task

        # Start worker for this task
        worker = asyncio.create_task(
            self._execute_streaming(task),
            name=f"streaming-{task.id}",
        )
        self._workers.append(worker)

        return task

    async def _execute_streaming(self, task: StreamingTask) -> None:
        """Execute a streaming task."""
        if self._semaphore is None:
            return

        async with self._semaphore:
            try:
                flags = self.executor.build_flags(task.options)

                async for event in self.executor.execute(
                    prompt=task.prompt,
                    flags=flags,
                    timeout=task.timeout,
                    cwd=task.cwd,
                ):
                    await task.put_event(event)

                await task.finish()

            except Exception as e:
                await task.finish(error=e)

    async def __aenter__(self) -> StreamingPool:
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.shutdown()


# =============================================================================
# Module-level Pool Management
# =============================================================================

_default_pool: ProcessPool | None = None
_default_streaming_pool: StreamingPool | None = None


def get_pool(config: PoolConfig | None = None) -> ProcessPool:
    """Get or create the default process pool.

    Note: The pool must be started before use.
    """
    global _default_pool
    if _default_pool is None:
        _default_pool = ProcessPool(config=config)
    return _default_pool


def get_streaming_pool(config: PoolConfig | None = None) -> StreamingPool:
    """Get or create the default streaming pool.

    Note: The pool must be started before use.
    """
    global _default_streaming_pool
    if _default_streaming_pool is None:
        _default_streaming_pool = StreamingPool(config=config)
    return _default_streaming_pool


async def reset_pools() -> None:
    """Reset all global pools."""
    global _default_pool, _default_streaming_pool

    if _default_pool is not None:
        await _default_pool.shutdown()
        _default_pool = None

    if _default_streaming_pool is not None:
        await _default_streaming_pool.shutdown()
        _default_streaming_pool = None
