"""
Subagent coordination for ccflow.

Provides SubagentCoordinator for parallel task execution with
context isolation and result aggregation.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from .agent import Agent, AgentRegistry, get_agent_registry

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .types import CLIAgentOptions, Message

logger = structlog.get_logger(__name__)


@dataclass
class SubagentTask:
    """Represents a subagent task.

    Attributes:
        task_id: Unique task identifier
        agent_name: Name of the agent to use
        prompt: Task prompt
        context: Optional context data
        status: Task status ("pending", "running", "completed", "failed")
        result: Task result when completed
        error: Error message if failed
    """

    task_id: str
    agent_name: str
    prompt: str
    context: dict[str, Any] | None = None
    status: str = "pending"
    result: str | None = None
    error: str | None = None


class SubagentCoordinator:
    """Coordinates parallel subagent execution.

    Manages spawning and running multiple subagents in parallel,
    with concurrency control and result aggregation.

    Example:
        >>> coordinator = SubagentCoordinator()
        >>> results = await coordinator.parallel([
        ...     ("code-reviewer", "Review auth.py"),
        ...     ("security-auditor", "Check for vulnerabilities"),
        ... ])
        >>> for result in results:
        ...     print(result)
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        max_concurrent: int = 10,
        default_options: CLIAgentOptions | None = None,
    ) -> None:
        """Initialize coordinator.

        Args:
            registry: Agent registry to use
            max_concurrent: Maximum concurrent subagents
            default_options: Default options for subagents
        """
        self._registry = registry or get_agent_registry()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._default_options = default_options
        self._active_tasks: dict[str, asyncio.Task] = {}
        self._task_results: dict[str, SubagentTask] = {}

    @property
    def max_concurrent(self) -> int:
        """Get maximum concurrent subagents."""
        return self._semaphore._value

    async def spawn(
        self,
        agent_name: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[Message]:
        """Spawn single subagent.

        Runs the specified agent with the given task prompt,
        respecting concurrency limits.

        Args:
            agent_name: Name of the agent to spawn
            task: Task prompt
            context: Optional context data

        Yields:
            Message events from the agent
        """
        task_id = str(uuid.uuid4())[:8]

        async with self._semaphore:
            logger.debug(
                "spawning_subagent",
                agent=agent_name,
                task_id=task_id,
            )

            try:
                agent = Agent(
                    agent_name,
                    parent_options=self._default_options,
                    registry=self._registry,
                )

                async for msg in agent.execute(task, context):
                    yield msg

                logger.debug(
                    "subagent_completed",
                    agent=agent_name,
                    task_id=task_id,
                )

            except Exception as e:
                logger.error(
                    "subagent_failed",
                    agent=agent_name,
                    task_id=task_id,
                    error=str(e),
                )
                raise

    async def spawn_simple(
        self,
        agent_name: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Spawn subagent and return aggregated result.

        Runs the agent and collects all text output into
        a single string.

        Args:
            agent_name: Name of the agent
            task: Task prompt
            context: Optional context data

        Returns:
            Aggregated text response
        """
        agent = Agent(
            agent_name,
            parent_options=self._default_options,
            registry=self._registry,
        )

        async with self._semaphore:
            return await agent.execute_simple(task, context)

    async def parallel(
        self,
        tasks: list[tuple[str, str]],
        context: dict[str, Any] | None = None,
    ) -> list[str]:
        """Execute multiple subagents in parallel.

        Runs all tasks concurrently (up to max_concurrent limit)
        and returns results in the same order as input.

        Args:
            tasks: List of (agent_name, prompt) tuples
            context: Optional shared context for all agents

        Returns:
            List of text results in same order as input
        """

        async def run_one(agent_name: str, prompt: str) -> str:
            return await self.spawn_simple(agent_name, prompt, context)

        # Create all tasks
        coros = [run_one(name, prompt) for name, prompt in tasks]

        # Run in parallel and gather results
        results = await asyncio.gather(*coros, return_exceptions=True)

        # Convert exceptions to error strings
        processed_results: list[str] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.warning(
                    "parallel_task_failed",
                    agent=tasks[i][0],
                    error=str(result),
                )
                processed_results.append(f"Error: {result}")
            else:
                processed_results.append(result)

        return processed_results

    async def spawn_background(
        self,
        agent_name: str,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> str:
        """Spawn subagent in background.

        Creates a background task for the agent execution,
        returning a task ID for later retrieval.

        Args:
            agent_name: Name of the agent
            task: Task prompt
            context: Optional context data

        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())[:8]

        task_record = SubagentTask(
            task_id=task_id,
            agent_name=agent_name,
            prompt=task,
            context=context,
            status="pending",
        )
        self._task_results[task_id] = task_record

        async def background_run():
            try:
                task_record.status = "running"
                result = await self.spawn_simple(agent_name, task, context)
                task_record.result = result
                task_record.status = "completed"
            except Exception as e:
                task_record.error = str(e)
                task_record.status = "failed"

        async_task = asyncio.create_task(background_run())
        self._active_tasks[task_id] = async_task

        logger.debug(
            "spawned_background_task",
            task_id=task_id,
            agent=agent_name,
        )

        return task_id

    async def get_task_result(
        self,
        task_id: str,
        wait: bool = True,
        timeout: float | None = None,
    ) -> SubagentTask | None:
        """Get result of a background task.

        Args:
            task_id: Task ID from spawn_background
            wait: Whether to wait for completion
            timeout: Maximum time to wait

        Returns:
            SubagentTask with result, or None if not found
        """
        if task_id not in self._task_results:
            return None

        task_record = self._task_results[task_id]

        if wait and task_id in self._active_tasks:
            async_task = self._active_tasks[task_id]
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(async_task, timeout=timeout)

        return task_record

    async def gather(self, timeout: float | None = None) -> list[SubagentTask]:
        """Wait for all active background tasks to complete.

        Args:
            timeout: Maximum time to wait

        Returns:
            List of all task results
        """
        if self._active_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._active_tasks.values(), return_exceptions=True),
                    timeout=timeout,
                )
            except TimeoutError:
                logger.warning("gather_timeout", pending=len(self._active_tasks))

        return list(self._task_results.values())

    def list_active(self) -> list[str]:
        """List active task IDs.

        Returns:
            List of task IDs that are still running
        """
        return [
            task_id
            for task_id, task_record in self._task_results.items()
            if task_record.status in ("pending", "running")
        ]

    def cancel(self, task_id: str) -> bool:
        """Cancel a background task.

        Args:
            task_id: Task ID to cancel

        Returns:
            True if task was cancelled
        """
        if task_id not in self._active_tasks:
            return False

        async_task = self._active_tasks[task_id]
        async_task.cancel()

        if task_id in self._task_results:
            self._task_results[task_id].status = "cancelled"

        logger.debug("task_cancelled", task_id=task_id)
        return True

    def cancel_all(self) -> int:
        """Cancel all active tasks.

        Returns:
            Number of tasks cancelled
        """
        count = 0
        for task_id in list(self._active_tasks.keys()):
            if self.cancel(task_id):
                count += 1
        return count


# Global coordinator singleton
_global_coordinator: SubagentCoordinator | None = None


def get_subagent_coordinator() -> SubagentCoordinator:
    """Get the global subagent coordinator singleton.

    Returns:
        The global SubagentCoordinator instance
    """
    global _global_coordinator
    if _global_coordinator is None:
        _global_coordinator = SubagentCoordinator()
    return _global_coordinator


def reset_subagent_coordinator() -> None:
    """Reset the global coordinator (for testing)."""
    global _global_coordinator
    _global_coordinator = None
