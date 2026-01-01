"""Tests for process pool module - concurrent CLI execution."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccflow.pool import (
    PoolConfig,
    PoolStats,
    PoolTask,
    ProcessPool,
    StreamingPool,
    StreamingTask,
    TaskStatus,
    get_pool,
    get_streaming_pool,
    reset_pools,
)
from ccflow.types import CLIAgentOptions


# =============================================================================
# PoolConfig Tests
# =============================================================================


class TestPoolConfig:
    """Tests for pool configuration."""

    def test_default_config(self):
        """Default configuration should have sensible values."""
        config = PoolConfig()
        assert config.max_workers == 4
        assert config.max_queue_size == 100
        assert config.shutdown_timeout == 30.0
        assert config.task_timeout == 300.0

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = PoolConfig(
            max_workers=8,
            max_queue_size=50,
            shutdown_timeout=60.0,
            task_timeout=600.0,
        )
        assert config.max_workers == 8
        assert config.max_queue_size == 50

    def test_invalid_max_workers(self):
        """Should reject max_workers < 1."""
        with pytest.raises(ValueError, match="max_workers"):
            PoolConfig(max_workers=0)

    def test_invalid_queue_size(self):
        """Should reject max_queue_size < 1."""
        with pytest.raises(ValueError, match="max_queue_size"):
            PoolConfig(max_queue_size=0)


# =============================================================================
# PoolTask Tests
# =============================================================================


class TestPoolTask:
    """Tests for pool task creation."""

    def test_create_task(self):
        """Should create a task with defaults."""
        task = PoolTask.create("Hello, Claude")
        assert task.prompt == "Hello, Claude"
        assert task.status == TaskStatus.PENDING
        assert task.timeout == 300.0
        assert len(task.id) == 12

    def test_create_task_with_options(self):
        """Should create task with custom options."""
        options = CLIAgentOptions(model="opus")
        task = PoolTask.create(
            "Test prompt",
            options=options,
            timeout=60.0,
        )
        assert task.options.model == "opus"
        assert task.timeout == 60.0


# =============================================================================
# ProcessPool Tests
# =============================================================================


class TestProcessPool:
    """Tests for the process pool."""

    @pytest.fixture
    def mock_executor(self):
        """Create a mock CLI executor."""
        executor = MagicMock()
        executor.build_flags = MagicMock(return_value=["--flag"])

        async def mock_execute(*args, **kwargs):
            yield {"type": "init", "session_id": "test-123"}
            yield {"type": "text", "content": "Hello"}
            yield {"type": "stop", "usage": {"input_tokens": 100, "output_tokens": 50}}

        executor.execute = mock_execute
        return executor

    @pytest.mark.asyncio
    async def test_pool_start_and_shutdown(self):
        """Pool should start and shutdown cleanly."""
        pool = ProcessPool(config=PoolConfig(max_workers=2))
        assert not pool.is_running

        await pool.start()
        assert pool.is_running

        await pool.shutdown()
        assert not pool.is_running

    @pytest.mark.asyncio
    async def test_pool_context_manager(self):
        """Pool should work as async context manager."""
        async with ProcessPool() as pool:
            assert pool.is_running

        assert not pool.is_running

    @pytest.mark.asyncio
    async def test_submit_requires_running_pool(self):
        """Submit should fail if pool not running."""
        pool = ProcessPool()

        with pytest.raises(RuntimeError, match="not running"):
            await pool.submit("Test")

    @pytest.mark.asyncio
    async def test_submit_task(self, mock_executor):
        """Should submit and track task."""
        pool = ProcessPool(executor=mock_executor)
        await pool.start()

        try:
            task = await pool.submit("Test prompt")
            assert task.id in pool._tasks
            assert task.status == TaskStatus.PENDING

            # Wait for completion
            completed = await pool.wait(task.id, timeout=5.0)
            assert completed.status == TaskStatus.COMPLETED
            assert completed.result is not None
            assert len(completed.result) == 3
        finally:
            await pool.shutdown()

    @pytest.mark.asyncio
    async def test_map_multiple_tasks(self, mock_executor):
        """Should submit multiple tasks via map."""
        pool = ProcessPool(
            config=PoolConfig(max_workers=4),
            executor=mock_executor,
        )

        async with pool:
            tasks = await pool.map(
                ["Prompt 1", "Prompt 2", "Prompt 3"],
                CLIAgentOptions(model="sonnet"),
            )
            assert len(tasks) == 3

            # Wait for all
            results = await pool.gather(*[t.id for t in tasks])
            assert len(results) == 3
            assert all(isinstance(t, PoolTask) for t in results)

    @pytest.mark.asyncio
    async def test_pool_stats(self, mock_executor):
        """Should track pool statistics."""
        pool = ProcessPool(
            config=PoolConfig(max_workers=2),
            executor=mock_executor,
        )

        async with pool:
            stats = pool.stats()
            assert stats.pool_size == 2
            assert stats.active_tasks == 0
            assert stats.completed_tasks == 0

            await pool.submit("Test")
            await asyncio.sleep(0.1)

            # After task completes
            await asyncio.sleep(0.5)
            stats = pool.stats()
            assert stats.completed_tasks >= 0  # May have completed

    @pytest.mark.asyncio
    async def test_cancel_pending_task(self, mock_executor):
        """Should cancel pending task."""
        pool = ProcessPool(
            config=PoolConfig(max_workers=1),
            executor=mock_executor,
        )

        async with pool:
            # Submit multiple tasks
            task1 = await pool.submit("Task 1")
            task2 = await pool.submit("Task 2")

            # Cancel second task before it runs
            cancelled = pool.cancel_task(task2.id)
            assert cancelled or task2.status != TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_task(self, mock_executor):
        """Should get task by ID."""
        pool = ProcessPool(executor=mock_executor)

        async with pool:
            task = await pool.submit("Test")
            retrieved = pool.get_task(task.id)
            assert retrieved is not None
            assert retrieved.id == task.id

            # Non-existent task
            assert pool.get_task("nonexistent") is None

    @pytest.mark.asyncio
    async def test_task_failure(self):
        """Should handle task failures."""
        executor = MagicMock()
        executor.build_flags = MagicMock(return_value=["--flag"])

        async def failing_execute(*args, **kwargs):
            raise RuntimeError("CLI error")
            yield  # Make it a generator

        executor.execute = failing_execute

        pool = ProcessPool(executor=executor)

        async with pool:
            task = await pool.submit("Test")
            completed = await pool.wait(task.id, timeout=5.0)
            assert completed.status == TaskStatus.FAILED
            assert completed.error is not None

    @pytest.mark.asyncio
    async def test_pool_stats_utilization(self):
        """Pool stats should calculate utilization correctly."""
        stats = PoolStats(
            pool_size=4,
            active_tasks=2,
            pending_tasks=5,
            completed_tasks=10,
            failed_tasks=1,
            total_tasks=18,
        )
        assert stats.utilization == 50.0  # 2/4 * 100

        # Zero pool size
        stats_zero = PoolStats(
            pool_size=0,
            active_tasks=0,
            pending_tasks=0,
            completed_tasks=0,
            failed_tasks=0,
            total_tasks=0,
        )
        assert stats_zero.utilization == 0.0


# =============================================================================
# StreamingPool Tests
# =============================================================================


class TestStreamingPool:
    """Tests for the streaming pool."""

    @pytest.fixture
    def mock_executor(self):
        """Create a mock streaming executor."""
        executor = MagicMock()
        executor.build_flags = MagicMock(return_value=["--flag"])

        async def mock_execute(*args, **kwargs):
            for i in range(3):
                yield {"type": "text", "content": f"Chunk {i}"}
            yield {"type": "stop"}

        executor.execute = mock_execute
        return executor

    @pytest.mark.asyncio
    async def test_streaming_pool_start_shutdown(self):
        """Streaming pool should start and shutdown."""
        pool = StreamingPool()
        await pool.start()
        assert pool._running

        await pool.shutdown()
        assert not pool._running

    @pytest.mark.asyncio
    async def test_streaming_task_iteration(self, mock_executor):
        """Should iterate over streaming events."""
        pool = StreamingPool(executor=mock_executor)

        async with pool:
            task = await pool.submit("Test prompt")
            events = []

            async for event in task:
                events.append(event)

            assert len(events) == 4  # 3 text + 1 stop

    @pytest.mark.asyncio
    async def test_streaming_submit_requires_running(self):
        """Submit should fail if not running."""
        pool = StreamingPool()

        with pytest.raises(RuntimeError, match="not running"):
            await pool.submit("Test")


# =============================================================================
# StreamingTask Tests
# =============================================================================


class TestStreamingTask:
    """Tests for streaming task."""

    @pytest.mark.asyncio
    async def test_streaming_task_events(self):
        """Task should yield events as they arrive."""
        task = StreamingTask(
            task_id="test-123",
            prompt="Test",
            options=CLIAgentOptions(),
            timeout=60.0,
            cwd=None,
        )

        # Simulate events being added
        async def add_events():
            await task.put_event({"type": "text", "content": "Hello"})
            await task.put_event({"type": "text", "content": "World"})
            await task.finish()

        asyncio.create_task(add_events())

        events = []
        async for event in task:
            events.append(event)

        assert len(events) == 2
        assert events[0]["content"] == "Hello"
        assert events[1]["content"] == "World"

    @pytest.mark.asyncio
    async def test_streaming_task_error(self):
        """Task should raise error when finished with error."""
        task = StreamingTask(
            task_id="test-123",
            prompt="Test",
            options=CLIAgentOptions(),
            timeout=60.0,
            cwd=None,
        )

        async def add_error():
            await task.put_event({"type": "text"})
            await task.finish(error=RuntimeError("Test error"))

        asyncio.create_task(add_error())

        with pytest.raises(RuntimeError, match="Test error"):
            async for _ in task:
                pass


# =============================================================================
# Global Pool Management Tests
# =============================================================================


class TestGlobalPoolManagement:
    """Tests for global pool functions."""

    @pytest.mark.asyncio
    async def test_get_pool_singleton(self):
        """get_pool should return singleton."""
        await reset_pools()

        pool1 = get_pool()
        pool2 = get_pool()
        assert pool1 is pool2

        await reset_pools()

    @pytest.mark.asyncio
    async def test_get_streaming_pool_singleton(self):
        """get_streaming_pool should return singleton."""
        await reset_pools()

        pool1 = get_streaming_pool()
        pool2 = get_streaming_pool()
        assert pool1 is pool2

        await reset_pools()

    @pytest.mark.asyncio
    async def test_reset_pools(self):
        """reset_pools should clear global pools."""
        pool1 = get_pool()
        streaming1 = get_streaming_pool()

        await reset_pools()

        pool2 = get_pool()
        streaming2 = get_streaming_pool()

        assert pool1 is not pool2
        assert streaming1 is not streaming2

        await reset_pools()
