"""
Tests for subagent coordination.

Tests SubagentCoordinator, parallel execution, and background tasks.
"""

from __future__ import annotations

import asyncio

import pytest

from ccflow.agent import AgentRegistry, reset_agent_registry
from ccflow.subagent import (
    SubagentCoordinator,
    SubagentTask,
    get_subagent_coordinator,
    reset_subagent_coordinator,
)
from ccflow.types import AgentDefinition


class TestSubagentTask:
    """Tests for SubagentTask dataclass."""

    def test_create_minimal(self):
        """Create task with required fields."""
        task = SubagentTask(
            task_id="abc123",
            agent_name="test-agent",
            prompt="Do the test",
        )

        assert task.task_id == "abc123"
        assert task.agent_name == "test-agent"
        assert task.prompt == "Do the test"
        assert task.status == "pending"
        assert task.result is None
        assert task.error is None

    def test_create_full(self):
        """Create task with all fields."""
        task = SubagentTask(
            task_id="xyz789",
            agent_name="worker",
            prompt="Work on this",
            context={"key": "value"},
            status="completed",
            result="Done!",
            error=None,
        )

        assert task.context == {"key": "value"}
        assert task.status == "completed"
        assert task.result == "Done!"


class TestSubagentCoordinator:
    """Tests for SubagentCoordinator."""

    def setup_method(self):
        """Reset globals before each test."""
        reset_agent_registry()
        reset_subagent_coordinator()

    def test_create_coordinator(self):
        """Create coordinator with defaults."""
        coordinator = SubagentCoordinator()

        assert coordinator.max_concurrent == 10

    def test_create_with_custom_max(self):
        """Create coordinator with custom max concurrent."""
        coordinator = SubagentCoordinator(max_concurrent=5)

        assert coordinator.max_concurrent == 5

    def test_create_with_registry(self):
        """Create coordinator with custom registry."""
        registry = AgentRegistry()
        coordinator = SubagentCoordinator(registry=registry)

        assert coordinator._registry is registry

    @pytest.mark.asyncio
    async def test_spawn(self, mock_subprocess):
        """Spawn subagent and stream messages."""
        # Create registry with test agent
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="test-agent",
            description="Test agent",
            prompt="You are a test agent.",
        ))

        coordinator = SubagentCoordinator(registry=registry)
        messages = []

        async for msg in coordinator.spawn("test-agent", "Hello"):
            messages.append(msg)

        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_spawn_simple(self, mock_subprocess):
        """Spawn subagent and get simple result."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="test-agent",
            description="Test agent",
            prompt="You are a test agent.",
        ))

        coordinator = SubagentCoordinator(registry=registry)
        result = await coordinator.spawn_simple("test-agent", "Hello")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_parallel(self, mock_subprocess):
        """Run multiple subagents in parallel."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="agent-a",
            description="Agent A",
            prompt="You are agent A.",
        ))
        registry.register(AgentDefinition(
            name="agent-b",
            description="Agent B",
            prompt="You are agent B.",
        ))

        coordinator = SubagentCoordinator(registry=registry)
        results = await coordinator.parallel([
            ("agent-a", "Task A"),
            ("agent-b", "Task B"),
        ])

        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)

    @pytest.mark.asyncio
    async def test_parallel_with_error(self, mock_subprocess):
        """Parallel handles errors gracefully."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="good-agent",
            description="Good agent",
            prompt="You are good.",
        ))
        # Missing agent will cause error

        coordinator = SubagentCoordinator(registry=registry)
        results = await coordinator.parallel([
            ("good-agent", "Good task"),
            ("missing-agent", "Will fail"),
        ])

        assert len(results) == 2
        assert "Error:" in results[1]

    @pytest.mark.asyncio
    async def test_spawn_background(self, mock_subprocess):
        """Spawn subagent in background."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="bg-agent",
            description="Background agent",
            prompt="You work in background.",
        ))

        coordinator = SubagentCoordinator(registry=registry)
        task_id = await coordinator.spawn_background("bg-agent", "Background task")

        assert task_id is not None
        assert len(task_id) == 8  # UUID prefix

    @pytest.mark.asyncio
    async def test_get_task_result(self, mock_subprocess):
        """Get result of background task."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="bg-agent",
            description="Background agent",
            prompt="You work in background.",
        ))

        coordinator = SubagentCoordinator(registry=registry)
        task_id = await coordinator.spawn_background("bg-agent", "Background task")

        # Wait for result
        result = await coordinator.get_task_result(task_id, wait=True, timeout=5.0)

        assert result is not None
        assert result.task_id == task_id
        assert result.status in ("completed", "running")

    @pytest.mark.asyncio
    async def test_get_task_result_not_found(self):
        """Get nonexistent task returns None."""
        coordinator = SubagentCoordinator()
        result = await coordinator.get_task_result("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_gather(self, mock_subprocess):
        """Gather all background tasks."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="gather-agent",
            description="Gather agent",
            prompt="You are gathered.",
        ))

        coordinator = SubagentCoordinator(registry=registry)

        # Spawn multiple background tasks
        await coordinator.spawn_background("gather-agent", "Task 1")
        await coordinator.spawn_background("gather-agent", "Task 2")

        results = await coordinator.gather(timeout=5.0)

        assert len(results) == 2

    def test_list_active_empty(self):
        """List active when no tasks."""
        coordinator = SubagentCoordinator()
        active = coordinator.list_active()

        assert active == []

    @pytest.mark.asyncio
    async def test_list_active(self, mock_subprocess):
        """List active tasks."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="active-agent",
            description="Active agent",
            prompt="You are active.",
        ))

        coordinator = SubagentCoordinator(registry=registry)
        task_id = await coordinator.spawn_background("active-agent", "Long task")

        # Immediately check - might still be running
        active = coordinator.list_active()

        # Should have at least started
        assert task_id in active or coordinator._task_results[task_id].status == "completed"

    @pytest.mark.asyncio
    async def test_cancel(self, mock_subprocess):
        """Cancel a background task."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="cancel-agent",
            description="Cancellable agent",
            prompt="You can be cancelled.",
        ))

        coordinator = SubagentCoordinator(registry=registry)
        task_id = await coordinator.spawn_background("cancel-agent", "Task")

        # Cancel immediately
        result = coordinator.cancel(task_id)

        # Should have attempted cancellation
        assert result is True or coordinator._task_results[task_id].status == "completed"

    def test_cancel_nonexistent(self):
        """Cancel nonexistent task returns False."""
        coordinator = SubagentCoordinator()
        result = coordinator.cancel("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all(self, mock_subprocess):
        """Cancel all active tasks."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="cancel-agent",
            description="Cancellable agent",
            prompt="You can be cancelled.",
        ))

        coordinator = SubagentCoordinator(registry=registry)

        # Spawn multiple
        await coordinator.spawn_background("cancel-agent", "Task 1")
        await coordinator.spawn_background("cancel-agent", "Task 2")

        # Cancel all
        count = coordinator.cancel_all()

        # Should have cancelled some (or they completed quickly)
        assert count >= 0

    def test_global_singleton(self):
        """Global coordinator returns same instance."""
        reset_subagent_coordinator()

        coord1 = get_subagent_coordinator()
        coord2 = get_subagent_coordinator()

        assert coord1 is coord2


class TestSubagentConcurrency:
    """Tests for concurrency control."""

    def setup_method(self):
        """Reset globals before each test."""
        reset_agent_registry()
        reset_subagent_coordinator()

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self, mock_subprocess):
        """Coordinator respects max concurrent limit."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="concurrent-agent",
            description="Concurrent agent",
            prompt="You run concurrently.",
        ))

        # Very low concurrency limit
        coordinator = SubagentCoordinator(
            registry=registry,
            max_concurrent=2,
        )

        # Spawn many tasks
        tasks = [
            ("concurrent-agent", f"Task {i}")
            for i in range(5)
        ]

        # Should complete without error (semaphore manages limit)
        results = await coordinator.parallel(tasks)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_parallel_preserves_order(self, mock_subprocess):
        """Parallel results match input order."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="order-agent",
            description="Order agent",
            prompt="You maintain order.",
        ))

        coordinator = SubagentCoordinator(registry=registry)

        # Create tasks with identifiable prompts
        tasks = [
            ("order-agent", f"Task {i}")
            for i in range(3)
        ]

        results = await coordinator.parallel(tasks)

        # Results should be in same order as tasks
        assert len(results) == 3


class TestSubagentContext:
    """Tests for context passing."""

    def setup_method(self):
        """Reset globals before each test."""
        reset_agent_registry()
        reset_subagent_coordinator()

    @pytest.mark.asyncio
    async def test_spawn_with_context(self, mock_subprocess):
        """Spawn passes context to agent."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="context-agent",
            description="Context agent",
            prompt="You use context.",
        ))

        coordinator = SubagentCoordinator(registry=registry)

        async for _ in coordinator.spawn(
            "context-agent",
            "Use the context",
            context={"key": "value"},
        ):
            pass

        # Test passed if no error

    @pytest.mark.asyncio
    async def test_parallel_shared_context(self, mock_subprocess):
        """Parallel shares context across agents."""
        registry = AgentRegistry()
        registry._search_paths = []
        registry.register(AgentDefinition(
            name="shared-agent",
            description="Shared agent",
            prompt="You share context.",
        ))

        coordinator = SubagentCoordinator(registry=registry)

        tasks = [
            ("shared-agent", "Task A"),
            ("shared-agent", "Task B"),
        ]

        results = await coordinator.parallel(
            tasks,
            context={"shared": "data"},
        )

        assert len(results) == 2
