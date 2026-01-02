"""
Tests for hook system.

Tests HookEvent, HookContext, HookMatcher, and HookRegistry classes.
"""

from __future__ import annotations

import asyncio

import pytest

from ccflow.hooks import (
    HookCallback,
    HookContext,
    HookEvent,
    HookMatcher,
    HookRegistry,
    get_hook_registry,
    reset_hook_registry,
)


class TestHookEvent:
    """Tests for HookEvent enum."""

    def test_all_sdk_events_exist(self):
        """All SDK-compatible events are defined."""
        assert HookEvent.PRE_TOOL_USE == "pre_tool_use"
        assert HookEvent.POST_TOOL_USE == "post_tool_use"
        assert HookEvent.USER_PROMPT_SUBMIT == "user_prompt_submit"
        assert HookEvent.STOP == "stop"
        assert HookEvent.SUBAGENT_STOP == "subagent_stop"
        assert HookEvent.PRE_COMPACT == "pre_compact"

    def test_event_count(self):
        """Exactly 6 SDK-compatible events."""
        assert len(HookEvent) == 6


class TestHookContext:
    """Tests for HookContext dataclass."""

    def test_create_minimal(self):
        """Create context with required fields only."""
        ctx = HookContext(
            session_id="test-session",
            hook_event=HookEvent.PRE_TOOL_USE,
        )

        assert ctx.session_id == "test-session"
        assert ctx.hook_event == HookEvent.PRE_TOOL_USE
        assert ctx.tool_name is None
        assert ctx.tool_input is None
        assert ctx.metadata == {}

    def test_create_for_tool_use(self):
        """Create context for tool use hook."""
        ctx = HookContext(
            session_id="session-123",
            hook_event=HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
            tool_input={"command": "ls -la"},
        )

        assert ctx.tool_name == "Bash"
        assert ctx.tool_input == {"command": "ls -la"}

    def test_create_for_post_tool(self):
        """Create context for post-tool hook."""
        ctx = HookContext(
            session_id="session-123",
            hook_event=HookEvent.POST_TOOL_USE,
            tool_name="Read",
            tool_result="file contents here",
        )

        assert ctx.tool_result == "file contents here"

    def test_create_for_user_prompt(self):
        """Create context for user prompt hook."""
        ctx = HookContext(
            session_id="session-123",
            hook_event=HookEvent.USER_PROMPT_SUBMIT,
            prompt="Write a function",
        )

        assert ctx.prompt == "Write a function"

    def test_create_with_error(self):
        """Create context with error."""
        error = ValueError("test error")
        ctx = HookContext(
            session_id="session-123",
            hook_event=HookEvent.STOP,
            error=error,
        )

        assert ctx.error is error

    def test_metadata_is_mutable(self):
        """Metadata can be modified."""
        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
        )

        ctx.metadata["key"] = "value"
        assert ctx.metadata["key"] == "value"


class TestHookMatcher:
    """Tests for HookMatcher."""

    def test_matches_without_pattern(self):
        """Matcher without pattern matches all tools."""

        async def callback(ctx: HookContext) -> HookContext:
            return ctx

        matcher = HookMatcher(callback=callback)

        assert matcher.matches("Bash")
        assert matcher.matches("Read")
        assert matcher.matches("anything")
        assert matcher.matches(None)

    def test_matches_with_exact_pattern(self):
        """Matcher with exact pattern (anchored regex)."""

        async def callback(ctx: HookContext) -> HookContext:
            return ctx

        # Use anchored regex for exact match
        matcher = HookMatcher(callback=callback, pattern="^Bash$")

        assert matcher.matches("Bash")
        assert not matcher.matches("Read")
        assert not matcher.matches("BashScript")

    def test_matches_with_regex_pattern(self):
        """Matcher with regex pattern."""

        async def callback(ctx: HookContext) -> HookContext:
            return ctx

        matcher = HookMatcher(callback=callback, pattern="Bash.*")

        assert matcher.matches("Bash")
        assert matcher.matches("BashCommand")
        assert not matcher.matches("Read")

    def test_matches_with_alternation_pattern(self):
        """Matcher with alternation pattern."""

        async def callback(ctx: HookContext) -> HookContext:
            return ctx

        matcher = HookMatcher(callback=callback, pattern="Edit|Write")

        assert matcher.matches("Edit")
        assert matcher.matches("Write")
        assert not matcher.matches("Read")

    def test_no_match_when_tool_is_none(self):
        """Pattern doesn't match when tool_name is None."""

        async def callback(ctx: HookContext) -> HookContext:
            return ctx

        matcher = HookMatcher(callback=callback, pattern="Bash")

        assert not matcher.matches(None)

    def test_default_values(self):
        """Default values are correct."""

        async def callback(ctx: HookContext) -> HookContext:
            return ctx

        matcher = HookMatcher(callback=callback)

        assert matcher.pattern is None
        assert matcher.priority == 0
        assert matcher.timeout == 30.0


class TestHookRegistry:
    """Tests for HookRegistry."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_hook_registry()

    def test_register_hook(self):
        """Register a hook callback."""
        registry = HookRegistry()

        async def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, my_hook)

        assert registry.count(HookEvent.PRE_TOOL_USE) == 1

    def test_register_with_pattern(self):
        """Register hook with pattern."""
        registry = HookRegistry()

        async def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, my_hook, pattern="Bash.*")

        hooks = registry.list_hooks(HookEvent.PRE_TOOL_USE)
        assert len(hooks) == 1
        assert hooks[0].pattern == "Bash.*"

    def test_unregister_hook(self):
        """Unregister a hook callback."""
        registry = HookRegistry()

        async def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, my_hook)
        assert registry.count(HookEvent.PRE_TOOL_USE) == 1

        result = registry.unregister(HookEvent.PRE_TOOL_USE, my_hook)
        assert result is True
        assert registry.count(HookEvent.PRE_TOOL_USE) == 0

    def test_unregister_nonexistent(self):
        """Unregister nonexistent hook returns False."""
        registry = HookRegistry()

        async def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        result = registry.unregister(HookEvent.PRE_TOOL_USE, my_hook)
        assert result is False

    @pytest.mark.asyncio
    async def test_run_hooks(self):
        """Run hooks and verify execution."""
        registry = HookRegistry()
        called = []

        async def hook1(ctx: HookContext) -> HookContext:
            called.append("hook1")
            return ctx

        async def hook2(ctx: HookContext) -> HookContext:
            called.append("hook2")
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, hook1)
        registry.register(HookEvent.PRE_TOOL_USE, hook2)

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
        )

        await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert "hook1" in called
        assert "hook2" in called

    @pytest.mark.asyncio
    async def test_run_hooks_priority_order(self):
        """Hooks run in priority order (highest first)."""
        registry = HookRegistry()
        order = []

        async def low_priority(ctx: HookContext) -> HookContext:
            order.append("low")
            return ctx

        async def high_priority(ctx: HookContext) -> HookContext:
            order.append("high")
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, low_priority, priority=1)
        registry.register(HookEvent.PRE_TOOL_USE, high_priority, priority=10)

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
        )

        await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_run_hooks_pattern_filtering(self):
        """Only matching hooks are run."""
        registry = HookRegistry()
        called = []

        async def bash_hook(ctx: HookContext) -> HookContext:
            called.append("bash")
            return ctx

        async def read_hook(ctx: HookContext) -> HookContext:
            called.append("read")
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, bash_hook, pattern="Bash")
        registry.register(HookEvent.PRE_TOOL_USE, read_hook, pattern="Read")

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
        )

        await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert called == ["bash"]

    @pytest.mark.asyncio
    async def test_run_hooks_modifies_context(self):
        """Hooks can modify context."""
        registry = HookRegistry()

        async def add_metadata(ctx: HookContext) -> HookContext:
            ctx.metadata["added"] = True
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, add_metadata)

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
        )

        result = await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert result.metadata["added"] is True

    @pytest.mark.asyncio
    async def test_run_hooks_chain_modifications(self):
        """Multiple hooks chain context modifications."""
        registry = HookRegistry()

        async def add_a(ctx: HookContext) -> HookContext:
            ctx.metadata["a"] = True
            return ctx

        async def add_b(ctx: HookContext) -> HookContext:
            ctx.metadata["b"] = True
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, add_a)
        registry.register(HookEvent.PRE_TOOL_USE, add_b)

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
        )

        result = await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert result.metadata["a"] is True
        assert result.metadata["b"] is True

    @pytest.mark.asyncio
    async def test_run_hooks_handles_timeout(self):
        """Hook timeout is handled gracefully."""
        registry = HookRegistry()

        async def slow_hook(ctx: HookContext) -> HookContext:
            await asyncio.sleep(10)
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, slow_hook, timeout=0.01)

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
        )

        result = await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert result.metadata.get("_hook_timeout") is True

    @pytest.mark.asyncio
    async def test_run_hooks_handles_exception(self):
        """Hook exceptions are handled gracefully."""
        registry = HookRegistry()

        async def bad_hook(ctx: HookContext) -> HookContext:
            raise ValueError("test error")

        registry.register(HookEvent.PRE_TOOL_USE, bad_hook)

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
        )

        result = await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert "test error" in result.metadata.get("_hook_error", "")

    def test_decorator_registration(self):
        """Register hook via decorator."""
        registry = HookRegistry()

        @registry.on(HookEvent.PRE_TOOL_USE)
        async def my_hook(ctx: HookContext) -> HookContext:
            return ctx

        assert registry.count(HookEvent.PRE_TOOL_USE) == 1

    def test_decorator_with_pattern(self):
        """Decorator registration with pattern."""
        registry = HookRegistry()

        @registry.on(HookEvent.PRE_TOOL_USE, pattern="Bash.*")
        async def bash_hook(ctx: HookContext) -> HookContext:
            return ctx

        hooks = registry.list_hooks(HookEvent.PRE_TOOL_USE)
        assert hooks[0].pattern == "Bash.*"

    def test_decorator_with_priority(self):
        """Decorator registration with priority."""
        registry = HookRegistry()

        @registry.on(HookEvent.PRE_TOOL_USE, priority=100)
        async def high_priority_hook(ctx: HookContext) -> HookContext:
            return ctx

        hooks = registry.list_hooks(HookEvent.PRE_TOOL_USE)
        assert hooks[0].priority == 100

    def test_clear_specific_event(self):
        """Clear hooks for specific event."""
        registry = HookRegistry()

        async def hook1(ctx: HookContext) -> HookContext:
            return ctx

        async def hook2(ctx: HookContext) -> HookContext:
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, hook1)
        registry.register(HookEvent.POST_TOOL_USE, hook2)

        registry.clear(HookEvent.PRE_TOOL_USE)

        assert registry.count(HookEvent.PRE_TOOL_USE) == 0
        assert registry.count(HookEvent.POST_TOOL_USE) == 1

    def test_clear_all_events(self):
        """Clear all hooks."""
        registry = HookRegistry()

        async def hook(ctx: HookContext) -> HookContext:
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, hook)
        registry.register(HookEvent.POST_TOOL_USE, hook)
        registry.register(HookEvent.STOP, hook)

        registry.clear()

        assert registry.count() == 0

    def test_count_specific_event(self):
        """Count hooks for specific event."""
        registry = HookRegistry()

        async def hook(ctx: HookContext) -> HookContext:
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, hook)
        registry.register(HookEvent.PRE_TOOL_USE, hook)
        registry.register(HookEvent.POST_TOOL_USE, hook)

        assert registry.count(HookEvent.PRE_TOOL_USE) == 2
        assert registry.count(HookEvent.POST_TOOL_USE) == 1

    def test_count_all_events(self):
        """Count all hooks."""
        registry = HookRegistry()

        async def hook(ctx: HookContext) -> HookContext:
            return ctx

        registry.register(HookEvent.PRE_TOOL_USE, hook)
        registry.register(HookEvent.POST_TOOL_USE, hook)
        registry.register(HookEvent.STOP, hook)

        assert registry.count() == 3

    def test_global_registry_singleton(self):
        """Global registry returns same instance."""
        reset_hook_registry()

        registry1 = get_hook_registry()
        registry2 = get_hook_registry()

        assert registry1 is registry2


class TestHookIntegration:
    """Integration tests for hook system."""

    @pytest.mark.asyncio
    async def test_multiple_events(self):
        """Register hooks for multiple events."""
        registry = HookRegistry()
        events_fired = []

        @registry.on(HookEvent.PRE_TOOL_USE)
        async def pre_hook(ctx: HookContext) -> HookContext:
            events_fired.append("pre")
            return ctx

        @registry.on(HookEvent.POST_TOOL_USE)
        async def post_hook(ctx: HookContext) -> HookContext:
            events_fired.append("post")
            return ctx

        # Fire pre-tool event
        pre_ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
        )
        await registry.run(HookEvent.PRE_TOOL_USE, pre_ctx)

        # Fire post-tool event
        post_ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.POST_TOOL_USE,
            tool_name="Bash",
            tool_result="success",
        )
        await registry.run(HookEvent.POST_TOOL_USE, post_ctx)

        assert events_fired == ["pre", "post"]

    @pytest.mark.asyncio
    async def test_hook_can_block_execution(self):
        """Hook can signal blocking via metadata."""
        registry = HookRegistry()

        @registry.on(HookEvent.PRE_TOOL_USE, pattern="Bash")
        async def block_bash(ctx: HookContext) -> HookContext:
            ctx.metadata["blocked"] = True
            ctx.metadata["block_reason"] = "Bash commands disabled"
            return ctx

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
            tool_input={"command": "rm -rf /"},
        )

        result = await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert result.metadata["blocked"] is True
        assert "disabled" in result.metadata["block_reason"]

    @pytest.mark.asyncio
    async def test_hook_can_modify_input(self):
        """Hook can modify tool input."""
        registry = HookRegistry()

        @registry.on(HookEvent.PRE_TOOL_USE, pattern="Bash")
        async def sanitize_bash(ctx: HookContext) -> HookContext:
            if ctx.tool_input and "command" in ctx.tool_input:
                # Add safety prefix
                ctx.tool_input["command"] = "set -e; " + ctx.tool_input["command"]
            return ctx

        ctx = HookContext(
            session_id="test",
            hook_event=HookEvent.PRE_TOOL_USE,
            tool_name="Bash",
            tool_input={"command": "ls -la"},
        )

        result = await registry.run(HookEvent.PRE_TOOL_USE, ctx)

        assert result.tool_input["command"] == "set -e; ls -la"
