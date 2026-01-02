"""
Hook system for ccflow.

Provides SDK-compatible lifecycle hooks for tool execution,
session events, and message handling.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    from .types import Message


class HookEvent(str, Enum):
    """Hook event types (SDK-compatible).

    These match the Claude Agent SDK's hook types:
    - PRE_TOOL_USE: Before tool execution
    - POST_TOOL_USE: After tool execution
    - USER_PROMPT_SUBMIT: User submits a prompt
    - STOP: Agent stops execution
    - SUBAGENT_STOP: Subagent finishes
    - PRE_COMPACT: Before context compaction
    """

    PRE_TOOL_USE = "pre_tool_use"
    POST_TOOL_USE = "post_tool_use"
    USER_PROMPT_SUBMIT = "user_prompt_submit"
    STOP = "stop"
    SUBAGENT_STOP = "subagent_stop"
    PRE_COMPACT = "pre_compact"


@dataclass
class HookContext:
    """Context passed to hook callbacks.

    Contains information about the current hook event and
    allows hooks to modify or inspect execution state.

    Attributes:
        session_id: Current session identifier
        hook_event: The type of hook being executed
        tool_name: Name of the tool (for tool hooks)
        tool_input: Input data for the tool
        tool_result: Result from tool execution (post-tool only)
        message: The message being processed
        prompt: User prompt (for user_prompt_submit)
        stop_reason: Reason for stopping (for stop hooks)
        error: Exception if an error occurred
        metadata: Additional context data
    """

    session_id: str
    hook_event: HookEvent
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: str | None = None
    message: Message | None = None
    prompt: str | None = None
    stop_reason: str | None = None
    error: Exception | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# Hook callback type: takes context, returns modified context or None
HookCallback: TypeAlias = Callable[[HookContext], Awaitable[HookContext | None]]


@dataclass
class HookMatcher:
    """Matches hooks against tool patterns.

    Supports regex patterns for tool names and async callbacks.

    Attributes:
        callback: The async callback function to invoke
        pattern: Regex pattern for tool names (None = match all)
        priority: Higher priority runs first (default: 0)
        timeout: Timeout in seconds for the callback (default: 30)
    """

    callback: HookCallback
    pattern: str | None = None
    priority: int = 0
    timeout: float = 30.0

    def matches(self, tool_name: str | None) -> bool:
        """Check if hook matches the given tool name.

        Args:
            tool_name: The tool name to check

        Returns:
            True if pattern matches or pattern is None
        """
        if self.pattern is None:
            return True
        if tool_name is None:
            return False
        return bool(re.match(self.pattern, tool_name))


class HookRegistry:
    """Registry for hook callbacks.

    Manages hook registration and execution with support for:
    - Pattern matching on tool names
    - Priority-based execution order
    - Async callbacks with timeout
    - Decorator-based registration

    Example:
        >>> registry = HookRegistry()
        >>> @registry.on(HookEvent.PRE_TOOL_USE, pattern="Bash.*")
        ... async def audit_bash(ctx: HookContext) -> HookContext:
        ...     print(f"Bash command: {ctx.tool_input}")
        ...     return ctx
    """

    def __init__(self) -> None:
        """Initialize empty hook registry."""
        self._hooks: dict[HookEvent, list[HookMatcher]] = {event: [] for event in HookEvent}

    def register(
        self,
        hook_event: HookEvent,
        callback: HookCallback,
        pattern: str | None = None,
        priority: int = 0,
        timeout: float = 30.0,
    ) -> None:
        """Register a hook callback.

        Args:
            hook_event: The event type to hook
            callback: Async callback function
            pattern: Regex pattern for tool names (optional)
            priority: Higher priority runs first (default: 0)
            timeout: Callback timeout in seconds (default: 30)
        """
        matcher = HookMatcher(
            callback=callback,
            pattern=pattern,
            priority=priority,
            timeout=timeout,
        )
        self._hooks[hook_event].append(matcher)

    def unregister(
        self,
        hook_event: HookEvent,
        callback: HookCallback,
    ) -> bool:
        """Unregister a hook callback.

        Args:
            hook_event: The event type
            callback: The callback to remove

        Returns:
            True if callback was found and removed
        """
        hooks = self._hooks[hook_event]
        for i, matcher in enumerate(hooks):
            if matcher.callback is callback:
                del hooks[i]
                return True
        return False

    async def run(
        self,
        hook_event: HookEvent,
        context: HookContext,
    ) -> HookContext:
        """Run all matching hooks for an event.

        Hooks are executed in priority order (highest first).
        Each hook can modify the context for the next hook.

        Args:
            hook_event: The event type
            context: Initial hook context

        Returns:
            Final context after all hooks have run
        """
        # Sort by priority (descending)
        matchers = sorted(
            self._hooks[hook_event],
            key=lambda m: -m.priority,
        )

        for matcher in matchers:
            if not matcher.matches(context.tool_name):
                continue

            try:
                result = await asyncio.wait_for(
                    matcher.callback(context),
                    timeout=matcher.timeout,
                )
                if result is not None:
                    context = result
            except TimeoutError:
                # Log timeout but continue with other hooks
                context.metadata["_hook_timeout"] = True
            except Exception as e:
                # Store error but continue
                context.metadata["_hook_error"] = str(e)

        return context

    def on(
        self,
        hook_event: HookEvent,
        pattern: str | None = None,
        priority: int = 0,
        timeout: float = 30.0,
    ) -> Callable[[HookCallback], HookCallback]:
        """Decorator for registering hooks.

        Example:
            >>> @hooks.on(HookEvent.PRE_TOOL_USE, pattern="Edit|Write")
            ... async def log_edits(ctx: HookContext) -> HookContext:
            ...     print(f"Editing: {ctx.tool_input}")
            ...     return ctx

        Args:
            hook_event: The event type to hook
            pattern: Regex pattern for tool names (optional)
            priority: Higher priority runs first (default: 0)
            timeout: Callback timeout in seconds (default: 30)

        Returns:
            Decorator function
        """

        def decorator(func: HookCallback) -> HookCallback:
            self.register(hook_event, func, pattern, priority, timeout)
            return func

        return decorator

    def clear(self, hook_event: HookEvent | None = None) -> None:
        """Clear registered hooks.

        Args:
            hook_event: Specific event to clear, or None to clear all
        """
        if hook_event is None:
            for event in HookEvent:
                self._hooks[event] = []
        else:
            self._hooks[hook_event] = []

    def list_hooks(self, hook_event: HookEvent) -> list[HookMatcher]:
        """List hooks for an event type.

        Args:
            hook_event: The event type

        Returns:
            List of registered hook matchers
        """
        return list(self._hooks[hook_event])

    def count(self, hook_event: HookEvent | None = None) -> int:
        """Count registered hooks.

        Args:
            hook_event: Specific event, or None for total count

        Returns:
            Number of registered hooks
        """
        if hook_event is None:
            return sum(len(hooks) for hooks in self._hooks.values())
        return len(self._hooks[hook_event])


# Global hook registry singleton
_global_hook_registry: HookRegistry | None = None


def get_hook_registry() -> HookRegistry:
    """Get the global hook registry singleton.

    Returns:
        The global HookRegistry instance
    """
    global _global_hook_registry
    if _global_hook_registry is None:
        _global_hook_registry = HookRegistry()
    return _global_hook_registry


def reset_hook_registry() -> None:
    """Reset the global hook registry (for testing)."""
    global _global_hook_registry
    _global_hook_registry = None
