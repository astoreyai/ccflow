#!/usr/bin/env python3
"""
Hook System Example (v0.2.0)

Demonstrates SDK-compatible lifecycle hooks for tool execution and session events.
"""

import asyncio

from ccflow import (
    CLIAgentOptions,
    HookContext,
    HookEvent,
    TextMessage,
    get_hook_registry,
    query,
    reset_hook_registry,
)


async def basic_hooks():
    """Register and use basic hooks."""
    print("=== Basic Hooks ===")

    # Reset registry for clean demo
    reset_hook_registry()
    hooks = get_hook_registry()

    # Track tool usage
    tool_calls = []

    @hooks.on(HookEvent.PRE_TOOL_USE)
    async def log_pre_tool(ctx: HookContext) -> HookContext:
        """Log before tool execution."""
        print(f"  [PRE] Tool: {ctx.tool_name}")
        tool_calls.append(ctx.tool_name)
        return ctx

    @hooks.on(HookEvent.POST_TOOL_USE)
    async def log_post_tool(ctx: HookContext) -> HookContext:
        """Log after tool execution."""
        result_preview = str(ctx.tool_result or "")[:50]
        print(f"  [POST] Tool: {ctx.tool_name} -> {result_preview}...")
        return ctx

    # Execute query that uses tools
    print("Executing query with hooks active...")
    options = CLIAgentOptions(
        model="haiku",
        allowed_tools=["Glob"],
    )

    async for msg in query("List *.py files in examples", options):
        if isinstance(msg, TextMessage):
            pass  # Suppress output for demo

    print(f"\nTools called: {tool_calls}")


async def pattern_matching_hooks():
    """Use pattern matching to filter hooks."""
    print("\n=== Pattern Matching Hooks ===")

    reset_hook_registry()
    hooks = get_hook_registry()

    # Only audit Bash commands
    @hooks.on(HookEvent.PRE_TOOL_USE, pattern=r"Bash.*")
    async def audit_bash(ctx: HookContext) -> HookContext:
        """Audit all Bash commands."""
        print(f"  [AUDIT] Bash command detected: {ctx.tool_input}")
        return ctx

    # Only log Read operations
    @hooks.on(HookEvent.PRE_TOOL_USE, pattern=r"Read")
    async def log_reads(ctx: HookContext) -> HookContext:
        """Log file reads."""
        print(f"  [READ] File: {ctx.tool_input}")
        return ctx

    print("Pattern-based hooks registered for Bash and Read tools")


async def priority_hooks():
    """Demonstrate hook priority execution order."""
    print("\n=== Priority Hooks ===")

    reset_hook_registry()
    hooks = get_hook_registry()

    @hooks.on(HookEvent.PRE_TOOL_USE, priority=10)
    async def high_priority(ctx: HookContext) -> HookContext:
        """Runs first (highest priority)."""
        print("  1. High priority hook (10)")
        return ctx

    @hooks.on(HookEvent.PRE_TOOL_USE, priority=5)
    async def medium_priority(ctx: HookContext) -> HookContext:
        """Runs second."""
        print("  2. Medium priority hook (5)")
        return ctx

    @hooks.on(HookEvent.PRE_TOOL_USE, priority=0)
    async def low_priority(ctx: HookContext) -> HookContext:
        """Runs last (default priority)."""
        print("  3. Low priority hook (0)")
        return ctx

    # Manually run hooks to demonstrate order
    ctx = HookContext(
        session_id="demo",
        hook_event=HookEvent.PRE_TOOL_USE,
        tool_name="Demo",
    )
    print("Running hooks in priority order:")
    await hooks.run(HookEvent.PRE_TOOL_USE, ctx)


async def hook_events_demo():
    """Show all available hook events."""
    print("\n=== Available Hook Events ===")

    for event in HookEvent:
        print(f"  - {event.value}")

    print("\nEvent descriptions:")
    print("  PRE_TOOL_USE    - Before tool execution (can modify input)")
    print("  POST_TOOL_USE   - After tool execution (can modify result)")
    print("  USER_PROMPT_SUBMIT - User submits a prompt")
    print("  STOP            - Agent stops execution")
    print("  SUBAGENT_STOP   - Subagent finishes")
    print("  PRE_COMPACT     - Before context compaction")


async def main():
    """Run all hook examples."""
    await basic_hooks()
    await pattern_matching_hooks()
    await priority_hooks()
    await hook_events_demo()
    print("\nAll hook examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
