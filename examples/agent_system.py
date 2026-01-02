#!/usr/bin/env python3
"""
Agent System Example (v0.2.0)

Demonstrates specialized agents with isolated tools, models, and system prompts.
"""

import asyncio

from ccflow import (
    Agent,
    AgentDefinition,
    CLIAgentOptions,
    TextMessage,
    get_agent_registry,
)


async def programmatic_agent():
    """Create and use an agent programmatically."""
    print("=== Programmatic Agent ===")

    # Define a code reviewer agent
    reviewer = Agent(
        AgentDefinition(
            name="code-reviewer",
            description="Expert code reviewer for quality and security",
            system_prompt="""You are a senior code reviewer focused on:
1. Security vulnerabilities
2. Code quality and maintainability
3. Best practices and patterns
Be concise and actionable in your feedback.""",
            tools=["Read", "Grep", "Glob"],
            model="sonnet",
        )
    )

    # Execute a review task
    print("Reviewing code...")
    async for msg in reviewer.execute("Review the examples directory for best practices"):
        if isinstance(msg, TextMessage):
            print(msg.content, end="", flush=True)

    print("\n")


async def registry_agent():
    """Use the agent registry for filesystem-discovered agents."""
    print("=== Registry Agent ===")

    registry = get_agent_registry()

    # List available agents
    print("Available agents:")
    for name in registry.list():
        agent_def = registry.get(name)
        if agent_def:
            print(f"  - {name}: {agent_def.description[:50]}...")

    # If you have agents in ~/.claude/agents/, you can use them:
    # agent = registry.get("my-agent")
    # if agent:
    #     async for msg in Agent(agent).execute("Do something"):
    #         print(msg.content, end="")


async def agent_with_options():
    """Create agent with custom options."""
    print("=== Agent with Options ===")

    # Create agent with parent options
    parent_options = CLIAgentOptions(
        timeout=120.0,
        verbose=True,
    )

    test_runner = Agent(
        AgentDefinition(
            name="test-runner",
            description="Runs tests and reports results",
            system_prompt="You run tests and report results clearly.",
            tools=["Bash", "Read"],
            model="haiku",  # Fast model for test running
            timeout=60.0,  # Override parent timeout
        ),
        parent_options=parent_options,
    )

    # Simple execution (returns aggregated result)
    result = await test_runner.execute_simple(
        "Show the first 5 Python files in examples directory"
    )
    print(f"Result: {result[:200]}...")


async def main():
    """Run all agent examples."""
    await programmatic_agent()
    await registry_agent()
    await agent_with_options()
    print("\nAll agent examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
