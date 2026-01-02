#!/usr/bin/env python3
"""
Subagent Coordination Example (v0.2.0)

Demonstrates parallel agent execution with concurrency control.
"""

import asyncio

from ccflow import (
    AgentDefinition,
    get_agent_registry,
    get_subagent_coordinator,
    reset_subagent_coordinator,
)


async def setup_demo_agents():
    """Set up demo agents for the examples."""
    registry = get_agent_registry()

    # Register demo agents
    agents = [
        AgentDefinition(
            name="analyzer",
            description="Analyzes code structure",
            system_prompt="You analyze code structure briefly.",
            tools=["Glob"],
            model="haiku",
        ),
        AgentDefinition(
            name="reviewer",
            description="Reviews code quality",
            system_prompt="You review code quality briefly.",
            tools=["Read"],
            model="haiku",
        ),
        AgentDefinition(
            name="tester",
            description="Runs tests",
            system_prompt="You describe test scenarios briefly.",
            tools=["Bash"],
            model="haiku",
        ),
    ]

    for agent in agents:
        registry.register(agent)

    print(f"Registered {len(agents)} demo agents")


async def parallel_execution():
    """Execute multiple agents in parallel."""
    print("\n=== Parallel Execution ===")

    reset_subagent_coordinator()
    coordinator = get_subagent_coordinator()

    print("Executing 3 agents in parallel...")
    print("(Note: This example shows the API - actual execution requires CLI)")

    # In real usage, this would execute agents concurrently:
    # results = await coordinator.parallel([
    #     ("analyzer", "Analyze the examples directory structure"),
    #     ("reviewer", "Review basic_query.py for quality"),
    #     ("tester", "Suggest tests for the session module"),
    # ])
    #
    # for i, result in enumerate(results):
    #     print(f"\nAgent {i+1} result: {result[:100]}...")

    print("Parallel execution pattern:")
    print("  results = await coordinator.parallel([")
    print('      ("agent1", "task1"),')
    print('      ("agent2", "task2"),')
    print("  ])")


async def streaming_spawn():
    """Stream results from a single subagent."""
    print("\n=== Streaming Spawn ===")

    reset_subagent_coordinator()
    coordinator = get_subagent_coordinator()

    print("Streaming from subagent:")
    print("  async for msg in coordinator.spawn('agent', 'task'):")
    print("      if isinstance(msg, TextMessage):")
    print("          print(msg.content)")

    # In real usage:
    # async for msg in coordinator.spawn("analyzer", "Analyze structure"):
    #     if isinstance(msg, TextMessage):
    #         print(msg.content, end="")


async def background_tasks():
    """Demonstrate background task execution."""
    print("\n=== Background Tasks ===")

    reset_subagent_coordinator()
    coordinator = get_subagent_coordinator()

    print("Background task workflow:")
    print("  # Spawn task in background")
    print('  task_id = await coordinator.spawn_background("agent", "task")')
    print("")
    print("  # Do other work...")
    print("")
    print("  # Get result when ready")
    print("  task = await coordinator.get_task_result(task_id, wait=True)")
    print("  print(task.status, task.result)")
    print("")
    print("  # Or cancel if needed")
    print("  coordinator.cancel(task_id)")

    # Demonstrate the data structure
    print("\nSubagentTask fields:")
    print("  - task_id: str")
    print("  - agent_name: str")
    print("  - prompt: str")
    print("  - context: dict | None")
    print("  - status: 'pending' | 'running' | 'completed' | 'failed'")
    print("  - result: str | None")
    print("  - error: str | None")


async def concurrency_control():
    """Demonstrate concurrency control."""
    print("\n=== Concurrency Control ===")

    reset_subagent_coordinator()

    # Create coordinator with custom max_concurrent
    from ccflow.subagent import SubagentCoordinator

    coordinator = SubagentCoordinator(max_concurrent=5)

    print(f"Max concurrent subagents: {coordinator.max_concurrent}")
    print("")
    print("The coordinator uses a semaphore to limit concurrency:")
    print("  coordinator = SubagentCoordinator(max_concurrent=5)")
    print("")
    print("When you spawn more than max_concurrent agents, extras wait.")


async def gather_all():
    """Demonstrate gathering all background tasks."""
    print("\n=== Gather All Tasks ===")

    reset_subagent_coordinator()
    coordinator = get_subagent_coordinator()

    print("Wait for all background tasks:")
    print("  # Spawn multiple background tasks")
    print('  await coordinator.spawn_background("agent1", "task1")')
    print('  await coordinator.spawn_background("agent2", "task2")')
    print("")
    print("  # Wait for all to complete")
    print("  all_tasks = await coordinator.gather(timeout=60.0)")
    print("  for task in all_tasks:")
    print("      print(task.agent_name, task.status)")
    print("")
    print("  # List active tasks")
    print("  active = coordinator.list_active()")
    print("")
    print("  # Cancel all active tasks")
    print("  cancelled = coordinator.cancel_all()")


async def main():
    """Run all subagent examples."""
    await setup_demo_agents()
    await parallel_execution()
    await streaming_spawn()
    await background_tasks()
    await concurrency_control()
    await gather_all()
    print("\nAll subagent examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
