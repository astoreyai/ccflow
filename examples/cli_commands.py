#!/usr/bin/env python3
"""
CLI Commands Example (v0.2.0)

Demonstrates the /command system for ccflow.
"""

import asyncio

from ccflow import (
    get_command_registry,
    handle_command,
    parse_command,
    reset_command_registry,
)


async def builtin_commands():
    """Show built-in commands."""
    print("=== Built-in Commands ===")

    reset_command_registry()
    registry = get_command_registry()

    # List all commands
    print("Available commands:")
    for cmd in registry.list():
        args = " ".join(f"<{a}>" for a in cmd.args)
        print(f"  /{cmd.name} {args}".rstrip())
        print(f"    {cmd.description}")

    print("\nBuilt-in commands include:")
    print("  /agents - List available agents")
    print("  /skills - List available skills")
    print("  /hooks  - List registered hooks")
    print("  /spawn  - Spawn a subagent")
    print("  /help   - Show this help")


async def custom_commands():
    """Register custom commands."""
    print("\n=== Custom Commands ===")

    reset_command_registry()
    registry = get_command_registry()

    # Register a custom command
    @registry.command("status", "Show project status")
    async def cmd_status() -> str:
        """Custom status command."""
        return "All systems operational"

    @registry.command("greet", "Greet someone", args=["name"])
    async def cmd_greet(name: str = "World") -> str:
        """Custom greeting command."""
        return f"Hello, {name}!"

    @registry.command("calc", "Simple calculator", args=["a", "b"])
    async def cmd_calc(a: str = "0", b: str = "0") -> str:
        """Add two numbers."""
        try:
            result = int(a) + int(b)
            return f"{a} + {b} = {result}"
        except ValueError:
            return "Error: Please provide valid numbers"

    print("Custom commands registered:")
    for cmd in registry.list():
        if cmd.name in ["status", "greet", "calc"]:
            print(f"  /{cmd.name}: {cmd.description}")


async def command_parsing():
    """Demonstrate command parsing."""
    print("\n=== Command Parsing ===")

    # Parse command strings
    test_inputs = [
        "/help",
        "/greet Aaron",
        "/calc 5 3",
        "/spawn code-reviewer Review main.py",
        "not a command",
    ]

    print("Parsing command strings:")
    for inp in test_inputs:
        result = parse_command(inp)
        if result:
            name, args = result
            print(f"  '{inp}' -> name='{name}', args={args}")
        else:
            print(f"  '{inp}' -> Not a command")


async def command_execution():
    """Demonstrate command execution."""
    print("\n=== Command Execution ===")

    reset_command_registry()
    registry = get_command_registry()

    # Register commands
    @registry.command("echo", "Echo back input", args=["text"])
    async def cmd_echo(text: str = "") -> str:
        return f"Echo: {text}"

    @registry.command("count", "Count characters", args=["text"])
    async def cmd_count(text: str = "") -> str:
        return f"Length: {len(text)}"

    # Execute commands
    commands = [
        "/echo Hello World",
        "/count Testing123",
        "/help",
    ]

    print("Executing commands:")
    for cmd in commands:
        result = await handle_command(cmd)
        if result:
            preview = result[:80].replace("\n", " ")
            print(f"  '{cmd}' -> {preview}...")


async def command_registry_api():
    """Show CommandRegistry API."""
    print("\n=== Command Registry API ===")

    reset_command_registry()
    registry = get_command_registry()

    print("CommandRegistry methods:")
    print("  register(cmd)     - Register a CommandDefinition")
    print("  command(name, desc, args=[])  - Decorator for registration")
    print("  get(name)         - Get command by name")
    print("  execute(name, args)  - Execute command")
    print("  list()            - List all commands")
    print("  help()            - Generate help text")

    print("\nCommandDefinition fields:")
    print("  name: str         - Command name (without /)")
    print("  description: str  - Help text")
    print("  handler: Callable - Async handler function")
    print("  args: list[str]   - Expected argument names")
    print("  metadata: dict    - Additional metadata")

    # Show help output
    print("\nGenerated help:")
    print(registry.help())


async def main():
    """Run all command examples."""
    await builtin_commands()
    await custom_commands()
    await command_parsing()
    await command_execution()
    await command_registry_api()
    print("\nAll command examples complete!")


if __name__ == "__main__":
    asyncio.run(main())
