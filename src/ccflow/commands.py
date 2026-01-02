"""
CLI command system for ccflow.

Provides CommandRegistry for registering and executing
/command style commands.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import structlog

from .agent import get_agent_registry
from .hooks import HookEvent, get_hook_registry
from .skills import get_skill_loader
from .subagent import get_subagent_coordinator

logger = structlog.get_logger(__name__)


@dataclass
class CommandDefinition:
    """Definition of a CLI command.

    Attributes:
        name: Command name (without /)
        description: Help text
        handler: Async handler function
        args: Expected argument names
        metadata: Additional command metadata
    """

    name: str
    description: str
    handler: Callable
    args: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class CommandRegistry:
    """Registry for CLI commands.

    Manages command registration and execution with support
    for argument parsing and help generation.

    Example:
        >>> registry = CommandRegistry()
        >>> @registry.command("greet", "Greet the user")
        ... async def greet(name: str = "World"):
        ...     return f"Hello, {name}!"
        >>> result = await registry.execute("greet", ["Aaron"])
    """

    def __init__(self) -> None:
        """Initialize command registry."""
        self._commands: dict[str, CommandDefinition] = {}

    def register(self, command: CommandDefinition) -> None:
        """Register a command definition.

        Args:
            command: Command to register
        """
        self._commands[command.name] = command

    def command(
        self,
        name: str,
        description: str,
        args: list[str] | None = None,
    ) -> Callable[[Callable], Callable]:
        """Decorator for registering commands.

        Example:
            >>> @registry.command("echo", "Echo back input", args=["text"])
            ... async def echo(text: str) -> str:
            ...     return text

        Args:
            name: Command name (without /)
            description: Help text
            args: Expected argument names

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            cmd = CommandDefinition(
                name=name,
                description=description,
                handler=func,
                args=args or [],
            )
            self.register(cmd)

            @functools.wraps(func)
            async def wrapper(*a, **kw):
                return await func(*a, **kw)

            return wrapper

        return decorator

    def get(self, name: str) -> CommandDefinition | None:
        """Get command by name.

        Args:
            name: Command name (without /)

        Returns:
            Command definition or None
        """
        return self._commands.get(name)

    async def execute(
        self,
        name: str,
        args: list[str] | None = None,
    ) -> Any:
        """Execute a command by name.

        Args:
            name: Command name (without /)
            args: Command arguments

        Returns:
            Command result

        Raises:
            ValueError: If command not found
        """
        command = self.get(name)
        if command is None:
            raise ValueError(f"Unknown command: /{name}")

        args = args or []

        # Map positional args to expected arg names
        kwargs = {}
        for i, arg_name in enumerate(command.args):
            if i < len(args):
                kwargs[arg_name] = args[i]

        logger.debug("executing_command", name=name, args=kwargs)

        return await command.handler(**kwargs)

    def list(self) -> list[CommandDefinition]:
        """List all registered commands.

        Returns:
            List of command definitions
        """
        return list(self._commands.values())

    def help(self) -> str:
        """Generate help text for all commands.

        Returns:
            Formatted help string
        """
        lines = ["Available commands:"]
        for cmd in sorted(self._commands.values(), key=lambda c: c.name):
            arg_str = " ".join(f"<{a}>" for a in cmd.args)
            lines.append(f"  /{cmd.name} {arg_str}".rstrip())
            lines.append(f"    {cmd.description}")
        return "\n".join(lines)


# Global command registry singleton
_global_command_registry: CommandRegistry | None = None


def get_command_registry() -> CommandRegistry:
    """Get the global command registry singleton.

    Returns:
        The global CommandRegistry instance
    """
    global _global_command_registry
    if _global_command_registry is None:
        _global_command_registry = CommandRegistry()
        _register_builtin_commands(_global_command_registry)
    return _global_command_registry


def reset_command_registry() -> None:
    """Reset the global command registry (for testing)."""
    global _global_command_registry
    _global_command_registry = None


def _register_builtin_commands(registry: CommandRegistry) -> None:
    """Register built-in commands."""

    @registry.command("agents", "List available agents")
    async def cmd_agents() -> str:
        """List all registered agents."""
        agent_registry = get_agent_registry()
        names = agent_registry.list()

        if not names:
            return "No agents registered."

        lines = ["Available agents:"]
        for name in sorted(names):
            agent = agent_registry.get(name)
            if agent:
                desc = (
                    agent.description[:50] + "..."
                    if len(agent.description) > 50
                    else agent.description
                )
                lines.append(f"  {name}: {desc}")

        return "\n".join(lines)

    @registry.command("skills", "List available skills")
    async def cmd_skills() -> str:
        """List all discovered skills."""
        loader = get_skill_loader()
        names = loader.list()

        if not names:
            return "No skills discovered."

        lines = ["Available skills:"]
        for name in sorted(names):
            skill = loader.get(name)
            if skill:
                desc = (
                    skill.description[:50] + "..."
                    if len(skill.description) > 50
                    else skill.description
                )
                lines.append(f"  {name}: {desc}")

        return "\n".join(lines)

    @registry.command("hooks", "List registered hooks")
    async def cmd_hooks() -> str:
        """List all registered hooks by event type."""
        hook_registry = get_hook_registry()

        lines = ["Registered hooks:"]
        for event in HookEvent:
            count = hook_registry.count(event)
            lines.append(f"  {event.value}: {count} hook(s)")

        total = hook_registry.count()
        lines.append(f"Total: {total} hook(s)")

        return "\n".join(lines)

    @registry.command("spawn", "Spawn a subagent", args=["agent", "task"])
    async def cmd_spawn(agent: str = "", task: str = "") -> str:
        """Spawn a subagent with a task."""
        if not agent or not task:
            return "Usage: /spawn <agent> <task>"

        coordinator = get_subagent_coordinator()

        try:
            result = await coordinator.spawn_simple(agent, task)
            return result
        except ValueError as e:
            return f"Error: {e}"

    @registry.command("help", "Show available commands")
    async def cmd_help() -> str:
        """Show help for all commands."""
        return registry.help()


def parse_command(input_str: str) -> tuple[str, list[str]] | None:
    """Parse /command input.

    Args:
        input_str: Input string starting with /

    Returns:
        Tuple of (command_name, args) or None if not a command
    """
    if not input_str.startswith("/"):
        return None

    # Remove leading / and split
    parts = input_str[1:].split(maxsplit=1)
    if not parts:
        return None

    name = parts[0]
    args = parts[1].split() if len(parts) > 1 else []

    return (name, args)


async def handle_command(input_str: str) -> str | None:
    """Handle /command input.

    Args:
        input_str: Input string (may or may not be a command)

    Returns:
        Command result or None if not a command
    """
    parsed = parse_command(input_str)
    if not parsed:
        return None

    name, args = parsed
    registry = get_command_registry()

    try:
        return await registry.execute(name, args)
    except ValueError as e:
        return str(e)
