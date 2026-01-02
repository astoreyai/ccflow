"""
Tests for CLI command system.

Tests CommandRegistry, command execution, and built-in commands.
"""

from __future__ import annotations

import pytest

from ccflow.agent import reset_agent_registry
from ccflow.commands import (
    CommandDefinition,
    CommandRegistry,
    get_command_registry,
    handle_command,
    parse_command,
    reset_command_registry,
)
from ccflow.hooks import reset_hook_registry
from ccflow.skills import reset_skill_loader
from ccflow.subagent import reset_subagent_coordinator


class TestCommandDefinition:
    """Tests for CommandDefinition dataclass."""

    def test_create_minimal(self):
        """Create command with required fields."""

        async def handler():
            return "result"

        cmd = CommandDefinition(
            name="test",
            description="Test command",
            handler=handler,
        )

        assert cmd.name == "test"
        assert cmd.description == "Test command"
        assert cmd.args == []

    def test_create_with_args(self):
        """Create command with arguments."""

        async def handler(arg1, arg2):
            return f"{arg1} {arg2}"

        cmd = CommandDefinition(
            name="test",
            description="Test command",
            handler=handler,
            args=["arg1", "arg2"],
        )

        assert cmd.args == ["arg1", "arg2"]


class TestCommandRegistry:
    """Tests for CommandRegistry."""

    def setup_method(self):
        """Reset globals before each test."""
        reset_command_registry()

    def test_register_command(self):
        """Register a command."""
        registry = CommandRegistry()

        async def handler():
            return "result"

        cmd = CommandDefinition(
            name="test",
            description="Test",
            handler=handler,
        )
        registry.register(cmd)

        assert registry.get("test") is not None

    def test_get_nonexistent(self):
        """Get nonexistent command returns None."""
        registry = CommandRegistry()
        result = registry.get("nonexistent")

        assert result is None

    def test_decorator_registration(self):
        """Register command via decorator."""
        registry = CommandRegistry()

        @registry.command("greet", "Greet someone")
        async def greet():
            return "Hello!"

        assert registry.get("greet") is not None

    def test_decorator_with_args(self):
        """Decorator registration with args."""
        registry = CommandRegistry()

        @registry.command("echo", "Echo text", args=["text"])
        async def echo(text: str) -> str:
            return text

        cmd = registry.get("echo")
        assert cmd is not None
        assert cmd.args == ["text"]

    @pytest.mark.asyncio
    async def test_execute_simple(self):
        """Execute simple command."""
        registry = CommandRegistry()

        @registry.command("hello", "Say hello")
        async def hello():
            return "Hello, World!"

        result = await registry.execute("hello")

        assert result == "Hello, World!"

    @pytest.mark.asyncio
    async def test_execute_with_args(self):
        """Execute command with arguments."""
        registry = CommandRegistry()

        @registry.command("greet", "Greet someone", args=["name"])
        async def greet(name: str = "World"):
            return f"Hello, {name}!"

        result = await registry.execute("greet", ["Aaron"])

        assert result == "Hello, Aaron!"

    @pytest.mark.asyncio
    async def test_execute_unknown(self):
        """Execute unknown command raises error."""
        registry = CommandRegistry()

        with pytest.raises(ValueError, match="Unknown command"):
            await registry.execute("nonexistent")

    def test_list_commands(self):
        """List registered commands."""
        registry = CommandRegistry()

        @registry.command("cmd1", "Command 1")
        async def cmd1():
            pass

        @registry.command("cmd2", "Command 2")
        async def cmd2():
            pass

        commands = registry.list()

        assert len(commands) == 2
        names = [c.name for c in commands]
        assert "cmd1" in names
        assert "cmd2" in names

    def test_help_text(self):
        """Generate help text."""
        registry = CommandRegistry()

        @registry.command("simple", "A simple command")
        async def simple():
            pass

        @registry.command("with_args", "Command with args", args=["arg1", "arg2"])
        async def with_args(arg1, arg2):
            pass

        help_text = registry.help()

        assert "Available commands:" in help_text
        assert "/simple" in help_text
        assert "A simple command" in help_text
        assert "/with_args <arg1> <arg2>" in help_text


class TestParseCommand:
    """Tests for command parsing."""

    def test_parse_simple_command(self):
        """Parse simple command."""
        result = parse_command("/help")

        assert result == ("help", [])

    def test_parse_command_with_args(self):
        """Parse command with arguments."""
        result = parse_command("/spawn agent-name task description")

        assert result is not None
        assert result[0] == "spawn"
        assert result[1] == ["agent-name", "task", "description"]

    def test_parse_not_command(self):
        """Non-command returns None."""
        result = parse_command("regular text")

        assert result is None

    def test_parse_empty_after_slash(self):
        """Empty after slash returns None."""
        result = parse_command("/")

        assert result is None

    def test_parse_preserves_quotes(self):
        """Parse handles arguments with spaces."""
        result = parse_command("/echo hello world")

        assert result is not None
        assert result[0] == "echo"
        assert result[1] == ["hello", "world"]


class TestHandleCommand:
    """Tests for command handling."""

    def setup_method(self):
        """Reset globals before each test."""
        reset_command_registry()

    @pytest.mark.asyncio
    async def test_handle_command(self):
        """Handle valid command."""
        registry = get_command_registry()

        @registry.command("test", "Test command")
        async def test_cmd():
            return "test result"

        result = await handle_command("/test")

        assert result == "test result"

    @pytest.mark.asyncio
    async def test_handle_not_command(self):
        """Handle non-command returns None."""
        result = await handle_command("regular text")

        assert result is None

    @pytest.mark.asyncio
    async def test_handle_unknown_command(self):
        """Handle unknown command returns error."""
        result = await handle_command("/unknown_command")

        assert result is not None
        assert "Unknown command" in result


class TestBuiltinCommands:
    """Tests for built-in commands."""

    def setup_method(self):
        """Reset globals before each test."""
        reset_command_registry()
        reset_agent_registry()
        reset_hook_registry()
        reset_skill_loader()
        reset_subagent_coordinator()

    @pytest.mark.asyncio
    async def test_agents_command(self):
        """Test /agents command."""
        registry = get_command_registry()

        result = await registry.execute("agents")

        # Should not error, may have no agents
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_skills_command(self):
        """Test /skills command."""
        registry = get_command_registry()

        result = await registry.execute("skills")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_hooks_command(self):
        """Test /hooks command."""
        registry = get_command_registry()

        result = await registry.execute("hooks")

        assert "Registered hooks:" in result
        assert "Total:" in result

    @pytest.mark.asyncio
    async def test_spawn_command_usage(self):
        """Test /spawn command without args."""
        registry = get_command_registry()

        result = await registry.execute("spawn")

        assert "Usage:" in result

    @pytest.mark.asyncio
    async def test_help_command(self):
        """Test /help command."""
        registry = get_command_registry()

        result = await registry.execute("help")

        assert "Available commands:" in result
        assert "/agents" in result
        assert "/skills" in result
        assert "/hooks" in result
        assert "/spawn" in result
        assert "/help" in result


class TestCommandRegistrySingleton:
    """Tests for global command registry."""

    def setup_method(self):
        """Reset globals before each test."""
        reset_command_registry()

    def test_singleton(self):
        """Global registry returns same instance."""
        registry1 = get_command_registry()
        registry2 = get_command_registry()

        assert registry1 is registry2

    def test_has_builtin_commands(self):
        """Global registry has built-in commands."""
        registry = get_command_registry()

        assert registry.get("agents") is not None
        assert registry.get("skills") is not None
        assert registry.get("hooks") is not None
        assert registry.get("spawn") is not None
        assert registry.get("help") is not None
