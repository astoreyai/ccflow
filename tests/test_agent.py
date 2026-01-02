"""
Tests for agent system.

Tests AgentDefinition, Agent, and AgentRegistry classes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ccflow.agent import (
    Agent,
    AgentRegistry,
    get_agent_registry,
    parse_yaml_frontmatter,
    reset_agent_registry,
)
from ccflow.types import AgentDefinition, CLIAgentOptions, PermissionMode


class TestParseYamlFrontmatter:
    """Tests for YAML frontmatter parsing."""

    def test_parse_with_frontmatter(self):
        """Parse content with valid frontmatter."""
        content = """---
name: test-agent
description: A test agent
tools:
  - Read
  - Grep
---
This is the body content.
"""
        metadata, body = parse_yaml_frontmatter(content)

        assert metadata["name"] == "test-agent"
        assert metadata["description"] == "A test agent"
        assert metadata["tools"] == ["Read", "Grep"]
        assert body == "This is the body content."

    def test_parse_without_frontmatter(self):
        """Parse content without frontmatter."""
        content = "Just body content without frontmatter."
        metadata, body = parse_yaml_frontmatter(content)

        assert metadata == {}
        assert body == content

    def test_parse_empty_frontmatter(self):
        """Parse content with empty frontmatter."""
        content = """---
---
Body after empty frontmatter.
"""
        metadata, body = parse_yaml_frontmatter(content)

        assert metadata == {}
        assert "Body after empty frontmatter." in body

    def test_parse_invalid_yaml(self):
        """Parse content with invalid YAML in frontmatter."""
        content = """---
invalid: yaml: content: here
---
Body content.
"""
        metadata, body = parse_yaml_frontmatter(content)

        # Should return empty metadata on invalid YAML
        assert metadata == {}


class TestAgentDefinition:
    """Tests for AgentDefinition dataclass."""

    def test_create_minimal(self):
        """Create agent with only required fields."""
        agent = AgentDefinition(
            description="A test agent",
            prompt="You are a test agent.",
        )

        assert agent.description == "A test agent"
        assert agent.prompt == "You are a test agent."
        assert agent.tools is None
        assert agent.model is None
        assert agent.name == ""

    def test_create_full(self):
        """Create agent with all fields."""
        agent = AgentDefinition(
            name="code-reviewer",
            description="Reviews code for quality",
            prompt="You are a code reviewer.",
            tools=["Read", "Grep", "Glob"],
            model="opus",
            permission_mode=PermissionMode.ACCEPT_EDITS,
            timeout=60.0,
            metadata={"version": "1.0"},
        )

        assert agent.name == "code-reviewer"
        assert agent.description == "Reviews code for quality"
        assert agent.tools == ["Read", "Grep", "Glob"]
        assert agent.model == "opus"
        assert agent.permission_mode == PermissionMode.ACCEPT_EDITS
        assert agent.timeout == 60.0
        assert agent.metadata == {"version": "1.0"}

    def test_to_sdk_dict_minimal(self):
        """Convert minimal agent to SDK dict."""
        agent = AgentDefinition(
            description="Test agent",
            prompt="Test prompt",
        )

        sdk_dict = agent.to_sdk_dict()

        assert sdk_dict == {
            "description": "Test agent",
            "prompt": "Test prompt",
        }

    def test_to_sdk_dict_full(self):
        """Convert full agent to SDK dict."""
        agent = AgentDefinition(
            name="test",
            description="Test agent",
            prompt="Test prompt",
            tools=["Read", "Write"],
            model="sonnet",
        )

        sdk_dict = agent.to_sdk_dict()

        assert sdk_dict == {
            "description": "Test agent",
            "prompt": "Test prompt",
            "tools": ["Read", "Write"],
            "model": "sonnet",
        }


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_agent_registry()

    def test_register_and_get(self):
        """Register and retrieve an agent."""
        registry = AgentRegistry()

        agent = AgentDefinition(
            name="test-agent",
            description="A test agent",
            prompt="Test prompt",
        )
        registry.register(agent)

        retrieved = registry.get("test-agent")
        assert retrieved is not None
        assert retrieved.name == "test-agent"
        assert retrieved.description == "A test agent"

    def test_register_without_name_raises(self):
        """Registering agent without name raises error."""
        registry = AgentRegistry()

        agent = AgentDefinition(
            description="No name agent",
            prompt="Test",
        )

        with pytest.raises(ValueError, match="must have a name"):
            registry.register(agent)

    def test_get_nonexistent_returns_none(self):
        """Getting nonexistent agent returns None."""
        registry = AgentRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_list_agents(self):
        """List registered agents."""
        registry = AgentRegistry()
        registry._search_paths = []  # Clear default paths for test isolation

        registry.register(AgentDefinition(
            name="agent-a",
            description="Agent A",
            prompt="Prompt A",
        ))
        registry.register(AgentDefinition(
            name="agent-b",
            description="Agent B",
            prompt="Prompt B",
        ))

        names = registry.list()
        assert set(names) == {"agent-a", "agent-b"}

    def test_discover_from_filesystem(self):
        """Discover agents from filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()

            # Create agent file
            agent_file = agents_dir / "my-agent.md"
            agent_file.write_text("""---
name: my-agent
description: My custom agent
tools:
  - Read
  - Grep
model: sonnet
---
You are a custom agent that helps with code review.
""")

            registry = AgentRegistry()
            registry._search_paths = [agents_dir]

            discovered = registry.discover()

            assert len(discovered) == 1
            agent = discovered[0]
            assert agent.name == "my-agent"
            assert agent.description == "My custom agent"
            assert agent.tools == ["Read", "Grep"]
            assert agent.model == "sonnet"
            assert "custom agent" in agent.prompt

    def test_discover_uses_filename_as_name(self):
        """Discover uses filename as name when not specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            agents_dir = Path(tmpdir) / "agents"
            agents_dir.mkdir()

            agent_file = agents_dir / "code-reviewer.md"
            agent_file.write_text("""---
description: Reviews code
---
Review code carefully.
""")

            registry = AgentRegistry()
            registry._search_paths = [agents_dir]

            discovered = registry.discover()

            assert len(discovered) == 1
            assert discovered[0].name == "code-reviewer"

    def test_add_search_path(self):
        """Add custom search path."""
        registry = AgentRegistry()
        custom_path = Path("/custom/agents")

        registry.add_search_path(custom_path)

        assert custom_path in registry._search_paths

    def test_global_registry_singleton(self):
        """Global registry returns same instance."""
        reset_agent_registry()

        registry1 = get_agent_registry()
        registry2 = get_agent_registry()

        assert registry1 is registry2


class TestAgent:
    """Tests for Agent class."""

    def setup_method(self):
        """Reset global registry before each test."""
        reset_agent_registry()

    def test_create_with_definition(self):
        """Create agent with AgentDefinition."""
        definition = AgentDefinition(
            name="test-agent",
            description="Test agent",
            prompt="You are a test agent.",
            tools=["Read"],
            model="sonnet",
        )

        agent = Agent(definition)

        assert agent.definition.name == "test-agent"
        assert agent.options.model == "sonnet"
        assert agent.options.allowed_tools == ["Read"]
        assert agent.options.system_prompt == "You are a test agent."

    def test_create_with_name_lookup(self):
        """Create agent by name lookup in registry."""
        registry = get_agent_registry()
        registry.register(AgentDefinition(
            name="registered-agent",
            description="A registered agent",
            prompt="Registered prompt",
        ))

        agent = Agent("registered-agent")

        assert agent.definition.name == "registered-agent"

    def test_create_with_unknown_name_raises(self):
        """Creating agent with unknown name raises error."""
        with pytest.raises(ValueError, match="not found"):
            Agent("unknown-agent")

    def test_inherits_parent_options(self):
        """Agent inherits from parent options."""
        parent_opts = CLIAgentOptions(
            model="opus",
            max_budget_usd=10.0,
            timeout=120.0,
        )

        definition = AgentDefinition(
            name="child-agent",
            description="Child agent",
            prompt="Child prompt",
        )

        agent = Agent(definition, parent_options=parent_opts)

        # Should inherit parent values
        assert agent.options.max_budget_usd == 10.0
        assert agent.options.timeout == 120.0
        # Model should still be overridden by agent (but agent has no model, so inherits)
        # Actually, agent has no model so parent model is kept
        # Wait, the definition has no model, so parent's opus should be kept
        # But the _build_options doesn't explicitly copy model if agent.model is None
        # Let me check the implementation...
        # Actually looking at the code, if definition.model is None, we don't override
        # So parent's opus should be kept
        # But wait, we start with a copy of parent_opts which has model="opus"
        # And only override if definition.model exists and is not "inherit"
        # Since definition.model is None, we don't override, so opus is kept

    def test_overrides_parent_with_agent_settings(self):
        """Agent settings override parent options."""
        parent_opts = CLIAgentOptions(
            model="opus",
            allowed_tools=["Read", "Write", "Bash"],
            timeout=120.0,
        )

        definition = AgentDefinition(
            name="child-agent",
            description="Child agent",
            prompt="Restricted agent",
            tools=["Read"],  # More restrictive
            model="haiku",  # Override model
        )

        agent = Agent(definition, parent_options=parent_opts)

        # Agent settings override parent
        assert agent.options.model == "haiku"
        assert agent.options.allowed_tools == ["Read"]
        # Parent settings preserved where not overridden
        assert agent.options.timeout == 120.0

    def test_model_inherit_keeps_parent_model(self):
        """Model 'inherit' keeps parent model."""
        parent_opts = CLIAgentOptions(model="opus")

        definition = AgentDefinition(
            name="inherit-agent",
            description="Inherits model",
            prompt="Test",
            model="inherit",
        )

        agent = Agent(definition, parent_options=parent_opts)

        assert agent.options.model == "opus"

    def test_permission_mode_override(self):
        """Agent can override permission mode."""
        definition = AgentDefinition(
            name="permissive-agent",
            description="Permissive agent",
            prompt="Test",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )

        agent = Agent(definition)

        assert agent.options.permission_mode == PermissionMode.ACCEPT_EDITS

    def test_timeout_override(self):
        """Agent can override timeout."""
        definition = AgentDefinition(
            name="quick-agent",
            description="Quick agent",
            prompt="Be fast",
            timeout=30.0,
        )

        agent = Agent(definition)

        assert agent.options.timeout == 30.0

    def test_custom_registry(self):
        """Use custom registry for name lookup."""
        custom_registry = AgentRegistry()
        custom_registry.register(AgentDefinition(
            name="custom-agent",
            description="Custom registry agent",
            prompt="Custom prompt",
        ))

        agent = Agent("custom-agent", registry=custom_registry)

        assert agent.definition.name == "custom-agent"


class TestAgentExecution:
    """Tests for Agent execution methods."""

    @pytest.mark.asyncio
    async def test_execute_simple_returns_text(self, mock_subprocess):
        """execute_simple returns aggregated text."""
        definition = AgentDefinition(
            name="test-agent",
            description="Test agent",
            prompt="You are helpful.",
        )

        agent = Agent(definition)
        result = await agent.execute_simple("Hello")

        # From mock_subprocess fixture, response is "Hello, world!"
        assert "Hello" in result or "world" in result

    @pytest.mark.asyncio
    async def test_execute_yields_messages(self, mock_subprocess):
        """execute yields message stream."""
        definition = AgentDefinition(
            name="test-agent",
            description="Test agent",
            prompt="You are helpful.",
        )

        agent = Agent(definition)
        messages = []

        async for msg in agent.execute("Hello"):
            messages.append(msg)

        # Should have multiple messages
        assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_execute_with_context(self, mock_subprocess):
        """execute passes context to options."""
        definition = AgentDefinition(
            name="test-agent",
            description="Test agent",
            prompt="You are helpful.",
        )

        agent = Agent(definition)

        async for _ in agent.execute("Hello", context={"key": "value"}):
            pass

        # Context should be set on options
        assert agent.options.context == {"key": "value"}

    @pytest.mark.asyncio
    async def test_last_session_available_after_execute(self, mock_subprocess):
        """last_session is available after execution."""
        definition = AgentDefinition(
            name="test-agent",
            description="Test agent",
            prompt="You are helpful.",
        )

        agent = Agent(definition)

        # Before execution
        assert agent.last_session is None

        async for _ in agent.execute("Hello"):
            pass

        # After execution
        assert agent.last_session is not None
