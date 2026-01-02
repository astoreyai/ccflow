"""
Agent system for ccflow.

Provides Agent class for specialized execution contexts and
AgentRegistry for managing agent definitions.
"""

from __future__ import annotations

import builtins
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from .types import AgentDefinition, CLIAgentOptions, Message, PermissionMode

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .session import Session


# Default search paths for agent definitions
AGENT_SEARCH_PATHS = [
    Path.home() / ".claude" / "agents",  # User agents
    Path.cwd() / ".claude" / "agents",  # Project agents
]


def parse_yaml_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from markdown content.

    Args:
        content: Markdown content with optional YAML frontmatter

    Returns:
        Tuple of (metadata_dict, body_content)
    """
    if not content.startswith("---"):
        return {}, content

    # Find the closing ---
    end_match = re.search(r"\n---\s*\n", content[3:])
    if not end_match:
        return {}, content

    frontmatter_end = end_match.start() + 3
    frontmatter = content[3:frontmatter_end]
    body = content[frontmatter_end + end_match.end() - end_match.start() :]

    try:
        metadata = yaml.safe_load(frontmatter) or {}
    except yaml.YAMLError:
        metadata = {}

    return metadata, body.strip()


@dataclass
class AgentRegistry:
    """Registry for agent definitions with filesystem discovery.

    Manages agent definitions with support for:
    - Programmatic registration
    - Filesystem discovery from ~/.claude/agents/ and .claude/agents/
    - YAML frontmatter parsing for markdown agent files

    Example:
        >>> registry = AgentRegistry()
        >>> registry.register(AgentDefinition(
        ...     name="code-reviewer",
        ...     description="Reviews code for quality and security",
        ...     prompt="You are a senior code reviewer...",
        ...     tools=["Read", "Grep", "Glob"],
        ... ))
        >>> agent_def = registry.get("code-reviewer")
    """

    _agents: dict[str, AgentDefinition] = field(default_factory=dict)
    _search_paths: list[Path] = field(default_factory=lambda: list(AGENT_SEARCH_PATHS))
    _discovered: bool = field(default=False, repr=False)

    def register(self, agent: AgentDefinition) -> None:
        """Register an agent definition.

        Args:
            agent: Agent definition to register

        Raises:
            ValueError: If agent has no name
        """
        if not agent.name:
            raise ValueError("Agent must have a name")
        self._agents[agent.name] = agent

    def get(self, name: str) -> AgentDefinition | None:
        """Get agent definition by name.

        Auto-discovers agents from filesystem on first access.

        Args:
            name: Agent name to look up

        Returns:
            Agent definition or None if not found
        """
        if not self._discovered:
            self.discover()

        return self._agents.get(name)

    def list(self) -> builtins.list[str]:
        """List all registered agent names.

        Auto-discovers agents from filesystem on first access.

        Returns:
            List of agent names
        """
        if not self._discovered:
            self.discover()

        return builtins.list(self._agents.keys())

    def discover(self) -> builtins.list[AgentDefinition]:
        """Discover agents from filesystem paths.

        Searches for .md files in configured search paths and
        parses them as agent definitions.

        Returns:
            List of discovered agent definitions
        """
        discovered = []

        for path in self._search_paths:
            if not path.exists():
                continue

            for md_file in path.glob("*.md"):
                try:
                    agent = self._load_agent_file(md_file)
                    if agent:
                        self._agents[agent.name] = agent
                        discovered.append(agent)
                except Exception:
                    # Skip files that fail to parse
                    continue

        self._discovered = True
        return discovered

    def _load_agent_file(self, path: Path) -> AgentDefinition | None:
        """Load agent definition from markdown file.

        Expects format:
            ---
            name: agent-name
            description: When to use this agent
            tools:
              - Read
              - Grep
            model: sonnet
            ---
            System prompt content here...

        Args:
            path: Path to markdown file

        Returns:
            AgentDefinition or None if invalid
        """
        content = path.read_text()
        metadata, body = parse_yaml_frontmatter(content)

        # Extract required fields
        name = metadata.get("name", path.stem)
        description = metadata.get("description", "")

        if not description:
            return None

        return AgentDefinition(
            name=name,
            description=description,
            prompt=body,
            tools=metadata.get("tools"),
            model=metadata.get("model"),
            permission_mode=(
                PermissionMode(metadata["permission_mode"])
                if "permission_mode" in metadata
                else None
            ),
            timeout=metadata.get("timeout"),
            metadata={
                k: v
                for k, v in metadata.items()
                if k not in {"name", "description", "tools", "model", "permission_mode", "timeout"}
            },
        )

    def add_search_path(self, path: Path) -> None:
        """Add a search path for agent discovery.

        Args:
            path: Directory path to search
        """
        if path not in self._search_paths:
            self._search_paths.append(path)
            self._discovered = False  # Force re-discovery


# Global registry singleton
_global_registry: AgentRegistry | None = None


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry singleton.

    Returns:
        The global AgentRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = AgentRegistry()
    return _global_registry


def reset_agent_registry() -> None:
    """Reset the global agent registry (for testing)."""
    global _global_registry
    _global_registry = None


class Agent:
    """Specialized execution context with isolated configuration.

    An Agent wraps an AgentDefinition and provides execution methods
    that use the agent's specialized tools, model, and system prompt.

    Example:
        >>> agent = Agent(AgentDefinition(
        ...     name="code-reviewer",
        ...     description="Reviews code for quality",
        ...     prompt="You are a code reviewer...",
        ...     tools=["Read", "Grep"],
        ... ))
        >>> async for msg in agent.execute("Review auth.py"):
        ...     print(msg)
    """

    def __init__(
        self,
        definition: AgentDefinition | str,
        parent_options: CLIAgentOptions | None = None,
        registry: AgentRegistry | None = None,
    ):
        """Initialize an Agent.

        Args:
            definition: AgentDefinition or name to look up in registry
            parent_options: Parent options to inherit from
            registry: Registry to use for name lookup

        Raises:
            ValueError: If definition is a string and not found in registry
        """
        self._registry = registry or get_agent_registry()

        if isinstance(definition, str):
            resolved = self._registry.get(definition)
            if resolved is None:
                raise ValueError(f"Agent '{definition}' not found in registry")
            self.definition = resolved
        else:
            self.definition = definition

        self._parent_options = parent_options
        self._options = self._build_options()
        self._session: Session | None = None

    def _build_options(self) -> CLIAgentOptions:
        """Build options by merging agent definition with parent options."""
        # Start with parent options or defaults
        if self._parent_options:
            # Copy parent options
            from dataclasses import asdict

            opts = CLIAgentOptions(**asdict(self._parent_options))
        else:
            opts = CLIAgentOptions()

        # Override with agent-specific settings
        if self.definition.model:
            opts.model = self.definition.model if self.definition.model != "inherit" else opts.model

        if self.definition.tools is not None:
            opts.allowed_tools = self.definition.tools

        if self.definition.prompt:
            opts.system_prompt = self.definition.prompt

        if self.definition.permission_mode is not None:
            opts.permission_mode = self.definition.permission_mode

        if self.definition.timeout is not None:
            opts.timeout = self.definition.timeout

        return opts

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[Message]:
        """Execute a task with this agent's configuration.

        Args:
            task: The task/prompt to execute
            context: Optional context data to inject

        Yields:
            Message events from execution
        """
        # Import here to avoid circular imports
        from .session import Session

        if context:
            self._options.context = context

        self._session = Session(options=self._options)
        async for msg in self._session.send_message(task):
            yield msg

    async def execute_simple(self, task: str, context: dict[str, Any] | None = None) -> str:
        """Execute task and return aggregated text response.

        Args:
            task: The task/prompt to execute
            context: Optional context data to inject

        Returns:
            Concatenated text content from response
        """
        from .types import TextMessage

        result_parts = []
        async for msg in self.execute(task, context):
            if isinstance(msg, TextMessage):
                result_parts.append(msg.content)

        return "".join(result_parts)

    @property
    def options(self) -> CLIAgentOptions:
        """Get the agent's effective options."""
        return self._options

    @property
    def last_session(self) -> Session | None:
        """Get the session from the last execution."""
        return self._session
