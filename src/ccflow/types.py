"""
Type definitions for ccflow middleware.

Provides Pydantic models and dataclasses for configuration,
messages, and results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class PermissionMode(str, Enum):
    """CLI permission modes for tool execution.

    Valid modes from CLI: acceptEdits, bypassPermissions, default, delegate, dontAsk, plan
    """

    DEFAULT = "default"
    PLAN = "plan"
    DONT_ASK = "dontAsk"
    ACCEPT_EDITS = "acceptEdits"
    DELEGATE = "delegate"
    BYPASS = "bypassPermissions"


@dataclass
class ToonConfig:
    """Configuration for TOON serialization.

    TOON (Token-Oriented Object Notation) reduces token consumption
    by 30-60% for structured data compared to JSON.

    Attributes:
        enabled: Whether to use TOON encoding for context
        delimiter: Field separator (comma, tab, or pipe)
        indent: Spaces per indentation level
        length_marker: Include length markers in output
        encode_context: Auto-encode context objects
        encode_tool_results: Encode tool results in TOON
        track_savings: Track compression metrics
    """

    enabled: bool = True
    delimiter: Literal[",", "\t", "|"] = ","
    indent: int = 2
    length_marker: bool = False
    encode_context: bool = True
    encode_tool_results: bool = True
    track_savings: bool = True

    # Internal tracking (not part of public API)
    _last_json_tokens: int = field(default=0, repr=False)
    _last_toon_tokens: int = field(default=0, repr=False)

    @property
    def last_compression_ratio(self) -> float:
        """Returns token savings as fraction (0.0-1.0).

        Example:
            >>> config.last_compression_ratio
            0.45  # 45% token savings
        """
        if self._last_json_tokens == 0:
            return 0.0
        return 1.0 - (self._last_toon_tokens / self._last_json_tokens)


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server.

    Attributes:
        command: Executable command (e.g., "python", "npx")
        args: Command arguments
        env: Environment variables
        transport: Transport type (stdio, sse, http)
        url: URL for SSE/HTTP transports
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    transport: Literal["stdio", "sse", "http"] = "stdio"
    url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to MCP config JSON format."""
        config: dict[str, Any] = {"command": self.command}
        if self.args:
            config["args"] = self.args
        if self.env:
            config["env"] = self.env
        if self.transport != "stdio":
            config["type"] = self.transport
        if self.url:
            config["url"] = self.url
        return config


@dataclass
class CLIAgentOptions:
    """Configuration options for CLI middleware.

    Matches Claude Agent SDK ClaudeAgentOptions interface where possible,
    translating to CLI flags internally.

    Attributes:
        model: Model to use (sonnet, opus, haiku, or full name)
        fallback_model: Fallback model on overload
        system_prompt: Replace entire system prompt
        append_system_prompt: Append to default system prompt
        permission_mode: Tool permission mode
        allowed_tools: Tools to allow without prompting
        disallowed_tools: Tools to deny
        session_id: Specific session UUID
        resume: Whether to resume previous session
        fork_session: Create new branch from resumed session
        max_budget_usd: Maximum dollar amount to spend on API calls
        timeout: Execution timeout in seconds
        cwd: Working directory
        add_dirs: Additional directories to include
        mcp_servers: MCP server configurations
        strict_mcp: Only use specified MCP servers
        toon: TOON serialization config
        context: Structured data to inject (auto-TOON encoded)
        verbose: Enable verbose logging
        include_partial: Include partial streaming events
        debug: Enable debug mode (True for all, or filter string like "api,hooks")
        json_schema: JSON Schema for structured output validation
        input_format: Input format ("text" or "stream-json")
        dangerously_skip_permissions: Bypass all permission checks (sandboxed only)
        tools: Built-in tools to enable (empty list disables all)
        continue_session: Continue most recent conversation
        no_session_persistence: Don't save session to disk
        agent: Agent for the session
        agents: Custom agents JSON object
        betas: Beta headers for API requests
        settings: Path to settings JSON or JSON string
        plugin_dirs: Directories to load plugins from
        disable_slash_commands: Disable all slash commands
        ultrathink: Enable extended thinking mode (prepends "ultrathink" to prompt)
    """

    # Model selection
    model: str = "sonnet"
    fallback_model: str | None = None

    # System prompt
    system_prompt: str | None = None
    append_system_prompt: str | None = None

    # Permissions
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    allowed_tools: list[str] | None = None
    disallowed_tools: list[str] | None = None

    # Session management
    session_id: str | None = None
    resume: bool = False
    fork_session: bool = False

    # Execution limits
    max_budget_usd: float | None = None
    max_tokens: int | None = None  # Max output tokens (API only)
    timeout: float = 300.0

    # Working directory
    cwd: str | None = None
    add_dirs: list[str] | None = None

    # MCP configuration
    mcp_servers: dict[str, MCPServerConfig] | None = None
    strict_mcp: bool = False

    # TOON serialization
    toon: ToonConfig = field(default_factory=ToonConfig)

    # Context data (will be TOON-encoded if enabled)
    context: dict[str, Any] | None = None

    # Output options
    verbose: bool = False
    include_partial: bool = False

    # Debug options
    debug: str | bool | None = None  # True for all, or filter string like "api,hooks"

    # Structured output
    json_schema: str | dict | None = None  # JSON Schema for output validation

    # Input streaming
    input_format: Literal["text", "stream-json"] | None = None

    # Permission bypass (for sandboxed environments)
    dangerously_skip_permissions: bool = False

    # Tool specification
    tools: list[str] | None = None  # Built-in tools to enable ("" disables all, "default" uses all)

    # Session options
    continue_session: bool = False  # Continue most recent conversation
    no_session_persistence: bool = False  # Don't save session to disk

    # Agent configuration
    agent: str | None = None  # Agent for the session
    agents: dict | None = None  # Custom agents JSON object

    # Beta features
    betas: list[str] | None = None  # Beta headers for API requests

    # Settings
    settings: str | None = None  # Path to settings JSON or JSON string

    # Plugins
    plugin_dirs: list[str] | None = None  # Directories to load plugins from

    # Slash commands
    disable_slash_commands: bool = False

    # Extended thinking
    ultrathink: bool = False  # Enable extended thinking mode


# Message types matching CLI stream-json output


@dataclass
class InitMessage:
    """Initial system message with session ID."""

    session_id: str
    type: Literal["system"] = "system"
    subtype: Literal["init"] = "init"


@dataclass
class TextMessage:
    """Text content from assistant response."""

    content: str
    type: Literal["message"] = "message"
    delta_type: Literal["text_delta"] = "text_delta"


@dataclass
class ThinkingMessage:
    """Extended thinking content from assistant.

    Represents the internal reasoning process when ultrathink mode is enabled.
    Contains the model's chain-of-thought reasoning before generating the response.
    """

    content: str
    thinking_tokens: int = 0  # Tokens used for thinking
    type: Literal["thinking"] = "thinking"


@dataclass
class ToolUseMessage:
    """Tool invocation by assistant."""

    tool: str
    args: dict[str, Any]
    type: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultMessage:
    """Result from tool execution."""

    tool: str
    result: str
    type: Literal["tool_result"] = "tool_result"


@dataclass
class ErrorMessage:
    """Error during execution."""

    message: str
    type: Literal["error"] = "error"
    code: str | None = None


@dataclass
class StopMessage:
    """Final system message with usage stats."""

    session_id: str
    usage: dict[str, int]  # {input_tokens, output_tokens}
    type: Literal["system"] = "system"
    subtype: Literal["stop"] = "stop"


@dataclass
class HookMessage:
    """Hook response system event."""

    hook_type: str
    content: str | None = None
    type: Literal["system"] = "system"
    subtype: Literal["hook_response"] = "hook_response"


@dataclass
class UnknownSystemMessage:
    """Unknown system event (for forward compatibility)."""

    subtype: str
    raw_data: dict[str, Any]
    type: Literal["system"] = "system"


@dataclass
class AssistantMessage:
    """Full assistant response message.

    Contains the complete message object from the CLI including
    model info, content blocks, and usage statistics.
    """

    content: list[dict[str, Any]]  # Content blocks (text, tool_use, etc.)
    model: str
    message_id: str
    session_id: str
    usage: dict[str, Any]
    stop_reason: str | None = None
    type: Literal["assistant"] = "assistant"

    @property
    def text_content(self) -> str:
        """Extract concatenated text from content blocks."""
        parts = []
        for block in self.content:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)


@dataclass
class UnknownMessage:
    """Unknown event type (for forward compatibility)."""

    event_type: str
    raw_data: dict[str, Any]


@dataclass
class UserMessage:
    """User message event (typically tool results).

    Represents a user message in the conversation, which typically
    contains tool results or other user-provided content blocks.
    """

    content: list[dict[str, Any]]  # Content blocks (tool_result, etc.)
    session_id: str
    tool_use_result: dict[str, Any] | None = None
    type: Literal["user"] = "user"


@dataclass
class ResultMessage:
    """Final result summary with cost and usage stats.

    Represents the completion of a query with aggregate statistics
    about the execution including duration, turns, and costs.
    """

    result: str
    session_id: str
    duration_ms: int
    num_turns: int
    total_cost_usd: float
    usage: dict[str, Any]
    is_error: bool = False
    type: Literal["result"] = "result"


# Union type for all messages
Message = (
    InitMessage
    | TextMessage
    | ThinkingMessage
    | ToolUseMessage
    | ToolResultMessage
    | ErrorMessage
    | StopMessage
    | HookMessage
    | UnknownSystemMessage
    | AssistantMessage
    | UserMessage
    | ResultMessage
    | UnknownMessage
)


@dataclass
class QueryResult:
    """Result from a completed query.

    Returned by batch_query() for each prompt processed.
    """

    session_id: str
    result: str
    input_tokens: int
    output_tokens: int
    duration_seconds: float
    toon_savings_ratio: float = 0.0
    error: str | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens

    @property
    def success(self) -> bool:
        """Whether query completed without error."""
        return self.error is None


@dataclass
class SessionStats:
    """Statistics from a completed session.

    Returned by Session.close().
    """

    session_id: str
    total_turns: int
    total_input_tokens: int
    total_output_tokens: int
    duration_seconds: float
    toon_savings_ratio: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.total_input_tokens + self.total_output_tokens
