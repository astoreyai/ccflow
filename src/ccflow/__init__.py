"""
ccflow - Claude Code CLI ↔ SDK Middleware

Production middleware bridging Claude Code CLI with SDK-like Python interfaces.
Enables subscription-based usage with TOON token optimization.

Example:
    >>> from ccflow import query, CLIAgentOptions
    >>> async for msg in query("Analyze this code", CLIAgentOptions(model="sonnet")):
    ...     print(msg.content, end="")
"""

from ccflow.api import batch_query, query
from ccflow.exceptions import (
    CCFlowError,
    CLIAuthenticationError,
    CLIExecutionError,
    CLINotFoundError,
    CLITimeoutError,
    ParseError,
    PermissionDeniedError,
    SessionNotFoundError,
    ToonEncodingError,
)
from ccflow.session import Session
from ccflow.types import (
    AssistantMessage,
    CLIAgentOptions,
    HookMessage,
    InitMessage,
    MCPServerConfig,
    Message,
    PermissionMode,
    QueryResult,
    ResultMessage,
    SessionStats,
    StopMessage,
    TextMessage,
    ToonConfig,
    ToolResultMessage,
    ToolUseMessage,
    UnknownMessage,
    UnknownSystemMessage,
    UserMessage,
)

__version__ = "0.1.0"
__all__ = [
    # Core API
    "query",
    "batch_query",
    "Session",
    # Configuration
    "CLIAgentOptions",
    "ToonConfig",
    "MCPServerConfig",
    "PermissionMode",
    # Message types
    "Message",
    "InitMessage",
    "TextMessage",
    "ToolUseMessage",
    "ToolResultMessage",
    "StopMessage",
    "HookMessage",
    "AssistantMessage",
    "UserMessage",
    "ResultMessage",
    "UnknownSystemMessage",
    "UnknownMessage",
    # Results
    "QueryResult",
    "SessionStats",
    # Exceptions
    "CCFlowError",
    "CLINotFoundError",
    "CLIAuthenticationError",
    "CLIExecutionError",
    "CLITimeoutError",
    "SessionNotFoundError",
    "ParseError",
    "ToonEncodingError",
    "PermissionDeniedError",
]
