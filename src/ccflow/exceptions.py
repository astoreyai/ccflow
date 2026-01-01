"""
Custom exceptions for ccflow middleware.

Provides a hierarchy of exceptions for different error scenarios
during CLI execution and middleware operations.
"""

from __future__ import annotations


class CCFlowError(Exception):
    """Base exception for all ccflow errors."""

    pass


class CLINotFoundError(CCFlowError):
    """Claude CLI executable not found in PATH.

    Raised when the 'claude' command is not available.
    Install Claude Code CLI: npm install -g @anthropic-ai/claude-code
    """

    def __init__(self, message: str = "Claude CLI not found in PATH") -> None:
        super().__init__(message)


class CLIAuthenticationError(CCFlowError):
    """Claude CLI not authenticated.

    Raised when CLI requires authentication.
    Run 'claude' interactively to authenticate.
    """

    def __init__(self, message: str = "Claude CLI not authenticated") -> None:
        super().__init__(message)


class CLIExecutionError(CCFlowError):
    """CLI subprocess execution failed.

    Attributes:
        stderr: Standard error output from CLI
        exit_code: Process exit code
    """

    def __init__(self, message: str, stderr: str = "", exit_code: int = 1) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.exit_code = exit_code

    def __str__(self) -> str:
        base = super().__str__()
        if self.stderr:
            return f"{base}\nstderr: {self.stderr}"
        return base


class CLITimeoutError(CCFlowError):
    """CLI execution exceeded timeout.

    Attributes:
        timeout: The timeout value in seconds
        partial_output: Any output received before timeout
    """

    def __init__(
        self,
        timeout: float,
        partial_output: str = "",
        message: str | None = None,
    ) -> None:
        self.timeout = timeout
        self.partial_output = partial_output
        msg = message or f"CLI execution exceeded {timeout}s timeout"
        super().__init__(msg)


class SessionNotFoundError(CCFlowError):
    """Requested session ID not found.

    Raised when attempting to resume a non-existent session.
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class ParseError(CCFlowError):
    """Failed to parse CLI output.

    Raised when NDJSON parsing fails or unexpected format received.

    Attributes:
        line: The problematic line content
        line_number: Line number in stream (if known)
    """

    def __init__(
        self,
        message: str,
        line: str = "",
        line_number: int | None = None,
    ) -> None:
        super().__init__(message)
        self.line = line
        self.line_number = line_number

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.line_number is not None:
            parts.append(f"line {self.line_number}")
        if self.line:
            parts.append(f"content: {self.line[:100]}")
        return " | ".join(parts)


class ToonEncodingError(CCFlowError):
    """TOON encoding/decoding failed.

    Raised when data cannot be serialized to/from TOON format.

    Attributes:
        data_type: Type of data that failed encoding
    """

    def __init__(self, message: str, data_type: str | None = None) -> None:
        super().__init__(message)
        self.data_type = data_type


class PermissionDeniedError(CCFlowError):
    """Tool permission was denied.

    Raised when a tool execution is blocked by permission settings.

    Attributes:
        tool: The tool that was denied
        reason: Why permission was denied
    """

    def __init__(self, tool: str, reason: str = "Permission denied") -> None:
        self.tool = tool
        self.reason = reason
        super().__init__(f"Permission denied for tool '{tool}': {reason}")


class RateLimitError(CCFlowError):
    """Rate limit exceeded.

    Raised when subscription quota or rate limits are hit.

    Attributes:
        retry_after: Seconds to wait before retry (if known)
    """

    def __init__(self, message: str = "Rate limit exceeded", retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class MCPConfigError(CCFlowError):
    """MCP configuration error.

    Raised when MCP server configuration is invalid or server fails to start.

    Attributes:
        server_name: Name of the problematic server
    """

    def __init__(self, message: str, server_name: str | None = None) -> None:
        super().__init__(message)
        self.server_name = server_name


class SessionStoreError(CCFlowError):
    """Session storage operation failed.

    Raised when database operations fail in SessionStore implementations.

    Attributes:
        operation: The operation that failed (save, load, delete, etc.)
    """

    def __init__(self, message: str, operation: str | None = None) -> None:
        super().__init__(message)
        self.operation = operation
