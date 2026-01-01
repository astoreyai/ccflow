"""Tests for ccflow custom exceptions."""

import pytest

from ccflow.exceptions import (
    CCFlowError,
    CLIAuthenticationError,
    CLIExecutionError,
    CLINotFoundError,
    CLITimeoutError,
    MCPConfigError,
    ParseError,
    PermissionDeniedError,
    RateLimitError,
    SessionNotFoundError,
    ToonEncodingError,
)


class TestCCFlowError:
    """Tests for base CCFlowError."""

    def test_base_error_with_message(self):
        """Test CCFlowError with message."""
        err = CCFlowError("Test error")
        assert str(err) == "Test error"

    def test_base_error_inheritance(self):
        """Test CCFlowError inherits from Exception."""
        assert issubclass(CCFlowError, Exception)


class TestCLINotFoundError:
    """Tests for CLINotFoundError."""

    def test_default_message(self):
        """Test default error message."""
        err = CLINotFoundError()
        assert "not found" in str(err).lower()

    def test_custom_message(self):
        """Test custom error message."""
        err = CLINotFoundError("Custom: CLI missing")
        assert str(err) == "Custom: CLI missing"

    def test_inheritance(self):
        """Test CLINotFoundError inherits from CCFlowError."""
        assert issubclass(CLINotFoundError, CCFlowError)


class TestCLIAuthenticationError:
    """Tests for CLIAuthenticationError."""

    def test_default_message(self):
        """Test default error message."""
        err = CLIAuthenticationError()
        assert "not authenticated" in str(err).lower()

    def test_custom_message(self):
        """Test custom error message."""
        err = CLIAuthenticationError("Please run 'claude' to login")
        assert str(err) == "Please run 'claude' to login"

    def test_inheritance(self):
        """Test CLIAuthenticationError inherits from CCFlowError."""
        assert issubclass(CLIAuthenticationError, CCFlowError)


class TestCLIExecutionError:
    """Tests for CLIExecutionError."""

    def test_basic_error(self):
        """Test basic error creation."""
        err = CLIExecutionError("Command failed")
        assert str(err) == "Command failed"
        assert err.stderr == ""
        assert err.exit_code == 1

    def test_with_stderr(self):
        """Test error with stderr output."""
        err = CLIExecutionError("Command failed", stderr="Error: invalid flag")
        assert "Command failed" in str(err)
        assert "stderr: Error: invalid flag" in str(err)

    def test_with_exit_code(self):
        """Test error with custom exit code."""
        err = CLIExecutionError("Segfault", exit_code=139)
        assert err.exit_code == 139

    def test_str_without_stderr(self):
        """Test __str__ without stderr."""
        err = CLIExecutionError("Failed", stderr="")
        assert str(err) == "Failed"

    def test_str_with_stderr(self):
        """Test __str__ includes stderr."""
        err = CLIExecutionError("Failed", stderr="Some error output")
        result = str(err)
        assert "Failed" in result
        assert "stderr:" in result
        assert "Some error output" in result


class TestCLITimeoutError:
    """Tests for CLITimeoutError."""

    def test_basic_timeout(self):
        """Test basic timeout error."""
        err = CLITimeoutError(30.0)
        assert err.timeout == 30.0
        assert "30" in str(err)
        assert "timeout" in str(err).lower()

    def test_with_partial_output(self):
        """Test timeout with partial output captured."""
        err = CLITimeoutError(60.0, partial_output="Partial response...")
        assert err.partial_output == "Partial response..."
        assert err.timeout == 60.0

    def test_custom_message(self):
        """Test timeout with custom message."""
        err = CLITimeoutError(120.0, message="Custom timeout message")
        assert str(err) == "Custom timeout message"
        assert err.timeout == 120.0

    def test_default_message_format(self):
        """Test default message includes timeout value."""
        err = CLITimeoutError(45.5)
        assert "45.5s" in str(err)


class TestSessionNotFoundError:
    """Tests for SessionNotFoundError."""

    def test_session_id_in_message(self):
        """Test session ID included in message."""
        err = SessionNotFoundError("session-abc-123")
        assert err.session_id == "session-abc-123"
        assert "session-abc-123" in str(err)
        assert "not found" in str(err).lower()

    def test_inheritance(self):
        """Test SessionNotFoundError inherits from CCFlowError."""
        assert issubclass(SessionNotFoundError, CCFlowError)


class TestParseError:
    """Tests for ParseError."""

    def test_basic_error(self):
        """Test basic parse error."""
        err = ParseError("Invalid JSON")
        assert "Invalid JSON" in str(err)
        assert err.line == ""
        assert err.line_number is None

    def test_with_line_content(self):
        """Test parse error with problematic line."""
        err = ParseError("Parse failed", line="{invalid json}")
        assert err.line == "{invalid json}"
        assert "content:" in str(err)

    def test_with_line_number(self):
        """Test parse error with line number."""
        err = ParseError("Unexpected token", line_number=42)
        assert err.line_number == 42
        assert "line 42" in str(err)

    def test_str_with_all_fields(self):
        """Test __str__ with line and line_number."""
        err = ParseError("Bad format", line="corrupted data", line_number=15)
        result = str(err)
        assert "Bad format" in result
        assert "line 15" in result
        assert "content:" in result
        assert "corrupted" in result

    def test_long_line_truncated(self):
        """Test long line content is truncated in __str__."""
        long_line = "x" * 200
        err = ParseError("Error", line=long_line)
        result = str(err)
        # Should only show first 100 chars
        assert len(result) < len(long_line) + 50


class TestToonEncodingError:
    """Tests for ToonEncodingError."""

    def test_basic_error(self):
        """Test basic encoding error."""
        err = ToonEncodingError("Cannot encode")
        assert str(err) == "Cannot encode"
        assert err.data_type is None

    def test_with_data_type(self):
        """Test encoding error with data type."""
        err = ToonEncodingError("Unsupported type", data_type="datetime")
        assert err.data_type == "datetime"
        assert str(err) == "Unsupported type"

    def test_inheritance(self):
        """Test ToonEncodingError inherits from CCFlowError."""
        assert issubclass(ToonEncodingError, CCFlowError)


class TestPermissionDeniedError:
    """Tests for PermissionDeniedError."""

    def test_basic_error(self):
        """Test basic permission denied error."""
        err = PermissionDeniedError("Bash")
        assert err.tool == "Bash"
        assert "Bash" in str(err)
        assert "permission denied" in str(err).lower()

    def test_with_reason(self):
        """Test permission denied with custom reason."""
        err = PermissionDeniedError("Write", reason="File in protected directory")
        assert err.tool == "Write"
        assert err.reason == "File in protected directory"
        assert "Write" in str(err)
        assert "File in protected directory" in str(err)

    def test_default_reason(self):
        """Test default reason is used."""
        err = PermissionDeniedError("Edit")
        assert err.reason == "Permission denied"


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_default_message(self):
        """Test default rate limit message."""
        err = RateLimitError()
        assert "rate limit" in str(err).lower()
        assert err.retry_after is None

    def test_custom_message(self):
        """Test custom rate limit message."""
        err = RateLimitError("Quota exceeded for today")
        assert str(err) == "Quota exceeded for today"

    def test_with_retry_after(self):
        """Test rate limit with retry-after value."""
        err = RateLimitError("Too many requests", retry_after=60.0)
        assert err.retry_after == 60.0

    def test_inheritance(self):
        """Test RateLimitError inherits from CCFlowError."""
        assert issubclass(RateLimitError, CCFlowError)


class TestMCPConfigError:
    """Tests for MCPConfigError."""

    def test_basic_error(self):
        """Test basic MCP config error."""
        err = MCPConfigError("Invalid configuration")
        assert str(err) == "Invalid configuration"
        assert err.server_name is None

    def test_with_server_name(self):
        """Test MCP config error with server name."""
        err = MCPConfigError("Server failed to start", server_name="github")
        assert err.server_name == "github"
        assert str(err) == "Server failed to start"

    def test_inheritance(self):
        """Test MCPConfigError inherits from CCFlowError."""
        assert issubclass(MCPConfigError, CCFlowError)


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_base(self):
        """Test all exceptions inherit from CCFlowError."""
        exceptions = [
            CLINotFoundError,
            CLIAuthenticationError,
            CLIExecutionError,
            CLITimeoutError,
            SessionNotFoundError,
            ParseError,
            ToonEncodingError,
            PermissionDeniedError,
            RateLimitError,
            MCPConfigError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, CCFlowError), f"{exc_class.__name__} should inherit from CCFlowError"

    def test_exceptions_can_be_caught_as_base(self):
        """Test all exceptions can be caught as CCFlowError."""
        try:
            raise CLIExecutionError("test")
        except CCFlowError as e:
            assert isinstance(e, CLIExecutionError)

        try:
            raise ParseError("test")
        except CCFlowError as e:
            assert isinstance(e, ParseError)

    def test_exceptions_are_also_exceptions(self):
        """Test all exceptions are proper Python exceptions."""
        exceptions = [
            CLINotFoundError(),
            CLIAuthenticationError(),
            CLIExecutionError("test"),
            CLITimeoutError(30),
            SessionNotFoundError("sess"),
            ParseError("test"),
            ToonEncodingError("test"),
            PermissionDeniedError("tool"),
            RateLimitError(),
            MCPConfigError("test"),
        ]
        for exc in exceptions:
            assert isinstance(exc, Exception)
