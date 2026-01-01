"""Tests for stream parser."""

import pytest

from ccflow.parser import (
    StreamParser,
    collect_text,
    extract_session_id,
    extract_usage,
    parse_event,
)
from ccflow.types import (
    AssistantMessage,
    ErrorMessage,
    HookMessage,
    InitMessage,
    ResultMessage,
    StopMessage,
    TextMessage,
    ToolResultMessage,
    ToolUseMessage,
    UnknownMessage,
    UnknownSystemMessage,
    UserMessage,
)
from ccflow.exceptions import ParseError


class TestStreamParser:
    """Tests for StreamParser class."""

    def test_parse_init_message(self):
        """Test parsing init system message."""
        event = {"type": "system", "subtype": "init", "session_id": "abc123"}
        msg = parse_event(event)

        assert isinstance(msg, InitMessage)
        assert msg.session_id == "abc123"
        assert msg.type == "system"
        assert msg.subtype == "init"

    def test_parse_stop_message(self):
        """Test parsing stop system message."""
        event = {
            "type": "system",
            "subtype": "stop",
            "session_id": "abc123",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        msg = parse_event(event)

        assert isinstance(msg, StopMessage)
        assert msg.session_id == "abc123"
        assert msg.usage["input_tokens"] == 100
        assert msg.usage["output_tokens"] == 50

    def test_parse_text_message(self):
        """Test parsing text message."""
        event = {"type": "message", "content": "Hello, world!", "delta_type": "text_delta"}
        msg = parse_event(event)

        assert isinstance(msg, TextMessage)
        assert msg.content == "Hello, world!"
        assert msg.delta_type == "text_delta"

    def test_parse_tool_use_message(self):
        """Test parsing tool use message."""
        event = {
            "type": "tool_use",
            "tool": "Read",
            "args": {"file_path": "/path/to/file.py"},
        }
        msg = parse_event(event)

        assert isinstance(msg, ToolUseMessage)
        assert msg.tool == "Read"
        assert msg.args["file_path"] == "/path/to/file.py"

    def test_parse_tool_result_message(self):
        """Test parsing tool result message."""
        event = {"type": "tool_result", "tool": "Read", "content": "file contents here"}
        msg = parse_event(event)

        assert isinstance(msg, ToolResultMessage)
        assert msg.tool == "Read"
        assert msg.result == "file contents here"

    def test_parse_error_message(self):
        """Test parsing error message."""
        event = {"type": "error", "message": "Rate limit exceeded", "code": "rate_limit"}
        msg = parse_event(event)

        assert isinstance(msg, ErrorMessage)
        assert msg.message == "Rate limit exceeded"
        assert msg.code == "rate_limit"

    def test_parse_unknown_type_returns_unknown_message(self):
        """Test that unknown event type returns UnknownMessage."""
        event = {"type": "future_type", "data": "something"}
        msg = parse_event(event)

        assert isinstance(msg, UnknownMessage)
        assert msg.event_type == "future_type"
        assert msg.raw_data == event

    def test_parse_assistant_message(self):
        """Test parsing assistant response message."""
        event = {
            "type": "assistant",
            "message": {
                "model": "claude-sonnet-4-5-20250929",
                "id": "msg_123",
                "content": [{"type": "text", "text": "Hello, world!"}],
                "usage": {"input_tokens": 100, "output_tokens": 50},
                "stop_reason": "end_turn",
            },
            "session_id": "session-abc",
        }
        msg = parse_event(event)

        assert isinstance(msg, AssistantMessage)
        assert msg.model == "claude-sonnet-4-5-20250929"
        assert msg.message_id == "msg_123"
        assert msg.session_id == "session-abc"
        assert msg.text_content == "Hello, world!"
        assert msg.usage["input_tokens"] == 100
        assert msg.stop_reason == "end_turn"

    def test_parse_unknown_system_subtype_returns_unknown_message(self):
        """Test that unknown system subtype returns UnknownSystemMessage."""
        event = {"type": "system", "subtype": "future_subtype", "session_id": "abc", "extra": "data"}
        msg = parse_event(event)

        assert isinstance(msg, UnknownSystemMessage)
        assert msg.subtype == "future_subtype"
        assert msg.raw_data == event

    def test_parse_hook_response_message(self):
        """Test parsing hook_response system message."""
        event = {
            "type": "system",
            "subtype": "hook_response",
            "hook_type": "SessionStart",
            "content": "Hook executed successfully",
        }
        msg = parse_event(event)

        assert isinstance(msg, HookMessage)
        assert msg.hook_type == "SessionStart"
        assert msg.content == "Hook executed successfully"

    def test_parse_user_message(self):
        """Test parsing user message event."""
        event = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_123",
                        "content": "File contents here",
                    }
                ],
            },
            "session_id": "session-abc",
            "tool_use_result": {"tool_use_id": "tool_123", "content": "File contents here"},
        }
        msg = parse_event(event)

        assert isinstance(msg, UserMessage)
        assert msg.session_id == "session-abc"
        assert msg.type == "user"
        assert len(msg.content) == 1
        assert msg.content[0]["type"] == "tool_result"
        assert msg.content[0]["tool_use_id"] == "tool_123"
        assert msg.tool_use_result is not None
        assert msg.tool_use_result["tool_use_id"] == "tool_123"

    def test_parse_user_message_minimal(self):
        """Test parsing user message with minimal data."""
        event = {"type": "user"}
        msg = parse_event(event)

        assert isinstance(msg, UserMessage)
        assert msg.session_id == ""
        assert msg.content == []
        assert msg.tool_use_result is None

    def test_parse_result_message(self):
        """Test parsing result message event."""
        event = {
            "type": "result",
            "result": "Task completed successfully",
            "session_id": "session-abc",
            "duration_ms": 5432,
            "num_turns": 3,
            "total_cost_usd": 0.0125,
            "usage": {"input_tokens": 1500, "output_tokens": 500},
            "is_error": False,
        }
        msg = parse_event(event)

        assert isinstance(msg, ResultMessage)
        assert msg.result == "Task completed successfully"
        assert msg.session_id == "session-abc"
        assert msg.duration_ms == 5432
        assert msg.num_turns == 3
        assert msg.total_cost_usd == 0.0125
        assert msg.usage["input_tokens"] == 1500
        assert msg.usage["output_tokens"] == 500
        assert msg.is_error is False
        assert msg.type == "result"

    def test_parse_result_message_with_error(self):
        """Test parsing result message with error flag."""
        event = {
            "type": "result",
            "result": "Execution failed: rate limit exceeded",
            "session_id": "session-xyz",
            "duration_ms": 1000,
            "num_turns": 1,
            "total_cost_usd": 0.001,
            "usage": {"input_tokens": 100, "output_tokens": 0},
            "is_error": True,
        }
        msg = parse_event(event)

        assert isinstance(msg, ResultMessage)
        assert msg.is_error is True
        assert msg.result == "Execution failed: rate limit exceeded"

    def test_parse_result_message_minimal(self):
        """Test parsing result message with minimal/default data."""
        event = {"type": "result"}
        msg = parse_event(event)

        assert isinstance(msg, ResultMessage)
        assert msg.result == ""
        assert msg.session_id == ""
        assert msg.duration_ms == 0
        assert msg.num_turns == 0
        assert msg.total_cost_usd == 0.0
        assert msg.usage == {}
        assert msg.is_error is False


class TestConvenienceFunctions:
    """Tests for parser convenience functions."""

    def test_extract_session_id(self, simple_ndjson_response):
        """Test extracting session ID from events."""
        session_id = extract_session_id(simple_ndjson_response)
        assert session_id == "test-session-123"

    def test_extract_session_id_not_found(self):
        """Test extracting session ID when not present."""
        events = [{"type": "message", "content": "Hello"}]
        session_id = extract_session_id(events)
        assert session_id is None

    def test_extract_usage(self, simple_ndjson_response):
        """Test extracting usage from events."""
        usage = extract_usage(simple_ndjson_response)
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5

    def test_extract_usage_not_found(self):
        """Test extracting usage when not present."""
        events = [{"type": "message", "content": "Hello"}]
        usage = extract_usage(events)
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

    def test_collect_text(self):
        """Test collecting text from messages."""
        messages = [
            TextMessage(content="Hello, "),
            ToolUseMessage(tool="Read", args={}),
            TextMessage(content="world!"),
            StopMessage(session_id="abc", usage={}),
        ]
        text = collect_text(messages)
        assert text == "Hello, world!"

    def test_collect_text_empty(self):
        """Test collecting text with no text messages."""
        messages = [
            InitMessage(session_id="abc"),
            ToolUseMessage(tool="Read", args={}),
        ]
        text = collect_text(messages)
        assert text == ""
