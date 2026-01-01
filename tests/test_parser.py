"""Tests for stream parser."""

import pytest

from ccflow.parser import (
    StreamParser,
    collect_text,
    collect_thinking,
    extract_session_id,
    extract_thinking_from_assistant,
    extract_thinking_tokens,
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
    ThinkingMessage,
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


class TestThinkingParsing:
    """Tests for ultrathink/extended thinking parsing."""

    def test_parse_thinking_message(self):
        """Test parsing thinking event."""
        event = {
            "type": "thinking",
            "content": "Let me analyze this step by step...",
            "thinking_tokens": 150,
        }
        msg = parse_event(event)

        assert isinstance(msg, ThinkingMessage)
        assert msg.content == "Let me analyze this step by step..."
        assert msg.thinking_tokens == 150
        assert msg.type == "thinking"

    def test_parse_thinking_message_with_thinking_field(self):
        """Test parsing thinking event with 'thinking' field instead of 'content'."""
        event = {
            "type": "thinking",
            "thinking": "First, I need to consider the implications...",
            "thinking_tokens": 200,
        }
        msg = parse_event(event)

        assert isinstance(msg, ThinkingMessage)
        assert msg.content == "First, I need to consider the implications..."
        assert msg.thinking_tokens == 200

    def test_parse_thinking_message_minimal(self):
        """Test parsing thinking event with minimal data."""
        event = {"type": "thinking"}
        msg = parse_event(event)

        assert isinstance(msg, ThinkingMessage)
        assert msg.content == ""
        assert msg.thinking_tokens == 0

    def test_parse_thinking_message_zero_tokens(self):
        """Test parsing thinking event with content but no token count."""
        event = {
            "type": "thinking",
            "content": "Some reasoning here",
        }
        msg = parse_event(event)

        assert isinstance(msg, ThinkingMessage)
        assert msg.content == "Some reasoning here"
        assert msg.thinking_tokens == 0


class TestCollectThinking:
    """Tests for collect_thinking function."""

    def test_collect_thinking_single(self):
        """Test collecting thinking from single message."""
        messages = [
            ThinkingMessage(content="Step 1: Analyze the problem", thinking_tokens=50),
        ]
        thinking = collect_thinking(messages)
        assert thinking == "Step 1: Analyze the problem"

    def test_collect_thinking_multiple(self):
        """Test collecting thinking from multiple messages."""
        messages = [
            ThinkingMessage(content="First, ", thinking_tokens=10),
            TextMessage(content="Here is my response"),
            ThinkingMessage(content="let me think ", thinking_tokens=15),
            ThinkingMessage(content="about this.", thinking_tokens=12),
        ]
        thinking = collect_thinking(messages)
        assert thinking == "First, let me think about this."

    def test_collect_thinking_mixed_messages(self):
        """Test collecting thinking ignores non-thinking messages."""
        messages = [
            InitMessage(session_id="abc"),
            ThinkingMessage(content="Reasoning here", thinking_tokens=100),
            TextMessage(content="Response text"),
            ToolUseMessage(tool="Read", args={}),
            ThinkingMessage(content=" and more reasoning", thinking_tokens=50),
            StopMessage(session_id="abc", usage={}),
        ]
        thinking = collect_thinking(messages)
        assert thinking == "Reasoning here and more reasoning"

    def test_collect_thinking_empty(self):
        """Test collecting thinking with no thinking messages."""
        messages = [
            InitMessage(session_id="abc"),
            TextMessage(content="Hello"),
            StopMessage(session_id="abc", usage={}),
        ]
        thinking = collect_thinking(messages)
        assert thinking == ""

    def test_collect_thinking_empty_list(self):
        """Test collecting thinking from empty list."""
        thinking = collect_thinking([])
        assert thinking == ""


class TestExtractThinkingFromAssistant:
    """Tests for extract_thinking_from_assistant function."""

    def test_extract_thinking_from_content_blocks(self):
        """Test extracting thinking from assistant message content blocks."""
        msg = AssistantMessage(
            content=[
                {"type": "thinking", "thinking": "Let me analyze this..."},
                {"type": "text", "text": "Here is my answer."},
            ],
            model="claude-sonnet-4-5-20250929",
            message_id="msg_123",
            session_id="session-abc",
            usage={},
        )
        thinking = extract_thinking_from_assistant(msg)
        assert thinking == "Let me analyze this..."

    def test_extract_thinking_multiple_blocks(self):
        """Test extracting thinking from multiple thinking blocks."""
        msg = AssistantMessage(
            content=[
                {"type": "thinking", "thinking": "First thought. "},
                {"type": "text", "text": "Some text."},
                {"type": "thinking", "thinking": "Second thought."},
            ],
            model="claude-sonnet-4-5-20250929",
            message_id="msg_123",
            session_id="session-abc",
            usage={},
        )
        thinking = extract_thinking_from_assistant(msg)
        assert thinking == "First thought. Second thought."

    def test_extract_thinking_no_thinking_blocks(self):
        """Test extracting thinking when no thinking blocks present."""
        msg = AssistantMessage(
            content=[
                {"type": "text", "text": "Here is my response."},
            ],
            model="claude-sonnet-4-5-20250929",
            message_id="msg_123",
            session_id="session-abc",
            usage={},
        )
        thinking = extract_thinking_from_assistant(msg)
        assert thinking == ""

    def test_extract_thinking_empty_content(self):
        """Test extracting thinking from empty content."""
        msg = AssistantMessage(
            content=[],
            model="claude-sonnet-4-5-20250929",
            message_id="msg_123",
            session_id="session-abc",
            usage={},
        )
        thinking = extract_thinking_from_assistant(msg)
        assert thinking == ""

    def test_extract_thinking_missing_thinking_field(self):
        """Test extracting thinking when thinking block has no thinking field."""
        msg = AssistantMessage(
            content=[
                {"type": "thinking"},  # Missing 'thinking' field
                {"type": "text", "text": "Response"},
            ],
            model="claude-sonnet-4-5-20250929",
            message_id="msg_123",
            session_id="session-abc",
            usage={},
        )
        thinking = extract_thinking_from_assistant(msg)
        assert thinking == ""


class TestExtractThinkingTokens:
    """Tests for extract_thinking_tokens function."""

    def test_extract_thinking_tokens_from_thinking_event(self):
        """Test extracting thinking tokens from thinking events."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "abc"},
            {"type": "thinking", "content": "Reasoning...", "thinking_tokens": 150},
            {"type": "message", "content": "Response"},
            {"type": "system", "subtype": "stop", "session_id": "abc", "usage": {}},
        ]
        tokens = extract_thinking_tokens(events)
        assert tokens == 150

    def test_extract_thinking_tokens_multiple_events(self):
        """Test extracting thinking tokens from multiple thinking events."""
        events = [
            {"type": "thinking", "content": "First", "thinking_tokens": 100},
            {"type": "thinking", "content": "Second", "thinking_tokens": 75},
            {"type": "thinking", "content": "Third", "thinking_tokens": 50},
        ]
        tokens = extract_thinking_tokens(events)
        assert tokens == 225

    def test_extract_thinking_tokens_from_stop_event(self):
        """Test extracting thinking tokens from stop event usage."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "abc"},
            {"type": "message", "content": "Response"},
            {
                "type": "system",
                "subtype": "stop",
                "session_id": "abc",
                "usage": {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 200},
            },
        ]
        tokens = extract_thinking_tokens(events)
        assert tokens == 200

    def test_extract_thinking_tokens_from_result_event(self):
        """Test extracting thinking tokens from result event usage."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "abc"},
            {
                "type": "result",
                "result": "Done",
                "usage": {"input_tokens": 100, "output_tokens": 50, "thinking_tokens": 300},
            },
        ]
        tokens = extract_thinking_tokens(events)
        assert tokens == 300

    def test_extract_thinking_tokens_combined(self):
        """Test extracting thinking tokens from both events and usage."""
        events = [
            {"type": "thinking", "content": "First", "thinking_tokens": 100},
            {"type": "thinking", "content": "Second", "thinking_tokens": 50},
            {
                "type": "system",
                "subtype": "stop",
                "session_id": "abc",
                "usage": {"thinking_tokens": 75},
            },
        ]
        tokens = extract_thinking_tokens(events)
        # 100 + 50 + 75 = 225
        assert tokens == 225

    def test_extract_thinking_tokens_no_thinking(self):
        """Test extracting thinking tokens when none present."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "abc"},
            {"type": "message", "content": "Response"},
            {"type": "system", "subtype": "stop", "session_id": "abc", "usage": {}},
        ]
        tokens = extract_thinking_tokens(events)
        assert tokens == 0

    def test_extract_thinking_tokens_empty_events(self):
        """Test extracting thinking tokens from empty list."""
        tokens = extract_thinking_tokens([])
        assert tokens == 0
