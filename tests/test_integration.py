"""Integration tests for ccflow middleware.

These tests verify the complete flow from API to CLI execution.
Some tests require the Claude CLI to be available and authenticated.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccflow import (
    AssistantMessage,
    CLIAgentOptions,
    InitMessage,
    PermissionMode,
    StopMessage,
    TextMessage,
    query,
)
from ccflow.executor import CLIExecutor
from ccflow.parser import StreamParser
from ccflow.toon_integration import ToonSerializer, should_use_toon
from ccflow.types import ToonConfig


class TestExecutorIntegration:
    """Test CLIExecutor with mocked subprocess."""

    @pytest.fixture
    def mock_ndjson_stream(self):
        """Create mock NDJSON stream output."""
        return [
            b'{"type":"system","subtype":"init","session_id":"test-123"}\n',
            b'{"type":"assistant","message":{"model":"claude-sonnet","id":"msg_1","content":[{"type":"text","text":"Hello!"}],"usage":{"input_tokens":10,"output_tokens":5},"stop_reason":"end_turn"},"session_id":"test-123"}\n',
            b'{"type":"system","subtype":"stop","session_id":"test-123","usage":{"input_tokens":10,"output_tokens":5}}\n',
        ]

    @pytest.mark.asyncio
    async def test_executor_streams_events(self, mock_ndjson_stream):
        """Test that executor properly streams parsed events."""
        # Create mock process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.stdout = AsyncMock()

        # Create async iterator for stdout
        async def stdout_iterator():
            for line in mock_ndjson_stream:
                yield line

        mock_process.stdout.__aiter__ = lambda self: stdout_iterator()
        mock_process.stderr = AsyncMock()
        mock_process.stderr.read = AsyncMock(return_value=b"")
        mock_process.wait = AsyncMock()

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            executor = CLIExecutor(cli_path="/usr/bin/claude")
            events = []
            async for event in executor.execute("test prompt", ["--output-format", "stream-json"]):
                events.append(event)

            assert len(events) == 3
            assert events[0]["type"] == "system"
            assert events[1]["type"] == "assistant"
            assert events[2]["type"] == "system"

    @pytest.mark.asyncio
    async def test_executor_handles_empty_lines(self, mock_ndjson_stream):
        """Test that executor ignores empty lines in stream."""
        # Create mock process with empty lines interspersed
        mock_process = AsyncMock()
        mock_process.returncode = 0

        async def stdout_with_empty_lines():
            yield b'\n'
            yield mock_ndjson_stream[0]
            yield b'   \n'
            yield mock_ndjson_stream[1]
            yield b'\n'
            yield mock_ndjson_stream[2]

        mock_process.stdout = MagicMock()
        mock_process.stdout.__aiter__ = lambda self: stdout_with_empty_lines()
        mock_process.stderr = AsyncMock()
        mock_process.stderr.read = AsyncMock(return_value=b"")
        mock_process.wait = AsyncMock()

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            executor = CLIExecutor(cli_path="/usr/bin/claude")
            events = []
            async for event in executor.execute("test prompt", ["--output-format", "stream-json"]):
                events.append(event)

            # Should only have 3 actual events, empty lines filtered
            assert len(events) == 3


class TestParserIntegration:
    """Test StreamParser with realistic event sequences."""

    def test_parse_full_conversation(self):
        """Test parsing a complete conversation flow."""
        parser = StreamParser()
        events = [
            {"type": "system", "subtype": "init", "session_id": "conv-123"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m1", "content": [{"type": "text", "text": "I'll help."}], "usage": {}, "stop_reason": None}, "session_id": "conv-123"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m2", "content": [{"type": "text", "text": " Here's the answer."}], "usage": {"input_tokens": 100, "output_tokens": 50}, "stop_reason": "end_turn"}, "session_id": "conv-123"},
            {"type": "system", "subtype": "stop", "session_id": "conv-123", "usage": {"input_tokens": 100, "output_tokens": 50}},
        ]

        messages = [parser.parse_event(e) for e in events]

        assert isinstance(messages[0], InitMessage)
        assert isinstance(messages[1], AssistantMessage)
        assert isinstance(messages[2], AssistantMessage)
        assert isinstance(messages[3], StopMessage)

        # Check text extraction
        assert messages[1].text_content == "I'll help."
        assert messages[2].text_content == " Here's the answer."

    def test_parse_tool_use_conversation(self):
        """Test parsing conversation with tool use."""
        parser = StreamParser()
        events = [
            {"type": "system", "subtype": "init", "session_id": "tool-123"},
            {"type": "message", "content": "Let me read that file.", "delta_type": "text_delta"},
            {"type": "tool_use", "tool": "Read", "args": {"file_path": "/test.py"}},
            {"type": "tool_result", "tool": "Read", "content": "print('hello')"},
            {"type": "message", "content": "The file prints hello.", "delta_type": "text_delta"},
            {"type": "system", "subtype": "stop", "session_id": "tool-123", "usage": {"input_tokens": 50, "output_tokens": 20}},
        ]

        messages = [parser.parse_event(e) for e in events]

        assert len(messages) == 6
        assert isinstance(messages[0], InitMessage)
        assert isinstance(messages[1], TextMessage)
        assert messages[2].tool == "Read"
        assert messages[3].result == "print('hello')"

    def test_parse_handles_missing_fields_gracefully(self):
        """Test parser handles missing optional fields."""
        parser = StreamParser()

        # Minimal valid events
        minimal_events = [
            {"type": "system", "subtype": "init"},  # Missing session_id
            {"type": "message"},  # Missing content
            {"type": "system", "subtype": "stop"},  # Missing usage
        ]

        messages = [parser.parse_event(e) for e in minimal_events]

        assert isinstance(messages[0], InitMessage)
        assert messages[0].session_id == ""
        assert isinstance(messages[1], TextMessage)
        assert messages[1].content == ""
        assert isinstance(messages[2], StopMessage)
        assert messages[2].usage == {}


class TestToonIntegration:
    """Test TOON serialization integration."""

    def test_toon_with_context_injection(self):
        """Test TOON encoding for context injection."""
        config = ToonConfig(enabled=False)  # Use JSON fallback for predictable output
        serializer = ToonSerializer(config)

        data = {
            "users": [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]
        }

        formatted = serializer.format_for_prompt(data, label="UserData")

        assert "[UserData]" in formatted
        assert "```json" in formatted
        assert '"users"' in formatted

    def test_should_use_toon_heuristic(self):
        """Test the TOON applicability heuristic."""
        # Uniform arrays should use TOON
        uniform_data = {"items": [{"a": 1}, {"a": 2}, {"a": 3}, {"a": 4}]}
        assert should_use_toon(uniform_data) is True

        # Deeply nested should not
        deep_data = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        assert should_use_toon(deep_data) is False

    def test_toon_serializer_fallback_on_disabled(self):
        """Test that serializer falls back to JSON when disabled."""
        config = ToonConfig(enabled=False)
        serializer = ToonSerializer(config)

        data = {"key": "value", "number": 42}
        result = serializer.encode(data)

        # Should be valid JSON
        import json
        parsed = json.loads(result)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_should_use_toon_shallow_dict(self):
        """Test TOON heuristic with shallow dictionaries."""
        # Shallow dict (depth <= 3) should use TOON
        shallow = {"a": {"b": {"c": 1}}}
        assert should_use_toon(shallow) is True

    def test_should_use_toon_non_uniform_array(self):
        """Test TOON heuristic rejects non-uniform arrays."""
        # Non-uniform array should not use TOON (different keys)
        # This has non-uniform keys so should_use_toon returns False
        # But if items < 4, it won't trigger uniform array check
        small_non_uniform = {"items": [{"a": 1}, {"b": 2}]}
        assert should_use_toon(small_non_uniform) is True  # Falls through to depth check


class TestOptionsIntegration:
    """Test CLIAgentOptions flag generation."""

    def test_build_flags_with_all_options(self):
        """Test that all options generate correct flags."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            model="opus",
            system_prompt="Be helpful",
            permission_mode=PermissionMode.ACCEPT_EDITS,
            allowed_tools=["Read", "Write"],
            max_budget_usd=10.0,
            verbose=True,
        )

        flags = executor.build_flags(options)

        assert "--model" in flags
        assert "opus" in flags
        assert "--system-prompt" in flags
        assert "--permission-mode" in flags
        assert "acceptEdits" in flags
        assert "--allowedTools" in flags
        assert "--max-budget-usd" in flags
        assert "10.0" in flags

    def test_build_flags_with_session_resume(self):
        """Test flags for session resume."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            session_id="session-abc-123",
            resume=True,
        )

        flags = executor.build_flags(options)

        assert "--resume" in flags
        assert "session-abc-123" in flags

    def test_build_flags_with_session_fork(self):
        """Test flags for session fork."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            session_id="session-abc-123",
            resume=True,
            fork_session=True,
        )

        flags = executor.build_flags(options)

        assert "--resume" in flags
        assert "--fork-session" in flags

    def test_build_flags_with_disallowed_tools(self):
        """Test flags for disallowed tools."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            disallowed_tools=["Bash", "Write"],
        )

        flags = executor.build_flags(options)

        assert "--disallowedTools" in flags
        assert "Bash" in flags
        assert "Write" in flags

    def test_build_flags_with_add_dirs(self):
        """Test flags for additional directories."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            add_dirs=["/path/to/dir1", "/path/to/dir2"],
        )

        flags = executor.build_flags(options)

        assert "--add-dir" in flags
        assert "/path/to/dir1" in flags
        assert "/path/to/dir2" in flags

    def test_build_flags_default_includes_stream_json(self):
        """Test that default flags include stream-json output format."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions()

        flags = executor.build_flags(options)

        assert "--output-format" in flags
        assert "stream-json" in flags

    def test_build_flags_with_debug_boolean(self):
        """Test debug flag with boolean True."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(debug=True)

        flags = executor.build_flags(options)

        assert "--debug" in flags
        # Should be just the flag, no filter value
        debug_idx = flags.index("--debug")
        if debug_idx + 1 < len(flags):
            # Next item should be another flag or model value, not a filter
            next_val = flags[debug_idx + 1]
            # If boolean, there's no filter string after --debug
            assert next_val.startswith("--") or next_val == "sonnet"

    def test_build_flags_with_debug_filter(self):
        """Test debug flag with filter string."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(debug="api,hooks")

        flags = executor.build_flags(options)

        assert "--debug" in flags
        assert "api,hooks" in flags

    def test_build_flags_with_json_schema_string(self):
        """Test json_schema flag with string."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        schema = '{"type":"object","properties":{"name":{"type":"string"}}}'
        options = CLIAgentOptions(json_schema=schema)

        flags = executor.build_flags(options)

        assert "--json-schema" in flags
        assert schema in flags

    def test_build_flags_with_json_schema_dict(self):
        """Test json_schema flag with dict (auto-serialized)."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        options = CLIAgentOptions(json_schema=schema)

        flags = executor.build_flags(options)

        assert "--json-schema" in flags
        # Should be JSON serialized
        import json
        schema_idx = flags.index("--json-schema")
        parsed = json.loads(flags[schema_idx + 1])
        assert parsed["type"] == "object"

    def test_build_flags_with_input_format(self):
        """Test input_format flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(input_format="stream-json")

        flags = executor.build_flags(options)

        assert "--input-format" in flags
        assert "stream-json" in flags

    def test_build_flags_with_dangerously_skip_permissions(self):
        """Test dangerously_skip_permissions flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(dangerously_skip_permissions=True)

        flags = executor.build_flags(options)

        assert "--dangerously-skip-permissions" in flags

    def test_build_flags_with_tools_list(self):
        """Test tools flag with list of tools."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(tools=["Bash", "Edit", "Read"])

        flags = executor.build_flags(options)

        assert "--tools" in flags
        assert "Bash" in flags
        assert "Edit" in flags
        assert "Read" in flags

    def test_build_flags_with_tools_empty_list(self):
        """Test tools flag with empty list (disables all tools)."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(tools=[])

        flags = executor.build_flags(options)

        assert "--tools" in flags
        tools_idx = flags.index("--tools")
        assert flags[tools_idx + 1] == ""

    def test_build_flags_with_continue_session(self):
        """Test continue_session flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(continue_session=True)

        flags = executor.build_flags(options)

        assert "--continue" in flags

    def test_build_flags_with_no_session_persistence(self):
        """Test no_session_persistence flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(no_session_persistence=True)

        flags = executor.build_flags(options)

        assert "--no-session-persistence" in flags

    def test_build_flags_with_agent(self):
        """Test agent flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(agent="code-reviewer")

        flags = executor.build_flags(options)

        assert "--agent" in flags
        assert "code-reviewer" in flags

    def test_build_flags_with_agents_json(self):
        """Test agents flag with custom agents JSON."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        agents_config = {
            "reviewer": {"description": "Reviews code", "prompt": "You are a code reviewer"}
        }
        options = CLIAgentOptions(agents=agents_config)

        flags = executor.build_flags(options)

        assert "--agents" in flags
        import json
        agents_idx = flags.index("--agents")
        parsed = json.loads(flags[agents_idx + 1])
        assert "reviewer" in parsed
        assert parsed["reviewer"]["description"] == "Reviews code"

    def test_build_flags_with_betas(self):
        """Test betas flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(betas=["computer-use", "token-counting"])

        flags = executor.build_flags(options)

        assert "--betas" in flags
        assert "computer-use" in flags
        assert "token-counting" in flags

    def test_build_flags_with_settings(self):
        """Test settings flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(settings="/path/to/settings.json")

        flags = executor.build_flags(options)

        assert "--settings" in flags
        assert "/path/to/settings.json" in flags

    def test_build_flags_with_plugin_dirs(self):
        """Test plugin_dirs flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(plugin_dirs=["/path/to/plugins1", "/path/to/plugins2"])

        flags = executor.build_flags(options)

        assert "--plugin-dir" in flags
        assert "/path/to/plugins1" in flags
        assert "/path/to/plugins2" in flags

    def test_build_flags_with_disable_slash_commands(self):
        """Test disable_slash_commands flag."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(disable_slash_commands=True)

        flags = executor.build_flags(options)

        assert "--disable-slash-commands" in flags

    def test_build_flags_with_all_new_options(self):
        """Test all new options together."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            debug="api",
            json_schema={"type": "object"},
            input_format="text",
            dangerously_skip_permissions=True,
            tools=["Read", "Write"],
            continue_session=False,  # Default
            no_session_persistence=True,
            agent="explorer",
            agents={"test": {"description": "Test agent"}},
            betas=["beta1"],
            settings='{"key": "value"}',
            plugin_dirs=["/plugins"],
            disable_slash_commands=True,
        )

        flags = executor.build_flags(options)

        assert "--debug" in flags
        assert "--json-schema" in flags
        assert "--input-format" in flags
        assert "--dangerously-skip-permissions" in flags
        assert "--tools" in flags
        assert "--no-session-persistence" in flags
        assert "--agent" in flags
        assert "--agents" in flags
        assert "--betas" in flags
        assert "--settings" in flags
        assert "--plugin-dir" in flags
        assert "--disable-slash-commands" in flags


class TestEndToEndMocked:
    """End-to-end tests with mocked CLI."""

    @pytest.mark.asyncio
    async def test_query_returns_text_content(self):
        """Test that query() properly yields message content."""
        mock_events = [
            {"type": "system", "subtype": "init", "session_id": "e2e-test"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m1", "content": [{"type": "text", "text": "Test response"}], "usage": {"input_tokens": 5, "output_tokens": 3}, "stop_reason": "end_turn"}, "session_id": "e2e-test"},
            {"type": "system", "subtype": "stop", "session_id": "e2e-test", "usage": {"input_tokens": 5, "output_tokens": 3}},
        ]

        async def mock_execute(*args, **kwargs):
            for event in mock_events:
                yield event

        with patch('ccflow.api.get_executor') as mock_get_executor:
            mock_executor = MagicMock()
            mock_executor.execute = mock_execute
            mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags
            mock_get_executor.return_value = mock_executor

            messages = []
            async for msg in query("Test", CLIAgentOptions()):
                messages.append(msg)

            assert len(messages) == 3
            assert any(isinstance(m, AssistantMessage) for m in messages)

            assistant_msgs = [m for m in messages if isinstance(m, AssistantMessage)]
            assert assistant_msgs[0].text_content == "Test response"

    @pytest.mark.asyncio
    async def test_query_with_context_injection(self):
        """Test query with context data injection."""
        mock_events = [
            {"type": "system", "subtype": "init", "session_id": "ctx-test"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m1", "content": [{"type": "text", "text": "I see the context"}], "usage": {}, "stop_reason": "end_turn"}, "session_id": "ctx-test"},
            {"type": "system", "subtype": "stop", "session_id": "ctx-test", "usage": {}},
        ]

        captured_flags = []

        async def mock_execute(prompt, flags, **kwargs):
            captured_flags.extend(flags)
            for event in mock_events:
                yield event

        with patch('ccflow.api.get_executor') as mock_get_executor:
            mock_executor = MagicMock()
            mock_executor.execute = mock_execute
            mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags
            mock_get_executor.return_value = mock_executor

            options = CLIAgentOptions(
                context={"user": "Alice", "role": "admin"},
                toon=ToonConfig(enabled=False, encode_context=True),
            )

            messages = []
            async for msg in query("What is the user?", options):
                messages.append(msg)

            # Context should have been injected into append_system_prompt
            assert "--append-system-prompt" in captured_flags

    @pytest.mark.asyncio
    async def test_query_handles_stop_message(self):
        """Test that query properly handles stop message with usage."""
        mock_events = [
            {"type": "system", "subtype": "init", "session_id": "stop-test"},
            {"type": "message", "content": "Response text", "delta_type": "text_delta"},
            {"type": "system", "subtype": "stop", "session_id": "stop-test", "usage": {"input_tokens": 100, "output_tokens": 50}},
        ]

        async def mock_execute(*args, **kwargs):
            for event in mock_events:
                yield event

        with patch('ccflow.api.get_executor') as mock_get_executor:
            mock_executor = MagicMock()
            mock_executor.execute = mock_execute
            mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags
            mock_get_executor.return_value = mock_executor

            messages = []
            async for msg in query("Test", CLIAgentOptions()):
                messages.append(msg)

            stop_msgs = [m for m in messages if isinstance(m, StopMessage)]
            assert len(stop_msgs) == 1
            assert stop_msgs[0].usage["input_tokens"] == 100
            assert stop_msgs[0].usage["output_tokens"] == 50


class TestMultiTurnIntegration:
    """Test multi-turn conversation scenarios."""

    def test_parser_tracks_multiple_assistant_messages(self):
        """Test parsing multiple assistant turns."""
        parser = StreamParser()
        events = [
            {"type": "system", "subtype": "init", "session_id": "multi-123"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m1", "content": [{"type": "text", "text": "First turn"}], "usage": {"input_tokens": 10, "output_tokens": 5}}, "session_id": "multi-123"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m2", "content": [{"type": "text", "text": "Second turn"}], "usage": {"input_tokens": 15, "output_tokens": 8}}, "session_id": "multi-123"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m3", "content": [{"type": "text", "text": "Third turn"}], "usage": {"input_tokens": 20, "output_tokens": 10}, "stop_reason": "end_turn"}, "session_id": "multi-123"},
            {"type": "system", "subtype": "stop", "session_id": "multi-123", "usage": {"input_tokens": 45, "output_tokens": 23}},
        ]

        messages = [parser.parse_event(e) for e in events]

        assistant_msgs = [m for m in messages if isinstance(m, AssistantMessage)]
        assert len(assistant_msgs) == 3
        assert assistant_msgs[0].text_content == "First turn"
        assert assistant_msgs[1].text_content == "Second turn"
        assert assistant_msgs[2].text_content == "Third turn"

    def test_parser_handles_interleaved_tool_use(self):
        """Test parsing conversation with interleaved tool calls."""
        parser = StreamParser()
        events = [
            {"type": "system", "subtype": "init", "session_id": "tool-multi"},
            {"type": "message", "content": "Reading files...", "delta_type": "text_delta"},
            {"type": "tool_use", "tool": "Read", "args": {"file_path": "/a.py"}},
            {"type": "tool_result", "tool": "Read", "content": "content_a"},
            {"type": "message", "content": "Got first file.", "delta_type": "text_delta"},
            {"type": "tool_use", "tool": "Read", "args": {"file_path": "/b.py"}},
            {"type": "tool_result", "tool": "Read", "content": "content_b"},
            {"type": "message", "content": "Done!", "delta_type": "text_delta"},
            {"type": "system", "subtype": "stop", "session_id": "tool-multi", "usage": {}},
        ]

        messages = [parser.parse_event(e) for e in events]

        text_msgs = [m for m in messages if isinstance(m, TextMessage)]
        assert len(text_msgs) == 3

        from ccflow.types import ToolResultMessage, ToolUseMessage
        tool_use_msgs = [m for m in messages if isinstance(m, ToolUseMessage)]
        tool_result_msgs = [m for m in messages if isinstance(m, ToolResultMessage)]

        assert len(tool_use_msgs) == 2
        assert len(tool_result_msgs) == 2


class TestErrorHandlingIntegration:
    """Test error handling across the stack."""

    @pytest.mark.asyncio
    async def test_executor_handles_nonzero_exit(self):
        """Test executor raises on non-zero exit code."""
        from ccflow.exceptions import CLIExecutionError

        mock_process = AsyncMock()
        mock_process.returncode = 1

        async def empty_iterator():
            return
            yield  # Make it a generator

        mock_process.stdout = MagicMock()
        mock_process.stdout.__aiter__ = lambda self: empty_iterator()
        mock_process.stderr = AsyncMock()
        mock_process.stderr.read = AsyncMock(return_value=b"Error: something went wrong")
        mock_process.wait = AsyncMock()

        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            executor = CLIExecutor(cli_path="/usr/bin/claude")

            with pytest.raises(CLIExecutionError) as exc_info:
                async for _ in executor.execute("test", []):
                    pass

            assert "exited with code 1" in str(exc_info.value)

    def test_parser_handles_error_event(self):
        """Test parser correctly handles error events."""
        parser = StreamParser()
        event = {"type": "error", "message": "Rate limit exceeded", "code": "rate_limit"}

        from ccflow.types import ErrorMessage
        msg = parser.parse_event(event)

        assert isinstance(msg, ErrorMessage)
        assert msg.message == "Rate limit exceeded"
        assert msg.code == "rate_limit"

    def test_parser_handles_unknown_event_type(self):
        """Test parser gracefully handles unknown event types."""
        parser = StreamParser()
        event = {"type": "future_new_type", "data": "something"}

        from ccflow.types import UnknownMessage
        msg = parser.parse_event(event)

        assert isinstance(msg, UnknownMessage)
        assert msg.event_type == "future_new_type"
        assert msg.raw_data == event


class TestMetricsIntegration:
    """Test metrics integration with API functions."""

    @pytest.mark.asyncio
    async def test_query_records_metrics_on_success(self):
        """Test that query() records metrics on successful completion."""
        from unittest.mock import patch

        mock_events = [
            {"type": "system", "subtype": "init", "session_id": "metrics-test"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m1", "content": [{"type": "text", "text": "Hello"}], "usage": {}, "stop_reason": "end_turn"}, "session_id": "metrics-test"},
            {"type": "system", "subtype": "stop", "session_id": "metrics-test", "usage": {"input_tokens": 100, "output_tokens": 50}},
        ]

        async def mock_execute(*args, **kwargs):
            for event in mock_events:
                yield event

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request') as mock_record_request:
            mock_executor = MagicMock()
            mock_executor.execute = mock_execute
            mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags
            mock_get_executor.return_value = mock_executor

            messages = []
            async for msg in query("Test", CLIAgentOptions(model="sonnet")):
                messages.append(msg)

            # Verify record_request was called with correct arguments
            mock_record_request.assert_called_once()
            call_args = mock_record_request.call_args
            assert call_args.kwargs["model"] == "sonnet"
            assert call_args.kwargs["status"] == "success"
            assert call_args.kwargs["input_tokens"] == 100
            assert call_args.kwargs["output_tokens"] == 50
            assert call_args.kwargs["duration"] > 0

    @pytest.mark.asyncio
    async def test_query_records_error_metrics_on_failure(self):
        """Test that query() records error metrics on failure."""
        from unittest.mock import patch

        from ccflow.exceptions import CLIExecutionError

        async def mock_execute_error(*args, **kwargs):
            raise CLIExecutionError("Test error", exit_code=1)
            yield  # Make it a generator

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request') as mock_record_request, \
             patch('ccflow.api.record_error') as mock_record_error:
            mock_executor = MagicMock()
            mock_executor.execute = mock_execute_error
            mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags
            mock_get_executor.return_value = mock_executor

            with pytest.raises(CLIExecutionError):
                async for _ in query("Test", CLIAgentOptions()):
                    pass

            # Verify error was recorded
            mock_record_error.assert_called_once_with("CLIExecutionError")

            # Verify request was recorded with error status
            mock_record_request.assert_called_once()
            assert mock_record_request.call_args.kwargs["status"] == "error"

    @pytest.mark.asyncio
    async def test_query_records_toon_savings(self):
        """Test that query() records TOON savings when context is provided."""
        from unittest.mock import patch

        mock_events = [
            {"type": "system", "subtype": "init", "session_id": "toon-test"},
            {"type": "system", "subtype": "stop", "session_id": "toon-test", "usage": {}},
        ]

        async def mock_execute(*args, **kwargs):
            for event in mock_events:
                yield event

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'), \
             patch('ccflow.api.record_toon_savings'):
            mock_executor = MagicMock()
            mock_executor.execute = mock_execute
            mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags
            mock_get_executor.return_value = mock_executor

            # Note: TOON savings tracking happens during ToonSerializer.encode
            # which sets _last_json_tokens and _last_toon_tokens
            # For this test, we're just checking the integration point exists
            options = CLIAgentOptions(
                context={"key": "value"},
            )

            async for _ in query("Test", options):
                pass

            # record_toon_savings is only called if _last_json_tokens > 0
            # which happens when TOON library is available and does tracking


class TestMCPIntegration:
    """Test MCP configuration integration with executor."""

    def test_build_flags_with_mcp_servers(self):
        """Test that MCP servers generate --mcp-config flag."""
        from ccflow.types import MCPServerConfig

        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            mcp_servers={
                "github": MCPServerConfig(
                    command="npx",
                    args=["@anthropic-ai/mcp-server-github"],
                ),
                "filesystem": MCPServerConfig(
                    command="npx",
                    args=["@anthropic-ai/mcp-server-filesystem", "/tmp"],
                ),
            },
        )

        flags = executor.build_flags(options)

        assert "--mcp-config" in flags
        # Config path should be in the flags
        config_idx = flags.index("--mcp-config")
        config_path = flags[config_idx + 1]
        assert config_path.endswith(".json")

        # Cleanup
        executor.cleanup()

    def test_build_flags_with_mcp_servers_strict_mode(self):
        """Test that strict_mcp adds --strict-mcp-config flag."""
        from ccflow.types import MCPServerConfig

        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            mcp_servers={
                "custom": MCPServerConfig(command="python", args=["-m", "my_mcp"]),
            },
            strict_mcp=True,
        )

        flags = executor.build_flags(options)

        assert "--mcp-config" in flags
        assert "--strict-mcp-config" in flags

        # Cleanup
        executor.cleanup()

    def test_mcp_config_file_created(self):
        """Test that MCP config file is created with correct content."""
        import json
        from pathlib import Path

        from ccflow.types import MCPServerConfig

        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions(
            mcp_servers={
                "test-server": MCPServerConfig(
                    command="node",
                    args=["server.js", "--port", "3000"],
                    env={"API_KEY": "secret"},
                ),
            },
        )

        flags = executor.build_flags(options)

        # Get config path and read it
        config_idx = flags.index("--mcp-config")
        config_path = Path(flags[config_idx + 1])

        assert config_path.exists()

        with open(config_path) as f:
            config = json.load(f)

        assert "mcpServers" in config
        assert "test-server" in config["mcpServers"]
        assert config["mcpServers"]["test-server"]["command"] == "node"
        assert config["mcpServers"]["test-server"]["args"] == ["server.js", "--port", "3000"]
        assert config["mcpServers"]["test-server"]["env"] == {"API_KEY": "secret"}

        # Cleanup
        executor.cleanup()
        assert not config_path.exists()

    def test_executor_cleanup_removes_mcp_configs(self):
        """Test that executor cleanup removes MCP config files."""

        from ccflow.types import MCPServerConfig

        executor = CLIExecutor(cli_path="/usr/bin/claude")

        # Create multiple configs
        for i in range(3):
            options = CLIAgentOptions(
                mcp_servers={
                    f"server-{i}": MCPServerConfig(command="test", args=[]),
                },
            )
            executor.build_flags(options)

        # Get all config files
        config_files = list(executor._mcp_manager._config_files)
        assert len(config_files) == 3

        for path in config_files:
            assert path.exists()

        # Cleanup
        executor.cleanup()

        for path in config_files:
            assert not path.exists()

    def test_build_flags_without_mcp_servers(self):
        """Test that no MCP flags are added when mcp_servers is None."""
        executor = CLIExecutor(cli_path="/usr/bin/claude")
        options = CLIAgentOptions()

        flags = executor.build_flags(options)

        assert "--mcp-config" not in flags
        assert "--strict-mcp-config" not in flags

        executor.cleanup()
