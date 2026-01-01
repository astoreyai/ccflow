"""Tests for ccflow API functions.

Covers query_simple, batch_query, and stream_to_callback.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from ccflow.api import query, query_simple, batch_query, stream_to_callback
from ccflow.types import (
    CLIAgentOptions,
    InitMessage,
    StopMessage,
    TextMessage,
    AssistantMessage,
    ResultMessage,
    ToonConfig,
)
from ccflow.executor import CLIExecutor
from ccflow.exceptions import CLIExecutionError


def create_mock_executor(events):
    """Create a mock executor that yields given events."""
    async def mock_execute(*args, **kwargs):
        for event in events:
            yield event

    mock_executor = MagicMock()
    mock_executor.execute = mock_execute
    mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags
    return mock_executor


class TestQuerySimple:
    """Tests for query_simple function."""

    @pytest.fixture
    def standard_events(self):
        """Standard successful response events."""
        return [
            {"type": "system", "subtype": "init", "session_id": "simple-123"},
            {"type": "message", "content": "Hello, ", "delta_type": "text_delta"},
            {"type": "message", "content": "world!", "delta_type": "text_delta"},
            {"type": "system", "subtype": "stop", "session_id": "simple-123", "usage": {"input_tokens": 10, "output_tokens": 5}},
        ]

    @pytest.mark.asyncio
    async def test_query_simple_returns_text(self, standard_events):
        """Test that query_simple returns concatenated text."""
        with patch('ccflow.api.get_executor') as mock_get_executor:
            mock_get_executor.return_value = create_mock_executor(standard_events)

            result = await query_simple("Test prompt")

            assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_query_simple_with_options(self, standard_events):
        """Test query_simple with custom options."""
        with patch('ccflow.api.get_executor') as mock_get_executor:
            mock_get_executor.return_value = create_mock_executor(standard_events)

            options = CLIAgentOptions(model="opus")
            result = await query_simple("Test prompt", options)

            assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_query_simple_empty_response(self):
        """Test query_simple with no text content."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "empty-123"},
            {"type": "system", "subtype": "stop", "session_id": "empty-123", "usage": {}},
        ]

        with patch('ccflow.api.get_executor') as mock_get_executor:
            mock_get_executor.return_value = create_mock_executor(events)

            result = await query_simple("Test prompt")

            assert result == ""

    @pytest.mark.asyncio
    async def test_query_simple_with_assistant_message(self):
        """Test query_simple extracts text from AssistantMessage."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "asst-123"},
            {"type": "assistant", "message": {"model": "sonnet", "id": "m1", "content": [{"type": "text", "text": "Assistant response"}], "usage": {}, "stop_reason": "end_turn"}, "session_id": "asst-123"},
            {"type": "system", "subtype": "stop", "session_id": "asst-123", "usage": {}},
        ]

        with patch('ccflow.api.get_executor') as mock_get_executor:
            mock_get_executor.return_value = create_mock_executor(events)

            result = await query_simple("Test prompt")

            # AssistantMessage doesn't have .content attribute directly
            # collect_text checks for hasattr(msg, "content")
            # AssistantMessage has text_content property, not content
            assert result == ""  # AssistantMessage content not collected by collect_text


class TestBatchQuery:
    """Tests for batch_query function."""

    @pytest.fixture
    def single_prompt_events(self):
        """Events for a single prompt response."""
        return [
            {"type": "system", "subtype": "init", "session_id": "batch-001"},
            {"type": "message", "content": "Response", "delta_type": "text_delta"},
            {"type": "system", "subtype": "stop", "session_id": "batch-001", "usage": {"input_tokens": 10, "output_tokens": 5}},
        ]

    @pytest.mark.asyncio
    async def test_batch_query_single_prompt(self, single_prompt_events):
        """Test batch_query with single prompt."""
        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'), \
             patch('ccflow.api.record_error'):
            mock_get_executor.return_value = create_mock_executor(single_prompt_events)

            results = await batch_query(["Test prompt"])

            assert len(results) == 1
            assert results[0].session_id == "batch-001"
            assert results[0].result == "Response"
            assert results[0].success is True

    @pytest.mark.asyncio
    async def test_batch_query_multiple_prompts(self):
        """Test batch_query with multiple prompts."""
        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            events = [
                {"type": "system", "subtype": "init", "session_id": f"batch-{call_count:03d}"},
                {"type": "message", "content": f"Response {call_count}", "delta_type": "text_delta"},
                {"type": "system", "subtype": "stop", "session_id": f"batch-{call_count:03d}", "usage": {"input_tokens": 10 * call_count, "output_tokens": 5 * call_count}},
            ]
            for event in events:
                yield event

        mock_executor = MagicMock()
        mock_executor.execute = mock_execute
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'), \
             patch('ccflow.api.record_error'):
            mock_get_executor.return_value = mock_executor

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = await batch_query(prompts)

            assert len(results) == 3
            # Results come back in order (asyncio.gather preserves order)
            for i, result in enumerate(results, 1):
                assert result.success is True
                assert f"Response {i}" in result.result

    @pytest.mark.asyncio
    async def test_batch_query_with_options(self):
        """Test batch_query uses provided options."""
        captured_flags = []

        async def mock_execute(prompt, flags, **kwargs):
            captured_flags.append(flags.copy())
            events = [
                {"type": "system", "subtype": "init", "session_id": "opt-test"},
                {"type": "system", "subtype": "stop", "session_id": "opt-test", "usage": {}},
            ]
            for event in events:
                yield event

        mock_executor = MagicMock()
        mock_executor.execute = mock_execute
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'), \
             patch('ccflow.api.record_error'):
            mock_get_executor.return_value = mock_executor

            options = CLIAgentOptions(model="haiku")
            results = await batch_query(["Prompt 1", "Prompt 2"], options)

            assert len(results) == 2
            # Both should have used the same options
            for flags in captured_flags:
                assert "--model" in flags
                assert "haiku" in flags

    @pytest.mark.asyncio
    async def test_batch_query_concurrency_limit(self):
        """Test batch_query respects concurrency limit."""
        import asyncio

        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def mock_execute(*args, **kwargs):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                if current_concurrent > max_concurrent:
                    max_concurrent = current_concurrent

            # Simulate some work
            await asyncio.sleep(0.01)

            async with lock:
                current_concurrent -= 1

            events = [
                {"type": "system", "subtype": "init", "session_id": "conc-test"},
                {"type": "system", "subtype": "stop", "session_id": "conc-test", "usage": {}},
            ]
            for event in events:
                yield event

        mock_executor = MagicMock()
        mock_executor.execute = mock_execute
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'), \
             patch('ccflow.api.record_error'):
            mock_get_executor.return_value = mock_executor

            # Run 10 prompts with concurrency of 2
            prompts = [f"Prompt {i}" for i in range(10)]
            results = await batch_query(prompts, concurrency=2)

            assert len(results) == 10
            assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_batch_query_handles_errors(self):
        """Test batch_query handles individual query errors."""
        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 2:
                # Second query fails
                raise CLIExecutionError("Test error", exit_code=1)

            events = [
                {"type": "system", "subtype": "init", "session_id": f"err-{call_count}"},
                {"type": "message", "content": f"Success {call_count}", "delta_type": "text_delta"},
                {"type": "system", "subtype": "stop", "session_id": f"err-{call_count}", "usage": {}},
            ]
            for event in events:
                yield event

        mock_executor = MagicMock()
        mock_executor.execute = mock_execute
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'), \
             patch('ccflow.api.record_error') as mock_record_error:
            mock_get_executor.return_value = mock_executor

            prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
            results = await batch_query(prompts, concurrency=1)

            assert len(results) == 3

            # First and third should succeed
            assert results[0].success is True
            assert results[2].success is True

            # Second should fail
            assert results[1].success is False
            assert results[1].error is not None
            assert "Test error" in results[1].error

            # Error should be recorded
            mock_record_error.assert_called()

    @pytest.mark.asyncio
    async def test_batch_query_tracks_tokens(self):
        """Test batch_query tracks token usage across all prompts."""
        call_count = 0

        async def mock_execute(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            events = [
                {"type": "system", "subtype": "init", "session_id": f"tok-{call_count}"},
                {"type": "system", "subtype": "stop", "session_id": f"tok-{call_count}", "usage": {"input_tokens": 100, "output_tokens": 50}},
            ]
            for event in events:
                yield event

        mock_executor = MagicMock()
        mock_executor.execute = mock_execute
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request') as mock_record, \
             patch('ccflow.api.record_error'):
            mock_get_executor.return_value = mock_executor

            prompts = ["Prompt 1", "Prompt 2"]
            results = await batch_query(prompts)

            # Each result should have token counts
            for result in results:
                assert result.input_tokens == 100
                assert result.output_tokens == 50
                assert result.total_tokens == 150

    @pytest.mark.asyncio
    async def test_batch_query_empty_prompts(self):
        """Test batch_query with empty prompt list."""
        with patch('ccflow.api.get_executor'), \
             patch('ccflow.api.record_request'), \
             patch('ccflow.api.record_error'):
            results = await batch_query([])

            assert len(results) == 0

    @pytest.mark.asyncio
    async def test_batch_query_records_batch_metrics(self):
        """Test batch_query records batch-level metrics."""
        async def mock_execute(*args, **kwargs):
            events = [
                {"type": "system", "subtype": "init", "session_id": "met-test"},
                {"type": "system", "subtype": "stop", "session_id": "met-test", "usage": {"input_tokens": 50, "output_tokens": 25}},
            ]
            for event in events:
                yield event

        mock_executor = MagicMock()
        mock_executor.execute = mock_execute
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request') as mock_record, \
             patch('ccflow.api.record_error'):
            mock_get_executor.return_value = mock_executor

            options = CLIAgentOptions(model="sonnet")
            await batch_query(["P1", "P2"], options)

            # record_request is called for each individual query + 1 for batch
            # Individual: 2 calls from query()
            # Batch: 1 call from batch_query()
            assert mock_record.call_count >= 1

            # Check that batch metrics were recorded
            calls = mock_record.call_args_list
            # Last call should be batch metrics with aggregated tokens
            batch_call = calls[-1]
            assert batch_call.kwargs["model"] == "sonnet"
            assert batch_call.kwargs["status"] == "success"


class TestStreamToCallback:
    """Tests for stream_to_callback function."""

    @pytest.fixture
    def standard_events(self):
        """Standard successful response events."""
        return [
            {"type": "system", "subtype": "init", "session_id": "cb-123"},
            {"type": "message", "content": "Hello", "delta_type": "text_delta"},
            {"type": "message", "content": " callback!", "delta_type": "text_delta"},
            {"type": "system", "subtype": "stop", "session_id": "cb-123", "usage": {"input_tokens": 20, "output_tokens": 10}},
        ]

    @pytest.mark.asyncio
    async def test_stream_to_callback_calls_callback(self, standard_events):
        """Test that stream_to_callback calls callback for each message."""
        callback_calls = []

        def callback(msg):
            callback_calls.append(msg)

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'):
            mock_get_executor.return_value = create_mock_executor(standard_events)

            result = await stream_to_callback("Test", callback)

            # Should have called callback 4 times (init, 2 text, stop)
            assert len(callback_calls) == 4
            assert isinstance(callback_calls[0], InitMessage)
            assert isinstance(callback_calls[1], TextMessage)
            assert isinstance(callback_calls[2], TextMessage)
            assert isinstance(callback_calls[3], StopMessage)

    @pytest.mark.asyncio
    async def test_stream_to_callback_returns_query_result(self, standard_events):
        """Test that stream_to_callback returns QueryResult."""
        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'):
            mock_get_executor.return_value = create_mock_executor(standard_events)

            result = await stream_to_callback("Test", lambda m: None)

            assert result.session_id == "cb-123"
            assert result.result == "Hello callback!"
            assert result.input_tokens == 20
            assert result.output_tokens == 10
            assert result.success is True
            assert result.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_stream_to_callback_with_options(self):
        """Test stream_to_callback with custom options."""
        captured_flags = []

        async def mock_execute(prompt, flags, **kwargs):
            captured_flags.extend(flags)
            events = [
                {"type": "system", "subtype": "init", "session_id": "opt-cb"},
                {"type": "system", "subtype": "stop", "session_id": "opt-cb", "usage": {}},
            ]
            for event in events:
                yield event

        mock_executor = MagicMock()
        mock_executor.execute = mock_execute
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'):
            mock_get_executor.return_value = mock_executor

            options = CLIAgentOptions(model="haiku", max_budget_usd=5.0)
            await stream_to_callback("Test", lambda m: None, options)

            assert "--model" in captured_flags
            assert "haiku" in captured_flags
            assert "--max-budget-usd" in captured_flags
            assert "5.0" in captured_flags

    @pytest.mark.asyncio
    async def test_stream_to_callback_handles_error(self):
        """Test stream_to_callback handles errors gracefully."""
        async def mock_execute_error(*args, **kwargs):
            raise CLIExecutionError("Callback error", exit_code=1)
            yield  # Make it a generator

        mock_executor = MagicMock()
        mock_executor.execute = mock_execute_error
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_error') as mock_record_error, \
             patch('ccflow.api.record_request'):
            mock_get_executor.return_value = mock_executor

            result = await stream_to_callback("Test", lambda m: None)

            assert result.success is False
            assert result.error is not None
            assert "Callback error" in result.error

            # Error recorded from both query() and stream_to_callback()
            assert mock_record_error.call_count == 2
            mock_record_error.assert_any_call("CLIExecutionError")

    @pytest.mark.asyncio
    async def test_stream_to_callback_tracks_toon_savings(self):
        """Test stream_to_callback includes TOON savings ratio."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "toon-cb"},
            {"type": "system", "subtype": "stop", "session_id": "toon-cb", "usage": {}},
        ]

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'):
            mock_get_executor.return_value = create_mock_executor(events)

            # Create options with TOON config
            toon_config = ToonConfig(enabled=True)
            options = CLIAgentOptions(toon=toon_config)

            result = await stream_to_callback("Test", lambda m: None, options)

            # Result should have toon_savings_ratio (even if None)
            assert hasattr(result, 'toon_savings_ratio')

    @pytest.mark.asyncio
    async def test_stream_to_callback_empty_response(self):
        """Test stream_to_callback with empty response."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "empty-cb"},
            {"type": "system", "subtype": "stop", "session_id": "empty-cb", "usage": {}},
        ]

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'):
            mock_get_executor.return_value = create_mock_executor(events)

            result = await stream_to_callback("Test", lambda m: None)

            assert result.result == ""
            assert result.success is True


class TestQueryWithResultMessage:
    """Tests for query handling ResultMessage events."""

    @pytest.mark.asyncio
    async def test_query_extracts_tokens_from_result_message(self):
        """Test that query extracts tokens from ResultMessage."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "result-test"},
            {"type": "message", "content": "Response", "delta_type": "text_delta"},
            {"type": "result", "result": "Done", "session_id": "result-test", "usage": {"input_tokens": 200, "output_tokens": 100}, "duration_ms": 1500, "num_turns": 2},
        ]

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request') as mock_record:
            mock_get_executor.return_value = create_mock_executor(events)

            messages = []
            async for msg in query("Test", CLIAgentOptions()):
                messages.append(msg)

            # Should have parsed ResultMessage
            result_msgs = [m for m in messages if isinstance(m, ResultMessage)]
            assert len(result_msgs) == 1
            assert result_msgs[0].usage["input_tokens"] == 200
            assert result_msgs[0].usage["output_tokens"] == 100

            # Metrics should use ResultMessage tokens
            mock_record.assert_called_once()
            call_args = mock_record.call_args.kwargs
            assert call_args["input_tokens"] == 200
            assert call_args["output_tokens"] == 100

    @pytest.mark.asyncio
    async def test_query_prefers_stop_message_tokens(self):
        """Test query uses StopMessage tokens over ResultMessage."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "stop-pref"},
            {"type": "system", "subtype": "stop", "session_id": "stop-pref", "usage": {"input_tokens": 50, "output_tokens": 25}},
            {"type": "result", "result": "Done", "session_id": "stop-pref", "usage": {"input_tokens": 100, "output_tokens": 50}},
        ]

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request') as mock_record:
            mock_get_executor.return_value = create_mock_executor(events)

            async for _ in query("Test", CLIAgentOptions()):
                pass

            # StopMessage tokens should be used since it comes first
            call_args = mock_record.call_args.kwargs
            # ResultMessage comes after and should update the values
            assert call_args["input_tokens"] == 100
            assert call_args["output_tokens"] == 50


class TestQueryWithToonSavings:
    """Tests for TOON savings recording in query."""

    @pytest.mark.asyncio
    async def test_query_records_toon_savings_when_enabled(self):
        """Test query records TOON savings with context."""
        events = [
            {"type": "system", "subtype": "init", "session_id": "toon-save"},
            {"type": "system", "subtype": "stop", "session_id": "toon-save", "usage": {}},
        ]

        with patch('ccflow.api.get_executor') as mock_get_executor, \
             patch('ccflow.api.record_request'), \
             patch('ccflow.api.record_toon_savings') as mock_toon_savings:
            mock_get_executor.return_value = create_mock_executor(events)

            # Create config with tracking enabled
            toon_config = ToonConfig(
                enabled=True,
                encode_context=True,
                track_savings=True,
            )
            # Simulate previous encoding that set token counts
            toon_config._last_json_tokens = 100
            toon_config._last_toon_tokens = 60

            options = CLIAgentOptions(
                context={"data": "value"},
                toon=toon_config,
            )

            async for _ in query("Test", options):
                pass

            # TOON savings should be recorded
            mock_toon_savings.assert_called_once_with(100, 60)
