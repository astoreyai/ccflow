"""Tests for session management."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

from ccflow.session import Session, resume_session
from ccflow.types import (
    CLIAgentOptions,
    InitMessage,
    StopMessage,
    TextMessage,
    SessionStats,
    PermissionMode,
)
from ccflow.executor import CLIExecutor


class TestSessionInit:
    """Tests for Session initialization."""

    def test_session_creates_with_defaults(self):
        """Test session creates with default values."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session()

        assert session.session_id is not None
        assert len(session.session_id) == 36  # UUID format
        assert session.turn_count == 0
        assert session.is_closed is False

    def test_session_uses_provided_session_id(self):
        """Test session uses provided session ID."""
        custom_id = "custom-session-123"
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session(session_id=custom_id)

        assert session.session_id == custom_id

    def test_session_uses_provided_options(self):
        """Test session uses provided options."""
        options = CLIAgentOptions(
            model="opus",
            permission_mode=PermissionMode.ACCEPT_EDITS,
        )
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session(options=options)

        assert session._options.model == "opus"
        assert session._options.permission_mode == PermissionMode.ACCEPT_EDITS

    def test_session_uses_provided_executor(self):
        """Test session uses provided executor."""
        mock_executor = MagicMock(spec=CLIExecutor)
        session = Session(executor=mock_executor)

        assert session._executor is mock_executor


class TestSessionProperties:
    """Tests for Session properties."""

    def test_session_id_property(self):
        """Test session_id property returns correct value."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session(session_id="test-123")

        assert session.session_id == "test-123"

    def test_turn_count_starts_at_zero(self):
        """Test turn_count starts at zero."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session()

        assert session.turn_count == 0

    def test_is_closed_starts_false(self):
        """Test is_closed starts as False."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session()

        assert session.is_closed is False


class TestSessionSendMessage:
    """Tests for Session.send_message()."""

    @pytest.fixture
    def mock_executor(self):
        """Create a mock executor."""
        executor = MagicMock(spec=CLIExecutor)
        executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags
        return executor

    @pytest.fixture
    def mock_events(self):
        """Create mock NDJSON events."""
        return [
            {"type": "system", "subtype": "init", "session_id": "cli-session-abc"},
            {"type": "message", "content": "Hello!", "delta_type": "text_delta"},
            {"type": "system", "subtype": "stop", "session_id": "cli-session-abc",
             "usage": {"input_tokens": 100, "output_tokens": 50}},
        ]

    @pytest.mark.asyncio
    async def test_send_message_streams_response(self, mock_executor, mock_events):
        """Test send_message streams response messages."""
        async def mock_execute(*args, **kwargs):
            for event in mock_events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(executor=mock_executor)

        messages = []
        async for msg in session.send_message("Test message"):
            messages.append(msg)

        assert len(messages) == 3
        assert isinstance(messages[0], InitMessage)
        assert isinstance(messages[1], TextMessage)
        assert isinstance(messages[2], StopMessage)

    @pytest.mark.asyncio
    async def test_send_message_increments_turn_count(self, mock_executor, mock_events):
        """Test send_message increments turn count."""
        async def mock_execute(*args, **kwargs):
            for event in mock_events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(executor=mock_executor)

        assert session.turn_count == 0

        async for _ in session.send_message("First message"):
            pass
        assert session.turn_count == 1

        async for _ in session.send_message("Second message"):
            pass
        assert session.turn_count == 2

    @pytest.mark.asyncio
    async def test_send_message_tracks_tokens(self, mock_executor, mock_events):
        """Test send_message tracks token usage."""
        async def mock_execute(*args, **kwargs):
            for event in mock_events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(executor=mock_executor)

        async for _ in session.send_message("Test"):
            pass

        assert session._total_input_tokens == 100
        assert session._total_output_tokens == 50

    @pytest.mark.asyncio
    async def test_send_message_accumulates_tokens(self, mock_executor):
        """Test send_message accumulates tokens across turns."""
        events1 = [
            {"type": "system", "subtype": "init", "session_id": "sess-1"},
            {"type": "system", "subtype": "stop", "session_id": "sess-1",
             "usage": {"input_tokens": 100, "output_tokens": 50}},
        ]
        events2 = [
            {"type": "system", "subtype": "init", "session_id": "sess-1"},
            {"type": "system", "subtype": "stop", "session_id": "sess-1",
             "usage": {"input_tokens": 150, "output_tokens": 75}},
        ]

        call_count = [0]
        async def mock_execute(*args, **kwargs):
            events = events1 if call_count[0] == 0 else events2
            call_count[0] += 1
            for event in events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(executor=mock_executor)

        async for _ in session.send_message("First"):
            pass
        async for _ in session.send_message("Second"):
            pass

        assert session._total_input_tokens == 250  # 100 + 150
        assert session._total_output_tokens == 125  # 50 + 75

    @pytest.mark.asyncio
    async def test_send_message_updates_session_id_on_first_message(self, mock_executor, mock_events):
        """Test send_message updates session ID from CLI on first message."""
        async def mock_execute(*args, **kwargs):
            for event in mock_events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(session_id="initial-id", executor=mock_executor)

        async for _ in session.send_message("Test"):
            pass

        # Session ID should be updated from CLI's init message
        assert session.session_id == "cli-session-abc"

    @pytest.mark.asyncio
    async def test_send_message_raises_on_closed_session(self, mock_executor):
        """Test send_message raises RuntimeError on closed session."""
        session = Session(executor=mock_executor)
        await session.close()

        with pytest.raises(RuntimeError, match="Cannot send message on closed session"):
            async for _ in session.send_message("Test"):
                pass

    @pytest.mark.asyncio
    async def test_send_message_sets_resume_after_first_message(self, mock_executor, mock_events):
        """Test send_message sets resume flag after first message."""
        captured_flags = []

        async def mock_execute(prompt, flags, **kwargs):
            captured_flags.append(list(flags))
            for event in mock_events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(executor=mock_executor)

        # First message - should not have --resume
        async for _ in session.send_message("First"):
            pass
        assert "--resume" not in captured_flags[0]

        # Second message - should have --resume
        async for _ in session.send_message("Second"):
            pass
        assert "--resume" in captured_flags[1]

    @pytest.mark.asyncio
    async def test_send_message_with_context(self, mock_executor, mock_events):
        """Test send_message with context injection."""
        captured_flags = []

        async def mock_execute(prompt, flags, **kwargs):
            captured_flags.append(list(flags))
            for event in mock_events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(executor=mock_executor)

        async for _ in session.send_message("Test", context={"key": "value"}):
            pass

        # Context should be injected via append_system_prompt
        assert "--append-system-prompt" in captured_flags[0]


class TestSessionFork:
    """Tests for Session.fork()."""

    @pytest.mark.asyncio
    async def test_fork_creates_new_session(self):
        """Test fork creates a new session instance."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_executor = MagicMock()
            mock_get_executor.return_value = mock_executor

            session = Session(session_id="parent-session")
            forked = await session.fork()

        assert forked is not session
        assert isinstance(forked, Session)

    @pytest.mark.asyncio
    async def test_fork_inherits_options(self):
        """Test forked session inherits parent options."""
        options = CLIAgentOptions(
            model="opus",
            permission_mode=PermissionMode.BYPASS,
            max_budget_usd=10.0,
        )

        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_executor = MagicMock()
            mock_get_executor.return_value = mock_executor

            session = Session(options=options)
            forked = await session.fork()

        assert forked._options.model == "opus"
        assert forked._options.max_budget_usd == 10.0

    @pytest.mark.asyncio
    async def test_fork_sets_fork_session_flag(self):
        """Test forked session has fork_session flag set."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_executor = MagicMock()
            mock_get_executor.return_value = mock_executor

            session = Session(session_id="parent-session")
            forked = await session.fork()

        assert forked._options.fork_session is True
        assert forked._options.resume is True

    @pytest.mark.asyncio
    async def test_fork_shares_executor(self):
        """Test forked session shares parent's executor."""
        mock_executor = MagicMock(spec=CLIExecutor)
        session = Session(executor=mock_executor)
        forked = await session.fork()

        assert forked._executor is mock_executor


class TestSessionClose:
    """Tests for Session.close()."""

    @pytest.mark.asyncio
    async def test_close_returns_session_stats(self):
        """Test close returns SessionStats."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session(session_id="test-session")
            stats = await session.close()

        assert isinstance(stats, SessionStats)
        assert stats.session_id == "test-session"

    @pytest.mark.asyncio
    async def test_close_sets_is_closed(self):
        """Test close sets is_closed flag."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session()

        assert session.is_closed is False
        await session.close()
        assert session.is_closed is True

    @pytest.mark.asyncio
    async def test_close_returns_correct_stats(self):
        """Test close returns accurate statistics."""
        mock_executor = MagicMock(spec=CLIExecutor)
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        events = [
            {"type": "system", "subtype": "init", "session_id": "stats-test"},
            {"type": "system", "subtype": "stop", "session_id": "stats-test",
             "usage": {"input_tokens": 200, "output_tokens": 100}},
        ]

        async def mock_execute(*args, **kwargs):
            for event in events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(executor=mock_executor)

        # Send two messages
        async for _ in session.send_message("First"):
            pass
        async for _ in session.send_message("Second"):
            pass

        stats = await session.close()

        assert stats.total_turns == 2
        assert stats.total_input_tokens == 400  # 200 * 2
        assert stats.total_output_tokens == 200  # 100 * 2
        assert stats.total_tokens == 600
        assert stats.duration_seconds > 0

    @pytest.mark.asyncio
    async def test_close_can_be_called_multiple_times(self):
        """Test close can be called multiple times without error."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = Session()

        await session.close()
        # Should not raise, just log warning
        await session.close()

        assert session.is_closed is True


class TestResumeSession:
    """Tests for resume_session function."""

    @pytest.mark.asyncio
    async def test_resume_session_returns_session(self):
        """Test resume_session returns a Session instance."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = await resume_session("existing-session-id")

        assert isinstance(session, Session)
        assert session.session_id == "existing-session-id"

    @pytest.mark.asyncio
    async def test_resume_session_sets_resume_flag(self):
        """Test resume_session sets resume flag."""
        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = await resume_session("existing-session-id")

        assert session._options.resume is True
        assert session._is_first_message is False

    @pytest.mark.asyncio
    async def test_resume_session_uses_provided_options(self):
        """Test resume_session uses provided options."""
        options = CLIAgentOptions(model="haiku", max_budget_usd=5.0)

        with patch('ccflow.session.get_executor') as mock_get_executor:
            mock_get_executor.return_value = MagicMock()
            session = await resume_session("existing-session-id", options=options)

        assert session._options.model == "haiku"
        assert session._options.max_budget_usd == 5.0
        assert session._options.resume is True

    @pytest.mark.asyncio
    async def test_resumed_session_uses_resume_on_first_send(self):
        """Test resumed session uses --resume on first send_message."""
        mock_executor = MagicMock(spec=CLIExecutor)
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        captured_flags = []
        events = [
            {"type": "system", "subtype": "init", "session_id": "resumed-session"},
            {"type": "system", "subtype": "stop", "session_id": "resumed-session", "usage": {}},
        ]

        async def mock_execute(prompt, flags, **kwargs):
            captured_flags.append(list(flags))
            for event in events:
                yield event

        mock_executor.execute = mock_execute

        with patch('ccflow.session.get_executor', return_value=mock_executor):
            session = await resume_session("existing-session-id")

        async for _ in session.send_message("Test"):
            pass

        # First message on resumed session should have --resume
        assert "--resume" in captured_flags[0]


class TestSessionIntegration:
    """Integration tests for Session with mocked executor."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test a multi-turn conversation flow."""
        mock_executor = MagicMock(spec=CLIExecutor)
        mock_executor.build_flags = CLIExecutor(cli_path="/usr/bin/claude").build_flags

        turn_responses = [
            [
                {"type": "system", "subtype": "init", "session_id": "multi-turn"},
                {"type": "message", "content": "I understand.", "delta_type": "text_delta"},
                {"type": "system", "subtype": "stop", "session_id": "multi-turn",
                 "usage": {"input_tokens": 50, "output_tokens": 25}},
            ],
            [
                {"type": "system", "subtype": "init", "session_id": "multi-turn"},
                {"type": "message", "content": "Here's more detail.", "delta_type": "text_delta"},
                {"type": "system", "subtype": "stop", "session_id": "multi-turn",
                 "usage": {"input_tokens": 75, "output_tokens": 40}},
            ],
            [
                {"type": "system", "subtype": "init", "session_id": "multi-turn"},
                {"type": "message", "content": "Final answer.", "delta_type": "text_delta"},
                {"type": "system", "subtype": "stop", "session_id": "multi-turn",
                 "usage": {"input_tokens": 100, "output_tokens": 50}},
            ],
        ]

        call_count = [0]

        async def mock_execute(*args, **kwargs):
            events = turn_responses[call_count[0]]
            call_count[0] += 1
            for event in events:
                yield event

        mock_executor.execute = mock_execute
        session = Session(executor=mock_executor)

        # Turn 1
        texts = []
        async for msg in session.send_message("Tell me about X"):
            if isinstance(msg, TextMessage):
                texts.append(msg.content)
        assert texts == ["I understand."]

        # Turn 2
        texts = []
        async for msg in session.send_message("More details please"):
            if isinstance(msg, TextMessage):
                texts.append(msg.content)
        assert texts == ["Here's more detail."]

        # Turn 3
        texts = []
        async for msg in session.send_message("Summarize"):
            if isinstance(msg, TextMessage):
                texts.append(msg.content)
        assert texts == ["Final answer."]

        # Close and verify stats
        stats = await session.close()
        assert stats.total_turns == 3
        assert stats.total_input_tokens == 225  # 50 + 75 + 100
        assert stats.total_output_tokens == 115  # 25 + 40 + 50
