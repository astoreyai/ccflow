"""Tests for Event System."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from ccflow.events import (
    CostIncurredEvent,
    CostTracker,
    Event,
    EventEmitter,
    EventType,
    LoggingHandler,
    MessageEvent,
    MetricsHandler,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionDeletedEvent,
    SessionErrorEvent,
    SessionEvent,
    SessionLoadedEvent,
    SessionPersistedEvent,
    SessionResumedEvent,
    TokensUsedEvent,
    ToolCalledEvent,
    ToolEvent,
    ToolResultEvent,
    TurnCompletedEvent,
    TurnEvent,
    TurnStartedEvent,
    get_emitter,
    reset_emitter,
)


class TestEventTypes:
    """Tests for EventType enum."""

    def test_session_lifecycle_events(self):
        """Test session lifecycle event types exist."""
        assert EventType.SESSION_CREATED == "session.created"
        assert EventType.SESSION_RESUMED == "session.resumed"
        assert EventType.SESSION_CLOSED == "session.closed"
        assert EventType.SESSION_ERROR == "session.error"

    def test_message_flow_events(self):
        """Test message flow event types exist."""
        assert EventType.TURN_STARTED == "turn.started"
        assert EventType.TURN_COMPLETED == "turn.completed"
        assert EventType.MESSAGE_RECEIVED == "message.received"

    def test_tool_events(self):
        """Test tool event types exist."""
        assert EventType.TOOL_CALLED == "tool.called"
        assert EventType.TOOL_RESULT == "tool.result"

    def test_usage_events(self):
        """Test usage tracking event types exist."""
        assert EventType.TOKENS_USED == "tokens.used"
        assert EventType.COST_INCURRED == "cost.incurred"

    def test_store_events(self):
        """Test store event types exist."""
        assert EventType.SESSION_PERSISTED == "session.persisted"
        assert EventType.SESSION_LOADED == "session.loaded"
        assert EventType.SESSION_DELETED == "session.deleted"


class TestBaseEvent:
    """Tests for base Event class."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event(type=EventType.SESSION_CREATED)

        assert event.type == EventType.SESSION_CREATED
        assert isinstance(event.timestamp, datetime)
        assert event.session_id is None
        assert event.metadata == {}

    def test_event_with_session_id(self):
        """Test event with session ID."""
        event = Event(
            type=EventType.SESSION_CREATED,
            session_id="test-123",
        )
        assert event.session_id == "test-123"

    def test_event_with_metadata(self):
        """Test event with metadata."""
        event = Event(
            type=EventType.SESSION_CREATED,
            metadata={"key": "value"},
        )
        assert event.metadata == {"key": "value"}


class TestSessionEvents:
    """Tests for session-related events."""

    def test_session_event(self):
        """Test SessionEvent base class."""
        event = SessionEvent(
            type=EventType.SESSION_CREATED,
            session_id="test-123",
            model="opus",
            turn_count=5,
            total_tokens=1000,
        )

        assert event.model == "opus"
        assert event.turn_count == 5
        assert event.total_tokens == 1000

    def test_session_created_event(self):
        """Test SessionCreatedEvent."""
        event = SessionCreatedEvent(
            session_id="test-123",
            model="sonnet",
            tags=["test", "dev"],
        )

        assert event.type == EventType.SESSION_CREATED
        assert event.tags == ["test", "dev"]

    def test_session_resumed_event(self):
        """Test SessionResumedEvent."""
        event = SessionResumedEvent(
            session_id="test-123",
            previous_turn_count=10,
        )

        assert event.type == EventType.SESSION_RESUMED
        assert event.previous_turn_count == 10

    def test_session_closed_event(self):
        """Test SessionClosedEvent."""
        event = SessionClosedEvent(
            session_id="test-123",
            duration_seconds=120.5,
            total_cost_usd=0.05,
        )

        assert event.type == EventType.SESSION_CLOSED
        assert event.duration_seconds == 120.5
        assert event.total_cost_usd == 0.05

    def test_session_error_event(self):
        """Test SessionErrorEvent."""
        event = SessionErrorEvent(
            session_id="test-123",
            error_type="CLITimeoutError",
            error_message="Command timed out",
        )

        assert event.type == EventType.SESSION_ERROR
        assert event.error_type == "CLITimeoutError"
        assert event.error_message == "Command timed out"


class TestTurnEvents:
    """Tests for turn-related events."""

    def test_turn_event(self):
        """Test TurnEvent base class."""
        event = TurnEvent(
            type=EventType.TURN_STARTED,
            turn_number=3,
            prompt="Hello",
        )

        assert event.turn_number == 3
        assert event.prompt == "Hello"

    def test_turn_started_event(self):
        """Test TurnStartedEvent."""
        event = TurnStartedEvent(
            session_id="test-123",
            turn_number=1,
            prompt="Hello",
            context_size=5000,
        )

        assert event.type == EventType.TURN_STARTED
        assert event.context_size == 5000

    def test_turn_completed_event(self):
        """Test TurnCompletedEvent."""
        event = TurnCompletedEvent(
            session_id="test-123",
            turn_number=1,
            prompt="Hello",
            response="Hi there!",
            input_tokens=10,
            output_tokens=20,
            duration_seconds=1.5,
        )

        assert event.type == EventType.TURN_COMPLETED
        assert event.response == "Hi there!"
        assert event.input_tokens == 10
        assert event.output_tokens == 20
        assert event.duration_seconds == 1.5


class TestToolEvents:
    """Tests for tool-related events."""

    def test_tool_event(self):
        """Test ToolEvent base class."""
        event = ToolEvent(
            type=EventType.TOOL_CALLED,
            tool_name="read_file",
            tool_id="tool-123",
        )

        assert event.tool_name == "read_file"
        assert event.tool_id == "tool-123"

    def test_tool_called_event(self):
        """Test ToolCalledEvent."""
        event = ToolCalledEvent(
            tool_name="read_file",
            tool_id="tool-123",
            arguments={"path": "/test/file.py"},
        )

        assert event.type == EventType.TOOL_CALLED
        assert event.arguments == {"path": "/test/file.py"}

    def test_tool_result_event(self):
        """Test ToolResultEvent."""
        event = ToolResultEvent(
            tool_name="read_file",
            tool_id="tool-123",
            result="file contents",
            success=True,
        )

        assert event.type == EventType.TOOL_RESULT
        assert event.result == "file contents"
        assert event.success is True
        assert event.error is None

    def test_tool_result_event_with_error(self):
        """Test ToolResultEvent with error."""
        event = ToolResultEvent(
            tool_name="read_file",
            tool_id="tool-123",
            result="",
            success=False,
            error="File not found",
        )

        assert event.success is False
        assert event.error == "File not found"


class TestUsageEvents:
    """Tests for usage tracking events."""

    def test_tokens_used_event(self):
        """Test TokensUsedEvent."""
        event = TokensUsedEvent(
            session_id="test-123",
            input_tokens=100,
            output_tokens=200,
            model="sonnet",
        )

        assert event.type == EventType.TOKENS_USED
        assert event.input_tokens == 100
        assert event.output_tokens == 200
        assert event.model == "sonnet"

    def test_cost_incurred_event(self):
        """Test CostIncurredEvent."""
        event = CostIncurredEvent(
            session_id="test-123",
            amount_usd=0.05,
            model="opus",
            input_tokens=100,
            output_tokens=200,
        )

        assert event.type == EventType.COST_INCURRED
        assert event.amount_usd == 0.05
        assert event.model == "opus"


class TestStoreEvents:
    """Tests for store-related events."""

    def test_session_persisted_event(self):
        """Test SessionPersistedEvent."""
        event = SessionPersistedEvent(session_id="test-123")
        assert event.type == EventType.SESSION_PERSISTED

    def test_session_loaded_event(self):
        """Test SessionLoadedEvent."""
        event = SessionLoadedEvent(
            session_id="test-123",
            turn_count=5,
            total_tokens=1000,
        )

        assert event.type == EventType.SESSION_LOADED
        assert event.turn_count == 5
        assert event.total_tokens == 1000

    def test_session_deleted_event(self):
        """Test SessionDeletedEvent."""
        event = SessionDeletedEvent(session_id="test-123")
        assert event.type == EventType.SESSION_DELETED


class TestEventEmitter:
    """Tests for EventEmitter class."""

    @pytest.fixture
    def emitter(self) -> EventEmitter:
        """Create event emitter for testing."""
        return EventEmitter()

    def test_init(self, emitter: EventEmitter):
        """Test emitter initialization."""
        assert emitter.handler_count == 0

    def test_on_decorator(self, emitter: EventEmitter):
        """Test on decorator for registering handlers."""

        @emitter.on(EventType.SESSION_CREATED)
        def handler(event: Event):
            pass

        assert emitter.handler_count == 1

    def test_on_decorator_global(self, emitter: EventEmitter):
        """Test on decorator for global handlers."""

        @emitter.on()  # No event type = global
        def handler(event: Event):
            pass

        assert emitter.handler_count == 1
        assert len(emitter._global_handlers) == 1

    def test_add_handler(self, emitter: EventEmitter):
        """Test add_handler method."""

        def handler(event: Event):
            pass

        emitter.add_handler(handler, EventType.SESSION_CREATED)
        assert emitter.handler_count == 1

    def test_add_handler_global(self, emitter: EventEmitter):
        """Test add_handler for global handlers."""

        def handler(event: Event):
            pass

        emitter.add_handler(handler)  # No event type = global
        assert len(emitter._global_handlers) == 1

    def test_remove_handler(self, emitter: EventEmitter):
        """Test remove_handler method."""

        def handler(event: Event):
            pass

        emitter.add_handler(handler, EventType.SESSION_CREATED)
        assert emitter.handler_count == 1

        result = emitter.remove_handler(handler, EventType.SESSION_CREATED)
        assert result is True
        assert emitter.handler_count == 0

    def test_remove_handler_not_found(self, emitter: EventEmitter):
        """Test remove_handler when handler not found."""

        def handler(event: Event):
            pass

        result = emitter.remove_handler(handler, EventType.SESSION_CREATED)
        assert result is False

    def test_remove_global_handler(self, emitter: EventEmitter):
        """Test removing global handler."""

        def handler(event: Event):
            pass

        emitter.add_handler(handler)
        result = emitter.remove_handler(handler)

        assert result is True
        assert len(emitter._global_handlers) == 0

    async def test_emit_sync_handler(self, emitter: EventEmitter):
        """Test emitting to sync handler."""
        called = []

        def handler(event: Event):
            called.append(event)

        emitter.add_handler(handler, EventType.SESSION_CREATED)
        event = SessionCreatedEvent(session_id="test-123")
        await emitter.emit(event)

        assert len(called) == 1
        assert called[0] is event

    async def test_emit_async_handler(self, emitter: EventEmitter):
        """Test emitting to async handler."""
        called = []

        async def handler(event: Event):
            called.append(event)

        emitter.add_handler(handler, EventType.SESSION_CREATED)
        event = SessionCreatedEvent(session_id="test-123")
        await emitter.emit(event)

        assert len(called) == 1

    async def test_emit_global_handler(self, emitter: EventEmitter):
        """Test emitting to global handler."""
        called = []

        def handler(event: Event):
            called.append(event.type)

        emitter.add_handler(handler)  # Global

        await emitter.emit(SessionCreatedEvent(session_id="test-1"))
        await emitter.emit(SessionClosedEvent(session_id="test-2"))

        assert len(called) == 2
        assert EventType.SESSION_CREATED in called
        assert EventType.SESSION_CLOSED in called

    async def test_emit_multiple_handlers(self, emitter: EventEmitter):
        """Test emitting to multiple handlers."""
        called = []

        def handler1(event: Event):
            called.append("handler1")

        def handler2(event: Event):
            called.append("handler2")

        emitter.add_handler(handler1, EventType.SESSION_CREATED)
        emitter.add_handler(handler2, EventType.SESSION_CREATED)

        await emitter.emit(SessionCreatedEvent(session_id="test-123"))

        assert called == ["handler1", "handler2"]

    async def test_emit_handler_error(self, emitter: EventEmitter):
        """Test that handler errors don't stop emission."""
        called = []

        def bad_handler(event: Event):
            raise ValueError("Handler error")

        def good_handler(event: Event):
            called.append(event)

        emitter.add_handler(bad_handler, EventType.SESSION_CREATED)
        emitter.add_handler(good_handler, EventType.SESSION_CREATED)

        # Should not raise
        await emitter.emit(SessionCreatedEvent(session_id="test-123"))

        # Good handler should still be called
        assert len(called) == 1

    def test_emit_sync(self, emitter: EventEmitter):
        """Test synchronous emit."""
        called = []

        def handler(event: Event):
            called.append(event)

        emitter.add_handler(handler, EventType.SESSION_CREATED)
        emitter.emit_sync(SessionCreatedEvent(session_id="test-123"))

        assert len(called) == 1

    def test_emit_sync_skips_async_handlers(self, emitter: EventEmitter):
        """Test emit_sync skips async handlers with warning."""
        called = []

        async def async_handler(event: Event):
            called.append(event)

        emitter.add_handler(async_handler, EventType.SESSION_CREATED)
        emitter.emit_sync(SessionCreatedEvent(session_id="test-123"))

        # Async handler should be skipped
        assert len(called) == 0

    def test_emit_sync_handler_error(self, emitter: EventEmitter):
        """Test emit_sync handles errors gracefully."""
        called = []

        def bad_handler(event: Event):
            raise ValueError("Handler error")

        def good_handler(event: Event):
            called.append(event)

        emitter.add_handler(bad_handler, EventType.SESSION_CREATED)
        emitter.add_handler(good_handler, EventType.SESSION_CREATED)

        emitter.emit_sync(SessionCreatedEvent(session_id="test-123"))

        assert len(called) == 1

    def test_clear_specific(self, emitter: EventEmitter):
        """Test clearing handlers for specific event type."""

        def handler(event: Event):
            pass

        emitter.add_handler(handler, EventType.SESSION_CREATED)
        emitter.add_handler(handler, EventType.SESSION_CLOSED)

        emitter.clear(EventType.SESSION_CREATED)

        # Only SESSION_CREATED handlers should be cleared
        assert EventType.SESSION_CREATED not in emitter._handlers or \
               len(emitter._handlers[EventType.SESSION_CREATED]) == 0

    def test_clear_all(self, emitter: EventEmitter):
        """Test clearing all handlers."""

        def handler(event: Event):
            pass

        emitter.add_handler(handler, EventType.SESSION_CREATED)
        emitter.add_handler(handler)  # Global

        emitter.clear()

        assert emitter.handler_count == 0


class TestLoggingHandler:
    """Tests for LoggingHandler."""

    def test_init_default(self):
        """Test default initialization."""
        handler = LoggingHandler()
        assert handler._level == "debug"

    def test_init_custom_level(self):
        """Test custom log level."""
        handler = LoggingHandler(level="info")
        assert handler._level == "info"

    def test_call(self):
        """Test handler callable."""
        handler = LoggingHandler()
        event = SessionCreatedEvent(
            session_id="test-123",
            metadata={"key": "value"},
        )

        # Should not raise
        handler(event)


class TestMetricsHandler:
    """Tests for MetricsHandler."""

    @pytest.fixture
    def handler(self) -> MetricsHandler:
        """Create metrics handler for testing."""
        return MetricsHandler()

    def test_init(self, handler: MetricsHandler):
        """Test initialization."""
        metrics = handler.metrics
        assert metrics["sessions_created"] == 0
        assert metrics["total_cost_usd"] == 0.0

    def test_metrics_returns_copy(self, handler: MetricsHandler):
        """Test metrics property returns copy."""
        metrics1 = handler.metrics
        metrics1["sessions_created"] = 100

        # Original should be unchanged
        assert handler.metrics["sessions_created"] == 0

    def test_reset(self, handler: MetricsHandler):
        """Test reset method."""
        handler._metrics["sessions_created"] = 10
        handler._metrics["total_cost_usd"] = 5.0

        handler.reset()

        assert handler._metrics["sessions_created"] == 0
        assert handler._metrics["total_cost_usd"] == 0.0

    def test_session_created(self, handler: MetricsHandler):
        """Test tracking session created."""
        handler(SessionCreatedEvent(session_id="test-123"))
        assert handler.metrics["sessions_created"] == 1

    def test_session_closed(self, handler: MetricsHandler):
        """Test tracking session closed."""
        handler(SessionClosedEvent(
            session_id="test-123",
            total_cost_usd=0.05,
        ))

        assert handler.metrics["sessions_closed"] == 1
        assert handler.metrics["total_cost_usd"] == 0.05

    def test_turn_completed(self, handler: MetricsHandler):
        """Test tracking turn completed."""
        handler(TurnCompletedEvent(
            session_id="test-123",
            turn_number=1,
            input_tokens=100,
            output_tokens=200,
        ))

        assert handler.metrics["turns_completed"] == 1
        assert handler.metrics["total_input_tokens"] == 100
        assert handler.metrics["total_output_tokens"] == 200

    def test_tokens_used(self, handler: MetricsHandler):
        """Test tracking tokens used."""
        handler(TokensUsedEvent(
            session_id="test-123",
            input_tokens=50,
            output_tokens=100,
            model="sonnet",
        ))

        assert handler.metrics["total_input_tokens"] == 50
        assert handler.metrics["total_output_tokens"] == 100

    def test_cost_incurred(self, handler: MetricsHandler):
        """Test tracking cost incurred."""
        handler(CostIncurredEvent(
            session_id="test-123",
            amount_usd=0.10,
            model="opus",
            input_tokens=100,
            output_tokens=200,
        ))

        assert handler.metrics["total_cost_usd"] == 0.10

    def test_tool_called(self, handler: MetricsHandler):
        """Test tracking tool called."""
        handler(ToolCalledEvent(
            tool_name="read_file",
            tool_id="tool-123",
        ))

        assert handler.metrics["tools_called"] == 1

    def test_session_error(self, handler: MetricsHandler):
        """Test tracking session error."""
        handler(SessionErrorEvent(
            session_id="test-123",
            error_type="CLITimeoutError",
            error_message="Timeout",
        ))

        assert handler.metrics["errors"] == 1


class TestCostTracker:
    """Tests for CostTracker."""

    @pytest.fixture
    def tracker(self) -> CostTracker:
        """Create cost tracker for testing."""
        return CostTracker()

    def test_init(self, tracker: CostTracker):
        """Test initialization."""
        assert tracker.get_total_cost() == 0.0

    def test_get_session_cost_unknown(self, tracker: CostTracker):
        """Test getting cost for unknown session."""
        assert tracker.get_session_cost("unknown") == 0.0

    def test_get_session_tokens_unknown(self, tracker: CostTracker):
        """Test getting tokens for unknown session."""
        tokens = tracker.get_session_tokens("unknown")
        assert tokens == {"input": 0, "output": 0}

    def test_turn_completed_tracking(self, tracker: CostTracker):
        """Test tracking from turn completed event."""
        tracker(TurnCompletedEvent(
            session_id="test-123",
            turn_number=1,
            input_tokens=1000,
            output_tokens=500,
            metadata={"model": "sonnet"},
        ))

        tokens = tracker.get_session_tokens("test-123")
        assert tokens["input"] == 1000
        assert tokens["output"] == 500

        cost = tracker.get_session_cost("test-123")
        assert cost > 0

    def test_tokens_used_tracking(self, tracker: CostTracker):
        """Test tracking from tokens used event."""
        tracker(TokensUsedEvent(
            session_id="test-123",
            input_tokens=1000,
            output_tokens=500,
            model="opus",
        ))

        tokens = tracker.get_session_tokens("test-123")
        assert tokens["input"] == 1000
        assert tokens["output"] == 500

    def test_cost_calculation_sonnet(self, tracker: CostTracker):
        """Test cost calculation for Sonnet model."""
        # 1M input tokens = $3, 1M output tokens = $15
        tracker(TokensUsedEvent(
            session_id="test-123",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="sonnet",
        ))

        cost = tracker.get_session_cost("test-123")
        assert abs(cost - 18.0) < 0.01  # $3 + $15

    def test_cost_calculation_opus(self, tracker: CostTracker):
        """Test cost calculation for Opus model."""
        # 1M input tokens = $15, 1M output tokens = $75
        tracker(TokensUsedEvent(
            session_id="test-123",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="opus",
        ))

        cost = tracker.get_session_cost("test-123")
        assert abs(cost - 90.0) < 0.01  # $15 + $75

    def test_cost_calculation_haiku(self, tracker: CostTracker):
        """Test cost calculation for Haiku model (Claude 3.5 Haiku)."""
        # 1M input tokens = $0.80, 1M output tokens = $4.00
        tracker(TokensUsedEvent(
            session_id="test-123",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="haiku",
        ))

        cost = tracker.get_session_cost("test-123")
        assert abs(cost - 4.8) < 0.01  # $0.80 + $4.00

    def test_cost_calculation_unknown_model(self, tracker: CostTracker):
        """Test cost calculation defaults to Sonnet for unknown model."""
        tracker(TokensUsedEvent(
            session_id="test-123",
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            model="unknown-model",
        ))

        cost = tracker.get_session_cost("test-123")
        assert abs(cost - 18.0) < 0.01  # Sonnet pricing

    def test_get_total_cost(self, tracker: CostTracker):
        """Test total cost across all sessions."""
        tracker(TokensUsedEvent(
            session_id="session-1",
            input_tokens=100_000,
            output_tokens=100_000,
            model="sonnet",
        ))
        tracker(TokensUsedEvent(
            session_id="session-2",
            input_tokens=100_000,
            output_tokens=100_000,
            model="sonnet",
        ))

        total = tracker.get_total_cost()
        expected = 2 * (0.1 * 3.0 + 0.1 * 15.0)  # 2 sessions
        assert abs(total - expected) < 0.01

    def test_clear(self, tracker: CostTracker):
        """Test clearing tracked costs."""
        tracker(TokensUsedEvent(
            session_id="test-123",
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
        ))

        tracker.clear()

        assert tracker.get_total_cost() == 0.0
        assert tracker.get_session_cost("test-123") == 0.0

    def test_no_session_id(self, tracker: CostTracker):
        """Test event without session ID is ignored."""
        tracker(TokensUsedEvent(
            session_id=None,  # type: ignore
            input_tokens=1000,
            output_tokens=500,
            model="sonnet",
        ))

        assert tracker.get_total_cost() == 0.0

    def test_cumulative_tracking(self, tracker: CostTracker):
        """Test cumulative token tracking."""
        tracker(TurnCompletedEvent(
            session_id="test-123",
            turn_number=1,
            input_tokens=100,
            output_tokens=50,
            metadata={"model": "sonnet"},
        ))
        tracker(TurnCompletedEvent(
            session_id="test-123",
            turn_number=2,
            input_tokens=200,
            output_tokens=100,
            metadata={"model": "sonnet"},
        ))

        tokens = tracker.get_session_tokens("test-123")
        assert tokens["input"] == 300
        assert tokens["output"] == 150


class TestGlobalEmitter:
    """Tests for global emitter functions."""

    def test_get_emitter(self):
        """Test get_emitter returns emitter."""
        reset_emitter()

        emitter = get_emitter()
        assert isinstance(emitter, EventEmitter)

    def test_get_emitter_singleton(self):
        """Test get_emitter returns same instance."""
        reset_emitter()

        emitter1 = get_emitter()
        emitter2 = get_emitter()

        assert emitter1 is emitter2

    def test_reset_emitter(self):
        """Test reset_emitter clears global emitter."""
        emitter1 = get_emitter()

        def handler(event: Event):
            pass

        emitter1.add_handler(handler)

        reset_emitter()
        emitter2 = get_emitter()

        # Should be a new instance
        assert emitter2.handler_count == 0


class TestMessageEvent:
    """Tests for MessageEvent."""

    def test_message_event(self):
        """Test MessageEvent creation."""
        event = MessageEvent(
            session_id="test-123",
            message_type="text",
            content="Hello world",
        )

        assert event.type == EventType.MESSAGE_RECEIVED
        assert event.message_type == "text"
        assert event.content == "Hello world"
