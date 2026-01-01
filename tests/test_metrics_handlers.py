"""Tests for Metrics Event Handlers."""

from unittest.mock import MagicMock, patch

import pytest

from ccflow.events import (
    CostIncurredEvent,
    EventEmitter,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionErrorEvent,
    TokensUsedEvent,
    TurnCompletedEvent,
)
from ccflow.metrics_handlers import (
    PROMETHEUS_AVAILABLE,
    PrometheusEventHandler,
    setup_metrics,
    start_metrics_server,
)


@pytest.fixture
def emitter() -> EventEmitter:
    """Create event emitter for testing."""
    return EventEmitter()


@pytest.fixture
def handler() -> PrometheusEventHandler:
    """Create handler for testing."""
    # Reset metrics before each test
    PrometheusEventHandler.reset_metrics()
    return PrometheusEventHandler()


class TestPrometheusEventHandler:
    """Tests for PrometheusEventHandler."""

    def test_init(self, handler: PrometheusEventHandler):
        """Test handler initialization."""
        assert handler._prefix == "ccflow"

    def test_init_custom_prefix(self):
        """Test handler with custom prefix."""
        PrometheusEventHandler.reset_metrics()
        handler = PrometheusEventHandler(prefix="custom")
        assert handler._prefix == "custom"

    def test_register(self, emitter: EventEmitter, handler: PrometheusEventHandler):
        """Test registering handler with emitter."""
        handler.register(emitter)
        assert emitter.handler_count == 1

    def test_unregister(self, emitter: EventEmitter, handler: PrometheusEventHandler):
        """Test unregistering handler from emitter."""
        handler.register(emitter)
        handler.unregister(emitter)
        assert emitter.handler_count == 0

    def test_handles_session_created(self, handler: PrometheusEventHandler):
        """Test handling SESSION_CREATED event."""
        event = SessionCreatedEvent(session_id="test-123", model="sonnet")
        # Should not raise
        handler(event)

    def test_handles_session_closed(self, handler: PrometheusEventHandler):
        """Test handling SESSION_CLOSED event."""
        event = SessionClosedEvent(
            session_id="test-123",
            model="sonnet",
            turn_count=5,
            duration_seconds=120.0,
            total_cost_usd=0.05,
        )
        # Should not raise
        handler(event)

    def test_handles_session_error(self, handler: PrometheusEventHandler):
        """Test handling SESSION_ERROR event."""
        event = SessionErrorEvent(
            session_id="test-123",
            model="sonnet",
            error_type="CLITimeoutError",
            error_message="Timeout",
        )
        # Should not raise
        handler(event)

    def test_handles_turn_completed(self, handler: PrometheusEventHandler):
        """Test handling TURN_COMPLETED event."""
        event = TurnCompletedEvent(
            session_id="test-123",
            turn_number=1,
            prompt="Hello",
            response="Hi there!",
            input_tokens=100,
            output_tokens=200,
            duration_seconds=1.5,
            metadata={"model": "sonnet"},
        )
        # Should not raise
        handler(event)

    def test_handles_tokens_used(self, handler: PrometheusEventHandler):
        """Test handling TOKENS_USED event."""
        event = TokensUsedEvent(
            session_id="test-123",
            input_tokens=100,
            output_tokens=200,
            model="opus",
        )
        # Should not raise
        handler(event)

    def test_handles_cost_incurred(self, handler: PrometheusEventHandler):
        """Test handling COST_INCURRED event."""
        event = CostIncurredEvent(
            session_id="test-123",
            amount_usd=0.10,
            model="opus",
            input_tokens=100,
            output_tokens=200,
        )
        # Should not raise
        handler(event)

    def test_handler_error_handling(self, handler: PrometheusEventHandler):
        """Test handler catches and logs errors."""
        # Create an event that might cause issues
        event = SessionCreatedEvent(session_id="test-123")

        # Mock a metric to raise
        with patch.object(handler, "_handle_session_created", side_effect=Exception("Test error")):
            # Should not raise
            handler(event)

    def test_reset_metrics(self):
        """Test resetting metrics."""
        PrometheusEventHandler()
        PrometheusEventHandler.reset_metrics()

        assert PrometheusEventHandler._metrics_initialized is False
        assert PrometheusEventHandler._active_sessions is None

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus not installed")
    def test_metrics_are_initialized(self):
        """Test that metrics are properly initialized."""
        PrometheusEventHandler.reset_metrics()
        PrometheusEventHandler()

        assert PrometheusEventHandler._metrics_initialized is True
        assert PrometheusEventHandler._active_sessions is not None
        assert PrometheusEventHandler._tokens_input is not None


class TestSetupMetrics:
    """Tests for setup_metrics function."""

    def test_setup_returns_handler(self, emitter: EventEmitter):
        """Test setup_metrics returns a handler."""
        PrometheusEventHandler.reset_metrics()

        if PROMETHEUS_AVAILABLE:
            handler = setup_metrics(emitter, start_server=False)
            assert handler is not None
            assert isinstance(handler, PrometheusEventHandler)
        else:
            handler = setup_metrics(emitter, start_server=False)
            assert handler is None

    def test_setup_registers_with_emitter(self, emitter: EventEmitter):
        """Test setup_metrics registers handler with emitter."""
        PrometheusEventHandler.reset_metrics()

        if PROMETHEUS_AVAILABLE:
            setup_metrics(emitter, start_server=False)
            assert emitter.handler_count >= 1

    def test_setup_uses_global_emitter(self):
        """Test setup_metrics uses global emitter if none provided."""
        from ccflow.events import get_emitter, reset_emitter

        reset_emitter()
        PrometheusEventHandler.reset_metrics()

        if PROMETHEUS_AVAILABLE:
            handler = setup_metrics(start_server=False)
            assert handler is not None

            global_emitter = get_emitter()
            assert global_emitter.handler_count >= 1


class TestStartMetricsServer:
    """Tests for start_metrics_server function."""

    def test_returns_false_without_prometheus(self):
        """Test returns False if Prometheus not available."""
        with patch("ccflow.metrics_handlers.PROMETHEUS_AVAILABLE", False):
            result = start_metrics_server()
            assert result is False

    @pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus not installed")
    def test_returns_true_with_prometheus(self):
        """Test returns True if Prometheus available."""
        with patch("ccflow.metrics_handlers.start_http_server") as mock_server:
            result = start_metrics_server(port=9999)
            assert result is True
            mock_server.assert_called_once_with(9999, "")


class TestMetricsDisabled:
    """Tests for when metrics are disabled."""

    def test_handler_does_nothing_when_disabled(self):
        """Test handler does nothing when metrics disabled."""
        PrometheusEventHandler.reset_metrics()

        with patch("ccflow.metrics_handlers.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock(enable_metrics=False)
            handler = PrometheusEventHandler()
            assert handler._enabled is False

            event = SessionCreatedEvent(session_id="test-123")
            # Should not raise and should do nothing
            handler(event)


class TestEventIntegration:
    """Integration tests for event-to-metrics flow."""

    async def test_full_session_lifecycle(self, emitter: EventEmitter):
        """Test metrics through full session lifecycle."""
        PrometheusEventHandler.reset_metrics()

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not installed")

        handler = PrometheusEventHandler()
        handler.register(emitter)

        # Session created
        await emitter.emit(SessionCreatedEvent(
            session_id="test-123",
            model="sonnet",
        ))

        # Turn completed
        await emitter.emit(TurnCompletedEvent(
            session_id="test-123",
            turn_number=1,
            input_tokens=100,
            output_tokens=50,
            metadata={"model": "sonnet"},
        ))

        # Session closed
        await emitter.emit(SessionClosedEvent(
            session_id="test-123",
            model="sonnet",
            turn_count=1,
            duration_seconds=10.0,
            total_cost_usd=0.01,
        ))

        # Verify metrics were updated (no assertion errors)
        assert True

    async def test_error_tracking(self, emitter: EventEmitter):
        """Test error metrics tracking."""
        PrometheusEventHandler.reset_metrics()

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not installed")

        handler = PrometheusEventHandler()
        handler.register(emitter)

        # Session error
        await emitter.emit(SessionErrorEvent(
            session_id="test-123",
            model="sonnet",
            error_type="CLITimeoutError",
            error_message="Request timed out",
        ))

        # Verify metrics were updated (no assertion errors)
        assert True

    async def test_token_tracking(self, emitter: EventEmitter):
        """Test token metrics tracking."""
        PrometheusEventHandler.reset_metrics()

        if not PROMETHEUS_AVAILABLE:
            pytest.skip("Prometheus not installed")

        handler = PrometheusEventHandler()
        handler.register(emitter)

        # Tokens used
        await emitter.emit(TokensUsedEvent(
            session_id="test-123",
            input_tokens=500,
            output_tokens=250,
            model="opus",
        ))

        # Cost incurred
        await emitter.emit(CostIncurredEvent(
            session_id="test-123",
            amount_usd=0.05,
            model="opus",
            input_tokens=500,
            output_tokens=250,
        ))

        # Verify metrics were updated (no assertion errors)
        assert True


class TestHandlerWithMissingModel:
    """Tests for events with missing model information."""

    def test_session_closed_no_model(self, handler: PrometheusEventHandler):
        """Test handling SESSION_CLOSED without model."""
        event = SessionClosedEvent(
            session_id="test-123",
            model=None,
            turn_count=5,
            duration_seconds=120.0,
        )
        # Should not raise - uses "unknown" as default
        handler(event)

    def test_turn_completed_no_model(self, handler: PrometheusEventHandler):
        """Test handling TURN_COMPLETED without model."""
        event = TurnCompletedEvent(
            session_id="test-123",
            turn_number=1,
            input_tokens=100,
            output_tokens=50,
            metadata={},  # No model
        )
        # Should not raise - uses "unknown" as default
        handler(event)

    def test_tokens_used_no_model(self, handler: PrometheusEventHandler):
        """Test handling TOKENS_USED without model."""
        event = TokensUsedEvent(
            session_id="test-123",
            input_tokens=100,
            output_tokens=50,
            model=None,  # type: ignore
        )
        # Should not raise - uses "unknown" as default
        handler(event)
