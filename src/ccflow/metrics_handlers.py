"""
Metrics Event Handlers - Bridge between event system and Prometheus metrics.

Provides event handlers that automatically update Prometheus metrics
when session events occur, enabling observability without code changes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from ccflow.config import get_settings
from ccflow.events import (
    CostIncurredEvent,
    Event,
    EventType,
    SessionClosedEvent,
    SessionErrorEvent,
    TokensUsedEvent,
    TurnCompletedEvent,
)

if TYPE_CHECKING:
    from ccflow.events import EventEmitter

logger = structlog.get_logger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore[misc,assignment]
    Gauge = None  # type: ignore[misc,assignment]
    Histogram = None  # type: ignore[misc,assignment]
    start_http_server = None  # type: ignore[assignment]


class PrometheusEventHandler:
    """Event handler that updates Prometheus metrics.

    Listens to session events and updates corresponding Prometheus
    metrics for monitoring and alerting.

    Example:
        >>> from ccflow.events import get_emitter
        >>> from ccflow.metrics_handlers import PrometheusEventHandler
        >>>
        >>> handler = PrometheusEventHandler()
        >>> emitter = get_emitter()
        >>> handler.register(emitter)
        >>>
        >>> # Now all session events will update Prometheus metrics
    """

    # Metric definitions (class-level, shared across instances)
    _metrics_initialized = False
    _requests_total: Counter | None = None
    _request_duration: Histogram | None = None
    _tokens_input: Counter | None = None
    _tokens_output: Counter | None = None
    _active_sessions: Gauge | None = None
    _session_turns: Histogram | None = None
    _session_duration: Histogram | None = None
    _session_cost: Counter | None = None
    _errors_total: Counter | None = None
    _turns_total: Counter | None = None

    def __init__(self, prefix: str = "ccflow") -> None:
        """Initialize Prometheus event handler.

        Args:
            prefix: Metric name prefix (default: ccflow)
        """
        self._prefix = prefix
        self._enabled = get_settings().enable_metrics and PROMETHEUS_AVAILABLE

        if self._enabled and not PrometheusEventHandler._metrics_initialized:
            self._initialize_metrics()

    def _initialize_metrics(self) -> None:
        """Initialize Prometheus metric objects."""
        if not PROMETHEUS_AVAILABLE:
            return

        prefix = self._prefix

        # Request/Turn metrics
        PrometheusEventHandler._requests_total = Counter(
            f"{prefix}_requests_total",
            "Total requests by model and status",
            ["model", "status"],
        )

        PrometheusEventHandler._request_duration = Histogram(
            f"{prefix}_request_duration_seconds",
            "Request duration in seconds",
            ["model"],
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        PrometheusEventHandler._turns_total = Counter(
            f"{prefix}_turns_total",
            "Total conversation turns",
            ["model"],
        )

        # Token metrics
        PrometheusEventHandler._tokens_input = Counter(
            f"{prefix}_tokens_input_total",
            "Total input tokens consumed",
            ["model"],
        )

        PrometheusEventHandler._tokens_output = Counter(
            f"{prefix}_tokens_output_total",
            "Total output tokens generated",
            ["model"],
        )

        # Session metrics
        PrometheusEventHandler._active_sessions = Gauge(
            f"{prefix}_active_sessions",
            "Currently active sessions",
        )

        PrometheusEventHandler._session_turns = Histogram(
            f"{prefix}_session_turns",
            "Number of turns per session",
            buckets=[1, 2, 3, 5, 10, 20, 50, 100],
        )

        PrometheusEventHandler._session_duration = Histogram(
            f"{prefix}_session_duration_seconds",
            "Session duration in seconds",
            buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],
        )

        PrometheusEventHandler._session_cost = Counter(
            f"{prefix}_session_cost_usd_total",
            "Total cost in USD",
            ["model"],
        )

        # Error metrics
        PrometheusEventHandler._errors_total = Counter(
            f"{prefix}_errors_total",
            "Total errors by type",
            ["error_type"],
        )

        PrometheusEventHandler._metrics_initialized = True
        logger.debug("prometheus_metrics_initialized", prefix=prefix)

    def register(self, emitter: EventEmitter) -> None:
        """Register this handler with an event emitter.

        Args:
            emitter: Event emitter to register with
        """
        # Register for all events
        emitter.add_handler(self)
        logger.debug("prometheus_handler_registered")

    def unregister(self, emitter: EventEmitter) -> None:
        """Unregister this handler from an event emitter.

        Args:
            emitter: Event emitter to unregister from
        """
        emitter.remove_handler(self)
        logger.debug("prometheus_handler_unregistered")

    def __call__(self, event: Event) -> None:
        """Handle event by updating metrics.

        Args:
            event: Event to process
        """
        if not self._enabled:
            return

        try:
            if event.type == EventType.SESSION_CREATED:
                self._handle_session_created(event)
            elif event.type == EventType.SESSION_CLOSED:
                self._handle_session_closed(event)
            elif event.type == EventType.SESSION_ERROR:
                self._handle_session_error(event)
            elif event.type == EventType.TURN_COMPLETED:
                self._handle_turn_completed(event)
            elif event.type == EventType.TOKENS_USED:
                self._handle_tokens_used(event)
            elif event.type == EventType.COST_INCURRED:
                self._handle_cost_incurred(event)
        except Exception as e:
            logger.warning(
                "prometheus_handler_error",
                event_type=event.type.value,
                error=str(e),
            )

    def _handle_session_created(self, _event: Event) -> None:
        """Handle session created event."""
        if self._active_sessions is not None:
            self._active_sessions.inc()

    def _handle_session_closed(self, event: Event) -> None:
        """Handle session closed event."""
        if not isinstance(event, SessionClosedEvent):
            return

        if self._active_sessions is not None:
            self._active_sessions.dec()

        if self._session_turns is not None:
            self._session_turns.observe(event.turn_count)

        if self._session_duration is not None:
            self._session_duration.observe(event.duration_seconds)

        if self._session_cost is not None and event.total_cost_usd > 0:
            model = event.model or "unknown"
            self._session_cost.labels(model=model).inc(event.total_cost_usd)

        if self._requests_total is not None:
            model = event.model or "unknown"
            self._requests_total.labels(model=model, status="success").inc()

        if self._request_duration is not None:
            model = event.model or "unknown"
            self._request_duration.labels(model=model).observe(event.duration_seconds)

    def _handle_session_error(self, event: Event) -> None:
        """Handle session error event."""
        if not isinstance(event, SessionErrorEvent):
            return

        if self._errors_total is not None:
            self._errors_total.labels(error_type=event.error_type).inc()

        if self._requests_total is not None:
            model = event.model or "unknown"
            self._requests_total.labels(model=model, status="error").inc()

    def _handle_turn_completed(self, event: Event) -> None:
        """Handle turn completed event."""
        if not isinstance(event, TurnCompletedEvent):
            return

        model = event.metadata.get("model", "unknown")

        if self._turns_total is not None:
            self._turns_total.labels(model=model).inc()

        if self._tokens_input is not None and event.input_tokens > 0:
            self._tokens_input.labels(model=model).inc(event.input_tokens)

        if self._tokens_output is not None and event.output_tokens > 0:
            self._tokens_output.labels(model=model).inc(event.output_tokens)

    def _handle_tokens_used(self, event: Event) -> None:
        """Handle tokens used event."""
        if not isinstance(event, TokensUsedEvent):
            return

        model = event.model or "unknown"

        if self._tokens_input is not None and event.input_tokens > 0:
            self._tokens_input.labels(model=model).inc(event.input_tokens)

        if self._tokens_output is not None and event.output_tokens > 0:
            self._tokens_output.labels(model=model).inc(event.output_tokens)

    def _handle_cost_incurred(self, event: Event) -> None:
        """Handle cost incurred event."""
        if not isinstance(event, CostIncurredEvent):
            return

        model = event.model or "unknown"

        if self._session_cost is not None and event.amount_usd > 0:
            self._session_cost.labels(model=model).inc(event.amount_usd)

    @classmethod
    def reset_metrics(cls) -> None:
        """Reset all metrics (for testing).

        This unregisters metrics from the Prometheus registry to allow
        re-initialization with potentially different configurations.
        """
        if PROMETHEUS_AVAILABLE:
            from prometheus_client import REGISTRY

            # Unregister all metrics from the registry
            collectors_to_remove = []
            for collector in REGISTRY._names_to_collectors.values():
                # Check if it's one of our metrics by name prefix
                if hasattr(collector, "_name"):
                    name = collector._name
                    if name.startswith("ccflow_") or name.startswith("custom_"):
                        collectors_to_remove.append(collector)

            import contextlib

            for collector in collectors_to_remove:
                with contextlib.suppress(Exception):
                    REGISTRY.unregister(collector)

        cls._metrics_initialized = False
        cls._requests_total = None
        cls._request_duration = None
        cls._tokens_input = None
        cls._tokens_output = None
        cls._active_sessions = None
        cls._session_turns = None
        cls._session_duration = None
        cls._session_cost = None
        cls._errors_total = None
        cls._turns_total = None


def start_metrics_server(port: int = 9090, addr: str = "") -> bool:
    """Start Prometheus metrics HTTP server.

    Args:
        port: Port to listen on (default: 9090)
        addr: Address to bind to (default: all interfaces)

    Returns:
        True if server started, False if Prometheus not available
    """
    if not PROMETHEUS_AVAILABLE or start_http_server is None:
        logger.warning(
            "metrics_server_unavailable",
            message="Install prometheus-client to enable metrics server",
        )
        return False

    start_http_server(port, addr)
    logger.info("metrics_server_started", port=port, addr=addr or "0.0.0.0")
    return True


def setup_metrics(
    emitter: EventEmitter | None = None,
    start_server: bool = False,
    server_port: int = 9090,
) -> PrometheusEventHandler | None:
    """Convenience function to set up metrics collection.

    Args:
        emitter: Event emitter to register with (uses global if None)
        start_server: Whether to start HTTP metrics server
        server_port: Port for metrics server

    Returns:
        PrometheusEventHandler if setup successful, None otherwise

    Example:
        >>> from ccflow.events import get_emitter
        >>> from ccflow.metrics_handlers import setup_metrics
        >>>
        >>> handler = setup_metrics(start_server=True, server_port=8080)
        >>> # Metrics now available at http://localhost:8080/metrics
    """
    if not PROMETHEUS_AVAILABLE:
        logger.warning(
            "prometheus_not_available",
            message="Install prometheus-client: pip install prometheus-client",
        )
        return None

    # Create and register handler
    handler = PrometheusEventHandler()

    if emitter is None:
        from ccflow.events import get_emitter

        emitter = get_emitter()

    handler.register(emitter)

    # Optionally start server
    if start_server:
        start_metrics_server(server_port)

    return handler
