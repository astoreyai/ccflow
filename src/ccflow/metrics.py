"""
Metrics & Observability - Prometheus metrics and monitoring.

Provides metrics collection for requests, tokens, TOON savings,
and session statistics.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Generator, TypeVar

import structlog

from ccflow.config import get_settings

logger = structlog.get_logger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import Counter, Gauge, Histogram, Info

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None  # type: ignore
    Gauge = None  # type: ignore
    Histogram = None  # type: ignore
    Info = None  # type: ignore
    logger.debug("prometheus_not_available", message="Install with: pip install prometheus-client")


# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


# Metric definitions (only created if prometheus is available)
if PROMETHEUS_AVAILABLE:
    # Request metrics
    REQUESTS_TOTAL = Counter(
        "ccflow_requests_total",
        "Total CLI requests",
        ["model", "status"],
    )

    REQUEST_DURATION = Histogram(
        "ccflow_request_duration_seconds",
        "Request duration in seconds",
        ["model"],
        buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
    )

    # Token metrics
    TOKENS_INPUT = Counter(
        "ccflow_tokens_input_total",
        "Total input tokens consumed",
        ["model"],
    )

    TOKENS_OUTPUT = Counter(
        "ccflow_tokens_output_total",
        "Total output tokens generated",
        ["model"],
    )

    # TOON metrics
    TOON_SAVINGS_RATIO = Histogram(
        "ccflow_toon_savings_ratio",
        "TOON compression ratio (0-1, higher is better)",
        buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    )

    TOON_TOKENS_SAVED = Counter(
        "ccflow_toon_tokens_saved_total",
        "Total tokens saved by TOON encoding",
    )

    # Session metrics
    ACTIVE_SESSIONS = Gauge(
        "ccflow_active_sessions",
        "Currently active sessions",
    )

    SESSION_TURNS = Histogram(
        "ccflow_session_turns",
        "Number of turns per session",
        buckets=[1, 2, 3, 5, 10, 20, 50],
    )

    # Error metrics
    ERRORS_TOTAL = Counter(
        "ccflow_errors_total",
        "Total errors by type",
        ["error_type"],
    )

    # Build info
    BUILD_INFO = Info(
        "ccflow_build",
        "Build information",
    )

else:
    # Dummy metrics when prometheus not available
    REQUESTS_TOTAL = None
    REQUEST_DURATION = None
    TOKENS_INPUT = None
    TOKENS_OUTPUT = None
    TOON_SAVINGS_RATIO = None
    TOON_TOKENS_SAVED = None
    ACTIVE_SESSIONS = None
    SESSION_TURNS = None
    ERRORS_TOTAL = None
    BUILD_INFO = None


def record_request(
    model: str,
    status: str,
    duration: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """Record metrics for a completed request.

    Args:
        model: Model used for request
        status: Request status (success, error, timeout)
        duration: Request duration in seconds
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
    """
    settings = get_settings()
    if not settings.enable_metrics or not PROMETHEUS_AVAILABLE:
        return

    REQUESTS_TOTAL.labels(model=model, status=status).inc()
    REQUEST_DURATION.labels(model=model).observe(duration)

    if input_tokens > 0:
        TOKENS_INPUT.labels(model=model).inc(input_tokens)
    if output_tokens > 0:
        TOKENS_OUTPUT.labels(model=model).inc(output_tokens)


def record_toon_savings(
    json_tokens: int,
    toon_tokens: int,
) -> None:
    """Record TOON compression metrics.

    Args:
        json_tokens: Token count for JSON encoding
        toon_tokens: Token count for TOON encoding
    """
    settings = get_settings()
    if not settings.enable_metrics or not PROMETHEUS_AVAILABLE:
        return

    if json_tokens > 0:
        ratio = 1.0 - (toon_tokens / json_tokens)
        saved = json_tokens - toon_tokens

        TOON_SAVINGS_RATIO.observe(ratio)
        TOON_TOKENS_SAVED.inc(saved)


def record_error(error_type: str) -> None:
    """Record an error occurrence.

    Args:
        error_type: Type/class of error
    """
    settings = get_settings()
    if not settings.enable_metrics or not PROMETHEUS_AVAILABLE:
        return

    ERRORS_TOTAL.labels(error_type=error_type).inc()


def track_session(increment: bool = True) -> None:
    """Track active session count.

    Args:
        increment: True to increment, False to decrement
    """
    settings = get_settings()
    if not settings.enable_metrics or not PROMETHEUS_AVAILABLE:
        return

    if increment:
        ACTIVE_SESSIONS.inc()
    else:
        ACTIVE_SESSIONS.dec()


def record_session_complete(turns: int) -> None:
    """Record completed session metrics.

    Args:
        turns: Number of turns in session
    """
    settings = get_settings()
    if not settings.enable_metrics or not PROMETHEUS_AVAILABLE:
        return

    SESSION_TURNS.observe(turns)
    ACTIVE_SESSIONS.dec()


@contextmanager
def timed_operation(
    model: str,
    operation: str = "request",
) -> Generator[dict[str, Any], None, None]:
    """Context manager for timing operations.

    Args:
        model: Model being used
        operation: Operation name for logging

    Yields:
        Dict to populate with result metadata
    """
    start = time.monotonic()
    result: dict[str, Any] = {
        "status": "success",
        "input_tokens": 0,
        "output_tokens": 0,
    }

    try:
        yield result
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        record_error(type(e).__name__)
        raise
    finally:
        duration = time.monotonic() - start
        result["duration"] = duration

        record_request(
            model=model,
            status=result["status"],
            duration=duration,
            input_tokens=result.get("input_tokens", 0),
            output_tokens=result.get("output_tokens", 0),
        )

        logger.info(
            f"{operation}_complete",
            model=model,
            status=result["status"],
            duration=f"{duration:.2f}s",
            tokens=result.get("input_tokens", 0) + result.get("output_tokens", 0),
        )


def set_build_info(version: str, **extra: str) -> None:
    """Set build information metric.

    Args:
        version: Package version
        **extra: Additional build info fields
    """
    if not PROMETHEUS_AVAILABLE or BUILD_INFO is None:
        return

    BUILD_INFO.info({
        "version": version,
        **extra,
    })


# Initialize build info on import
if PROMETHEUS_AVAILABLE and BUILD_INFO is not None:
    try:
        from ccflow import __version__
        set_build_info(__version__)
    except ImportError:
        pass
