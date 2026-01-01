"""
Reliability patterns for production middleware.

Provides circuit breaker, retry with backoff, and health checking
for robust CLI execution.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections.abc import Awaitable, Callable  # noqa: TC003
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

import structlog

logger = structlog.get_logger(__name__)

# Context variable for request correlation ID
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="")

T = TypeVar("T")


def get_correlation_id() -> str:
    """Get current correlation ID or generate new one."""
    cid = correlation_id_var.get()
    if not cid:
        cid = str(uuid.uuid4())[:8]
        correlation_id_var.set(cid)
    return cid


def set_correlation_id(cid: str | None = None) -> str:
    """Set correlation ID for current context.

    Args:
        cid: Correlation ID to set, or None to generate new one

    Returns:
        The correlation ID that was set
    """
    if cid is None:
        cid = str(uuid.uuid4())[:8]
    correlation_id_var.set(cid)
    return cid


def bind_correlation_id(log: Any = None) -> structlog.BoundLogger:
    """Bind correlation ID to logger.

    Args:
        log: Optional logger to bind to, uses module logger if None

    Returns:
        Logger with correlation_id bound
    """
    log = log or logger
    return log.bind(correlation_id=get_correlation_id())


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests rejected immediately
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker.

    Attributes:
        failure_threshold: Failures before opening circuit
        success_threshold: Successes in half-open before closing
        reset_timeout: Seconds before attempting half-open
        half_open_max_calls: Max concurrent calls in half-open state
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    reset_timeout: float = 30.0
    half_open_max_calls: int = 1


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker."""

    state: CircuitState
    failure_count: int
    success_count: int
    last_failure_time: float | None
    last_success_time: float | None
    total_calls: int
    total_failures: int
    total_rejections: int


class CircuitBreakerError(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, message: str = "Circuit breaker is open", retry_after: float = 0.0):
        super().__init__(message)
        self.retry_after = retry_after


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service
    is experiencing issues.

    Example:
        >>> breaker = CircuitBreaker()
        >>> async with breaker.call():
        ...     result = await risky_operation()

        # Or with decorator
        >>> @breaker.protect
        ... async def my_function():
        ...     return await risky_operation()
    """

    def __init__(
        self,
        config: CircuitBreakerConfig | None = None,
        name: str = "default",
    ):
        self.config = config or CircuitBreakerConfig()
        self.name = name
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._last_success_time: float | None = None
        self._half_open_calls = 0
        self._lock = asyncio.Lock()

        # Statistics
        self._total_calls = 0
        self._total_failures = 0
        self._total_rejections = 0

        self._log = logger.bind(circuit=name)

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state

    @property
    def stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return CircuitBreakerStats(
            state=self._state,
            failure_count=self._failure_count,
            success_count=self._success_count,
            last_failure_time=self._last_failure_time,
            last_success_time=self._last_success_time,
            total_calls=self._total_calls,
            total_failures=self._total_failures,
            total_rejections=self._total_rejections,
        )

    async def _check_state(self) -> None:
        """Check and potentially transition state."""
        if self._state == CircuitState.OPEN and self._last_failure_time is not None:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self.config.reset_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0
                self._half_open_calls = 0
                self._log.info(
                    "circuit_half_open",
                    elapsed=elapsed,
                    correlation_id=get_correlation_id(),
                )

    async def _record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._last_success_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self._half_open_calls -= 1

                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._log.info(
                        "circuit_closed",
                        success_count=self._success_count,
                        correlation_id=get_correlation_id(),
                    )
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    async def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        async with self._lock:
            self._failure_count += 1
            self._total_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls -= 1
                self._state = CircuitState.OPEN
                self._log.warning(
                    "circuit_reopened",
                    error=str(error),
                    correlation_id=get_correlation_id(),
                )
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._log.warning(
                        "circuit_opened",
                        failure_count=self._failure_count,
                        threshold=self.config.failure_threshold,
                        correlation_id=get_correlation_id(),
                    )

    @asynccontextmanager
    async def call(self):
        """Context manager for protected calls.

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            await self._check_state()

            if self._state == CircuitState.OPEN:
                self._total_rejections += 1
                retry_after = 0.0
                if self._last_failure_time:
                    elapsed = time.monotonic() - self._last_failure_time
                    retry_after = max(0, self.config.reset_timeout - elapsed)
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open",
                    retry_after=retry_after,
                )

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.half_open_max_calls:
                    self._total_rejections += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is half-open, max calls reached",
                        retry_after=1.0,
                    )
                self._half_open_calls += 1

            self._total_calls += 1

        try:
            yield
            await self._record_success()
        except Exception as e:
            await self._record_failure(e)
            raise

    def protect(
        self, func: Callable[..., Awaitable[T]]
    ) -> Callable[..., Awaitable[T]]:
        """Decorator to protect an async function with circuit breaker.

        Example:
            >>> @breaker.protect
            ... async def risky_call():
            ...     return await external_service()
        """

        async def wrapper(*args: Any, **kwargs: Any) -> T:
            async with self.call():
                return await func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper  # type: ignore[return-value]

    async def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        async with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._log.info("circuit_reset", correlation_id=get_correlation_id())


# Global circuit breaker for CLI execution
_cli_circuit_breaker: CircuitBreaker | None = None


def get_cli_circuit_breaker() -> CircuitBreaker:
    """Get the global CLI circuit breaker."""
    global _cli_circuit_breaker
    if _cli_circuit_breaker is None:
        _cli_circuit_breaker = CircuitBreaker(name="cli")
    return _cli_circuit_breaker


def reset_cli_circuit_breaker() -> None:
    """Reset the global CLI circuit breaker."""
    global _cli_circuit_breaker
    _cli_circuit_breaker = None


# =============================================================================
# Retry with Backoff
# =============================================================================


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        exponential_base: Base for exponential backoff
        jitter: Add random jitter (0.0-1.0 as fraction of delay)
        retryable_errors: Exception types that should trigger retry
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: float = 0.1
    retryable_errors: tuple[type[Exception], ...] = field(
        default_factory=lambda: (
            ConnectionError,
            TimeoutError,
            OSError,
        )
    )


@dataclass
class RetryStats:
    """Statistics from a retry operation."""

    attempts: int
    total_delay: float
    success: bool
    final_error: Exception | None = None


class RetryExhaustedError(Exception):
    """Raised when all retries are exhausted."""

    def __init__(
        self,
        message: str,
        attempts: int,
        last_error: Exception | None = None,
    ):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for a given retry attempt.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        Delay in seconds with jitter applied
    """
    # Exponential backoff
    delay = config.base_delay * (config.exponential_base**attempt)

    # Cap at max delay
    delay = min(delay, config.max_delay)

    # Add jitter
    if config.jitter > 0:
        jitter_range = delay * config.jitter
        delay += random.uniform(-jitter_range, jitter_range)

    return max(0, delay)


async def retry_with_backoff(
    func: Callable[..., Awaitable[T]],
    *args: Any,
    config: RetryConfig | None = None,
    **kwargs: Any,
) -> T:
    """Execute function with retry and exponential backoff.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        Result from successful function call

    Raises:
        RetryExhaustedError: If all retries are exhausted

    Example:
        >>> result = await retry_with_backoff(
        ...     risky_call,
        ...     arg1, arg2,
        ...     config=RetryConfig(max_retries=5),
        ... )
    """
    config = config or RetryConfig()
    log = bind_correlation_id()

    last_error: Exception | None = None
    total_delay = 0.0

    for attempt in range(config.max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except config.retryable_errors as e:
            last_error = e

            if attempt == config.max_retries:
                log.warning(
                    "retry_exhausted",
                    attempts=attempt + 1,
                    error=str(e),
                    total_delay=total_delay,
                )
                raise RetryExhaustedError(
                    f"All {config.max_retries + 1} attempts failed",
                    attempts=attempt + 1,
                    last_error=e,
                ) from e

            delay = calculate_delay(attempt, config)
            total_delay += delay

            log.info(
                "retry_attempt",
                attempt=attempt + 1,
                max_retries=config.max_retries,
                delay=delay,
                error=str(e),
            )

            await asyncio.sleep(delay)
        except Exception:
            # Non-retryable error, re-raise immediately
            raise

    # Should never reach here
    raise RetryExhaustedError(
        "Unexpected retry loop exit",
        attempts=config.max_retries + 1,
        last_error=last_error,
    )


def with_retry(
    config: RetryConfig | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator for retry with backoff.

    Example:
        >>> @with_retry(RetryConfig(max_retries=5))
        ... async def risky_operation():
        ...     return await external_call()
    """
    cfg = config or RetryConfig()

    def decorator(
        func: Callable[..., Awaitable[T]],
    ) -> Callable[..., Awaitable[T]]:
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await retry_with_backoff(func, *args, config=cfg, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper  # type: ignore[return-value]

    return decorator  # type: ignore[return-value]


# =============================================================================
# Health Check
# =============================================================================


@dataclass
class HealthStatus:
    """Health check result."""

    healthy: bool
    cli_available: bool
    cli_executable: bool
    cli_authenticated: bool
    last_check_time: float
    latency_ms: float | None = None
    error: str | None = None
    version: str | None = None


class HealthChecker:
    """Deep health checking for CLI availability.

    Performs cached health checks to avoid hammering the CLI.

    Example:
        >>> checker = HealthChecker()
        >>> status = await checker.check()
        >>> if status.healthy:
        ...     print("CLI is ready")
    """

    def __init__(
        self,
        cache_ttl: float = 30.0,
        timeout: float = 10.0,
    ):
        """Initialize health checker.

        Args:
            cache_ttl: How long to cache health check results (seconds)
            timeout: Timeout for health check operations (seconds)
        """
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._cached_status: HealthStatus | None = None
        self._lock = asyncio.Lock()
        self._log = logger.bind(component="health_checker")

    async def check(self, force: bool = False) -> HealthStatus:
        """Perform health check.

        Args:
            force: Bypass cache and perform fresh check

        Returns:
            Health status
        """
        async with self._lock:
            # Check cache
            if not force and self._cached_status is not None:
                elapsed = time.monotonic() - self._cached_status.last_check_time
                if elapsed < self.cache_ttl:
                    return self._cached_status

            # Perform fresh check
            status = await self._perform_check()
            self._cached_status = status

            log = bind_correlation_id(self._log)
            if status.healthy:
                log.debug(
                    "health_check_passed",
                    latency_ms=status.latency_ms,
                    version=status.version,
                )
            else:
                log.warning(
                    "health_check_failed",
                    error=status.error,
                    cli_available=status.cli_available,
                    cli_executable=status.cli_executable,
                    cli_authenticated=status.cli_authenticated,
                )

            return status

    async def _perform_check(self) -> HealthStatus:
        """Execute the actual health check."""
        import shutil

        start_time = time.monotonic()
        check_time = start_time

        # Step 1: Check CLI is in PATH
        cli_path = shutil.which("claude")
        if cli_path is None:
            return HealthStatus(
                healthy=False,
                cli_available=False,
                cli_executable=False,
                cli_authenticated=False,
                last_check_time=check_time,
                error="Claude CLI not found in PATH",
            )

        # Step 2: Check CLI is executable (version check)
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "claude",
                    "--version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self.timeout,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )

            if proc.returncode != 0:
                return HealthStatus(
                    healthy=False,
                    cli_available=True,
                    cli_executable=False,
                    cli_authenticated=False,
                    last_check_time=check_time,
                    error=f"CLI version check failed: {stderr.decode().strip()}",
                )

            version = stdout.decode().strip()

        except TimeoutError:
            return HealthStatus(
                healthy=False,
                cli_available=True,
                cli_executable=False,
                cli_authenticated=False,
                last_check_time=check_time,
                error="CLI version check timed out",
            )
        except Exception as e:
            return HealthStatus(
                healthy=False,
                cli_available=True,
                cli_executable=False,
                cli_authenticated=False,
                last_check_time=check_time,
                error=f"CLI execution error: {e}",
            )

        # Step 3: Deep check - execute minimal query
        # We use a simple prompt that should return quickly
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    "claude",
                    "-p",
                    "--print",
                    "--output-format",
                    "json",
                    "--max-budget-usd",
                    "0.01",
                    "Reply with exactly: OK",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=self.timeout,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )

            latency_ms = (time.monotonic() - start_time) * 1000

            if proc.returncode != 0:
                stderr_text = stderr.decode().strip()
                # Check for auth errors
                if "auth" in stderr_text.lower() or "login" in stderr_text.lower():
                    return HealthStatus(
                        healthy=False,
                        cli_available=True,
                        cli_executable=True,
                        cli_authenticated=False,
                        last_check_time=check_time,
                        latency_ms=latency_ms,
                        version=version,
                        error=f"CLI not authenticated: {stderr_text}",
                    )
                return HealthStatus(
                    healthy=False,
                    cli_available=True,
                    cli_executable=True,
                    cli_authenticated=False,
                    last_check_time=check_time,
                    latency_ms=latency_ms,
                    version=version,
                    error=f"CLI query failed: {stderr_text}",
                )

            # Success!
            return HealthStatus(
                healthy=True,
                cli_available=True,
                cli_executable=True,
                cli_authenticated=True,
                last_check_time=check_time,
                latency_ms=latency_ms,
                version=version,
            )

        except TimeoutError:
            latency_ms = (time.monotonic() - start_time) * 1000
            return HealthStatus(
                healthy=False,
                cli_available=True,
                cli_executable=True,
                cli_authenticated=False,
                last_check_time=check_time,
                latency_ms=latency_ms,
                version=version,
                error="CLI query timed out",
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            return HealthStatus(
                healthy=False,
                cli_available=True,
                cli_executable=True,
                cli_authenticated=False,
                last_check_time=check_time,
                latency_ms=latency_ms,
                version=version,
                error=f"CLI query error: {e}",
            )

    def invalidate_cache(self) -> None:
        """Clear cached health status."""
        self._cached_status = None


# Global health checker
_health_checker: HealthChecker | None = None


def get_health_checker() -> HealthChecker:
    """Get the global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker


def reset_health_checker() -> None:
    """Reset the global health checker."""
    global _health_checker
    _health_checker = None
