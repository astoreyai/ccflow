"""
Rate Limiting & Concurrency Control.

Provides rate limiting and concurrency control for CLI requests
to prevent overloading and ensure fair resource usage.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, TypeVar

import structlog

from ccflow.config import get_settings
from ccflow.exceptions import CCFlowError

logger = structlog.get_logger(__name__)

T = TypeVar("T")


class RateLimitExceededError(CCFlowError):
    """Raised when rate limit is exceeded and no wait is desired."""

    def __init__(self, message: str, retry_after: float):
        super().__init__(message)
        self.retry_after = retry_after


class ConcurrencyLimitExceededError(CCFlowError):
    """Raised when concurrency limit is exceeded."""

    def __init__(self, message: str, current: int, limit: int):
        super().__init__(message)
        self.current = current
        self.limit = limit


@dataclass
class RateLimiterStats:
    """Statistics for rate limiter."""

    total_requests: int = 0
    total_waits: int = 0
    total_wait_time: float = 0.0
    rejected_requests: int = 0

    @property
    def average_wait_time(self) -> float:
        """Average wait time per waited request."""
        if self.total_waits == 0:
            return 0.0
        return self.total_wait_time / self.total_waits


class TokenBucketRateLimiter:
    """Token bucket rate limiter.

    Allows burst traffic while enforcing average rate over time.
    Tokens are added at a constant rate and consumed by requests.

    Example:
        >>> limiter = TokenBucketRateLimiter(rate=60, burst=10)  # 60/min, burst 10
        >>> async with limiter.acquire():
        ...     await make_request()
    """

    def __init__(
        self,
        rate: float,
        burst: int | None = None,
        *,
        wait_timeout: float | None = None,
    ) -> None:
        """Initialize token bucket rate limiter.

        Args:
            rate: Tokens per minute (requests per minute)
            burst: Maximum burst size (defaults to rate/10 or 1)
            wait_timeout: Maximum time to wait for token (None = wait forever)
        """
        self._rate = rate
        self._burst = burst or max(1, int(rate / 10))
        self._wait_timeout = wait_timeout

        # Token state
        self._tokens = float(self._burst)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

        # Stats
        self._stats = RateLimiterStats()

        logger.debug(
            "rate_limiter_initialized",
            rate=rate,
            burst=self._burst,
        )

    @property
    def rate(self) -> float:
        """Requests per minute."""
        return self._rate

    @property
    def burst(self) -> int:
        """Maximum burst size."""
        return self._burst

    @property
    def stats(self) -> RateLimiterStats:
        """Get current statistics."""
        return self._stats

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate)."""
        self._refill()
        return self._tokens

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self._last_update
        self._last_update = now

        # Add tokens based on elapsed time (rate is per minute)
        tokens_to_add = elapsed * (self._rate / 60.0)
        self._tokens = min(self._burst, self._tokens + tokens_to_add)

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens, waiting if necessary.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            Wait time in seconds (0 if no wait needed)

        Raises:
            RateLimitExceededError: If wait_timeout exceeded
        """
        async with self._lock:
            self._refill()
            self._stats.total_requests += 1

            if self._tokens >= tokens:
                self._tokens -= tokens
                return 0.0

            # Calculate wait time
            tokens_needed = tokens - self._tokens
            wait_time = tokens_needed / (self._rate / 60.0)

            # Check timeout
            if self._wait_timeout is not None and wait_time > self._wait_timeout:
                self._stats.rejected_requests += 1
                raise RateLimitExceededError(
                    f"Rate limit exceeded, retry after {wait_time:.2f}s",
                    retry_after=wait_time,
                )

            # Wait and consume
            self._stats.total_waits += 1
            self._stats.total_wait_time += wait_time

            logger.debug("rate_limit_wait", wait_time=f"{wait_time:.2f}s")

        # Release lock while waiting
        await asyncio.sleep(wait_time)

        async with self._lock:
            self._refill()
            self._tokens -= tokens

        return wait_time

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Try to acquire tokens without waiting.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if tokens acquired, False otherwise
        """
        self._refill()
        self._stats.total_requests += 1

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True

        self._stats.rejected_requests += 1
        return False

    @asynccontextmanager
    async def limit(self) -> AsyncIterator[float]:
        """Context manager for rate-limited operations.

        Yields:
            Wait time before operation started

        Example:
            >>> async with limiter.limit() as wait_time:
            ...     print(f"Waited {wait_time}s")
            ...     await do_work()
        """
        wait_time = await self.acquire()
        yield wait_time

    def reset(self) -> None:
        """Reset limiter to initial state."""
        self._tokens = float(self._burst)
        self._last_update = time.monotonic()
        self._stats = RateLimiterStats()


class SlidingWindowRateLimiter:
    """Sliding window rate limiter.

    Tracks requests in a sliding time window for more accurate
    rate limiting than token bucket.

    Example:
        >>> limiter = SlidingWindowRateLimiter(rate=60, window=60)  # 60/min
        >>> if limiter.try_acquire():
        ...     await make_request()
    """

    def __init__(
        self,
        rate: int,
        window: float = 60.0,
        *,
        wait_timeout: float | None = None,
    ) -> None:
        """Initialize sliding window rate limiter.

        Args:
            rate: Maximum requests per window
            window: Window size in seconds (default: 60s)
            wait_timeout: Maximum time to wait (None = wait forever)
        """
        self._rate = rate
        self._window = window
        self._wait_timeout = wait_timeout

        # Request timestamps
        self._requests: list[float] = []
        self._lock = asyncio.Lock()

        # Stats
        self._stats = RateLimiterStats()

        logger.debug(
            "sliding_window_limiter_initialized",
            rate=rate,
            window=window,
        )

    @property
    def rate(self) -> int:
        """Maximum requests per window."""
        return self._rate

    @property
    def window(self) -> float:
        """Window size in seconds."""
        return self._window

    @property
    def stats(self) -> RateLimiterStats:
        """Get current statistics."""
        return self._stats

    @property
    def current_count(self) -> int:
        """Current request count in window."""
        self._cleanup()
        return len(self._requests)

    def _cleanup(self) -> None:
        """Remove expired requests from window."""
        now = time.monotonic()
        cutoff = now - self._window
        self._requests = [t for t in self._requests if t > cutoff]

    async def acquire(self) -> float:
        """Acquire permission to make request, waiting if necessary.

        Returns:
            Wait time in seconds (0 if no wait needed)

        Raises:
            RateLimitExceededError: If wait_timeout exceeded
        """
        async with self._lock:
            self._cleanup()
            self._stats.total_requests += 1

            if len(self._requests) < self._rate:
                self._requests.append(time.monotonic())
                return 0.0

            # Calculate wait time until oldest request expires
            oldest = self._requests[0]
            now = time.monotonic()
            wait_time = (oldest + self._window) - now

            if wait_time <= 0:
                self._requests.append(time.monotonic())
                return 0.0

            # Check timeout
            if self._wait_timeout is not None and wait_time > self._wait_timeout:
                self._stats.rejected_requests += 1
                raise RateLimitExceededError(
                    f"Rate limit exceeded, retry after {wait_time:.2f}s",
                    retry_after=wait_time,
                )

            self._stats.total_waits += 1
            self._stats.total_wait_time += wait_time

            logger.debug("rate_limit_wait", wait_time=f"{wait_time:.2f}s")

        # Release lock while waiting
        await asyncio.sleep(wait_time)

        async with self._lock:
            self._cleanup()
            self._requests.append(time.monotonic())

        return wait_time

    def try_acquire(self) -> bool:
        """Try to acquire without waiting.

        Returns:
            True if acquired, False otherwise
        """
        self._cleanup()
        self._stats.total_requests += 1

        if len(self._requests) < self._rate:
            self._requests.append(time.monotonic())
            return True

        self._stats.rejected_requests += 1
        return False

    @asynccontextmanager
    async def limit(self) -> AsyncIterator[float]:
        """Context manager for rate-limited operations."""
        wait_time = await self.acquire()
        yield wait_time

    def reset(self) -> None:
        """Reset limiter to initial state."""
        self._requests.clear()
        self._stats = RateLimiterStats()


class ConcurrencyLimiter:
    """Limits concurrent operations using a semaphore.

    Example:
        >>> limiter = ConcurrencyLimiter(max_concurrent=5)
        >>> async with limiter.acquire():
        ...     await do_work()
    """

    def __init__(
        self,
        max_concurrent: int,
        *,
        wait_timeout: float | None = None,
    ) -> None:
        """Initialize concurrency limiter.

        Args:
            max_concurrent: Maximum concurrent operations
            wait_timeout: Maximum time to wait for slot (None = wait forever)
        """
        self._max_concurrent = max_concurrent
        self._wait_timeout = wait_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._current = 0
        self._lock = asyncio.Lock()

        # Stats
        self._total_acquired = 0
        self._total_waits = 0
        self._total_rejected = 0
        self._peak_concurrent = 0

        logger.debug("concurrency_limiter_initialized", max_concurrent=max_concurrent)

    @property
    def max_concurrent(self) -> int:
        """Maximum concurrent operations."""
        return self._max_concurrent

    @property
    def current(self) -> int:
        """Current concurrent operations."""
        return self._current

    @property
    def available(self) -> int:
        """Available slots."""
        return self._max_concurrent - self._current

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """Acquire a concurrency slot.

        Raises:
            ConcurrencyLimitExceededError: If wait_timeout exceeded
        """
        if self._wait_timeout is not None:
            try:
                acquired = await asyncio.wait_for(
                    self._semaphore.acquire(),
                    timeout=self._wait_timeout,
                )
            except asyncio.TimeoutError:
                self._total_rejected += 1
                raise ConcurrencyLimitExceededError(
                    f"Concurrency limit exceeded ({self._current}/{self._max_concurrent})",
                    current=self._current,
                    limit=self._max_concurrent,
                )
        else:
            await self._semaphore.acquire()

        async with self._lock:
            self._current += 1
            self._total_acquired += 1
            self._peak_concurrent = max(self._peak_concurrent, self._current)

        try:
            yield
        finally:
            self._semaphore.release()
            async with self._lock:
                self._current -= 1

    def try_acquire(self) -> bool:
        """Try to acquire slot without waiting.

        Returns:
            True if acquired, False otherwise
        """
        if self._semaphore.locked():
            if self._current >= self._max_concurrent:
                self._total_rejected += 1
                return False

        # Non-blocking acquire attempt
        try:
            self._semaphore._value -= 1
            if self._semaphore._value < 0:
                self._semaphore._value += 1
                self._total_rejected += 1
                return False
        except Exception:
            return False

        self._current += 1
        self._total_acquired += 1
        self._peak_concurrent = max(self._peak_concurrent, self._current)
        return True

    def release(self) -> None:
        """Release a previously acquired slot."""
        self._semaphore.release()
        self._current -= 1

    @property
    def stats(self) -> dict:
        """Get current statistics."""
        return {
            "total_acquired": self._total_acquired,
            "total_waits": self._total_waits,
            "total_rejected": self._total_rejected,
            "peak_concurrent": self._peak_concurrent,
            "current_concurrent": self._current,
        }


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on: tuple[type[Exception], ...] = field(
        default_factory=lambda: (RateLimitExceededError,)
    )


class RetryHandler:
    """Handles retries with exponential backoff.

    Example:
        >>> retry = RetryHandler(RetryConfig(max_retries=3))
        >>> result = await retry.execute(make_request)
    """

    def __init__(self, config: RetryConfig | None = None) -> None:
        """Initialize retry handler.

        Args:
            config: Retry configuration
        """
        self._config = config or RetryConfig()

        # Stats
        self._total_attempts = 0
        self._total_retries = 0
        self._total_failures = 0

        logger.debug(
            "retry_handler_initialized",
            max_retries=self._config.max_retries,
        )

    @property
    def config(self) -> RetryConfig:
        """Get retry configuration."""
        return self._config

    @property
    def stats(self) -> dict:
        """Get current statistics."""
        return {
            "total_attempts": self._total_attempts,
            "total_retries": self._total_retries,
            "total_failures": self._total_failures,
        }

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        import random

        delay = self._config.initial_delay * (
            self._config.exponential_base ** attempt
        )
        delay = min(delay, self._config.max_delay)

        if self._config.jitter:
            delay *= 0.5 + random.random()

        return delay

    async def execute(
        self,
        func: Callable[..., T],
        *args,
        **kwargs,
    ) -> T:
        """Execute function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            Exception: If all retries exhausted
        """
        last_exception: Exception | None = None

        for attempt in range(self._config.max_retries + 1):
            self._total_attempts += 1

            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except self._config.retry_on as e:
                last_exception = e
                self._total_retries += 1

                if attempt < self._config.max_retries:
                    delay = self._calculate_delay(attempt)

                    # Use retry_after from RateLimitExceededError if available
                    if isinstance(e, RateLimitExceededError):
                        delay = max(delay, e.retry_after)

                    logger.warning(
                        "retry_attempt",
                        attempt=attempt + 1,
                        max_retries=self._config.max_retries,
                        delay=f"{delay:.2f}s",
                        error=str(e),
                    )

                    await asyncio.sleep(delay)
                else:
                    self._total_failures += 1
                    raise

            except Exception:
                self._total_failures += 1
                raise

        # Should not reach here
        if last_exception:
            raise last_exception
        raise RuntimeError("Retry logic error")


class CombinedLimiter:
    """Combines rate limiting and concurrency control.

    Example:
        >>> limiter = CombinedLimiter(rate=60, max_concurrent=10)
        >>> async with limiter.acquire():
        ...     await make_request()
    """

    def __init__(
        self,
        rate: float | None = None,
        max_concurrent: int | None = None,
        *,
        burst: int | None = None,
        wait_timeout: float | None = None,
    ) -> None:
        """Initialize combined limiter.

        Args:
            rate: Requests per minute (None = no rate limit)
            max_concurrent: Maximum concurrent requests (None = no limit)
            burst: Rate limiter burst size
            wait_timeout: Maximum wait time
        """
        settings = get_settings()

        self._rate_limiter: TokenBucketRateLimiter | None = None
        self._concurrency_limiter: ConcurrencyLimiter | None = None

        rate = rate or settings.rate_limit_per_minute
        max_concurrent = max_concurrent or settings.max_concurrent_requests

        if rate and rate > 0:
            self._rate_limiter = TokenBucketRateLimiter(
                rate=rate,
                burst=burst,
                wait_timeout=wait_timeout,
            )

        if max_concurrent and max_concurrent > 0:
            self._concurrency_limiter = ConcurrencyLimiter(
                max_concurrent=max_concurrent,
                wait_timeout=wait_timeout,
            )

        logger.debug(
            "combined_limiter_initialized",
            rate=rate,
            max_concurrent=max_concurrent,
        )

    @property
    def rate_limiter(self) -> TokenBucketRateLimiter | None:
        """Get rate limiter."""
        return self._rate_limiter

    @property
    def concurrency_limiter(self) -> ConcurrencyLimiter | None:
        """Get concurrency limiter."""
        return self._concurrency_limiter

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[float]:
        """Acquire both rate limit token and concurrency slot.

        Yields:
            Rate limit wait time
        """
        wait_time = 0.0

        # Acquire rate limit first
        if self._rate_limiter:
            wait_time = await self._rate_limiter.acquire()

        # Then acquire concurrency slot
        if self._concurrency_limiter:
            async with self._concurrency_limiter.acquire():
                yield wait_time
        else:
            yield wait_time

    @property
    def stats(self) -> dict:
        """Get combined statistics."""
        stats = {}

        if self._rate_limiter:
            rl_stats = self._rate_limiter.stats
            stats["rate_limiter"] = {
                "total_requests": rl_stats.total_requests,
                "total_waits": rl_stats.total_waits,
                "average_wait_time": rl_stats.average_wait_time,
                "rejected_requests": rl_stats.rejected_requests,
            }

        if self._concurrency_limiter:
            stats["concurrency_limiter"] = self._concurrency_limiter.stats

        return stats


# Global limiter instance
_global_limiter: CombinedLimiter | None = None


def get_limiter() -> CombinedLimiter:
    """Get or create global combined limiter."""
    global _global_limiter
    if _global_limiter is None:
        _global_limiter = CombinedLimiter()
    return _global_limiter


def reset_limiter() -> None:
    """Reset global limiter."""
    global _global_limiter
    _global_limiter = None
