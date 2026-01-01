"""Tests for Rate Limiting & Concurrency Control."""

import asyncio
import time

import pytest

from ccflow.rate_limiting import (
    CombinedLimiter,
    ConcurrencyLimiter,
    ConcurrencyLimitExceededError,
    RateLimiterStats,
    RateLimitExceededError,
    RetryConfig,
    RetryHandler,
    SlidingWindowRateLimiter,
    TokenBucketRateLimiter,
    get_limiter,
    reset_limiter,
)


class TestRateLimiterStats:
    """Tests for RateLimiterStats."""

    def test_initial_state(self):
        """Test initial stats are zero."""
        stats = RateLimiterStats()
        assert stats.total_requests == 0
        assert stats.total_waits == 0
        assert stats.total_wait_time == 0.0
        assert stats.rejected_requests == 0

    def test_average_wait_time_zero(self):
        """Test average wait time with no waits."""
        stats = RateLimiterStats()
        assert stats.average_wait_time == 0.0

    def test_average_wait_time(self):
        """Test average wait time calculation."""
        stats = RateLimiterStats(
            total_waits=4,
            total_wait_time=8.0,
        )
        assert stats.average_wait_time == 2.0


class TestTokenBucketRateLimiter:
    """Tests for TokenBucketRateLimiter."""

    def test_init_default_burst(self):
        """Test default burst calculation."""
        limiter = TokenBucketRateLimiter(rate=100)
        assert limiter.rate == 100
        assert limiter.burst == 10  # rate/10

    def test_init_custom_burst(self):
        """Test custom burst size."""
        limiter = TokenBucketRateLimiter(rate=60, burst=5)
        assert limiter.burst == 5

    def test_init_minimum_burst(self):
        """Test minimum burst of 1."""
        limiter = TokenBucketRateLimiter(rate=5)
        assert limiter.burst == 1

    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        """Test immediate acquisition when tokens available."""
        limiter = TokenBucketRateLimiter(rate=60, burst=10)
        wait_time = await limiter.acquire()
        assert wait_time == 0.0
        assert limiter.stats.total_requests == 1
        assert limiter.stats.total_waits == 0

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        """Test multiple acquisitions within burst."""
        limiter = TokenBucketRateLimiter(rate=60, burst=5)
        for _i in range(5):
            wait_time = await limiter.acquire()
            assert wait_time == 0.0

        assert limiter.stats.total_requests == 5

    @pytest.mark.asyncio
    async def test_acquire_wait(self):
        """Test waiting when tokens exhausted."""
        limiter = TokenBucketRateLimiter(rate=600, burst=1)  # 10/sec = 0.1s per token

        # First acquisition immediate
        await limiter.acquire()

        # Second should wait
        start = time.monotonic()
        wait_time = await limiter.acquire()
        elapsed = time.monotonic() - start

        assert wait_time > 0
        assert elapsed >= wait_time * 0.9  # Allow small timing variance
        assert limiter.stats.total_waits == 1

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test timeout when wait exceeds limit."""
        limiter = TokenBucketRateLimiter(rate=60, burst=1, wait_timeout=0.05)

        # Exhaust burst
        await limiter.acquire()

        # Should timeout
        with pytest.raises(RateLimitExceededError) as exc_info:
            await limiter.acquire()

        assert exc_info.value.retry_after > 0
        assert limiter.stats.rejected_requests == 1

    def test_try_acquire_success(self):
        """Test non-blocking acquisition success."""
        limiter = TokenBucketRateLimiter(rate=60, burst=5)
        assert limiter.try_acquire() is True
        assert limiter.stats.total_requests == 1

    def test_try_acquire_failure(self):
        """Test non-blocking acquisition failure."""
        limiter = TokenBucketRateLimiter(rate=60, burst=1)
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False
        assert limiter.stats.rejected_requests == 1

    @pytest.mark.asyncio
    async def test_limit_context_manager(self):
        """Test context manager usage."""
        limiter = TokenBucketRateLimiter(rate=60, burst=5)

        async with limiter.limit() as wait_time:
            assert wait_time == 0.0

        assert limiter.stats.total_requests == 1

    def test_reset(self):
        """Test limiter reset."""
        limiter = TokenBucketRateLimiter(rate=60, burst=5)
        limiter.try_acquire()
        limiter.try_acquire()

        limiter.reset()

        assert limiter.available_tokens == 5.0
        assert limiter.stats.total_requests == 0

    def test_available_tokens(self):
        """Test available tokens property."""
        limiter = TokenBucketRateLimiter(rate=60, burst=5)
        assert limiter.available_tokens == pytest.approx(5.0, abs=0.01)

        limiter.try_acquire()
        assert limiter.available_tokens == pytest.approx(4.0, abs=0.01)


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    def test_init(self):
        """Test initialization."""
        limiter = SlidingWindowRateLimiter(rate=60, window=60.0)
        assert limiter.rate == 60
        assert limiter.window == 60.0

    @pytest.mark.asyncio
    async def test_acquire_immediate(self):
        """Test immediate acquisition."""
        limiter = SlidingWindowRateLimiter(rate=10, window=1.0)
        wait_time = await limiter.acquire()
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_within_limit(self):
        """Test acquisitions within limit."""
        limiter = SlidingWindowRateLimiter(rate=5, window=1.0)
        for _ in range(5):
            wait_time = await limiter.acquire()
            assert wait_time == 0.0

        assert limiter.current_count == 5

    @pytest.mark.asyncio
    async def test_acquire_wait(self):
        """Test waiting when window full."""
        limiter = SlidingWindowRateLimiter(rate=2, window=0.1)

        # Fill window
        await limiter.acquire()
        await limiter.acquire()

        # Should wait for window to expire
        start = time.monotonic()
        wait_time = await limiter.acquire()
        elapsed = time.monotonic() - start

        assert wait_time > 0 or elapsed >= 0.05  # Either waited or took time

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test timeout when window full."""
        limiter = SlidingWindowRateLimiter(rate=1, window=60.0, wait_timeout=0.01)

        await limiter.acquire()

        with pytest.raises(RateLimitExceededError):
            await limiter.acquire()

    def test_try_acquire_success(self):
        """Test non-blocking success."""
        limiter = SlidingWindowRateLimiter(rate=5, window=1.0)
        assert limiter.try_acquire() is True

    def test_try_acquire_failure(self):
        """Test non-blocking failure."""
        limiter = SlidingWindowRateLimiter(rate=1, window=60.0)
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

    @pytest.mark.asyncio
    async def test_limit_context_manager(self):
        """Test context manager."""
        limiter = SlidingWindowRateLimiter(rate=10, window=1.0)

        async with limiter.limit() as wait_time:
            assert wait_time == 0.0

    def test_reset(self):
        """Test reset."""
        limiter = SlidingWindowRateLimiter(rate=5, window=1.0)
        limiter.try_acquire()
        limiter.try_acquire()

        limiter.reset()

        assert limiter.current_count == 0
        assert limiter.stats.total_requests == 0

    def test_current_count(self):
        """Test current count property."""
        limiter = SlidingWindowRateLimiter(rate=10, window=60.0)
        assert limiter.current_count == 0

        limiter.try_acquire()
        assert limiter.current_count == 1


class TestConcurrencyLimiter:
    """Tests for ConcurrencyLimiter."""

    def test_init(self):
        """Test initialization."""
        limiter = ConcurrencyLimiter(max_concurrent=5)
        assert limiter.max_concurrent == 5
        assert limiter.current == 0
        assert limiter.available == 5

    @pytest.mark.asyncio
    async def test_acquire_release(self):
        """Test basic acquire/release."""
        limiter = ConcurrencyLimiter(max_concurrent=3)

        async with limiter.acquire():
            assert limiter.current == 1
            assert limiter.available == 2

        assert limiter.current == 0
        assert limiter.available == 3

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        """Test multiple concurrent acquisitions."""
        limiter = ConcurrencyLimiter(max_concurrent=3)

        async def worker(limiter, results, index):
            async with limiter.acquire():
                results[index] = limiter.current
                await asyncio.sleep(0.05)

        results = {}
        await asyncio.gather(
            worker(limiter, results, 0),
            worker(limiter, results, 1),
            worker(limiter, results, 2),
        )

        # All should have been concurrent
        assert max(results.values()) <= 3

    @pytest.mark.asyncio
    async def test_acquire_timeout(self):
        """Test timeout when limit reached."""
        limiter = ConcurrencyLimiter(max_concurrent=1, wait_timeout=0.05)

        async with limiter.acquire():
            with pytest.raises(ConcurrencyLimitExceededError) as exc_info:
                async with limiter.acquire():
                    pass

            assert exc_info.value.current == 1
            assert exc_info.value.limit == 1

    @pytest.mark.asyncio
    async def test_waiting_queue(self):
        """Test requests queue when at limit."""
        limiter = ConcurrencyLimiter(max_concurrent=1)
        results = []

        async def worker(limiter, index):
            async with limiter.acquire():
                results.append(f"start_{index}")
                await asyncio.sleep(0.02)
                results.append(f"end_{index}")

        await asyncio.gather(
            worker(limiter, 0),
            worker(limiter, 1),
        )

        # Should have sequential execution
        assert results == ["start_0", "end_0", "start_1", "end_1"]

    def test_try_acquire_success(self):
        """Test non-blocking success."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        assert limiter.try_acquire() is True
        assert limiter.current == 1

    def test_try_acquire_failure(self):
        """Test non-blocking failure."""
        limiter = ConcurrencyLimiter(max_concurrent=1)
        assert limiter.try_acquire() is True
        assert limiter.try_acquire() is False

    def test_release(self):
        """Test manual release."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        limiter.try_acquire()
        assert limiter.current == 1

        limiter.release()
        assert limiter.current == 0

    def test_stats(self):
        """Test statistics."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        limiter.try_acquire()
        limiter.try_acquire()
        limiter.try_acquire()  # Should fail

        stats = limiter.stats
        assert stats["total_acquired"] == 2
        assert stats["total_rejected"] >= 1
        assert stats["peak_concurrent"] == 2


class TestRetryConfig:
    """Tests for RetryConfig."""

    def test_defaults(self):
        """Test default values."""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert RateLimitExceededError in config.retry_on

    def test_custom_values(self):
        """Test custom values."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            max_delay=30.0,
            jitter=False,
        )
        assert config.max_retries == 5
        assert config.initial_delay == 0.5


class TestRetryHandler:
    """Tests for RetryHandler."""

    def test_init_default(self):
        """Test default initialization."""
        handler = RetryHandler()
        assert handler.config.max_retries == 3

    def test_init_custom(self):
        """Test custom config."""
        config = RetryConfig(max_retries=5)
        handler = RetryHandler(config)
        assert handler.config.max_retries == 5

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        handler = RetryHandler()

        async def success():
            return "result"

        result = await handler.execute(success)
        assert result == "result"
        assert handler.stats["total_attempts"] == 1
        assert handler.stats["total_retries"] == 0

    @pytest.mark.asyncio
    async def test_execute_retry_then_success(self):
        """Test retry then success."""
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.01,
            jitter=False,
        )
        handler = RetryHandler(config)

        attempts = [0]

        async def fail_then_succeed():
            attempts[0] += 1
            if attempts[0] < 3:
                raise RateLimitExceededError("Rate limited", retry_after=0.01)
            return "success"

        result = await handler.execute(fail_then_succeed)
        assert result == "success"
        assert attempts[0] == 3
        assert handler.stats["total_retries"] == 2

    @pytest.mark.asyncio
    async def test_execute_max_retries_exceeded(self):
        """Test max retries exceeded."""
        config = RetryConfig(
            max_retries=2,
            initial_delay=0.01,
            jitter=False,
        )
        handler = RetryHandler(config)

        async def always_fail():
            raise RateLimitExceededError("Rate limited", retry_after=0.01)

        with pytest.raises(RateLimitExceededError):
            await handler.execute(always_fail)

        assert handler.stats["total_attempts"] == 3  # 1 + 2 retries
        assert handler.stats["total_failures"] == 1

    @pytest.mark.asyncio
    async def test_execute_non_retryable_error(self):
        """Test non-retryable errors are not retried."""
        handler = RetryHandler()

        async def raise_value_error():
            raise ValueError("Not retryable")

        with pytest.raises(ValueError):
            await handler.execute(raise_value_error)

        assert handler.stats["total_attempts"] == 1
        assert handler.stats["total_retries"] == 0

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """Test with sync function."""
        handler = RetryHandler()

        def sync_func():
            return "sync_result"

        result = await handler.execute(sync_func)
        assert result == "sync_result"

    def test_calculate_delay_exponential(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=False,
        )
        handler = RetryHandler(config)

        assert handler._calculate_delay(0) == 1.0
        assert handler._calculate_delay(1) == 2.0
        assert handler._calculate_delay(2) == 4.0
        assert handler._calculate_delay(3) == 8.0

    def test_calculate_delay_max_cap(self):
        """Test delay capped at max."""
        config = RetryConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False,
        )
        handler = RetryHandler(config)

        # 2^10 = 1024, but should be capped at 10
        assert handler._calculate_delay(10) == 10.0


class TestCombinedLimiter:
    """Tests for CombinedLimiter."""

    def test_init_no_limits(self):
        """Test with no limits configured."""
        reset_limiter()
        from unittest.mock import patch

        with patch("ccflow.rate_limiting.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_per_minute = 0
            mock_settings.return_value.max_concurrent_requests = 0

            limiter = CombinedLimiter(rate=0, max_concurrent=0)
            assert limiter.rate_limiter is None
            assert limiter.concurrency_limiter is None

    def test_init_rate_only(self):
        """Test with rate limit only."""
        from unittest.mock import patch

        with patch("ccflow.rate_limiting.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_per_minute = 0
            mock_settings.return_value.max_concurrent_requests = 0

            limiter = CombinedLimiter(rate=60, max_concurrent=0)
            assert limiter.rate_limiter is not None
            assert limiter.concurrency_limiter is None

    def test_init_concurrency_only(self):
        """Test with concurrency limit only."""
        from unittest.mock import patch

        with patch("ccflow.rate_limiting.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_per_minute = 0
            mock_settings.return_value.max_concurrent_requests = 0

            limiter = CombinedLimiter(rate=0, max_concurrent=5)
            assert limiter.rate_limiter is None
            assert limiter.concurrency_limiter is not None

    def test_init_both(self):
        """Test with both limits."""
        limiter = CombinedLimiter(rate=60, max_concurrent=5)
        assert limiter.rate_limiter is not None
        assert limiter.concurrency_limiter is not None

    @pytest.mark.asyncio
    async def test_acquire_no_limits(self):
        """Test acquire with no limits."""
        limiter = CombinedLimiter(rate=0, max_concurrent=0)

        async with limiter.acquire() as wait_time:
            assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_rate_only(self):
        """Test acquire with rate limit only."""
        limiter = CombinedLimiter(rate=600, max_concurrent=0, burst=5)

        async with limiter.acquire() as wait_time:
            assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_concurrency_only(self):
        """Test acquire with concurrency only."""
        limiter = CombinedLimiter(rate=0, max_concurrent=5)

        async with limiter.acquire() as wait_time:
            assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_both(self):
        """Test acquire with both limits."""
        limiter = CombinedLimiter(rate=600, max_concurrent=5, burst=5)

        async with limiter.acquire() as wait_time:
            assert isinstance(wait_time, float)

    @pytest.mark.asyncio
    async def test_concurrency_respected(self):
        """Test concurrency limit is respected."""
        limiter = CombinedLimiter(rate=6000, max_concurrent=2, burst=100)
        active_count = [0]
        max_active = [0]

        async def worker():
            async with limiter.acquire():
                active_count[0] += 1
                max_active[0] = max(max_active[0], active_count[0])
                await asyncio.sleep(0.02)
                active_count[0] -= 1

        await asyncio.gather(*[worker() for _ in range(5)])
        assert max_active[0] <= 2

    def test_stats_empty(self):
        """Test stats when no limits."""
        from unittest.mock import patch

        with patch("ccflow.rate_limiting.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_per_minute = 0
            mock_settings.return_value.max_concurrent_requests = 0

            limiter = CombinedLimiter(rate=0, max_concurrent=0)
            stats = limiter.stats
            assert stats == {}

    def test_stats_with_limits(self):
        """Test stats with limits."""
        limiter = CombinedLimiter(rate=60, max_concurrent=5)
        stats = limiter.stats

        assert "rate_limiter" in stats
        assert "concurrency_limiter" in stats


class TestGlobalLimiter:
    """Tests for global limiter functions."""

    def test_get_limiter(self):
        """Test getting global limiter."""
        reset_limiter()
        limiter = get_limiter()
        assert isinstance(limiter, CombinedLimiter)

    def test_get_limiter_singleton(self):
        """Test limiter is singleton."""
        reset_limiter()
        limiter1 = get_limiter()
        limiter2 = get_limiter()
        assert limiter1 is limiter2

    def test_reset_limiter(self):
        """Test resetting global limiter."""
        reset_limiter()
        limiter1 = get_limiter()
        reset_limiter()
        limiter2 = get_limiter()
        assert limiter1 is not limiter2


class TestExceptions:
    """Tests for rate limiting exceptions."""

    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError."""
        error = RateLimitExceededError("Rate limited", retry_after=5.0)
        assert str(error) == "Rate limited"
        assert error.retry_after == 5.0

    def test_concurrency_limit_exceeded_error(self):
        """Test ConcurrencyLimitExceededError."""
        error = ConcurrencyLimitExceededError(
            "Concurrency limit exceeded",
            current=10,
            limit=10,
        )
        assert str(error) == "Concurrency limit exceeded"
        assert error.current == 10
        assert error.limit == 10


class TestExecutorIntegration:
    """Tests for rate limiting integration with executor."""

    @pytest.mark.asyncio
    async def test_executor_with_limiter(self):
        """Test executor uses limiter."""
        from unittest.mock import patch

        from ccflow.executor import CLIExecutor
        from ccflow.rate_limiting import CombinedLimiter

        limiter = CombinedLimiter(rate=60, max_concurrent=5)

        with patch.object(CLIExecutor, "_find_cli", return_value="/usr/bin/claude"):
            executor = CLIExecutor(limiter=limiter)
            assert executor.limiter is limiter

    @pytest.mark.asyncio
    async def test_executor_uses_global_limiter(self):
        """Test executor uses global limiter."""
        from unittest.mock import patch

        from ccflow.executor import CLIExecutor

        reset_limiter()

        with patch.object(CLIExecutor, "_find_cli", return_value="/usr/bin/claude"):
            executor = CLIExecutor(use_global_limiter=True)
            assert executor.limiter is get_limiter()

    @pytest.mark.asyncio
    async def test_executor_no_limiter(self):
        """Test executor without limiter."""
        from unittest.mock import patch

        from ccflow.executor import CLIExecutor

        with patch.object(CLIExecutor, "_find_cli", return_value="/usr/bin/claude"):
            executor = CLIExecutor(use_global_limiter=False)
            assert executor.limiter is None
