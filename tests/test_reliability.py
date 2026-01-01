"""Tests for reliability module - circuit breaker, retry, health check, correlation IDs."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from ccflow.reliability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    HealthChecker,
    RetryConfig,
    RetryExhaustedError,
    calculate_delay,
    get_cli_circuit_breaker,
    get_correlation_id,
    get_health_checker,
    reset_cli_circuit_breaker,
    reset_health_checker,
    retry_with_backoff,
    set_correlation_id,
    with_retry,
)

# =============================================================================
# Correlation ID Tests
# =============================================================================


class TestCorrelationId:
    """Tests for correlation ID management."""

    def test_get_correlation_id_generates_if_empty(self):
        """Should generate new correlation ID if none exists."""
        # Reset to empty state
        from ccflow.reliability import correlation_id_var
        correlation_id_var.set("")

        cid = get_correlation_id()
        assert cid
        assert len(cid) == 8  # UUID prefix

    def test_set_correlation_id_custom(self):
        """Should set custom correlation ID."""
        custom_id = "test-123"
        result = set_correlation_id(custom_id)
        assert result == custom_id
        assert get_correlation_id() == custom_id

    def test_set_correlation_id_generates(self):
        """Should generate ID if None passed."""
        result = set_correlation_id(None)
        assert result
        assert len(result) == 8

    def test_correlation_id_persists(self):
        """Correlation ID should persist across calls."""
        set_correlation_id("persist-test")
        assert get_correlation_id() == "persist-test"
        assert get_correlation_id() == "persist-test"


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker pattern."""

    @pytest.fixture
    def breaker(self):
        """Create a circuit breaker with low thresholds for testing."""
        reset_cli_circuit_breaker()
        return CircuitBreaker(
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                reset_timeout=0.1,  # 100ms for fast tests
                half_open_max_calls=1,
            ),
            name="test",
        )

    @pytest.mark.asyncio
    async def test_initial_state_is_closed(self, breaker):
        """Circuit breaker should start in closed state."""
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_successful_calls_keep_circuit_closed(self, breaker):
        """Successful calls should keep circuit closed."""
        for _ in range(5):
            async with breaker.call():
                pass  # Success

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.total_calls == 5
        assert breaker.stats.total_failures == 0

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self, breaker):
        """Reaching failure threshold should open circuit."""
        for i in range(3):
            with pytest.raises(ValueError):
                async with breaker.call():
                    raise ValueError(f"Error {i}")

        assert breaker.state == CircuitState.OPEN
        assert breaker.stats.failure_count == 3

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, breaker):
        """Open circuit should reject calls immediately."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker.call():
                    raise ValueError("Error")

        # Should reject
        with pytest.raises(CircuitBreakerError) as exc_info:
            async with breaker.call():
                pass

        assert "open" in str(exc_info.value).lower()
        assert breaker.stats.total_rejections == 1

    @pytest.mark.asyncio
    async def test_circuit_transitions_to_half_open(self, breaker):
        """Circuit should transition to half-open after timeout."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker.call():
                    raise ValueError("Error")

        assert breaker.state == CircuitState.OPEN

        # Wait for reset timeout
        await asyncio.sleep(0.15)

        # Next call should be allowed (half-open)
        async with breaker.call():
            pass

        # Still half-open after 1 success (need 2 for this config)
        assert breaker.state == CircuitState.HALF_OPEN

        # Wait for another half-open slot
        await asyncio.sleep(0.15)

        # Second success should close it
        async with breaker.call():
            pass

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self, breaker):
        """Successful calls in half-open should close circuit."""
        # Modify to require 1 success
        breaker.config.success_threshold = 1

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker.call():
                    raise ValueError("Error")

        await asyncio.sleep(0.15)

        # Success in half-open
        async with breaker.call():
            pass

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self, breaker):
        """Failure in half-open should reopen circuit."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker.call():
                    raise ValueError("Error")

        await asyncio.sleep(0.15)

        # Failure in half-open
        with pytest.raises(ValueError):
            async with breaker.call():
                raise ValueError("Still failing")

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, breaker):
        """Reset should clear all state."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker.call():
                    raise ValueError("Error")

        assert breaker.state == CircuitState.OPEN

        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.stats.failure_count == 0

    @pytest.mark.asyncio
    async def test_protect_decorator(self, breaker):
        """Protect decorator should wrap function with circuit breaker."""
        call_count = 0

        @breaker.protect
        async def my_function():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await my_function()
        assert result == "success"
        assert call_count == 1

    def test_stats_tracking(self, breaker):
        """Stats should track calls correctly."""
        stats = breaker.stats
        assert stats.state == CircuitState.CLOSED
        assert stats.total_calls == 0
        assert stats.total_failures == 0
        assert stats.total_rejections == 0

    @pytest.mark.asyncio
    async def test_retry_after_on_open(self, breaker):
        """CircuitBreakerError should include retry_after."""
        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                async with breaker.call():
                    raise ValueError("Error")

        with pytest.raises(CircuitBreakerError) as exc_info:
            async with breaker.call():
                pass

        assert exc_info.value.retry_after >= 0


class TestGlobalCircuitBreaker:
    """Tests for global circuit breaker."""

    def setup_method(self):
        reset_cli_circuit_breaker()

    def test_get_cli_circuit_breaker_singleton(self):
        """Should return same instance."""
        breaker1 = get_cli_circuit_breaker()
        breaker2 = get_cli_circuit_breaker()
        assert breaker1 is breaker2

    def test_reset_cli_circuit_breaker(self):
        """Reset should create new instance."""
        breaker1 = get_cli_circuit_breaker()
        reset_cli_circuit_breaker()
        breaker2 = get_cli_circuit_breaker()
        assert breaker1 is not breaker2


# =============================================================================
# Retry Tests
# =============================================================================


class TestRetry:
    """Tests for retry with backoff."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self):
        """Should return immediately on success."""
        async def success_func():
            return "success"

        result = await retry_with_backoff(success_func)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_retry_on_retryable_error(self):
        """Should retry on retryable errors."""
        attempts = 0

        async def flaky_func():
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError("Connection failed")
            return "success"

        config = RetryConfig(
            max_retries=3,
            base_delay=0.01,
            retryable_errors=(ConnectionError,),
        )

        result = await retry_with_backoff(flaky_func, config=config)
        assert result == "success"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_exhaust_retries(self):
        """Should raise RetryExhaustedError when retries exhausted."""
        async def always_fail():
            raise ConnectionError("Always fails")

        config = RetryConfig(
            max_retries=2,
            base_delay=0.01,
            retryable_errors=(ConnectionError,),
        )

        with pytest.raises(RetryExhaustedError) as exc_info:
            await retry_with_backoff(always_fail, config=config)

        assert exc_info.value.attempts == 3  # 1 + 2 retries
        assert isinstance(exc_info.value.last_error, ConnectionError)

    @pytest.mark.asyncio
    async def test_non_retryable_error_not_retried(self):
        """Non-retryable errors should not be retried."""
        attempts = 0

        async def fail_with_value_error():
            nonlocal attempts
            attempts += 1
            raise ValueError("Not retryable")

        config = RetryConfig(
            max_retries=3,
            base_delay=0.01,
            retryable_errors=(ConnectionError,),
        )

        with pytest.raises(ValueError):
            await retry_with_backoff(fail_with_value_error, config=config)

        assert attempts == 1  # No retries

    @pytest.mark.asyncio
    async def test_with_retry_decorator(self):
        """with_retry decorator should work correctly."""
        attempts = 0

        @with_retry(RetryConfig(max_retries=2, base_delay=0.01))
        async def decorated_func():
            nonlocal attempts
            attempts += 1
            if attempts < 2:
                raise ConnectionError("Retry me")
            return "decorated success"

        result = await decorated_func()
        assert result == "decorated success"
        assert attempts == 2


class TestCalculateDelay:
    """Tests for delay calculation."""

    def test_exponential_backoff(self):
        """Delay should increase exponentially."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=100.0,
            exponential_base=2.0,
            jitter=0.0,
        )

        delays = [calculate_delay(i, config) for i in range(5)]
        assert delays == [1.0, 2.0, 4.0, 8.0, 16.0]

    def test_max_delay_cap(self):
        """Delay should be capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=2.0,
            jitter=0.0,
        )

        delay = calculate_delay(10, config)  # Would be 1024 without cap
        assert delay == 5.0

    def test_jitter_adds_randomness(self):
        """Jitter should add randomness to delay."""
        config = RetryConfig(
            base_delay=10.0,
            max_delay=100.0,
            exponential_base=1.0,  # No exponential growth
            jitter=0.5,  # 50% jitter
        )

        # Run multiple times to verify variance
        delays = [calculate_delay(0, config) for _ in range(100)]
        unique_delays = set(delays)

        # Should have variation due to jitter
        assert len(unique_delays) > 1
        # Should be within jitter range (10 ± 5)
        assert all(5 <= d <= 15 for d in delays)


# =============================================================================
# Health Checker Tests
# =============================================================================


class TestHealthChecker:
    """Tests for deep health checking."""

    @pytest.fixture
    def checker(self):
        reset_health_checker()
        return HealthChecker(cache_ttl=0.1, timeout=5.0)

    @pytest.mark.asyncio
    async def test_health_check_caching(self, checker):
        """Health check results should be cached."""
        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                # Mock successful execution
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"1.0.0", b""))
                mock_proc.wait = AsyncMock()
                mock_exec.return_value = mock_proc

                # First call
                result1 = await checker.check()

                # Second call should use cache
                result2 = await checker.check()

                # Should only have called subprocess once (for version)
                # The second call should use cache
                assert result1.last_check_time == result2.last_check_time

    @pytest.mark.asyncio
    async def test_force_bypasses_cache(self, checker):
        """Force=True should bypass cache."""
        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"1.0.0", b""))
                mock_proc.wait = AsyncMock()
                mock_exec.return_value = mock_proc

                result1 = await checker.check()
                result2 = await checker.check(force=True)

                # Force should have triggered new check
                assert result1.last_check_time <= result2.last_check_time

    @pytest.mark.asyncio
    async def test_cli_not_in_path(self, checker):
        """Should report unhealthy if CLI not in PATH."""
        with patch("shutil.which", return_value=None):
            result = await checker.check()

            assert not result.healthy
            assert not result.cli_available
            assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalidate_cache(self, checker):
        """invalidate_cache should clear cached result."""
        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("asyncio.create_subprocess_exec") as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"1.0.0", b""))
                mock_proc.wait = AsyncMock()
                mock_exec.return_value = mock_proc

                await checker.check()
                checker.invalidate_cache()

                assert checker._cached_status is None


class TestGlobalHealthChecker:
    """Tests for global health checker."""

    def setup_method(self):
        reset_health_checker()

    def test_get_health_checker_singleton(self):
        """Should return same instance."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()
        assert checker1 is checker2

    def test_reset_health_checker(self):
        """Reset should create new instance."""
        checker1 = get_health_checker()
        reset_health_checker()
        checker2 = get_health_checker()
        assert checker1 is not checker2


# =============================================================================
# Integration Tests
# =============================================================================


class TestReliabilityIntegration:
    """Integration tests combining reliability features."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_retry(self):
        """Circuit breaker should track retry failures."""
        breaker = CircuitBreaker(
            CircuitBreakerConfig(failure_threshold=5),
            name="integration-test",
        )

        async def flaky_operation():
            raise ConnectionError("Service down")

        config = RetryConfig(
            max_retries=2,
            base_delay=0.01,
            retryable_errors=(ConnectionError,),
        )

        # Each retry attempt counts as a failure for circuit breaker
        for _ in range(2):
            try:
                async with breaker.call():
                    await retry_with_backoff(flaky_operation, config=config)
            except RetryExhaustedError:
                pass

        # Should have recorded failures
        assert breaker.stats.total_failures == 2

    @pytest.mark.asyncio
    async def test_correlation_id_in_retry_loop(self):
        """Correlation ID should persist across retries."""
        set_correlation_id("retry-correlation-test")
        collected_ids = []

        async def collect_id():
            collected_ids.append(get_correlation_id())
            if len(collected_ids) < 3:
                raise ConnectionError("Retry")
            return "done"

        config = RetryConfig(
            max_retries=3,
            base_delay=0.01,
            retryable_errors=(ConnectionError,),
        )

        await retry_with_backoff(collect_id, config=config)

        # All attempts should have same correlation ID
        assert all(cid == "retry-correlation-test" for cid in collected_ids)
