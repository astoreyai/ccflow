#!/usr/bin/env python3
"""
Reliability Patterns Example

Demonstrates circuit breaker, retry with backoff, and health checking
for robust Claude CLI interactions.
"""

import asyncio
import random

from ccflow.reliability import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    RetryConfig,
    RetryExhaustedError,
    retry_with_backoff,
    with_retry,
    HealthChecker,
    get_cli_circuit_breaker,
    set_correlation_id,
)


async def demo_circuit_breaker():
    """Circuit breaker prevents cascading failures."""
    print("\n1. Circuit Breaker")
    print("-" * 40)

    # Configure circuit breaker
    config = CircuitBreakerConfig(
        failure_threshold=3,      # Open after 3 failures
        success_threshold=2,      # Close after 2 successes
        reset_timeout=2.0,        # Try again after 2 seconds
        half_open_max_calls=1,    # 1 test call in half-open
    )

    breaker = CircuitBreaker(config, name="demo")
    print(f"  Initial state: {breaker.state}")

    # Simulate failing calls
    async def failing_call():
        async with breaker.call():
            raise RuntimeError("Simulated failure")

    # Trigger failures to open circuit
    for i in range(4):
        try:
            await failing_call()
        except RuntimeError:
            print(f"  Call {i+1} failed, state: {breaker.state}")
        except CircuitBreakerError as e:
            print(f"  Call {i+1} rejected (circuit open)")

    print(f"\n  Circuit is now: {breaker.state}")
    print("  Waiting for reset timeout...")
    await asyncio.sleep(2.5)

    print(f"  After timeout: {breaker.state}")

    # Successful call in half-open closes circuit
    async def successful_call():
        async with breaker.call():
            return "success"

    try:
        result = await successful_call()
        print(f"  Test call succeeded: {result}")
        result = await successful_call()
        print(f"  Second call succeeded, state: {breaker.state}")
    except CircuitBreakerError:
        print("  Still rejected")

    print(f"\n  Final stats: {breaker.stats}")


async def demo_retry_with_backoff():
    """Retry with exponential backoff for transient failures."""
    print("\n2. Retry with Exponential Backoff")
    print("-" * 40)

    attempt_count = 0

    async def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        print(f"  Attempt {attempt_count}...")

        if attempt_count < 3:
            raise ConnectionError("Transient failure")
        return "Success!"

    config = RetryConfig(
        max_retries=4,
        base_delay=0.5,
        max_delay=5.0,
        exponential_base=2.0,      # 0.5s, 1s, 2s, 4s
        jitter=0.1,                # Add 10% randomness
        retryable_errors=(ConnectionError,),
    )

    try:
        result = await retry_with_backoff(flaky_operation, config=config)
        print(f"  Final result: {result}")
    except RetryExhaustedError as e:
        print(f"  All retries exhausted: {e}")


async def demo_retry_decorator():
    """Using @with_retry decorator."""
    print("\n3. @with_retry Decorator")
    print("-" * 40)

    call_count = 0

    @with_retry(RetryConfig(max_retries=2, base_delay=0.1, retryable_errors=(ValueError,)))
    async def decorated_operation():
        nonlocal call_count
        call_count += 1
        print(f"  Decorated call #{call_count}")

        if call_count < 2:
            raise ValueError("Simulated error")
        return "Decorated success!"

    try:
        result = await decorated_operation()
        print(f"  Result: {result}")
    except RetryExhaustedError:
        print("  All retries failed")


async def demo_circuit_breaker_decorator():
    """Using circuit breaker as decorator."""
    print("\n4. Circuit Breaker Decorator")
    print("-" * 40)

    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=2),
        name="decorator_demo",
    )

    @breaker.protect
    async def protected_operation(should_fail: bool = False):
        if should_fail:
            raise RuntimeError("Deliberate failure")
        return "Protected success!"

    # Successful calls
    result = await protected_operation(should_fail=False)
    print(f"  First call: {result}")

    # Trigger circuit opening
    for i in range(3):
        try:
            await protected_operation(should_fail=True)
        except RuntimeError:
            print(f"  Failed call {i+1}")
        except CircuitBreakerError:
            print(f"  Rejected call {i+1} (circuit open)")


async def demo_health_checker():
    """Health checking for CLI availability."""
    print("\n5. Health Checker")
    print("-" * 40)

    checker = HealthChecker()

    print("  Checking CLI health...")
    status = await checker.check()

    print(f"  Healthy: {status.healthy}")
    print(f"  CLI available: {status.cli_available}")
    print(f"  Latency: {status.latency_ms:.1f}ms")

    if not status.healthy:
        print(f"  Error: {status.error}")


async def demo_correlation_ids():
    """Correlation IDs for request tracing."""
    print("\n6. Correlation IDs")
    print("-" * 40)

    # Set correlation ID for a request flow
    cid = set_correlation_id("req-12345")
    print(f"  Set correlation ID: {cid}")

    # Auto-generate if not provided
    cid2 = set_correlation_id()
    print(f"  Auto-generated: {cid2}")

    # Logs will include correlation ID
    print("  (Correlation IDs appear in structured logs)")


async def demo_combined_patterns():
    """Combining circuit breaker with retry."""
    print("\n7. Combined: Circuit Breaker + Retry")
    print("-" * 40)

    breaker = CircuitBreaker(
        CircuitBreakerConfig(failure_threshold=5),
        name="combined",
    )

    retry_config = RetryConfig(max_retries=2, base_delay=0.1)
    call_count = 0

    async def operation():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError(f"Attempt {call_count} failed")
        return f"Success on attempt {call_count}"

    async def combined_call():
        async with breaker.call():
            return await retry_with_backoff(operation, config=retry_config)

    try:
        result = await combined_call()
        print(f"  Result: {result}")
        print(f"  Total attempts: {call_count}")
        print(f"  Circuit state: {breaker.state}")
    except (RetryExhaustedError, CircuitBreakerError) as e:
        print(f"  Failed: {e}")


async def main():
    """Run all reliability demos."""
    print("=" * 60)
    print("Reliability Patterns Examples")
    print("=" * 60)

    await demo_circuit_breaker()
    await demo_retry_with_backoff()
    await demo_retry_decorator()
    await demo_circuit_breaker_decorator()
    await demo_health_checker()
    await demo_correlation_ids()
    await demo_combined_patterns()

    print("\n" + "=" * 60)
    print("All reliability demos complete!")


if __name__ == "__main__":
    asyncio.run(main())
