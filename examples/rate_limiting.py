#!/usr/bin/env python3
"""
Rate Limiting Example

Demonstrates configuring and using rate limiters to control request throughput
and prevent overloading Claude.
"""

import asyncio
import time

from ccflow import query, CLIAgentOptions  # noqa: F401 - shown in comments
from ccflow.rate_limiting import (
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    ConcurrencyLimiter,
    CombinedLimiter,
    get_limiter,
    RateLimitExceededError,
)


async def demo_token_bucket():
    """Token bucket allows bursts up to bucket size."""
    print("\n1. Token Bucket Rate Limiter")
    print("-" * 40)

    # 120 requests per minute (2/sec), burst of 3
    limiter = TokenBucketRateLimiter(rate=120.0, burst=3)

    async def make_request(i: int):
        wait_time = await limiter.acquire()
        print(f"  Request {i}: acquired at {time.time():.2f} (waited {wait_time:.2f}s)")
        await asyncio.sleep(0.1)  # Simulate work

    # First 3 should go immediately (burst), then rate-limited
    start = time.time()
    for i in range(5):
        await make_request(i)

    print(f"  Total time: {time.time() - start:.2f}s (expected ~1.5s)")


async def demo_sliding_window():
    """Sliding window provides smooth rate limiting."""
    print("\n2. Sliding Window Rate Limiter")
    print("-" * 40)

    # 5 requests per 2-second window
    limiter = SlidingWindowRateLimiter(rate=5, window=2.0)

    print(f"  Initial count: {limiter.current_count}")

    async def make_request(i: int):
        wait_time = await limiter.acquire()
        print(f"  Request {i}: acquired at {time.time():.2f} (waited {wait_time:.2f}s)")

    start = time.time()
    for i in range(6):
        await make_request(i)

    print(f"  Total time: {time.time() - start:.2f}s")
    print(f"  Final count: {limiter.current_count}")


async def demo_concurrency_limiter():
    """Concurrency limiter controls parallel execution."""
    print("\n3. Concurrency Limiter")
    print("-" * 40)

    # Max 3 concurrent requests
    limiter = ConcurrencyLimiter(max_concurrent=3)

    async def make_request(i: int):
        async with limiter.acquire():
            print(f"  Request {i}: started (concurrent: {limiter._semaphore._value})")
            await asyncio.sleep(0.5)  # Simulate work
            print(f"  Request {i}: finished")

    # Run 5 requests - only 3 at a time
    start = time.time()
    await asyncio.gather(*[make_request(i) for i in range(5)])
    print(f"  Total time: {time.time() - start:.2f}s (expected ~1s)")


async def demo_combined_limiter():
    """Combined limiter applies both rate and concurrency limits."""
    print("\n4. Combined Limiter (Rate + Concurrency)")
    print("-" * 40)

    # 240 requests/minute (4/sec), max 2 concurrent
    limiter = CombinedLimiter(rate=240.0, max_concurrent=2)

    async def make_request(i: int):
        async with limiter.acquire():
            print(f"  Request {i}: acquired")
            await asyncio.sleep(0.3)

    start = time.time()
    await asyncio.gather(*[make_request(i) for i in range(4)])
    print(f"  Total time: {time.time() - start:.2f}s")
    print(f"  Stats: {limiter.stats}")


async def demo_with_ccflow():
    """Use rate limiting with actual ccflow queries."""
    print("\n5. Rate Limiting with ccflow Queries")
    print("-" * 40)

    # Configure a strict limiter for demo (120/min = 2/sec)
    limiter = CombinedLimiter(rate=120.0, max_concurrent=1)

    questions = [
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
    ]

    async def rate_limited_query(q: str):
        async with limiter.acquire():
            print(f"  Querying: {q[:20]}...")
            # In real usage, you'd call query() here
            # response = await query(q)
            await asyncio.sleep(0.2)  # Simulated
            return f"Answer to: {q}"

    start = time.time()
    results = await asyncio.gather(*[rate_limited_query(q) for q in questions])
    print(f"  Completed {len(results)} queries in {time.time() - start:.2f}s")


async def demo_rate_limit_exceeded():
    """Demonstrate handling rate limit exceeded errors."""
    print("\n6. Handling Rate Limit Exceeded")
    print("-" * 40)

    # Strict limiter with no wait (60/min = 1/sec)
    limiter = TokenBucketRateLimiter(rate=60.0, burst=1, wait_timeout=0.0)

    # First request succeeds
    await limiter.acquire()
    print("  First request: success")

    # Second request fails immediately (no waiting)
    try:
        await limiter.acquire()
        print("  Second request: success")
    except RateLimitExceededError as e:
        print(f"  Second request: rate limited (wait {e.retry_after:.2f}s)")


async def main():
    """Run all rate limiting demos."""
    print("=" * 60)
    print("Rate Limiting Examples")
    print("=" * 60)

    await demo_token_bucket()
    await demo_sliding_window()
    await demo_concurrency_limiter()
    await demo_combined_limiter()
    await demo_with_ccflow()
    await demo_rate_limit_exceeded()

    print("\n" + "=" * 60)
    print("All demos complete!")


if __name__ == "__main__":
    asyncio.run(main())
