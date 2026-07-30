#!/usr/bin/env python3
"""
Retry and reliability demo.

Demonstrates:
  - RetryExecutor with exponential backoff
  - Exception filtering (retryable vs non-retryable)
  - Custom retry policies
  - Structured logging with AtlasLogger
"""

import asyncio
import random
from typing import Any

from app.core.log import AtlasLogger
from app.core.reliability import RetryExecutor, RetryPolicy, RetryResult

log = AtlasLogger("retry-demo", level="INFO")


# ---------------------------------------------------------------------------
# Simulated flaky operations
# ---------------------------------------------------------------------------

attempt_counter: dict[str, int] = {}


async def flaky_network_call(url: str) -> str:
    """Simulates a network call that fails 60% of the time."""
    attempt_counter[url] = attempt_counter.get(url, 0) + 1
    if random.random() < 0.6:
        raise ConnectionError(f"Connection refused: {url}")
    return f"Response from {url}"


def flaky_db_query(query: str) -> list[str]:
    """Simulates a database query that fails 40% of the time."""
    attempt_counter[query] = attempt_counter.get(query, 0) + 1
    if random.random() < 0.4:
        raise TimeoutError(f"Query timed out: {query}")
    return [f"result_{query}"]


def always_fails() -> None:
    """A function that always fails with a non-retryable error."""
    raise ValueError("Invalid input — this should NOT be retried")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def demonstrate_policy(
    name: str,
    policy: RetryPolicy,
    fn: Any,
    *args: Any,
) -> None:
    """Run a retry demo with a specific policy."""
    attempt_counter.clear()
    executor = RetryExecutor(policy)

    log.info(f"Running: {name}", policy=str(policy))
    result = await executor.execute(fn, *args)

    log.info(
        f"Result: {'SUCCESS' if result.success else 'FAILURE'}",
        attempts=result.attempts,
        duration_ms=result.duration_ms,
        success=result.success,
    )

    # Check how many times the function was actually called
    total_calls = sum(attempt_counter.values())
    print(f"  Function called {total_calls} time(s)")
    return result


async def main() -> None:
    print("=" * 60)
    print("Retry & Reliability Demo")
    print("=" * 60)

    # Policy 1: Aggressive retry for network calls
    network_policy = RetryPolicy(
        max_attempts=5,
        initial_delay_ms=10.0,
        backoff_multiplier=2.0,
        max_delay_ms=1000.0,
        retry_exceptions=(ConnectionError,),
    )

    # Policy 2: Conservative retry for DB
    db_policy = RetryPolicy(
        max_attempts=3,
        initial_delay_ms=5.0,
        backoff_multiplier=1.0,  # constant delay
        max_delay_ms=100.0,
        retry_exceptions=(TimeoutError,),
    )

    # Policy 3: Single attempt (no retry)
    no_retry_policy = RetryPolicy(
        max_attempts=1,
        initial_delay_ms=0,
    )

    print("\n--- Network call (retry on ConnectionError) ---")
    random.seed(42)
    await demonstrate_policy(
        "flaky_network_call", network_policy,
        flaky_network_call, "https://api.example.com/data",
    )

    print("\n--- DB query (retry on TimeoutError) ---")
    random.seed(42)
    await demonstrate_policy(
        "flaky_db_query", db_policy,
        flaky_db_query, "SELECT * FROM users",
    )

    print("\n--- Non-retryable error (fails immediately) ---")
    result = await demonstrate_policy(
        "always_fails", network_policy,
        always_fails,
    )
    print(f"  Last error: {result.metadata.get('last_error', 'none')}")

    print("\n--- Single attempt (no retry) ---")
    await demonstrate_policy(
        "no_retry", no_retry_policy,
        flaky_network_call, "https://api.example.com/ping",
    )

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
