"""RetryExecutor — execute callables with exponential backoff retry.

Supports both sync and async callables.  Retry delays follow an
exponential backoff pattern capped at ``max_delay_ms``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.reliability.errors import ReliabilityError
from app.core.reliability.models import RetryPolicy, RetryResult


class RetryExecutor:
    """Executes a callable with exponential backoff retry.

    Usage::

        executor = RetryExecutor()

        # Retry a sync callable
        result = await executor.execute(maybe_flaky_fn, arg1, arg2)

        # Retry an async callable
        result = await executor.execute(async_fn, key="value")

        # Custom policy
        policy = RetryPolicy(max_attempts=5, initial_delay_ms=50.0)
        result = await executor.execute(fn, retry_policy=policy)
    """

    def __init__(
        self,
        policy: RetryPolicy | None = None,
    ) -> None:
        self._policy = policy or RetryPolicy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def policy(self) -> RetryPolicy:
        """Return the executor's default retry policy."""
        return self._policy

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    async def execute(
        self,
        fn: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *args: Any,
        retry_policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute a callable with retry on failure.

        The *fn* can be sync or async.  If a custom
        *retry_policy* is provided it is used instead of the
        executor's default.

        Args:
            fn: The function or callable to execute.
            *args: Positional arguments forwarded to *fn*.
            retry_policy: Optional override policy for this execution.
            **kwargs: Keyword arguments forwarded to *fn*.

        Returns:
            A ``RetryResult`` with attempt count and timing.

        Raises:
            ReliabilityError: If *fn* is not callable.
        """
        import builtins

        if not builtins.callable(fn):
            raise ReliabilityError(
                "RetryExecutor requires a callable",
                details={"received_type": type(fn).__name__},
            )

        policy = retry_policy or self._policy
        policy.validate()

        start = time.perf_counter()
        last_exception: Exception | None = None
        last_error_type: str = ""
        attempts_made = 0

        for attempt in range(1, policy.max_attempts + 1):
            attempts_made = attempt

            try:
                val = fn(*args, **kwargs)
                if isinstance(val, Awaitable):
                    await val
            except Exception as exc:
                last_exception = exc
                last_error_type = type(exc).__name__

                # Check if this exception type should trigger a retry
                if policy.retry_exceptions and not isinstance(exc, policy.retry_exceptions):
                    duration = (time.perf_counter() - start) * 1000.0
                    return RetryResult(
                        attempts=attempt,
                        success=False,
                        duration_ms=round(duration, 2),
                        metadata={
                            "last_error": str(exc),
                            "last_error_type": last_error_type,
                            "duration": round(duration, 2),
                        },
                    )

                # Last attempt — no sleep, just fail
                if attempt == policy.max_attempts:
                    duration = (time.perf_counter() - start) * 1000.0
                    return RetryResult(
                        attempts=attempt,
                        success=False,
                        duration_ms=round(duration, 2),
                        metadata={
                            "last_error": str(exc),
                            "last_error_type": last_error_type,
                            "duration": round(duration, 2),
                        },
                    )

                # Calculate backoff delay
                delay_ms = policy.initial_delay_ms * (policy.backoff_multiplier ** (attempt - 1))
                delay_ms = min(delay_ms, policy.max_delay_ms)
                await asyncio.sleep(delay_ms / 1000.0)
                continue

            # Success
            duration = (time.perf_counter() - start) * 1000.0
            return RetryResult(
                attempts=attempt,
                success=True,
                duration_ms=round(duration, 2),
                metadata={
                    "success_on_attempt": attempt,
                    "duration": round(duration, 2),
                },
            )

        # Should not reach here, but safety net
        duration = (time.perf_counter() - start) * 1000.0
        return RetryResult(
            attempts=attempts_made,
            success=False,
            duration_ms=round(duration, 2),
            metadata={
                "last_error": str(last_exception) if last_exception else "unknown",
                "last_error_type": last_error_type if last_exception else "unknown",
                "duration": round(duration, 2),
            },
        )
