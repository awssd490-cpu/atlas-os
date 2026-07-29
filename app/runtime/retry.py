"""Retry and recovery framework for Atlas runtime.

The ``RetryManager`` encapsulates retry decisions behind a unified API.
Execution code never decides whether to retry — it simply calls
``retry_manager.execute()`` which handles the full lifecycle.

Providers, tools, and other runtime components remain unaware of retries.
"""

from __future__ import annotations

import asyncio
import enum
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Retry strategies
# ---------------------------------------------------------------------------


class RetryStrategy(str, enum.Enum):
    """Strategy for calculating retry delays."""

    NO_RETRY = "no_retry"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    EXPONENTIAL_WITH_JITTER = "exponential_with_jitter"


# ---------------------------------------------------------------------------
# Retry reasons
# ---------------------------------------------------------------------------


class RetryReason(str, enum.Enum):
    """Classification for why a retry is occurring."""

    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ErrorClassification:
    """Result of classifying an exception.

    Attributes:
        retryable: Whether the error can be retried.
        reason: The classification reason.
        permanent: Whether the error is permanent (no retry possible).
        timeout: Whether the error is a timeout.
    """

    retryable: bool = False
    reason: RetryReason = RetryReason.UNKNOWN
    permanent: bool = False
    timeout: bool = False


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behaviour.

    Attributes:
        strategy: The retry delay strategy to use.
        max_attempts: Maximum number of attempts (including first).
            Default 3 (first attempt + 2 retries).
        initial_delay: Initial delay in seconds before retrying.
            Default 0.5.
        maximum_delay: Maximum delay in seconds between retries.
            Default 8.0.
        backoff_multiplier: Multiplier for exponential backoff.
            Default 2.0.
        jitter: Whether to add random jitter to delay.  Default True.
        retryable_exceptions: Tuple of exception types that are retryable.
            Default includes common transient failures.
        retry_on_timeout: Whether to retry on timeout.  Default True.
        retry_on_cancelled: Whether to retry on cancellation.  Default False.
    """

    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_WITH_JITTER
    max_attempts: int = 3
    initial_delay: float = 0.5
    maximum_delay: float = 8.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple[type[Exception], ...] = (
        TimeoutError,
        ConnectionError,
        ConnectionRefusedError,
        ConnectionResetError,
    )
    retry_on_timeout: bool = True
    retry_on_cancelled: bool = False

    def __post_init__(self) -> None:
        """Validate policy values."""
        if self.max_attempts < 1:
            raise ValueError(
                f"max_attempts must be >= 1, got {self.max_attempts}"
            )
        if self.initial_delay < 0:
            raise ValueError(
                f"initial_delay must be >= 0, got {self.initial_delay}"
            )
        if self.maximum_delay < self.initial_delay:
            raise ValueError(
                f"maximum_delay ({self.maximum_delay}) must be >= "
                f"initial_delay ({self.initial_delay})"
            )
        if self.backoff_multiplier < 1.0:
            raise ValueError(
                f"backoff_multiplier must be >= 1.0, got {self.backoff_multiplier}"
            )

    @classmethod
    def no_retry(cls) -> "RetryPolicy":
        """Return a policy that never retries."""
        return cls(strategy=RetryStrategy.NO_RETRY, max_attempts=1)

    @classmethod
    def default(cls) -> "RetryPolicy":
        """Return the default retry policy."""
        return cls()


# ---------------------------------------------------------------------------
# Retry context
# ---------------------------------------------------------------------------


@dataclass
class RetryContext:
    """Runtime context for a single retry session.

    This is mutable — it tracks state across attempts.

    Attributes:
        attempt: Current attempt number (1-based).
        last_error: The last exception that occurred (if any).
        last_error_type: Type name of the last error.
        start_time: Time the execution started.
        delays: List of delays between attempts.
    """

    attempt: int = 0
    last_error: Exception | None = None
    last_error_type: str = ""
    start_time: float = 0.0
    delays: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retry result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryResult:
    """The result of a retry-managed execution.

    Attributes:
        success: Whether execution succeeded.
        value: The return value of the executed callable (if success).
        error: The last error (if any).
        error_type: Type name of the last error.
        attempts: Total attempts made.
        total_delay: Total delay incurred across retries.
        elapsed: Total elapsed time.
        exhausted: Whether retries were exhausted.
        retried: Whether any retry was attempted.
    """

    success: bool = True
    value: Any = None
    error: str = ""
    error_type: str = ""
    attempts: int = 1
    total_delay: float = 0.0
    elapsed: float = 0.0
    exhausted: bool = False
    retried: bool = False


# ---------------------------------------------------------------------------
# Retry manager
# ---------------------------------------------------------------------------


class RetryManager:
    """Central retry and recovery manager.

    Encapsulates retry decisions.  Execution code simply calls
    ``execute()`` and the manager handles the full lifecycle.

    Usage::

        manager = RetryManager(policy=RetryPolicy.default())
        result = await manager.execute(provider.generate, request)
        if result.success:
            response = result.value
    """

    def __init__(self, policy: RetryPolicy | None = None) -> None:
        self._policy = policy or RetryPolicy.default()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        fn: Callable[..., Any],
        *args: Any,
        policy: RetryPolicy | None = None,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute *fn* with retry logic.

        The callable is invoked with *args* and **kwargs*.  If it raises
        a retryable exception, the manager will retry according to the
        configured policy.

        Args:
            fn: The async callable to execute.
            *args: Positional arguments for *fn*.
            policy: Optional override policy for this execution.
            **kwargs: Keyword arguments for *fn*.

        Returns:
            A ``RetryResult`` with the outcome.
        """
        resolved_policy = policy or self._policy
        ctx = RetryContext(start_time=time.monotonic())

        if resolved_policy.strategy == RetryStrategy.NO_RETRY:
            return await self._execute_once(fn, ctx, *args, **kwargs)

        for attempt in range(resolved_policy.max_attempts):
            ctx.attempt = attempt + 1

            if attempt > 0:
                # Calculate delay
                delay = self._calculate_delay(resolved_policy, attempt)
                ctx.delays.append(delay)

                # Wait
                await asyncio.sleep(delay)

            result = await self._execute_once(fn, ctx, *args, **kwargs)

            if result.success:
                return result

            # Check if we should retry
            if not self._should_retry(ctx, resolved_policy):
                return result

        # Exhausted
        return RetryResult(
            success=False,
            value=ctx.last_error,
            error=str(ctx.last_error) if ctx.last_error else "Retry exhausted",
            error_type=ctx.last_error_type or "Exception",
            attempts=resolved_policy.max_attempts,
            total_delay=sum(ctx.delays),
            elapsed=time.monotonic() - ctx.start_time,
            exhausted=True,
            retried=len(ctx.delays) > 0,
        )

    # ------------------------------------------------------------------
    # Delay calculation
    # ------------------------------------------------------------------

    def _calculate_delay(
        self,
        policy: RetryPolicy,
        attempt: int,
    ) -> float:
        """Calculate delay for the given attempt.

        Args:
            policy: The retry policy.
            attempt: The attempt number (1-based, 1 = first retry).

        Returns:
            Delay in seconds.
        """
        if policy.strategy == RetryStrategy.FIXED_DELAY:
            delay = policy.initial_delay
        elif policy.strategy in (
            RetryStrategy.EXPONENTIAL_BACKOFF,
            RetryStrategy.EXPONENTIAL_WITH_JITTER,
        ):
            delay = policy.initial_delay * (
                policy.backoff_multiplier ** (attempt - 1)
            )
        else:
            delay = 0.0

        # Cap at maximum
        delay = min(delay, policy.maximum_delay)

        # Add jitter
        if policy.jitter and delay > 0:
            delay = delay * (0.5 + random.random() * 0.5)

        return delay

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _execute_once(
        self,
        fn: Callable[..., Any],
        ctx: RetryContext,
        *args: Any,
        **kwargs: Any,
    ) -> RetryResult:
        """Execute the callable once and capture the result."""
        if ctx.attempt == 0:
            ctx.attempt = 1

        try:
            value = await fn(*args, **kwargs)
            return RetryResult(
                success=True,
                value=value,
                attempts=ctx.attempt,
                total_delay=sum(ctx.delays),
                elapsed=time.monotonic() - ctx.start_time,
                retried=len(ctx.delays) > 0,
            )
        except Exception as exc:
            ctx.last_error = exc
            ctx.last_error_type = type(exc).__name__
            return RetryResult(
                success=False,
                error=str(exc),
                error_type=type(exc).__name__,
                attempts=ctx.attempt,
                total_delay=sum(ctx.delays),
                elapsed=time.monotonic() - ctx.start_time,
                retried=len(ctx.delays) > 0,
            )

    @staticmethod
    def _should_retry(
        ctx: RetryContext,
        policy: RetryPolicy,
    ) -> bool:
        """Determine whether to retry based on error and policy."""
        if ctx.last_error is None:
            return False

        # Check for asyncio.CancelledError
        if isinstance(ctx.last_error, asyncio.CancelledError):
            return policy.retry_on_cancelled

        # Check for TimeoutError
        if isinstance(ctx.last_error, TimeoutError):
            return policy.retry_on_timeout

        # Check retryable exception types
        for exc_type in policy.retryable_exceptions:
            if isinstance(ctx.last_error, exc_type):
                return True

        return False
