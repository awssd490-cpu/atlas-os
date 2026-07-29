"""Tests for retry and recovery framework."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.runtime.retry import (
    ErrorClassification,
    RetryContext,
    RetryManager,
    RetryPolicy,
    RetryResult,
    RetryStrategy,
)


# ---------------------------------------------------------------------------
# RetryPolicy tests
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_default_policy(self) -> None:
        policy = RetryPolicy.default()
        assert policy.max_attempts == 3
        assert policy.strategy == RetryStrategy.EXPONENTIAL_WITH_JITTER
        assert policy.retry_on_timeout is True

    def test_no_retry_policy(self) -> None:
        policy = RetryPolicy.no_retry()
        assert policy.max_attempts == 1
        assert policy.strategy == RetryStrategy.NO_RETRY

    def test_custom_policy(self) -> None:
        policy = RetryPolicy(
            max_attempts=5,
            strategy=RetryStrategy.FIXED_DELAY,
            initial_delay=1.0,
        )
        assert policy.max_attempts == 5
        assert policy.initial_delay == 1.0

    def test_invalid_max_attempts_raises(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(max_attempts=0)

    def test_invalid_initial_delay_raises(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(initial_delay=-1.0)

    def test_max_delay_less_than_initial_raises(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(initial_delay=5.0, maximum_delay=1.0)

    def test_invalid_backoff_raises(self) -> None:
        with pytest.raises(ValueError):
            RetryPolicy(backoff_multiplier=0.5)

    def test_retry_strategy_values(self) -> None:
        assert RetryStrategy.NO_RETRY.value == "no_retry"
        assert RetryStrategy.FIXED_DELAY.value == "fixed_delay"
        assert RetryStrategy.EXPONENTIAL_BACKOFF.value == "exponential_backoff"
        assert RetryStrategy.EXPONENTIAL_WITH_JITTER.value == "exponential_with_jitter"


# ---------------------------------------------------------------------------
# RetryContext tests
# ---------------------------------------------------------------------------


class TestRetryContext:
    def test_default_context(self) -> None:
        ctx = RetryContext()
        assert ctx.attempt == 0
        assert ctx.last_error is None
        assert ctx.delays == []

    def test_mutable(self) -> None:
        ctx = RetryContext()
        ctx.attempt = 2
        ctx.delays.append(0.5)
        assert ctx.attempt == 2
        assert len(ctx.delays) == 1


# ---------------------------------------------------------------------------
# RetryResult tests
# ---------------------------------------------------------------------------


class TestRetryResult:
    def test_success_default(self) -> None:
        result = RetryResult()
        assert result.success is True
        assert result.attempts == 1

    def test_failure(self) -> None:
        result = RetryResult(
            success=False,
            error="timeout",
            error_type="TimeoutError",
            exhausted=True,
        )
        assert result.success is False
        assert result.error == "timeout"
        assert result.exhausted is True


# ---------------------------------------------------------------------------
# Delay calculation tests
# ---------------------------------------------------------------------------


class TestDelayCalculation:
    def test_no_retry_zero_delay(self) -> None:
        manager = RetryManager(RetryPolicy(strategy=RetryStrategy.NO_RETRY))
        delay = manager._calculate_delay(RetryPolicy(strategy=RetryStrategy.NO_RETRY), 1)
        assert delay == 0.0

    def test_fixed_delay(self) -> None:
        policy = RetryPolicy(strategy=RetryStrategy.FIXED_DELAY, initial_delay=1.0, jitter=False)
        manager = RetryManager(policy)
        delay = manager._calculate_delay(policy, 1)
        assert delay == 1.0

    def test_exponential_backoff(self) -> None:
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay=1.0,
            backoff_multiplier=2.0,
            jitter=False,
        )
        manager = RetryManager(policy)
        delay1 = manager._calculate_delay(policy, 1)
        delay2 = manager._calculate_delay(policy, 2)
        delay3 = manager._calculate_delay(policy, 3)
        assert delay1 == 1.0
        assert delay2 == 2.0
        assert delay3 == 4.0

    def test_exponential_capped(self) -> None:
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            initial_delay=1.0,
            backoff_multiplier=10.0,
            maximum_delay=5.0,
            jitter=False,
        )
        manager = RetryManager(policy)
        delay = manager._calculate_delay(policy, 3)
        assert delay == 5.0  # capped

    def test_jitter_adds_variation(self) -> None:
        policy = RetryPolicy(
            strategy=RetryStrategy.EXPONENTIAL_WITH_JITTER,
            initial_delay=1.0,
            jitter=True,
        )
        manager = RetryManager(policy)
        delays = [manager._calculate_delay(policy, 1) for _ in range(10)]
        # With jitter, delays should vary
        assert min(delays) < max(delays)


# ---------------------------------------------------------------------------
# RetryManager execution tests
# ---------------------------------------------------------------------------


class TestRetryManagerExecution:
    async def test_successful_execution(self) -> None:
        """Successful execution returns immediately."""
        manager = RetryManager()
        result = await manager.execute(_success_fn)
        assert result.success is True
        assert result.value == "ok"
        assert result.attempts == 1
        assert result.retried is False

    async def test_no_retry_on_failure(self) -> None:
        """No retry policy fails immediately."""
        manager = RetryManager(RetryPolicy.no_retry())
        result = await manager.execute(_always_fail_fn)
        assert result.success is False
        assert result.attempts == 1
        assert result.exhausted is False  # not exhausted, just failed

    async def test_retry_then_succeed(self) -> None:
        """Retry succeeds on second attempt."""
        fn = _FailThenSucceed(fail_count=1, fail_message="first fail")
        manager = RetryManager(RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.FIXED_DELAY,
            initial_delay=0.01,
            jitter=False,
            retryable_exceptions=(ValueError,),
        ))
        result = await manager.execute(fn)
        assert result.success is True
        assert result.value == "success"
        assert result.attempts == 2
        assert result.retried is True

    async def test_retry_exhausted(self) -> None:
        """All retries exhausted returns failure."""
        fn = _FailThenSucceed(fail_count=10, fail_message="keeps failing")
        manager = RetryManager(RetryPolicy(
            max_attempts=3,
            strategy=RetryStrategy.FIXED_DELAY,
            initial_delay=0.01,
            jitter=False,
            retryable_exceptions=(ValueError,),
        ))
        result = await manager.execute(fn)
        assert result.success is False
        assert result.attempts == 3
        assert result.exhausted is True
        assert result.retried is True

    async def test_retry_count_matches(self) -> None:
        """With max_attempts=1, no retries happen."""
        fn = _FailThenSucceed(fail_count=5, fail_message="always fail")
        manager = RetryManager(RetryPolicy(max_attempts=1))
        result = await manager.execute(fn)
        assert result.attempts == 1
        assert result.retried is False

    async def test_custom_policy_per_execution(self) -> None:
        """Per-execution policy overrides manager default."""
        fn = _FailThenSucceed(fail_count=5, fail_message="fail")
        manager = RetryManager(RetryPolicy.no_retry())
        # Override with a retry policy
        result = await manager.execute(
            fn,
            policy=RetryPolicy(
                max_attempts=2,
                strategy=RetryStrategy.FIXED_DELAY,
                initial_delay=0.01,
                jitter=False,
                retryable_exceptions=(ValueError,),
            ),
        )
        assert result.attempts == 2


# ---------------------------------------------------------------------------
# Error classification tests
# ---------------------------------------------------------------------------


async def _success_fn() -> str:
    return "ok"


async def _always_fail_fn() -> str:
    raise ValueError("always fails")


class _FailThenSucceed:
    """Callable that fails N times then succeeds."""

    def __init__(self, fail_count: int, fail_message: str = "fail"):
        self._remaining = fail_count
        self._fail_message = fail_message

    async def __call__(self) -> str:
        if self._remaining > 0:
            self._remaining -= 1
            raise ValueError(self._fail_message)
        return "success"


class TestErrorClassification:
    def test_most_errors_retryable(self) -> None:
        retryable_types = RetryPolicy.default().retryable_exceptions
        assert TimeoutError in retryable_types
        assert ConnectionError in retryable_types

    def test_non_retryable_types(self) -> None:
        """ValueError is not in default retryable exceptions."""
        policy = RetryPolicy.default()
        for exc_type in policy.retryable_exceptions:
            assert not issubclass(ValueError, exc_type)


# ---------------------------------------------------------------------------
# Parallel compatibility
# ---------------------------------------------------------------------------


class TestParallelCompatibility:
    async def test_retry_in_parallel(self) -> None:
        """RetryManager works correctly with parallel tool execution."""
        policy = RetryPolicy(
            max_attempts=2,
            strategy=RetryStrategy.FIXED_DELAY,
            initial_delay=0.01,
            jitter=False,
            retryable_exceptions=(ValueError,),
        )
        manager = RetryManager(policy)

        async def execute_with_retry(tool_fn):
            return await manager.execute(tool_fn)

        # Simulate parallel execution with independent retries
        fns = [
            _FailThenSucceed(fail_count=1),
            _success_fn,
            _FailThenSucceed(fail_count=0),  # succeeds immediately
        ]
        results = await asyncio.gather(
            *[execute_with_retry(fn) for fn in fns],
        )
        assert results[0].success is True
        assert results[1].success is True
        assert results[2].success is True


# ---------------------------------------------------------------------------
# Config integration tests
# ---------------------------------------------------------------------------


class TestRetryConfig:
    def test_default_retry_config(self) -> None:
        from app.agent.config import AgentConfig

        config = AgentConfig.default()
        assert config.retry_enabled is True
        assert config.retry_strategy == "exponential_with_jitter"
        assert config.max_retry_attempts == 3

    def test_invalid_retry_strategy(self) -> None:
        from app.agent.config import AgentConfig

        with pytest.raises(ValueError):
            AgentConfig(retry_strategy="invalid")

    def test_invalid_retry_attempts(self) -> None:
        from app.agent.config import AgentConfig

        with pytest.raises(ValueError):
            AgentConfig(max_retry_attempts=0)

    def test_invalid_retry_delay(self) -> None:
        from app.agent.config import AgentConfig

        with pytest.raises(ValueError):
            AgentConfig(retry_initial_delay=-1)

    def test_max_delay_validation(self) -> None:
        from app.agent.config import AgentConfig

        with pytest.raises(ValueError):
            AgentConfig(retry_initial_delay=5.0, retry_max_delay=2.0)
