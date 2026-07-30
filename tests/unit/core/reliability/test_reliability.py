"""Tests for the Atlas reliability and retry subsystem."""

from __future__ import annotations

import time
from typing import Any

import pytest

from app.core.reliability import (
    InvalidRetryPolicy,
    ReliabilityError,
    RetryExecutor,
    RetryPolicy,
    RetryResult,
)
from app.core.reliability.errors import InvalidRetryPolicy as InvalidRetryPolicy_Impl
from app.core.reliability.errors import ReliabilityError as ReliabilityError_Impl
from app.core.reliability.models import RetryPolicy as RetryPolicy_Impl
from app.core.reliability.models import RetryResult as RetryResult_Impl
from app.core.reliability.retry import RetryExecutor as RetryExecutor_Impl
from app.core.errors import AtlasError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_retry_policy_imported(self) -> None:
        assert RetryPolicy is RetryPolicy_Impl

    def test_retry_result_imported(self) -> None:
        assert RetryResult is RetryResult_Impl

    def test_retry_executor_imported(self) -> None:
        assert RetryExecutor is RetryExecutor_Impl

    def test_reliability_error_imported(self) -> None:
        assert ReliabilityError is ReliabilityError_Impl

    def test_invalid_retry_policy_imported(self) -> None:
        assert InvalidRetryPolicy is InvalidRetryPolicy_Impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(ReliabilityError, AtlasError)
        assert issubclass(InvalidRetryPolicy, ReliabilityError)


# ======================================================================
# RetryPolicy
# ======================================================================


class TestRetryPolicy:
    """RetryPolicy frozen dataclass and validation."""

    def test_default_values(self) -> None:
        p = RetryPolicy()
        assert p.max_attempts == 3
        assert p.initial_delay_ms == 100.0
        assert p.backoff_multiplier == 2.0
        assert p.max_delay_ms == 5000.0
        assert p.retry_exceptions == ()
        assert p.metadata == {}

    def test_custom_values(self) -> None:
        p = RetryPolicy(
            max_attempts=5,
            initial_delay_ms=50.0,
            backoff_multiplier=1.5,
            max_delay_ms=2000.0,
            retry_exceptions=(ValueError,),
            metadata={"name": "custom"},
        )
        assert p.max_attempts == 5
        assert p.initial_delay_ms == 50.0
        assert p.backoff_multiplier == 1.5
        assert p.max_delay_ms == 2000.0
        assert p.retry_exceptions == (ValueError,)

    def test_immutable(self) -> None:
        p = RetryPolicy()
        with pytest.raises(AttributeError):
            p.max_attempts = 5  # type: ignore[misc]

    def test_validate_passes(self) -> None:
        RetryPolicy(max_attempts=1, initial_delay_ms=0, backoff_multiplier=1.0).validate()

    def test_validate_max_attempts_zero(self) -> None:
        with pytest.raises(InvalidRetryPolicy):
            RetryPolicy(max_attempts=0).validate()

    def test_validate_max_attempts_negative(self) -> None:
        with pytest.raises(InvalidRetryPolicy):
            RetryPolicy(max_attempts=-1).validate()

    def test_validate_initial_delay_negative(self) -> None:
        with pytest.raises(InvalidRetryPolicy):
            RetryPolicy(initial_delay_ms=-1.0).validate()

    def test_validate_backoff_less_than_one(self) -> None:
        with pytest.raises(InvalidRetryPolicy):
            RetryPolicy(backoff_multiplier=0.5).validate()

    def test_validate_max_delay_negative(self) -> None:
        with pytest.raises(InvalidRetryPolicy):
            RetryPolicy(max_delay_ms=-1.0).validate()


# ======================================================================
# RetryResult
# ======================================================================


class TestRetryResult:
    """RetryResult frozen dataclass."""

    def test_default_values(self) -> None:
        r = RetryResult()
        assert r.attempts == 0
        assert r.success is False
        assert r.duration_ms == 0.0
        assert r.metadata == {}

    def test_custom_values(self) -> None:
        r = RetryResult(
            attempts=3,
            success=False,
            duration_ms=150.0,
            metadata={"last_error": "timeout"},
        )
        assert r.attempts == 3
        assert r.success is False
        assert r.duration_ms == 150.0
        assert r.metadata == {"last_error": "timeout"}

    def test_immutable(self) -> None:
        r = RetryResult()
        with pytest.raises(AttributeError):
            r.success = True  # type: ignore[misc]


# ======================================================================
# RetryExecutor — successful execution
# ======================================================================


class TestRetryExecutorSuccess:
    """RetryExecutor with immediately-successful calls."""

    @pytest.fixture
    def executor(self) -> RetryExecutor:
        return RetryExecutor()

    @pytest.mark.asyncio
    async def test_sync_success(self, executor: RetryExecutor) -> None:
        result = await executor.execute(lambda: "ok")
        assert result.success is True
        assert result.attempts == 1
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_async_success(self, executor: RetryExecutor) -> None:
        async def async_fn() -> str:
            return "ok"
        result = await executor.execute(async_fn)
        assert result.success is True
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_sync_with_args(self, executor: RetryExecutor) -> None:
        result = await executor.execute(lambda a, b: a + b, 1, b=2)
        assert result.success is True
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_metadata_on_success(self, executor: RetryExecutor) -> None:
        result = await executor.execute(lambda: "ok")
        assert "success_on_attempt" in result.metadata
        assert result.metadata["success_on_attempt"] == 1


# ======================================================================
# RetryExecutor — retry then succeed
# ======================================================================


class TestRetryExecutorRetryThenSuccess:
    """RetryExecutor with transient failures."""

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self) -> None:
        """Fail twice, succeed on third attempt."""
        counter: list[int] = [0]

        def flaky() -> None:
            counter[0] += 1
            if counter[0] < 3:
                raise ValueError("not ready")

        policy = RetryPolicy(
            max_attempts=5,
            initial_delay_ms=1.0,  # very fast for testing
            backoff_multiplier=1.0,
        )
        executor = RetryExecutor(policy=policy)
        result = await executor.execute(flaky)

        assert result.success is True
        assert result.attempts == 3
        assert counter[0] == 3

    @pytest.mark.asyncio
    async def test_async_retry_then_succeed(self) -> None:
        counter: list[int] = [0]

        async def flaky() -> None:
            counter[0] += 1
            if counter[0] < 2:
                raise RuntimeError("transient")

        policy = RetryPolicy(
            max_attempts=3,
            initial_delay_ms=1.0,
            backoff_multiplier=1.0,
        )
        executor = RetryExecutor(policy=policy)
        result = await executor.execute(flaky)

        assert result.success is True
        assert result.attempts == 2


# ======================================================================
# RetryExecutor — final failure
# ======================================================================


class TestRetryExecutorFailure:
    """RetryExecutor when all attempts fail."""

    @pytest.mark.asyncio
    async def test_all_attempts_fail(self) -> None:
        def always_fails() -> None:
            raise ValueError("persistent error")

        policy = RetryPolicy(
            max_attempts=3,
            initial_delay_ms=1.0,
            backoff_multiplier=1.0,
        )
        executor = RetryExecutor(policy=policy)
        result = await executor.execute(always_fails)

        assert result.success is False
        assert result.attempts == 3
        assert "persistent error" in str(result.metadata.get("last_error", ""))

    @pytest.mark.asyncio
    async def test_single_attempt_failure(self) -> None:
        policy = RetryPolicy(
            max_attempts=1,
            initial_delay_ms=1.0,
        )
        executor = RetryExecutor(policy=policy)
        result = await executor.execute(lambda: (_ for _ in ()).throw(ValueError("fail")))  # noqa: E731

        assert result.success is False
        assert result.attempts == 1


# ======================================================================
# RetryExecutor — exception filtering
# ======================================================================


class TestRetryExecutorExceptionFilter:
    """RetryExecutor exception type filtering."""

    @pytest.mark.asyncio
    async def test_retryable_exception(self) -> None:
        """ValueError is retryable, should retry."""
        counter: list[int] = [0]

        def flaky() -> None:
            counter[0] += 1
            raise ValueError("retryable")

        policy = RetryPolicy(
            max_attempts=3,
            initial_delay_ms=1.0,
            backoff_multiplier=1.0,
            retry_exceptions=(ValueError,),
        )
        executor = RetryExecutor(policy=policy)
        result = await executor.execute(flaky)

        assert result.success is False
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self) -> None:
        """TypeError is not retryable, should fail immediately."""
        counter: list[int] = [0]

        def flaky() -> None:
            counter[0] += 1
            raise TypeError("not retryable")

        policy = RetryPolicy(
            max_attempts=5,
            initial_delay_ms=100.0,
            retry_exceptions=(ValueError,),
        )
        executor = RetryExecutor(policy=policy)
        result = await executor.execute(flaky)

        assert result.success is False
        assert result.attempts == 1  # no retry
        assert counter[0] == 1


# ======================================================================
# RetryExecutor — backoff calculation
# ======================================================================


class TestRetryExecutorBackoff:
    """Backoff delay calculation."""

    @pytest.mark.asyncio
    async def test_exponential_backoff(self) -> None:
        """Delay should increase exponentially."""
        counter: list[int] = [0]

        def flaky() -> None:
            counter[0] += 1
            raise ValueError("fail")

        policy = RetryPolicy(
            max_attempts=4,
            initial_delay_ms=10.0,
            backoff_multiplier=2.0,
            max_delay_ms=10000.0,
        )
        executor = RetryExecutor(policy=policy)
        start = time.monotonic()
        result = await executor.execute(flaky)
        elapsed = (time.monotonic() - start) * 1000.0

        assert result.success is False
        assert result.attempts == 4
        # Expected delays: 10 + 20 + 40 = 70ms minimum
        assert elapsed >= 60.0, f"Elapsed {elapsed}ms, expected >= 60ms"

    @pytest.mark.asyncio
    async def test_max_delay_cap(self) -> None:
        """Delay should be capped at max_delay_ms."""
        counter: list[int] = [0]

        def flaky() -> None:
            counter[0] += 1
            raise ValueError("fail")

        policy = RetryPolicy(
            max_attempts=5,
            initial_delay_ms=100.0,
            backoff_multiplier=10.0,
            max_delay_ms=50.0,  # cap at 50ms
        )
        executor = RetryExecutor(policy=policy)
        start = time.monotonic()
        result = await executor.execute(flaky)
        elapsed = (time.monotonic() - start) * 1000.0

        assert result.success is False
        assert result.attempts == 5
        # With cap at 50ms: 4 delays × 50ms = 200ms expected
        assert elapsed >= 150.0, f"Elapsed {elapsed}ms, expected >= 150ms"


# ======================================================================
# RetryExecutor — edge cases
# ======================================================================


class TestRetryExecutorEdgeCases:
    """RetryExecutor edge cases."""

    @pytest.mark.asyncio
    async def test_non_callable_raises(self) -> None:
        executor = RetryExecutor()
        with pytest.raises(ReliabilityError) as exc:
            await executor.execute("not callable")  # type: ignore[arg-type]
        assert "callable" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_deterministic_result(self) -> None:
        def fn() -> str:
            return "ok"

        executor = RetryExecutor()
        r1 = await executor.execute(fn)
        r2 = await executor.execute(fn)
        assert r1.success == r2.success
        assert r1.attempts == r2.attempts

    @pytest.mark.asyncio
    async def test_custom_policy_override(self) -> None:
        default_policy = RetryPolicy(max_attempts=3)
        override_policy = RetryPolicy(max_attempts=1, initial_delay_ms=1.0)
        executor = RetryExecutor(policy=default_policy)

        def fails() -> None:
            raise ValueError("fail")

        result = await executor.execute(fails, retry_policy=override_policy)
        assert result.attempts == 1  # override used

    @pytest.mark.asyncio
    async def test_immutable_result(self) -> None:
        executor = RetryExecutor()
        result = await executor.execute(lambda: "ok")
        with pytest.raises(AttributeError):
            result.success = False  # type: ignore[misc]
