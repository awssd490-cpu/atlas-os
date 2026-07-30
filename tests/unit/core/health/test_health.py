"""Tests for the Atlas health monitoring subsystem."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.health import (
    DuplicateHealthCheck,
    HealthCheck,
    HealthCheckNotFound,
    HealthError,
    HealthMonitor,
    HealthStatus,
    clear_checks,
    get,
    list_checks,
    register,
    unregister,
)
from app.core.health.errors import DuplicateHealthCheck as DuplicateHealthCheck_Impl
from app.core.health.errors import HealthCheckNotFound as HealthCheckNotFound_Impl
from app.core.health.errors import HealthError as HealthError_Impl
from app.core.health.models import HealthCheck as HealthCheck_Impl
from app.core.health.models import HealthStatus as HealthStatus_Impl
from app.core.health.monitor import HealthMonitor as HealthMonitor_Impl
from app.core.errors import AtlasError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_health_status_imported(self) -> None:
        assert HealthStatus is HealthStatus_Impl

    def test_health_check_imported(self) -> None:
        assert HealthCheck is HealthCheck_Impl

    def test_health_monitor_imported(self) -> None:
        assert HealthMonitor is HealthMonitor_Impl

    def test_health_error_imported(self) -> None:
        assert HealthError is HealthError_Impl

    def test_health_check_not_found_imported(self) -> None:
        assert HealthCheckNotFound is HealthCheckNotFound_Impl

    def test_duplicate_health_check_imported(self) -> None:
        assert DuplicateHealthCheck is DuplicateHealthCheck_Impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(HealthError, AtlasError)
        assert issubclass(HealthCheckNotFound, HealthError)
        assert issubclass(DuplicateHealthCheck, HealthError)


# ======================================================================
# HealthStatus
# ======================================================================


class TestHealthStatus:
    """HealthStatus enum values and ordering."""

    def test_values(self) -> None:
        assert HealthStatus.UNKNOWN.value == 0
        assert HealthStatus.HEALTHY.value == 1
        assert HealthStatus.DEGRADED.value == 2
        assert HealthStatus.UNHEALTHY.value == 3

    def test_ordering(self) -> None:
        assert HealthStatus.UNKNOWN.value < HealthStatus.HEALTHY.value
        assert HealthStatus.HEALTHY.value < HealthStatus.DEGRADED.value
        assert HealthStatus.DEGRADED.value < HealthStatus.UNHEALTHY.value


# ======================================================================
# HealthCheck
# ======================================================================


class TestHealthCheck:
    """HealthCheck frozen dataclass."""

    def test_default_values(self) -> None:
        hc = HealthCheck()
        assert hc.name == ""
        assert hc.status == HealthStatus.UNKNOWN
        assert hc.message == ""
        assert hc.duration_ms == 0.0
        assert hc.metadata == {}

    def test_custom_values(self) -> None:
        hc = HealthCheck(
            name="db",
            status=HealthStatus.HEALTHY,
            message="Database is up",
            duration_ms=5.2,
            metadata={"host": "localhost"},
        )
        assert hc.name == "db"
        assert hc.status == HealthStatus.HEALTHY
        assert hc.message == "Database is up"
        assert hc.duration_ms == 5.2
        assert hc.metadata == {"host": "localhost"}

    def test_immutable(self) -> None:
        hc = HealthCheck()
        with pytest.raises(AttributeError):
            hc.name = "changed"  # type: ignore[misc]


# ======================================================================
# HealthRegistry
# ======================================================================


class TestHealthRegistry:
    """Registry functions."""

    def setup_method(self) -> None:
        clear_checks()

    def teardown_method(self) -> None:
        clear_checks()

    def test_register_and_get(self) -> None:
        fn = lambda: HealthStatus.HEALTHY  # noqa: E731
        register("disk", fn)
        assert get("disk") is fn

    def test_register_duplicate(self) -> None:
        fn = lambda: HealthStatus.HEALTHY  # noqa: E731
        register("db", fn)
        with pytest.raises(DuplicateHealthCheck):
            register("db", fn)

    def test_get_unknown(self) -> None:
        with pytest.raises(HealthCheckNotFound):
            get("nonexistent")

    def test_unregister(self) -> None:
        register("x", lambda: HealthStatus.HEALTHY)  # noqa: E731
        unregister("x")
        assert "x" not in list_checks()

    def test_unregister_unknown(self) -> None:
        with pytest.raises(HealthCheckNotFound):
            unregister("nonexistent")

    def test_list_checks(self) -> None:
        register("a", lambda: HealthStatus.HEALTHY)  # noqa: E731
        register("b", lambda: HealthStatus.HEALTHY)  # noqa: E731
        names = list_checks()
        assert "a" in names
        assert "b" in names

    def test_clear_checks(self) -> None:
        register("x", lambda: HealthStatus.HEALTHY)  # noqa: E731
        clear_checks()
        assert list_checks() == []


# ======================================================================
# HealthMonitor — sync checks
# ======================================================================


class TestHealthMonitorSync:
    """HealthMonitor with sync health checks."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        clear_checks()
        yield
        clear_checks()

    async def test_sync_healthy(self) -> None:
        HealthMonitor.register("disk", lambda: HealthStatus.HEALTHY)
        result = await HealthMonitor.check("disk")
        assert result.status == HealthStatus.HEALTHY
        assert result.name == "disk"
        assert result.duration_ms >= 0

    async def test_sync_unhealthy(self) -> None:
        HealthMonitor.register("disk", lambda: HealthStatus.UNHEALTHY)
        result = await HealthMonitor.check("disk")
        assert result.status == HealthStatus.UNHEALTHY

    async def test_sync_bool_true(self) -> None:
        HealthMonitor.register("ok", lambda: True)
        result = await HealthMonitor.check("ok")
        assert result.status == HealthStatus.HEALTHY

    async def test_sync_bool_false(self) -> None:
        HealthMonitor.register("bad", lambda: False)
        result = await HealthMonitor.check("bad")
        assert result.status == HealthStatus.UNHEALTHY

    async def test_sync_tuple(self) -> None:
        HealthMonitor.register("svc", lambda: (HealthStatus.DEGRADED, "high load"))
        result = await HealthMonitor.check("svc")
        assert result.status == HealthStatus.DEGRADED
        assert "high load" in result.message

    async def test_sync_string(self) -> None:
        HealthMonitor.register("msg", lambda: "all good")
        result = await HealthMonitor.check("msg")
        assert result.status == HealthStatus.HEALTHY

    async def test_check_all_order(self) -> None:
        HealthMonitor.register("a", lambda: HealthStatus.HEALTHY)
        HealthMonitor.register("b", lambda: HealthStatus.HEALTHY)
        HealthMonitor.register("c", lambda: HealthStatus.HEALTHY)
        results = await HealthMonitor.check_all()
        assert [r.name for r in results] == ["a", "b", "c"]

    async def test_unknown_check(self) -> None:
        with pytest.raises(HealthCheckNotFound):
            await HealthMonitor.check("nonexistent")


# ======================================================================
# HealthMonitor — async checks
# ======================================================================


class TestHealthMonitorAsync:
    """HealthMonitor with async health checks."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        clear_checks()
        yield
        clear_checks()

    async def test_async_healthy(self) -> None:
        async def check() -> HealthStatus:
            return HealthStatus.HEALTHY
        HealthMonitor.register("async", check)
        result = await HealthMonitor.check("async")
        assert result.status == HealthStatus.HEALTHY

    async def test_async_unhealthy(self) -> None:
        async def check() -> HealthStatus:
            return HealthStatus.UNHEALTHY
        HealthMonitor.register("async", check)
        result = await HealthMonitor.check("async")
        assert result.status == HealthStatus.UNHEALTHY

    async def test_async_tuple(self) -> None:
        async def check() -> tuple:
            return (HealthStatus.DEGRADED, "slow response")
        HealthMonitor.register("api", check)
        result = await HealthMonitor.check("api")
        assert result.status == HealthStatus.DEGRADED
        assert "slow" in result.message

    async def test_async_bool(self) -> None:
        async def check() -> bool:
            return True
        HealthMonitor.register("ok", check)
        result = await HealthMonitor.check("ok")
        assert result.status == HealthStatus.HEALTHY


# ======================================================================
# HealthMonitor — exception handling
# ======================================================================


class TestHealthMonitorExceptions:
    """HealthMonitor exception-safe execution."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        clear_checks()
        yield
        clear_checks()

    async def test_exception_returns_unhealthy(self) -> None:
        def failing() -> None:
            raise RuntimeError("connection refused")
        HealthMonitor.register("db", failing)
        result = await HealthMonitor.check("db")
        assert result.status == HealthStatus.UNHEALTHY
        assert "connection refused" in result.message

    async def test_async_exception(self) -> None:
        async def failing() -> None:
            raise ValueError("timeout")
        HealthMonitor.register("api", failing)
        result = await HealthMonitor.check("api")
        assert result.status == HealthStatus.UNHEALTHY
        assert "timeout" in result.message

    async def test_exception_in_check_all(self) -> None:
        HealthMonitor.register("good", lambda: HealthStatus.HEALTHY)
        HealthMonitor.register("bad", lambda: (_ for _ in ()).throw(RuntimeError("fail")))  # noqa: E731
        HealthMonitor.register("also_good", lambda: HealthStatus.HEALTHY)
        results = await HealthMonitor.check_all()
        assert results[0].status == HealthStatus.HEALTHY
        assert results[1].status == HealthStatus.UNHEALTHY
        assert results[2].status == HealthStatus.HEALTHY

    async def test_duration_on_exception(self) -> None:
        def slow_fail() -> None:
            import time
            time.sleep(0.01)
            raise ValueError("fail")
        HealthMonitor.register("slow", slow_fail)
        result = await HealthMonitor.check("slow")
        assert result.duration_ms >= 10.0
        assert result.status == HealthStatus.UNHEALTHY


# ======================================================================
# HealthMonitor — immutable results
# ======================================================================


class TestHealthMonitorResults:
    """HealthMonitor returns immutable results."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        clear_checks()
        yield
        clear_checks()

    async def test_result_immutable(self) -> None:
        HealthMonitor.register("x", lambda: HealthStatus.HEALTHY)
        result = await HealthMonitor.check("x")
        with pytest.raises(AttributeError):
            result.name = "changed"  # type: ignore[misc]

    async def test_deterministic(self) -> None:
        HealthMonitor.register("stable", lambda: HealthStatus.HEALTHY)
        r1 = await HealthMonitor.check("stable")
        r2 = await HealthMonitor.check("stable")
        assert r1.status == r2.status

    async def test_metadata(self) -> None:
        HealthMonitor.register("meta", lambda: (HealthStatus.HEALTHY, "ok", {"version": "1.0"}))
        result = await HealthMonitor.check("meta")
        assert result.metadata == {"version": "1.0"}
