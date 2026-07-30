"""HealthMonitor — execute registered health checks and collect results.

Produces immutable ``HealthCheck`` result objects.  Supports both
sync and async health check functions.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.core.health.errors import HealthCheckNotFound, HealthError
from app.core.health.models import HealthCheck, HealthStatus
from app.core.health.registry import get, list_checks, register


class HealthMonitor:
    """Executes health checks and returns immutable results.

    Usage::

        async def db_check() -> HealthStatus:
            return HealthStatus.HEALTHY

        monitor = HealthMonitor()
        monitor.register("database", db_check)
        monitor.register("disk", lambda: HealthStatus.HEALTHY)

        results = await monitor.check_all()
        for check in results:
            print(f"{check.name}: {check.status}")
    """

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    @staticmethod
    def register(name: str, fn: Callable[..., Any] | Callable[..., Awaitable[Any]]) -> None:
        """Register a health check function.

        Args:
            name: Unique check name.
            fn: A sync or async callable.

        Raises:
            DuplicateHealthCheck: If *name* is already registered.
        """
        register(name, fn)

    @staticmethod
    def unregister(name: str) -> None:
        """Unregister a health check.

        Args:
            name: The check name.

        Raises:
            HealthCheckNotFound: If not registered.
        """
        from app.core.health.registry import unregister as _unregister
        _unregister(name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @staticmethod
    async def check(name: str) -> HealthCheck:
        """Execute a single health check by name.

        Args:
            name: The registered check name.

        Returns:
            An immutable ``HealthCheck`` with the result.

        Raises:
            HealthCheckNotFound: If not registered.
        """
        fn = get(name)
        return await _run_check(name, fn)

    @staticmethod
    async def check_all() -> list[HealthCheck]:
        """Execute all registered health checks.

        Checks are executed in registration order.  If a check raises,
        its result captures the exception rather than propagating.

        Returns:
            A list of ``HealthCheck`` results, one per registered check.
        """
        results: list[HealthCheck] = []
        for name in list_checks():
            fn = get(name)
            result = await _run_check(name, fn)
            results.append(result)
        return results

    @staticmethod
    def list_checks() -> list[str]:
        """Return the names of all registered health checks."""
        return list_checks()


# ------------------------------------------------------------------
# Internal runner
# ------------------------------------------------------------------


async def _run_check(name: str, fn: Callable[..., Any] | Callable[..., Awaitable[Any]]) -> HealthCheck:
    """Execute a health check and produce an immutable result.

    Catches any exception and returns an UNHEALTHY status.
    """
    start = time.perf_counter()

    try:
        result = fn()
        if isinstance(result, Awaitable):
            result = await result
    except BaseException as exc:
        elapsed = (time.perf_counter() - start) * 1000.0
        return HealthCheck(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=f"Health check raised: {exc}",
            duration_ms=round(elapsed, 2),
            metadata={"error_type": type(exc).__name__},
        )

    elapsed = (time.perf_counter() - start) * 1000.0

    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    metadata: dict[str, Any] = {}

    if isinstance(result, HealthStatus):
        status = result
        message = _default_message(name, status)
    elif isinstance(result, tuple) and len(result) >= 2:
        if isinstance(result[0], HealthStatus):
            status = result[0]
            message = str(result[1])
            if len(result) > 2 and isinstance(result[2], dict):
                metadata = result[2]
        else:
            status = HealthStatus.HEALTHY if result[0] else HealthStatus.UNHEALTHY
            message = str(result[1])
    elif isinstance(result, bool):
        status = HealthStatus.HEALTHY if result else HealthStatus.UNHEALTHY
        message = _default_message(name, status)
    elif isinstance(result, str):
        status = HealthStatus.HEALTHY
        message = result

    if not message:
        message = _default_message(name, status)

    return HealthCheck(
        name=name,
        status=status,
        message=message,
        duration_ms=round(elapsed, 2),
        metadata=metadata,
    )


def _default_message(name: str, status: HealthStatus) -> str:
    """Generate a default message for a health check result."""
    return f"{name} is {status.name.lower()}"
