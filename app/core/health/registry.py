"""Health check registry.

A global registry that maps check names to callables.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

# A health check callable can be sync or async, returning HealthStatus
# or a tuple[HealthStatus, str] or a full HealthCheck object.
# Type is intentionally broad for flexibility.
HealthCheckFn = Callable[..., Any | Awaitable[Any]]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_checks: dict[str, HealthCheckFn] = {}


def register(name: str, fn: HealthCheckFn) -> None:
    """Register a health check function.

    Args:
        name: Unique check name.
        fn: A sync or async callable that performs the check.

    Raises:
        DuplicateHealthCheck: If *name* is already registered.
    """
    from app.core.health.errors import DuplicateHealthCheck

    if name in _checks:
        raise DuplicateHealthCheck(name)
    _checks[name] = fn


def unregister(name: str) -> None:
    """Unregister a previously registered health check.

    Args:
        name: The check name to unregister.

    Raises:
        HealthCheckNotFound: If the check is not registered.
    """
    from app.core.health.errors import HealthCheckNotFound

    if name not in _checks:
        raise HealthCheckNotFound(name)
    del _checks[name]


def get(name: str) -> HealthCheckFn:
    """Look up a registered health check function by name.

    Args:
        name: The check name.

    Returns:
        The registered callable.

    Raises:
        HealthCheckNotFound: If the check is not registered.
    """
    from app.core.health.errors import HealthCheckNotFound

    try:
        return _checks[name]
    except KeyError:
        raise HealthCheckNotFound(name) from None


def list_checks() -> list[str]:
    """Return the names of all registered health checks."""
    return list(_checks)


def clear_checks() -> None:
    """Remove all registered health checks (used in tests)."""
    _checks.clear()
