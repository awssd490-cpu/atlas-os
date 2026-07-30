"""Health monitoring error hierarchy.

All health-specific errors derive from :class:`HealthError`
which inherits from :class:`AtlasError`.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class HealthError(AtlasError):
    """Base class for all health monitoring errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "HEALTH_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class HealthCheckNotFound(HealthError):
    """Raised when a requested health check is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Health check {name!r} not found" if name else "Health check not found"
        super().__init__(msg, code="HEALTH_CHECK_NOT_FOUND", details=details)


class DuplicateHealthCheck(HealthError):
    """Raised when a health check is registered under an already-used name."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Health check {name!r} is already registered" if name else "Duplicate health check"
        super().__init__(msg, code="DUPLICATE_HEALTH_CHECK", details=details)
