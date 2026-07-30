"""Concurrency error hierarchy.

All concurrency-specific errors derive from :class:`ConcurrencyError`
which inherits from :class:`AtlasError`.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class ConcurrencyError(AtlasError):
    """Base class for all concurrency and resource errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "CONCURRENCY_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ResourceNotFound(ConcurrencyError):
    """Raised when a requested resource is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Resource {name!r} not found" if name else "Resource not found"
        super().__init__(msg, code="RESOURCE_NOT_FOUND", details=details)


class DuplicateResource(ConcurrencyError):
    """Raised when a resource is registered under an already-used name."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Resource {name!r} is already registered" if name else "Duplicate resource"
        super().__init__(msg, code="DUPLICATE_RESOURCE", details=details)
