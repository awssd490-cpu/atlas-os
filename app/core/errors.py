"""ATLAS core exception hierarchy.

Every ATLAS-specific error derives from :class:`AtlasError` so callers can
catch platform errors distinctly from programming errors. Each subsystem
defines narrow subclasses here to keep the hierarchy discoverable in one
place.
"""

from __future__ import annotations

from typing import Any


class AtlasError(Exception):
    """Base class for all ATLAS platform errors.

    Args:
        message: Human-readable error description.
        code: Stable, machine-readable error code (SCREAMING_SNAKE_CASE).
        details: Structured context for logs and API error payloads.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "ATLAS_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error for API responses and structured logs."""
        return {"code": self.code, "message": self.message, "details": self.details}


class ConfigurationError(AtlasError):
    """Raised when configuration is missing, invalid, or inconsistent."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="CONFIGURATION_ERROR", details=details)


class DependencyResolutionError(AtlasError):
    """Raised when the DI container cannot resolve a dependency."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="DEPENDENCY_RESOLUTION_ERROR", details=details)


class ModuleError(AtlasError):
    """Base class for module registration and lifecycle errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "MODULE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ModuleNotFoundError_(ModuleError):
    """Raised when a requested module is not registered.

    Note: trailing underscore avoids shadowing the built-in
    :class:`ModuleNotFoundError`.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MODULE_NOT_FOUND", details=details)


class ModuleDependencyError(ModuleError):
    """Raised when module dependencies are missing or cyclic."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MODULE_DEPENDENCY_ERROR", details=details)


class ModuleBootError(ModuleError):
    """Raised when a module fails to boot."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="MODULE_BOOT_ERROR", details=details)


class LifecycleError(AtlasError):
    """Raised on invalid kernel lifecycle transitions (e.g. double boot)."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="LIFECYCLE_ERROR", details=details)


class EventBusError(AtlasError):
    """Raised on event bus misuse (e.g. emitting a non-Event object)."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, code="EVENT_BUS_ERROR", details=details)
