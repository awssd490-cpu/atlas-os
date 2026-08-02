"""Release engineering exception hierarchy.

Every release-specific error derives from :class:`ReleaseError`, which in
turn derives from :class:`AtlasError` so callers can catch platform errors
distinctly from programming errors.  Each operation defines a narrow
subclass here to keep the hierarchy discoverable in one place.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class ReleaseError(AtlasError):
    """Base class for all release engineering errors.

    Args:
        message: Human-readable error description.
        code: Stable, machine-readable error code.  Defaults to
            ``"RELEASE_ERROR"``.
        details: Structured context for logs and API error payloads.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "RELEASE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidReleaseConfiguration(ReleaseError):
    """Raised when release configuration is missing or invalid."""

    def __init__(
        self, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message, code="INVALID_RELEASE_CONFIGURATION", details=details
        )


class VersionError(ReleaseError):
    """Raised when a version string cannot be parsed or is invalid."""

    def __init__(
        self, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="VERSION_ERROR", details=details)


class VersionNotFound(ReleaseError):
    """Raised when the current project version cannot be resolved."""

    def __init__(
        self, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="VERSION_NOT_FOUND", details=details)


class ChangelogError(ReleaseError):
    """Raised when changelog construction or rendering fails."""

    def __init__(
        self, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="CHANGELOG_ERROR", details=details)


class ArtifactError(ReleaseError):
    """Raised when release artifact discovery or validation fails."""

    def __init__(
        self, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code="ARTIFACT_ERROR", details=details)
