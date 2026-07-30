"""Configuration error hierarchy.

All configuration-specific errors derive from :class:`ConfigurationError`
which inherits from :class:`AtlasError`.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class ConfigurationError(AtlasError):
    """Base class for all configuration errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "CONFIGURATION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidConfiguration(ConfigurationError):
    """Raised when configuration values are invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_CONFIGURATION", details=details)
