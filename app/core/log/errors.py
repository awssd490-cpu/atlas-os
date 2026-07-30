"""Logging error hierarchy.

All logging-specific errors derive from :class:`LoggingError`
which inherits from :class:`AtlasError`.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class LoggingError(AtlasError):
    """Base class for all logging errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "LOGGING_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidLogLevel(LoggingError):
    """Raised when an invalid log level is used."""

    def __init__(
        self,
        level: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Invalid log level: {level!r}" if level else "Invalid log level"
        super().__init__(msg, code="INVALID_LOG_LEVEL", details=details)
