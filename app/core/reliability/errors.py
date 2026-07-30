"""Reliability error hierarchy.

All reliability-specific errors derive from :class:`ReliabilityError`
which inherits from :class:`AtlasError`.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class ReliabilityError(AtlasError):
    """Base class for all reliability errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RELIABILITY_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidRetryPolicy(ReliabilityError):
    """Raised when a retry policy is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_RETRY_POLICY", details=details)
