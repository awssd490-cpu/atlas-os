"""Hybrid retrieval error hierarchy.

All hybrid-retrieval errors derive from :class:`HybridError`
which inherits from :class:`KnowledgeError`.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import KnowledgeError


class HybridError(KnowledgeError):
    """Base class for all hybrid retrieval errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "HYBRID_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidHybridConfiguration(HybridError):
    """Raised when hybrid retrieval configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_HYBRID_CONFIGURATION", details=details)


class FusionError(HybridError):
    """Raised when score fusion fails."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="FUSION_ERROR", details=details)
