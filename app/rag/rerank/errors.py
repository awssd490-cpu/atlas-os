"""Reranking error hierarchy.

All reranking-specific errors derive from :class:`RerankError`
which inherits from :class:`KnowledgeError`.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import KnowledgeError


class RerankError(KnowledgeError):
    """Base class for all reranking errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "RERANK_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidRerankConfiguration(RerankError):
    """Raised when reranking configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_RERANK_CONFIGURATION", details=details)


class RerankerNotFound(RerankError):
    """Raised when a requested reranker is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Reranker {name!r} not found" if name else "Reranker not found"
        super().__init__(msg, code="RERANKER_NOT_FOUND", details=details)
