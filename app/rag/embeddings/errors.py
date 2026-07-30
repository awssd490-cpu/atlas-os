"""Embedding error hierarchy.

All embedding-specific errors derive from :class:`EmbeddingError`
which inherits from :class:`KnowledgeError`.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import KnowledgeError


class EmbeddingError(KnowledgeError):
    """Base class for all embedding errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EMBEDDING_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidEmbeddingConfiguration(EmbeddingError):
    """Raised when embedding configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_EMBEDDING_CONFIGURATION", details=details)


class EmbeddingProviderError(EmbeddingError):
    """Raised when an embedding provider encounters an error."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="EMBEDDING_PROVIDER_ERROR", details=details)


class UnsupportedEmbeddingProvider(EmbeddingError):
    """Raised when a requested provider is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Unsupported embedding provider: {name!r}" if name else "Unsupported provider"
        super().__init__(msg, code="UNSUPPORTED_EMBEDDING_PROVIDER", details=details)
