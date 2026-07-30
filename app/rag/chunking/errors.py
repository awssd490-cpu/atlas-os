"""Chunking error hierarchy.

All chunking-specific errors derive from :class:`ChunkingError`
which inherits from :class:`KnowledgeError`.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import KnowledgeError


class ChunkingError(KnowledgeError):
    """Base class for all chunking errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "CHUNKING_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ChunkingConfigError(ChunkingError):
    """Raised when chunking configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="CHUNKING_CONFIG_ERROR", details=details)


class ChunkingStrategyError(ChunkingError):
    """Raised when a chunking strategy encounters an error."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="CHUNKING_STRATEGY_ERROR", details=details)


class ChunkingEngineError(ChunkingError):
    """Raised when the chunking engine encounters an error."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="CHUNKING_ENGINE_ERROR", details=details)


class UnsupportedStrategyError(ChunkingError):
    """Raised when a requested strategy is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Unsupported chunking strategy: {name!r}" if name else "Unsupported strategy"
        super().__init__(msg, code="CHUNKING_UNSUPPORTED_STRATEGY", details=details)
