"""Vector store error hierarchy.

All vector-store-specific errors derive from :class:`VectorStoreError`
which inherits from :class:`KnowledgeError`.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import KnowledgeError


class VectorStoreError(KnowledgeError):
    """Base class for all vector store errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "VECTOR_STORE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidVectorStoreConfiguration(VectorStoreError):
    """Raised when vector store configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_VECTOR_STORE_CONFIGURATION", details=details)


class VectorStoreFullError(VectorStoreError):
    """Raised when the vector store has reached its maximum capacity."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="VECTOR_STORE_FULL", details=details)


class VectorDimensionMismatchError(VectorStoreError):
    """Raised when a vector has the wrong dimensionality."""

    def __init__(
        self,
        expected: int,
        actual: int,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Vector dimension mismatch: expected {expected}, got {actual}"
        super().__init__(
            msg,
            code="VECTOR_DIMENSION_MISMATCH",
            details={"expected": expected, "actual": actual, **(details or {})},
        )


class VectorNotFoundError(VectorStoreError):
    """Raised when a requested vector is not found."""

    def __init__(
        self,
        chunk_id: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Vector {chunk_id!r} not found" if chunk_id else "Vector not found"
        super().__init__(msg, code="VECTOR_NOT_FOUND", details=details)
