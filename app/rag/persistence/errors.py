"""Persistence error hierarchy.

All persistence-specific errors derive from :class:`PersistenceError`
which inherits from :class:`KnowledgeError`.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import KnowledgeError


class PersistenceError(KnowledgeError):
    """Base class for all persistence errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "PERSISTENCE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidPersistenceConfiguration(PersistenceError):
    """Raised when persistence configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_PERSISTENCE_CONFIGURATION", details=details)


class PersistenceNotFound(PersistenceError):
    """Raised when a requested persistence backend is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Persistence backend {name!r} not found" if name else "Persistence backend not found"
        super().__init__(msg, code="PERSISTENCE_NOT_FOUND", details=details)
