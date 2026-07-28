"""Storage-layer exception hierarchy.

Every storage-specific error derives from :class:`StorageError`, which
itself derives from :class:`AtlasError` so callers can catch either layer.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class StorageError(AtlasError):
    """Base class for all storage errors."""

    _default_code: str = "STORAGE_ERROR"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code or self._default_code, details=details)


class ConnectionError_(StorageError):
    """Raised when a storage connection cannot be established or is lost.

    Note: trailing underscore avoids shadowing the built-in ``ConnectionError``.
    """

    _default_code: str = "CONNECTION_ERROR"


class MigrationError(StorageError):
    """Raised when a migration cannot be applied or rolled back."""

    _default_code: str = "MIGRATION_ERROR"


class RecordNotFoundError(StorageError):
    """Raised when a requested record is not found."""

    _default_code: str = "RECORD_NOT_FOUND"


class VersionConflictError(StorageError):
    """Raised when an optimistic-lock version check fails."""

    _default_code: str = "VERSION_CONFLICT"


class CacheError(StorageError):
    """Raised on cache backend failures (not cache misses)."""

    _default_code: str = "CACHE_ERROR"


class TransactionError(StorageError):
    """Raised when a transaction operation fails."""

    _default_code: str = "TRANSACTION_ERROR"
