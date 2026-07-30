"""Persistence domain models.

Every model in this module is immutable.  They represent the data types
for the persistence layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PersistenceStats:
    """Statistics for a persistence backend.

    Attributes:
        documents: Total number of documents persisted.
        chunks: Total number of chunks across all documents.
        embeddings: Total number of embedding vectors persisted.
        vectors: Total number of vector-store entries persisted.
        size_bytes: Total size of persisted data in bytes.
    """

    documents: int = 0
    chunks: int = 0
    embeddings: int = 0
    vectors: int = 0
    size_bytes: int = 0


@dataclass(frozen=True)
class PersistenceResult:
    """The result of a persistence operation.

    Attributes:
        success: Whether the operation completed successfully.
        metadata: Optional metadata about the operation (path,
            elapsed time, byte count, etc.).
    """

    success: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
