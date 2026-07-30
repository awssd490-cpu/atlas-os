"""Base abstractions for the vector store layer.

Defines the ``VectorStore`` abstract base class that all vector store
implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.rag.vectorstore.config import VectorStoreConfig
from app.rag.vectorstore.models import SearchResult


class VectorStore(ABC):
    """Abstract base class for vector stores.

    Every concrete store (in-memory, FAISS, pgvector, etc.) must subclass
    this and implement all abstract methods.
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        self._config = config or VectorStoreConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> VectorStoreConfig:
        """Return the store's configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    @abstractmethod
    def add(self, chunk_id: str, vector: tuple[float, ...]) -> None:
        """Add a vector to the store.

        If *chunk_id* already exists it is overwritten.

        Args:
            chunk_id: Unique identifier for this vector.
            vector: The embedding vector.

        Raises:
            VectorDimensionMismatchError: If dimension validation is
                enabled and *vector* has the wrong dimensionality.
            VectorStoreFullError: If the store is at capacity.
        """
        ...

    @abstractmethod
    def add_batch(self, items: Sequence[tuple[str, tuple[float, ...]]]) -> None:
        """Add multiple vectors in a single batch call.

        Args:
            items: A sequence of ``(chunk_id, vector)`` pairs.

        Raises:
            Same as ``add()``.  If any item fails the store is left
            in an undefined state.
        """
        ...

    @abstractmethod
    def remove(self, chunk_id: str) -> bool:
        """Remove a vector from the store.

        Args:
            chunk_id: The identifier of the vector to remove.

        Returns:
            ``True`` if the vector existed and was removed.
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all vectors from the store."""
        ...

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    @abstractmethod
    def get(self, chunk_id: str) -> tuple[float, ...] | None:
        """Retrieve a vector by its chunk ID.

        Args:
            chunk_id: The identifier.

        Returns:
            The vector, or ``None`` if not found.
        """
        ...

    @abstractmethod
    def contains(self, chunk_id: str) -> bool:
        """Check if a vector exists in the store."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of vectors in the store."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Search for the *top_k* nearest vectors.

        Args:
            query_vector: The query embedding vector.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``SearchResult`` sorted by descending similarity.
        """
        ...
