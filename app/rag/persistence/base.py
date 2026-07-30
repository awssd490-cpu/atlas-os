"""Base abstractions for the persistence layer.

Defines the ``PersistenceBackend`` abstract base class that all
persistence implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.persistence.config import PersistenceConfig
from app.rag.persistence.models import PersistenceResult, PersistenceStats


class PersistenceBackend(ABC):
    """Abstract base class for persistence backends.

    A persistence backend serialises and deserialises the state of a
    knowledge pipeline (documents, chunks, embeddings, vectors) to
    and from durable storage.

    Concrete subclasses must implement ``save()``, ``load()``,
    ``exists()``, ``delete()``, and ``stats()``.
    """

    def __init__(self, config: PersistenceConfig | None = None) -> None:
        self._config = config or PersistenceConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> PersistenceConfig:
        """Return the backend's configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Persistence API
    # ------------------------------------------------------------------

    @abstractmethod
    async def save(
        self,
        path: str,
        data: object,
        **kwargs: object,
    ) -> PersistenceResult:
        """Persist data to durable storage.

        Args:
            path: Target path or resource identifier.
            data: The data to persist (typically a knowledge base
                snapshot or pipeline state).
            **kwargs: Implementation-specific options.

        Returns:
            A ``PersistenceResult`` indicating success and metadata.

        Raises:
            PersistenceError: On persistence failures.
        """
        ...

    @abstractmethod
    async def load(
        self,
        path: str,
        **kwargs: object,
    ) -> PersistenceResult:
        """Load data from durable storage.

        Args:
            path: Source path or resource identifier.
            **kwargs: Implementation-specific options.

        Returns:
            A ``PersistenceResult`` with loaded data and metadata.

        Raises:
            PersistenceError: On load failures.
        """
        ...

    @abstractmethod
    async def exists(
        self,
        path: str,
        **kwargs: object,
    ) -> bool:
        """Check whether persisted data exists at *path*.

        Args:
            path: Path or resource identifier to check.
            **kwargs: Implementation-specific options.

        Returns:
            ``True`` if data exists at *path*, ``False`` otherwise.

        Raises:
            PersistenceError: On backend failures.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        path: str,
        **kwargs: object,
    ) -> PersistenceResult:
        """Delete persisted data at *path*.

        Args:
            path: Path or resource identifier to delete.
            **kwargs: Implementation-specific options.

        Returns:
            A ``PersistenceResult`` indicating success and metadata.

        Raises:
            PersistenceError: On deletion failures.
        """
        ...

    @abstractmethod
    async def stats(
        self,
        path: str,
        **kwargs: object,
    ) -> PersistenceStats:
        """Return statistics about persisted data at *path*.

        Args:
            path: Path or resource identifier.
            **kwargs: Implementation-specific options.

        Returns:
            A ``PersistenceStats`` snapshot.

        Raises:
            PersistenceError: On backend failures.
        """
        ...
