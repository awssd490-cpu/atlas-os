"""Persistence configuration.

All configuration objects are immutable frozen dataclasses, following the
convention established in ``app.rag.models``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PersistenceConfig:
    """Configuration for a persistence backend.

    Attributes:
        compress: Whether to compress serialised data.
            Default ``True``.
        overwrite: Whether to overwrite an existing persistence
            target.  Default ``False``.
        include_embeddings: Whether to include embedding vectors
            in the persisted output.  Default ``True``.
        include_vectors: Whether to include vector-store entries
            in the persisted output.  Default ``True``.
    """

    compress: bool = True
    overwrite: bool = False
    include_embeddings: bool = True
    include_vectors: bool = True

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidPersistenceConfiguration: If any value is out of range
                or invalid.
        """
        from app.rag.persistence.errors import InvalidPersistenceConfiguration

        # All fields currently have valid ranges; this method exists
        # for forward compatibility and consistency with other subsystems.
        pass
