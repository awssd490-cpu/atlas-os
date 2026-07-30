"""Vector store configuration.

All configuration objects are immutable frozen dataclasses, following the
convention established in ``app.rag.models``.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.vectorstore.metrics import SimilarityMetric


@dataclass(frozen=True)
class VectorStoreConfig:
    """Configuration for a vector store.

    Attributes:
        metric: The similarity metric to use for searches.
            Default ``SimilarityMetric.COSINE``.
        max_vectors: Maximum number of vectors the store can hold.
            ``0`` means unlimited.  Default 0.
        validate_dimensions: Whether to enforce that all vectors have
            the same dimensionality.  Default ``True``.
    """

    metric: SimilarityMetric = SimilarityMetric.COSINE
    max_vectors: int = 0
    validate_dimensions: bool = True

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidVectorStoreConfiguration: If any value is out of range.
        """
        from app.rag.vectorstore.errors import InvalidVectorStoreConfiguration

        if self.max_vectors < 0:
            raise InvalidVectorStoreConfiguration(
                "max_vectors must be non-negative",
                details={"max_vectors": self.max_vectors},
            )
