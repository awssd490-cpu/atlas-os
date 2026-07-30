"""Embedding configuration.

All configuration objects are immutable frozen dataclasses, following the
convention established in ``app.rag.models``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for an embedding provider.

    Attributes:
        provider_name: The embedding provider identifier (e.g. ``"openai"``,
            ``"ollama"``).  Default ``"openai"``.
        dimensions: Dimensionality of the embedding vectors.  Must be >= 1.
            Default 768.
        batch_size: Maximum number of texts to embed in a single batch call.
            Must be >= 1.  Default 32.
        normalize_embeddings: Whether to L2-normalise output vectors.
            Default ``True``.
        timeout: Request timeout in seconds.  Must be > 0.  Default 30.0.
    """

    provider_name: str = "openai"
    dimensions: int = 768
    batch_size: int = 32
    normalize_embeddings: bool = True
    timeout: float = 30.0

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidEmbeddingConfiguration: If any value is out of range
                or invalid.
        """
        from app.rag.embeddings.errors import InvalidEmbeddingConfiguration

        if self.dimensions < 1:
            raise InvalidEmbeddingConfiguration(
                "dimensions must be at least 1",
                details={"dimensions": self.dimensions},
            )
        if self.batch_size < 1:
            raise InvalidEmbeddingConfiguration(
                "batch_size must be at least 1",
                details={"batch_size": self.batch_size},
            )
        if self.timeout <= 0:
            raise InvalidEmbeddingConfiguration(
                "timeout must be positive",
                details={"timeout": self.timeout},
            )
        if not self.provider_name:
            raise InvalidEmbeddingConfiguration(
                "provider_name must not be empty",
                details={"provider_name": self.provider_name},
            )
