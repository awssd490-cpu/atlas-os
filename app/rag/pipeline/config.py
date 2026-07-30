"""Pipeline configuration.

All configuration objects are immutable frozen dataclasses, following the
convention established in ``app.rag.models``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for a knowledge pipeline.

    Attributes:
        auto_embed: Whether to automatically embed documents during
            ingestion.  Default ``True``.
        auto_index: Whether to automatically index documents into the
            vector store during ingestion.  Default ``True``.
        auto_rerank: Whether to automatically rerank search results.
            Default ``True``.
        batch_size: Maximum number of items to process in a single
            batch during ingestion.  Must be > 0.  Default 32.
    """

    auto_embed: bool = True
    auto_index: bool = True
    auto_rerank: bool = True
    batch_size: int = 32

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidPipelineConfiguration: If any value is out of range
                or invalid.
        """
        from app.rag.pipeline.errors import InvalidPipelineConfiguration

        if self.batch_size < 1:
            raise InvalidPipelineConfiguration(
                "batch_size must be at least 1",
                details={"batch_size": self.batch_size},
            )
