"""Hybrid retrieval configuration.

All configuration objects are immutable frozen dataclasses, following the
convention established in ``app.rag.models``.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.hybrid.fusion import FusionStrategy


@dataclass(frozen=True)
class HybridConfig:
    """Configuration for hybrid retrieval.

    Attributes:
        keyword_weight: Weight for keyword (lexical) scores in fusion.
            Must be >= 0.  Default 0.5.
        semantic_weight: Weight for semantic (vector) scores in fusion.
            Must be >= 0.  Default 0.5.
        max_candidates: Maximum number of candidate chunks from each
            retrieval arm that enter fusion.  Must be > 0.  Default 20.
        fusion_strategy: The strategy used to combine scores.
            Default ``FusionStrategy.WEIGHTED_SUM``.
    """

    keyword_weight: float = 0.5
    semantic_weight: float = 0.5
    max_candidates: int = 20
    fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED_SUM

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidHybridConfiguration: If any value is out of range.
        """
        from app.rag.hybrid.errors import InvalidHybridConfiguration

        if self.keyword_weight < 0:
            raise InvalidHybridConfiguration(
                "keyword_weight must be non-negative",
                details={"keyword_weight": self.keyword_weight},
            )
        if self.semantic_weight < 0:
            raise InvalidHybridConfiguration(
                "semantic_weight must be non-negative",
                details={"semantic_weight": self.semantic_weight},
            )
        if self.keyword_weight + self.semantic_weight <= 0:
            raise InvalidHybridConfiguration(
                "keyword_weight and semantic_weight must sum to more than 0",
                details={
                    "keyword_weight": self.keyword_weight,
                    "semantic_weight": self.semantic_weight,
                },
            )
        if self.max_candidates < 1:
            raise InvalidHybridConfiguration(
                "max_candidates must be at least 1",
                details={"max_candidates": self.max_candidates},
            )
