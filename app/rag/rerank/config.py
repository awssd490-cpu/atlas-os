"""Reranking configuration.

All configuration objects are immutable frozen dataclasses, following the
convention established in ``app.rag.models``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RerankConfig:
    """Configuration for a reranker.

    Attributes:
        enabled: Whether reranking is active.  Default ``True``.
        top_k: Maximum number of results to return after reranking.
            Must be > 0.  Default 10.
        score_threshold: Minimum ``final_score`` for a result to be
            included.  Must be in ``[0, 1]``.  Default 0.0 (no
            threshold).
    """

    enabled: bool = True
    top_k: int = 10
    score_threshold: float = 0.0

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidRerankConfiguration: If any value is out of range.
        """
        from app.rag.rerank.errors import InvalidRerankConfiguration

        if self.top_k < 1:
            raise InvalidRerankConfiguration(
                "top_k must be at least 1",
                details={"top_k": self.top_k},
            )
        if self.score_threshold < 0 or self.score_threshold > 1:
            raise InvalidRerankConfiguration(
                "score_threshold must be in [0, 1]",
                details={"score_threshold": self.score_threshold},
            )
