"""Hybrid retrieval domain models.

Every model in this module is immutable.  They represent the score
and result types for the hybrid retrieval layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RetrievalScore:
    """Aggregated scores for a single chunk from hybrid retrieval.

    Attributes:
        chunk_id: The identifier of the chunk.
        keyword_score: The score from keyword (lexical) retrieval.
            ``0.0`` if not available.
        semantic_score: The score from semantic (vector) retrieval.
            ``0.0`` if not available.
        final_score: The fused score combining keyword and semantic
            scores.  Interpretation depends on the fusion strategy.
    """

    chunk_id: str = ""
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    final_score: float = 0.0


@dataclass(frozen=True)
class HybridResult:
    """The result of a hybrid retrieval query.

    Attributes:
        results: The fused retrieval scores sorted by descending
            ``final_score``.
        metadata: Optional metadata about the retrieval (timing,
            candidate counts, fusion strategy used).
    """

    results: tuple[RetrievalScore, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
