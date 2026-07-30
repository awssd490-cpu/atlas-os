"""Reranking domain models.

Every model in this module is immutable.  They represent the score
and result types for the reranking layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RerankedResult:
    """A single reranked result.

    Attributes:
        chunk_id: The identifier of the chunk.
        original_score: The score before reranking (from the retrieval
            stage).  ``0.0`` if not available.
        rerank_score: The score produced by the reranker.
            ``0.0`` if not available.
        final_score: The combined or final score after reranking.
            How this is computed depends on the concrete reranker.
    """

    chunk_id: str = ""
    original_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0


@dataclass(frozen=True)
class RerankResponse:
    """The response from a reranker.

    Attributes:
        results: The reranked results sorted by descending
            ``final_score``.
        metadata: Optional metadata about the reranking (timing,
            model name, etc.).
    """

    results: tuple[RerankedResult, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
