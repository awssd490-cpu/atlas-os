"""Fusion strategies for hybrid retrieval.

Provides the ``FusionStrategy`` enum and function implementations
for combining keyword and semantic retrieval scores.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence


class FusionStrategy(enum.Enum):
    """Supported fusion strategies for combining retrieval scores.

    Each strategy defines how keyword and semantic scores are merged
    into a single final score.
    """

    WEIGHTED_SUM = "weighted_sum"
    RECIPROCAL_RANK_FUSION = "reciprocal_rank_fusion"


def weighted_sum(
    keyword_scores: Mapping[str, float],
    semantic_scores: Mapping[str, float],
    keyword_weight: float = 0.5,
    semantic_weight: float = 0.5,
) -> dict[str, float]:
    """Combine scores via weighted sum.

    For every chunk ID present in either set of scores, the final score
    is ``keyword_weight * keyword_score + semantic_weight * semantic_score``.
    Missing scores are treated as ``0.0``.

    Args:
        keyword_scores: Map of chunk_id → keyword score.
        semantic_scores: Map of chunk_id → semantic score.
        keyword_weight: Weight for keyword scores.
        semantic_weight: Weight for semantic scores.

    Returns:
        A dict mapping chunk_id → fused score.

    Raises:
        FusionError: If both weights are zero.
    """
    from app.rag.hybrid.errors import FusionError

    if keyword_weight == 0.0 and semantic_weight == 0.0:
        raise FusionError(
            "At least one weight must be non-zero for weighted sum",
            details={"keyword_weight": keyword_weight, "semantic_weight": semantic_weight},
        )

    all_ids = set(keyword_scores) | set(semantic_scores)
    fused: dict[str, float] = {}
    for cid in all_ids:
        ks = keyword_scores.get(cid, 0.0)
        ss = semantic_scores.get(cid, 0.0)
        fused[cid] = keyword_weight * ks + semantic_weight * ss
    return fused


def reciprocal_rank_fusion(
    keyword_ranked: Sequence[str],
    semantic_ranked: Sequence[str],
    k: int = 60,
) -> dict[str, float]:
    """Combine rankings via reciprocal rank fusion (RRF).

    For every unique chunk across both ranked lists, the fused score
    is ``1 / (k + rank_keyword) + 1 / (k + rank_semantic)``.
    Ranks are 1-indexed.  Items not present in a list receive the
    effective rank ``len(list) + 1``.

    Args:
        keyword_ranked: Chunk IDs ranked by keyword score (best first).
        semantic_ranked: Chunk IDs ranked by semantic score (best first).
        k: The RRF constant (default 60).

    Returns:
        A dict mapping chunk_id → fused RRF score.
    """
    all_ids = set(keyword_ranked) | set(semantic_ranked)
    kw_len = len(keyword_ranked)
    sem_len = len(semantic_ranked)

    kw_rank = {cid: i + 1 for i, cid in enumerate(keyword_ranked)}
    sem_rank = {cid: i + 1 for i, cid in enumerate(semantic_ranked)}

    fused: dict[str, float] = {}
    for cid in all_ids:
        kr = kw_rank.get(cid, kw_len + 1)
        sr = sem_rank.get(cid, sem_len + 1)
        fused[cid] = 1.0 / (k + kr) + 1.0 / (k + sr)
    return fused
