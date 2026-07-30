"""Hybrid Retrieval — fused keyword + semantic retrieval for the Knowledge Layer.

Provides a pluggable hybrid retriever abstraction with configurable
fusion strategies (weighted sum and reciprocal rank fusion).  It is
independent of the concrete keyword and vector retrievers — those are
injected by concrete subclasses.
"""

from __future__ import annotations

from app.rag.hybrid.base import HybridRetriever
from app.rag.hybrid.config import HybridConfig
from app.rag.hybrid.errors import (
    FusionError,
    HybridError,
    InvalidHybridConfiguration,
)
from app.rag.hybrid.fusion import FusionStrategy, reciprocal_rank_fusion, weighted_sum
from app.rag.hybrid.models import HybridResult, RetrievalScore

__all__ = [
    "FusionError",
    "FusionStrategy",
    "HybridConfig",
    "HybridError",
    "HybridResult",
    "HybridRetriever",
    "InvalidHybridConfiguration",
    "RetrievalScore",
    "reciprocal_rank_fusion",
    "weighted_sum",
]
