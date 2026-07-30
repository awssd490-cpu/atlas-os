"""Reranking — result reranking for the Knowledge Layer.

Provides a pluggable reranker abstraction that re-orders retrieval
results using a secondary scoring model.  It is independent of the
concrete retrieval and fusion layers.
"""

from __future__ import annotations

from app.rag.rerank.base import Reranker
from app.rag.rerank.config import RerankConfig
from app.rag.rerank.errors import (
    InvalidRerankConfiguration,
    RerankError,
    RerankerNotFound,
)
from app.rag.rerank.models import RerankResponse, RerankedResult
from app.rag.rerank.registry import (
    clear_rerankers,
    get_reranker,
    list_rerankers,
    register_reranker,
)

__all__ = [
    "InvalidRerankConfiguration",
    "RerankConfig",
    "RerankError",
    "RerankResponse",
    "RerankedResult",
    "Reranker",
    "RerankerNotFound",
    "clear_rerankers",
    "get_reranker",
    "list_rerankers",
    "register_reranker",
]
