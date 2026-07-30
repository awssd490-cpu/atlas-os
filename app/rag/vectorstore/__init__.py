"""Vector Store — vector similarity search for the Knowledge Layer.

Provides a pluggable vector store abstraction backed by an in-memory
implementation.  Supports cosine similarity, dot product, and
negative Euclidean distance for nearest-neighbour search.
"""

from __future__ import annotations

from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.config import VectorStoreConfig
from app.rag.vectorstore.errors import (
    InvalidVectorStoreConfiguration,
    VectorDimensionMismatchError,
    VectorNotFoundError,
    VectorStoreError,
    VectorStoreFullError,
)
from app.rag.vectorstore.memory import MemoryVectorStore
from app.rag.vectorstore.metrics import SimilarityMetric, compute_similarity
from app.rag.vectorstore.models import SearchResult

__all__ = [
    "InvalidVectorStoreConfiguration",
    "MemoryVectorStore",
    "SearchResult",
    "SimilarityMetric",
    "VectorDimensionMismatchError",
    "VectorNotFoundError",
    "VectorStore",
    "VectorStoreConfig",
    "VectorStoreError",
    "VectorStoreFullError",
    "compute_similarity",
]
