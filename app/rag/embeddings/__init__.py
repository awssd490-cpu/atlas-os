"""Embeddings — vector embedding for the Knowledge Layer.

The embeddings sub-package provides a pluggable framework for generating
vector embeddings from text.  It is independent of ``KnowledgeBase``,
``ChunkingEngine``, and ``KnowledgeRetriever`` — embeddings are a
pre-processing step that happens before vector storage and search.
"""

from __future__ import annotations

from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.config import EmbeddingConfig
from app.rag.embeddings.errors import (
    EmbeddingError,
    EmbeddingProviderError,
    InvalidEmbeddingConfiguration,
    UnsupportedEmbeddingProvider,
)
from app.rag.embeddings.models import EmbeddingResult, EmbeddingVector
from app.rag.embeddings.registry import (
    clear_providers,
    get_provider,
    list_providers,
    register_provider,
)

__all__ = [
    "EmbeddingConfig",
    "EmbeddingError",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingResult",
    "EmbeddingVector",
    "InvalidEmbeddingConfiguration",
    "UnsupportedEmbeddingProvider",
    "clear_providers",
    "get_provider",
    "list_providers",
    "register_provider",
]
