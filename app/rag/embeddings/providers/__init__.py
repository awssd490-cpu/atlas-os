"""Embedding provider implementations.

Built-in providers are automatically registered with the global
registry when this module is imported.
"""

from __future__ import annotations

from app.rag.embeddings.providers.deterministic import DeterministicEmbeddingProvider
from app.rag.embeddings.providers.mock import MockEmbeddingProvider
from app.rag.embeddings.registry import register_provider

# Auto-register both providers at import time
register_provider("deterministic", DeterministicEmbeddingProvider)
register_provider("mock", MockEmbeddingProvider)

__all__ = [
    "DeterministicEmbeddingProvider",
    "MockEmbeddingProvider",
]
