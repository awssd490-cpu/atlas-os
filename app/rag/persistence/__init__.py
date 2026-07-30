"""Persistence — durable storage for the Knowledge Layer.

The persistence sub-package provides a pluggable framework for
serialising and deserialising knowledge pipeline state (documents,
chunks, embeddings, vectors) to and from durable storage.  It is
independent of the concrete knowledge base, embedding, vector store,
and reranking layers — backends compose those components into a
coherent save/load lifecycle.
"""

from __future__ import annotations

from app.rag.persistence.base import PersistenceBackend
from app.rag.persistence.config import PersistenceConfig
from app.rag.persistence.errors import (
    InvalidPersistenceConfiguration,
    PersistenceError,
    PersistenceNotFound,
)
from app.rag.persistence.models import PersistenceResult, PersistenceStats
from app.rag.persistence.registry import (
    clear_backends,
    get,
    list_backends,
    register,
    unregister,
)

__all__ = [
    "InvalidPersistenceConfiguration",
    "PersistenceBackend",
    "PersistenceConfig",
    "PersistenceError",
    "PersistenceNotFound",
    "PersistenceResult",
    "PersistenceStats",
    "clear_backends",
    "get",
    "list_backends",
    "register",
    "unregister",
]
