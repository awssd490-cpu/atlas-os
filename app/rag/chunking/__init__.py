"""Chunking — document chunking for the Knowledge Layer.

The chunking sub-package provides a pluggable framework for splitting
knowledge documents into ``KnowledgeChunk`` objects.  It is independent
of ``KnowledgeBase`` and ``KnowledgeRetriever`` — chunking is a
pre-processing step that happens before storage.
"""

from __future__ import annotations

from app.rag.chunking.base import ChunkResult, ChunkingStrategy
from app.rag.chunking.chunker import ChunkingEngine
from app.rag.chunking.config import ChunkingConfig
from app.rag.chunking.errors import (
    ChunkingConfigError,
    ChunkingEngineError,
    ChunkingError,
    ChunkingStrategyError,
    UnsupportedStrategyError,
)
from app.rag.chunking.metadata import ChunkMetadata

__all__ = [
    "ChunkingConfig",
    "ChunkingEngine",
    "ChunkingError",
    "ChunkingConfigError",
    "ChunkingEngineError",
    "ChunkingStrategyError",
    "ChunkingStrategy",
    "ChunkMetadata",
    "ChunkResult",
    "UnsupportedStrategyError",
]
