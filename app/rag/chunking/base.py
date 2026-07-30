"""Base abstractions for the chunking layer.

Defines ``ChunkResult`` and the abstract ``ChunkingStrategy`` protocol
that all strategy implementations follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.rag.chunking.config import ChunkingConfig
from app.rag.chunking.metadata import ChunkMetadata
from app.rag.models import KnowledgeChunk


@dataclass(frozen=True)
class ChunkResult:
    """The result of chunking a single document or text.

    Attributes:
        chunks: The list of knowledge chunks produced.
        metadata: Per-chunk metadata, parallel to ``chunks``.
        config: The configuration used for chunking.
        total_chunks: Total number of chunks produced.
        original_length: Length in characters of the source text.
    """

    chunks: tuple[KnowledgeChunk, ...] = ()
    metadata: tuple[ChunkMetadata, ...] = ()
    config: ChunkingConfig = field(default_factory=ChunkingConfig)
    total_chunks: int = 0
    original_length: int = 0


class ChunkingStrategy(Protocol):
    """Protocol that every chunking strategy must satisfy.

    A strategy is a callable that accepts a text string and a
    ``ChunkingConfig``, returning a ``ChunkResult``.
    """

    __name__: str

    def __call__(
        self,
        text: str,
        config: ChunkingConfig,
    ) -> ChunkResult:
        """Chunk *text* according to *config*.

        Args:
            text: The document text to chunk.
            config: Configuration controlling chunk size, overlap, etc.

        Returns:
            A ``ChunkResult`` containing the produced chunks and metadata.
        """
        ...
