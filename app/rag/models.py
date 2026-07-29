"""RAG domain models.

Every model in this module is immutable and provider-independent.
They represent the canonical data types for the Knowledge Layer.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class KnowledgeMetadata:
    """Metadata attached to a knowledge document or chunk.

    Attributes:
        source: Original source identifier (e.g. filename, URL).
        author: Creator or maintainer.
        version: Document version string.
        tags: Categorisation tags.
        category: Document category.
        language: Document language code.
        timestamp: When the document was created/added.
    """

    source: str = ""
    author: str = ""
    version: str = "1.0"
    tags: tuple[str, ...] = ()
    category: str = ""
    language: str = ""
    timestamp: float = 0.0


@dataclass(frozen=True)
class KnowledgeChunk:
    """A single chunk of a knowledge document.

    Documents are split into chunks for targeted retrieval.
    Each chunk carries its own metadata and provenance.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        document_id: ID of the parent document.
        content: The chunk text content.
        index: Position within the parent document (0-based).
        metadata: Chunk-level metadata (may override document metadata).
    """

    chunk_id: str = ""
    document_id: str = ""
    content: str = ""
    index: int = 0
    metadata: KnowledgeMetadata = field(default_factory=KnowledgeMetadata)


@dataclass(frozen=True)
class KnowledgeDocument:
    """A complete knowledge document.

    The fundamental unit of knowledge storage.  Documents are registered
    with a ``KnowledgeBase`` and retrieved by ``KnowledgeRetriever``.

    Attributes:
        document_id: Unique identifier.
        title: Document title.
        content: Full document content.
        chunks: Pre-computed chunks (if any).
        metadata: Document-level metadata.
    """

    document_id: str = ""
    title: str = ""
    content: str = ""
    chunks: tuple[KnowledgeChunk, ...] = ()
    metadata: KnowledgeMetadata = field(default_factory=KnowledgeMetadata)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def content_length(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "content_length": self.content_length,
            "chunk_count": self.chunk_count,
        }


@dataclass(frozen=True)
class KnowledgeSource:
    """Provenance — where a piece of knowledge came from.

    Preserved through retrieval so the application can cite sources.
    """

    document_id: str = ""
    chunk_id: str = ""
    title: str = ""
    score: float = 0.0


@dataclass(frozen=True)
class KnowledgeQuery:
    """A query against the knowledge base.

    Attributes:
        query: The search text.
        max_results: Maximum results to return.
        min_score: Minimum relevance score threshold.
        filters: Optional metadata filters.
    """

    query: str = ""
    max_results: int = 10
    min_score: float = 0.0
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class KnowledgeResult:
    """The result of a knowledge query.

    Attributes:
        chunks: The matching knowledge chunks.
        sources: Source provenance for each chunk.
        query: The original query.
        total: Total matching chunks found.
        elapsed_ms: Query execution time.
    """

    chunks: list[KnowledgeChunk] = field(default_factory=list)
    sources: list[KnowledgeSource] = field(default_factory=list)
    query: str = ""
    total: int = 0
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class KnowledgeContext:
    """Merged knowledge context ready for provider injection.

    Attributes:
        text: Formatted text for injection.
        chunks: The chunks that contributed.
        sources: Provenance information.
        total_chunks: Total chunks retrieved.
    """

    text: str = ""
    chunks: list[KnowledgeChunk] = field(default_factory=list)
    sources: list[KnowledgeSource] = field(default_factory=list)
    total_chunks: int = 0
