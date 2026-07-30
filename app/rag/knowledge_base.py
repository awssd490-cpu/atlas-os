"""KnowledgeBase — the primary knowledge storage.

Manages registration, removal, and lookup of ``KnowledgeDocument``
objects.  Does NOT perform retrieval or embeddings — that is the
responsibility of ``KnowledgeRetriever``.

The ``KnowledgeBase`` is the single source of truth for all registered
knowledge.  It supports thousands of documents with efficient lookups.
"""

from __future__ import annotations

from typing import Any

from app.rag.chunking import ChunkingConfig, ChunkingEngine
from app.rag.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
)
from app.rag.models import KnowledgeChunk, KnowledgeDocument


class KnowledgeBase:
    """Central registry for knowledge documents.

    Supports registration, removal, and enumeration of documents.
    Storage-agnostic — currently uses in-memory dicts.

    Adding a document via ``add_document()`` automatically chunks the
    content through the built-in ``ChunkingEngine``.  Callers that need
    full control over chunk boundaries can still use ``register()`` with
    pre-built chunks.

    Usage::

        kb = KnowledgeBase()
        doc = KnowledgeDocument(document_id="doc_1", title="Paris", content="...")
        kb.add_document(doc)
        found = kb.get("doc_1")
        kb.remove("doc_1")
    """

    def __init__(
        self,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self._documents: dict[str, KnowledgeDocument] = {}
        self._chunks: dict[str, KnowledgeChunk] = {}
        self._chunking_engine = ChunkingEngine(config=chunking_config)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """Register a document in the knowledge base.

        The document is stored as-is with its existing ``chunks`` tuple.
        No automatic chunking is performed — use ``add_document()``
        when you want the document content to be chunked automatically.

        Args:
            document: The document to register.

        Returns:
            The registered document.

        Raises:
            DuplicateDocumentError: If a document with the same ID is
                already registered.
        """
        self._raise_if_duplicate(document)
        self._store_document(document)
        return document

    def add_document(
        self,
        document: KnowledgeDocument,
        *,
        config: ChunkingConfig | None = None,
    ) -> KnowledgeDocument:
        """Add a document and automatically chunk its content.

        The document content is passed through the ``ChunkingEngine``
        which produces ``KnowledgeChunk`` objects according to the
        configured (or passed) strategy.  The resulting chunks are
        stored alongside the document.

        Args:
            document: The document to add.
            config: Optional per-call chunking configuration.  Falls
                back to the engine's default if omitted.

        Returns:
            The document with its ``chunks`` tuple populated.

        Raises:
            DuplicateDocumentError: If a document with the same ID is
                already registered.
        """
        self._raise_if_duplicate(document)

        result = self._chunking_engine.chunk(
            document.content,
            config=config,
            document_id=document.document_id,
        )

        chunked_doc = KnowledgeDocument(
            document_id=document.document_id,
            title=document.title,
            content=document.content,
            chunks=result.chunks,
            metadata=document.metadata,
        )

        self._store_document(chunked_doc)
        return chunked_doc

    def remove(self, document_id: str) -> bool:
        """Remove a document from the knowledge base.

        Args:
            document_id: The document ID to remove.

        Returns:
            ``True`` if removed, ``False`` if not found.
        """
        doc = self._documents.pop(document_id, None)
        if doc is None:
            return False

        # Remove associated chunks
        for chunk in doc.chunks:
            self._chunks.pop(chunk.chunk_id, None)

        return True

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, document_id: str) -> KnowledgeDocument | None:
        """Look up a document by ID.

        Args:
            document_id: The document identifier.

        Returns:
            The ``KnowledgeDocument`` or ``None``.
        """
        return self._documents.get(document_id)

    def exists(self, document_id: str) -> bool:
        """Check if a document is registered."""
        return document_id in self._documents

    def get_chunk(self, chunk_id: str) -> KnowledgeChunk | None:
        """Look up a single chunk by ID.

        Args:
            chunk_id: The chunk identifier.

        Returns:
            The ``KnowledgeChunk`` or ``None``.
        """
        return self._chunks.get(chunk_id)

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_documents(self) -> list[KnowledgeDocument]:
        """Return all registered documents."""
        return list(self._documents.values())

    def list_chunks(self) -> list[KnowledgeChunk]:
        """Return all registered chunks across all documents."""
        return list(self._chunks.values())

    def count(self) -> int:
        """Return the number of registered documents."""
        return len(self._documents)

    def clear(self) -> None:
        """Remove all documents and chunks."""
        self._documents.clear()
        self._chunks.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _raise_if_duplicate(self, document: KnowledgeDocument) -> None:
        if document.document_id in self._documents:
            raise DuplicateDocumentError(
                name=document.document_id,
                details={"title": document.title},
            )

    def _store_document(self, document: KnowledgeDocument) -> None:
        self._documents[document.document_id] = document
        for chunk in document.chunks:
            self._chunks[chunk.chunk_id] = chunk

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def chunking_config(self) -> ChunkingConfig:
        """Return the chunking configuration in use."""
        return self._chunking_engine.config
