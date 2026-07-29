"""KnowledgeBase — the primary knowledge storage.

Manages registration, removal, and lookup of ``KnowledgeDocument``
objects.  Does NOT perform retrieval or embeddings — that is the
responsibility of ``KnowledgeRetriever``.

The ``KnowledgeBase`` is the single source of truth for all registered
knowledge.  It supports thousands of documents with efficient lookups.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import (
    DocumentNotFoundError,
    DuplicateDocumentError,
)
from app.rag.models import KnowledgeChunk, KnowledgeDocument


class KnowledgeBase:
    """Central registry for knowledge documents.

    Supports registration, removal, and enumeration of documents.
    Storage-agnostic — currently uses in-memory dicts.

    Usage::

        kb = KnowledgeBase()
        doc = KnowledgeDocument(document_id="doc_1", title="Paris", content="...")
        kb.register(doc)
        found = kb.get("doc_1")
        kb.remove("doc_1")
    """

    def __init__(self) -> None:
        self._documents: dict[str, KnowledgeDocument] = {}
        self._chunks: dict[str, KnowledgeChunk] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, document: KnowledgeDocument) -> KnowledgeDocument:
        """Register a document in the knowledge base.

        Args:
            document: The document to register.

        Returns:
            The registered document.

        Raises:
            DuplicateDocumentError: If a document with the same ID is
                already registered.
        """
        if document.document_id in self._documents:
            raise DuplicateDocumentError(
                name=document.document_id,
                details={"title": document.title},
            )

        self._documents[document.document_id] = document

        # Index chunks
        for chunk in document.chunks:
            self._chunks[chunk.chunk_id] = chunk

        return document

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
