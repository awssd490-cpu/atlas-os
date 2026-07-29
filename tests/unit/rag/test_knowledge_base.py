"""Tests for KnowledgeBase."""

from __future__ import annotations

import pytest

from app.rag.errors import DuplicateDocumentError
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeChunk, KnowledgeDocument, KnowledgeMetadata


class TestKnowledgeBase:
    def test_register(self) -> None:
        kb = KnowledgeBase()
        doc = KnowledgeDocument(document_id="doc_1", title="Paris", content="Paris is the capital")
        kb.register(doc)
        assert kb.count() == 1
        assert kb.exists("doc_1")

    def test_register_duplicate_raises(self) -> None:
        kb = KnowledgeBase()
        doc = KnowledgeDocument(document_id="doc_1")
        kb.register(doc)
        with pytest.raises(DuplicateDocumentError):
            kb.register(doc)

    def test_get(self) -> None:
        kb = KnowledgeBase()
        doc = KnowledgeDocument(document_id="d1", title="Test")
        kb.register(doc)
        assert kb.get("d1") is doc
        assert kb.get("missing") is None

    def test_remove(self) -> None:
        kb = KnowledgeBase()
        kb.register(KnowledgeDocument(document_id="d1"))
        assert kb.remove("d1") is True
        assert kb.count() == 0

    def test_remove_missing(self) -> None:
        kb = KnowledgeBase()
        assert kb.remove("missing") is False

    def test_list_documents(self) -> None:
        kb = KnowledgeBase()
        kb.register(KnowledgeDocument(document_id="a"))
        kb.register(KnowledgeDocument(document_id="b"))
        assert len(kb.list_documents()) == 2

    def test_clear(self) -> None:
        kb = KnowledgeBase()
        kb.register(KnowledgeDocument(document_id="a"))
        kb.register(KnowledgeDocument(document_id="b"))
        kb.clear()
        assert kb.count() == 0

    def test_register_with_chunks(self) -> None:
        kb = KnowledgeBase()
        doc = KnowledgeDocument(
            document_id="d1",
            chunks=(
                KnowledgeChunk(chunk_id="c1", document_id="d1", content="chunk 1"),
                KnowledgeChunk(chunk_id="c2", document_id="d1", content="chunk 2"),
            ),
        )
        kb.register(doc)
        assert kb.get_chunk("c1") is not None
        assert kb.get_chunk("c1").content == "chunk 1"
        assert len(kb.list_chunks()) == 2

    def test_remove_also_removes_chunks(self) -> None:
        kb = KnowledgeBase()
        doc = KnowledgeDocument(
            document_id="d1",
            chunks=(KnowledgeChunk(chunk_id="c1", document_id="d1", content="x"),),
        )
        kb.register(doc)
        kb.remove("d1")
        assert kb.get_chunk("c1") is None

    def test_empty_base(self) -> None:
        kb = KnowledgeBase()
        assert kb.count() == 0
        assert kb.list_documents() == []
        assert kb.list_chunks() == []
