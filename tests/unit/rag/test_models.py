"""Tests for RAG domain models."""

from __future__ import annotations

from app.rag.models import (
    KnowledgeChunk,
    KnowledgeContext,
    KnowledgeDocument,
    KnowledgeMetadata,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeSource,
)


class TestKnowledgeMetadata:
    def test_create(self) -> None:
        meta = KnowledgeMetadata(
            source="test.txt",
            author="tester",
            tags=("example",),
            category="reference",
        )
        assert meta.source == "test.txt"
        assert meta.author == "tester"


class TestKnowledgeChunk:
    def test_create(self) -> None:
        chunk = KnowledgeChunk(
            chunk_id="chunk_1",
            document_id="doc_1",
            content="Paris is the capital of France",
            index=0,
        )
        assert chunk.content == "Paris is the capital of France"
        assert chunk.index == 0


class TestKnowledgeDocument:
    def test_create(self) -> None:
        doc = KnowledgeDocument(
            document_id="doc_1",
            title="Paris",
            content="Paris is the capital...",
        )
        assert doc.document_id == "doc_1"
        assert doc.content_length == 23
        assert doc.chunk_count == 0

    def test_to_dict(self) -> None:
        doc = KnowledgeDocument(document_id="d1", title="Test")
        d = doc.to_dict()
        assert d["document_id"] == "d1"
        assert d["title"] == "Test"


class TestKnowledgeSource:
    def test_create(self) -> None:
        src = KnowledgeSource(
            document_id="doc_1",
            chunk_id="chunk_1",
            title="Paris",
            score=0.95,
        )
        assert src.score == 0.95


class TestKnowledgeQuery:
    def test_create(self) -> None:
        q = KnowledgeQuery(query="What is Paris?", max_results=5)
        assert q.query == "What is Paris?"
        assert q.max_results == 5

    def test_with_filters(self) -> None:
        q = KnowledgeQuery(query="test", filters={"category": "science"})
        assert q.filters["category"] == "science"


class TestKnowledgeResult:
    def test_empty(self) -> None:
        r = KnowledgeResult(query="test")
        assert r.total == 0
        assert r.chunks == []


class TestKnowledgeContext:
    def test_create(self) -> None:
        ctx = KnowledgeContext(
            text="Some knowledge",
            total_chunks=3,
        )
        assert ctx.text == "Some knowledge"
        assert ctx.total_chunks == 3

    def test_defaults(self) -> None:
        ctx = KnowledgeContext()
        assert ctx.text == ""
        assert ctx.total_chunks == 0
