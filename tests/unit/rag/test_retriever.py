"""Tests for KnowledgeRetriever."""

from __future__ import annotations

import pytest

from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeMetadata,
    KnowledgeQuery,
)
from app.rag.retriever import KnowledgeRetriever


class TestKnowledgeRetriever:
    @pytest.fixture
    def kb(self) -> KnowledgeBase:
        base = KnowledgeBase()

        paris = KnowledgeDocument(
            document_id="paris",
            title="Paris",
            chunks=(
                KnowledgeChunk(
                    chunk_id="p1",
                    document_id="paris",
                    content="Paris is the capital of France. It is known for the Eiffel Tower.",
                ),
            ),
        )
        base.register(paris)

        london = KnowledgeDocument(
            document_id="london",
            title="London",
            chunks=(
                KnowledgeChunk(
                    chunk_id="l1",
                    document_id="london",
                    content="London is the capital of the United Kingdom. It has Big Ben.",
                ),
            ),
        )
        base.register(london)

        python = KnowledgeDocument(
            document_id="python",
            title="Python",
            chunks=(
                KnowledgeChunk(
                    chunk_id="py1",
                    document_id="python",
                    content="Python is a programming language created by Guido van Rossum.",
                ),
            ),
        )
        base.register(python)

        return base

    async def test_retrieve_by_keyword(self, kb: KnowledgeBase) -> None:
        retriever = KnowledgeRetriever(kb)
        result = await retriever.retrieve(
            KnowledgeQuery(query="capital", max_results=5)
        )
        assert len(result.chunks) >= 2  # Paris + London
        assert result.total >= 2
        assert result.query == "capital"

    async def test_retrieve_limit(self, kb: KnowledgeBase) -> None:
        retriever = KnowledgeRetriever(kb)
        result = await retriever.retrieve(
            KnowledgeQuery(query="capital", max_results=1)
        )
        assert len(result.chunks) == 1

    async def test_retrieve_no_match(self, kb: KnowledgeBase) -> None:
        retriever = KnowledgeRetriever(kb)
        result = await retriever.retrieve(
            KnowledgeQuery(query="xyznonexistent", max_results=5)
        )
        assert len(result.chunks) == 0
        assert result.total == 0

    async def test_retrieve_min_score(self, kb: KnowledgeBase) -> None:
        retriever = KnowledgeRetriever(kb)
        result = await retriever.retrieve(
            KnowledgeQuery(query="capital", max_results=5, min_score=50.0)
        )
        assert len(result.chunks) == 0

    async def test_retrieve_empty_query(self, kb: KnowledgeBase) -> None:
        retriever = KnowledgeRetriever(kb)
        result = await retriever.retrieve(
            KnowledgeQuery(query="", max_results=5)
        )
        assert len(result.chunks) == 0

    async def test_sources_in_result(self, kb: KnowledgeBase) -> None:
        retriever = KnowledgeRetriever(kb)
        result = await retriever.retrieve(
            KnowledgeQuery(query="capital", max_results=5)
        )
        for source in result.sources:
            assert source.document_id in ("paris", "london", "python")
            assert source.chunk_id != ""

    async def test_empty_knowledge_base(self) -> None:
        retriever = KnowledgeRetriever(KnowledgeBase())
        result = await retriever.retrieve(
            KnowledgeQuery(query="anything", max_results=5)
        )
        assert len(result.chunks) == 0
