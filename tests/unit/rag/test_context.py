"""Tests for KnowledgeContextBuilder."""

from __future__ import annotations

import pytest

from app.rag.context import KnowledgeContextBuilder
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import (
    KnowledgeChunk,
    KnowledgeDocument,
    KnowledgeMetadata,
)
from app.rag.retriever import KnowledgeRetriever


class TestKnowledgeContextBuilder:
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
                    content="Paris is the capital of France.",
                ),
            ),
        )
        base.register(paris)

        weather = KnowledgeDocument(
            document_id="weather",
            title="Weather",
            chunks=(
                KnowledgeChunk(
                    chunk_id="w1",
                    document_id="weather",
                    content="The weather in London is often rainy.",
                ),
            ),
        )
        base.register(weather)

        return base

    async def test_build_with_query(self, kb: KnowledgeBase) -> None:
        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="capital of France")
        assert context.total_chunks >= 1
        assert "Paris" in context.text

    async def test_build_no_query(self, kb: KnowledgeBase) -> None:
        builder = KnowledgeContextBuilder(kb)
        context = await builder.build()
        assert context.total_chunks == 0
        assert context.text == ""

    async def test_build_no_match(self, kb: KnowledgeBase) -> None:
        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="xyznothing")
        assert context.total_chunks == 0

    async def test_build_without_kb(self) -> None:
        builder = KnowledgeContextBuilder()
        context = await builder.build(query="test")
        assert context.total_chunks == 0

    async def test_build_max_chunks(self, kb: KnowledgeBase) -> None:
        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="the", max_chunks=1)
        assert context.total_chunks <= 1

    async def test_build_with_explicit_retriever(self, kb: KnowledgeBase) -> None:
        retriever = KnowledgeRetriever(kb)
        builder = KnowledgeContextBuilder(kb, retriever=retriever)
        context = await builder.build(query="capital")
        assert context.total_chunks >= 1
        assert "France" in context.text or "Paris" in context.text

    async def test_sources_in_context(self, kb: KnowledgeBase) -> None:
        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="weather")
        assert len(context.sources) >= 1
        assert context.sources[0].document_id == "weather"
