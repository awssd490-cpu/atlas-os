"""Tests for KnowledgeBase — including chunking integration."""

from __future__ import annotations

import pytest

from app.rag.chunking import (
    ChunkingConfig,
    ChunkingEngine,
    STRATEGY_FIXED_SIZE,
    STRATEGY_WHOLE_DOCUMENT,
    STRATEGY_SENTENCE,
    STRATEGY_PARAGRAPH,
    STRATEGY_SLIDING_WINDOW,
)
from app.rag.errors import DuplicateDocumentError
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeChunk, KnowledgeDocument, KnowledgeMetadata


class TestKnowledgeBase:
    """Original tests — every one must still pass unchanged."""

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


# ======================================================================
# add_document() — new integration tests
# ======================================================================


class TestAddDocument:
    def test_add_document_auto_chunks(self) -> None:
        """add_document() automatically chunks content."""
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
        )
        text = "A" * 25
        doc = KnowledgeDocument(document_id="d1", title="Test", content=text)
        stored = kb.add_document(doc)

        # Document is stored
        assert kb.count() == 1
        assert kb.exists("d1")

        # Chunks were generated — 3 chunks of 10, 10, 5
        assert stored.chunk_count >= 2
        assert len(stored.chunks) >= 2

        # Chunks are indexed in the KB
        assert len(kb.list_chunks()) >= 2

    def test_add_document_whole_document(self) -> None:
        """whole_document strategy produces exactly one chunk."""
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
        )
        doc = KnowledgeDocument(
            document_id="d1",
            title="Paris",
            content="Paris is the capital of France.",
        )
        stored = kb.add_document(doc)
        assert stored.chunk_count == 1
        assert stored.chunks[0].content == "Paris is the capital of France."

    def test_add_document_fixed_size(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
        )
        doc = KnowledgeDocument(
            document_id="d1",
            title="Test",
            content="Hello World! This is a longer text.",
        )
        stored = kb.add_document(doc)
        assert stored.chunk_count > 1
        for chunk in stored.chunks:
            assert chunk.document_id == "d1"
            assert chunk.chunk_id.startswith("fixed_size:")

    def test_add_document_sentence(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_SENTENCE),
        )
        doc = KnowledgeDocument(
            document_id="d1",
            title="Sentences",
            content="First sentence. Second sentence. Third!",
        )
        stored = kb.add_document(doc)
        assert stored.chunk_count == 3

    def test_add_document_paragraph(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_PARAGRAPH),
        )
        doc = KnowledgeDocument(
            document_id="d1",
            title="Paragraphs",
            content="Para A.\n\nPara B.\n\nPara C.",
        )
        stored = kb.add_document(doc)
        assert stored.chunk_count == 3

    def test_add_document_sliding_window(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_SLIDING_WINDOW,
                window_size=10,
                stride=5,
                min_chunk_size=1,
            ),
        )
        doc = KnowledgeDocument(
            document_id="d1",
            title="Sliding",
            content="A" * 30,
        )
        stored = kb.add_document(doc)
        assert stored.chunk_count >= 4

    def test_add_document_preserves_document_id(self) -> None:
        kb = KnowledgeBase()
        doc = KnowledgeDocument(document_id="doc_42", content="Hello world.")
        stored = kb.add_document(doc)
        assert stored.document_id == "doc_42"
        for chunk in stored.chunks:
            assert chunk.document_id == "doc_42"

    def test_add_document_preserves_metadata(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
        )
        meta = KnowledgeMetadata(source="test.txt", category="reference")
        doc = KnowledgeDocument(
            document_id="d1",
            title="Test",
            content="Hello world.",
            metadata=meta,
        )
        stored = kb.add_document(doc)
        assert stored.metadata.source == "test.txt"
        assert stored.metadata.category == "reference"

    def test_add_document_empty_content(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
        )
        doc = KnowledgeDocument(document_id="d1", content="")
        stored = kb.add_document(doc)
        assert stored.chunk_count == 0

    def test_add_document_duplicate_raises(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
        )
        doc = KnowledgeDocument(document_id="d1", content="Some text.")
        kb.add_document(doc)
        with pytest.raises(DuplicateDocumentError):
            kb.add_document(doc)

    def test_add_document_deterministic_chunk_ids(self) -> None:
        """Calling add_document twice produces identical chunks."""
        kb1 = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
        )
        kb2 = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
        )
        text = "Hello World! This is stable content."
        doc1 = KnowledgeDocument(document_id="d1", content=text)
        doc2 = KnowledgeDocument(document_id="d1", content=text)

        stored1 = kb1.add_document(doc1)
        stored2 = kb2.add_document(doc2)

        for c1, c2 in zip(stored1.chunks, stored2.chunks):
            assert c1.chunk_id == c2.chunk_id
            assert c1.content == c2.content
            assert c1.index == c2.index

    def test_add_document_per_call_config(self) -> None:
        """Per-call config overrides the KB default."""
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
        )
        doc = KnowledgeDocument(
            document_id="d1",
            content="First. Second. Third!",
        )
        config = ChunkingConfig(strategy=STRATEGY_SENTENCE)
        stored = kb.add_document(doc, config=config)
        assert stored.chunk_count == 3  # sentence split, not whole doc

    def test_add_document_unicode(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
        )
        text = "Hello 世界! Unicode ✅"
        doc = KnowledgeDocument(document_id="d1", content=text)
        stored = kb.add_document(doc)
        assert stored.chunk_count >= 1
        # Verify the chunks are stored and retrievable
        total_chunks = len(kb.list_chunks())
        assert total_chunks == stored.chunk_count

    def test_add_document_then_remove(self) -> None:
        """Chunks are removed when the document is removed."""
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
        )
        doc = KnowledgeDocument(document_id="d1", content="A" * 30)
        kb.add_document(doc)
        assert len(kb.list_chunks()) == 3

        kb.remove("d1")
        assert kb.count() == 0
        assert len(kb.list_chunks()) == 0

    def test_add_document_multiple_documents(self) -> None:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
        )
        doc1 = KnowledgeDocument(document_id="d1", content="A" * 30)
        doc2 = KnowledgeDocument(document_id="d2", content="B" * 20)
        kb.add_document(doc1)
        kb.add_document(doc2)

        assert kb.count() == 2
        assert len(kb.list_chunks()) == 5  # 3 + 2

        d1 = kb.get("d1")
        d2 = kb.get("d2")
        assert d1 is not None and d1.chunk_count == 3
        assert d2 is not None and d2.chunk_count == 2


# ======================================================================
# Retrieval compatibility
# ======================================================================


class TestRetrievalCompatibility:
    """The retriever must work with auto-chunked documents."""

    @pytest.fixture
    def kb_with_auto_chunks(self) -> KnowledgeBase:
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_SENTENCE,
            ),
        )

        paris = KnowledgeDocument(
            document_id="paris",
            title="Paris",
            content="Paris is the capital of France. It is known for the Eiffel Tower.",
        )
        kb.add_document(paris)

        london = KnowledgeDocument(
            document_id="london",
            title="London",
            content="London is the capital of the UK. Big Ben is a famous landmark.",
        )
        kb.add_document(london)

        return kb

    @pytest.fixture
    def retriever(self, kb_with_auto_chunks: KnowledgeBase):
        from app.rag.retriever import KnowledgeRetriever
        return KnowledgeRetriever(kb_with_auto_chunks)

    async def test_retrieve_after_add_document(self, retriever) -> None:
        from app.rag.models import KnowledgeQuery
        result = await retriever.retrieve(
            KnowledgeQuery(query="capital", max_results=5)
        )
        assert len(result.chunks) >= 2
        assert result.total >= 2

    async def test_retrieve_limit(self, retriever) -> None:
        from app.rag.models import KnowledgeQuery
        result = await retriever.retrieve(
            KnowledgeQuery(query="capital", max_results=1)
        )
        assert len(result.chunks) == 1

    async def test_retrieve_no_match(self, retriever) -> None:
        from app.rag.models import KnowledgeQuery
        result = await retriever.retrieve(
            KnowledgeQuery(query="xyznonexistent", max_results=5)
        )
        assert len(result.chunks) == 0

    async def test_retrieve_sources(self, retriever) -> None:
        from app.rag.models import KnowledgeQuery
        result = await retriever.retrieve(
            KnowledgeQuery(query="capital", max_results=5)
        )
        for source in result.sources:
            assert source.document_id in ("paris", "london")
            assert source.chunk_id != ""

    async def test_retrieve_single_chunk(self, retriever) -> None:
        """A single sentence should match only one chunk."""
        from app.rag.models import KnowledgeQuery
        result = await retriever.retrieve(
            KnowledgeQuery(query="Eiffel Tower", max_results=5)
        )
        assert len(result.chunks) == 1
        assert "Eiffel Tower" in result.chunks[0].content

    async def test_retrieve_multiple_docs(self, retriever) -> None:
        from app.rag.models import KnowledgeQuery
        result = await retriever.retrieve(
            KnowledgeQuery(query="landmark", max_results=5)
        )
        assert len(result.chunks) >= 1


# ======================================================================
# Chunking config / property
# ======================================================================


class TestKnowledgeBaseChunking:
    def test_default_chunking_config(self) -> None:
        kb = KnowledgeBase()
        assert kb.chunking_config.strategy == "fixed_size"
        assert kb.chunking_config.chunk_size == 512

    def test_custom_chunking_config(self) -> None:
        config = ChunkingConfig(
            strategy=STRATEGY_SENTENCE,
            chunk_size=256,
            chunk_overlap=32,
        )
        kb = KnowledgeBase(chunking_config=config)
        assert kb.chunking_config.strategy == "sentence"
        assert kb.chunking_config.chunk_size == 256
        assert kb.chunking_config.chunk_overlap == 32

    def test_chunking_config_not_required(self) -> None:
        """KnowledgeBase() works without explicit chunking config."""
        kb = KnowledgeBase()
        assert kb.chunking_config is not None

    def test_register_still_works(self) -> None:
        """register() with pre-built chunks is unaffected."""
        kb = KnowledgeBase()
        doc = KnowledgeDocument(
            document_id="d1",
            chunks=(KnowledgeChunk(chunk_id="c1", document_id="d1", content="x"),),
        )
        kb.register(doc)
        assert len(kb.list_chunks()) == 1
