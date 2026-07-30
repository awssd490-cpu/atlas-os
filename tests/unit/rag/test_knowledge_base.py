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
from app.rag.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingConfig as EmbeddingConfig_Impl,
    MockEmbeddingProvider,
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


# ======================================================================
# Embedding integration
# ======================================================================


class TestKnowledgeBaseEmbeddings:
    """Tests for the optional embedding integration in KnowledgeBase."""

    def test_no_embedding_provider_by_default(self) -> None:
        """KnowledgeBase without embedding provider — no embeddings stored."""
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
        )
        assert kb.embedding_provider is None
        doc = KnowledgeDocument(document_id="d1", content="Hello world.")
        kb.add_document(doc)
        assert len(kb.list_embeddings()) == 0
        assert kb.get_embedding("any_id") is None

    def test_with_deterministic_provider(self) -> None:
        """Embeddings are generated when a deterministic provider is configured."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=4,
            normalize_embeddings=False,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
            embedding_provider=provider,
        )

        doc = KnowledgeDocument(document_id="d1", content="A" * 25)
        kb.add_document(doc)

        # Embeddings should exist for all chunks
        assert len(kb.list_embeddings()) >= 2
        embeddings = kb.list_embeddings()
        for vec in embeddings:
            assert len(vec.vector) == 4
            assert vec.provider == "deterministic"

    def test_embedding_lookup_by_chunk_id(self) -> None:
        """Look up an embedding by chunk ID."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=3,
            normalize_embeddings=False,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
            embedding_provider=provider,
        )

        doc = KnowledgeDocument(document_id="d1", content="A" * 30)
        kb.add_document(doc)

        # Get first chunk's embedding
        first_chunk = list(kb.list_chunks())[0]
        vec = kb.get_embedding(first_chunk.chunk_id)
        assert vec is not None
        assert len(vec.vector) == 3

        # Missing chunk returns None
        assert kb.get_embedding("nonexistent") is None

    def test_with_mock_provider(self) -> None:
        """Mock provider returns expected zero vectors."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="mock_test",
            dimensions=2,
        )
        provider = MockEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_WHOLE_DOCUMENT,
            ),
            embedding_provider=provider,
        )

        doc = KnowledgeDocument(document_id="d1", content="Hello world.")
        kb.add_document(doc)

        assert len(kb.list_embeddings()) == 1
        vec = list(kb.list_embeddings())[0]
        assert vec.vector == (0.0, 0.0)
        assert vec.dimensions == 2

    def test_deterministic_output(self) -> None:
        """Same input → same embeddings across separate KB instances."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=4,
            normalize_embeddings=False,
        )

        def make_kb() -> KnowledgeBase:
            return KnowledgeBase(
                chunking_config=ChunkingConfig(
                    strategy=STRATEGY_FIXED_SIZE,
                    chunk_size=20,
                    chunk_overlap=0,
                    min_chunk_size=1,
                ),
                embedding_provider=DeterministicEmbeddingProvider(emb_config),
            )

        kb1 = make_kb()
        kb2 = make_kb()

        text = "This is a deterministic test."
        doc1 = KnowledgeDocument(document_id="d1", content=text)
        doc2 = KnowledgeDocument(document_id="d1", content=text)

        kb1.add_document(doc1)
        kb2.add_document(doc2)

        vecs1 = kb1.list_embeddings()
        vecs2 = kb2.list_embeddings()

        assert len(vecs1) == len(vecs2)
        for v1, v2 in zip(vecs1, vecs2):
            assert v1.vector == v2.vector

    def test_empty_document_no_embeddings(self) -> None:
        """Empty document produces no chunks, no embeddings."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=4,
            normalize_embeddings=False,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            embedding_provider=provider,
        )

        doc = KnowledgeDocument(document_id="d1", content="")
        kb.add_document(doc)
        assert kb.list_chunks() == []
        assert kb.list_embeddings() == []

    def test_unicode_text(self) -> None:
        """Unicode text produce embeddings correctly."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=4,
            normalize_embeddings=False,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            embedding_provider=provider,
        )

        doc = KnowledgeDocument(document_id="d1", content="Hello 世界! 🌍✨")
        kb.add_document(doc)
        assert len(kb.list_embeddings()) == 1
        vec = kb.list_embeddings()[0]
        assert len(vec.vector) == 4

    def test_duplicate_document_still_raises(self) -> None:
        """Duplicate detection works even with embedding provider."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=4,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            embedding_provider=provider,
        )

        kb.add_document(KnowledgeDocument(document_id="d1", content="Text."))
        with pytest.raises(DuplicateDocumentError):
            kb.add_document(KnowledgeDocument(document_id="d1", content="More text."))

    def test_remove_removes_embeddings(self) -> None:
        """Removing a document also removes its embeddings."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=3,
            normalize_embeddings=False,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
            embedding_provider=provider,
        )

        doc = KnowledgeDocument(document_id="d1", content="A" * 30)
        kb.add_document(doc)
        assert len(kb.list_embeddings()) >= 2

        kb.remove("d1")
        assert len(kb.list_embeddings()) == 0

    def test_clear_removes_embeddings(self) -> None:
        """Clearing the KB removes all embeddings."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=3,
            normalize_embeddings=False,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            embedding_provider=provider,
        )

        kb.add_document(KnowledgeDocument(document_id="d1", content="Text."))
        kb.add_document(KnowledgeDocument(document_id="d2", content="More."))
        assert len(kb.list_embeddings()) == 2

        kb.clear()
        assert len(kb.list_embeddings()) == 0

    def test_embedding_provider_property(self) -> None:
        """The embedding_provider property returns the configured provider."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=4,
        )
        provider = DeterministicEmbeddingProvider(emb_config)
        kb = KnowledgeBase(embedding_provider=provider)
        assert kb.embedding_provider is provider

    def test_embedding_vectors_have_metadata(self) -> None:
        """Embedding vectors include provider and metadata."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=4,
            normalize_embeddings=False,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            embedding_provider=provider,
        )

        kb.add_document(KnowledgeDocument(document_id="d1", content="Test text."))
        vec = kb.list_embeddings()[0]
        assert vec.provider == "deterministic"
        assert vec.dimensions == 4
        assert "text_length" in vec.metadata

    def test_mock_provider_with_custom_factory(self) -> None:
        """Mock provider with custom vector factory produces custom embeddings."""
        def factory(text: str) -> tuple[float, ...]:
            return (1.0, 2.0, 3.0)

        emb_config = EmbeddingConfig_Impl(
            provider_name="mock_custom",
            dimensions=3,
        )
        provider = MockEmbeddingProvider(emb_config, vector_factory=factory)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            embedding_provider=provider,
        )

        kb.add_document(KnowledgeDocument(document_id="d1", content="Test."))
        vec = kb.list_embeddings()[0]
        assert vec.vector == (1.0, 2.0, 3.0)

    def test_batch_embedding_generation(self) -> None:
        """Multiple chunks produce multiple embeddings, one per chunk."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=2,
            normalize_embeddings=False,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE,
                chunk_size=10,
                chunk_overlap=0,
                min_chunk_size=1,
            ),
            embedding_provider=provider,
        )

        doc = KnowledgeDocument(document_id="d1", content="A" * 30)
        kb.add_document(doc)

        chunks = kb.list_chunks()
        embeddings = kb.list_embeddings()
        assert len(embeddings) == len(chunks) == 3

    def test_embeddings_survive_register(self) -> None:
        """register() with pre-built chunks does not interact with embeddings."""
        emb_config = EmbeddingConfig_Impl(
            provider_name="deterministic",
            dimensions=2,
        )
        provider = DeterministicEmbeddingProvider(emb_config)

        kb = KnowledgeBase(
            embedding_provider=provider,
        )

        # Plain register — no automatic embedding generation
        doc = KnowledgeDocument(
            document_id="d1",
            chunks=(KnowledgeChunk(chunk_id="c1", document_id="d1", content="x"),),
        )
        kb.register(doc)
        assert len(kb.list_embeddings()) == 0
        assert kb.get_embedding("c1") is None
