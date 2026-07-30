"""Comprehensive tests for the vector store subsystem."""

from __future__ import annotations

import math

import pytest

from app.rag.vectorstore import (
    InvalidVectorStoreConfiguration,
    MemoryVectorStore,
    SearchResult,
    SimilarityMetric,
    VectorDimensionMismatchError,
    VectorNotFoundError,
    VectorStore,
    VectorStoreConfig,
    VectorStoreError,
    VectorStoreFullError,
    compute_similarity,
)
from app.rag.vectorstore.base import VectorStore as VectorStore_Impl
from app.rag.errors import KnowledgeError


# ======================================================================
# Imports & architecture
# ======================================================================


class TestArchitecture:
    def test_imports(self) -> None:
        assert VectorStore is VectorStore_Impl
        assert issubclass(VectorStoreError, KnowledgeError)
        assert issubclass(InvalidVectorStoreConfiguration, VectorStoreError)
        assert issubclass(VectorStoreFullError, VectorStoreError)
        assert issubclass(VectorDimensionMismatchError, VectorStoreError)
        assert issubclass(VectorNotFoundError, VectorStoreError)

    def test_vector_store_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            VectorStore()  # type: ignore[abstract]

    def test_abstract_methods_exist(self) -> None:
        assert hasattr(VectorStore, "add")
        assert hasattr(VectorStore, "add_batch")
        assert hasattr(VectorStore, "remove")
        assert hasattr(VectorStore, "clear")
        assert hasattr(VectorStore, "get")
        assert hasattr(VectorStore, "contains")
        assert hasattr(VectorStore, "count")
        assert hasattr(VectorStore, "search")

    def test_search_result(self) -> None:
        sr = SearchResult(chunk_id="c1", score=0.95, vector=(0.1, 0.2))
        assert sr.chunk_id == "c1"
        assert sr.score == 0.95
        assert sr.vector == (0.1, 0.2)
        with pytest.raises(AttributeError):
            sr.score = 0.5  # type: ignore[misc]

    def test_similarity_metric_values(self) -> None:
        assert SimilarityMetric.COSINE.value == "cosine"
        assert SimilarityMetric.DOT_PRODUCT.value == "dot_product"
        assert SimilarityMetric.EUCLIDEAN.value == "euclidean"


# ======================================================================
# VectorStoreConfig
# ======================================================================


class TestVectorStoreConfig:
    def test_defaults(self) -> None:
        cfg = VectorStoreConfig()
        assert cfg.metric == SimilarityMetric.COSINE
        assert cfg.max_vectors == 0
        assert cfg.validate_dimensions is True

    def test_custom(self) -> None:
        cfg = VectorStoreConfig(
            metric=SimilarityMetric.EUCLIDEAN,
            max_vectors=1000,
            validate_dimensions=False,
        )
        assert cfg.metric == SimilarityMetric.EUCLIDEAN
        assert cfg.max_vectors == 1000
        assert cfg.validate_dimensions is False

    def test_validate_passes(self) -> None:
        VectorStoreConfig(max_vectors=0).validate()
        VectorStoreConfig(max_vectors=100).validate()

    def test_validate_negative(self) -> None:
        with pytest.raises(InvalidVectorStoreConfiguration):
            VectorStoreConfig(max_vectors=-1).validate()

    def test_immutable(self) -> None:
        cfg = VectorStoreConfig()
        with pytest.raises(AttributeError):
            cfg.max_vectors = 10  # type: ignore[misc]


# ======================================================================
# Similarity metrics
# ======================================================================


class TestSimilarityMetrics:
    def test_cosine_identical(self) -> None:
        v = (1.0, 0.0, 0.0)
        assert math.isclose(compute_similarity(v, v, SimilarityMetric.COSINE), 1.0)

    def test_cosine_orthogonal(self) -> None:
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        assert math.isclose(compute_similarity(a, b, SimilarityMetric.COSINE), 0.0)

    def test_cosine_opposite(self) -> None:
        a = (1.0, 0.0)
        b = (-1.0, 0.0)
        assert math.isclose(compute_similarity(a, b, SimilarityMetric.COSINE), -1.0)

    def test_cosine_zero_vector(self) -> None:
        """Zero vector returns 0.0 similarity."""
        a = (1.0, 0.0)
        b = (0.0, 0.0)
        assert math.isclose(compute_similarity(a, b, SimilarityMetric.COSINE), 0.0)

    def test_dot_product(self) -> None:
        a = (1.0, 2.0)
        b = (3.0, 4.0)
        assert math.isclose(compute_similarity(a, b, SimilarityMetric.DOT_PRODUCT), 11.0)

    def test_dot_product_zero(self) -> None:
        a = (1.0, 0.0)
        b = (0.0, 1.0)
        assert math.isclose(compute_similarity(a, b, SimilarityMetric.DOT_PRODUCT), 0.0)

    def test_euclidean_identical(self) -> None:
        v = (1.0, 2.0, 3.0)
        assert math.isclose(compute_similarity(v, v, SimilarityMetric.EUCLIDEAN), 0.0)

    def test_euclidean_different(self) -> None:
        a = (0.0, 0.0)
        b = (3.0, 4.0)
        # -sqrt(9+16) = -5.0
        assert math.isclose(compute_similarity(a, b, SimilarityMetric.EUCLIDEAN), -5.0)

    def test_cosine_helper(self) -> None:
        from app.rag.vectorstore.metrics import cosine_similarity
        a = (2.0, 0.0)
        b = (0.0, 2.0)
        assert math.isclose(cosine_similarity(a, b), 0.0)


# ======================================================================
# MemoryVectorStore — basic operations
# ======================================================================


class TestMemoryVectorStoreBasics:
    def test_add_and_get(self) -> None:
        store = MemoryVectorStore()
        store.add("c1", (1.0, 0.0, 0.0))
        assert store.get("c1") == (1.0, 0.0, 0.0)
        assert store.get("nonexistent") is None

    def test_count(self) -> None:
        store = MemoryVectorStore()
        assert store.count() == 0
        store.add("c1", (0.1,))
        store.add("c2", (0.2,))
        assert store.count() == 2

    def test_contains(self) -> None:
        store = MemoryVectorStore()
        store.add("c1", (0.1,))
        assert store.contains("c1") is True
        assert store.contains("missing") is False

    def test_remove(self) -> None:
        store = MemoryVectorStore()
        store.add("c1", (0.1,))
        assert store.remove("c1") is True
        assert store.count() == 0
        assert store.remove("c1") is False

    def test_remove_nonexistent_returns_false(self) -> None:
        store = MemoryVectorStore()
        assert store.remove("nope") is False

    def test_clear(self) -> None:
        store = MemoryVectorStore()
        store.add("c1", (0.1,))
        store.add("c2", (0.2,))
        store.clear()
        assert store.count() == 0
        assert store.get("c1") is None

    def test_overwrite(self) -> None:
        store = MemoryVectorStore()
        store.add("c1", (1.0, 0.0))
        store.add("c1", (2.0, 3.0))
        assert store.get("c1") == (2.0, 3.0)
        assert store.count() == 1

    def test_properties(self) -> None:
        cfg = VectorStoreConfig(metric=SimilarityMetric.DOT_PRODUCT)
        store = MemoryVectorStore(config=cfg)
        assert store.config.metric == SimilarityMetric.DOT_PRODUCT


# ======================================================================
# MemoryVectorStore — add_batch
# ======================================================================


class TestMemoryVectorStoreBatch:
    def test_add_batch(self) -> None:
        store = MemoryVectorStore()
        items = [("c1", (0.1,)), ("c2", (0.2,)), ("c3", (0.3,))]
        store.add_batch(items)
        assert store.count() == 3
        assert store.get("c2") == (0.2,)

    def test_add_batch_empty(self) -> None:
        store = MemoryVectorStore()
        store.add_batch([])
        assert store.count() == 0

    def test_add_batch_overwrite(self) -> None:
        store = MemoryVectorStore()
        store.add("c1", (1.0,))
        store.add_batch([("c1", (99.0,))])
        assert store.get("c1") == (99.0,)


# ======================================================================
# MemoryVectorStore — dimension validation
# ======================================================================


class TestMemoryVectorStoreDimensions:
    def test_dimension_mismatch_raises(self) -> None:
        store = MemoryVectorStore()
        store.add("c1", (0.1, 0.2, 0.3))
        with pytest.raises(VectorDimensionMismatchError):
            store.add("c2", (0.1, 0.2))

    def test_dimension_mismatch_in_search(self) -> None:
        store = MemoryVectorStore()
        store.add("c1", (0.1, 0.2, 0.3))
        with pytest.raises(VectorDimensionMismatchError):
            store.search((0.1, 0.2), top_k=1)

    def test_dimension_validation_disabled(self) -> None:
        store = MemoryVectorStore(
            config=VectorStoreConfig(validate_dimensions=False),
        )
        store.add("c1", (0.1, 0.2))
        store.add("c2", (0.1, 0.2, 0.3))  # no error
        assert store.count() == 2


# ======================================================================
# MemoryVectorStore — capacity limits
# ======================================================================


class TestMemoryVectorStoreCapacity:
    def test_max_vectors(self) -> None:
        store = MemoryVectorStore(
            config=VectorStoreConfig(max_vectors=2),
        )
        store.add("c1", (0.1,))
        store.add("c2", (0.2,))
        with pytest.raises(VectorStoreFullError):
            store.add("c3", (0.3,))

    def test_max_vectors_zero_means_unlimited(self) -> None:
        store = MemoryVectorStore()
        for i in range(1000):
            store.add(f"c{i}", (float(i),))
        assert store.count() == 1000

    def test_max_vectors_allows_overwrite(self) -> None:
        """Overwriting an existing vector does not increase count."""
        store = MemoryVectorStore(
            config=VectorStoreConfig(max_vectors=1),
        )
        store.add("c1", (0.1,))
        store.add("c1", (0.2,))  # overwrite, not new
        assert store.count() == 1


# ======================================================================
# MemoryVectorStore — search
# ======================================================================


class TestMemoryVectorStoreSearch:
    def _store(
        self,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> MemoryVectorStore:
        store = MemoryVectorStore(
            config=VectorStoreConfig(metric=metric),
        )
        store.add("c1", (1.0, 0.0))
        store.add("c2", (0.0, 1.0))
        store.add("c3", (0.5, 0.5))
        store.add("c4", (-1.0, 0.0))
        return store

    def test_search_returns_results(self) -> None:
        store = self._store()
        results = store.search((1.0, 0.0), top_k=2)
        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_ordering_cosine(self) -> None:
        store = self._store(SimilarityMetric.COSINE)
        results = store.search((1.0, 0.0), top_k=4)
        # c1 (identical) should be first, c3 (0.5,0.5) second, c2 orthogonal, c4 opposite
        ids = [r.chunk_id for r in results]
        assert ids[0] == "c1"  # 1.0
        assert ids[-1] == "c4"  # -1.0 (opposite)

    def test_search_top_k(self) -> None:
        store = self._store()
        results = store.search((1.0, 0.0), top_k=1)
        assert len(results) == 1

    def test_search_top_k_larger_than_store(self) -> None:
        store = self._store()
        results = store.search((1.0, 0.0), top_k=100)
        assert len(results) == 4

    def test_search_empty_store(self) -> None:
        store = MemoryVectorStore()
        results = store.search((0.1, 0.2), top_k=5)
        assert results == []

    def test_search_identical_vectors_cosine(self) -> None:
        """All identical → all scores should be 1.0."""
        store = MemoryVectorStore(
            config=VectorStoreConfig(metric=SimilarityMetric.COSINE),
        )
        store.add("c1", (0.5, 0.5))
        store.add("c2", (0.5, 0.5))
        results = store.search((0.5, 0.5), top_k=2)
        assert len(results) == 2
        assert math.isclose(results[0].score, 1.0)
        assert math.isclose(results[1].score, 1.0)

    def test_search_deterministic_ordering(self) -> None:
        """Same store + same query = same order."""
        store = self._store()
        q = (1.0, 0.0)
        r1 = store.search(q, top_k=4)
        r2 = store.search(q, top_k=4)
        for a, b in zip(r1, r2):
            assert a.chunk_id == b.chunk_id
            assert math.isclose(a.score, b.score)

    def test_search_with_dot_product(self) -> None:
        store = self._store(SimilarityMetric.DOT_PRODUCT)
        results = store.search((1.0, 0.0), top_k=4)
        # dot product: c1=1.0, c3=0.5, c2=0.0, c4=-1.0
        assert results[0].chunk_id == "c1"
        assert results[-1].chunk_id == "c4"

    def test_search_with_euclidean(self) -> None:
        store = self._store(SimilarityMetric.EUCLIDEAN)
        results = store.search((1.0, 0.0), top_k=4)
        # negative euclidean: c1=0.0, c3=-0.707, c4=-2.0, c2=-1.414... wait
        # c2 is (0,1) with query (1,0): dist = sqrt((1-0)^2 + (0-1)^2) = sqrt(2) ≈ -1.414
        # c4 is (-1,0) with query (1,0): dist = sqrt(4) = -2.0
        # Order should be: c1 (0.0), c3 (-0.707), c2 (-1.414), c4 (-2.0)
        ids = [r.chunk_id for r in results]
        assert ids[0] == "c1"


# ======================================================================
# MemoryVectorStore — search result scores
# ======================================================================


class TestMemoryVectorStoreScores:
    def test_cosine_scores(self) -> None:
        store = MemoryVectorStore(
            config=VectorStoreConfig(metric=SimilarityMetric.COSINE),
        )
        store.add("c1", (1.0, 0.0))
        results = store.search((1.0, 0.0), top_k=1)
        assert math.isclose(results[0].score, 1.0)

    def test_dot_product_scores(self) -> None:
        store = MemoryVectorStore(
            config=VectorStoreConfig(metric=SimilarityMetric.DOT_PRODUCT),
        )
        store.add("c1", (2.0, 3.0))
        results = store.search((4.0, 5.0), top_k=1)
        assert math.isclose(results[0].score, 23.0)

    def test_euclidean_scores(self) -> None:
        store = MemoryVectorStore(
            config=VectorStoreConfig(metric=SimilarityMetric.EUCLIDEAN),
        )
        store.add("c1", (0.0, 0.0))
        results = store.search((3.0, 4.0), top_k=1)
        assert math.isclose(results[0].score, -5.0)


# ======================================================================
# KnowledgeBase integration
# ======================================================================


class TestKnowledgeBaseIntegration:
    """Vector store integration with KnowledgeBase."""

    def test_no_vector_store_by_default(self) -> None:
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.chunking import ChunkingConfig, STRATEGY_WHOLE_DOCUMENT

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
        )
        assert kb.vector_store is None
        doc = create_doc("d1", "Hello.")
        kb.add_document(doc)
        assert kb.list_embeddings() == []

    def test_vector_store_empty_without_embeddings(self) -> None:
        """Vector store is not populated unless an embedding provider is set."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.chunking import ChunkingConfig, STRATEGY_WHOLE_DOCUMENT

        vs = MemoryVectorStore()
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            vector_store=vs,
        )
        doc = create_doc("d1", "Hello.")
        kb.add_document(doc)
        # No embedding provider → no embeddings → nothing in vector store
        assert vs.count() == 0
        assert kb.vector_store is vs

    def test_automatic_insertion(self) -> None:
        """Vectors are automatically inserted when using embedding provider + vector store."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.chunking import ChunkingConfig, STRATEGY_FIXED_SIZE
        from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig

        emb_config = EConfig(provider_name="deterministic", dimensions=4, normalize_embeddings=False)
        provider = DeterministicEmbeddingProvider(emb_config)
        vs = MemoryVectorStore()

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE, chunk_size=10, chunk_overlap=0, min_chunk_size=1,
            ),
            embedding_provider=provider,
            vector_store=vs,
        )

        doc = create_doc("d1", "A" * 30)
        kb.add_document(doc)

        # Vector store should have vectors matching chunk count
        assert vs.count() == 3
        # Each chunk should have a corresponding vector
        for chunk in kb.list_chunks():
            assert vs.contains(chunk.chunk_id)
            vec = vs.get(chunk.chunk_id)
            assert len(vec) == 4

    def test_automatic_cleanup_on_remove(self) -> None:
        """Removing a document removes its vectors from the store."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.chunking import ChunkingConfig, STRATEGY_FIXED_SIZE
        from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig

        emb_config = EConfig(provider_name="deterministic", dimensions=2, normalize_embeddings=False)
        vs = MemoryVectorStore()
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE, chunk_size=10, chunk_overlap=0, min_chunk_size=1,
            ),
            embedding_provider=DeterministicEmbeddingProvider(emb_config),
            vector_store=vs,
        )

        doc = create_doc("d1", "A" * 30)
        kb.add_document(doc)
        assert vs.count() == 3

        kb.remove("d1")
        assert vs.count() == 0

    def test_automatic_cleanup_on_clear(self) -> None:
        """Clearing the KB clears the vector store."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.chunking import ChunkingConfig, STRATEGY_WHOLE_DOCUMENT
        from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig

        emb_config = EConfig(provider_name="deterministic", dimensions=2, normalize_embeddings=False)
        vs = MemoryVectorStore()
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            embedding_provider=DeterministicEmbeddingProvider(emb_config),
            vector_store=vs,
        )

        kb.add_document(create_doc("d1", "Hello."))
        kb.add_document(create_doc("d2", "World."))
        assert vs.count() == 2

        kb.clear()
        assert vs.count() == 0

    def test_registration_without_embedding(self) -> None:
        """register() does not add to vector store."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeChunk

        vs = MemoryVectorStore()
        kb = KnowledgeBase(vector_store=vs)
        doc = create_doc("d1", "Hello.", chunks=(KnowledgeChunk(chunk_id="c1", content="Hello."),))
        kb.register(doc)
        assert vs.count() == 0

    def test_search_after_ingestion(self) -> None:
        """Search returns results after documents are added."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.chunking import ChunkingConfig, STRATEGY_FIXED_SIZE
        from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig

        emb_config = EConfig(provider_name="deterministic", dimensions=4, normalize_embeddings=False)
        vs = MemoryVectorStore()

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(
                strategy=STRATEGY_FIXED_SIZE, chunk_size=10, chunk_overlap=0, min_chunk_size=1,
            ),
            embedding_provider=DeterministicEmbeddingProvider(emb_config),
            vector_store=vs,
        )

        kb.add_document(create_doc("d1", "Hello World ABC DEF."))
        kb.add_document(create_doc("d2", "Another document here."))

        # Search with one of the stored vectors
        first_vec = vs.get(list(kb.list_chunks())[0].chunk_id)
        results = vs.search(first_vec, top_k=5)
        assert len(results) >= 1
        assert results[0].score > 0  # identical vector has highest score

    def test_provider_property(self) -> None:
        """vector_store property returns the configured store."""
        from app.rag.knowledge_base import KnowledgeBase

        vs = MemoryVectorStore()
        kb = KnowledgeBase(vector_store=vs)
        assert kb.vector_store is vs

    def test_no_vector_store_embedding_still_works(self) -> None:
        """Embeddings are stored in KB even without vector store."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.chunking import ChunkingConfig, STRATEGY_WHOLE_DOCUMENT
        from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig

        emb_config = EConfig(provider_name="deterministic", dimensions=3, normalize_embeddings=False)
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
            embedding_provider=DeterministicEmbeddingProvider(emb_config),
        )
        kb.add_document(create_doc("d1", "Hello."))
        assert len(kb.list_embeddings()) == 1
        assert kb.get_embedding(kb.list_chunks()[0].chunk_id) is not None


# ======================================================================
# Error hierarchy
# ======================================================================


class TestVectorStoreErrors:
    def test_vector_store_error(self) -> None:
        err = VectorStoreError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.code == "VECTOR_STORE_ERROR"

    def test_invalid_config(self) -> None:
        err = InvalidVectorStoreConfiguration("Bad config")
        assert err.code == "INVALID_VECTOR_STORE_CONFIGURATION"
        assert isinstance(err, VectorStoreError)

    def test_full_error(self) -> None:
        err = VectorStoreFullError("Full")
        assert err.code == "VECTOR_STORE_FULL"
        assert isinstance(err, VectorStoreError)

    def test_dimension_mismatch(self) -> None:
        err = VectorDimensionMismatchError(expected=3, actual=2)
        assert "expected 3, got 2" in str(err)
        assert err.code == "VECTOR_DIMENSION_MISMATCH"
        assert err.details["expected"] == 3
        assert err.details["actual"] == 2

    def test_not_found(self) -> None:
        err = VectorNotFoundError("c1")
        assert "c1" in str(err)
        assert err.code == "VECTOR_NOT_FOUND"

    def test_not_found_empty(self) -> None:
        err = VectorNotFoundError()
        assert str(err) == "Vector not found"

    def test_knowledge_error_is_base(self) -> None:
        assert issubclass(VectorStoreError, KnowledgeError)


# ======================================================================
# Helper
# ======================================================================


def create_doc(
    document_id: str,
    content: str,
    **kw: object,
) -> object:
    from app.rag.models import KnowledgeDocument
    return KnowledgeDocument(document_id=document_id, content=content, **kw)
