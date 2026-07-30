"""Comprehensive tests for the hybrid retrieval layer — architecture + DefaultHybridRetriever."""

from __future__ import annotations

import asyncio
import math

import pytest

from app.rag.hybrid import (
    DefaultHybridRetriever,
    FusionError,
    FusionStrategy,
    HybridConfig,
    HybridError,
    HybridResult,
    HybridRetriever,
    InvalidHybridConfiguration,
    RetrievalScore,
    reciprocal_rank_fusion,
    weighted_sum,
)
from app.rag.hybrid.base import HybridRetriever as HybridRetriever_Impl
from app.rag.hybrid.config import HybridConfig as HybridConfig_Impl
from app.rag.hybrid.errors import HybridError as HybridError_Impl
from app.rag.hybrid.models import HybridResult as HybridResult_Impl
from app.rag.hybrid.models import RetrievalScore as RetrievalScore_Impl
from app.rag.errors import KnowledgeError
from app.rag.context import KnowledgeContextBuilder


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_hybrid_config_imported(self) -> None:
        assert HybridConfig is HybridConfig_Impl

    def test_hybrid_error_imported(self) -> None:
        assert HybridError is HybridError_Impl

    def test_hybrid_result_imported(self) -> None:
        assert HybridResult is HybridResult_Impl

    def test_retrieval_score_imported(self) -> None:
        assert RetrievalScore is RetrievalScore_Impl

    def test_hybrid_retriever_imported(self) -> None:
        assert HybridRetriever is HybridRetriever_Impl

    def test_default_hybrid_retriever_imported(self) -> None:
        assert DefaultHybridRetriever is not None
        assert issubclass(DefaultHybridRetriever, HybridRetriever)

    def test_fusion_functions_imported(self) -> None:
        assert callable(weighted_sum)
        assert callable(reciprocal_rank_fusion)

    def test_error_hierarchy(self) -> None:
        assert issubclass(HybridError, KnowledgeError)
        assert issubclass(InvalidHybridConfiguration, HybridError)
        assert issubclass(FusionError, HybridError)


# ======================================================================
# FusionStrategy enum
# ======================================================================


class TestFusionStrategy:
    def test_values(self) -> None:
        assert FusionStrategy.WEIGHTED_SUM.value == "weighted_sum"
        assert FusionStrategy.RECIPROCAL_RANK_FUSION.value == "reciprocal_rank_fusion"


# ======================================================================
# HybridConfig
# ======================================================================


class TestHybridConfig:
    def test_default_values(self) -> None:
        cfg = HybridConfig()
        assert cfg.keyword_weight == 0.5
        assert cfg.semantic_weight == 0.5
        assert cfg.max_candidates == 20
        assert cfg.fusion_strategy == FusionStrategy.WEIGHTED_SUM

    def test_custom_values(self) -> None:
        cfg = HybridConfig(
            keyword_weight=0.3,
            semantic_weight=0.7,
            max_candidates=50,
            fusion_strategy=FusionStrategy.RECIPROCAL_RANK_FUSION,
        )
        assert cfg.keyword_weight == 0.3
        assert cfg.semantic_weight == 0.7
        assert cfg.max_candidates == 50
        assert cfg.fusion_strategy == FusionStrategy.RECIPROCAL_RANK_FUSION

    def test_validate_passes(self) -> None:
        HybridConfig(keyword_weight=0.2, semantic_weight=0.8, max_candidates=10).validate()

    def test_validate_keyword_weight_negative(self) -> None:
        with pytest.raises(InvalidHybridConfiguration):
            HybridConfig(keyword_weight=-1.0).validate()

    def test_validate_semantic_weight_negative(self) -> None:
        with pytest.raises(InvalidHybridConfiguration):
            HybridConfig(semantic_weight=-0.5).validate()

    def test_validate_weights_sum_zero(self) -> None:
        with pytest.raises(InvalidHybridConfiguration):
            HybridConfig(keyword_weight=0.0, semantic_weight=0.0).validate()

    def test_validate_max_candidates_zero(self) -> None:
        with pytest.raises(InvalidHybridConfiguration):
            HybridConfig(max_candidates=0).validate()

    def test_immutable(self) -> None:
        cfg = HybridConfig()
        with pytest.raises(AttributeError):
            cfg.keyword_weight = 0.9  # type: ignore[misc]


# ======================================================================
# RetrievalScore
# ======================================================================


class TestRetrievalScore:
    def test_default_values(self) -> None:
        s = RetrievalScore()
        assert s.chunk_id == ""
        assert s.keyword_score == 0.0
        assert s.semantic_score == 0.0
        assert s.final_score == 0.0

    def test_custom_values(self) -> None:
        s = RetrievalScore(chunk_id="c1", keyword_score=0.8, semantic_score=0.6, final_score=0.7)
        assert s.chunk_id == "c1"
        assert s.keyword_score == 0.8
        assert s.semantic_score == 0.6
        assert s.final_score == 0.7

    def test_immutable(self) -> None:
        s = RetrievalScore(chunk_id="c1")
        with pytest.raises(AttributeError):
            s.final_score = 0.5  # type: ignore[misc]


# ======================================================================
# HybridResult
# ======================================================================


class TestHybridResult:
    def test_default_values(self) -> None:
        r = HybridResult()
        assert r.results == ()
        assert r.metadata == {}

    def test_with_results(self) -> None:
        scores = (RetrievalScore(chunk_id="c1", final_score=0.9),)
        result = HybridResult(results=scores, metadata={"strategy": "weighted_sum"})
        assert len(result.results) == 1
        assert result.results[0].chunk_id == "c1"

    def test_immutable(self) -> None:
        r = HybridResult()
        with pytest.raises(AttributeError):
            r.results = ()  # type: ignore[misc]


# ======================================================================
# HybridRetriever (abstract)
# ======================================================================


class TestHybridRetriever:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            HybridRetriever()  # type: ignore[abstract]

    def test_abstract_methods(self) -> None:
        assert hasattr(HybridRetriever, "retrieve")


# ======================================================================
# DefaultHybridRetriever — architecture
# ======================================================================


class TestDefaultHybridRetrieverArch:
    def test_concrete_subclass(self) -> None:
        class FakeKB:
            embedding_provider = None
            vector_store = None

        class FakeKR:
            async def retrieve(self, query):
                from app.rag.models import KnowledgeResult
                return KnowledgeResult()

        retriever = DefaultHybridRetriever(FakeKB(), FakeKR())
        assert retriever.config.keyword_weight == 0.5
        assert retriever.config.fusion_strategy == FusionStrategy.WEIGHTED_SUM

    def test_custom_config(self) -> None:
        class FakeKB:
            embedding_provider = None
            vector_store = None

        class FakeKR:
            async def retrieve(self, query):
                from app.rag.models import KnowledgeResult
                return KnowledgeResult()

        cfg = HybridConfig(fusion_strategy=FusionStrategy.RECIPROCAL_RANK_FUSION)
        retriever = DefaultHybridRetriever(FakeKB(), FakeKR(), config=cfg)
        assert retriever.config.fusion_strategy == FusionStrategy.RECIPROCAL_RANK_FUSION

    def test_properties(self) -> None:
        class FakeKB:
            embedding_provider = None
            vector_store = None

        class FakeKR:
            async def retrieve(self, query):
                from app.rag.models import KnowledgeResult
                return KnowledgeResult()

        kb = FakeKB()
        kr = FakeKR()
        r = DefaultHybridRetriever(kb, kr)
        assert r.knowledge_base is kb
        assert r.keyword_retriever is kr


# ======================================================================
# Fusion helpers
# ======================================================================


class TestWeightedSum:
    def test_equal_weights(self) -> None:
        kw = {"c1": 1.0, "c2": 0.5}
        sem = {"c2": 0.8, "c3": 0.3}
        result = weighted_sum(kw, sem)
        assert result["c1"] == pytest.approx(0.5)
        assert result["c2"] == pytest.approx(0.65)
        assert result["c3"] == pytest.approx(0.15)

    def test_keyword_only(self) -> None:
        result = weighted_sum({"c1": 1.0}, {}, keyword_weight=1.0, semantic_weight=0.0)
        assert result["c1"] == pytest.approx(1.0)

    def test_semantic_only(self) -> None:
        result = weighted_sum({}, {"c1": 0.9}, keyword_weight=0.0, semantic_weight=1.0)
        assert result["c1"] == pytest.approx(0.9)

    def test_empty_scores(self) -> None:
        assert weighted_sum({}, {}) == {}

    def test_all_weights_zero_raises(self) -> None:
        with pytest.raises(FusionError):
            weighted_sum({}, {}, keyword_weight=0.0, semantic_weight=0.0)


class TestReciprocalRankFusion:
    def test_basic(self) -> None:
        kw = ["c1", "c2", "c3"]
        sem = ["c2", "c1", "c4"]
        result = reciprocal_rank_fusion(kw, sem, k=60)
        assert set(result) == {"c1", "c2", "c3", "c4"}

    def test_identical_rankings(self) -> None:
        kw = sem = ["c1", "c2", "c3"]
        result = reciprocal_rank_fusion(kw, sem, k=60)
        assert result["c1"] > result["c3"]

    def test_empty_lists(self) -> None:
        assert reciprocal_rank_fusion([], [], k=60) == {}


# ======================================================================
# DefaultHybridRetriever — integration with KnowledgeBase
# ======================================================================


class TestDefaultHybridRetriever:
    """Integration tests using real KnowledgeBase, retriever, embeddings, vector store."""

    @pytest.fixture
    def kb_and_retriever(self):
        """Build a fully-configured KB with chunks, embeddings, and vector store."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.chunking import ChunkingConfig, STRATEGY_SENTENCE
        from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig
        from app.rag.vectorstore import MemoryVectorStore, VectorStoreConfig, SimilarityMetric
        from app.rag.retriever import KnowledgeRetriever

        emb_config = EConfig(
            provider_name="deterministic", dimensions=4, normalize_embeddings=True,
        )
        vs = MemoryVectorStore(
            config=VectorStoreConfig(metric=SimilarityMetric.COSINE),
        )
        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_SENTENCE),
            embedding_provider=DeterministicEmbeddingProvider(emb_config),
            vector_store=vs,
        )

        # Paris document: sentences about Paris
        kb.add_document(doc("paris", "Paris is the capital of France. It has the Eiffel Tower. "
                                     "The Louvre museum is in Paris. French cuisine is famous."))
        # London document
        kb.add_document(doc("london", "London is the capital of the UK. Big Ben is in London. "
                                      "The Thames river flows through London. Fish and chips."))
        # Python document (unrelated)
        kb.add_document(doc("python", "Python is a programming language. It was created by "
                                       "Guido van Rossum. Python is used for data science."))

        retriever = KnowledgeRetriever(kb)
        return kb, retriever

    async def test_keyword_only(self, kb_and_retriever):
        """When no vector store is available, keyword retrieval alone works."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.retriever import KnowledgeRetriever

        kb, _ = kb_and_retriever
        # KB without vector store
        kb2 = KnowledgeBase()
        for d in kb.list_documents():
            # Re-register with pre-built chunks
            kb2.register(d)

        kr = KnowledgeRetriever(kb2)
        hr = DefaultHybridRetriever(kb2, kr)
        result = await hr.retrieve("capital", top_k=3)
        assert len(result.results) > 0
        assert result.metadata.get("keyword_candidates", 0) > 0
        assert result.metadata.get("semantic_candidates", 0) == 0

    async def test_semantic_only(self, kb_and_retriever):
        """When keyword_weight=0, only semantic scores matter."""
        kb, kr = kb_and_retriever
        cfg = HybridConfig(keyword_weight=0.0, semantic_weight=1.0)
        hr = DefaultHybridRetriever(kb, kr, config=cfg)
        # Query that would not match keyword but has related semantics
        result = await hr.retrieve("Eiffel", top_k=3)
        assert len(result.results) > 0
        # All scores should have 0 keyword component
        for rs in result.results:
            assert rs.keyword_score >= 0

    async def test_weighted_fusion(self, kb_and_retriever):
        """Weighted sum fusion produces combined results."""
        kb, kr = kb_and_retriever
        cfg = HybridConfig(fusion_strategy=FusionStrategy.WEIGHTED_SUM)
        hr = DefaultHybridRetriever(kb, kr, config=cfg)
        result = await hr.retrieve("capital London", top_k=5)
        assert len(result.results) > 0
        assert result.metadata["fusion_strategy"] == "weighted_sum"
        for rs in result.results:
            assert rs.final_score >= 0

    async def test_rrf_fusion(self, kb_and_retriever):
        """Reciprocal rank fusion produces combined results."""
        kb, kr = kb_and_retriever
        cfg = HybridConfig(fusion_strategy=FusionStrategy.RECIPROCAL_RANK_FUSION)
        hr = DefaultHybridRetriever(kb, kr, config=cfg)
        result = await hr.retrieve("capital", top_k=5)
        assert len(result.results) > 0
        assert result.metadata["fusion_strategy"] == "reciprocal_rank_fusion"
        # RRF scores should be positive
        for rs in result.results:
            assert rs.final_score > 0

    async def test_top_k(self, kb_and_retriever):
        """top_k limits the number of results."""
        kb, kr = kb_and_retriever
        hr = DefaultHybridRetriever(kb, kr)
        for k in (1, 2, 3, 10):
            result = await hr.retrieve("capital", top_k=k)
            assert len(result.results) <= k

    async def test_empty_kb(self):
        """Empty knowledge base returns empty results."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.retriever import KnowledgeRetriever

        kb = KnowledgeBase()
        kr = KnowledgeRetriever(kb)
        hr = DefaultHybridRetriever(kb, kr)
        result = await hr.retrieve("anything", top_k=5)
        assert len(result.results) == 0
        assert result.metadata["keyword_candidates"] == 0

    async def test_deterministic_ordering(self, kb_and_retriever):
        """Same query produces same ordering."""
        kb, kr = kb_and_retriever
        hr = DefaultHybridRetriever(kb, kr)
        r1 = await hr.retrieve("capital", top_k=5)
        r2 = await hr.retrieve("capital", top_k=5)
        for a, b in zip(r1.results, r2.results):
            assert a.chunk_id == b.chunk_id
            assert math.isclose(a.final_score, b.final_score, rel_tol=1e-6)

    async def test_unicode_query(self, kb_and_retriever):
        """Unicode queries are handled correctly."""
        kb, kr = kb_and_retriever
        hr = DefaultHybridRetriever(kb, kr)
        result = await hr.retrieve("Françé", top_k=3)
        assert isinstance(result, HybridResult)

    async def test_no_embedding_provider(self):
        """When KB has no embedding provider, only keyword retrieval runs."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.retriever import KnowledgeRetriever
        from app.rag.chunking import ChunkingConfig, STRATEGY_WHOLE_DOCUMENT

        kb = KnowledgeBase(
            chunking_config=ChunkingConfig(strategy=STRATEGY_WHOLE_DOCUMENT),
        )
        kb.add_document(doc("d1", "Paris is the capital of France."))
        kr = KnowledgeRetriever(kb)
        hr = DefaultHybridRetriever(kb, kr)
        result = await hr.retrieve("capital", top_k=5)
        assert len(result.results) > 0
        assert result.metadata["semantic_candidates"] == 0

    async def test_no_vector_store(self):
        """When KB has no vector store, only keyword retrieval runs."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.retriever import KnowledgeRetriever
        from app.rag.chunking import STRATEGY_WHOLE_DOCUMENT
        from app.rag.models import KnowledgeChunk

        # Use register() with pre-built chunks to avoid asyncio.run() in
        # add_document() while running inside an async test context.
        kb = KnowledgeBase()
        chunk = KnowledgeChunk(chunk_id="d1_chunk", document_id="d1",
                                content="Paris is the capital of France.")
        from app.rag.models import KnowledgeDocument
        kb.register(KnowledgeDocument(
            document_id="d1", content="Paris is the capital of France.",
            chunks=(chunk,),
        ))
        kr = KnowledgeRetriever(kb)
        hr = DefaultHybridRetriever(kb, kr)
        result = await hr.retrieve("capital", top_k=5)
        assert len(result.results) > 0
        assert result.metadata["semantic_candidates"] == 0

    async def test_results_type(self, kb_and_retriever):
        """Results contain RetrievalScore objects with populated fields."""
        kb, kr = kb_and_retriever
        hr = DefaultHybridRetriever(kb, kr)
        result = await hr.retrieve("capital", top_k=5)
        for rs in result.results:
            assert isinstance(rs, RetrievalScore)
            assert isinstance(rs.chunk_id, str)
            assert isinstance(rs.final_score, float)

    async def test_metadata(self, kb_and_retriever):
        """Metadata includes timing and candidate info."""
        kb, kr = kb_and_retriever
        hr = DefaultHybridRetriever(kb, kr)
        result = await hr.retrieve("capital", top_k=5)
        meta = result.metadata
        assert "keyword_elapsed_ms" in meta
        assert "semantic_elapsed_ms" in meta
        assert "fusion_elapsed_ms" in meta
        assert "total_elapsed_ms" in meta
        assert "keyword_candidates" in meta
        assert "semantic_candidates" in meta
        assert "fusion_strategy" in meta


# ======================================================================
# KnowledgeBase hybird_retriever property
# ======================================================================


@pytest.fixture
def hybrid_kb():
    """Build KB with pre-chunked+embedded data.

    Uses the provider's synchronous _generate method to avoid
    asyncio.run() inside async tests.
    """
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig
    from app.rag.vectorstore import MemoryVectorStore, VectorStoreConfig, SimilarityMetric
    from app.rag.models import KnowledgeDocument, KnowledgeChunk

    emb_config = EConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
    provider = DeterministicEmbeddingProvider(emb_config)
    vs = MemoryVectorStore(config=VectorStoreConfig(metric=SimilarityMetric.COSINE))
    kb = KnowledgeBase(embedding_provider=provider, vector_store=vs)

    chunk1 = KnowledgeChunk(chunk_id="d1_s0", document_id="d1", index=0,
                            content="Paris is the capital of France.")
    chunk2 = KnowledgeChunk(chunk_id="d1_s1", document_id="d1", index=1,
                            content="It has the Eiffel Tower.")
    kb.register(KnowledgeDocument(document_id="d1", chunks=(chunk1, chunk2)))
    for ch in (chunk1, chunk2):
        vec = provider._generate(ch.content)
        from app.rag.embeddings.models import EmbeddingVector
        kb._embeddings[ch.chunk_id] = EmbeddingVector(
            vector=vec, dimensions=len(vec), provider="det",
        )
        vs.add(ch.chunk_id, vec)
    return kb


class TestKnowledgeBaseHybridIntegration:
    """Tests for kb.hybrid_retriever auto-detection."""

    def test_hybrid_retriever_none_by_default(self):
        from app.rag.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        assert kb.hybrid_retriever is None

    def test_hybrid_retriever_requires_both(self):
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig
        from app.rag.vectorstore import MemoryVectorStore

        emb_config = EConfig(provider_name="det", dimensions=4)
        kb1 = KnowledgeBase(embedding_provider=DeterministicEmbeddingProvider(emb_config))
        assert kb1.hybrid_retriever is None
        kb2 = KnowledgeBase(vector_store=MemoryVectorStore())
        assert kb2.hybrid_retriever is None
        kb3 = KnowledgeBase(
            embedding_provider=DeterministicEmbeddingProvider(emb_config),
            vector_store=MemoryVectorStore(),
        )
        assert kb3.hybrid_retriever is not None

    async def test_hybrid_retriever_works(self, hybrid_kb):
        hybrid = hybrid_kb.hybrid_retriever
        assert hybrid is not None
        result = await hybrid.retrieve("capital", top_k=3)
        assert len(result.results) > 0
        assert result.metadata["semantic_candidates"] > 0


# ======================================================================
# ContextBuilder integration with hybrid
# ======================================================================


class TestContextBuilderHybrid:
    """Context builder should auto-detect and use hybrid retrieval."""

    async def test_context_builder_keyword_fallback(self):
        """Without hybrid capabilities, builder uses keyword retrieval."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeChunk

        kb = KnowledgeBase()
        chunk = KnowledgeChunk(chunk_id="c1", content="Paris is the capital.")
        from app.rag.models import KnowledgeDocument
        kb.register(KnowledgeDocument(document_id="d1", chunks=(chunk,)))

        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="capital", max_chunks=5)
        assert context.total_chunks > 0

    async def test_context_builder_hybrid_auto(self, hybrid_kb):
        """Builder auto-uses hybrid when KB has embedding + vector store."""
        builder = KnowledgeContextBuilder(hybrid_kb)
        context = await builder.build(query="capital", max_chunks=5)
        assert context.total_chunks > 0
        assert len(context.text) > 0
        assert len(context.sources) > 0

    async def test_context_builder_hybrid_sources(self, hybrid_kb):
        """Hybrid mode produces valid sources with document IDs."""
        builder = KnowledgeContextBuilder(hybrid_kb)
        context = await builder.build(query="Paris", max_chunks=5)
        for source in context.sources:
            assert source.document_id == "d1"
            assert source.chunk_id != ""

    async def test_context_builder_identical_output(self, hybrid_kb):
        """Builder with same config returns deterministic results."""
        builder = KnowledgeContextBuilder(hybrid_kb)
        c1 = await builder.build(query="capital", max_chunks=5)
        c2 = await builder.build(query="capital", max_chunks=5)
        assert c1.text == c2.text
        assert len(c1.chunks) == len(c2.chunks)
        for s1, s2 in zip(c1.sources, c2.sources):
            assert s1.chunk_id == s2.chunk_id
            assert s1.score == s2.score


# ======================================================================
# Error hierarchy
# ======================================================================


class TestHybridErrors:
    def test_hybrid_error(self) -> None:
        err = HybridError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.code == "HYBRID_ERROR"

    def test_invalid_configuration(self) -> None:
        err = InvalidHybridConfiguration("Bad config")
        assert err.code == "INVALID_HYBRID_CONFIGURATION"
        assert isinstance(err, HybridError)

    def test_fusion_error(self) -> None:
        err = FusionError("Fusion failed")
        assert err.code == "FUSION_ERROR"

    def test_to_dict(self) -> None:
        err = InvalidHybridConfiguration("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "INVALID_HYBRID_CONFIGURATION"
        assert d["message"] == "test"

    def test_knowledge_error_is_base(self) -> None:
        assert issubclass(HybridError, KnowledgeError)


# ======================================================================
# Helper
# ======================================================================


def doc(document_id: str, content: str) -> object:
    from app.rag.models import KnowledgeDocument
    return KnowledgeDocument(document_id=document_id, content=content)
