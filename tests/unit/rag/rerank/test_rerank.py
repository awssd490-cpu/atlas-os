"""Tests for the reranking architecture and DefaultReranker."""

from __future__ import annotations

import math

import pytest

from app.rag.rerank import (
    DefaultReranker,
    InvalidRerankConfiguration,
    RerankConfig,
    RerankError,
    RerankResponse,
    RerankedResult,
    Reranker,
    RerankerNotFound,
    clear_rerankers,
    get_reranker,
    list_rerankers,
    register_reranker,
)
from app.rag.rerank.base import Reranker as Reranker_Impl
from app.rag.rerank.config import RerankConfig as RerankConfig_Impl
from app.rag.rerank.errors import RerankError as RerankError_Impl
from app.rag.rerank.models import RerankResponse as RerankResponse_Impl
from app.rag.rerank.models import RerankedResult as RerankedResult_Impl
from app.rag.errors import KnowledgeError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_rerank_config_imported(self) -> None:
        assert RerankConfig is RerankConfig_Impl

    def test_rerank_error_imported(self) -> None:
        assert RerankError is RerankError_Impl

    def test_reranker_imported(self) -> None:
        assert Reranker is Reranker_Impl

    def test_reranked_result_imported(self) -> None:
        assert RerankedResult is RerankedResult_Impl

    def test_rerank_response_imported(self) -> None:
        assert RerankResponse is RerankResponse_Impl

    def test_default_reranker_imported(self) -> None:
        assert DefaultReranker is not None
        assert issubclass(DefaultReranker, Reranker)

    def test_error_hierarchy(self) -> None:
        assert issubclass(RerankError, KnowledgeError)
        assert issubclass(InvalidRerankConfiguration, RerankError)
        assert issubclass(RerankerNotFound, RerankError)

    def test_registry_functions_imported(self) -> None:
        assert callable(register_reranker)
        assert callable(get_reranker)
        assert callable(list_rerankers)
        assert callable(clear_rerankers)


# ======================================================================
# RerankConfig
# ======================================================================


class TestRerankConfig:
    def test_default_values(self) -> None:
        cfg = RerankConfig()
        assert cfg.enabled is True
        assert cfg.top_k == 10
        assert cfg.score_threshold == 0.0

    def test_custom_values(self) -> None:
        cfg = RerankConfig(enabled=False, top_k=5, score_threshold=0.3)
        assert cfg.enabled is False
        assert cfg.top_k == 5
        assert cfg.score_threshold == 0.3

    def test_immutable(self) -> None:
        cfg = RerankConfig()
        with pytest.raises(AttributeError):
            cfg.top_k = 5  # type: ignore[misc]

    def test_validate_passes(self) -> None:
        RerankConfig(top_k=1, score_threshold=0.0).validate()
        RerankConfig(top_k=100, score_threshold=1.0).validate()

    def test_validate_top_k_zero(self) -> None:
        with pytest.raises(InvalidRerankConfiguration):
            RerankConfig(top_k=0).validate()

    def test_validate_top_k_negative(self) -> None:
        with pytest.raises(InvalidRerankConfiguration):
            RerankConfig(top_k=-1).validate()

    def test_validate_score_threshold_negative(self) -> None:
        with pytest.raises(InvalidRerankConfiguration):
            RerankConfig(score_threshold=-0.1).validate()

    def test_validate_score_threshold_greater_than_one(self) -> None:
        with pytest.raises(InvalidRerankConfiguration):
            RerankConfig(score_threshold=1.1).validate()


# ======================================================================
# RerankedResult
# ======================================================================


class TestRerankedResult:
    def test_default_values(self) -> None:
        r = RerankedResult()
        assert r.chunk_id == ""
        assert r.original_score == 0.0
        assert r.rerank_score == 0.0
        assert r.final_score == 0.0

    def test_custom_values(self) -> None:
        r = RerankedResult(chunk_id="c1", original_score=0.8, rerank_score=0.9, final_score=0.85)
        assert r.chunk_id == "c1"
        assert r.original_score == 0.8
        assert r.rerank_score == 0.9
        assert r.final_score == 0.85

    def test_immutable(self) -> None:
        r = RerankedResult(chunk_id="c1")
        with pytest.raises(AttributeError):
            r.final_score = 0.5  # type: ignore[misc]


# ======================================================================
# RerankResponse
# ======================================================================


class TestRerankResponse:
    def test_default_values(self) -> None:
        r = RerankResponse()
        assert r.results == ()
        assert r.metadata == {}

    def test_with_results(self) -> None:
        results = (RerankedResult(chunk_id="c1", final_score=0.9),)
        response = RerankResponse(results=results, metadata={"model": "test"})
        assert len(response.results) == 1
        assert response.results[0].chunk_id == "c1"

    def test_immutable(self) -> None:
        r = RerankResponse()
        with pytest.raises(AttributeError):
            r.results = ()  # type: ignore[misc]


# ======================================================================
# Reranker ABC + Registry
# ======================================================================


class TestReranker:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Reranker()  # type: ignore[abstract]

    def test_abstract_methods(self) -> None:
        assert hasattr(Reranker, "rerank")


class TestRerankerRegistry:
    def test_register_and_get(self) -> None:
        class Fake(Reranker):
            async def rerank(self, query: str, results: list[tuple[str, float]]) -> RerankResponse:
                return RerankResponse()
        register_reranker("fake", Fake)
        assert get_reranker("fake") is Fake
        clear_rerankers()

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(RerankerNotFound):
            get_reranker("nonexistent")


# ======================================================================
# DefaultReranker — score() method
# ======================================================================


class TestDefaultRerankerScore:
    """Tests for the core scoring algorithm via score()."""

    @pytest.fixture
    def reranker(self) -> DefaultReranker:
        return DefaultReranker()

    def test_lexical_overlap(self, reranker: DefaultReranker) -> None:
        """Matching query terms produce a higher rerank_score."""
        r1 = reranker.score("capital city", "The capital of France is Paris.")
        r2 = reranker.score("capital city", "The weather today is sunny.")
        assert r1.rerank_score > r2.rerank_score

    def test_exact_phrase_bonus(self, reranker: DefaultReranker) -> None:
        """Exact query match gives bonus."""
        r1 = reranker.score("capital of France", "The capital of France is Paris.")
        r2 = reranker.score("capital of France", "A capital city named Paris.")
        assert r1.rerank_score > r2.rerank_score

    def test_length_penalty(self, reranker: DefaultReranker) -> None:
        """Very short chunks get lower scores."""
        r1 = reranker.score("test", "test " * 50)       # ~250 chars
        r2 = reranker.score("test", "test")              # 4 chars
        assert r1.rerank_score != r2.rerank_score

    def test_final_score_combines(self, reranker: DefaultReranker) -> None:
        """final_score = original_score + rerank_weight * rerank_score."""
        result = reranker.score("capital", "The capital is Paris.", original_score=0.5)
        expected = 0.5 + 1.0 * result.rerank_score
        assert math.isclose(result.final_score, expected)

    def test_deterministic(self, reranker: DefaultReranker) -> None:
        """Same inputs produce same outputs."""
        r1 = reranker.score("hello world", "Hello world, this is a test.")
        r2 = reranker.score("hello world", "Hello world, this is a test.")
        assert r1.rerank_score == r2.rerank_score
        assert r1.final_score == r2.final_score

    def test_empty_query(self, reranker: DefaultReranker) -> None:
        result = reranker.score("", "Some content.", original_score=0.5)
        assert result.rerank_score == 0.0
        assert result.final_score == 0.5

    def test_empty_content(self, reranker: DefaultReranker) -> None:
        result = reranker.score("query", "", original_score=0.5)
        assert result.rerank_score == 0.0
        assert result.final_score == 0.5

    def test_unicode(self, reranker: DefaultReranker) -> None:
        result = reranker.score("capital", "Paris est la capitale de la France.")
        assert result.rerank_score >= 0
        assert result.final_score >= 0

    def test_partial_term_overlap(self, reranker: DefaultReranker) -> None:
        """Fewer matching terms → lower score."""
        r1 = reranker.score("cat dog bird", "cat dog bird fish")
        r2 = reranker.score("cat dog bird", "cat only")
        assert r1.rerank_score > r2.rerank_score


# ======================================================================
# DefaultReranker — rerank() with content_provider
# ======================================================================


class TestDefaultRerankerRerank:
    """Tests for the rerank() method with a content provider."""

    @pytest.fixture
    def content_map(self) -> dict[str, str]:
        return {
            "c1": "Paris is the capital of France.",
            "c2": "London is the capital of the UK.",
            "c3": "Python is a programming language.",
        }

    @pytest.fixture
    def reranker(self, content_map: dict[str, str]) -> DefaultReranker:
        return DefaultReranker(content_provider=content_map.get)  # type: ignore[arg-type]

    async def test_rerank_returns_results(self, reranker: DefaultReranker) -> None:
        results = [("c1", 0.9), ("c2", 0.8), ("c3", 0.5)]
        response = await reranker.rerank("capital", results)
        assert len(response.results) > 0
        assert isinstance(response, RerankResponse)

    async def test_rerank_ordering(self, reranker: DefaultReranker) -> None:
        """Higher-scoring chunks are ranked first."""
        results = [("c1", 0.9), ("c2", 0.8), ("c3", 0.5)]
        response = await reranker.rerank("capital", results)
        # c1 matches "capital" and "France" — should rank highest
        assert response.results[0].chunk_id == "c1"

    async def test_top_k(self, reranker: DefaultReranker) -> None:
        results = [("c1", 0.9), ("c2", 0.8), ("c3", 0.5)]
        response = await reranker.rerank("capital", results)
        assert len(response.results) <= len(results)

    async def test_custom_top_k(self) -> None:
        config = RerankConfig(top_k=2)
        content = {"c1": "Paris.", "c2": "London.", "c3": "Berlin."}
        reranker = DefaultReranker(config=config, content_provider=content.get)  # type: ignore[arg-type]
        results = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7)]
        response = await reranker.rerank("city", results)
        assert len(response.results) <= 2

    async def test_score_threshold(self) -> None:
        """Results below threshold are filtered out."""
        config = RerankConfig(top_k=10, score_threshold=1.0)
        content = {"c1": "Paris.", "c2": "London."}
        reranker = DefaultReranker(config=config, content_provider=content.get)  # type: ignore[arg-type]
        results = [("c1", 0.1), ("c2", 0.05)]
        response = await reranker.rerank("city", results)
        assert len(response.results) == 0

    async def test_deterministic_ordering(self, reranker: DefaultReranker) -> None:
        results = [("c1", 0.9), ("c2", 0.8), ("c3", 0.5)]
        r1 = await reranker.rerank("capital", list(results))
        r2 = await reranker.rerank("capital", list(results))
        for a, b in zip(r1.results, r2.results):
            assert a.chunk_id == b.chunk_id
            assert math.isclose(a.final_score, b.final_score)

    async def test_empty_results(self, reranker: DefaultReranker) -> None:
        response = await reranker.rerank("query", [])
        assert len(response.results) == 0

    async def test_empty_query(self, reranker: DefaultReranker) -> None:
        response = await reranker.rerank("", [("c1", 0.9)])
        assert len(response.results) == 0

    async def test_no_content_provider(self) -> None:
        """Without content provider, final_score = original_score."""
        reranker = DefaultReranker()
        results = [("c1", 0.9), ("c2", 0.5)]
        response = await reranker.rerank("capital", results)
        for r in response.results:
            assert math.isclose(r.final_score, r.original_score)
            assert r.rerank_score == 0.0

    async def test_missing_content(self, content_map: dict[str, str]) -> None:
        """Missing chunk content is handled gracefully."""
        reranker = DefaultReranker(content_provider=content_map.get)  # type: ignore[arg-type]
        results = [("c1", 0.9), ("nonexistent", 0.8)]
        response = await reranker.rerank("capital", results)
        assert len(response.results) == 2

    async def test_metadata(self, reranker: DefaultReranker) -> None:
        results = [("c1", 0.9)]
        response = await reranker.rerank("capital", results)
        meta = response.metadata
        assert "rerank_weight" in meta
        assert "total_candidates" in meta
        assert "returned" in meta
        assert "elapsed_ms" in meta

    async def test_unicode_content(self) -> None:
        content = {"c1": "Paris est la capitale de la France."}
        reranker = DefaultReranker(content_provider=content.get)  # type: ignore[arg-type]
        response = await reranker.rerank("capitale", [("c1", 0.9)])
        assert len(response.results) == 1

    async def test_scoring_improves_ranking(self) -> None:
        """Reranking should promote chunks with better content match."""
        content = {
            "irrelevant": "The weather today is sunny.",
            "relevant": "The capital of France is Paris.",
        }
        reranker = DefaultReranker(content_provider=content.get)  # type: ignore[arg-type]
        # Both start with same original score
        results = [("irrelevant", 1.0), ("relevant", 1.0)]
        response = await reranker.rerank("capital France Paris", results)
        # "relevant" should now be ranked first after reranking
        assert response.results[0].chunk_id == "relevant"


# ======================================================================
# DefaultReranker — rerank() with custom weights
# ======================================================================


class TestDefaultRerankerWeights:
    async def test_custom_rerank_weight(self) -> None:
        """Higher rerank_weight increases the impact of reranking."""
        content = {"c1": "Paris is the capital of France.", "c2": "Unrelated text."}
        base = DefaultReranker(content_provider=content.get, rerank_weight=1.0)  # type: ignore[arg-type]
        heavy = DefaultReranker(content_provider=content.get, rerank_weight=5.0)  # type: ignore[arg-type]

        results = [("c1", 0.5), ("c2", 0.5)]
        r_base = await base.rerank("capital France", results)
        r_heavy = await heavy.rerank("capital France", results)

        # The reranked scores should differ
        assert not math.isclose(r_base.results[0].final_score,
                                r_heavy.results[0].final_score)


# ======================================================================
# DefaultReranker — architecture
# ======================================================================


class TestDefaultRerankerArch:
    def test_subclass_of_reranker(self) -> None:
        assert issubclass(DefaultReranker, Reranker)

    def test_default_config(self) -> None:
        reranker = DefaultReranker()
        assert reranker.config.enabled is True
        assert reranker.config.top_k == 10

    def test_custom_config(self) -> None:
        config = RerankConfig(top_k=3)
        reranker = DefaultReranker(config=config)
        assert reranker.config.top_k == 3

    def test_rerank_weight_property(self) -> None:
        reranker = DefaultReranker(rerank_weight=2.0)
        assert reranker.rerank_weight == 2.0

    def test_content_provider_property(self) -> None:
        provider = lambda x: x  # noqa: E731
        reranker = DefaultReranker(content_provider=provider)
        assert reranker.content_provider is provider


# ======================================================================
# Error hierarchy
# ======================================================================


class TestRerankErrors:
    def test_rerank_error_message(self) -> None:
        err = RerankError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.code == "RERANK_ERROR"

    def test_invalid_configuration_error(self) -> None:
        err = InvalidRerankConfiguration("Bad config")
        assert err.code == "INVALID_RERANK_CONFIGURATION"

    def test_reranker_not_found_with_name(self) -> None:
        err = RerankerNotFound("cross_encoder")
        assert "cross_encoder" in str(err)

    def test_reranker_not_found_empty(self) -> None:
        err = RerankerNotFound()
        assert str(err) == "Reranker not found"

    def test_to_dict(self) -> None:
        err = InvalidRerankConfiguration("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "INVALID_RERANK_CONFIGURATION"

    def test_knowledge_error_is_base(self) -> None:
        assert issubclass(RerankError, KnowledgeError)


# ======================================================================
# KnowledgeBase reranker integration
# ======================================================================


class TestKnowledgeBaseReranker:
    """Reranker integration with KnowledgeBase."""

    def test_no_reranker_by_default(self) -> None:
        from app.rag.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        assert kb.reranker is None

    def test_reranker_property(self) -> None:
        from app.rag.knowledge_base import KnowledgeBase
        reranker = DefaultReranker()
        kb = KnowledgeBase(reranker=reranker)
        assert kb.reranker is reranker

    async def test_keyword_reranker_pipeline(self) -> None:
        """Keyword retrieval + reranker produces reranked context."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeDocument, KnowledgeChunk
        from app.rag.context import KnowledgeContextBuilder

        reranker = DefaultReranker()
        kb = KnowledgeBase(reranker=reranker)
        chunk = KnowledgeChunk(chunk_id="c1", document_id="d1",
                               content="Paris is the capital of France.")
        kb.register(KnowledgeDocument(document_id="d1", chunks=(chunk,)))

        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="capital France", max_chunks=5)
        assert context.total_chunks > 0
        assert len(context.text) > 0

    async def test_reranker_disabled(self) -> None:
        """Reranker with enabled=False passes through chunks unchanged."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeDocument, KnowledgeChunk
        from app.rag.context import KnowledgeContextBuilder

        config = RerankConfig(enabled=False)
        reranker = DefaultReranker(config=config)
        kb = KnowledgeBase(reranker=reranker)
        chunk = KnowledgeChunk(chunk_id="c1", document_id="d1",
                               content="Paris is the capital of France.")
        kb.register(KnowledgeDocument(document_id="d1", chunks=(chunk,)))

        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="capital France", max_chunks=5)
        assert context.total_chunks > 0

    async def test_empty_results(self) -> None:
        """Reranker handles empty retrieval results."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.context import KnowledgeContextBuilder

        reranker = DefaultReranker()
        kb = KnowledgeBase(reranker=reranker)
        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="anything", max_chunks=5)
        assert context.total_chunks == 0

    async def test_hybrid_reranker_pipeline(self) -> None:
        """Hybrid retrieval + reranker works end-to-end."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.embeddings import DeterministicEmbeddingProvider, EmbeddingConfig as EConfig
        from app.rag.vectorstore import MemoryVectorStore, VectorStoreConfig, SimilarityMetric
        from app.rag.context import KnowledgeContextBuilder

        reranker = DefaultReranker()
        emb_config = EConfig(provider_name="det", dimensions=4, normalize_embeddings=True)
        provider = DeterministicEmbeddingProvider(emb_config)
        vs = MemoryVectorStore(config=VectorStoreConfig(metric=SimilarityMetric.COSINE))
        kb = KnowledgeBase(
            embedding_provider=provider,
            vector_store=vs,
            reranker=reranker,
        )

        from app.rag.models import KnowledgeDocument, KnowledgeChunk
        chunk = KnowledgeChunk(chunk_id="c1", document_id="d1",
                               content="Paris is the capital of France.")
        kb.register(KnowledgeDocument(document_id="d1", chunks=(chunk,)))
        vec = provider._generate(chunk.content)
        from app.rag.embeddings.models import EmbeddingVector
        kb._embeddings["c1"] = EmbeddingVector(
            vector=vec, dimensions=len(vec), provider="det",
        )
        vs.add("c1", vec)

        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="capital France", max_chunks=5)
        assert context.total_chunks > 0
        assert len(context.text) > 0

    async def test_unicode(self) -> None:
        """Reranker works with unicode content."""
        from app.rag.knowledge_base import KnowledgeBase
        from app.rag.models import KnowledgeDocument, KnowledgeChunk
        from app.rag.context import KnowledgeContextBuilder

        reranker = DefaultReranker()
        kb = KnowledgeBase(reranker=reranker)
        chunk = KnowledgeChunk(chunk_id="c1", document_id="d1",
                               content="Paris est la capitale de la France.")
        kb.register(KnowledgeDocument(document_id="d1", chunks=(chunk,)))

        builder = KnowledgeContextBuilder(kb)
        context = await builder.build(query="capitale", max_chunks=5)
        assert context.total_chunks > 0
