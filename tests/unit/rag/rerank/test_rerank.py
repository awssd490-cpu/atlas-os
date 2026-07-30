"""Tests for the reranking architecture.

Checkpoint 1 — verifies imports, configuration, models, registry,
error hierarchy, and abstract base class.
"""

from __future__ import annotations

import pytest

from app.rag.rerank import (
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
        cfg = RerankConfig(top_k=1, score_threshold=0.0)
        cfg.validate()
        cfg = RerankConfig(top_k=100, score_threshold=1.0)
        cfg.validate()

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
        r = RerankedResult(
            chunk_id="c1",
            original_score=0.8,
            rerank_score=0.9,
            final_score=0.85,
        )
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
        response = RerankResponse(
            results=results,
            metadata={"model": "cross_encoder", "elapsed_ms": 5.2},
        )
        assert len(response.results) == 1
        assert response.results[0].chunk_id == "c1"
        assert response.metadata["model"] == "cross_encoder"

    def test_immutable(self) -> None:
        r = RerankResponse()
        with pytest.raises(AttributeError):
            r.results = ()  # type: ignore[misc]


# ======================================================================
# Reranker (abstract)
# ======================================================================


class TestReranker:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            Reranker()  # type: ignore[abstract]

    def test_abstract_methods(self) -> None:
        assert hasattr(Reranker, "rerank")

    def test_concrete_subclass(self) -> None:
        class TestRerankerImpl(Reranker):
            async def rerank(
                self,
                query: str,
                results: list[tuple[str, float]],  # type: ignore[override]
            ) -> RerankResponse:
                return RerankResponse()

        reranker = TestRerankerImpl()
        assert reranker.config.enabled is True
        assert reranker.config.top_k == 10

    def test_custom_config(self) -> None:
        class TestRerankerImpl(Reranker):
            async def rerank(
                self,
                query: str,
                results: list[tuple[str, float]],  # type: ignore[override]
            ) -> RerankResponse:
                return RerankResponse()

        cfg = RerankConfig(top_k=5, score_threshold=0.2)
        reranker = TestRerankerImpl(config=cfg)
        assert reranker.config.top_k == 5
        assert reranker.config.score_threshold == 0.2

    def test_config_validation(self) -> None:
        class ValidatingReranker(Reranker):
            def __init__(self, config: RerankConfig) -> None:
                config.validate()
                super().__init__(config)

            async def rerank(
                self,
                query: str,
                results: list[tuple[str, float]],  # type: ignore[override]
            ) -> RerankResponse:
                return RerankResponse()

        with pytest.raises(InvalidRerankConfiguration):
            ValidatingReranker(RerankConfig(top_k=0))


# ======================================================================
# Registry
# ======================================================================


class TestRerankerRegistry:
    def test_register_and_get(self) -> None:
        class FakeReranker(Reranker):
            async def rerank(
                self,
                query: str,
                results: list[tuple[str, float]],  # type: ignore[override]
            ) -> RerankResponse:
                return RerankResponse()

        register_reranker("fake", FakeReranker)
        cls = get_reranker("fake")
        assert cls is FakeReranker
        clear_rerankers()

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(RerankerNotFound):
            get_reranker("nonexistent")

    def test_register_duplicate_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.rag.rerank.registry._rerankers", {})

        class P1(Reranker):
            async def rerank(
                self,
                query: str,
                results: list[tuple[str, float]],  # type: ignore[override]
            ) -> RerankResponse:
                return RerankResponse()

        class P2(Reranker):
            async def rerank(
                self,
                query: str,
                results: list[tuple[str, float]],  # type: ignore[override]
            ) -> RerankResponse:
                return RerankResponse()

        register_reranker("dup", P1)
        with pytest.raises(ValueError, match="already registered"):
            register_reranker("dup", P2)

    def test_list_rerankers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.rag.rerank.registry._rerankers", {})

        class P(Reranker):
            async def rerank(
                self,
                query: str,
                results: list[tuple[str, float]],  # type: ignore[override]
            ) -> RerankResponse:
                return RerankResponse()

        assert list_rerankers() == []
        register_reranker("a", P)
        register_reranker("b", P)
        assert set(list_rerankers()) == {"a", "b"}

    def test_clear_rerankers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("app.rag.rerank.registry._rerankers", {})

        class P(Reranker):
            async def rerank(
                self,
                query: str,
                results: list[tuple[str, float]],  # type: ignore[override]
            ) -> RerankResponse:
                return RerankResponse()

        register_reranker("p", P)
        assert list_rerankers() == ["p"]
        clear_rerankers()
        assert list_rerankers() == []


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
        assert isinstance(err, RerankError)

    def test_reranker_not_found_with_name(self) -> None:
        err = RerankerNotFound("cross_encoder")
        assert "cross_encoder" in str(err)
        assert err.code == "RERANKER_NOT_FOUND"

    def test_reranker_not_found_empty(self) -> None:
        err = RerankerNotFound()
        assert str(err) == "Reranker not found"
        assert err.code == "RERANKER_NOT_FOUND"

    def test_to_dict(self) -> None:
        err = InvalidRerankConfiguration("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "INVALID_RERANK_CONFIGURATION"
        assert d["message"] == "test"
        assert d["details"] == {"key": "val"}

    def test_knowledge_error_is_base(self) -> None:
        assert issubclass(RerankError, KnowledgeError)
