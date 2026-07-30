"""Architecture tests for the hybrid retrieval layer.

Checkpoint 1 — verifies imports, configuration, fusion strategy enum,
models, fusion functions, error hierarchy, and immutability.
"""

from __future__ import annotations

import pytest

from app.rag.hybrid import (
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

    def test_str(self) -> None:
        assert str(FusionStrategy.WEIGHTED_SUM) == "FusionStrategy.WEIGHTED_SUM"


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
        cfg = HybridConfig(keyword_weight=0.2, semantic_weight=0.8, max_candidates=10)
        cfg.validate()

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

    def test_validate_max_candidates_negative(self) -> None:
        with pytest.raises(InvalidHybridConfiguration):
            HybridConfig(max_candidates=-1).validate()

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
        s = RetrievalScore(
            chunk_id="c1",
            keyword_score=0.8,
            semantic_score=0.6,
            final_score=0.7,
        )
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
        result = HybridResult(
            results=scores,
            metadata={"strategy": "weighted_sum", "candidates": 5},
        )
        assert len(result.results) == 1
        assert result.results[0].chunk_id == "c1"
        assert result.metadata["strategy"] == "weighted_sum"

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

    def test_concrete_subclass(self) -> None:
        class TestRetriever(HybridRetriever):
            async def retrieve(self, query: str, top_k: int = 5) -> HybridResult:
                return HybridResult()

        retriever = TestRetriever()
        assert retriever.config.keyword_weight == 0.5
        assert retriever.config.fusion_strategy == FusionStrategy.WEIGHTED_SUM

    def test_custom_config(self) -> None:
        class TestRetriever(HybridRetriever):
            async def retrieve(self, query: str, top_k: int = 5) -> HybridResult:
                return HybridResult()

        cfg = HybridConfig(fusion_strategy=FusionStrategy.RECIPROCAL_RANK_FUSION)
        retriever = TestRetriever(config=cfg)
        assert retriever.config.fusion_strategy == FusionStrategy.RECIPROCAL_RANK_FUSION

    def test_config_validation(self) -> None:
        class ValidatingRetriever(HybridRetriever):
            def __init__(self, config: HybridConfig) -> None:
                config.validate()
                super().__init__(config)

            async def retrieve(self, query: str, top_k: int = 5) -> HybridResult:
                return HybridResult()

        with pytest.raises(InvalidHybridConfiguration):
            ValidatingRetriever(HybridConfig(keyword_weight=-1.0))


# ======================================================================
# weighted_sum
# ======================================================================


class TestWeightedSum:
    def test_equal_weights(self) -> None:
        kw = {"c1": 1.0, "c2": 0.5}
        sem = {"c2": 0.8, "c3": 0.3}
        result = weighted_sum(kw, sem, keyword_weight=0.5, semantic_weight=0.5)
        assert result["c1"] == pytest.approx(0.5)  # 0.5*1.0 + 0.5*0.0
        assert result["c2"] == pytest.approx(0.65)  # 0.5*0.5 + 0.5*0.8
        assert result["c3"] == pytest.approx(0.15)  # 0.5*0.0 + 0.5*0.3

    def test_keyword_only(self) -> None:
        kw = {"c1": 1.0}
        sem = {}
        result = weighted_sum(kw, sem, keyword_weight=1.0, semantic_weight=0.0)
        assert result["c1"] == pytest.approx(1.0)

    def test_semantic_only(self) -> None:
        kw = {}
        sem = {"c1": 0.9}
        result = weighted_sum(kw, sem, keyword_weight=0.0, semantic_weight=1.0)
        assert result["c1"] == pytest.approx(0.9)

    def test_skewed_weights(self) -> None:
        kw = {"c1": 1.0}
        sem = {"c1": 0.5}
        result = weighted_sum(kw, sem, keyword_weight=0.8, semantic_weight=0.2)
        assert result["c1"] == pytest.approx(0.9)  # 0.8 + 0.1

    def test_empty_scores(self) -> None:
        result = weighted_sum({}, {})
        assert result == {}

    def test_all_weights_zero_raises(self) -> None:
        with pytest.raises(FusionError, match="non-zero"):
            weighted_sum({}, {}, keyword_weight=0.0, semantic_weight=0.0)

    def test_preserves_ids(self) -> None:
        kw = {"a": 0.5, "b": 0.3}
        sem = {"b": 0.7, "c": 0.2}
        result = weighted_sum(kw, sem)
        assert set(result) == {"a", "b", "c"}


# ======================================================================
# reciprocal_rank_fusion
# ======================================================================


class TestReciprocalRankFusion:
    def test_basic(self) -> None:
        kw = ["c1", "c2", "c3"]
        sem = ["c2", "c1", "c4"]
        result = reciprocal_rank_fusion(kw, sem, k=60)
        assert "c1" in result
        assert "c2" in result
        assert "c3" in result
        assert "c4" in result
        # c2 appears in both at rank 2 → higher score than c3 (only in kw)
        assert result["c2"] > result["c3"]

    def test_identical_rankings(self) -> None:
        kw = ["c1", "c2", "c3"]
        sem = ["c1", "c2", "c3"]
        result = reciprocal_rank_fusion(kw, sem, k=60)
        # c1: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.0328
        # c3: 1/(60+3) + 1/(60+3) = 2/63 ≈ 0.0317
        assert result["c1"] > result["c3"]

    def test_only_keyword(self) -> None:
        kw = ["c1", "c2"]
        sem = []
        result = reciprocal_rank_fusion(kw, sem, k=60)
        # c1: 1/61 + 1/(60+3) (sem rank = len(sem)+1 = 3)
        assert len(result) == 2

    def test_only_semantic(self) -> None:
        kw = []
        sem = ["c1", "c2"]
        result = reciprocal_rank_fusion(kw, sem, k=60)
        assert len(result) == 2

    def test_empty_lists(self) -> None:
        result = reciprocal_rank_fusion([], [], k=60)
        assert result == {}

    def test_custom_k(self) -> None:
        kw = ["c1"]
        sem = ["c1"]
        result = reciprocal_rank_fusion(kw, sem, k=1)
        # c1: 1/(1+1) + 1/(1+1) = 1.0
        assert result["c1"] == pytest.approx(1.0)

    def test_ordering(self) -> None:
        """RRF results have correct ordering."""
        kw = ["c3", "c1", "c2"]
        sem = ["c2", "c3", "c1"]
        result = reciprocal_rank_fusion(kw, sem, k=60)
        # All items should have scores
        assert set(result) == {"c1", "c2", "c3"}
        # c3: kw_rank=1, sem_rank=2 → 1/61 + 1/62
        # c2: kw_rank=3, sem_rank=1 → 1/63 + 1/61
        # c1: kw_rank=2, sem_rank=3 → 1/62 + 1/63
        # c3 should have highest score, then c2, then c1
        assert result["c3"] > result["c1"]


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
        assert isinstance(err, HybridError)

    def test_to_dict(self) -> None:
        err = InvalidHybridConfiguration("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "INVALID_HYBRID_CONFIGURATION"
        assert d["message"] == "test"
        assert d["details"] == {"key": "val"}

    def test_knowledge_error_is_base(self) -> None:
        assert issubclass(HybridError, KnowledgeError)
