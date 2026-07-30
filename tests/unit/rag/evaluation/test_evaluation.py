"""Tests for the evaluation architecture."""

from __future__ import annotations

from typing import Any

import pytest

from app.rag.evaluation import (
    BenchmarkResult,
    EvaluationConfig,
    EvaluationError,
    EvaluationNotFound,
    EvaluationResult,
    EvaluationRunner,
    InvalidEvaluationConfiguration,
    RetrievalMetrics,
    clear_runners,
    get,
    list_runners,
    register,
    unregister,
)
from app.rag.evaluation.base import EvaluationRunner as EvaluationRunner_Impl
from app.rag.evaluation.config import EvaluationConfig as EvaluationConfig_Impl
from app.rag.evaluation.errors import EvaluationError as EvaluationError_Impl
from app.rag.evaluation.errors import EvaluationNotFound as EvaluationNotFound_Impl
from app.rag.evaluation.models import BenchmarkResult as BenchmarkResult_Impl
from app.rag.evaluation.models import EvaluationResult as EvaluationResult_Impl
from app.rag.evaluation.retrieval_metrics import RetrievalMetrics as RetrievalMetrics_Impl
from app.rag.errors import KnowledgeError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_evaluation_config_imported(self) -> None:
        assert EvaluationConfig is EvaluationConfig_Impl

    def test_evaluation_error_imported(self) -> None:
        assert EvaluationError is EvaluationError_Impl

    def test_evaluation_not_found_imported(self) -> None:
        assert EvaluationNotFound is EvaluationNotFound_Impl

    def test_evaluation_runner_imported(self) -> None:
        assert EvaluationRunner is EvaluationRunner_Impl

    def test_evaluation_result_imported(self) -> None:
        assert EvaluationResult is EvaluationResult_Impl

    def test_benchmark_result_imported(self) -> None:
        assert BenchmarkResult is BenchmarkResult_Impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(EvaluationError, KnowledgeError)
        assert issubclass(InvalidEvaluationConfiguration, EvaluationError)
        assert issubclass(EvaluationNotFound, EvaluationError)

    def test_registry_functions_imported(self) -> None:
        assert callable(register)
        assert callable(unregister)
        assert callable(get)
        assert callable(list_runners)
        assert callable(clear_runners)

    def test_retrieval_metrics_imported(self) -> None:
        assert RetrievalMetrics is RetrievalMetrics_Impl


# ======================================================================
# EvaluationConfig
# ======================================================================


class TestEvaluationConfig:
    def test_default_values(self) -> None:
        cfg = EvaluationConfig()
        assert cfg.enabled is True
        assert cfg.metrics == ()
        assert cfg.warmup_runs == 3
        assert cfg.benchmark_runs == 10
        assert cfg.random_seed == 42

    def test_custom_values(self) -> None:
        cfg = EvaluationConfig(
            enabled=False,
            metrics=("precision", "recall"),
            warmup_runs=5,
            benchmark_runs=20,
            random_seed=7,
        )
        assert cfg.enabled is False
        assert cfg.metrics == ("precision", "recall")
        assert cfg.warmup_runs == 5
        assert cfg.benchmark_runs == 20
        assert cfg.random_seed == 7

    def test_immutable(self) -> None:
        cfg = EvaluationConfig()
        with pytest.raises(AttributeError):
            cfg.benchmark_runs = 5  # type: ignore[misc]

    def test_validate_passes(self) -> None:
        EvaluationConfig(benchmark_runs=1, warmup_runs=0).validate()
        EvaluationConfig(benchmark_runs=100, warmup_runs=10).validate()

    def test_validate_benchmark_runs_zero(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration):
            EvaluationConfig(benchmark_runs=0).validate()

    def test_validate_benchmark_runs_negative(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration):
            EvaluationConfig(benchmark_runs=-1).validate()

    def test_validate_warmup_runs_negative(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration):
            EvaluationConfig(warmup_runs=-1).validate()


# ======================================================================
# EvaluationResult
# ======================================================================


class TestEvaluationResult:
    def test_default_values(self) -> None:
        r = EvaluationResult()
        assert r.score == 0.0
        assert r.metrics == {}
        assert r.metadata == {}
        assert r.duration == 0.0

    def test_custom_values(self) -> None:
        r = EvaluationResult(
            score=0.85,
            metrics={"precision": 0.9, "recall": 0.8},
            metadata={"component": "retrieval", "dataset": "test"},
            duration=12.5,
        )
        assert r.score == 0.85
        assert r.metrics == {"precision": 0.9, "recall": 0.8}
        assert r.metadata == {"component": "retrieval", "dataset": "test"}
        assert r.duration == 12.5

    def test_immutable(self) -> None:
        r = EvaluationResult()
        with pytest.raises(AttributeError):
            r.score = 0.5  # type: ignore[misc]


# ======================================================================
# BenchmarkResult
# ======================================================================


class TestBenchmarkResult:
    def test_default_values(self) -> None:
        r = BenchmarkResult()
        assert r.latency_ms == 0.0
        assert r.throughput == 0.0
        assert r.memory_bytes == 0
        assert r.metadata == {}

    def test_custom_values(self) -> None:
        r = BenchmarkResult(
            latency_ms=45.2,
            throughput=220.0,
            memory_bytes=1048576,
            metadata={"batch_size": 32},
        )
        assert r.latency_ms == 45.2
        assert r.throughput == 220.0
        assert r.memory_bytes == 1048576
        assert r.metadata == {"batch_size": 32}

    def test_immutable(self) -> None:
        r = BenchmarkResult()
        with pytest.raises(AttributeError):
            r.latency_ms = 10.0  # type: ignore[misc]


# ======================================================================
# EvaluationRunner ABC + Registry
# ======================================================================


class TestEvaluationRunner:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            EvaluationRunner()  # type: ignore[abstract]

    def test_abstract_methods(self) -> None:
        assert hasattr(EvaluationRunner, "evaluate")
        assert hasattr(EvaluationRunner, "benchmark")
        assert hasattr(EvaluationRunner, "profile")

    def test_default_config(self) -> None:
        class MinimalRunner(EvaluationRunner):
            async def evaluate(
                self, component: object, dataset: object, **kwargs: Any
            ) -> EvaluationResult:
                return EvaluationResult()

            async def benchmark(
                self, component: object, dataset: object, **kwargs: Any
            ) -> BenchmarkResult:
                return BenchmarkResult()

            async def profile(
                self, component: object, dataset: object, **kwargs: Any
            ) -> BenchmarkResult:
                return BenchmarkResult()

        runner = MinimalRunner()
        assert isinstance(runner.config, EvaluationConfig)
        assert runner.config.enabled is True
        assert runner.config.benchmark_runs == 10

    def test_custom_config(self) -> None:
        class MinimalRunner(EvaluationRunner):
            async def evaluate(
                self, component: object, dataset: object, **kwargs: Any
            ) -> EvaluationResult:
                return EvaluationResult()

            async def benchmark(
                self, component: object, dataset: object, **kwargs: Any
            ) -> BenchmarkResult:
                return BenchmarkResult()

            async def profile(
                self, component: object, dataset: object, **kwargs: Any
            ) -> BenchmarkResult:
                return BenchmarkResult()

        config = EvaluationConfig(benchmark_runs=25, warmup_runs=5)
        runner = MinimalRunner(config=config)
        assert runner.config.benchmark_runs == 25
        assert runner.config.warmup_runs == 5


class TestEvaluationRegistry:
    def test_register_and_get(self) -> None:
        class FakeRunner(EvaluationRunner):
            async def evaluate(
                self, component: object, dataset: object, **kwargs: Any
            ) -> EvaluationResult:
                return EvaluationResult()
            async def benchmark(
                self, component: object, dataset: object, **kwargs: Any
            ) -> BenchmarkResult:
                return BenchmarkResult()
            async def profile(
                self, component: object, dataset: object, **kwargs: Any
            ) -> BenchmarkResult:
                return BenchmarkResult()

        register("fake", FakeRunner)
        assert get("fake") is FakeRunner
        clear_runners()

    def test_register_duplicate_raises(self) -> None:
        class R1(EvaluationRunner):
            async def evaluate(self, component: object, dataset: object, **kwargs: Any) -> EvaluationResult:
                return EvaluationResult()
            async def benchmark(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()
            async def profile(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()

        class R2(EvaluationRunner):
            async def evaluate(self, component: object, dataset: object, **kwargs: Any) -> EvaluationResult:
                return EvaluationResult()
            async def benchmark(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()
            async def profile(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()

        register("dup", R1)
        with pytest.raises(ValueError, match="already registered"):
            register("dup", R2)
        clear_runners()

    def test_get_unknown_raises(self) -> None:
        with pytest.raises(EvaluationNotFound):
            get("nonexistent")

    def test_unregister(self) -> None:
        class R(EvaluationRunner):
            async def evaluate(self, component: object, dataset: object, **kwargs: Any) -> EvaluationResult:
                return EvaluationResult()
            async def benchmark(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()
            async def profile(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()

        register("to_remove", R)
        unregister("to_remove")
        assert "to_remove" not in list_runners()
        clear_runners()

    def test_unregister_unknown_raises(self) -> None:
        with pytest.raises(EvaluationNotFound):
            unregister("nonexistent")

    def test_list_runners(self) -> None:
        class R(EvaluationRunner):
            async def evaluate(self, component: object, dataset: object, **kwargs: Any) -> EvaluationResult:
                return EvaluationResult()
            async def benchmark(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()
            async def profile(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()

        register("a", R)
        register("b", R)
        names = list_runners()
        assert "a" in names
        assert "b" in names
        clear_runners()

    def test_clear_runners(self) -> None:
        class R(EvaluationRunner):
            async def evaluate(self, component: object, dataset: object, **kwargs: Any) -> EvaluationResult:
                return EvaluationResult()
            async def benchmark(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()
            async def profile(self, component: object, dataset: object, **kwargs: Any) -> BenchmarkResult:
                return BenchmarkResult()

        register("x", R)
        clear_runners()
        assert list_runners() == []


# ======================================================================
# Error hierarchy
# ======================================================================


class TestEvaluationErrors:
    def test_evaluation_error_message(self) -> None:
        err = EvaluationError("Something went wrong")
        assert str(err) == "Something went wrong"
        assert err.code == "EVALUATION_ERROR"

    def test_invalid_configuration_error(self) -> None:
        err = InvalidEvaluationConfiguration("Bad config")
        assert err.code == "INVALID_EVALUATION_CONFIGURATION"

    def test_evaluation_not_found_with_name(self) -> None:
        err = EvaluationNotFound("retrieval_runner")
        assert "retrieval_runner" in str(err)

    def test_evaluation_not_found_empty(self) -> None:
        err = EvaluationNotFound()
        assert str(err) == "Evaluation runner not found"

    def test_to_dict(self) -> None:
        err = InvalidEvaluationConfiguration("test", details={"key": "val"})
        d = err.to_dict()
        assert d["code"] == "INVALID_EVALUATION_CONFIGURATION"

    def test_knowledge_error_is_base(self) -> None:
        assert issubclass(EvaluationError, KnowledgeError)


# ======================================================================
# RetrievalMetrics — precision_at_k
# ======================================================================


class TestPrecisionAtK:
    """precision_at_k tests."""

    @pytest.fixture
    def metrics(self) -> RetrievalMetrics:
        return RetrievalMetrics()

    def test_all_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert result == 1.0

    def test_partial_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.precision_at_k(["a", "b", "c", "d"], {"a", "c"}, k=4)
        assert result == 0.5

    def test_none_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.precision_at_k(["a", "b"], {"c", "d"}, k=2)
        assert result == 0.0

    def test_k_smaller_than_list(self, metrics: RetrievalMetrics) -> None:
        result = metrics.precision_at_k(["a", "b", "c", "d"], {"a", "d"}, k=2)
        assert result == 0.5

    def test_k_larger_than_list(self, metrics: RetrievalMetrics) -> None:
        result = metrics.precision_at_k(["a", "b"], {"a", "c"}, k=10)
        assert result == 0.1

    def test_k_is_zero(self, metrics: RetrievalMetrics) -> None:
        assert metrics.precision_at_k(["a", "b"], {"a"}, k=0) == 0.0

    def test_k_is_negative(self, metrics: RetrievalMetrics) -> None:
        assert metrics.precision_at_k(["a", "b"], {"a"}, k=-1) == 0.0

    def test_empty_retrieved(self, metrics: RetrievalMetrics) -> None:
        assert metrics.precision_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.precision_at_k(["a", "b"], set(), k=2)
        assert result == 0.0

    def test_unicode_ids(self, metrics: RetrievalMetrics) -> None:
        result = metrics.precision_at_k(["東京", "巴黎", "倫敦"], {"東京", "倫敦"}, k=3)
        assert result == pytest.approx(2.0 / 3.0)

    def test_deterministic(self, metrics: RetrievalMetrics) -> None:
        a = metrics.precision_at_k(["x", "y"], {"x"}, k=2)
        b = metrics.precision_at_k(["x", "y"], {"x"}, k=2)
        assert a == b


# ======================================================================
# RetrievalMetrics — recall_at_k
# ======================================================================


class TestRecallAtK:
    """recall_at_k tests."""

    @pytest.fixture
    def metrics(self) -> RetrievalMetrics:
        return RetrievalMetrics()

    def test_all_relevant_found(self, metrics: RetrievalMetrics) -> None:
        result = metrics.recall_at_k(["a", "b", "c"], {"a", "b"}, k=3)
        assert result == 1.0

    def test_partial_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.recall_at_k(["a", "b"], {"a", "c", "d"}, k=2)
        assert result == 1.0 / 3.0

    def test_none_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.recall_at_k(["a", "b"], {"c"}, k=2)
        assert result == 0.0

    def test_k_smaller_than_list(self, metrics: RetrievalMetrics) -> None:
        result = metrics.recall_at_k(["a", "b", "c"], {"a", "c"}, k=1)
        assert result == 0.5

    def test_k_larger_than_list(self, metrics: RetrievalMetrics) -> None:
        result = metrics.recall_at_k(["a", "b"], {"a", "c"}, k=10)
        assert result == 0.5

    def test_k_is_zero(self, metrics: RetrievalMetrics) -> None:
        assert metrics.recall_at_k(["a", "b"], {"a"}, k=0) == 0.0

    def test_k_is_negative(self, metrics: RetrievalMetrics) -> None:
        assert metrics.recall_at_k(["a", "b"], {"a"}, k=-1) == 0.0

    def test_empty_retrieved(self, metrics: RetrievalMetrics) -> None:
        assert metrics.recall_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.recall_at_k(["a", "b"], set(), k=2)
        assert result == 1.0

    def test_unicode_ids(self, metrics: RetrievalMetrics) -> None:
        result = metrics.recall_at_k(["東京", "巴黎"], {"東京", "倫敦"}, k=2)
        assert result == 0.5

    def test_deterministic(self, metrics: RetrievalMetrics) -> None:
        a = metrics.recall_at_k(["x", "y"], {"x"}, k=2)
        b = metrics.recall_at_k(["x", "y"], {"x"}, k=2)
        assert a == b


# ======================================================================
# RetrievalMetrics — f1_at_k
# ======================================================================


class TestF1AtK:
    """F1 at k tests."""

    @pytest.fixture
    def metrics(self) -> RetrievalMetrics:
        return RetrievalMetrics()

    def test_perfect_precision_recall(self, metrics: RetrievalMetrics) -> None:
        result = metrics.f1_at_k(["a", "b"], {"a", "b"}, k=2)
        assert result == 1.0

    def test_harmonic_mean(self, metrics: RetrievalMetrics) -> None:
        # P=0.5, R=1.0 → F1 = 2*0.5*1.0 / (0.5+1.0) = 1.0/1.5 = 0.667
        result = metrics.f1_at_k(["a", "b", "c", "d"], {"a", "b"}, k=4)
        assert result == pytest.approx(2.0 / 3.0)

    def test_no_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.f1_at_k(["a", "b"], {"c"}, k=2)
        assert result == 0.0

    def test_empty_retrieved(self, metrics: RetrievalMetrics) -> None:
        assert metrics.f1_at_k([], {"a"}, k=5) == 0.0

    def test_empty_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.f1_at_k(["a"], set(), k=1)
        assert result == 0.0

    def test_unbalanced(self, metrics: RetrievalMetrics) -> None:
        # P=0.5 (1/2), R=0.5 (1/2) → F1 = 0.5
        result = metrics.f1_at_k(["a", "b"], {"a", "c"}, k=2)
        assert result == 0.5

    def test_k_is_zero(self, metrics: RetrievalMetrics) -> None:
        assert metrics.f1_at_k(["a"], {"a"}, k=0) == 0.0

    def test_deterministic(self, metrics: RetrievalMetrics) -> None:
        a = metrics.f1_at_k(["x", "y"], {"x"}, k=2)
        b = metrics.f1_at_k(["x", "y"], {"x"}, k=2)
        assert a == b


# ======================================================================
# RetrievalMetrics — mean_reciprocal_rank
# ======================================================================


class TestMeanReciprocalRank:
    """MRR tests."""

    @pytest.fixture
    def metrics(self) -> RetrievalMetrics:
        return RetrievalMetrics()

    def test_first_is_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.mean_reciprocal_rank(["a", "b", "c"], {"a"})
        assert result == 1.0

    def test_second_is_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.mean_reciprocal_rank(["a", "b", "c"], {"b"})
        assert result == 0.5

    def test_third_is_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.mean_reciprocal_rank(["a", "b", "c"], {"c"})
        assert result == 1.0 / 3.0

    def test_none_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.mean_reciprocal_rank(["a", "b"], {"c"})
        assert result == 0.0

    def test_first_of_multiple(self, metrics: RetrievalMetrics) -> None:
        result = metrics.mean_reciprocal_rank(["a", "b", "c"], {"b", "c"})
        assert result == 0.5

    def test_empty_retrieved(self, metrics: RetrievalMetrics) -> None:
        assert metrics.mean_reciprocal_rank([], {"a"}) == 0.0

    def test_empty_relevant(self, metrics: RetrievalMetrics) -> None:
        assert metrics.mean_reciprocal_rank(["a", "b"], set()) == 0.0

    def test_relevant_at_first_rank(self, metrics: RetrievalMetrics) -> None:
        result = metrics.mean_reciprocal_rank(["a", "b", "c"], {"a", "b"})
        assert result == 1.0

    def test_deterministic(self, metrics: RetrievalMetrics) -> None:
        a = metrics.mean_reciprocal_rank(["x", "y", "z"], {"y"})
        b = metrics.mean_reciprocal_rank(["x", "y", "z"], {"y"})
        assert a == b


# ======================================================================
# RetrievalMetrics — average_precision
# ======================================================================


class TestAveragePrecision:
    """Average precision (AP) tests."""

    @pytest.fixture
    def metrics(self) -> RetrievalMetrics:
        return RetrievalMetrics()

    def test_all_relevant_at_top(self, metrics: RetrievalMetrics) -> None:
        result = metrics.average_precision(["a", "b", "c"], {"a", "b", "c"})
        assert result == 1.0

    def test_standard_case(self, metrics: RetrievalMetrics) -> None:
        result = metrics.average_precision(
            ["a", "b", "c", "d", "e"],
            {"a", "c", "e"},
        )
        expected = (1.0 + 2.0 / 3.0 + 3.0 / 5.0) / 3.0
        assert result == pytest.approx(expected)

    def test_none_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.average_precision(["a", "b"], {"c"})
        assert result == 0.0

    def test_empty_retrieved(self, metrics: RetrievalMetrics) -> None:
        assert metrics.average_precision([], {"a"}) == 0.0

    def test_empty_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.average_precision(["a", "b"], set())
        assert result == 1.0

    def test_single_relevant_last(self, metrics: RetrievalMetrics) -> None:
        result = metrics.average_precision(
            ["a", "b", "c", "d", "e"], {"e"},
        )
        assert result == 0.2

    def test_deterministic(self, metrics: RetrievalMetrics) -> None:
        a = metrics.average_precision(["x", "y", "z"], {"x", "z"})
        b = metrics.average_precision(["x", "y", "z"], {"x", "z"})
        assert a == b


# ======================================================================
# RetrievalMetrics — normalized_dcg
# ======================================================================


class TestNormalizedDCG:
    """nDCG@k tests."""

    @pytest.fixture
    def metrics(self) -> RetrievalMetrics:
        return RetrievalMetrics()

    def test_perfect_ranking(self, metrics: RetrievalMetrics) -> None:
        import math
        result = metrics.normalized_dcg(["a", "b", "c"], {"a", "b", "c"}, k=3)
        assert result == pytest.approx(1.0)

    def test_partial_ranking(self, metrics: RetrievalMetrics) -> None:
        import math
        result = metrics.normalized_dcg(
            ["a", "b", "c", "d"], {"b", "d"}, k=4,
        )
        dcg = 1.0 / math.log2(3.0) + 1.0 / math.log2(5.0)
        idcg = 1.0 / math.log2(2.0) + 1.0 / math.log2(3.0)
        assert result == pytest.approx(dcg / idcg)

    def test_no_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.normalized_dcg(["a", "b", "c"], {"d"}, k=3)
        assert result == 0.0

    def test_k_smaller_than_results(self, metrics: RetrievalMetrics) -> None:
        import math
        result = metrics.normalized_dcg(
            ["a", "b", "c", "d"], {"a", "d"}, k=2,
        )
        assert result == pytest.approx(1.0)

    def test_k_is_zero(self, metrics: RetrievalMetrics) -> None:
        assert metrics.normalized_dcg(["a", "b"], {"a"}, k=0) == 0.0

    def test_k_is_negative(self, metrics: RetrievalMetrics) -> None:
        assert metrics.normalized_dcg(["a", "b"], {"a"}, k=-1) == 0.0

    def test_empty_retrieved(self, metrics: RetrievalMetrics) -> None:
        assert metrics.normalized_dcg([], {"a"}, k=5) == 0.0

    def test_empty_relevant(self, metrics: RetrievalMetrics) -> None:
        result = metrics.normalized_dcg(["a", "b"], set(), k=2)
        assert result == 1.0

    def test_unicode_ids(self, metrics: RetrievalMetrics) -> None:
        import math
        result = metrics.normalized_dcg(["東京", "巴黎"], {"東京"}, k=2)
        dcg = 1.0 / math.log2(2.0)
        idcg = 1.0 / math.log2(2.0)
        assert result == pytest.approx(dcg / idcg)

    def test_deterministic(self, metrics: RetrievalMetrics) -> None:
        a = metrics.normalized_dcg(["x", "y", "z"], {"x"}, k=3)
        b = metrics.normalized_dcg(["x", "y", "z"], {"x"}, k=3)
        assert a == b


# ======================================================================
# RetrievalMetrics — edge cases
# ======================================================================


class TestRetrievalMetricsEdgeCases:
    """Edge cases and cross-metric consistency."""

    @pytest.fixture
    def metrics(self) -> RetrievalMetrics:
        return RetrievalMetrics()

    def test_precision_recall_consistency(self, metrics: RetrievalMetrics) -> None:
        """All metric values must be in [0.0, 1.0]."""
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e"}
        for k in [1, 3, 5]:
            p = metrics.precision_at_k(retrieved, relevant, k)
            r = metrics.recall_at_k(retrieved, relevant, k)
            f = metrics.f1_at_k(retrieved, relevant, k)
            ap = metrics.average_precision(retrieved, relevant)
            ndcg = metrics.normalized_dcg(retrieved, relevant, k)
            for val, name in [(p, "P"), (r, "R"), (f, "F1"), (ap, "AP"), (ndcg, "nDCG")]:
                assert 0.0 <= val <= 1.0, f"{name}={val} outside [0, 1] at k={k}"

    def test_mrr_range(self, metrics: RetrievalMetrics) -> None:
        val = metrics.mean_reciprocal_rank(["a", "b"], {"c"})
        assert 0.0 <= val <= 1.0

    def test_duplicates_handled(self, metrics: RetrievalMetrics) -> None:
        """Duplicate IDs in retrieved list are counted as-is."""
        p = metrics.precision_at_k(["a", "a", "b"], {"a"}, k=3)
        assert p == pytest.approx(2.0 / 3.0)

    def test_f1_with_one_empty(self, metrics: RetrievalMetrics) -> None:
        """F1 where P=0, R=1 (empty relevant) should be 0."""
        f = metrics.f1_at_k(["a"], set(), k=1)
        assert f == 0.0
