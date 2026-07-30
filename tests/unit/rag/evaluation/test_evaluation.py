"""Tests for the evaluation architecture."""

from __future__ import annotations

from typing import Any

import pytest

from app.rag.evaluation import (
    BenchmarkResult,
    BenchmarkRunner,
    DatasetLoader,
    EvaluationConfig,
    EvaluationDataset,
    EvaluationError,
    EvaluationNotFound,
    EvaluationResult,
    EvaluationRunner,
    EvaluationSample,
    InvalidEvaluationConfiguration,
    RetrievalMetrics,
    clear_runners,
    get,
    list_runners,
    register,
    unregister,
)
from app.rag.evaluation.base import EvaluationRunner as EvaluationRunner_Impl
from app.rag.evaluation.benchmark import BenchmarkRunner as BenchmarkRunner_Impl
from app.rag.evaluation.config import EvaluationConfig as EvaluationConfig_Impl
from app.rag.evaluation.datasets import DatasetLoader as DatasetLoader_Impl
from app.rag.evaluation.datasets import EvaluationDataset as EvaluationDataset_Impl
from app.rag.evaluation.datasets import EvaluationSample as EvaluationSample_Impl
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

    def test_benchmark_runner_imported(self) -> None:
        assert BenchmarkRunner is BenchmarkRunner_Impl

    def test_evaluation_sample_imported(self) -> None:
        assert EvaluationSample is EvaluationSample_Impl

    def test_evaluation_dataset_imported(self) -> None:
        assert EvaluationDataset is EvaluationDataset_Impl

    def test_dataset_loader_imported(self) -> None:
        assert DatasetLoader is DatasetLoader_Impl


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
        assert r.average_latency_ms == 0.0
        assert r.min_latency_ms == 0.0
        assert r.max_latency_ms == 0.0
        assert r.throughput_qps == 0.0
        assert r.total_queries == 0
        assert r.total_duration == 0.0

    def test_custom_values(self) -> None:
        r = BenchmarkResult(
            latency_ms=45.2,
            throughput=220.0,
            average_latency_ms=45.2,
            min_latency_ms=12.0,
            max_latency_ms=98.0,
            throughput_qps=220.0,
            total_queries=50,
            total_duration=250.0,
            memory_bytes=1048576,
            metadata={"batch_size": 32},
        )
        assert r.latency_ms == 45.2
        assert r.throughput == 220.0
        assert r.average_latency_ms == 45.2
        assert r.min_latency_ms == 12.0
        assert r.max_latency_ms == 98.0
        assert r.throughput_qps == 220.0
        assert r.total_queries == 50
        assert r.total_duration == 250.0
        assert r.memory_bytes == 1048576
        assert r.metadata == {"batch_size": 32}

    def test_immutable(self) -> None:
        r = BenchmarkResult()
        with pytest.raises(AttributeError):
            r.latency_ms = 10.0  # type: ignore[misc]
        with pytest.raises(AttributeError):
            r.average_latency_ms = 5.0  # type: ignore[misc]

    def test_backward_compat_latency_alias(self) -> None:
        r = BenchmarkResult(latency_ms=33.0, average_latency_ms=33.0)
        assert r.latency_ms == r.average_latency_ms

    def test_backward_compat_throughput_alias(self) -> None:
        r = BenchmarkResult(throughput=150.0, throughput_qps=150.0)
        assert r.throughput == r.throughput_qps


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


# ======================================================================
# BenchmarkRunner
# ======================================================================


class _AsyncId:
    """Async callable that returns its input (instant)."""

    async def __call__(self, query: str) -> str:
        return query


class _AsyncDelayed:
    """Async callable that simulates a fixed delay."""

    def __init__(self, delay_s: float = 0.01) -> None:
        self._delay = delay_s

    async def __call__(self, query: str) -> str:
        import asyncio
        await asyncio.sleep(self._delay)
        return query


class _AsyncFailing:
    """Async callable that raises for a specific query."""

    def __init__(self, fail_on: str = "fail") -> None:
        self._fail_on = fail_on

    async def __call__(self, query: str) -> str:
        if query == self._fail_on:
            raise ValueError(f"Failed on: {query}")
        return query


class TestBenchmarkRunner:
    """BenchmarkRunner construction and configuration."""

    def test_default_config(self) -> None:
        runner = BenchmarkRunner()
        assert runner.config.warmup_runs == 3
        assert runner.config.benchmark_runs == 10

    def test_custom_config(self) -> None:
        config = EvaluationConfig(warmup_runs=5, benchmark_runs=20)
        runner = BenchmarkRunner(config=config)
        assert runner.config.warmup_runs == 5
        assert runner.config.benchmark_runs == 20


class TestBenchmarkRunnerRun:
    """BenchmarkRunner.run() behaviour."""

    @pytest.mark.asyncio
    async def test_basic_run(self) -> None:
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncId(),
            ["q1", "q2", "q3"],
            warmup_runs=1,
            benchmark_runs=2,
        )
        assert result.total_queries == 6  # 3 queries × 2 benchmark_runs
        assert result.average_latency_ms >= 0
        assert result.min_latency_ms >= 0
        assert result.max_latency_ms >= 0
        assert result.min_latency_ms <= result.average_latency_ms <= result.max_latency_ms

    @pytest.mark.asyncio
    async def test_latency_measurement(self) -> None:
        """Delayed component should produce measurable latency."""
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncDelayed(0.005),
            ["q1"],
            warmup_runs=1,
            benchmark_runs=3,
        )
        # Each call should take at least 5ms
        assert result.average_latency_ms >= 5.0
        assert result.min_latency_ms >= 5.0
        assert result.total_queries == 3

    @pytest.mark.asyncio
    async def test_throughput_qps(self) -> None:
        """Throughput should be positive for non-zero duration."""
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncId(),
            ["q1", "q2"],
            warmup_runs=1,
            benchmark_runs=5,
        )
        assert result.throughput_qps > 0
        assert result.throughput > 0

    @pytest.mark.asyncio
    async def test_throughput_decreases_with_delay(self) -> None:
        """Slower component lower throughput."""
        runner = BenchmarkRunner()
        fast = await runner.run(
            _AsyncId(),
            ["q1"],
            warmup_runs=1,
            benchmark_runs=5,
        )
        slow = await runner.run(
            _AsyncDelayed(0.01),
            ["q1"],
            warmup_runs=1,
            benchmark_runs=5,
        )
        assert fast.throughput_qps > slow.throughput_qps

    @pytest.mark.asyncio
    async def test_empty_dataset(self) -> None:
        """Empty dataset should return a nominal result."""
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncId(),
            [],
            warmup_runs=1,
            benchmark_runs=5,
        )
        assert result.total_queries == 0
        assert result.average_latency_ms == 0.0

    @pytest.mark.asyncio
    async def test_single_query(self) -> None:
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncId(),
            ["only_query"],
            warmup_runs=0,
            benchmark_runs=1,
        )
        assert result.total_queries == 1
        assert result.average_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_benchmark_runs_one(self) -> None:
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncId(),
            ["q1", "q2"],
            warmup_runs=0,
            benchmark_runs=1,
        )
        assert result.total_queries == 2

    @pytest.mark.asyncio
    async def test_warmup_skipped_from_timing(self) -> None:
        """Warmup iterations should not affect timing."""
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncDelayed(0.001),
            ["q1"],
            warmup_runs=10,
            benchmark_runs=3,
        )
        assert result.total_queries == 3  # Only benchmark runs counted

    @pytest.mark.asyncio
    async def test_custom_run_counts(self) -> None:
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncId(),
            ["q1"],
            warmup_runs=0,
            benchmark_runs=7,
        )
        assert result.total_queries == 7

    @pytest.mark.asyncio
    async def test_uses_default_config(self) -> None:
        config = EvaluationConfig(warmup_runs=2, benchmark_runs=5)
        runner = BenchmarkRunner(config=config)
        result = await runner.run(
            _AsyncId(),
            ["q1"],
        )
        assert result.total_queries == 5  # from config

    @pytest.mark.asyncio
    async def test_exception_during_warmup(self) -> None:
        runner = BenchmarkRunner()
        with pytest.raises(EvaluationError) as exc:
            await runner.run(
                _AsyncFailing(fail_on="bad"),
                ["bad"],
                warmup_runs=1,
                benchmark_runs=0,
            )
        assert "warm-up" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_exception_during_benchmark(self) -> None:
        runner = BenchmarkRunner()
        with pytest.raises(EvaluationError) as exc:
            await runner.run(
                _AsyncFailing(fail_on="bad"),
                ["good", "bad"],
                warmup_runs=0,
                benchmark_runs=1,
            )
        assert "benchmark" in str(exc.value).lower()

    @pytest.mark.asyncio
    async def test_deterministic_latency(self) -> None:
        """Same component and dataset should produce same stats."""
        runner = BenchmarkRunner()
        r1 = await runner.run(
            _AsyncDelayed(0.002),
            ["q1", "q2"],
            warmup_runs=1,
            benchmark_runs=5,
        )
        r2 = await runner.run(
            _AsyncDelayed(0.002),
            ["q1", "q2"],
            warmup_runs=1,
            benchmark_runs=5,
        )
        assert abs(r1.average_latency_ms - r2.average_latency_ms) < 10.0

    @pytest.mark.asyncio
    async def test_metadata(self) -> None:
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncId(),
            ["q1", "q2"],
            warmup_runs=2,
            benchmark_runs=3,
        )
        meta = result.metadata
        assert meta["warmup_runs"] == 2
        assert meta["benchmark_runs"] == 3
        assert meta["dataset_size"] == 2
        assert meta["latency_samples"] == 6  # 2 queries * 3 runs

    @pytest.mark.asyncio
    async def test_min_max_latency(self) -> None:
        """min <= avg <= max should always hold."""
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncDelayed(0.003),
            ["q1", "q2", "q3"],
            warmup_runs=1,
            benchmark_runs=10,
        )
        assert result.min_latency_ms <= result.average_latency_ms <= result.max_latency_ms

    @pytest.mark.asyncio
    async def test_total_duration(self) -> None:
        """Total duration should be positive when benchmark runs complete."""
        runner = BenchmarkRunner()
        result = await runner.run(
            _AsyncDelayed(0.002),
            ["q1"],
            warmup_runs=0,
            benchmark_runs=5,
        )
        assert result.total_duration > 0


# ======================================================================
# EvaluationSample
# ======================================================================


class TestEvaluationSample:
    """EvaluationSample frozen dataclass."""

    def test_default_values(self) -> None:
        s = EvaluationSample()
        assert s.query == ""
        assert s.relevant_ids == frozenset()
        assert s.metadata == {}

    def test_custom_values(self) -> None:
        s = EvaluationSample(
            query="capital of France",
            relevant_ids=frozenset({"doc1", "doc2"}),
            metadata={"difficulty": "easy"},
        )
        assert s.query == "capital of France"
        assert s.relevant_ids == frozenset({"doc1", "doc2"})
        assert s.metadata == {"difficulty": "easy"}

    def test_immutable(self) -> None:
        s = EvaluationSample(query="test")
        with pytest.raises(AttributeError):
            s.query = "changed"  # type: ignore[misc]

    def test_relevant_ids_is_frozenset(self) -> None:
        s = EvaluationSample(relevant_ids=frozenset({"a", "b"}))
        assert isinstance(s.relevant_ids, frozenset)

    def test_unicode(self) -> None:
        s = EvaluationSample(
            query="東京の首都",
            relevant_ids=frozenset({"doc_東京"}),
        )
        assert "東京" in s.query
        assert "doc_東京" in s.relevant_ids


# ======================================================================
# EvaluationDataset
# ======================================================================


class TestEvaluationDataset:
    """EvaluationDataset frozen dataclass."""

    def test_default_values(self) -> None:
        d = EvaluationDataset()
        assert d.name == ""
        assert d.samples == ()
        assert d.metadata == {}
        assert d.size == 0
        assert d.is_empty is True

    def test_custom_values(self) -> None:
        samples = (
            EvaluationSample(query="q1", relevant_ids=frozenset({"a"})),
            EvaluationSample(query="q2", relevant_ids=frozenset({"b"})),
        )
        d = EvaluationDataset(
            name="test-ds",
            samples=samples,
            metadata={"version": "1.0"},
        )
        assert d.name == "test-ds"
        assert d.size == 2
        assert d.is_empty is False
        assert d.metadata == {"version": "1.0"}

    def test_immutable(self) -> None:
        d = EvaluationDataset(name="ds")
        with pytest.raises(AttributeError):
            d.name = "changed"  # type: ignore[misc]

    def test_size_property(self) -> None:
        d = EvaluationDataset(samples=(
            EvaluationSample(query="a"),
            EvaluationSample(query="b"),
        ))
        assert d.size == 2

    def test_is_empty_true(self) -> None:
        d = EvaluationDataset()
        assert d.is_empty is True

    def test_is_empty_false(self) -> None:
        d = EvaluationDataset(samples=(EvaluationSample(query="a"),))
        assert d.is_empty is False

    def test_queries(self) -> None:
        d = EvaluationDataset(samples=(
            EvaluationSample(query="q1", relevant_ids=frozenset({"a"})),
            EvaluationSample(query="q2", relevant_ids=frozenset({"b"})),
        ))
        assert d.queries() == ["q1", "q2"]

    def test_relevant_sets(self) -> None:
        d = EvaluationDataset(samples=(
            EvaluationSample(query="q1", relevant_ids=frozenset({"a", "b"})),
        ))
        result = d.relevant_sets()
        assert len(result) == 1
        assert result[0] == frozenset({"a", "b"})

    def test_sample(self) -> None:
        s1 = EvaluationSample(query="q1")
        s2 = EvaluationSample(query="q2")
        d = EvaluationDataset(samples=(s1, s2))
        assert d.sample(0) is s1
        assert d.sample(1) is s2

    def test_sample_out_of_range(self) -> None:
        d = EvaluationDataset()
        with pytest.raises(IndexError):
            d.sample(0)

    def test_unicode(self) -> None:
        s = EvaluationSample(query="東京", relevant_ids=frozenset({"doc_1"}))
        d = EvaluationDataset(name="日本語", samples=(s,))
        assert "日本語" in d.name
        assert d.sample(0).query == "東京"

    def test_empty_samples(self) -> None:
        d = EvaluationDataset(samples=())
        assert d.size == 0
        assert d.queries() == []
        assert d.relevant_sets() == []


# ======================================================================
# DatasetLoader — from_dict
# ======================================================================


class TestDatasetLoaderFromDict:
    """DatasetLoader.from_dict() tests."""

    def test_basic(self) -> None:
        data = {
            "name": "test",
            "samples": [
                {"query": "q1", "relevant_ids": ["a", "b"]},
                {"query": "q2", "relevant_ids": ["c"]},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        assert ds.name == "test"
        assert ds.size == 2
        assert ds.sample(0).query == "q1"
        assert ds.sample(0).relevant_ids == frozenset({"a", "b"})
        assert ds.sample(1).query == "q2"

    def test_with_metadata(self) -> None:
        data = {
            "name": "test",
            "metadata": {"version": "2", "source": "wiki"},
            "samples": [
                {"query": "q", "relevant_ids": ["a"], "metadata": {"difficulty": "hard"}},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        assert ds.metadata == {"version": "2", "source": "wiki"}
        assert ds.sample(0).metadata == {"difficulty": "hard"}

    def test_empty_samples(self) -> None:
        ds = DatasetLoader.from_dict({"name": "empty", "samples": []})
        assert ds.size == 0
        assert ds.is_empty

    def test_default_name(self) -> None:
        ds = DatasetLoader.from_dict({"samples": []})
        assert ds.name == ""

    def test_unicode(self) -> None:
        data = {
            "name": "unicode-test",
            "samples": [
                {"query": "東京の首都", "relevant_ids": ["doc_東京"]},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        assert "東京" in ds.sample(0).query
        assert "doc_東京" in ds.sample(0).relevant_ids

    def test_not_a_dict(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict([])  # type: ignore[arg-type]
        assert "object" in str(exc.value).lower()

    def test_name_not_string(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({"name": 123, "samples": []})
        assert "name" in str(exc.value).lower()

    def test_missing_samples(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({"name": "x"})
        assert "samples" in str(exc.value).lower()

    def test_samples_not_list(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({"name": "x", "samples": "not a list"})
        assert "samples" in str(exc.value).lower()

    def test_sample_not_dict(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({"name": "x", "samples": ["string"]})
        assert "object" in str(exc.value).lower()

    def test_query_not_string(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({"name": "x", "samples": [{"query": 42}]})
        assert "query" in str(exc.value).lower()

    def test_relevant_ids_not_list(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({"name": "x", "samples": [{"query": "q", "relevant_ids": "a"}]})
        assert "relevant_ids" in str(exc.value).lower()

    def test_relevant_id_not_string(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({"name": "x", "samples": [{"query": "q", "relevant_ids": [1]}]})
        assert "relevant_ids" in str(exc.value).lower()

    def test_metadata_not_dict(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({"name": "x", "samples": [], "metadata": "bad"})
        assert "metadata" in str(exc.value).lower()

    def test_sample_metadata_not_dict(self) -> None:
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_dict({
                "name": "x", "samples": [{"query": "q", "metadata": "bad"}],
            })
        assert "metadata" in str(exc.value).lower()

    def test_duplicate_ids_deduped(self) -> None:
        ds = DatasetLoader.from_dict({
            "name": "test",
            "samples": [{"query": "q", "relevant_ids": ["a", "a", "b"]}],
        })
        assert ds.sample(0).relevant_ids == frozenset({"a", "b"})


# ======================================================================
# DatasetLoader — to_dict / to_json
# ======================================================================


class TestDatasetLoaderToDict:
    """DatasetLoader.to_dict() tests."""

    def test_round_trip(self) -> None:
        data = {
            "name": "roundtrip",
            "metadata": {"v": 1},
            "samples": [
                {"query": "q1", "relevant_ids": ["b", "a"], "metadata": {}},
                {"query": "q2", "relevant_ids": ["c"], "metadata": {}},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        result = DatasetLoader.to_dict(ds)
        assert result["name"] == "roundtrip"
        assert result["metadata"] == {"v": 1}
        assert result["samples"][0]["relevant_ids"] == ["a", "b"]

    def test_deterministic(self) -> None:
        ds = DatasetLoader.from_dict({
            "name": "det",
            "samples": [{"query": "z", "relevant_ids": ["c", "a", "b"]}],
        })
        a = DatasetLoader.to_dict(ds)
        b = DatasetLoader.to_dict(ds)
        assert a == b

    def test_unicode_round_trip(self) -> None:
        ds = DatasetLoader.from_dict({
            "name": "日本語",
            "samples": [{"query": "東京", "relevant_ids": ["doc_東京"]}],
        })
        result = DatasetLoader.to_dict(ds)
        assert result["name"] == "日本語"
        assert result["samples"][0]["query"] == "東京"


class TestDatasetLoaderToJson:
    """DatasetLoader.to_json() and from_json() tests."""

    def test_round_trip(self, tmp_path: str) -> None:
        import json
        path = str(tmp_path / "dataset.json")
        data = {
            "name": "json-test",
            "metadata": {"version": "1"},
            "samples": [
                {"query": "capital of France", "relevant_ids": ["doc1", "doc2"]},
                {"query": "capital of Japan", "relevant_ids": ["doc3"]},
            ],
        }
        ds = DatasetLoader.from_dict(data)
        DatasetLoader.to_json(ds, path)

        loaded = DatasetLoader.from_json(path)
        assert loaded.name == "json-test"
        assert loaded.size == 2
        assert loaded.sample(0).query == "capital of France"

    def test_deterministic_output(self, tmp_path: str) -> None:
        import json
        path = str(tmp_path / "det.json")
        ds = DatasetLoader.from_dict({
            "name": "det",
            "samples": [{"query": "q", "relevant_ids": ["c", "a"]}],
        })
        DatasetLoader.to_json(ds, path)
        DatasetLoader.to_json(ds, path)
        with open(path, encoding="utf-8") as f:
            first = json.load(f)

        with open(path, encoding="utf-8") as f:
            second = json.load(f)
        assert first == second

    def test_unicode_file(self, tmp_path: str) -> None:
        path = str(tmp_path / "unicode.json")
        ds = DatasetLoader.from_dict({
            "name": "日本語",
            "samples": [{"query": "東京", "relevant_ids": ["doc_東京"]}],
        })
        DatasetLoader.to_json(ds, path)
        loaded = DatasetLoader.from_json(path)
        assert "東京" in loaded.sample(0).query

    def test_file_not_found(self) -> None:
        with pytest.raises(EvaluationError) as exc:
            DatasetLoader.from_json("/nonexistent/path.json")
        assert "does not exist" in str(exc.value)

    def test_invalid_json(self, tmp_path: str) -> None:
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json")
        with pytest.raises(EvaluationError) as exc:
            DatasetLoader.from_json(path)
        assert "parse" in str(exc.value).lower()

    def test_root_not_object(self, tmp_path: str) -> None:
        import json
        path = str(tmp_path / "list.json")
        with open(path, "w") as f:
            json.dump([], f)
        with pytest.raises(InvalidEvaluationConfiguration) as exc:
            DatasetLoader.from_json(path)
        assert "object" in str(exc.value).lower()


# ======================================================================
# DatasetLoader — empty dataset serialization
# ======================================================================


class TestDatasetLoaderEmpty:
    """Empty dataset serialization round-trip."""

    def test_empty_round_trip_dict(self) -> None:
        data = {"name": "empty", "samples": []}
        ds = DatasetLoader.from_dict(data)
        result = DatasetLoader.to_dict(ds)
        assert result["name"] == "empty"
        assert result["samples"] == []

    def test_empty_to_json(self, tmp_path: str) -> None:
        path = str(tmp_path / "empty.json")
        ds = DatasetLoader.from_dict({"name": "empty", "samples": []})
        DatasetLoader.to_json(ds, path)
        loaded = DatasetLoader.from_json(path)
        assert loaded.size == 0
        assert loaded.is_empty
