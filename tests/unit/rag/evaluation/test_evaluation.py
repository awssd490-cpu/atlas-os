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
