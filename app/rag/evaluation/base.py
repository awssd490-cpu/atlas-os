"""Base abstractions for the evaluation layer.

Defines the ``EvaluationRunner`` abstract base class that all evaluation
runner implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.rag.evaluation.config import EvaluationConfig
from app.rag.evaluation.models import BenchmarkResult, EvaluationResult


class EvaluationRunner(ABC):
    """Abstract base class for evaluation runners.

    An evaluation runner measures the quality and performance of a
    component (e.g. a retrieval pipeline, a reranker, a knowledge base)
    through three operations:

    * ``evaluate()`` — compute quality metrics (precision, recall, F1,
      hit rate, etc.) against a ground-truth dataset.
    * ``benchmark()`` — measure latency, throughput, and memory usage.
    * ``profile()`` — deep-dive performance profiling with per-operation
      breakdowns.

    Concrete subclasses must implement all three methods.
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self._config = config or EvaluationConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> EvaluationConfig:
        """Return the runner's configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Evaluation API
    # ------------------------------------------------------------------

    @abstractmethod
    async def evaluate(
        self,
        component: object,
        dataset: object,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a component against a ground-truth dataset.

        Args:
            component: The component to evaluate (e.g. a
                ``DefaultKnowledgePipeline``, a ``Reranker``).
            dataset: A ground-truth dataset with queries and expected
                results.
            **kwargs: Implementation-specific options.

        Returns:
            An ``EvaluationResult`` with aggregate score and per-metric
            breakdown.

        Raises:
            EvaluationError: On evaluation failures.
        """
        ...

    @abstractmethod
    async def benchmark(
        self,
        component: object,
        dataset: object,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Benchmark a component for latency, throughput, and memory.

        Args:
            component: The component to benchmark.
            dataset: The dataset to use for benchmarking.
            **kwargs: Implementation-specific options (e.g. batch
                size, concurrency level).

        Returns:
            A ``BenchmarkResult`` with measured performance data.

        Raises:
            EvaluationError: On benchmarking failures.
        """
        ...

    @abstractmethod
    async def profile(
        self,
        component: object,
        dataset: object,
        **kwargs: Any,
    ) -> BenchmarkResult:
        """Profile a component with per-operation breakdowns.

        Args:
            component: The component to profile.
            dataset: The dataset to use for profiling.
            **kwargs: Implementation-specific options.

        Returns:
            A ``BenchmarkResult`` with detailed profiling data.

        Raises:
            EvaluationError: On profiling failures.
        """
        ...
