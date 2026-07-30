"""BenchmarkRunner — generic benchmark execution engine.

Measures the latency and throughput of an arbitrary callable component
over a dataset of queries, with configurable warm-up and measurement
rounds.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from app.rag.evaluation.config import EvaluationConfig
from app.rag.evaluation.errors import EvaluationError
from app.rag.evaluation.models import BenchmarkResult


class BenchmarkRunner:
    """Generic benchmark runner.

    Executes a component callable over a dataset of queries, first
    through warm-up iterations (excluded from timing), then through
    measured benchmark iterations.

    Timing is performed with ``time.perf_counter()`` for maximum
    precision.

    Usage::

        async def search(query: str) -> list[str]:
            return await pipeline.search(query)

        runner = BenchmarkRunner()
        result = await runner.run(
            component=search,
            dataset=["capital of France", "capital of Japan"],
            warmup_runs=5,
            benchmark_runs=20,
        )
        print(f"Average latency: {result.average_latency_ms:.1f} ms")
        print(f"Throughput: {result.throughput_qps:.0f} qps")
    """

    def __init__(
        self,
        config: EvaluationConfig | None = None,
    ) -> None:
        self._config = config or EvaluationConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> EvaluationConfig:
        """Return the runner's configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run(
        self,
        component: Callable[[Any], Awaitable[Any]],
        dataset: Iterable[Any],
        *,
        warmup_runs: int | None = None,
        benchmark_runs: int | None = None,
    ) -> BenchmarkResult:
        """Benchmark a component by running it against a dataset.

        The *component* is an async callable that accepts a single
        dataset item (e.g. a query string) and returns a result.

        Warm-up iterations are executed first but their timing is
        discarded.  Benchmark iterations are measured with
        ``time.perf_counter()``.

        Args:
            component: An async callable that accepts a dataset item
                and returns a result.
            dataset: An iterable of query items to benchmark.
            warmup_runs: Override the configured warmup runs.
            benchmark_runs: Override the configured benchmark runs.

        Returns:
            A ``BenchmarkResult`` with latency and throughput
            statistics.

        Raises:
            EvaluationError: If the component raises during measurement.
        """
        resolved_warmup = (
            warmup_runs if warmup_runs is not None else self._config.warmup_runs
        )
        resolved_benchmark = (
            benchmark_runs if benchmark_runs is not None else self._config.benchmark_runs
        )

        queries = list(dataset)
        if not queries:
            return BenchmarkResult(
                metadata={
                    "warmup_runs": resolved_warmup,
                    "benchmark_runs": resolved_benchmark,
                    "dataset_size": 0,
                },
            )

        # Warm-up phase — excluded from all timing
        for _ in range(resolved_warmup):
            for query in queries:
                try:
                    await component(query)
                except Exception as exc:
                    raise EvaluationError(
                        f"Component raised during warm-up: {exc}",
                        details={"query": str(query)},
                    ) from exc

        # Benchmark phase — measured
        latencies: list[float] = []
        total_start = time.perf_counter()

        for _ in range(resolved_benchmark):
            for query in queries:
                try:
                    op_start = time.perf_counter()
                    await component(query)
                    op_end = time.perf_counter()
                except Exception as exc:
                    raise EvaluationError(
                        f"Component raised during benchmark: {exc}",
                        details={"query": str(query)},
                    ) from exc

                elapsed_s = op_end - op_start
                latencies.append(elapsed_s * 1000.0)  # convert to ms

        total_end = time.perf_counter()
        total_duration_s = total_end - total_start
        total_duration_ms = total_duration_s * 1000.0

        total_queries = len(queries) * resolved_benchmark

        if not latencies:
            return BenchmarkResult(
                metadata={
                    "warmup_runs": resolved_warmup,
                    "benchmark_runs": resolved_benchmark,
                    "dataset_size": len(queries),
                },
            )

        avg_latency_ms = statistics.mean(latencies)
        min_latency_ms = min(latencies)
        max_latency_ms = max(latencies)
        throughput_qps = total_queries / total_duration_s if total_duration_s > 0 else 0.0

        return BenchmarkResult(
            latency_ms=avg_latency_ms,
            throughput=throughput_qps,
            average_latency_ms=avg_latency_ms,
            min_latency_ms=min_latency_ms,
            max_latency_ms=max_latency_ms,
            throughput_qps=throughput_qps,
            total_queries=total_queries,
            total_duration=total_duration_ms,
            metadata={
                "warmup_runs": resolved_warmup,
                "benchmark_runs": resolved_benchmark,
                "dataset_size": len(queries),
                "latency_samples": len(latencies),
            },
        )
