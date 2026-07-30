"""Evaluation domain models.

Every model in this module is immutable.  They represent the data types
for the evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """The result of an evaluation run.

    Attributes:
        score: A single aggregate score for the evaluated component.
        metrics: A mapping of metric names to their computed values.
        metadata: Optional metadata about the evaluation (configuration
            used, component name, data split, etc.).
        duration: Total evaluation duration in seconds.
    """

    score: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0


@dataclass(frozen=True)
class BenchmarkResult:
    """The result of a benchmark run.

    Attributes:
        latency_ms: Average latency in milliseconds (alias for
            ``average_latency_ms``).
        throughput: Measured throughput in operations per second
            (alias for ``throughput_qps``).
        average_latency_ms: Mean latency across all benchmark runs.
        min_latency_ms: Minimum observed latency.
        max_latency_ms: Maximum observed latency.
        throughput_qps: Queries per second (total queries / total
            duration).
        total_queries: Number of queries benchmarked.
        total_duration: Total benchmark duration in milliseconds.
        memory_bytes: Memory usage in bytes during the benchmark.
        metadata: Optional metadata about the benchmark (system info,
            configuration, environment, etc.).
    """

    latency_ms: float = 0.0
    throughput: float = 0.0
    average_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    throughput_qps: float = 0.0
    total_queries: int = 0
    total_duration: float = 0.0
    memory_bytes: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
