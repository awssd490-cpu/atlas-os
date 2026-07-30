"""Evaluation configuration.

All configuration objects are immutable frozen dataclasses, following the
convention established in ``app.rag.models``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for an evaluation runner.

    Attributes:
        enabled: Whether the runner is active.  Default ``True``.
        metrics: Tuple of metric names to compute (e.g.
            ``("precision", "recall", "f1")``).  Default empty.
        warmup_runs: Number of warm-up iterations before measurement.
            Must be >= 0.  Default 3.
        benchmark_runs: Number of benchmark iterations for measurement.
            Must be > 0.  Default 10.
        random_seed: Seed for reproducible random operations.
            Default 42.
    """

    enabled: bool = True
    metrics: tuple[str, ...] = ()
    warmup_runs: int = 3
    benchmark_runs: int = 10
    random_seed: int = 42

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidEvaluationConfiguration: If any value is out of range
                or invalid.
        """
        from app.rag.evaluation.errors import InvalidEvaluationConfiguration

        if self.benchmark_runs < 1:
            raise InvalidEvaluationConfiguration(
                "benchmark_runs must be at least 1",
                details={"benchmark_runs": self.benchmark_runs},
            )
        if self.warmup_runs < 0:
            raise InvalidEvaluationConfiguration(
                "warmup_runs must be non-negative",
                details={"warmup_runs": self.warmup_runs},
            )
