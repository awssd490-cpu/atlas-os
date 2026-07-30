"""Reliability domain models.

All models in this module are immutable frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behaviour.

    Attributes:
        max_attempts: Maximum number of execution attempts (including
            the first).  Must be >= 1.  Default 3.
        initial_delay_ms: Delay before the first retry in milliseconds.
            Must be >= 0.  Default 100.0.
        backoff_multiplier: Multiplicative factor applied to the delay
            after each retry.  Must be >= 1.0.  Default 2.0.
        max_delay_ms: Maximum delay between any two retries in
            milliseconds.  Must be >= 0.  Default 5000.0.
        retry_exceptions: Tuple of exception types that should trigger
            a retry.  If empty, ALL exceptions trigger a retry.
            Default ``()``.
        metadata: Optional structured metadata.
    """

    max_attempts: int = 3
    initial_delay_ms: float = 100.0
    backoff_multiplier: float = 2.0
    max_delay_ms: float = 5000.0
    retry_exceptions: tuple[type[Exception], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate policy values.

        Raises:
            InvalidRetryPolicy: If any value is out of range.
        """
        from app.core.reliability.errors import InvalidRetryPolicy

        if self.max_attempts < 1:
            raise InvalidRetryPolicy(
                f"max_attempts must be at least 1, got {self.max_attempts}",
                details={"max_attempts": self.max_attempts},
            )
        if self.initial_delay_ms < 0:
            raise InvalidRetryPolicy(
                f"initial_delay_ms must be non-negative, got {self.initial_delay_ms}",
                details={"initial_delay_ms": self.initial_delay_ms},
            )
        if self.backoff_multiplier < 1.0:
            raise InvalidRetryPolicy(
                f"backoff_multiplier must be at least 1.0, got {self.backoff_multiplier}",
                details={"backoff_multiplier": self.backoff_multiplier},
            )
        if self.max_delay_ms < 0:
            raise InvalidRetryPolicy(
                f"max_delay_ms must be non-negative, got {self.max_delay_ms}",
                details={"max_delay_ms": self.max_delay_ms},
            )


@dataclass(frozen=True)
class RetryResult:
    """Immutable result of a retry attempt.

    Attributes:
        attempts: Number of attempts made.
        success: Whether any attempt succeeded.
        duration_ms: Total elapsed time for all attempts in ms.
        metadata: Additional metadata (last error, attempt times, etc.).
    """

    attempts: int = 0
    success: bool = False
    duration_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
