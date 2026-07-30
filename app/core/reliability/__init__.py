"""Reliability — lightweight retry and reliability utilities for Atlas.

Provides ``RetryExecutor`` for executing callables with exponential
backoff retry, ``RetryPolicy`` for configuring retry behaviour, and
``RetryResult`` for inspecting retry outcomes.
"""

from __future__ import annotations

from app.core.reliability.errors import InvalidRetryPolicy, ReliabilityError
from app.core.reliability.models import RetryPolicy, RetryResult
from app.core.reliability.retry import RetryExecutor

__all__ = [
    "InvalidRetryPolicy",
    "ReliabilityError",
    "RetryExecutor",
    "RetryPolicy",
    "RetryResult",
]
