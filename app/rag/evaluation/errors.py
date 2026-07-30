"""Evaluation error hierarchy.

All evaluation-specific errors derive from :class:`EvaluationError`
which inherits from :class:`KnowledgeError`.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import KnowledgeError


class EvaluationError(KnowledgeError):
    """Base class for all evaluation errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "EVALUATION_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidEvaluationConfiguration(EvaluationError):
    """Raised when evaluation configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_EVALUATION_CONFIGURATION", details=details)


class EvaluationNotFound(EvaluationError):
    """Raised when a requested evaluation runner is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Evaluation runner {name!r} not found" if name else "Evaluation runner not found"
        super().__init__(msg, code="EVALUATION_NOT_FOUND", details=details)
