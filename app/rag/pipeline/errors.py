"""Pipeline error hierarchy.

All pipeline-specific errors derive from :class:`PipelineError`
which inherits from :class:`KnowledgeError`.
"""

from __future__ import annotations

from typing import Any

from app.rag.errors import KnowledgeError


class PipelineError(KnowledgeError):
    """Base class for all pipeline errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "PIPELINE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class InvalidPipelineConfiguration(PipelineError):
    """Raised when pipeline configuration is invalid."""

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code="INVALID_PIPELINE_CONFIGURATION", details=details)


class PipelineNotFound(PipelineError):
    """Raised when a requested pipeline is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Pipeline {name!r} not found" if name else "Pipeline not found"
        super().__init__(msg, code="PIPELINE_NOT_FOUND", details=details)
