"""Pipeline — knowledge ingestion and search orchestration.

The pipeline sub-package provides a pluggable framework for
orchestrating document ingestion, search, and lifecycle management.
It is independent of the concrete knowledge base, embedding, vector
store, and reranking layers — pipelines compose those components
into a coherent end-to-end flow.
"""

from __future__ import annotations

from app.rag.pipeline.base import KnowledgePipeline
from app.rag.pipeline.config import PipelineConfig
from app.rag.pipeline.default import DefaultKnowledgePipeline
from app.rag.pipeline.errors import (
    InvalidPipelineConfiguration,
    PipelineError,
    PipelineNotFound,
)
from app.rag.pipeline.models import PipelineResult, PipelineStats
from app.rag.pipeline.registry import (
    clear_pipelines,
    get,
    list_pipelines,
    register,
    unregister,
)

__all__ = [
    "DefaultKnowledgePipeline",
    "InvalidPipelineConfiguration",
    "KnowledgePipeline",
    "PipelineConfig",
    "PipelineError",
    "PipelineNotFound",
    "PipelineResult",
    "PipelineStats",
    "clear_pipelines",
    "get",
    "list_pipelines",
    "register",
    "unregister",
]
