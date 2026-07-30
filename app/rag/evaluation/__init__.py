"""Evaluation — quality and performance measurement for the Knowledge Layer.

The evaluation sub-package provides a pluggable framework for measuring
the quality and performance of RAG components (retrieval, reranking,
pipelines, knowledge bases).  It is independent of the concrete
components being evaluated — runners compose those components into
a coherent evaluation lifecycle.
"""

from __future__ import annotations

from app.rag.evaluation.base import EvaluationRunner
from app.rag.evaluation.config import EvaluationConfig
from app.rag.evaluation.errors import (
    EvaluationError,
    EvaluationNotFound,
    InvalidEvaluationConfiguration,
)
from app.rag.evaluation.models import BenchmarkResult, EvaluationResult
from app.rag.evaluation.registry import (
    clear_runners,
    get,
    list_runners,
    register,
    unregister,
)
from app.rag.evaluation.retrieval_metrics import RetrievalMetrics

__all__ = [
    "BenchmarkResult",
    "EvaluationConfig",
    "EvaluationError",
    "EvaluationNotFound",
    "EvaluationResult",
    "EvaluationRunner",
    "InvalidEvaluationConfiguration",
    "RetrievalMetrics",
    "clear_runners",
    "get",
    "list_runners",
    "register",
    "unregister",
]
