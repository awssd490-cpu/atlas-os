"""Base abstractions for the knowledge pipeline layer.

Defines the ``KnowledgePipeline`` abstract base class that all pipeline
implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.rag.pipeline.config import PipelineConfig
from app.rag.pipeline.models import PipelineResult, PipelineStats


class KnowledgePipeline(ABC):
    """Abstract base class for knowledge pipelines.

    A pipeline orchestrates the end-to-end flow of knowledge: ingesting
    documents, searching for relevant content, tracking statistics, and
    clearing state.  Concrete subclasses integrate with the underlying
    knowledge base, embedding providers, vector stores, and rerankers.

    Concrete subclasses must implement ``ingest()``, ``search()``,
    ``clear()``, and ``stats()``.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> PipelineConfig:
        """Return the pipeline's configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Pipeline API
    # ------------------------------------------------------------------

    @abstractmethod
    async def ingest(
        self,
        documents: list[Any],
        **kwargs: Any,
    ) -> int:
        """Ingest documents into the knowledge base.

        Args:
            documents: A list of documents to ingest.
            **kwargs: Implementation-specific options.

        Returns:
            The number of documents successfully ingested.

        Raises:
            PipelineError: On pipeline-level failures.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> PipelineResult:
        """Search the knowledge base for relevant content.

        Args:
            query: The search query string.
            **kwargs: Implementation-specific options (e.g. top_k,
                min_score, filters).

        Returns:
            A ``PipelineResult`` containing the matched context.

        Raises:
            PipelineError: On pipeline-level failures.
        """
        ...

    @abstractmethod
    async def clear(self, **kwargs: Any) -> None:
        """Clear all data from the pipeline.

        Args:
            **kwargs: Implementation-specific options.

        Raises:
            PipelineError: On pipeline-level failures.
        """
        ...

    @abstractmethod
    async def stats(self, **kwargs: Any) -> PipelineStats:
        """Return current pipeline statistics.

        Args:
            **kwargs: Implementation-specific options.

        Returns:
            A ``PipelineStats`` snapshot.

        Raises:
            PipelineError: On pipeline-level failures.
        """
        ...
