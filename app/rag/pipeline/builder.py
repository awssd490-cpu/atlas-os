"""PipelineBuilder — fluent builder for DefaultKnowledgePipeline.

Provides a chainable API for constructing pipeline instances without
manually wiring constructor arguments.  Validates that all required
components are present before building.
"""

from __future__ import annotations

from typing import Any

from app.rag.chunking import ChunkingEngine
from app.rag.knowledge_base import KnowledgeBase
from app.rag.pipeline.config import PipelineConfig
from app.rag.pipeline.default import DefaultKnowledgePipeline
from app.rag.pipeline.errors import InvalidPipelineConfiguration

if True:  # TYPE_CHECKING-compatible import guard
    from collections.abc import Callable

    from app.rag.embeddings.base import EmbeddingProvider
    from app.rag.rerank.base import Reranker
    from app.rag.vectorstore.base import VectorStore

    Loader = Callable[[str], list[Any]]


class PipelineBuilder:
    """Fluent builder for constructing ``DefaultKnowledgePipeline`` instances.

    Usage::

        pipeline = (
            PipelineBuilder()
            .loader(my_loader)
            .chunker(ChunkingEngine())
            .knowledge_base(KnowledgeBase())
            .embedding_provider(provider)
            .vector_store(store)
            .config(PipelineConfig(auto_embed=True))
            .build()
        )
    """

    def __init__(self) -> None:
        self._loader: Loader | None = None
        self._chunker: ChunkingEngine | None = None
        self._knowledge_base: KnowledgeBase | None = None
        self._embedding_provider: EmbeddingProvider | None = None
        self._vector_store: VectorStore | None = None
        self._reranker: Reranker | None = None
        self._config: PipelineConfig | None = None

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def loader(
        self,
        loader: Loader,
    ) -> PipelineBuilder:
        """Set the loader callable.

        Args:
            loader: A callable that accepts a path string and returns a
                list of ``KnowledgeDocument`` objects.

        Returns:
            The builder for chaining.
        """
        self._loader = loader
        return self

    def chunker(
        self,
        chunker: ChunkingEngine,
    ) -> PipelineBuilder:
        """Set the chunking engine.

        Args:
            chunker: A ``ChunkingEngine`` instance.

        Returns:
            The builder for chaining.
        """
        self._chunker = chunker
        return self

    def knowledge_base(
        self,
        knowledge_base: KnowledgeBase,
    ) -> PipelineBuilder:
        """Set the knowledge base.

        Args:
            knowledge_base: A ``KnowledgeBase`` instance.

        Returns:
            The builder for chaining.
        """
        self._knowledge_base = knowledge_base
        return self

    def embedding_provider(
        self,
        provider: EmbeddingProvider | None,
    ) -> PipelineBuilder:
        """Set the embedding provider (optional).

        Args:
            provider: An ``EmbeddingProvider`` instance, or ``None``.

        Returns:
            The builder for chaining.
        """
        self._embedding_provider = provider
        return self

    def vector_store(
        self,
        store: VectorStore | None,
    ) -> PipelineBuilder:
        """Set the vector store (optional).

        Args:
            store: A ``VectorStore`` instance, or ``None``.

        Returns:
            The builder for chaining.
        """
        self._vector_store = store
        return self

    def reranker(
        self,
        reranker: Reranker | None,
    ) -> PipelineBuilder:
        """Set the reranker (optional).

        The reranker is wired onto the knowledge base before the
        pipeline is constructed, enabling automatic reranking during
        search.

        Args:
            reranker: A ``Reranker`` instance, or ``None``.

        Returns:
            The builder for chaining.
        """
        self._reranker = reranker
        return self

    def config(
        self,
        config: PipelineConfig,
    ) -> PipelineBuilder:
        """Set the pipeline configuration (optional).

        Args:
            config: A ``PipelineConfig`` instance.

        Returns:
            The builder for chaining.
        """
        self._config = config
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> DefaultKnowledgePipeline:
        """Construct a ``DefaultKnowledgePipeline`` with the configured
        components.

        Each call returns a **fresh** pipeline instance.  The builder's
        stored references are not mutated, so multiple calls to
        ``build()`` produce independent pipelines.

        Returns:
            A new ``DefaultKnowledgePipeline`` instance.

        Raises:
            InvalidPipelineConfiguration: If any required component
                (loader, chunker, knowledge_base) has not been set.
        """
        missing: list[str] = []

        if self._loader is None:
            missing.append("loader")
        if self._chunker is None:
            missing.append("chunker")
        if self._knowledge_base is None:
            missing.append("knowledge_base")

        if missing:
            msg = "Required pipeline components are missing: " + ", ".join(missing)
            raise InvalidPipelineConfiguration(
                msg,
                details={"missing": missing},
            )

        # Resolve the knowledge base — when a reranker is provided, wire
        # it onto the KB before building the pipeline.
        kb = self._knowledge_base
        if self._reranker is not None:
            # The KnowledgeBase stores its reranker as a private attribute
            # with no public setter, so we use the private API directly.
            kb._reranker = self._reranker  # type: ignore[attr-defined]

        return DefaultKnowledgePipeline(
            loader=self._loader,
            chunker=self._chunker,
            knowledge_base=kb,
            embedding_provider=self._embedding_provider,
            vector_store=self._vector_store,
            config=self._config,
        )
