"""DefaultKnowledgePipeline — concrete pipeline implementation.

Orchestrates document ingestion and search: loading, chunking,
registering in the knowledge base, optionally generating embeddings
and indexing vectors, and retrieving results through the existing
retrieval stack (keyword, hybrid, and reranking).
"""

from __future__ import annotations

import time
from typing import Any

from app.rag.chunking import ChunkingEngine
from app.rag.context import KnowledgeContextBuilder
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeDocument
from app.rag.pipeline.base import KnowledgePipeline
from app.rag.pipeline.config import PipelineConfig
from app.rag.pipeline.errors import PipelineError
from app.rag.pipeline.models import PipelineResult, PipelineStats

if True:  # TYPE_CHECKING-compatible import guard
    from collections.abc import Callable

    from app.rag.embeddings.base import EmbeddingProvider
    from app.rag.vectorstore.base import VectorStore

    Loader = Callable[[str], list[KnowledgeDocument]]


class DefaultKnowledgePipeline(KnowledgePipeline):
    """Concrete pipeline that ingests documents and searches a KnowledgeBase.

    The pipeline uses an external **loader** callable to produce
    ``KnowledgeDocument`` objects from a path, a **chunker** (typically
    a ``ChunkingEngine``) to split document text into chunks, and an
    optional **embedding provider** and **vector store** for automatic
    embedding and indexing.

    Search is delegated to the existing retrieval stack via
    :class:`KnowledgeContextBuilder`, which automatically selects
    keyword or hybrid retrieval based on the knowledge base's
    configuration and applies a configured reranker when available.

    Behaviour is controlled by :class:`PipelineConfig`:

    * ``auto_embed`` — generate embeddings during ingestion.
    * ``auto_index`` — insert vectors into the vector store.
    * ``batch_size`` — chunk count per embedding batch.
    * ``auto_rerank`` — (reserved for future use; reranking is
      configured on the ``KnowledgeBase`` directly).

    Usage::

        pipeline = DefaultKnowledgePipeline(
            loader=my_loader,
            chunker=ChunkingEngine(),
            knowledge_base=KnowledgeBase(),
            embedding_provider=provider,
            vector_store=store,
        )
        count = await pipeline.ingest("/path/to/docs")
        result = await pipeline.search("search query")
    """

    def __init__(
        self,
        loader: Loader,
        chunker: ChunkingEngine,
        knowledge_base: KnowledgeBase,
        embedding_provider: EmbeddingProvider | None = None,
        vector_store: VectorStore | None = None,
        config: PipelineConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._loader = loader
        self._chunker = chunker
        self._kb = knowledge_base
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store

        # Context builder — delegates to the existing retrieval stack
        from app.rag.retriever import KnowledgeRetriever

        self._context_builder = KnowledgeContextBuilder(
            knowledge_base=knowledge_base,
            retriever=KnowledgeRetriever(knowledge_base) if knowledge_base else None,
        )

        # Mutable counters (PipelineStats is frozen)
        self._doc_count: int = 0
        self._chunk_count: int = 0
        self._vector_count: int = 0
        self._search_count: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def loader(self) -> Loader:
        """Return the loader callable."""
        return self._loader

    @property
    def chunker(self) -> ChunkingEngine:
        """Return the chunking engine."""
        return self._chunker

    @property
    def knowledge_base(self) -> KnowledgeBase:
        """Return the underlying knowledge base."""
        return self._kb

    @property
    def embedding_provider(self) -> EmbeddingProvider | None:
        """Return the embedding provider, or ``None``."""
        return self._embedding_provider

    @property
    def vector_store(self) -> VectorStore | None:
        """Return the vector store, or ``None``."""
        return self._vector_store

    # ------------------------------------------------------------------
    # Ingestion API
    # ------------------------------------------------------------------

    async def ingest(
        self,
        path: str,
        **kwargs: Any,
    ) -> int:
        """Load documents from *path* and ingest them.

        Args:
            path: File-system path or resource identifier understood by
                the configured loader.
            **kwargs: Forwarded to :meth:`ingest_documents`.

        Returns:
            The number of documents successfully ingested.
        """
        documents = self._loader(path)
        result = await self.ingest_documents(documents, **kwargs)
        return result.metadata.get("documents_ingested", 0)

    async def ingest_documents(
        self,
        documents: list[KnowledgeDocument],
        **kwargs: Any,
    ) -> PipelineResult:
        """Ingest pre-loaded documents.

        Each document is chunked, registered in the knowledge base, and
        optionally embedded and indexed.

        Args:
            documents: One or more ``KnowledgeDocument`` objects.
            **kwargs: Implementation-specific options (currently unused).

        Returns:
            A ``PipelineResult`` with metadata about the ingestion.

        Raises:
            PipelineError: On critical pipeline failures.
        """
        start = time.monotonic()
        config = self._config

        ingested_docs = 0
        ingested_chunks = 0
        ingested_vectors = 0

        for doc in documents:
            # Chunk the document text
            try:
                result = self._chunker.chunk(
                    doc.content,
                    document_id=doc.document_id,
                )
            except Exception as exc:
                raise PipelineError(
                    f"Failed to chunk document {doc.document_id!r}: {exc}",
                    details={"document_id": doc.document_id},
                ) from exc

            chunks = result.chunks

            # Build a document with the produced chunks
            chunked_doc = KnowledgeDocument(
                document_id=doc.document_id,
                title=doc.title,
                content=doc.content,
                chunks=chunks,
                metadata=doc.metadata,
            )

            # Register in the knowledge base (skip duplicates)
            try:
                self._kb.register(chunked_doc)
            except Exception:
                continue

            ingested_docs += 1
            ingested_chunks += len(chunks)

            # --- Auto-embed & auto-index ---
            if config.auto_embed and self._embedding_provider is not None and chunks:
                vec_count = await self._embed_and_index(chunks, config)
                ingested_vectors += vec_count
            elif config.auto_index and self._vector_store is not None and chunks:
                # Index-only mode when embedding is disabled but the
                # knowledge base already stored embeddings internally.
                for c in chunks:
                    vec = self._kb.get_embedding(c.chunk_id)
                    if vec is not None:
                        self._vector_store.add(c.chunk_id, vec.vector)
                        ingested_vectors += 1

        elapsed = time.monotonic() - start

        # Update accumulated counters
        self._doc_count += ingested_docs
        self._chunk_count += ingested_chunks
        self._vector_count += ingested_vectors

        return PipelineResult(
            metadata={
                "documents_ingested": ingested_docs,
                "chunks_created": ingested_chunks,
                "vectors_created": ingested_vectors,
                "elapsed_time": round(elapsed, 4),
                "embedding_enabled": config.auto_embed
                and self._embedding_provider is not None,
                "indexing_enabled": config.auto_index
                and self._vector_store is not None,
            },
        )

    # ------------------------------------------------------------------
    # Search API
    # ------------------------------------------------------------------

    async def search(
        self,
        query: str,
        **kwargs: Any,
    ) -> PipelineResult:
        """Search the knowledge base for relevant content.

        Delegates to the existing ``KnowledgeContextBuilder`` which
        automatically selects keyword or hybrid retrieval based on the
        knowledge base's configuration and applies a configured reranker
        when available.

        Args:
            query: The search query string.
            **kwargs: Forwarded to ``KnowledgeContextBuilder.build()``.
                Supported options include ``max_chunks``, ``min_score``,
                and ``format_as``.

        Returns:
            A ``PipelineResult`` with the generated context text and
            metadata about the search execution.

        Raises:
            PipelineError: On retrieval failures.
        """
        if not query or not query.strip():
            return PipelineResult()

        start = time.monotonic()

        max_chunks = kwargs.get("max_chunks", 10)
        min_score = kwargs.get("min_score", 0.0)
        format_as = kwargs.get("format_as", "text")

        try:
            context = await self._context_builder.build(
                query=query,
                max_chunks=max_chunks,
                min_score=min_score,
                format_as=format_as,
            )
        except Exception as exc:
            raise PipelineError(
                f"Search failed: {exc}",
                details={"query": query},
            ) from exc

        elapsed = time.monotonic() - start

        # Update search counter
        self._search_count += 1

        # Determine retrieval mode from the knowledge base's capabilities
        hybrid = self._kb.hybrid_retriever is not None
        reranker = self._kb.reranker
        reranking_enabled = reranker is not None and reranker.config.enabled

        return PipelineResult(
            context=context.text,
            metadata={
                "query": query,
                "retrieval_mode": "hybrid" if hybrid else "keyword",
                "reranking_enabled": reranking_enabled,
                "chunks_returned": context.total_chunks,
                "elapsed_time": round(elapsed, 4),
            },
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def clear(self, **kwargs: Any) -> None:
        """Clear all data from the pipeline.

        Resets the knowledge base, the vector store (if present), and
        all internal counters.
        """
        self._kb.clear()
        if self._vector_store is not None:
            self._vector_store.clear()
        self._doc_count = 0
        self._chunk_count = 0
        self._vector_count = 0
        self._search_count = 0

    async def stats(self, **kwargs: Any) -> PipelineStats:
        """Return current pipeline statistics.

        Returns:
            A ``PipelineStats`` snapshot with accumulated document,
            chunk, vector, and search counts.
        """
        return PipelineStats(
            documents=self._doc_count,
            chunks=self._chunk_count,
            vectors=self._vector_count,
            searches=self._search_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _embed_and_index(
        self,
        chunks: Any,
        config: PipelineConfig,
    ) -> int:
        """Generate embeddings for *chunks* and optionally index them.

        Returns the number of vectors produced.
        """
        texts = [c.content for c in chunks]
        batch_size = config.batch_size
        total = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_chunks = chunks[i : i + batch_size]

            try:
                emb_result = await self._embedding_provider.embed_batch(  # type: ignore[union-attr]
                    batch_texts
                )
            except Exception as exc:
                raise PipelineError(
                    f"Embedding batch failed at offset {i}: {exc}",
                    details={"batch_offset": i, "batch_size": len(batch_texts)},
                ) from exc

            for chunk, vec in zip(batch_chunks, emb_result.embeddings):
                # Store the embedding on the knowledge base
                self._kb._embeddings[chunk.chunk_id] = vec  # type: ignore[attr-defined]

                # Index in the vector store when auto_index is enabled
                if config.auto_index and self._vector_store is not None:
                    self._vector_store.add(chunk.chunk_id, vec.vector)

                total += 1

        return total
