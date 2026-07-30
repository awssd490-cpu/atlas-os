"""KnowledgeContextBuilder — merges retrieved knowledge into provider context.

Operates independently of Memory context.  The Agent Runtime merges
both contexts before sending to the provider.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import (
    KnowledgeChunk,
    KnowledgeContext,
    KnowledgeDocument,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeSource,
)
from app.rag.retriever import KnowledgeRetriever

if TYPE_CHECKING:
    from app.rag.rerank.base import Reranker


class KnowledgeContextBuilder:
    """Retrieves knowledge and formats it for provider injection.

    Uses hybrid retrieval when the knowledge base has both an embedding
    provider and a vector store configured; otherwise falls back to
    keyword retrieval.  If a reranker is configured on the knowledge
    base, retrieved results are reranked automatically.

    Usage::

        builder = KnowledgeContextBuilder(knowledge_base)
        context = await builder.build("What is the capital of France?")
        print(context.text)
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase | None = None,
        retriever: KnowledgeRetriever | None = None,
    ) -> None:
        self._kb = knowledge_base
        self._retriever = retriever or (
            KnowledgeRetriever(knowledge_base) if knowledge_base else None
        )

    async def build(
        self,
        query: str = "",
        *,
        max_chunks: int = 10,
        min_score: float = 0.0,
        format_as: str = "text",
    ) -> KnowledgeContext:
        """Retrieve and format knowledge for provider context.

        When the underlying ``KnowledgeBase`` has both an embedding
        provider and a vector store configured, hybrid retrieval is
        used automatically.  Otherwise keyword retrieval is used.

        If a reranker is configured on the knowledge base, results are
        reranked before being formatted into the context.

        Args:
            query: The search query.
            max_chunks: Maximum chunks to include.
            min_score: Minimum relevance score.
            format_as: Output format (``"text"`` only currently).

        Returns:
            A ``KnowledgeContext`` with formatted text and sources.
        """
        if not self._retriever or not query:
            return KnowledgeContext()

        # Auto-detect hybrid retriever when available
        hybrid = getattr(self._kb, "hybrid_retriever", None) if self._kb else None

        if hybrid is not None:
            return await self._build_hybrid(hybrid, query, max_chunks)

        return await self._build_keyword(query, max_chunks, min_score)

    async def _build_keyword(
        self,
        query: str,
        max_chunks: int,
        min_score: float,
    ) -> KnowledgeContext:
        """Build context using keyword retrieval, optionally reranked."""
        kg_query = KnowledgeQuery(
            query=query,
            max_results=max_chunks,
            min_score=min_score,
        )
        result = await self._retriever.retrieve(kg_query)  # type: ignore[union-attr]

        if not result.chunks:
            return KnowledgeContext(total_chunks=0)

        # Rerank if configured
        reranker = getattr(self._kb, "reranker", None) if self._kb else None
        if reranker is not None:
            return await self._apply_reranker(reranker, query, result.chunks, max_chunks)

        text = self._format_chunks(result.chunks)

        return KnowledgeContext(
            text=text,
            chunks=result.chunks,
            sources=result.sources,
            total_chunks=len(result.chunks),
        )

    async def _build_hybrid(
        self,
        hybrid: Any,
        query: str,
        max_chunks: int,
    ) -> KnowledgeContext:
        """Build context using hybrid retrieval, optionally reranked."""
        hy_result = await hybrid.retrieve(query, top_k=max_chunks)

        if not hy_result.results:
            return KnowledgeContext(total_chunks=0)

        chunks: list[KnowledgeChunk] = []
        sources: list[KnowledgeSource] = []

        for rs in hy_result.results:
            chunk = self._kb.get_chunk(rs.chunk_id) if self._kb else None  # type: ignore[union-attr]
            if chunk is None:
                continue

            chunks.append(chunk)

            doc = self._kb.get(chunk.document_id) if self._kb else None  # type: ignore[union-attr]
            sources.append(KnowledgeSource(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                title=doc.title if doc else "",
                score=rs.final_score,
            ))

        # Rerank if configured
        reranker = getattr(self._kb, "reranker", None) if self._kb else None
        if reranker is not None:
            return await self._apply_reranker(reranker, query, chunks, max_chunks)

        text = self._format_chunks(chunks)

        return KnowledgeContext(
            text=text,
            chunks=chunks,
            sources=sources,
            total_chunks=len(chunks),
        )

    async def _apply_reranker(
        self,
        reranker: Reranker,
        query: str,
        chunks: list[KnowledgeChunk],
        max_chunks: int,
    ) -> KnowledgeContext:
        """Apply a reranker to a list of chunks and produce a KnowledgeContext.

        If the reranker is disabled via ``config.enabled``, the chunks
        are returned in their original order without reranking.
        """
        # Check if reranking is enabled
        if not reranker.config.enabled:
            text = self._format_chunks(chunks)
            sources = self._build_sources(chunks)
            return KnowledgeContext(text=text, chunks=chunks, sources=sources, total_chunks=len(chunks))

        # Build content lookup from the chunk list
        content_map = {c.chunk_id: c.content for c in chunks}
        if hasattr(reranker, 'content_provider') and reranker.content_provider is None:  # type: ignore
            reranker.content_provider = content_map.get  # type: ignore

        # Collect (chunk_id, score) pairs
        results: list[tuple[str, float]] = [
            (c.chunk_id, 0.0) for c in chunks
        ]

        response = await reranker.rerank(query, results)
        reranked_ids = {r.chunk_id: r.final_score for r in response.results}

        # Reorder chunks by reranked score
        id_order = [r.chunk_id for r in response.results]
        chunk_map = {c.chunk_id: c for c in chunks}

        reranked_chunks: list[KnowledgeChunk] = []
        for cid in id_order:
            chunk = chunk_map.get(cid)
            if chunk is not None:
                reranked_chunks.append(chunk)

        sources = [
            KnowledgeSource(
                document_id=reranked_chunks[i].document_id,
                chunk_id=reranked_chunks[i].chunk_id,
                title=(
                    doc.title
                    if (doc := (self._kb.get(reranked_chunks[i].document_id) if self._kb else None))
                    else ""
                ),
                score=reranked_ids.get(reranked_chunks[i].chunk_id, 0.0),
            )
            for i in range(len(reranked_chunks))
        ]

        text = self._format_chunks(reranked_chunks)

        return KnowledgeContext(
            text=text,
            chunks=reranked_chunks,
            sources=sources,
            total_chunks=len(reranked_chunks),
        )

    def _build_sources(self, chunks: list[KnowledgeChunk]) -> list[KnowledgeSource]:
        """Build KnowledgeSource entries for a list of chunks."""
        sources: list[KnowledgeSource] = []
        for chunk in chunks:
            doc = self._kb.get(chunk.document_id) if self._kb else None  # type: ignore[union-attr]
            sources.append(KnowledgeSource(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                title=doc.title if doc else "",
                score=0.0,
            ))
        return sources

    @staticmethod
    def _format_chunks(chunks: list[Any]) -> str:
        """Format chunks as plain text.

        Args:
            chunks: The knowledge chunks to format.

        Returns:
            A plain-text string ready for injection.
        """
        if not chunks:
            return ""

        lines: list[str] = ["Relevant knowledge:"]
        for chunk in chunks:
            lines.append(f"- {chunk.content}")

        return "\n".join(lines)
