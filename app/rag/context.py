"""KnowledgeContextBuilder — merges retrieved knowledge into provider context.

Operates independently of Memory context.  The Agent Runtime merges
both contexts before sending to the provider.
"""

from __future__ import annotations

from typing import Any

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


class KnowledgeContextBuilder:
    """Retrieves knowledge and formats it for provider injection.

    Uses hybrid retrieval when the knowledge base has both an embedding
    provider and a vector store configured; otherwise falls back to
    keyword retrieval.

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
        """Build context using keyword retrieval."""
        kg_query = KnowledgeQuery(
            query=query,
            max_results=max_chunks,
            min_score=min_score,
        )
        result = await self._retriever.retrieve(kg_query)  # type: ignore[union-attr]

        if not result.chunks:
            return KnowledgeContext(total_chunks=0)

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
        """Build context using hybrid retrieval, then map to KnowledgeChunks."""
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

        text = self._format_chunks(chunks)

        return KnowledgeContext(
            text=text,
            chunks=chunks,
            sources=sources,
            total_chunks=len(chunks),
        )

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
