"""KnowledgeContextBuilder — merges retrieved knowledge into provider context.

Operates independently of Memory context.  The Agent Runtime merges
both contexts before sending to the provider.
"""

from __future__ import annotations

from typing import Any

from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import (
    KnowledgeContext,
    KnowledgeDocument,
    KnowledgeQuery,
    KnowledgeResult,
)
from app.rag.retriever import KnowledgeRetriever


class KnowledgeContextBuilder:
    """Retrieves knowledge and formats it for provider injection.

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

        kg_query = KnowledgeQuery(
            query=query,
            max_results=max_chunks,
            min_score=min_score,
        )
        result = await self._retriever.retrieve(kg_query)

        if not result.chunks:
            return KnowledgeContext(total_chunks=0)

        text = self._format_chunks(result.chunks)

        return KnowledgeContext(
            text=text,
            chunks=result.chunks,
            sources=result.sources,
            total_chunks=len(result.chunks),
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
