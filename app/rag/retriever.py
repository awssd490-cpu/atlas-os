"""KnowledgeRetriever — retrieves relevant knowledge chunks.

Current implementation uses simple keyword matching.  The public
interface is designed to remain stable when future implementations
add embeddings, BM25, hybrid search, or vector DB support.

Retriever is independent of KnowledgeBase storage — it only queries
through the ``KnowledgeBase`` interface.
"""

from __future__ import annotations

import time
from typing import Any

from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import (
    KnowledgeChunk,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeSource,
)


class KnowledgeRetriever:
    """Retrieves relevant knowledge chunks for a query.

    Current strategy: simple case-insensitive keyword matching against
    chunk content and document title/description.

    Future strategies (interface remains the same):
    - Embedding similarity
    - BM25
    - Hybrid search
    - Vector DB

    Usage::

        retriever = KnowledgeRetriever(knowledge_base)
        result = await retriever.retrieve(
            KnowledgeQuery(query="What is Paris?", max_results=5)
        )
    """

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base

    async def retrieve(
        self,
        query: KnowledgeQuery,
    ) -> KnowledgeResult:
        """Retrieve knowledge chunks matching the query.

        Args:
            query: The query with search parameters.

        Returns:
            A ``KnowledgeResult`` with matching chunks and sources.
        """
        start = time.monotonic()
        query_lower = query.query.lower().strip()
        if not query_lower:
            return KnowledgeResult(query=query.query)

        query_terms = query_lower.split()

        scored_chunks: list[tuple[float, KnowledgeChunk]] = []

        for chunk in self._kb.list_chunks():
            score = self._score_chunk(chunk, query_lower, query_terms)
            if score > 0 and score >= query.min_score:
                scored_chunks.append((score, chunk))

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)

        # Apply limit
        top_chunks = scored_chunks[: query.max_results]

        sources: list[KnowledgeSource] = []
        for score, chunk in top_chunks:
            doc = self._kb.get(chunk.document_id)
            sources.append(KnowledgeSource(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                title=doc.title if doc else "",
                score=score,
            ))

        elapsed = (time.monotonic() - start) * 1000

        return KnowledgeResult(
            chunks=[c for _, c in top_chunks],
            sources=sources,
            query=query.query,
            total=len(scored_chunks),
            elapsed_ms=round(elapsed, 2),
        )

    @staticmethod
    def _score_chunk(
        chunk: KnowledgeChunk,
        query_lower: str,
        query_terms: list[str],
    ) -> float:
        """Score a chunk against the query.

        Simple keyword frequency scoring.  A chunk scores higher when
        more query terms appear in its content.

        Args:
            chunk: The chunk to score.
            query_lower: The full query in lowercase.
            query_terms: Individual query terms.

        Returns:
            A relevance score (0.0 = no match).
        """
        content_lower = chunk.content.lower()
        score = 0.0

        # Exact phrase match
        if query_lower in content_lower:
            score += 10.0

        # Individual term matches
        for term in query_terms:
            count = content_lower.count(term)
            score += count

        # Bonus for term in title/proximity
        if query_lower in content_lower[:200]:
            score += 5.0

        return score
