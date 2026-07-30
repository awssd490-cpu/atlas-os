"""
Custom reranker example.

Subclasses ``Reranker`` and implements a simple length-based scorer.
"""

from app.rag.rerank import Reranker
from app.rag.rerank.models import RerankResponse, RerankedResult


class LengthReranker(Reranker):
    """Reranker that prefers shorter chunks."""

    async def rerank(
        self,
        query: str,
        results: list[tuple[str, float]],
    ) -> RerankResponse:
        reranked: list[RerankedResult] = []
        for chunk_id, score in results:
            chunk_length = len(chunk_id)  # simplified — use content length in practice
            length_bonus = max(0.0, 1.0 - chunk_length / 100.0)
            final_score = score + length_bonus * 0.1
            reranked.append(RerankedResult(
                chunk_id=chunk_id,
                original_score=score,
                rerank_score=length_bonus * 0.1,
                final_score=final_score,
            ))
        reranked.sort(key=lambda r: r.final_score, reverse=True)
        return RerankResponse(results=tuple(reranked))
