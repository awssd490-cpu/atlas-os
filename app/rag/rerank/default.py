"""DefaultReranker — deterministic reranking using lightweight heuristics.

No external models, no ML libraries, no network calls.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from app.rag.rerank.base import Reranker
from app.rag.rerank.config import RerankConfig
from app.rag.rerank.models import RerankResponse, RerankedResult


class DefaultReranker(Reranker):
    """Deterministic reranker using lightweight text heuristics.

    Computes a ``rerank_score`` for each result based on:
    - **Lexical overlap**: fraction of query terms present in the chunk
    - **Query term coverage**: proportion of query terms matched
    - **Chunk length penalty**: chunks far from optimal length score lower
    - **Exact phrase bonus**: bonus if the exact query appears as a substring

    The final score is ``original_score + rerank_weight * rerank_score``.

    A ``content_provider`` callable can be supplied to look up chunk content
    by chunk ID.  Without one, only ``original_score`` is used.
    """

    def __init__(
        self,
        config: RerankConfig | None = None,
        *,
        content_provider: Callable[[str], str | None] | None = None,
        rerank_weight: float = 1.0,
        length_penalty_exponent: float = 0.3,
        phrase_bonus: float = 0.5,
    ) -> None:
        super().__init__(config)
        self._content_provider = content_provider
        self._rerank_weight = rerank_weight
        self._length_penalty_exponent = length_penalty_exponent
        self._phrase_bonus = phrase_bonus

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rerank_weight(self) -> float:
        return self._rerank_weight

    @property
    def content_provider(self) -> Callable[[str], str | None] | None:
        return self._content_provider

    @content_provider.setter
    def content_provider(self, provider: Callable[[str], str | None] | None) -> None:
        """Set the content provider after construction.

        This allows the ContextBuilder to inject a content lookup
        function before calling ``rerank()``.
        """
        self._content_provider = provider

    # ------------------------------------------------------------------
    # Reranking API
    # ------------------------------------------------------------------

    async def rerank(
        self,
        query: str,
        results: list[tuple[str, float]],
    ) -> RerankResponse:
        """Rerank a list of retrieval results.

        When a ``content_provider`` is configured, each chunk's content
        is fetched to compute the heuristic ``rerank_score``.  Without
        one, the original score is used as the final score.

        Args:
            query: The original search query.
            results: A sequence of ``(chunk_id, score)`` pairs from
                the retrieval stage, ordered by descending score.

        Returns:
            A ``RerankResponse`` with reranked results.
        """
        start = time.monotonic()

        metadata: dict[str, Any] = {
            "rerank_weight": self._rerank_weight,
            "total_candidates": len(results),
        }

        if not query or not results:
            return RerankResponse(metadata=metadata)

        reranked: list[RerankedResult] = []

        for chunk_id, original_score in results:
            rerank_score = 0.0

            if self._content_provider is not None:
                content = self._content_provider(chunk_id)
                if content:
                    result = self._score_content(query, content, original_score)
                    rerank_score = result.rerank_score
                    final_score = result.final_score
                else:
                    final_score = original_score
            else:
                final_score = original_score

            reranked.append(RerankedResult(
                chunk_id=chunk_id,
                original_score=original_score,
                rerank_score=rerank_score,
                final_score=final_score,
            ))

        # Sort by final_score descending, then original_score descending
        # (deterministic tie-break)
        reranked.sort(key=lambda r: (r.final_score, r.original_score), reverse=True)

        # Apply top_k and score_threshold
        threshold = self._config.score_threshold
        top_k = self._config.top_k

        filtered = [
            r for r in reranked
            if r.final_score >= threshold
        ][:top_k]

        elapsed = (time.monotonic() - start) * 1000

        metadata.update({
            "returned": len(filtered),
            "elapsed_ms": round(elapsed, 2),
        })

        return RerankResponse(results=tuple(filtered), metadata=metadata)

    # ------------------------------------------------------------------
    # Public scoring methods
    # ------------------------------------------------------------------

    def score(
        self,
        query: str,
        chunk_content: str,
        original_score: float = 0.0,
    ) -> RerankedResult:
        """Compute a reranked score for a single chunk given its content.

        This is the core scoring method.  Callers with access to chunk
        content can use this directly; the ``rerank()`` method delegates
        to ``_score_content()`` when a ``content_provider`` is set.

        Args:
            query: The search query.
            chunk_content: The chunk text content.
            original_score: The score from the retrieval stage.

        Returns:
            A ``RerankedResult`` with computed scores.  ``chunk_id`` is
            empty — the caller should fill it in.
        """
        result = self._score_content(query, chunk_content, original_score)
        return RerankedResult(
            chunk_id="",
            original_score=result.original_score,
            rerank_score=result.rerank_score,
            final_score=result.final_score,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_content(
        self,
        query: str,
        content: str,
        original_score: float,
    ) -> RerankedResult:
        """Core scoring: compute rerank_score from query and chunk content."""
        query_lower = query.lower().strip()
        chunk_lower = content.lower().strip()

        query_terms = [t for t in query_lower.split() if t]
        if not query_terms or not chunk_lower:
            return RerankedResult(
                original_score=original_score,
                rerank_score=0.0,
                final_score=original_score,
            )

        # --- Lexical overlap & term coverage ---
        matched = sum(1 for t in query_terms if t in chunk_lower)
        term_coverage = matched / len(query_terms)

        # --- Chunk length penalty ---
        length = len(chunk_lower)
        # Ideal length: 200 chars.  Penalty is a power curve centred on 200.
        norm_length = length / 200.0
        length_factor = max(0.1, min(1.0, norm_length ** self._length_penalty_exponent
                                      if norm_length < 1.0
                                      else norm_length ** (-self._length_penalty_exponent)))
        # Invert: 1.0 at ideal, approaching 0 for very far
        length_bonus = length_factor

        # --- Exact phrase bonus ---
        phrase_bonus = self._phrase_bonus if query_lower in chunk_lower else 0.0

        # --- Compute rerank score ---
        rerank_score = (
            term_coverage * 0.5
            + length_bonus * 0.25
            + phrase_bonus * 0.25
        )
        rerank_score = max(0.0, min(2.0, rerank_score))

        final_score = original_score + self._rerank_weight * rerank_score

        return RerankedResult(
            original_score=original_score,
            rerank_score=rerank_score,
            final_score=final_score,
        )
