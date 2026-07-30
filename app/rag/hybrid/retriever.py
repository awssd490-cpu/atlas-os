"""DefaultHybridRetriever — concrete hybrid retrieval implementation.

Combines keyword (lexical) and semantic (vector) retrieval using a
configurable fusion strategy.
"""

from __future__ import annotations

import time

from app.rag.hybrid.base import HybridRetriever
from app.rag.hybrid.config import HybridConfig
from app.rag.hybrid.errors import HybridError, InvalidHybridConfiguration
from app.rag.hybrid.fusion import FusionStrategy, reciprocal_rank_fusion, weighted_sum
from app.rag.hybrid.models import HybridResult, RetrievalScore
from app.rag.models import KnowledgeQuery


class DefaultHybridRetriever(HybridRetriever):
    """Concrete hybrid retriever that fuses keyword and semantic search.

    Requires a ``KnowledgeBase`` that has both a ``KnowledgeRetriever``
    and (optionally) an embedding provider + vector store.  Semantic
    retrieval is skipped if the vector store or embedding provider is
    not available; keyword retrieval is always attempted.

    Usage::

        retriever = DefaultHybridRetriever(knowledge_base, kb_retriever)
        result = await retriever.retrieve("What is Paris?")
    """

    def __init__(
        self,
        knowledge_base: object,
        keyword_retriever: object,
        config: HybridConfig | None = None,
    ) -> None:
        super().__init__(config)
        self._kb = knowledge_base
        self._keyword_retriever = keyword_retriever

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def knowledge_base(self) -> object:
        """Return the underlying knowledge base."""
        return self._kb

    @property
    def keyword_retriever(self) -> object:
        """Return the underlying keyword retriever."""
        return self._keyword_retriever

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int = 5) -> HybridResult:
        """Run hybrid retrieval for the given query.

        Args:
            query: The search query text.
            top_k: Maximum number of results to return.

        Returns:
            A ``HybridResult`` with fused and ranked scores.

        Raises:
            HybridError: On retrieval failure.
        """
        metadata: dict = {}
        start = time.monotonic()

        try:
            self._config.validate()
        except InvalidHybridConfiguration as exc:
            raise HybridError(
                "Invalid hybrid configuration",
                details={"config_error": str(exc)},
            ) from exc

        max_candidates = self._config.max_candidates

        # --- Step 1: keyword retrieval ---
        kw_start = time.monotonic()
        kw_scores: dict[str, float] = {}
        kw_ranked: list[str] = []

        try:
            kw_query = KnowledgeQuery(query=query, max_results=max_candidates)
            kw_result = await self._keyword_retriever.retrieve(kw_query)  # type: ignore[union-attr]
            for i, chunk in enumerate(kw_result.chunks or []):
                score = kw_result.sources[i].score if i < len(kw_result.sources) else 0.0
                kw_scores[chunk.chunk_id] = score
                kw_ranked.append(chunk.chunk_id)
        except Exception as exc:
            raise HybridError(
                f"Keyword retrieval failed: {exc}",
                details={"query": query},
            ) from exc

        kw_elapsed = (time.monotonic() - kw_start) * 1000
        metadata["keyword_elapsed_ms"] = round(kw_elapsed, 2)
        metadata["keyword_candidates"] = len(kw_scores)

        # --- Step 2: semantic retrieval ---
        sem_scores: dict[str, float] = {}
        sem_ranked: list[str] = []
        sem_elapsed = 0.0

        embedding_provider = getattr(self._kb, "embedding_provider", None)
        vector_store = getattr(self._kb, "vector_store", None)

        if embedding_provider is not None and vector_store is not None and query:
            sem_start = time.monotonic()
            try:
                emb_result = await embedding_provider.embed(query)  # type: ignore[union-attr]
                query_vec = emb_result.embeddings[0].vector if emb_result.embeddings else ()

                if query_vec:
                    vs_results = vector_store.search(query_vec, top_k=max_candidates)  # type: ignore[union-attr]
                    for sr in vs_results:
                        sem_scores[sr.chunk_id] = sr.score
                        sem_ranked.append(sr.chunk_id)
            except Exception as exc:
                raise HybridError(
                    f"Semantic retrieval failed: {exc}",
                    details={"query": query},
                ) from exc
            sem_elapsed = (time.monotonic() - sem_start) * 1000

        metadata["semantic_elapsed_ms"] = round(sem_elapsed, 2)
        metadata["semantic_candidates"] = len(sem_scores)

        # --- Step 3: fusion ---
        fusion_start = time.monotonic()
        cfg = self._config

        if cfg.fusion_strategy == FusionStrategy.RECIPROCAL_RANK_FUSION:
            fused = reciprocal_rank_fusion(kw_ranked, sem_ranked)
        else:
            fused = weighted_sum(
                kw_scores, sem_scores,
                keyword_weight=cfg.keyword_weight,
                semantic_weight=cfg.semantic_weight,
            )

        # Sort by descending final score
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)

        # Build RetrievalScore objects
        results: list[RetrievalScore] = []
        seen: set[str] = set()
        for cid, fscore in ranked:
            if cid in seen:
                continue
            seen.add(cid)
            results.append(RetrievalScore(
                chunk_id=cid,
                keyword_score=kw_scores.get(cid, 0.0),
                semantic_score=sem_scores.get(cid, 0.0),
                final_score=fscore,
            ))
            if len(results) >= top_k:
                break

        fusion_elapsed = (time.monotonic() - fusion_start) * 1000
        metadata["fusion_elapsed_ms"] = round(fusion_elapsed, 2)
        metadata["fusion_strategy"] = cfg.fusion_strategy.value
        metadata["total_elapsed_ms"] = round((time.monotonic() - start) * 1000, 2)

        return HybridResult(
            results=tuple(results),
            metadata=metadata,
        )
