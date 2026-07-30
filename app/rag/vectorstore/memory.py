"""MemoryVectorStore — an in-memory vector store implementation.

Dictionary-backed, deterministic ordering, O(1) lookup, O(n) search.
Supports cosine similarity, dot product, and negative Euclidean distance.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.config import VectorStoreConfig
from app.rag.vectorstore.errors import (
    VectorDimensionMismatchError,
    VectorNotFoundError,
    VectorStoreFullError,
)
from app.rag.vectorstore.metrics import SimilarityMetric, compute_similarity
from app.rag.vectorstore.models import SearchResult


class MemoryVectorStore(VectorStore):
    """In-memory vector store backed by a plain dict.

    All vectors are held in memory.  Search is O(n) — every vector is
    scored against the query.  Results are returned in descending order
    of similarity.

    Usage::

        store = MemoryVectorStore()
        store.add("chunk_1", (0.1, 0.2, 0.3))
        results = store.search((0.1, 0.2, 0.3), top_k=5)
    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        super().__init__(config)
        self._vectors: dict[str, tuple[float, ...]] = {}
        self._dimensions: int | None = None

    # ------------------------------------------------------------------
    # Mutation API
    # ------------------------------------------------------------------

    def add(self, chunk_id: str, vector: tuple[float, ...]) -> None:
        if self._config.max_vectors > 0 and chunk_id not in self._vectors:
            if len(self._vectors) >= self._config.max_vectors:
                raise VectorStoreFullError(
                    f"Vector store at capacity ({self._config.max_vectors})",
                    details={"max_vectors": self._config.max_vectors},
                )

        if self._config.validate_dimensions:
            self._validate_dimensions(vector)

        self._vectors[chunk_id] = vector

    def add_batch(self, items: Sequence[tuple[str, tuple[float, ...]]]) -> None:
        for chunk_id, vector in items:
            self.add(chunk_id, vector)

    def remove(self, chunk_id: str) -> bool:
        return self._vectors.pop(chunk_id, None) is not None

    def clear(self) -> None:
        self._vectors.clear()
        self._dimensions = None

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def get(self, chunk_id: str) -> tuple[float, ...] | None:
        return self._vectors.get(chunk_id)

    def contains(self, chunk_id: str) -> bool:
        return chunk_id in self._vectors

    def count(self) -> int:
        return len(self._vectors)

    def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int = 5,
    ) -> list[SearchResult]:
        if self._config.validate_dimensions:
            if self._dimensions is not None:
                if len(query_vector) != self._dimensions:
                    raise VectorDimensionMismatchError(
                        self._dimensions, len(query_vector),
                    )

        scored = []
        for chunk_id, vec in self._vectors.items():
            score = compute_similarity(query_vector, vec, self._config.metric)
            scored.append((score, chunk_id, vec))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(chunk_id=cid, score=sc, vector=vec)
            for sc, cid, vec in scored[:top_k]
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_dimensions(self, vector: tuple[float, ...]) -> None:
        if self._dimensions is None:
            self._dimensions = len(vector)
        elif len(vector) != self._dimensions:
            raise VectorDimensionMismatchError(self._dimensions, len(vector))
