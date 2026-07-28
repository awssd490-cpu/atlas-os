"""In-memory vector store for testing and development.

Pure-Python implementation using basic math (no numpy).  Suitable for
unit tests and small-scale development.  NOT for production.
"""

from __future__ import annotations

import math
from typing import Any

from app.storage.interfaces import (
    SearchResult,
    VectorRecord,
    VectorStore,
)


class InMemoryVectorStore(VectorStore):
    """Vector store backed by an in-memory dict.

    Similarity is computed via cosine similarity.  Namespaces are
    isolated by prefixing the internal key.
    """

    def __init__(self) -> None:
        # namespace -> {id: VectorRecord}
        self._data: dict[str, dict[str, VectorRecord]] = {}
        self._default_namespace = "default"

    async def upsert(self, record: VectorRecord) -> None:
        """Insert or update a vector record."""
        ns = record.namespace or self._default_namespace
        if ns not in self._data:
            self._data[ns] = {}
        self._data[ns][record.id] = record

    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Nearest-neighbor search via cosine similarity.

        Args:
            vector: Query embedding.
            limit: Maximum results.
            namespace: Restrict search.  ``None`` searches all namespaces.
            filter: Optional metadata filter (key-value equality match).

        Returns:
            Results sorted by similarity, highest first.
        """
        candidates: list[tuple[str, VectorRecord, float]] = []

        namespaces_to_search = (
            [namespace] if namespace else list(self._data.keys())
        )

        for ns in namespaces_to_search:
            ns_data = self._data.get(ns, {})
            for record_id, record in ns_data.items():
                # Metadata filter
                if filter is not None:
                    if not self._matches_filter(record.metadata, filter):
                        continue

                score = self._cosine_similarity(vector, record.vector)
                candidates.append((record_id, record, score))

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        return [
            SearchResult(
                id=cid,
                score=score,
                metadata=record.metadata,
            )
            for cid, record, score in candidates[:limit]
        ]

    async def delete(self, id: str, *, namespace: str | None = None) -> None:
        """Remove a vector by ID."""
        ns = namespace or self._default_namespace
        ns_data = self._data.get(ns, {})
        ns_data.pop(id, None)

    async def list_ids(self, *, namespace: str | None = None) -> list[str]:
        """Return all vector IDs in a namespace."""
        ns = namespace or self._default_namespace
        return list(self._data.get(ns, {}).keys())

    async def count(self, *, namespace: str | None = None) -> int:
        """Return the number of vectors in a namespace."""
        ns = namespace or self._default_namespace
        return len(self._data.get(ns, {}))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors.

        Returns 0.0 for zero-vector inputs to avoid division by zero.
        """
        if not a or not b or len(a) != len(b):
            return 0.0

        dot_product = sum(x * y for x, y in zip(a, b, strict=False))
        magnitude_a = math.sqrt(sum(x * x for x in a))
        magnitude_b = math.sqrt(sum(y * y for y in b))

        if magnitude_a == 0.0 or magnitude_b == 0.0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filter: dict[str, Any]) -> bool:
        """Check if metadata matches all key-value pairs in the filter."""
        for key, value in filter.items():
            if metadata.get(key) != value:
                return False
        return True
