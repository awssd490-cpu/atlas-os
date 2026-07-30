"""Base abstractions for the reranking layer.

Defines the ``Reranker`` abstract base class that all reranker
implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.rag.rerank.config import RerankConfig
from app.rag.rerank.models import RerankResponse


class Reranker(ABC):
    """Abstract base class for rerankers.

    A reranker takes a query and a list of retrieval results and
    produces a new ranked list with improved ordering.

    Concrete subclasses must implement ``rerank()``.
    """

    def __init__(self, config: RerankConfig | None = None) -> None:
        self._config = config or RerankConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> RerankConfig:
        """Return the reranker's configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Reranking API
    # ------------------------------------------------------------------

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: Sequence[tuple[str, float]],
    ) -> RerankResponse:
        """Rerank a list of retrieval results.

        Args:
            query: The original search query.
            results: A sequence of ``(chunk_id, score)`` pairs from
                the retrieval stage, ordered by descending score.

        Returns:
            A ``RerankResponse`` with reranked results.
        """
        ...
