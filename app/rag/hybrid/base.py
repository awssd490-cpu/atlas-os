"""Base abstractions for the hybrid retrieval layer.

Defines the ``HybridRetriever`` abstract base class that all hybrid
retriever implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.rag.hybrid.config import HybridConfig
from app.rag.hybrid.models import HybridResult


class HybridRetriever(ABC):
    """Abstract base class for hybrid retrievers.

    A hybrid retriever combines keyword (lexical) and semantic (vector)
    retrieval into a single ranked result set using a configurable
    fusion strategy.

    Concrete subclasses must implement ``retrieve()``.
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        self._config = config or HybridConfig()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> HybridConfig:
        """Return the retriever's configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Retrieval API
    # ------------------------------------------------------------------

    @abstractmethod
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
        ...
