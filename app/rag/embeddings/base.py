"""Base abstractions for the embedding layer.

Defines the ``EmbeddingProvider`` abstract base class that all embedding
provider implementations must subclass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.rag.embeddings.config import EmbeddingConfig
from app.rag.embeddings.models import EmbeddingResult


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Every concrete provider (OpenAI, Ollama, etc.) must subclass this
    and implement ``embed()`` and ``embed_batch()``.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        self._config = config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def config(self) -> EmbeddingConfig:
        """Return the provider's configuration."""
        return self._config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a human-readable provider name."""

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------

    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            An ``EmbeddingResult`` containing a single embedding vector.

        Raises:
            EmbeddingProviderError: On provider failure.
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a batch of texts.

        Args:
            texts: The texts to embed.

        Returns:
            An ``EmbeddingResult`` with one vector per input text.

        Raises:
            EmbeddingProviderError: On provider failure.
        """
        ...
