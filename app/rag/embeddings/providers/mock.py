"""Mock embedding provider for testing and dependency injection.

Provides configurable fixed vectors and failure injection for use in
unit tests.  No actual computation is performed.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.config import EmbeddingConfig
from app.rag.embeddings.errors import EmbeddingProviderError
from app.rag.embeddings.models import EmbeddingResult, EmbeddingVector


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing.

    By default returns a zero vector of ``config.dimensions`` for every
    input.  Callers can provide a ``vector_factory`` for custom vectors
    or use ``fail_on`` to inject failures for specific inputs.

    Usage::

        cfg = EmbeddingConfig(dimensions=4)
        provider = MockEmbeddingProvider(cfg)
        result = await provider.embed("hello")
        assert len(result.embeddings[0].vector) == 4
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        vector_factory: Callable[[str], tuple[float, ...]] | None = None,
        fail_on: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(config)
        self._vector_factory = vector_factory or self._default_vector
        self._fail_on = fail_on or (lambda _: False)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.config.provider_name or "mock"

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        if self._fail_on(text):
            raise EmbeddingProviderError(
                "Mock provider configured to fail",
                details={"text": text[:100]},
            )

        start = time.monotonic()
        vector = self._vector_factory(text)
        elapsed = (time.monotonic() - start) * 1000

        vec = EmbeddingVector(
            vector=vector,
            dimensions=len(vector),
            provider=self.name,
            created_at=time.time(),
            metadata={"text_length": len(text)},
        )

        return EmbeddingResult(
            embeddings=(vec,),
            provider=self.name,
            config=self.config,
            total_texts=1,
            elapsed_ms=round(elapsed, 2),
        )

    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed a batch of texts, preserving order."""
        vectors: list[EmbeddingVector] = []
        start = time.monotonic()

        for text in texts:
            if self._fail_on(text):
                raise EmbeddingProviderError(
                    "Mock provider configured to fail on batch input",
                    details={"text": text[:100]},
                )

            vector = self._vector_factory(text)

            vectors.append(
                EmbeddingVector(
                    vector=vector,
                    dimensions=len(vector),
                    provider=self.name,
                    created_at=time.time(),
                    metadata={"text_length": len(text)},
                )
            )

        elapsed = (time.monotonic() - start) * 1000

        return EmbeddingResult(
            embeddings=tuple(vectors),
            provider=self.name,
            config=self.config,
            total_texts=len(vectors),
            elapsed_ms=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Default factory
    # ------------------------------------------------------------------

    def _default_vector(self, text: str) -> tuple[float, ...]:
        """Return a zero vector of the configured dimensionality."""
        return tuple(0.0 for _ in range(self.config.dimensions))
