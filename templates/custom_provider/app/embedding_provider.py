"""
Custom embedding provider example.

Subclasses ``EmbeddingProvider`` and implements a simple
hash-based vector generation.
"""

import math
from collections.abc import Sequence

from app.rag.embeddings import EmbeddingProvider, EmbeddingConfig, register_provider
from app.rag.embeddings.models import EmbeddingResult, EmbeddingVector


class SimpleHashProvider(EmbeddingProvider):
    """A minimal embedding provider for demonstration purposes.

    Generates vectors using character hashing — NOT suitable for
    production use. Replace with an actual ML model provider.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "simple_hash"

    async def embed(self, text: str) -> EmbeddingResult:
        vector = self._make_vector(text)
        vec = EmbeddingVector(
            vector=vector,
            dimensions=len(vector),
            provider=self.name,
        )
        return EmbeddingResult(
            embeddings=(vec,),
            provider=self.name,
            config=self.config,
            total_texts=1,
        )

    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors: list[EmbeddingVector] = []
        for text in texts:
            vector = self._make_vector(text)
            vectors.append(EmbeddingVector(
                vector=vector, dimensions=len(vector), provider=self.name,
            ))
        return EmbeddingResult(
            embeddings=tuple(vectors),
            provider=self.name,
            config=self.config,
            total_texts=len(vectors),
        )

    def _make_vector(self, text: str) -> tuple[float, ...]:
        dims = self.config.dimensions
        raw = []
        for dim in range(dims):
            val = sum(ord(c) * (dim + 1) for c in text) % 1000 / 1000.0
            raw.append(val)
        norm = math.sqrt(sum(v * v for v in raw))
        if norm > 0:
            raw = [v / norm for v in raw]
        return tuple(raw)


# Register for discovery
register_provider("simple_hash", SimpleHashProvider)
