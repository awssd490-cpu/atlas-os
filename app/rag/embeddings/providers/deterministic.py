"""Deterministic embedding provider for testing.

Produces deterministic, reproducible embeddings using SHA-256 hashing
with no randomness, network, or external ML dependencies.
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
from collections.abc import Sequence

from app.rag.embeddings.base import EmbeddingProvider
from app.rag.embeddings.config import EmbeddingConfig
from app.rag.embeddings.errors import EmbeddingProviderError
from app.rag.embeddings.models import EmbeddingResult, EmbeddingVector


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Embedding provider that produces deterministic vectors from text.

    Each input text is hashed with SHA-256, and the digest is used to
    generate ``dimensions`` floating-point values.  The same text always
    produces the same vector; different texts produce different vectors.

    When ``normalize_embeddings`` is ``True`` the vector is L2-normalised.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.config.provider_name or "deterministic"

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------

    async def embed(self, text: str) -> EmbeddingResult:
        """Embed a single text string."""
        start = time.monotonic()

        try:
            vector = self._generate(text)
        except Exception as exc:
            raise EmbeddingProviderError(
                f"Deterministic embedding failed: {exc}",
                details={"text_length": len(text)},
            ) from exc

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
        start = time.monotonic()
        vectors: list[EmbeddingVector] = []

        cfg = self.config
        batch_size = cfg.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            for text in batch:
                try:
                    vector = self._generate(text)
                except Exception as exc:
                    raise EmbeddingProviderError(
                        f"Deterministic batch embedding failed: {exc}",
                        details={"batch_index": i, "text_length": len(text)},
                    ) from exc

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
            config=cfg,
            total_texts=len(vectors),
            elapsed_ms=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Internal — deterministic vector generation
    # ------------------------------------------------------------------

    def _generate(self, text: str) -> tuple[float, ...]:
        """Generate a deterministic ``dimensions``-dimensional vector.

        Each dimension is derived from SHA-256(text + ``":"`` + str(dim)),
        producing a float in ``[-1, 1]``.  The result is optionally
        L2-normalised based on config.
        """
        dims = self.config.dimensions
        raw: list[float] = []

        for dim in range(dims):
            seed_input = f"{text}:{dim}"
            digest = hashlib.sha256(seed_input.encode("utf-8")).digest()
            # Use first 8 bytes as a uint64, map to [-1, 1]
            (int_val,) = struct.unpack_from(">Q", digest, 0)
            # Normalize to [0, 1], then map to [-1, 1]
            normalised = (int_val / (2**64 - 1)) * 2.0 - 1.0
            raw.append(normalised)

        if self.config.normalize_embeddings:
            return self._l2_normalize(tuple(raw))
        return tuple(raw)

    @staticmethod
    def _l2_normalize(vec: tuple[float, ...]) -> tuple[float, ...]:
        """L2-normalise the vector in place."""
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return tuple(v / norm for v in vec)
