"""Embedding domain models.

Every model in this module is immutable and provider-independent.
They represent the canonical data types for the embedding layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EmbeddingVector:
    """A single embedding vector produced by a provider.

    Attributes:
        vector: The embedding values as a tuple of floats.
        dimensions: Dimensionality of the vector.
        provider: Name of the provider that produced this vector.
        created_at: Unix timestamp when the vector was produced.
        metadata: Optional extra information (e.g. input text length).
    """

    vector: tuple[float, ...] = ()
    dimensions: int = 0
    provider: str = ""
    created_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EmbeddingResult:
    """The result of an embedding call.

    Attributes:
        embeddings: The list of embedding vectors, one per input text.
        provider: Name of the provider that produced these embeddings.
        config: The configuration used.
        total_texts: Number of texts embedded.
        elapsed_ms: Execution time in milliseconds.
    """

    embeddings: tuple[EmbeddingVector, ...] = ()
    provider: str = ""
    config: Any = None
    total_texts: int = 0
    elapsed_ms: float = 0.0
