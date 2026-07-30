"""Pipeline domain models.

Every model in this module is immutable.  They represent the data types
for the knowledge pipeline layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PipelineStats:
    """Statistics for a knowledge pipeline.

    Attributes:
        documents: Total number of documents ingested.
        chunks: Total number of chunks across all documents.
        vectors: Total number of embedding vectors stored.
        searches: Total number of searches performed.
    """

    documents: int = 0
    chunks: int = 0
    vectors: int = 0
    searches: int = 0


@dataclass(frozen=True)
class PipelineResult:
    """The result of a pipeline search operation.

    Attributes:
        context: The knowledge context produced by the search,
            containing formatted text and source provenance.
        metadata: Optional metadata about the search execution
            (timing, scores, etc.).
    """

    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
