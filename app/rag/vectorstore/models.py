"""Vector store domain models.

Every model in this module is immutable.  They represent the canonical
data types for the vector store layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    """A single search result from a vector store query.

    Attributes:
        chunk_id: The identifier of the chunk this result corresponds to.
        score: The similarity score.  Higher values indicate closer
            matches.  The interpretation depends on the metric used
            during the search.
        vector: The vector associated with the chunk (may be empty if
            not requested).
    """

    chunk_id: str = ""
    score: float = 0.0
    vector: tuple[float, ...] = ()
