"""Similarity metrics for vector store search.

Provides the ``SimilarityMetric`` enum and helper functions for computing
similarity scores between vectors.
"""

from __future__ import annotations

import enum
import math


class SimilarityMetric(enum.Enum):
    """Supported similarity metrics for vector search.

    Each metric returns a score where **higher is more similar**.
    """

    COSINE = "cosine"
    DOT_PRODUCT = "dot_product"
    EUCLIDEAN = "euclidean"


def cosine_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value in ``[-1, 1]``, where ``1.0`` means identical
    direction and ``-1.0`` means opposite direction.  A zero vector
    returns ``0.0``.
    """
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for va, vb in zip(a, b):
        dot += va * vb
        norm_a += va * va
        norm_b += vb * vb

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def dot_product_similarity(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute dot-product similarity between two vectors.

    Higher values indicate greater alignment.  No upper bound.
    """
    dot = 0.0
    for va, vb in zip(a, b):
        dot += va * vb
    return dot


def negative_euclidean_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute the negative Euclidean distance between two vectors.

    Returns ``-distance`` so that higher values mean closer vectors.
    A result of ``0.0`` means identical vectors.
    """
    dist_sq = 0.0
    for va, vb in zip(a, b):
        diff = va - vb
        dist_sq += diff * diff
    return -math.sqrt(dist_sq)


# Map metric enum to function
_METRIC_FN = {
    SimilarityMetric.COSINE: cosine_similarity,
    SimilarityMetric.DOT_PRODUCT: dot_product_similarity,
    SimilarityMetric.EUCLIDEAN: negative_euclidean_distance,
}


def compute_similarity(
    a: tuple[float, ...],
    b: tuple[float, ...],
    metric: SimilarityMetric,
) -> float:
    """Compute similarity between two vectors using the given metric."""
    return _METRIC_FN[metric](a, b)
