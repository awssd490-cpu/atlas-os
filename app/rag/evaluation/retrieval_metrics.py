"""RetrievalMetrics — standard information retrieval quality metrics.

All metrics operate on retrieved and relevant ID sets and return
values in ``[0.0, 1.0]``.

Formulas (standard IR definitions):

    precision@k  = |retrieved[:k] ∩ relevant| / k
    recall@k     = |retrieved[:k] ∩ relevant| / |relevant|
    F1@k         = 2 * P@k * R@k / (P@k + R@k)

    MRR          = 1 / rank_of_first_relevant
    MAP          = mean of average_precision across queries
    nDCG@k       = DCG@k / IDCG@k
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


class RetrievalMetrics:
    """Collection of standard information retrieval metrics.

    All methods are static — no state is shared between calls.
    Each method takes *retrieved_ids* (the ordered list of chunk or
    document IDs returned by a retrieval system) and *relevant_ids*
    (the set of ground-truth relevant IDs).

    Usage::

        metrics = RetrievalMetrics()
        p5 = metrics.precision_at_k(["a", "b", "c"], {"a", "d"}, k=2)
        r5 = metrics.recall_at_k(["a", "b", "c"], {"a", "d"}, k=2)
    """

    # ------------------------------------------------------------------
    # Precision
    # ------------------------------------------------------------------

    @staticmethod
    def precision_at_k(
        retrieved_ids: Sequence[Any],
        relevant_ids: set[Any],
        k: int,
    ) -> float:
        """Compute precision at k (P@k).

        ``precision@k = |retrieved[:k] ∩ relevant| / k``

        Args:
            retrieved_ids: Ordered list of retrieved IDs.
            relevant_ids: Set of ground-truth relevant IDs.
            k: The cutoff rank.

        Returns:
            Precision at k in ``[0.0, 1.0]``.  Returns ``0.0`` when
            *k* ≤ 0 or the retrieved list is empty.
        """
        if k <= 0 or not retrieved_ids:
            return 0.0

        relevant_at_k = sum(
            1 for rid in retrieved_ids[:k]
            if rid in relevant_ids
        )
        return min(1.0, relevant_at_k / k)

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    @staticmethod
    def recall_at_k(
        retrieved_ids: Sequence[Any],
        relevant_ids: set[Any],
        k: int,
    ) -> float:
        """Compute recall at k (R@k).

        ``recall@k = |retrieved[:k] ∩ relevant| / |relevant|``

        Args:
            retrieved_ids: Ordered list of retrieved IDs.
            relevant_ids: Set of ground-truth relevant IDs.
            k: The cutoff rank.

        Returns:
            Recall at k in ``[0.0, 1.0]``.  Returns ``1.0`` when the
            relevant set is empty (all relevant items were trivially
            retrieved).  Returns ``0.0`` when *k* ≤ 0 or the retrieved
            list is empty.
        """
        if k <= 0 or not retrieved_ids:
            return 0.0
        if not relevant_ids:
            return 1.0

        relevant_at_k = sum(
            1 for rid in retrieved_ids[:k]
            if rid in relevant_ids
        )
        return min(1.0, relevant_at_k / len(relevant_ids))

    # ------------------------------------------------------------------
    # F1
    # ------------------------------------------------------------------

    @staticmethod
    def f1_at_k(
        retrieved_ids: Sequence[Any],
        relevant_ids: set[Any],
        k: int,
    ) -> float:
        """Compute F1 score at k.

        ``F1@k = 2 * P@k * R@k / (P@k + R@k)``

        When both P@k and R@k are 0.0 the F1 is defined as 0.0.

        Args:
            retrieved_ids: Ordered list of retrieved IDs.
            relevant_ids: Set of ground-truth relevant IDs.
            k: The cutoff rank.

        Returns:
            F1 at k in ``[0.0, 1.0]``.
        """
        p = RetrievalMetrics.precision_at_k(retrieved_ids, relevant_ids, k)
        r = RetrievalMetrics.recall_at_k(retrieved_ids, relevant_ids, k)

        if p == 0.0 and r == 0.0:
            return 0.0

        return 2.0 * p * r / (p + r)

    # ------------------------------------------------------------------
    # Mean Reciprocal Rank
    # ------------------------------------------------------------------

    @staticmethod
    def mean_reciprocal_rank(
        retrieved_ids: Sequence[Any],
        relevant_ids: set[Any],
    ) -> float:
        """Compute Mean Reciprocal Rank (MRR).

        ``MRR = 1 / rank_of_first_relevant``

        The reciprocal rank is 0 if no relevant item is found.

        Args:
            retrieved_ids: Ordered list of retrieved IDs.
            relevant_ids: Set of ground-truth relevant IDs.

        Returns:
            MRR in ``[0.0, 1.0]``.
        """
        if not retrieved_ids or not relevant_ids:
            return 0.0

        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                return 1.0 / rank

        return 0.0

    # ------------------------------------------------------------------
    # Average Precision
    # ------------------------------------------------------------------

    @staticmethod
    def average_precision(
        retrieved_ids: Sequence[Any],
        relevant_ids: set[Any],
    ) -> float:
        """Compute Average Precision (AP).

        ``AP = sum(P@i for i where retrieved[i] is relevant) / |relevant|``

        Args:
            retrieved_ids: Ordered list of retrieved IDs.
            relevant_ids: Set of ground-truth relevant IDs.

        Returns:
            Average precision in ``[0.0, 1.0]``.  Returns ``1.0`` when
            the relevant set is empty (trivial perfect score).
        """
        if not retrieved_ids:
            return 0.0
        if not relevant_ids:
            return 1.0

        score = 0.0
        relevant_count = 0

        for i, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                relevant_count += 1
                score += relevant_count / i

        return min(1.0, score / len(relevant_ids))

    # ------------------------------------------------------------------
    # Normalised Discounted Cumulative Gain
    # ------------------------------------------------------------------

    @staticmethod
    def normalized_dcg(
        retrieved_ids: Sequence[Any],
        relevant_ids: set[Any],
        k: int,
    ) -> float:
        """Compute Normalised Discounted Cumulative Gain at k (nDCG@k).

        Uses binary relevance: 1 if the ID is in *relevant_ids*, 0
        otherwise.

        ``DCG@k = sum((2^rel_i - 1) / log2(i + 1) for i in 1..k)``
        ``IDCG@k = DCG of ideal (perfectly sorted) ranking``
        ``nDCG@k = DCG@k / IDCG@k``

        Args:
            retrieved_ids: Ordered list of retrieved IDs.
            relevant_ids: Set of ground-truth relevant IDs.
            k: The cutoff rank.

        Returns:
            nDCG at k in ``[0.0, 1.0]``.  Returns ``1.0`` when the
            relevant set is empty (trivial perfect score).  Returns
            ``0.0`` when *k* ≤ 0 or the retrieved list is empty.
        """
        if k <= 0 or not retrieved_ids:
            return 0.0
        if not relevant_ids:
            return 1.0

        actual_k = min(k, len(retrieved_ids))
        if actual_k == 0:
            return 0.0

        def dcg_at_k(ids: Sequence[Any], k: int) -> float:
            """Compute DCG@k for a sequence of IDs."""
            dcg = 0.0
            for i, rid in enumerate(ids[:k], start=1):
                rel = 1.0 if rid in relevant_ids else 0.0
                dcg += (2.0**rel - 1.0) / math.log2(i + 1.0)
            return dcg

        # DCG of the actual ranking
        dcg = dcg_at_k(retrieved_ids, actual_k)

        # IDCG: ideal ranking — all relevant IDs first, then non-relevant
        ideal_ids: list[Any] = sorted(
            retrieved_ids[:actual_k],
            key=lambda rid: (rid in relevant_ids, 0),
            reverse=True,
        )
        idcg = dcg_at_k(ideal_ids, actual_k)

        if idcg == 0.0:
            return 0.0

        return min(1.0, dcg / idcg)
