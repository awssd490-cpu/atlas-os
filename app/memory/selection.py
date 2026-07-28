"""Memory Selection Engine — decides which memories survive for context.

The Context Builder orchestrates.  The Selection Engine decides.

Given a collection of retrieved memories, the engine produces the
optimal subset for the available context budget using configurable
scoring factors and policies.

Every component is independently testable and replaceable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.memory.memory import Memory, MemoryState


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectionScore:
    """A scored memory with full breakdown.

    The ``composite`` field is the final weighted score used for ranking.
    Individual factor scores enable debugging and explainability.
    """

    memory_id: str = ""
    composite: float = 0.0
    importance_score: float = 0.0
    recency_score: float = 0.0
    frequency_score: float = 0.0
    type_score: float = 0.0
    namespace_score: float = 0.0
    pinned_bonus: float = 0.0
    archived_penalty: float = 0.0


@dataclass(frozen=True)
class SelectionReason:
    """Why a memory was accepted or rejected."""

    memory_id: str = ""
    accepted: bool = True
    reason: str = ""


@dataclass(frozen=True)
class SelectionStatistics:
    """Aggregated statistics for a selection run."""

    total_input: int = 0
    total_selected: int = 0
    total_rejected: int = 0
    total_pinned: int = 0
    total_required: int = 0
    policy_enforcement_count: int = 0
    lowest_selected_score: float = 0.0
    highest_rejected_score: float = 0.0


@dataclass(frozen=True)
class SelectionResult:
    """The output of a selection run.

    Contains everything needed for debugging and explainability.
    """

    selected: list[Memory] = field(default_factory=list)
    selected_scores: list[SelectionScore] = field(default_factory=list)
    rejected: list[Memory] = field(default_factory=list)
    rejected_scores: list[SelectionScore] = field(default_factory=list)
    rejection_reasons: list[SelectionReason] = field(default_factory=list)
    pinned_memories: list[Memory] = field(default_factory=list)
    statistics: SelectionStatistics = field(default_factory=SelectionStatistics)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_count": len(self.selected),
            "rejected_count": len(self.rejected),
            "statistics": {
                "total_input": self.statistics.total_input,
                "total_selected": self.statistics.total_selected,
                "total_rejected": self.statistics.total_rejected,
            },
            "rejection_reasons": [
                {"memory_id": r.memory_id, "reason": r.reason}
                for r in self.rejection_reasons
            ],
        }


# ---------------------------------------------------------------------------
# SelectionConfig
# ---------------------------------------------------------------------------


@dataclass
class SelectionConfig:
    """Configuration for the Memory Selection Engine.

    Every scoring factor weight is configurable.  Set a weight to 0.0
    to disable that factor.  Policies are independently configurable.
    """

    # Score weights
    importance_weight: float = 1.0
    recency_weight: float = 0.5
    frequency_weight: float = 0.3
    type_weight: float = 0.2
    namespace_weight: float = 0.2
    pinned_bonus: float = 0.5
    archived_penalty: float = -0.5

    # Policies
    minimum_score: float = 0.0
    max_memories: int = 100
    require_pinned: bool = True
    diversity_ratio: float = 0.3  # max fraction from a single namespace/type
    recent_memory_count: int = 3  # guaranteed slots for recent memories
    long_term_guarantee: int = 2  # guaranteed slots for long-term facts

    # Namespace and type priority maps (name → multiplier)
    namespace_priorities: dict[str, float] = field(default_factory=dict)
    type_priorities: dict[str, float] = field(default_factory=dict)

    # Pinned memory IDs — always included
    pinned_memory_ids: set[str] = field(default_factory=set)

    # Required memory IDs — must be included if present
    required_memory_ids: set[str] = field(default_factory=set)

    # Pinned namespaces — all memories from these namespaces get a boost
    pinned_namespaces: set[str] = field(default_factory=set)

    # Bypass penalty for archived memories (True = archived gets penalty applied)
    penalize_archived: bool = True

    # Deterministic tie-breaking by ID when scores are equal
    deterministic_tie_break: bool = True


# ---------------------------------------------------------------------------
# MemorySelectionEngine
# ---------------------------------------------------------------------------


class MemorySelectionEngine:
    """Decides which memories survive for context.

    Uses a multi-factor scoring model and configurable policies to
    produce the optimal subset of memories for the available budget.
    """

    def __init__(self, config: SelectionConfig | None = None) -> None:
        self._config = config or SelectionConfig()

    @property
    def config(self) -> SelectionConfig:
        return self._config

    async def select(
        self,
        memories: list[Memory],
        *,
        max_memories: int | None = None,
    ) -> SelectionResult:
        """Select the optimal subset of *memories*.

        Args:
            memories: The candidate memories to select from.
            max_memories: Override the configured max_memories.

        Returns:
            A ``SelectionResult`` with selected and rejected memories.
        """
        cfg = self._config

        # Score each memory
        scored: list[tuple[Memory, SelectionScore]] = []
        for mem in memories:
            score = await self._score(mem)
            scored.append((mem, score))

        # Sort by composite score descending, with deterministic tie-break
        if cfg.deterministic_tie_break:
            scored.sort(key=lambda pair: (-pair[1].composite, pair[0].id.value))
        else:
            scored.sort(key=lambda pair: -pair[1].composite)

        total_input = len(scored)
        max_mem = max_memories or cfg.max_memories

        # Phase 1: Identify pinned and required memories
        pinned: list[Memory] = []
        pinned_scores: list[SelectionScore] = []
        remaining: list[tuple[Memory, SelectionScore]] = []

        for mem, score in scored:
            mid = mem.id.value
            if mid in cfg.pinned_memory_ids or mem.namespace in cfg.pinned_namespaces:
                pinned.append(mem)
                pinned_scores.append(score)
            else:
                remaining.append((mem, score))

        required: list[Memory] = []
        required_scores: list[SelectionScore] = []
        still_remaining: list[tuple[Memory, SelectionScore]] = []

        for mem, score in remaining:
            mid = mem.id.value
            if mid in cfg.required_memory_ids:
                required.append(mem)
                required_scores.append(score)
            else:
                still_remaining.append((mem, score))
        remaining = still_remaining

        # Phase 2: Filter by minimum score
        rejected: list[Memory] = []
        rejected_scores: list[SelectionScore] = []
        rejection_reasons: list[SelectionReason] = []

        above_threshold: list[tuple[Memory, SelectionScore]] = []
        for mem, score in remaining:
            if score.composite < cfg.minimum_score:
                rejected.append(mem)
                rejected_scores.append(score)
                rejection_reasons.append(
                    SelectionReason(
                        memory_id=mem.id.value,
                        accepted=False,
                        reason=f"score {score.composite:.3f} below minimum {cfg.minimum_score}",
                    )
                )
            else:
                above_threshold.append((mem, score))

        # Phase 3: Apply diversity policy (namespace and type quotas)
        diversity_pool: list[tuple[Memory, SelectionScore]] = []
        namespace_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}

        for mem, score in above_threshold:
            ns = mem.namespace
            mt = mem.memory_type

            ns_count = namespace_counts.get(ns, 0)
            max_ns = max(1, int(max_mem * cfg.diversity_ratio))
            if ns_count >= max_ns:
                rejected.append(mem)
                rejected_scores.append(score)
                rejection_reasons.append(
                    SelectionReason(
                        memory_id=mem.id.value,
                        accepted=False,
                        reason=f"namespace {ns!r} at quota ({ns_count}/{max_ns})",
                    )
                )
                continue

            mt_count = type_counts.get(mt, 0)
            max_mt = max(1, int(max_mem * cfg.diversity_ratio))
            if mt_count >= max_mt:
                rejected.append(mem)
                rejected_scores.append(score)
                rejection_reasons.append(
                    SelectionReason(
                        memory_id=mem.id.value,
                        accepted=False,
                        reason=f"memory type {mt!r} at quota ({mt_count}/{max_mt})",
                    )
                )
                continue

            diversity_pool.append((mem, score))
            namespace_counts[ns] = ns_count + 1
            type_counts[mt] = mt_count + 1

        # Phase 4: Build the selected set
        selected: list[Memory] = []
        selected_scores_list: list[SelectionScore] = []

        # Pinned always go first
        selected.extend(pinned)
        selected_scores_list.extend(pinned_scores)

        # Required go next
        selected.extend(required)
        selected_scores_list.extend(required_scores)

        # Guarantee recent memories
        recent_added = 0
        recent_guarantee = cfg.recent_memory_count
        remaining_for_select: list[tuple[Memory, SelectionScore]] = []
        for mem, score in diversity_pool:
            if recent_added < recent_guarantee and self._is_recent(mem):
                if len(selected) < max_mem:
                    selected.append(mem)
                    selected_scores_list.append(score)
                    recent_added += 1
                    continue
            remaining_for_select.append((mem, score))
        diversity_pool = remaining_for_select

        # Guarantee long-term facts
        lt_added = 0
        lt_guarantee = cfg.long_term_guarantee
        remaining_for_select = []
        for mem, score in diversity_pool:
            if lt_added < lt_guarantee and mem.namespace in ("long_term", "semantic") and mem.importance >= 0.5:
                if len(selected) < max_mem:
                    selected.append(mem)
                    selected_scores_list.append(score)
                    lt_added += 1
                    continue
            remaining_for_select.append((mem, score))
        diversity_pool = remaining_for_select

        # Fill remaining slots by score
        for mem, score in diversity_pool:
            if len(selected) >= max_mem:
                rejected.append(mem)
                rejected_scores.append(score)
                rejection_reasons.append(
                    SelectionReason(
                        memory_id=mem.id.value,
                        accepted=False,
                        reason=f"max memories ({max_mem}) reached",
                    )
                )
                continue
            selected.append(mem)
            selected_scores_list.append(score)

        # Build statistics
        lowest_selected = min((s.composite for s in selected_scores_list), default=0.0)
        highest_rejected = max((s.composite for s in rejected_scores), default=0.0)

        stats = SelectionStatistics(
            total_input=total_input,
            total_selected=len(selected),
            total_rejected=len(rejected),
            total_pinned=len(pinned),
            total_required=len(required),
            policy_enforcement_count=len(rejection_reasons),
            lowest_selected_score=lowest_selected,
            highest_rejected_score=highest_rejected,
        )

        return SelectionResult(
            selected=selected,
            selected_scores=selected_scores_list,
            rejected=rejected,
            rejected_scores=rejected_scores,
            rejection_reasons=rejection_reasons,
            pinned_memories=pinned,
            statistics=stats,
            metadata={
                "config_max_memories": max_mem,
                "config_min_score": cfg.minimum_score,
                "config_diversity_ratio": cfg.diversity_ratio,
            },
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    async def _score(self, memory: Memory) -> SelectionScore:
        """Compute a full score breakdown for *memory*."""
        cfg = self._config

        imp_score = memory.importance * cfg.importance_weight

        rec_score = self._recency_multiplier(memory) * cfg.recency_weight

        freq_score = self._frequency_multiplier(memory) * cfg.frequency_weight

        type_mult = cfg.type_priorities.get(memory.memory_type, 1.0)
        type_score = type_mult * cfg.type_weight

        ns_mult = cfg.namespace_priorities.get(memory.namespace, 1.0)
        namespace_score = ns_mult * cfg.namespace_weight

        pinned = 0.0
        if memory.id.value in cfg.pinned_memory_ids or memory.namespace in cfg.pinned_namespaces:
            pinned = cfg.pinned_bonus

        archived = 0.0
        if cfg.penalize_archived and memory.state == MemoryState.ARCHIVED:
            archived = cfg.archived_penalty

        composite = imp_score + rec_score + freq_score + type_score + namespace_score + pinned + archived
        composite = max(0.0, composite)

        return SelectionScore(
            memory_id=memory.id.value,
            composite=composite,
            importance_score=imp_score,
            recency_score=rec_score,
            frequency_score=freq_score,
            type_score=type_score,
            namespace_score=namespace_score,
            pinned_bonus=pinned,
            archived_penalty=archived,
        )

    @staticmethod
    def _recency_multiplier(memory: Memory) -> float:
        """Score recency — returns multiplier in [0.0, 2.0]."""
        if memory.accessed_at is None:
            return 0.5
        age_hours = (_now_utc() - memory.accessed_at).total_seconds() / 3600.0
        if age_hours < 1:
            return 2.0
        if age_hours < 24:
            return 1.5
        if age_hours < 168:
            return 1.0
        if age_hours < 720:
            return 0.5
        return 0.0

    @staticmethod
    def _frequency_multiplier(memory: Memory) -> float:
        """Score access frequency — returns multiplier in [0.0, 2.0]."""
        if memory.access_count == 0:
            return 0.0
        if memory.access_count > 100:
            return 2.0
        if memory.access_count > 50:
            return 1.5
        if memory.access_count > 10:
            return 1.0
        if memory.access_count > 3:
            return 0.5
        return 0.2

    @staticmethod
    def _is_recent(memory: Memory) -> bool:
        """Return True if the memory was accessed in the last 24 hours."""
        if memory.accessed_at is None:
            return False
        age_hours = (_now_utc() - memory.accessed_at).total_seconds() / 3600.0
        return age_hours < 24


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)
