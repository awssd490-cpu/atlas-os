"""Memory policies: retention, importance scoring, type configuration.

Policies drive every memory lifecycle decision.  No hardcoded behavior.
"""

from __future__ import annotations

from typing import Any

from app.memory.memory import Memory, MemoryState, MemoryType


# ---------------------------------------------------------------------------
# Memory type policy
# ---------------------------------------------------------------------------


class MemoryTypePolicy:
    """Configuration for a single memory type.

    Each type gets a distinct policy profile.  Adding a new memory type
    means adding a policy entry, not writing a subclass.
    """

    def __init__(
        self,
        *,
        ttl: float | None = 86400.0,
        max_count: int = 100,
        default_importance: float = 0.5,
        decay_rate: float = 0.1,
        compress_by_default: bool = False,
        auto_archive: bool = True,
    ) -> None:
        self.ttl = ttl
        self.max_count = max_count
        self.default_importance = default_importance
        self.decay_rate = decay_rate
        self.compress_by_default = compress_by_default
        self.auto_archive = auto_archive


# Default type policies
DEFAULT_TYPE_POLICIES: dict[str, MemoryTypePolicy] = {
    MemoryType.SHORT_TERM.value: MemoryTypePolicy(
        ttl=86400.0, max_count=100, default_importance=0.4, decay_rate=0.2,
    ),
    MemoryType.WORKING.value: MemoryTypePolicy(
        ttl=300.0, max_count=20, default_importance=0.7, decay_rate=0.5,
    ),
    MemoryType.LONG_TERM.value: MemoryTypePolicy(
        ttl=None, max_count=10000, default_importance=0.6, decay_rate=0.01,
        compress_by_default=True,
    ),
    MemoryType.SEMANTIC.value: MemoryTypePolicy(
        ttl=None, max_count=5000, default_importance=0.7, decay_rate=0.02,
        compress_by_default=True,
    ),
    MemoryType.EPISODIC.value: MemoryTypePolicy(
        ttl=86400.0 * 7, max_count=500, default_importance=0.5, decay_rate=0.1,
        compress_by_default=True,
    ),
    MemoryType.PROCEDURAL.value: MemoryTypePolicy(
        ttl=None, max_count=1000, default_importance=0.8, decay_rate=0.005,
    ),
    MemoryType.CONVERSATION.value: MemoryTypePolicy(
        ttl=86400.0, max_count=200, default_importance=0.3, decay_rate=0.3,
    ),
    MemoryType.PROJECT.value: MemoryTypePolicy(
        ttl=None, max_count=5000, default_importance=0.7, decay_rate=0.01,
        compress_by_default=True,
    ),
    MemoryType.KNOWLEDGE.value: MemoryTypePolicy(
        ttl=None, max_count=20000, default_importance=0.6, decay_rate=0.005,
        compress_by_default=True,
    ),
    MemoryType.REFERENCE.value: MemoryTypePolicy(
        ttl=None, max_count=10000, default_importance=0.5, decay_rate=0.01,
        compress_by_default=True,
    ),
}


# ---------------------------------------------------------------------------
# Importance scorer
# ---------------------------------------------------------------------------


class ImportanceScorer:
    """Computes and evolves memory importance scores.

    The base importance is set on creation.  It evolves via:
    - Recency: recently accessed memories score higher
    - Frequency: often-accessed memories score higher
    - Explicit promotion: user or system can boost
    - Decay: unpromoted memories lose importance over time
    """

    @staticmethod
    def compute_base(*, confidence: float = 1.0, source_weight: float = 0.5) -> float:
        """Compute initial importance from creation-time factors."""
        return 0.5 * confidence + 0.3 * source_weight

    @staticmethod
    def apply_recency(memory: Memory, *, now: float | None = None) -> float:
        """Boost importance based on recency of access.

        Returns a multiplier in [1.0, 1.5].
        """
        if memory.accessed_at is None or memory.created_at is None:
            return 1.0
        hours_since_access = memory.age_seconds / 3600.0
        if hours_since_access < 1:
            return 1.5
        if hours_since_access < 24:
            return 1.3
        if hours_since_access < 168:
            return 1.1
        return 1.0

    @staticmethod
    def apply_frequency(memory: Memory) -> float:
        """Boost importance based on access frequency.

        Returns a multiplier in [1.0, 1.3].
        """
        if memory.access_count == 0:
            return 1.0
        if memory.access_count > 50:
            return 1.3
        if memory.access_count > 10:
            return 1.15
        return 1.05

    @classmethod
    def effective_importance(cls, memory: Memory) -> float:
        """Compute the effective importance of a memory considering
        recency, frequency, and base score."""
        base = memory.importance
        recency = cls.apply_recency(memory)
        frequency = cls.apply_frequency(memory)
        return min(1.0, base * recency * frequency)


# ---------------------------------------------------------------------------
# Retention policy evaluator
# ---------------------------------------------------------------------------


class RetentionPolicy:
    """Evaluates whether a memory should be archived, forgotten, or deleted.

    All thresholds are configurable.  This is the single place where
    retention decisions are made.
    """

    def __init__(
        self,
        *,
        archive_threshold: float = 0.2,
        importance_decay_rate: float = 0.1,
        grace_period_seconds: float = 604800.0,
        max_per_namespace: dict[str, int] | None = None,
        enable_auto_archive: bool = True,
    ) -> None:
        self.archive_threshold = archive_threshold
        self.importance_decay_rate = importance_decay_rate
        self.grace_period_seconds = grace_period_seconds
        self.max_per_namespace = max_per_namespace or {}
        self.enable_auto_archive = enable_auto_archive

    def should_archive(self, memory: Memory) -> bool:
        """Return True when a memory should be archived."""
        if not self.enable_auto_archive:
            return False
        if memory.state != MemoryState.ACTIVE:
            return False
        effective = ImportanceScorer.effective_importance(memory)
        return effective < self.archive_threshold

    def should_forget(self, memory: Memory) -> bool:
        """Return True when a memory should be forgotten (beyond archiving)."""
        if memory.state == MemoryState.ARCHIVED:
            effective = ImportanceScorer.effective_importance(memory)
            return effective < (self.archive_threshold * 0.5)
        if memory.state == MemoryState.ACTIVE:
            return memory.is_expired
        return False

    def should_delete(self, memory: Memory) -> bool:
        """Return True when a forgotten memory's grace period has elapsed."""
        if memory.state != MemoryState.FORGOTTEN:
            return False
        if memory.forgotten_at is None:
            return False
        elapsed = (_now_utc() - memory.forgotten_at).total_seconds()
        return elapsed > self.grace_period_seconds

    def exceeded_capacity(self, memory: Memory, current_count: int) -> bool:
        """Return True when the namespace exceeds its capacity."""
        max_count = self.max_per_namespace.get(memory.namespace, 0)
        if max_count <= 0:
            return False
        return current_count > max_count


def _now_utc():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc)
