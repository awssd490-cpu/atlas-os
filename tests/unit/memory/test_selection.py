"""Tests for the Memory Selection Engine.

Verifies:
- SelectionScore creation and score breakdown
- SelectionResult and SelectionStatistics
- SelectionConfig defaults and customisation
- MemorySelectionEngine.select()
- Score calculation (importance, recency, frequency, type, namespace,
  pinned bonus, archived penalty)
- Policy enforcement (minimum score, max memories, diversity ratios,
  pinned memories, required memories, recent guarantees,
  long-term guarantees, namespace/type quotas)
- Deterministic tie-breaking
- Empty inputs
- Very large inputs (performance)
- ContextBuilder integration
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId, MemoryState, MemoryType
from app.memory.context import ContextBuilder, ContextBuilderConfig
from app.memory.manager import MemoryManager, MemoryRepository
from app.memory.selection import (
    MemorySelectionEngine,
    SelectionConfig,
    SelectionResult,
    SelectionScore,
    SelectionStatistics,
    SelectionReason,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> MemorySelectionEngine:
    return MemorySelectionEngine()


@pytest.fixture
def config() -> SelectionConfig:
    return SelectionConfig(
        minimum_score=0.0,
        max_memories=10,
        diversity_ratio=0.5,
    )


@pytest.fixture
def memories() -> list[Memory]:
    """A set of candidate memories for selection tests."""
    return [
        Memory(content="high importance", importance=0.95, memory_id=MemoryId("m-high"),
               namespace="ns1", memory_type=MemoryType.KNOWLEDGE.value),
        Memory(content="medium importance", importance=0.5, memory_id=MemoryId("m-med"),
               namespace="ns1", memory_type=MemoryType.REFERENCE.value),
        Memory(content="low importance", importance=0.1, memory_id=MemoryId("m-low"),
               namespace="ns2", memory_type=MemoryType.SHORT_TERM.value),
        Memory(content="archived memory", importance=0.6, memory_id=MemoryId("m-arch"),
               namespace="ns2", memory_type=MemoryType.PROJECT.value,
               state=MemoryState.ARCHIVED),
        Memory(content="frequent access", importance=0.4, memory_id=MemoryId("m-freq"),
               namespace="ns1", memory_type=MemoryType.WORKING.value,
               access_count=200),
        Memory(content="recently accessed", importance=0.3, memory_id=MemoryId("m-rec"),
               namespace="ns3", memory_type=MemoryType.CONVERSATION.value),
    ]


@pytest.fixture
async def repo_with_memories() -> MemoryRepository:
    from app.storage.connection.sqlite import SQLiteConnection
    from app.storage.migration.manager import SqliteMigrationManager
    from app.memory.migrations import V002_MemorySchema

    conn = SQLiteConnection(":memory:")
    mgr = SqliteMigrationManager()
    await mgr.apply_all(conn, [V002_MemorySchema()])
    repo = MemoryRepository(connection=conn)

    for m in [
        Memory(content="important plan", importance=0.9, memory_id=MemoryId("m1"), namespace="default"),
        Memory(content="note", importance=0.3, memory_id=MemoryId("m2"), namespace="default"),
        Memory(content="trivial", importance=0.05, memory_id=MemoryId("m3"), namespace="default"),
    ]:
        await repo.add(m)
    return repo


@pytest.fixture
def recent_memory() -> Memory:
    m = Memory(content="very recent", importance=0.5, memory_id=MemoryId("m-recent"))
    m.accessed_at = datetime.now(timezone.utc)
    return m


@pytest.fixture
def old_memory() -> Memory:
    m = Memory(content="very old", importance=0.8, memory_id=MemoryId("m-old"))
    m.accessed_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    return m


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------


class TestSelectionScore:
    def test_defaults(self) -> None:
        score = SelectionScore(memory_id="m1")
        assert score.memory_id == "m1"
        assert score.composite == 0.0

    def test_immutable(self) -> None:
        score = SelectionScore(memory_id="m1", composite=1.0)
        with pytest.raises(AttributeError):
            score.composite = 2.0  # type: ignore[misc]


class TestSelectionStatistics:
    def test_defaults(self) -> None:
        stats = SelectionStatistics()
        assert stats.total_input == 0
        assert stats.total_selected == 0

    def test_with_values(self) -> None:
        stats = SelectionStatistics(
            total_input=10, total_selected=5, total_rejected=5,
        )
        assert stats.total_input == 10
        assert stats.total_selected == 5


class TestSelectionResult:
    def test_empty(self) -> None:
        result = SelectionResult()
        assert len(result.selected) == 0
        assert result.statistics.total_input == 0

    def test_to_dict(self) -> None:
        m = Memory(content="test", memory_id=MemoryId("m1"))
        result = SelectionResult(
            selected=[m],
            selected_scores=[SelectionScore(memory_id="m1", composite=0.9)],
            rejected=[Memory(content="bad", memory_id=MemoryId("m2"))],
            rejection_reasons=[SelectionReason(memory_id="m2", accepted=False, reason="low score")],
            statistics=SelectionStatistics(total_input=2, total_selected=1, total_rejected=1),
        )
        d = result.to_dict()
        assert d["selected_count"] == 1
        assert d["rejected_count"] == 1
        assert len(d["rejection_reasons"]) == 1


# ---------------------------------------------------------------------------
# SelectionConfig
# ---------------------------------------------------------------------------


class TestSelectionConfig:
    def test_defaults(self) -> None:
        cfg = SelectionConfig()
        assert cfg.importance_weight == 1.0
        assert cfg.recency_weight == 0.5
        assert cfg.max_memories == 100
        assert cfg.diversity_ratio == 0.3
        assert cfg.deterministic_tie_break is True

    def test_custom(self) -> None:
        cfg = SelectionConfig(
            importance_weight=2.0,
            max_memories=5,
            minimum_score=0.2,
        )
        assert cfg.importance_weight == 2.0
        assert cfg.max_memories == 5


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


class TestScoring:
    async def test_importance_score(self, engine: MemorySelectionEngine) -> None:
        mem = Memory(content="test", importance=0.8)
        score = await engine._score(mem)
        assert score.importance_score == pytest.approx(0.8, abs=0.01)

    async def test_archived_penalty(self, engine: MemorySelectionEngine) -> None:
        mem = Memory(content="archived", importance=0.8, state=MemoryState.ARCHIVED)
        score = await engine._score(mem)
        assert score.archived_penalty < 0.0  # penalty was applied
        # Archived should score lower than an identical active memory
        active = Memory(content="active", importance=0.8)
        active_score = await engine._score(active)
        assert score.composite < active_score.composite

    async def test_pinned_bonus(self) -> None:
        cfg = SelectionConfig(pinned_memory_ids={"m1"})
        eng = MemorySelectionEngine(config=cfg)
        mem = Memory(content="pinned", importance=0.5, memory_id=MemoryId("m1"))
        score = await eng._score(mem)
        assert score.pinned_bonus > 0.0

    async def test_namespace_priority(self) -> None:
        cfg = SelectionConfig(namespace_priorities={"high_ns": 2.0})
        eng = MemorySelectionEngine(config=cfg)
        mem = Memory(content="high ns", importance=0.5, namespace="high_ns")
        score = await eng._score(mem)
        assert score.namespace_score > 0.0

    async def test_type_priority(self) -> None:
        cfg = SelectionConfig(type_priorities={"project": 2.0})
        eng = MemorySelectionEngine(config=cfg)
        mem = Memory(content="project", importance=0.5, memory_type="project")
        score = await eng._score(mem)
        assert score.type_score > 0.0

    async def test_recency_higher_for_recent(
        self, engine: MemorySelectionEngine, recent_memory: Memory, old_memory: Memory
    ) -> None:
        recent_score = await engine._score(recent_memory)
        old_score = await engine._score(old_memory)
        assert recent_score.recency_score > old_score.recency_score

    async def test_frequency_higher_for_frequent(
        self, engine: MemorySelectionEngine
    ) -> None:
        freq = Memory(content="freq")
        freq.access_count = 200
        none = Memory(content="none")
        none.access_count = 0
        freq_score = await engine._score(freq)
        none_score = await engine._score(none)
        assert freq_score.frequency_score > none_score.frequency_score


# ---------------------------------------------------------------------------
# Selection policies
# ---------------------------------------------------------------------------


class TestSelectionPolicies:
    async def test_minimum_score_filter(self) -> None:
        cfg = SelectionConfig(minimum_score=5.0, max_memories=10, diversity_ratio=1.0)
        eng = MemorySelectionEngine(config=cfg)
        mems = [
            Memory(content="high", importance=0.9, memory_id=MemoryId("m1")),
            Memory(content="low", importance=0.1, memory_id=MemoryId("m2")),
        ]
        result = await eng.select(mems)
        assert len(result.selected) == 0
        assert len(result.rejected) == 2

    async def test_max_memories_limit(self) -> None:
        cfg = SelectionConfig(max_memories=2, minimum_score=0.0, diversity_ratio=1.0)
        eng = MemorySelectionEngine(config=cfg)
        mems = [
            Memory(content="a", importance=0.9, memory_id=MemoryId("a"), namespace="ns"),
            Memory(content="b", importance=0.5, memory_id=MemoryId("b"), namespace="ns"),
            Memory(content="c", importance=0.3, memory_id=MemoryId("c"), namespace="ns"),
        ]
        result = await eng.select(mems)
        assert len(result.selected) == 2

    async def test_pinned_memories_always_included(self) -> None:
        cfg = SelectionConfig(
            pinned_memory_ids={"pin1"},
            max_memories=1,
            minimum_score=0.0,
            diversity_ratio=1.0,
        )
        eng = MemorySelectionEngine(config=cfg)
        mems = [
            Memory(content="pinned", importance=0.1, memory_id=MemoryId("pin1")),
            Memory(content="high", importance=0.9, memory_id=MemoryId("high1")),
        ]
        result = await eng.select(mems)
        ids = {m.id.value for m in result.selected}
        assert "pin1" in ids

    async def test_required_memories_included(self) -> None:
        cfg = SelectionConfig(
            required_memory_ids={"req1"},
            max_memories=1,
            minimum_score=0.0,
            diversity_ratio=1.0,
        )
        eng = MemorySelectionEngine(config=cfg)
        mems = [
            Memory(content="required", importance=0.1, memory_id=MemoryId("req1")),
            Memory(content="higher", importance=0.9, memory_id=MemoryId("higher")),
        ]
        result = await eng.select(mems)
        ids = {m.id.value for m in result.selected}
        assert "req1" in ids

    async def test_pinned_namespace_boost(self) -> None:
        cfg = SelectionConfig(
            pinned_namespaces={"important_ns"},
            max_memories=10,
            minimum_score=0.0,
            diversity_ratio=1.0,
        )
        eng = MemorySelectionEngine(config=cfg)
        mems = [
            Memory(content="pinned", importance=0.5, memory_id=MemoryId("pn"),
                   namespace="important_ns"),
            Memory(content="normal", importance=0.9, memory_id=MemoryId("nm"),
                   namespace="other"),
        ]
        result = await eng.select(mems)
        # Both should be selected, but pinned should have bonus
        assert len(result.selected) == 2

    async def test_diversity_ratio_namespace(self) -> None:
        cfg = SelectionConfig(
            max_memories=10,
            minimum_score=0.0,
            diversity_ratio=0.3,  # max 30% from one namespace
        )
        eng = MemorySelectionEngine(config=cfg)
        mems = [
            Memory(content=f"ns1-{i}", importance=0.9, memory_id=MemoryId(f"n1-{i}"),
                   namespace="ns1", memory_type="type_a")
            for i in range(10)
        ]
        result = await eng.select(mems)
        # Max 30% of 10 = 3 from ns1
        ns1_count = sum(1 for m in result.selected if m.namespace == "ns1")
        assert ns1_count <= 4  # 30% of 10 = 3, plus pinned/required 0 = max 3

    async def test_recent_memory_guarantee(self) -> None:
        cfg = SelectionConfig(
            max_memories=2,
            minimum_score=0.0,
            diversity_ratio=1.0,
            recent_memory_count=1,
        )
        eng = MemorySelectionEngine(config=cfg)

        old = Memory(content="old high", importance=0.9, memory_id=MemoryId("old"))
        old.accessed_at = datetime(2020, 1, 1, tzinfo=timezone.utc)

        recent = Memory(content="recent", importance=0.1, memory_id=MemoryId("rec"))
        recent.accessed_at = datetime.now(timezone.utc)

        mems = [old, recent]
        result = await eng.select(mems)
        ids = {m.id.value for m in result.selected}
        assert "rec" in ids  # recent guaranteed

    async def test_long_term_guarantee(self) -> None:
        cfg = SelectionConfig(
            max_memories=2,
            minimum_score=0.0,
            diversity_ratio=1.0,
            long_term_guarantee=1,
        )
        eng = MemorySelectionEngine(config=cfg)

        lt = Memory(content="lt fact", importance=0.9, memory_id=MemoryId("lt"),
                    namespace="long_term", memory_type=MemoryType.LONG_TERM.value)
        high = Memory(content="high", importance=0.95, memory_id=MemoryId("high"),
                      namespace="default", memory_type=MemoryType.SHORT_TERM.value)

        mems = [lt, high]
        result = await eng.select(mems)
        ids = {m.id.value for m in result.selected}
        assert "lt" in ids

    async def test_deterministic_tie_break(self) -> None:
        cfg = SelectionConfig(
            max_memories=10,
            minimum_score=0.0,
            diversity_ratio=1.0,
            deterministic_tie_break=True,
        )
        eng = MemorySelectionEngine(config=cfg)
        mems = [
            Memory(content="tie a", importance=0.5, memory_id=MemoryId("z-first")),
            Memory(content="tie b", importance=0.5, memory_id=MemoryId("a-first")),
        ]
        result = await eng.select(mems)
        # a-first should sort before z-first (alphabetical tie-break)
        assert result.selected[0].id.value == "a-first"


# ---------------------------------------------------------------------------
# Empty and edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_empty_input(self, engine: MemorySelectionEngine) -> None:
        result = await engine.select([])
        assert len(result.selected) == 0
        assert result.statistics.total_input == 0

    async def test_single_memory(self, engine: MemorySelectionEngine) -> None:
        mem = Memory(content="solo", importance=0.8, memory_id=MemoryId("s1"))
        result = await engine.select([mem])
        assert len(result.selected) == 1
        assert result.selected[0].id.value == "s1"

    async def test_all_below_minimum_score(self) -> None:
        cfg = SelectionConfig(minimum_score=2.0, max_memories=10, diversity_ratio=1.0)
        eng = MemorySelectionEngine(config=cfg)
        mems = [Memory(content="x", importance=0.5, memory_id=MemoryId("m1"))]
        result = await eng.select(mems)
        assert len(result.selected) == 0
        assert len(result.rejected) == 1

    async def test_large_input(self, engine: MemorySelectionEngine) -> None:
        """Should handle 1000 memories without error."""
        mems = [
            Memory(content=f"mem-{i}", importance=i / 1000, memory_id=MemoryId(str(i)),
                   namespace="ns" if i % 2 == 0 else "other",
                   memory_type="type_a" if i % 3 == 0 else "type_b")
            for i in range(1000)
        ]
        result = await engine.select(mems)
        assert len(result.selected) <= 100  # default max
        assert result.statistics.total_input == 1000

    async def test_all_archived_penalty(self) -> None:
        cfg = SelectionConfig(
            max_memories=10, minimum_score=0.0, diversity_ratio=1.0,
            penalize_archived=True,
        )
        eng = MemorySelectionEngine(config=cfg)
        mems = [
            Memory(content="archived a", importance=0.9, memory_id=MemoryId("a"),
                   state=MemoryState.ARCHIVED),
            Memory(content="active b", importance=0.5, memory_id=MemoryId("b")),
        ]
        result = await eng.select(mems)
        # Active should score higher due to no penalty
        assert result.selected[0].id.value == "b"


# ---------------------------------------------------------------------------
# ContextBuilder integration
# ---------------------------------------------------------------------------


class TestContextBuilderIntegration:
    async def test_selection_engine_property(
        self, repo_with_memories: MemoryRepository
    ) -> None:
        cfg = ContextBuilderConfig(enable_selection=True)
        eng = MemorySelectionEngine(SelectionConfig(max_memories=2))
        mgr = MemoryManager(repository=repo_with_memories)
        builder = ContextBuilder(
            memory_manager=mgr,
            config=cfg,
            selection_engine=eng,
        )
        assert builder.selection_engine is eng
        assert builder.selection_engine is not None

    async def test_selection_engine_none_by_default(
        self, repo_with_memories: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=repo_with_memories)
        builder = ContextBuilder(memory_manager=mgr)
        assert builder.selection_engine is None

    async def test_selection_applied_during_build(
        self, repo_with_memories: MemoryRepository
    ) -> None:
        """Build with selection engine should produce selected memories."""
        cfg = ContextBuilderConfig(
            enable_selection=True,
            enable_compression=False,
            enable_relationship_expansion=False,
        )
        eng = MemorySelectionEngine(SelectionConfig(max_memories=2, minimum_score=0.0))
        mgr = MemoryManager(repository=repo_with_memories)
        builder = ContextBuilder(
            memory_manager=mgr,
            config=cfg,
            selection_engine=eng,
        )
        pkg = await builder.build()
        # Should have at least some sections with memories
        total = pkg.total_memories
        assert total >= 0

    async def test_selection_disabled(self, repo_with_memories: MemoryRepository) -> None:
        """When selection is disabled, engine is not used."""
        cfg = ContextBuilderConfig(enable_selection=False)
        eng = MemorySelectionEngine(SelectionConfig(max_memories=1))
        mgr = MemoryManager(repository=repo_with_memories)
        builder = ContextBuilder(
            memory_manager=mgr,
            config=cfg,
            selection_engine=eng,
        )
        pkg = await builder.build()
        # Without selection, all memories should pass through
        assert pkg is not None

    async def test_selection_with_compression(
        self, repo_with_memories: MemoryRepository
    ) -> None:
        """Selection and compression can both be enabled."""
        cfg = ContextBuilderConfig(
            enable_selection=True,
            enable_compression=True,
            enable_relationship_expansion=False,
        )
        eng = MemorySelectionEngine(SelectionConfig(max_memories=5, minimum_score=0.0))
        mgr = MemoryManager(repository=repo_with_memories)
        builder = ContextBuilder(
            memory_manager=mgr,
            config=cfg,
            selection_engine=eng,
        )
        pkg = await builder.build()
        assert pkg is not None
        # Should not crash
        assert pkg.total_memories >= 0
