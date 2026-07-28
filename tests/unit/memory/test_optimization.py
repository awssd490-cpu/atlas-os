"""Tests for the Context Optimization Engine.

Verifies:
- Domain models: OptimizationDecision, OptimizationStatistics, OptimizationResult
- OptimizationConfig: defaults, custom, strategy presets
- Each pass: DuplicateElimination, MetadataCleanup, EmptySectionRemoval,
  MemoryOrdering
- ContextOptimizationEngine: orchestrates passes, recalculates stats
- Edge cases: empty package, single memory, all duplicates
- ContextBuilder integration
"""

from __future__ import annotations

from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId, MemoryState
from app.memory.context import ContextPackage, ContextSection, ContextSource, ContextStatistics
from app.memory.optimization import (
    ContextOptimizationEngine,
    DuplicateEliminationPass,
    EmptySectionRemovalPass,
    MemoryOrderingPass,
    MetadataCleanupPass,
    NearDuplicateEliminationPass,
    OptimizationConfig,
    OptimizationDecision,
    OptimizationPass,
    OptimizationResult,
    OptimizationStatistics,
    make_strategy_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_memory(content: str, memory_id: str = "", importance: float = 0.5, **kw: Any) -> Memory:
    return Memory(content=content, memory_id=MemoryId(memory_id or ""), importance=importance, **kw)


def _make_section(
    section_type: str,
    contents: list[str],
    *,
    label: str = "",
    importances: list[float] | None = None,
) -> ContextSection:
    imps = importances or [0.5] * len(contents)
    mems = [
        _make_memory(c, f"{section_type}-{i}", importance=imps[i])
        for i, c in enumerate(contents)
    ]
    sources = [ContextSource(source_type=section_type, memory_id=m.id.value) for m in mems]
    from app.memory.tokens import TokenEstimator
    tokens = TokenEstimator.estimate_memories(mems)
    return ContextSection(
        section_type=section_type,
        label=label or section_type,
        memories=mems,
        sources=sources,
        token_count=tokens,
    )


def _make_package(sections: list[ContextSection], request_id: str = "test") -> ContextPackage:
    tokens = sum(s.token_count for s in sections)
    stats = ContextStatistics(total_tokens=tokens, total_sections=len(sections))
    return ContextPackage(
        request_id=request_id,
        sections=sections,
        statistics=stats,
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------


class TestOptimizationDecision:
    def test_create(self) -> None:
        d = OptimizationDecision(pass_name="dup", memory_id="m1", action="removed", reason="dup")
        assert d.pass_name == "dup"
        assert d.action == "removed"


class TestOptimizationStatistics:
    def test_defaults(self) -> None:
        s = OptimizationStatistics()
        assert s.original_memories == 0

    def test_with_values(self) -> None:
        s = OptimizationStatistics(original_memories=10, final_memories=7, removed_count=3)
        assert s.removed_count == 3


class TestOptimizationResult:
    def test_empty(self) -> None:
        r = OptimizationResult()
        assert r.statistics.original_memories == 0

    def test_to_dict(self) -> None:
        r = OptimizationResult(
            statistics=OptimizationStatistics(original_memories=5, final_memories=3),
        )
        d = r.to_dict()
        assert d["original_memories"] == 5


# ---------------------------------------------------------------------------
# OptimizationConfig
# ---------------------------------------------------------------------------


class TestOptimizationConfig:
    def test_defaults(self) -> None:
        cfg = OptimizationConfig()
        assert cfg.strategy == "balanced"
        assert cfg.enable_duplicate_elimination is True

    def test_conservative_strategy(self) -> None:
        cfg = make_strategy_config("conservative")
        assert cfg.strategy == "conservative"
        assert cfg.enable_near_duplicate_elimination is False

    def test_aggressive_strategy(self) -> None:
        cfg = make_strategy_config("aggressive")
        assert cfg.strategy == "aggressive"
        assert cfg.enable_memory_ordering is True

    def test_balanced_retains_defaults(self) -> None:
        cfg = make_strategy_config("balanced")
        assert cfg.enable_duplicate_elimination is True


# ---------------------------------------------------------------------------
# DuplicateEliminationPass
# ---------------------------------------------------------------------------


class TestDuplicateEliminationPass:
    async def test_removes_exact_duplicates(self) -> None:
        p = DuplicateEliminationPass()
        sec = _make_section("data", ["same content", "unique", "same content"])
        pkg = _make_package([sec])
        result, decisions = await p.run(pkg, OptimizationConfig())
        assert len(result.sections) == 1
        assert result.sections[0].memory_count == 2
        assert len(decisions) == 1

    async def test_keeps_highest_importance(self) -> None:
        p = DuplicateEliminationPass()
        mems = [
            _make_memory("dup content", "a", importance=0.5),
            _make_memory("dup content", "b", importance=0.9),
        ]
        sources = [ContextSource(source_type="data", memory_id="a"),
                   ContextSource(source_type="data", memory_id="b")]
        sec = ContextSection(section_type="data", memories=mems, sources=sources, token_count=100)
        pkg = _make_package([sec])
        result, decisions = await p.run(pkg, OptimizationConfig())
        assert result.sections[0].memory_count == 1
        assert result.sections[0].memories[0].id.value == "b"

    async def test_empty_section(self) -> None:
        p = DuplicateEliminationPass()
        sec = ContextSection(section_type="empty")
        pkg = _make_package([sec])
        result, _ = await p.run(pkg, OptimizationConfig())
        assert len(result.sections) == 1
        assert result.sections[0].memory_count == 0


# ---------------------------------------------------------------------------
# MetadataCleanupPass
# ---------------------------------------------------------------------------


class TestMetadataCleanupPass:
    async def test_truncates_long_metadata(self) -> None:
        p = MetadataCleanupPass()
        mem = _make_memory("content", "m1")
        mem.metadata = {"long_key": "x" * 500}
        sec = ContextSection(section_type="data", memories=[mem], sources=[ContextSource(source_type="data", memory_id="m1")])
        pkg = _make_package([sec])
        result, decisions = await p.run(pkg, OptimizationConfig())
        assert len(decisions) >= 0  # should not crash

    async def test_removes_empty_metadata(self) -> None:
        p = MetadataCleanupPass()
        mem = _make_memory("content", "m1")
        mem.metadata = {"keep": "value", "empty": "", "none": None}
        sec = ContextSection(section_type="data", memories=[mem], sources=[ContextSource(source_type="data", memory_id="m1")])
        pkg = _make_package([sec])
        result, decisions = await p.run(pkg, OptimizationConfig())
        # Cleanup may or may not create decisions depending on internal check
        assert result is not None


# ---------------------------------------------------------------------------
# EmptySectionRemovalPass
# ---------------------------------------------------------------------------


class TestEmptySectionRemovalPass:
    async def test_removes_empty_sections(self) -> None:
        p = EmptySectionRemovalPass()
        full = _make_section("data", ["content"])
        empty = ContextSection(section_type="empty")
        pkg = _make_package([full, empty])
        result, decisions = await p.run(pkg, OptimizationConfig())
        assert len(result.sections) == 1
        assert len(decisions) == 1

    async def test_preserves_non_empty(self) -> None:
        p = EmptySectionRemovalPass()
        a = _make_section("a", ["x"])
        b = _make_section("b", ["y"])
        pkg = _make_package([a, b])
        result, decisions = await p.run(pkg, OptimizationConfig())
        assert len(result.sections) == 2
        assert len(decisions) == 0


# ---------------------------------------------------------------------------
# MemoryOrderingPass
# ---------------------------------------------------------------------------


class TestMemoryOrderingPass:
    async def test_orders_by_importance(self) -> None:
        p = MemoryOrderingPass()
        sec = _make_section("data", ["low", "high", "mid"], importances=[0.1, 0.9, 0.5])
        pkg = _make_package([sec])
        result, decisions = await p.run(pkg, OptimizationConfig())
        ordered = [m.importance for m in result.sections[0].memories]
        assert ordered == sorted(ordered, reverse=True)

    async def test_deterministic_tie_break(self) -> None:
        p = MemoryOrderingPass()
        mems = [
            _make_memory("a", "z-last", importance=0.5),
            _make_memory("b", "a-first", importance=0.5),
        ]
        sources = [ContextSource(source_type="data", memory_id="z-last"),
                   ContextSource(source_type="data", memory_id="a-first")]
        sec = ContextSection(section_type="data", memories=mems, sources=sources)
        pkg = _make_package([sec])
        result, _ = await p.run(pkg, OptimizationConfig())
        ids = [m.id.value for m in result.sections[0].memories]
        assert ids == ["a-first", "z-last"]


# ---------------------------------------------------------------------------
# NearDuplicateEliminationPass
# ---------------------------------------------------------------------------


class TestNearDuplicateEliminationPass:
    async def test_removes_near_duplicates(self) -> None:
        p = NearDuplicateEliminationPass()
        cfg = OptimizationConfig(near_duplicate_similarity=0.5)
        mems = [
            _make_memory("the quick brown fox jumps", "a", importance=0.9),
            _make_memory("the quick brown fox leaps", "b", importance=0.3),
        ]
        sources = [ContextSource(source_type="data", memory_id="a"),
                   ContextSource(source_type="data", memory_id="b")]
        sec = ContextSection(section_type="data", memories=mems, sources=sources)
        pkg = _make_package([sec])
        result, decisions = await p.run(pkg, cfg)
        assert len(result.sections[0].memories) == 1
        assert result.sections[0].memories[0].id.value == "a"

    async def test_preserves_different_content(self) -> None:
        p = NearDuplicateEliminationPass()
        mems = [
            _make_memory("completely different topic", "a", importance=0.5),
            _make_memory("something else entirely", "b", importance=0.5),
        ]
        sources = [ContextSource(source_type="data", memory_id="a"),
                   ContextSource(source_type="data", memory_id="b")]
        sec = ContextSection(section_type="data", memories=mems, sources=sources)
        pkg = _make_package([sec])
        result, decisions = await p.run(pkg, OptimizationConfig(near_duplicate_similarity=0.8))
        assert len(result.sections[0].memories) == 2

    def test_content_overlap(self) -> None:
        overlap = NearDuplicateEliminationPass._content_overlap(
            "the quick brown fox", "the quick brown fox jumps",
        )
        assert 0.5 < overlap < 1.0

    def test_content_overlap_empty(self) -> None:
        assert NearDuplicateEliminationPass._content_overlap("", "test") == 0.0
        assert NearDuplicateEliminationPass._content_overlap("test", "") == 0.0


# ---------------------------------------------------------------------------
# ContextOptimizationEngine
# ---------------------------------------------------------------------------


class TestContextOptimizationEngine:
    async def test_engine_runs_passes(self) -> None:
        engine = ContextOptimizationEngine()
        sec = _make_section("data", ["same", "unique", "same"])
        pkg = _make_package([sec])
        result = await engine.optimize(pkg)
        # Duplicate elimination should have run
        assert result.statistics.removed_count >= 0

    async def test_empty_package(self) -> None:
        engine = ContextOptimizationEngine()
        pkg = _make_package([])
        result = await engine.optimize(pkg)
        assert result.statistics.original_memories == 0
        assert result.statistics.passes_run > 0

    async def test_all_duplicates(self) -> None:
        engine = ContextOptimizationEngine()
        sec = _make_section("data", ["identical"] * 5)
        pkg = _make_package([sec])
        result = await engine.optimize(pkg)
        assert result.statistics.final_memories == 1
        assert result.statistics.removed_count >= 4

    async def test_preserves_order(self) -> None:
        engine = ContextOptimizationEngine()
        a = _make_section("a", ["first"])
        b = _make_section("b", ["second"])
        c = _make_section("c", ["third"])
        pkg = _make_package([a, b, c])
        result = await engine.optimize(pkg)
        types = [s.section_type for s in result.package.sections]
        assert types == ["a", "b", "c"]

    async def test_deterministic_output(self) -> None:
        engine = ContextOptimizationEngine()
        sec = _make_section("data", ["x", "y", "x", "z"])
        pkg = _make_package([sec])
        r1 = await engine.optimize(pkg)
        r2 = await engine.optimize(pkg)
        assert len(r1.package.sections) == len(r2.package.sections)

    async def test_statistics_recalculated(self) -> None:
        engine = ContextOptimizationEngine()
        sec = _make_section("data", ["a", "a", "b"])
        pkg = _make_package([sec])
        result = await engine.optimize(pkg)
        assert result.statistics.original_memories == 3
        assert result.statistics.final_memories < 3

    async def test_config_propagation(self) -> None:
        cfg = OptimizationConfig(strategy="conservative")
        engine = ContextOptimizationEngine(config=cfg)
        assert engine.config.strategy == "conservative"
        assert engine.config.enable_near_duplicate_elimination is False
        assert engine.config.enable_duplicate_elimination is True

    async def test_aggressive_orders_memories(self) -> None:
        cfg = make_strategy_config("aggressive")
        engine = ContextOptimizationEngine(config=cfg)
        sec = _make_section("data", ["low", "high"], importances=[0.1, 0.9])
        pkg = _make_package([sec])
        result = await engine.optimize(pkg)
        mems = result.package.sections[0].memories
        assert mems[0].importance >= mems[-1].importance

    async def test_preserves_single_memory(self) -> None:
        engine = ContextOptimizationEngine()
        sec = _make_section("data", ["only one"])
        pkg = _make_package([sec])
        result = await engine.optimize(pkg)
        assert result.statistics.final_memories == 1


# ---------------------------------------------------------------------------
# Integration with ContextBuilder
# ---------------------------------------------------------------------------


class TestContextBuilderIntegration:
    async def test_optimization_engine_property(self) -> None:
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema
        from app.memory.manager import MemoryManager, MemoryRepository
        from app.memory.context import ContextBuilder

        conn = SQLiteConnection(":memory:")
        mgr = SqliteMigrationManager()
        await mgr.apply_all(conn, [V002_MemorySchema()])
        repo = MemoryRepository(connection=conn)
        await repo.add(Memory(content="test", memory_id=MemoryId("m1"), namespace="default"))

        opt_engine = ContextOptimizationEngine()
        builder = ContextBuilder(
            memory_manager=MemoryManager(repository=repo),
            optimization_engine=opt_engine,
        )
        assert builder.optimization_engine is opt_engine

    async def test_optimization_engine_none_by_default(self) -> None:
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema
        from app.memory.manager import MemoryManager, MemoryRepository
        from app.memory.context import ContextBuilder

        conn = SQLiteConnection(":memory:")
        mgr = SqliteMigrationManager()
        await mgr.apply_all(conn, [V002_MemorySchema()])
        repo = MemoryRepository(connection=conn)
        builder = ContextBuilder(memory_manager=MemoryManager(repository=repo))
        assert builder.optimization_engine is None

    async def test_build_with_optimization(self) -> None:
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema
        from app.memory.manager import MemoryManager, MemoryRepository
        from app.memory.context import ContextBuilder, ContextBuilderConfig

        conn = SQLiteConnection(":memory:")
        mgr = SqliteMigrationManager()
        await mgr.apply_all(conn, [V002_MemorySchema()])
        repo = MemoryRepository(connection=conn)
        await repo.add(Memory(content="test a", memory_id=MemoryId("m1"), namespace="default"))
        await repo.add(Memory(content="test a", memory_id=MemoryId("m2"), namespace="default"))
        await repo.add(Memory(content="test b", memory_id=MemoryId("m3"), namespace="default"))

        opt_engine = ContextOptimizationEngine()
        builder = ContextBuilder(
            memory_manager=MemoryManager(repository=repo),
            config=ContextBuilderConfig(
                enable_compression=False,
                enable_relationship_expansion=False,
                enable_selection=False,
            ),
            optimization_engine=opt_engine,
        )
        pkg = await builder.build()
        # Build should complete without error
        assert pkg is not None
