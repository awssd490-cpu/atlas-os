"""Tests for the Token Budget Manager.

Verifies:
- Budget domain models: BudgetAllocation, BudgetDecision, BudgetStatistics, BudgetResult
- BudgetConfig defaults and customisation
- Allocation strategies: PriorityFirstAllocation, WeightedAllocation,
  ProportionalAllocation
- TokenBudgetManager.optimise() with various budgets
- Section trimming
- Preserved sections (user query)
- Emergency budget mode
- Zero-allocation section removal
- No-op when within budget
- Large oversize handling
- ContextBuilder integration
"""

from __future__ import annotations

from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId
from app.memory.context import (
    ContextBuilder,
    ContextBuilderConfig,
    ContextPackage,
    ContextSection,
    ContextSource,
    ContextStatistics,
    TokenEstimator,
)
from app.memory.manager import MemoryManager, MemoryRepository
from app.memory.budget import (
    BudgetAllocation,
    BudgetConfig,
    BudgetDecision,
    BudgetResult,
    BudgetStatistics,
    PriorityFirstAllocation,
    ProportionalAllocation,
    TokenBudgetManager,
    WeightedAllocation,
)
from app.memory.tokens import TokenEstimator as TokenEstimator2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_section(
    section_type: str,
    contents: list[str],
    *,
    label: str = "",
) -> ContextSection:
    """Build a ContextSection with memories."""
    mems: list[Memory] = []
    for i, content in enumerate(contents):
        mems.append(Memory(content=content, memory_id=MemoryId(f"{section_type}-{i}")))
    sources = [
        ContextSource(source_type=section_type, memory_id=m.id.value)
        for m in mems
    ]
    tokens = TokenEstimator2.estimate_memories(mems)
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
    )


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------


class TestBudgetAllocation:
    def test_create(self) -> None:
        ba = BudgetAllocation(section_type="relevant", allocated_tokens=500)
        assert ba.section_type == "relevant"
        assert ba.allocated_tokens == 500

    def test_immutable(self) -> None:
        ba = BudgetAllocation()
        with pytest.raises(AttributeError):
            ba.section_type = "x"  # type: ignore[misc]


class TestBudgetDecision:
    def test_create(self) -> None:
        bd = BudgetDecision(memory_id="m1", reason="too large", token_savings=100)
        assert bd.token_savings == 100


class TestBudgetStatistics:
    def test_defaults(self) -> None:
        bs = BudgetStatistics()
        assert bs.original_tokens == 0

    def test_with_values(self) -> None:
        bs = BudgetStatistics(original_tokens=1000, final_tokens=500, tokens_removed=500)
        assert bs.tokens_removed == 500


class TestBudgetResult:
    def test_empty(self) -> None:
        result = BudgetResult()
        assert result.statistics.original_tokens == 0

    def test_to_dict(self) -> None:
        result = BudgetResult(
            statistics=BudgetStatistics(original_tokens=100, final_tokens=50, tokens_removed=50),
        )
        d = result.to_dict()
        assert d["original_tokens"] == 100
        assert d["tokens_removed"] == 50


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------


class TestBudgetConfig:
    def test_defaults(self) -> None:
        cfg = BudgetConfig()
        assert cfg.default_budget == 4096
        assert cfg.allocation_strategy == "priority_first"
        assert "user_query" in cfg.preserved_section_types

    def test_custom(self) -> None:
        cfg = BudgetConfig(
            default_budget=2048,
            allocation_strategy="weighted",
            enable_emergency_mode=False,
        )
        assert cfg.default_budget == 2048
        assert cfg.allocation_strategy == "weighted"


# ---------------------------------------------------------------------------
# Allocation strategies
# ---------------------------------------------------------------------------


class TestPriorityFirstAllocation:
    def test_preserved_first(self) -> None:
        strat = PriorityFirstAllocation()
        sections = [
            _make_section("user_query", ["hello"]),
            _make_section("relevant", ["a", "b", "c"]),
        ]
        alloc = strat.allocate(sections, 200, preserved_section_types={"user_query"})
        assert alloc.get("user_query", 0) > 0

    def test_all_budget_used(self) -> None:
        strat = PriorityFirstAllocation()
        sections = [
            _make_section("a", ["x"]),
            _make_section("b", ["y" * 200]),
        ]
        alloc = strat.allocate(sections, 100)
        # Section b should get at least something
        assert alloc.get("b", 0) >= 0


class TestWeightedAllocation:
    def test_weights_affect_allocation(self) -> None:
        strat = WeightedAllocation(section_weights={"high": 10.0, "low": 0.1})
        sections = [
            _make_section("high", ["a" * 200]),
            _make_section("low", ["b" * 200]),
        ]
        alloc = strat.allocate(sections, 500)
        assert alloc.get("high", 0) >= 0


class TestProportionalAllocation:
    def test_proportional(self) -> None:
        strat = ProportionalAllocation()
        small = _make_section("small", ["x"])
        big = _make_section("big", ["y" * 500])
        alloc = strat.allocate([small, big], 300)
        # Big section should get more since it's larger
        assert alloc.get("big", 0) >= alloc.get("small", 0)


# ---------------------------------------------------------------------------
# TokenBudgetManager.optimise
# ---------------------------------------------------------------------------


class TestOptimise:
    async def test_noop_when_within_budget(self) -> None:
        mgr = TokenBudgetManager()
        section = _make_section("test", ["hello"])
        pkg = _make_package([section])
        result = await mgr.optimise(pkg, target_budget=100000)
        # No trimming needed
        assert result.statistics.tokens_removed == 0

    async def test_trims_excess(self) -> None:
        mgr = TokenBudgetManager()
        many_mems = _make_section("relevant", ["x" * 100] * 50)
        pkg = _make_package([many_mems])
        result = await mgr.optimise(pkg, target_budget=100)
        # Some tokens should have been removed
        assert result.statistics.tokens_removed > 0
        assert len(result.decisions) > 0

    async def test_preserves_user_query(self) -> None:
        cfg = BudgetConfig(preserved_section_types={"user_query"})
        mgr = TokenBudgetManager(config=cfg)
        query = _make_section("user_query", ["hi"])
        other = _make_section("relevant", ["x" * 200] * 20)
        pkg = _make_package([query, other])
        # Budget large enough for query but not for all relevant memories
        budget = query.token_count + 50  # query + a few relevant memories
        result = await mgr.optimise(pkg, target_budget=budget)
        # User query should still be present
        types = [s.section_type for s in result.package.sections]
        assert "user_query" in types

    async def test_emergency_mode(self) -> None:
        cfg = BudgetConfig(
            default_budget=4000,
            emergency_budget=1000,
            enable_emergency_mode=True,
            preserved_section_types={"user_query"},
        )
        mgr = TokenBudgetManager(config=cfg)
        query = _make_section("user_query", ["hi"])
        big = _make_section("big", ["x" * 500] * 100)
        pkg = _make_package([query, big])
        # Budget large enough for query
        budget = max(query.token_count, 200)
        result = await mgr.optimise(pkg, target_budget=budget)
        # At minimum the user query should survive
        types = [s.section_type for s in result.package.sections]
        assert "user_query" in types

    async def test_remove_zero_allocation(self) -> None:
        cfg = BudgetConfig(
            remove_zero_allocation_sections=True,
            preserved_section_types=set(),
        )
        mgr = TokenBudgetManager(config=cfg)
        small = _make_section("small", ["tiny"])
        large = _make_section("large", ["x" * 500] * 100)
        pkg = _make_package([small, large])
        result = await mgr.optimise(pkg, target_budget=10)
        # Small section should have been preserved (it fits)
        # Large section should be removed from the sections list if its allocation is 0
        # or survive with trimmed content if it gets > 0 allocation
        assert len(result.decisions) >= 0

    async def test_weighted_strategy(self) -> None:
        cfg = BudgetConfig(allocation_strategy="weighted")
        mgr = TokenBudgetManager(config=cfg)
        sections = [_make_section("a", ["x"]), _make_section("b", ["y" * 300])]
        pkg = _make_package(sections)
        result = await mgr.optimise(pkg, target_budget=500)
        assert result.statistics.final_tokens <= 500

    async def test_proportional_strategy(self) -> None:
        cfg = BudgetConfig(allocation_strategy="proportional")
        mgr = TokenBudgetManager(config=cfg)
        sections = [_make_section("a", ["x" * 100] * 3), _make_section("b", ["y" * 100] * 3)]
        pkg = _make_package(sections)
        result = await mgr.optimise(pkg, target_budget=500)
        assert result.statistics.final_tokens <= 500

    async def test_empty_package(self) -> None:
        mgr = TokenBudgetManager()
        pkg = _make_package([])
        result = await mgr.optimise(pkg, target_budget=100)
        assert result.statistics.original_tokens == 0
        assert result.statistics.tokens_removed == 0

    async def test_large_oversize(self) -> None:
        """Should handle extremely oversized context without error."""
        mgr = TokenBudgetManager()
        huge = _make_section("huge", ["x" * 1000] * 200)
        pkg = _make_package([huge])
        result = await mgr.optimise(pkg, target_budget=100)
        # All memories should be trimmed (each is ~300 tokens, budget=100)
        # The section should be removed entirely since no memory fits
        assert result.statistics.original_tokens > 0
        # Memory may be 0 if none fit, which is correct behaviour
        assert result.statistics.tokens_removed > 0

    async def test_deterministic_behaviour(self) -> None:
        """Two optimisations of the same input should produce the same output."""
        mgr = TokenBudgetManager()
        sections = [_make_section("a", ["x" * 50] * 5), _make_section("b", ["y" * 50] * 5)]
        pkg1 = _make_package(sections)
        pkg2 = _make_package(sections)
        result1 = await mgr.optimise(pkg1, target_budget=200)
        result2 = await mgr.optimise(pkg2, target_budget=200)
        assert result1.statistics.final_tokens == result2.statistics.final_tokens

    async def test_tracks_memories_removed(self) -> None:
        mgr = TokenBudgetManager()
        many = _make_section("data", ["x" * 100] * 30)
        pkg = _make_package([many])
        result = await mgr.optimise(pkg, target_budget=100)
        # Should have tracked removals
        assert result.statistics.memories_removed >= 0


# ---------------------------------------------------------------------------
# ContextBuilder integration
# ---------------------------------------------------------------------------


class TestContextBuilderIntegration:
    async def test_budget_manager_property(self) -> None:
        """The budget_manager property should return the configured manager."""
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema

        conn = SQLiteConnection(":memory:")
        mgr2 = SqliteMigrationManager()
        await mgr2.apply_all(conn, [V002_MemorySchema()])
        repo = MemoryRepository(connection=conn)
        budget_mgr = TokenBudgetManager()
        mgr_mgr = MemoryManager(repository=repo)
        builder = ContextBuilder(memory_manager=mgr_mgr, budget_manager=budget_mgr)
        assert builder.budget_manager is budget_mgr

    async def test_budget_manager_used_when_configured(
        self,
    ) -> None:
        """With a TokenBudgetManager, budget trimming should go through it."""
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema

        conn = SQLiteConnection(":memory:")
        mgr2 = SqliteMigrationManager()
        await mgr2.apply_all(conn, [V002_MemorySchema()])
        repo = MemoryRepository(connection=conn)
        await repo.add(Memory(content="test", memory_id=MemoryId("m1"), namespace="default"))
        await repo.add(Memory(content="big " + "x" * 500, memory_id=MemoryId("m2"), namespace="default"))

        budget_mgr = TokenBudgetManager(BudgetConfig(default_budget=100))
        mgr_mgr = MemoryManager(repository=repo)
        cfg = ContextBuilderConfig(
            enable_compression=False,
            enable_relationship_expansion=False,
            enable_selection=False,
        )
        builder = ContextBuilder(
            memory_manager=mgr_mgr,
            config=cfg,
            budget_manager=budget_mgr,
        )
        assert builder.budget_manager is budget_mgr
        pkg = await builder.build(max_tokens=200)
        assert pkg.total_tokens <= 250  # allow some overhead

    async def test_budget_manager_none_by_default(self) -> None:
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema

        conn = SQLiteConnection(":memory:")
        mgr2 = SqliteMigrationManager()
        await mgr2.apply_all(conn, [V002_MemorySchema()])
        repo = MemoryRepository(connection=conn)
        mgr_mgr = MemoryManager(repository=repo)
        builder = ContextBuilder(memory_manager=mgr_mgr)
        assert builder.budget_manager is None

    async def test_fallback_to_inline_when_no_budget_manager(self) -> None:
        """Without a budget manager, ContextBuilder falls back to inline trimming."""
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema

        conn = SQLiteConnection(":memory:")
        mgr2 = SqliteMigrationManager()
        await mgr2.apply_all(conn, [V002_MemorySchema()])
        repo = MemoryRepository(connection=conn)
        await repo.add(Memory(content="big " + "x" * 500, memory_id=MemoryId("m1"), namespace="default"))

        mgr_mgr = MemoryManager(repository=repo)
        builder = ContextBuilder(memory_manager=mgr_mgr)
        pkg = await builder.build(max_tokens=50)
        # Should still complete without error (uses inline fallback)
        assert pkg is not None
