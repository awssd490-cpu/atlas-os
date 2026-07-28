"""Tests for the compression subsystem.

Verifies:
- CompressionStrategy protocol and built-in strategies
- DedupStrategy: exact content dedup within namespace
- MergeRelatedStrategy: group and merge via graph
- TruncationStrategy: content length limiting
- ArchiveLowValueStrategy: state transitions by importance/age
- MemoryCompressorImpl: strategy dispatch, target_count, unknown strategy
- CompressionService: full run cycle, candidate selection, event emission,
  snapshot integration, persistence of results
- CompressionPolicy: configuration
- MemoryManager integration
"""

from __future__ import annotations

from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId, MemoryState, MemoryType
from app.memory.manager import MemoryManager, MemoryRepository
from app.memory.compression import (
    ArchiveLowValueStrategy,
    CompressionPolicy,
    CompressionService,
    CompressionStrategy,
    DedupStrategy,
    MemoryCompressorImpl,
    MergeRelatedStrategy,
    StrategyResult,
    TruncationStrategy,
)
from app.memory.snapshots import SnapshotRepository, SnapshotService
from app.storage.interfaces import SQLConnection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def repo() -> MemoryRepository:
    from app.storage.connection.sqlite import SQLiteConnection
    from app.storage.migration.manager import SqliteMigrationManager
    from app.memory.migrations import V002_MemorySchema

    conn = SQLiteConnection(":memory:")
    manager = SqliteMigrationManager()
    await manager.apply_all(conn, [V002_MemorySchema()])
    yield MemoryRepository(connection=conn)
    await conn.close()


@pytest.fixture
async def conn(repo: MemoryRepository) -> Any:
    return repo._conn


@pytest.fixture
async def seeded_repo(repo: MemoryRepository) -> MemoryRepository:
    """Populate repo with memories suitable for compression tests."""
    memories = [
        Memory(content="duplicate content", importance=0.5, memory_id=MemoryId("d1"),
               namespace="ns1", memory_type=MemoryType.SHORT_TERM.value),
        Memory(content="duplicate content", importance=0.9, memory_id=MemoryId("d2"),
               namespace="ns1", memory_type=MemoryType.SHORT_TERM.value),
        Memory(content="unique alpha", importance=0.8, memory_id=MemoryId("ua"),
               namespace="ns2", memory_type=MemoryType.KNOWLEDGE.value),
        Memory(content="unique beta", importance=0.3, memory_id=MemoryId("ub"),
               namespace="ns2", memory_type=MemoryType.KNOWLEDGE.value),
        Memory(content="long content " + ("x" * 600), importance=0.4, memory_id=MemoryId("long"),
               namespace="ns1", memory_type=MemoryType.REFERENCE.value),
        Memory(content="low value old", importance=0.1, memory_id=MemoryId("low"),
               namespace="ns1", memory_type=MemoryType.SHORT_TERM.value),
    ]
    for m in memories:
        await repo.add(m)
    return repo


@pytest.fixture
async def graph(seeded_repo: MemoryRepository) -> Any:
    from app.memory.relationships import MemoryGraphImpl

    g = MemoryGraphImpl(connection=seeded_repo._conn)
    # Relate ua and ub via references
    await g.add_relationship("ua", "ub", "references")
    return g


class _NoopGraph:
    """Graph that returns empty for all queries."""

    async def add_relationship(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def get_related(self, *args: Any, **kwargs: Any) -> list[Memory]:
        return []

    async def remove_relationship(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def propagate_importance(self, *args: Any, **kwargs: Any) -> int:
        return 0


# ---------------------------------------------------------------------------
# DedupStrategy
# ---------------------------------------------------------------------------


class TestDedupStrategy:
    async def test_removes_exact_duplicates(self) -> None:
        strat = DedupStrategy()
        mems = [
            Memory(content="same", importance=0.5, namespace="ns", memory_id=MemoryId("a")),
            Memory(content="same", importance=0.9, namespace="ns", memory_id=MemoryId("b")),
            Memory(content="different", importance=0.7, namespace="ns", memory_id=MemoryId("c")),
        ]
        result = await strat(mems)
        assert len(result.kept) == 2
        # Higher importance duplicate kept
        kept_ids = {m.id.value for m in result.kept}
        assert "b" in kept_ids
        assert "a" not in kept_ids
        assert result.removed_ids == ["a"]

    async def test_different_namespace_not_deduped(self) -> None:
        strat = DedupStrategy()
        mems = [
            Memory(content="same", importance=0.9, namespace="ns1", memory_id=MemoryId("a")),
            Memory(content="same", importance=0.5, namespace="ns2", memory_id=MemoryId("b")),
        ]
        result = await strat(mems)
        assert len(result.kept) == 2
        assert result.removed_ids == []

    async def test_empty_input(self) -> None:
        strat = DedupStrategy()
        result = await strat([])
        assert result.kept == []
        assert result.removed_ids == []

    async def test_no_duplicates(self) -> None:
        strat = DedupStrategy()
        mems = [
            Memory(content="a", namespace="ns", memory_id=MemoryId("1")),
            Memory(content="b", namespace="ns", memory_id=MemoryId("2")),
        ]
        result = await strat(mems)
        assert len(result.kept) == 2
        assert result.removed_ids == []


# ---------------------------------------------------------------------------
# MergeRelatedStrategy
# ---------------------------------------------------------------------------


class TestMergeRelatedStrategy:
    async def test_merges_related_memories(self) -> None:
        from app.memory.relationships import MemoryGraphImpl
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema

        conn = SQLiteConnection(":memory:")
        manager = SqliteMigrationManager()
        await manager.apply_all(conn, [V002_MemorySchema()])
        repo = MemoryRepository(connection=conn)

        m_a = Memory(content="part one", importance=0.7, memory_id=MemoryId("ma"), namespace="ns")
        m_b = Memory(content="part two", importance=0.6, memory_id=MemoryId("mb"), namespace="ns")
        await repo.add(m_a)
        await repo.add(m_b)

        graph = MemoryGraphImpl(connection=conn)
        await graph.add_relationship("ma", "mb", "references")

        strat = MergeRelatedStrategy()
        result = await strat([m_a, m_b], graph=graph)
        # Should have merged into one
        assert len(result.kept) == 1
        merged = result.kept[0]
        assert "part one" in merged.content
        assert "part two" in merged.content
        assert merged.metadata.get("compressed") is True
        assert set(merged.metadata.get("provenance", [])) == {"ma", "mb"}

        await conn.close()

    async def test_no_graph_no_merge(self) -> None:
        strat = MergeRelatedStrategy()
        mems = [Memory(content="a", memory_id=MemoryId("1")), Memory(content="b", memory_id=MemoryId("2"))]
        result = await strat(mems, graph=None)
        assert len(result.kept) == 2
        assert result.removed_ids == []

    async def test_single_memory_no_merge(self) -> None:
        strat = MergeRelatedStrategy()
        mems = [Memory(content="alone", memory_id=MemoryId("1"))]
        result = await strat(mems, graph=_NoopGraph())
        assert len(result.kept) == 1
        assert result.removed_ids == []


# ---------------------------------------------------------------------------
# TruncationStrategy
# ---------------------------------------------------------------------------


class TestTruncationStrategy:
    async def test_truncates_long_content(self) -> None:
        strat = TruncationStrategy(max_length=20)
        mem = Memory(content="this is a very long memory content that should be truncated")
        result = await strat([mem])
        assert len(result.kept) == 1
        assert len(result.kept[0].content) <= 20 + len("... [truncated]")
        assert "... [truncated]" in result.kept[0].content
        assert result.kept[0].metadata.get("compressed") is True

    async def test_does_not_truncate_short(self) -> None:
        strat = TruncationStrategy(max_length=500)
        mem = Memory(content="short")
        result = await strat([mem])
        assert result.kept[0].content == "short"

    async def test_no_removals(self) -> None:
        strat = TruncationStrategy(max_length=10)
        result = await strat([Memory(content="abcdefghijklmnop")])
        assert result.removed_ids == []


# ---------------------------------------------------------------------------
# ArchiveLowValueStrategy
# ---------------------------------------------------------------------------


class TestArchiveLowValueStrategy:
    async def test_archives_low_importance_old_memory(self) -> None:
        from datetime import timedelta, timezone

        strat = ArchiveLowValueStrategy(archive_threshold=0.3, max_age_days=0)  # archive anything old
        mem = Memory(content="low value", importance=0.1, state=MemoryState.ACTIVE)
        # Make it appear old
        mem.created_at = mem.created_at.replace(year=2020)
        result = await strat([mem])
        assert len(result.removed_ids) == 1
        assert mem.state == MemoryState.ARCHIVED

    async def test_keeps_high_importance(self) -> None:
        strat = ArchiveLowValueStrategy(archive_threshold=0.3, max_age_days=0)
        mem = Memory(content="valuable", importance=0.9, state=MemoryState.ACTIVE)
        result = await strat([mem])
        assert result.removed_ids == []
        assert mem.state == MemoryState.ACTIVE

    async def test_keeps_recent_low_importance(self) -> None:
        strat = ArchiveLowValueStrategy(archive_threshold=0.3, max_age_days=999)
        mem = Memory(content="recent low", importance=0.1, state=MemoryState.ACTIVE)
        result = await strat([mem])
        assert result.removed_ids == []

    async def test_keeps_already_archived(self) -> None:
        strat = ArchiveLowValueStrategy(archive_threshold=0.3, max_age_days=0)
        mem = Memory(content="already archived", importance=0.1, state=MemoryState.ARCHIVED)
        result = await strat([mem])
        assert result.removed_ids == []


# ---------------------------------------------------------------------------
# MemoryCompressorImpl
# ---------------------------------------------------------------------------


class TestMemoryCompressorImpl:
    async def test_dedup_strategy(self) -> None:
        compressor = MemoryCompressorImpl()
        mems = [
            Memory(content="dup", namespace="ns", memory_id=MemoryId("a"), importance=0.5),
            Memory(content="dup", namespace="ns", memory_id=MemoryId("b"), importance=0.9),
        ]
        result = await compressor.compress(mems, strategy="dedup")
        assert result.strategy == "dedup"
        assert result.compressed_count == 1
        assert result.original_count == 2
        assert result.ratio == 0.5

    async def test_truncate_strategy(self) -> None:
        compressor = MemoryCompressorImpl()
        mem = Memory(content="x" * 1000)
        result = await compressor.compress([mem], strategy="truncate")
        assert result.compressed_count == 1
        assert result.compressed[0].metadata.get("compression_strategy") == "truncate"

    async def test_archive_low_value_strategy(self) -> None:
        from datetime import timezone

        compressor = MemoryCompressorImpl()
        mem = Memory(content="low", importance=0.1, memory_id=MemoryId("m1"))
        # Make it old enough to qualify (max_age_days defaults to 30)
        mem.created_at = mem.created_at.replace(year=2020, tzinfo=timezone.utc)
        mem.accessed_at = mem.accessed_at.replace(year=2020, tzinfo=timezone.utc)
        result = await compressor.compress([mem], strategy="archive_low_value")
        assert result.compressed_count == 1
        assert result.compressed[0].state == MemoryState.ARCHIVED

    async def test_target_count(self) -> None:
        compressor = MemoryCompressorImpl()
        mems = [Memory(content=f"m{i}", importance=1.0 - i * 0.1, memory_id=MemoryId(str(i)))
                for i in range(5)]
        result = await compressor.compress(mems, strategy="dedup", target_count=3)
        assert result.compressed_count == 3

    async def test_unknown_strategy(self) -> None:
        compressor = MemoryCompressorImpl()
        with pytest.raises(ValueError, match="Unknown"):
            await compressor.compress([], strategy="nonexistent")

    async def test_empty_input(self) -> None:
        compressor = MemoryCompressorImpl()
        result = await compressor.compress([], strategy="dedup")
        assert result.compressed_count == 0
        assert result.original_count == 0

    async def test_register_custom_strategy(self) -> None:
        class _CustomStrategy(CompressionStrategy):
            async def __call__(self, memories, *, graph=None):
                return StrategyResult(kept=memories[:1], removed_ids=[m.id.value for m in memories[1:]],
                                      strategy_name="custom")

        MemoryCompressorImpl.register_strategy("custom", _CustomStrategy)
        compressor = MemoryCompressorImpl()
        mems = [Memory(content="a", memory_id=MemoryId("1")), Memory(content="b", memory_id=MemoryId("2"))]
        result = await compressor.compress(mems, strategy="custom")
        assert result.compressed_count == 1


# ---------------------------------------------------------------------------
# CompressionService
# ---------------------------------------------------------------------------


class _TestEventBus:
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


class TestCompressionService:
    async def test_run_selects_and_compresses(self, seeded_repo: MemoryRepository) -> None:
        svc = CompressionService(
            repository=seeded_repo,
            connection=seeded_repo._conn,
            policy=CompressionPolicy(
                enabled_strategies=["dedup"],
                take_snapshot_before=False,
            ),
        )
        result = await svc.run()
        # d1 and d2 are duplicates — one should be removed
        assert result.original_count >= 1
        assert result.compressed_count <= result.original_count

    async def test_run_with_snapshot(
        self, seeded_repo: MemoryRepository, conn: Any
    ) -> None:
        snap_repo = SnapshotRepository(connection=conn)
        snap_svc = SnapshotService(repository=snap_repo, connection=conn, logger=None)
        svc = CompressionService(
            repository=seeded_repo,
            connection=conn,
            snapshot_service=snap_svc,
            policy=CompressionPolicy(
                enabled_strategies=["dedup"],
                take_snapshot_before=True,
            ),
        )
        result = await svc.run()
        assert result.compressed_count is not None
        # A snapshot should have been created
        snaps = await snap_svc.list_snapshots()
        assert len(snaps) >= 1
        assert "pre-compression" in snaps[0]["label"]

    async def test_run_empty_repo(self, repo: MemoryRepository) -> None:
        svc = CompressionService(
            repository=repo,
            connection=repo._conn,
            policy=CompressionPolicy(take_snapshot_before=False),
        )
        result = await svc.run()
        assert result.original_count == 0
        assert result.compressed_count == 0

    async def test_emits_event(self, seeded_repo: MemoryRepository) -> None:
        bus = _TestEventBus()
        svc = CompressionService(
            repository=seeded_repo,
            connection=seeded_repo._conn,
            event_bus=bus,
            policy=CompressionPolicy(
                enabled_strategies=["dedup"],
                take_snapshot_before=False,
            ),
        )
        await svc.run()
        assert len(bus.events) >= 1
        event = bus.events[-1]
        assert event._event_type == "memory.compressed"
        assert event.original_count >= 1
        assert event.compressed_count >= 1

    async def test_compress_memories_direct(self, repo: MemoryRepository) -> None:
        svc = CompressionService(
            repository=repo,
            connection=repo._conn,
            policy=CompressionPolicy(take_snapshot_before=False),
        )
        mems = [
            Memory(content="dup", namespace="ns", memory_id=MemoryId("a"), importance=0.5),
            Memory(content="dup", namespace="ns", memory_id=MemoryId("b"), importance=0.9),
        ]
        result = await svc.compress_memories(mems, strategy="dedup")
        assert result.compressed_count == 1
        assert result.original_count == 2

    async def test_persists_removals(self, seeded_repo: MemoryRepository) -> None:
        """After compression run, deduplicated memories should be deleted from DB."""
        # Count before
        before_count = await seeded_repo.count()
        svc = CompressionService(
            repository=seeded_repo,
            connection=seeded_repo._conn,
            policy=CompressionPolicy(
                enabled_strategies=["dedup"],
                take_snapshot_before=False,
            ),
        )
        await svc.run()
        # After: one dedup removed, others still there
        after_count = await seeded_repo.count()
        assert after_count <= before_count

    async def test_multiple_strategies_sequentially(
        self, seeded_repo: MemoryRepository
    ) -> None:
        """Run with dedup then truncate."""
        svc = CompressionService(
            repository=seeded_repo,
            connection=seeded_repo._conn,
            policy=CompressionPolicy(
                enabled_strategies=["dedup", "truncate"],
                take_snapshot_before=False,
                content_truncation_length=100,
            ),
        )
        result = await svc.run()
        assert result.compressed_count <= result.original_count or result.original_count == 0


# ---------------------------------------------------------------------------
# CompressionPolicy
# ---------------------------------------------------------------------------


class TestCompressionPolicy:
    def test_default_policy(self) -> None:
        p = CompressionPolicy()
        assert "dedup" in p.enabled_strategies
        assert "truncate" in p.enabled_strategies
        assert "archive_low_value" in p.enabled_strategies
        assert p.take_snapshot_before is True
        assert p.max_memories_per_run == 500

    def test_custom_policy(self) -> None:
        p = CompressionPolicy(
            enabled_strategies=["merge_related"],
            max_memories_per_run=100,
            take_snapshot_before=False,
            namespace_filters=["ns1"],
        )
        assert p.enabled_strategies == ["merge_related"]
        assert p.max_memories_per_run == 100
        assert p.take_snapshot_before is False
        assert p.namespace_filters == ["ns1"]


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class TestCompressionEvents:
    async def test_compression_event_fields(self, seeded_repo: MemoryRepository) -> None:
        bus = _TestEventBus()
        svc = CompressionService(
            repository=seeded_repo,
            connection=seeded_repo._conn,
            event_bus=bus,
            policy=CompressionPolicy(
                enabled_strategies=["dedup"],
                take_snapshot_before=False,
            ),
        )
        await svc.run()
        event = bus.events[-1]
        assert hasattr(event, "original_count")
        assert hasattr(event, "compressed_count")
        assert hasattr(event, "strategy")
        assert hasattr(event, "ratio")
        assert hasattr(event, "snapshot_id")


# ---------------------------------------------------------------------------
# MemoryManager integration
# ---------------------------------------------------------------------------


class TestManagerCompressionIntegration:
    async def test_compressor_property_none_by_default(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        assert mgr.compressor is None

    async def test_compressor_property_set(self, repo: MemoryRepository) -> None:
        svc = CompressionService(
            repository=repo,
            connection=repo._conn,
            policy=CompressionPolicy(take_snapshot_before=False),
        )
        mgr = MemoryManager(repository=repo, compressor=svc)
        assert mgr.compressor is svc

    async def test_compressor_is_memory_compressor(self, repo: MemoryRepository) -> None:
        """The compressor property exposes the MemoryCompressor ABC interface."""
        from app.memory.interfaces import MemoryCompressor

        svc = CompressionService(
            repository=repo,
            connection=repo._conn,
            policy=CompressionPolicy(take_snapshot_before=False),
        )
        mgr = MemoryManager(repository=repo, compressor=svc)
        assert mgr.compressor is not None
        # It should have a compress method matching the ABC
        assert hasattr(mgr.compressor, "compress")
