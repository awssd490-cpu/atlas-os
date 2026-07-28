"""Tests for the snapshot subsystem.

Verifies:
- SnapshotRepository CRUD
- MemorySnapshotImpl immutability, restore, to_dict
- SnapshotService create_snapshot, list_snapshots, get_snapshot,
  restore_snapshot, delete_snapshot, count_snapshots
- Snapshot creation captures both memories and relationships
- Snapshot restore clears existing data and re-inserts
- Snapshot events are emitted
- MemoryManager integration (snapshots property)
- Edge cases: empty database, non-existent snapshot,
  snapshot without a connection
"""

from __future__ import annotations

from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId, MemoryState
from app.memory.manager import MemoryManager, MemoryRepository
from app.memory.snapshots import (
    MemorySnapshotImpl,
    SnapshotRepository,
    SnapshotService,
)
from app.storage.interfaces import SQLConnection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def conn() -> Any:
    """Return an in-memory SQLite connection with the V002 schema applied."""
    from app.storage.connection.sqlite import SQLiteConnection
    from app.storage.migration.manager import SqliteMigrationManager
    from app.memory.migrations import V002_MemorySchema

    c = SQLiteConnection(":memory:")
    manager = SqliteMigrationManager()
    await manager.apply_all(c, [V002_MemorySchema()])
    yield c
    await c.close()


@pytest.fixture
async def repo(conn: Any) -> MemoryRepository:
    """Return a MemoryRepository backed by *conn*."""
    return MemoryRepository(connection=conn)


@pytest.fixture
async def snapshot_repo(conn: Any) -> SnapshotRepository:
    return SnapshotRepository(connection=conn)


@pytest.fixture
async def seeded_repo(repo: MemoryRepository, conn: Any) -> tuple[MemoryRepository, SQLConnection]:
    """Repo with 3 memories and 1 relationship."""
    m_a = Memory(content="alpha", importance=0.9, memory_id=MemoryId("ma"))
    m_b = Memory(content="beta", importance=0.5, memory_id=MemoryId("mb"))
    m_c = Memory(content="gamma", importance=0.3, memory_id=MemoryId("mc"))
    await repo.add(m_a)
    await repo.add(m_b)
    await repo.add(m_c)

    conn = repo._conn
    await conn.execute(
        "INSERT INTO memory_relationships (source_id, target_id, rel_type) VALUES (:s, :t, :r)",
        {"s": "ma", "t": "mb", "r": "references"},
    )
    return repo, conn


@pytest.fixture
async def seeded_service(
    seeded_repo: tuple[MemoryRepository, SQLConnection],
    snapshot_repo: SnapshotRepository,
) -> SnapshotService:
    _, conn = seeded_repo
    return SnapshotService(
        repository=snapshot_repo,
        connection=conn,
    )


# ---------------------------------------------------------------------------
# SnapshotRepository
# ---------------------------------------------------------------------------


class TestSnapshotRepository:
    async def test_add_and_get(self, snapshot_repo: SnapshotRepository) -> None:
        from app.memory.snapshots import _SnapshotRow

        row = _SnapshotRow(id="snap1", label="test", data="{}", created_at="2025-01-01")
        await snapshot_repo.add(row)

        retrieved = await snapshot_repo.get("snap1")
        assert retrieved is not None
        assert retrieved.id == "snap1"
        assert retrieved.label == "test"

    async def test_get_nonexistent(self, snapshot_repo: SnapshotRepository) -> None:
        retrieved = await snapshot_repo.get("no-such")
        assert retrieved is None

    async def test_list_empty(self, snapshot_repo: SnapshotRepository) -> None:
        rows = await snapshot_repo.list()
        assert rows == []

    async def test_list_multiple(self, snapshot_repo: SnapshotRepository) -> None:
        from app.memory.snapshots import _SnapshotRow

        for i in range(3):
            await snapshot_repo.add(
                _SnapshotRow(id=f"snap{i}", label=f"test {i}", data="{}", created_at=f"2025-01-0{i+1}")
            )
        rows = await snapshot_repo.list()
        assert len(rows) == 3

    async def test_count(self, snapshot_repo: SnapshotRepository) -> None:
        from app.memory.snapshots import _SnapshotRow

        assert await snapshot_repo.count() == 0
        await snapshot_repo.add(_SnapshotRow(id="s1", label="x", data="{}"))
        assert await snapshot_repo.count() == 1

    async def test_delete(self, snapshot_repo: SnapshotRepository) -> None:
        from app.memory.snapshots import _SnapshotRow

        await snapshot_repo.add(_SnapshotRow(id="s1", label="x", data="{}"))
        await snapshot_repo.delete("s1")
        assert await snapshot_repo.get("s1") is None

    async def test_list_orders_by_created_at_desc(
        self, snapshot_repo: SnapshotRepository
    ) -> None:
        from app.memory.snapshots import _SnapshotRow

        await snapshot_repo.add(
            _SnapshotRow(id="old", label="old", data="{}", created_at="2024-01-01")
        )
        await snapshot_repo.add(
            _SnapshotRow(id="new", label="new", data="{}", created_at="2025-01-01")
        )
        rows = await snapshot_repo.list()
        assert rows[0].id == "new"
        assert rows[1].id == "old"


# ---------------------------------------------------------------------------
# MemorySnapshotImpl
# ---------------------------------------------------------------------------


class TestMemorySnapshotImpl:
    def test_creates_with_default_id(self) -> None:
        snap = MemorySnapshotImpl()
        assert snap.snapshot_id != ""
        assert snap.snapshot_id is not None

    def test_to_dict_returns_metadata(self) -> None:
        snap = MemorySnapshotImpl(
            snapshot_id="s1",
            label="my snap",
            data='{"memories": [{"id": "m1"}]}',
            created_at="2025-01-01",
        )
        d = snap.to_dict()
        assert d["snapshot_id"] == "s1"
        assert d["label"] == "my snap"
        assert d["created_at"] == "2025-01-01"
        assert d["memory_count"] == 1

    async def test_restore_requires_connection(self) -> None:
        snap = MemorySnapshotImpl()
        with pytest.raises(RuntimeError, match="no database connection"):
            await snap.restore()

    async def test_restore_replaces_memories(self, conn: Any) -> None:
        """Restoring a snapshot should clear existing data and insert snapshot data."""
        # Seed some existing data
        await conn.execute(
            "INSERT INTO memories (id, content) VALUES (:id, :c)",
            {"id": "old", "c": "to-be-cleared"},
        )

        # Create a snapshot with replacement data
        payload = '{"memories": [{"id": "new1", "content": "replacement", "memory_type": "short_term", "namespace": "default", "importance": 0.5, "confidence": 1.0, "ttl": null, "state": "active", "source": "manual", "owner": "system", "tags": "", "metadata": "{}", "correlation_id": "", "created_at": "2025-01-01", "updated_at": "2025-01-01", "accessed_at": "2025-01-01", "archived_at": null, "forgotten_at": null, "deleted_at": null, "access_count": 0, "version": 1}], "relationships": []}'
        snap = MemorySnapshotImpl(
            snapshot_id="s1",
            data=payload,
            connection=conn,
        )
        count = await snap.restore()
        assert count == 1

        # Old data should be gone
        row_old = await conn.fetchone("SELECT * FROM memories WHERE id = :id", {"id": "old"})
        assert row_old is None

        # New data should be present
        row_new = await conn.fetchone("SELECT * FROM memories WHERE id = :id", {"id": "new1"})
        assert row_new is not None
        assert row_new["content"] == "replacement"

    async def test_restore_also_restores_relationships(self, conn: Any) -> None:
        """Restoring should also rebuild the relationship table."""
        payload = json.dumps({
            "memories": [
                {
                    "id": "ma", "content": "a", "memory_type": "short_term",
                    "namespace": "default", "importance": 0.5, "confidence": 1.0,
                    "ttl": None, "state": "active", "source": "manual", "owner": "system",
                    "tags": "", "metadata": "{}", "correlation_id": "",
                    "created_at": "2025-01-01", "updated_at": "2025-01-01",
                    "accessed_at": "2025-01-01",
                    "archived_at": None, "forgotten_at": None, "deleted_at": None,
                    "access_count": 0, "version": 1,
                },
                {
                    "id": "mb", "content": "b", "memory_type": "short_term",
                    "namespace": "default", "importance": 0.5, "confidence": 1.0,
                    "ttl": None, "state": "active", "source": "manual", "owner": "system",
                    "tags": "", "metadata": "{}", "correlation_id": "",
                    "created_at": "2025-01-01", "updated_at": "2025-01-01",
                    "accessed_at": "2025-01-01",
                    "archived_at": None, "forgotten_at": None, "deleted_at": None,
                    "access_count": 0, "version": 1,
                },
            ],
            "relationships": [
                {"source_id": "ma", "target_id": "mb", "rel_type": "references",
                 "properties": "{}", "created_at": "2025-01-01"},
            ],
        })
        snap = MemorySnapshotImpl(snapshot_id="s1", data=payload, connection=conn)
        count = await snap.restore()
        assert count == 2

        # Relationship should exist
        rel_row = await conn.fetchone(
            "SELECT * FROM memory_relationships WHERE source_id = :s AND target_id = :t",
            {"s": "ma", "t": "mb"},
        )
        assert rel_row is not None
        assert rel_row["rel_type"] == "references"

    def test_immutable_after_creation(self) -> None:
        """Snapshot fields should not be mutable after construction."""
        snap = MemorySnapshotImpl(snapshot_id="s1", label="fixed", data="{}")
        assert snap.snapshot_id == "s1"
        # There should be no setter that changes the id
        with pytest.raises(AttributeError):
            snap.snapshot_id = "different"  # type: ignore[assignment]


import json  # noqa: E402 (needed by the test above)


# ---------------------------------------------------------------------------
# SnapshotService
# ---------------------------------------------------------------------------


class TestSnapshotService:
    async def test_create_snapshot(
        self, seeded_service: SnapshotService
    ) -> None:
        snap = await seeded_service.create_snapshot(label="test snap")
        assert snap.snapshot_id != ""
        assert snap.to_dict()["label"] == "test snap"
        assert snap.to_dict()["memory_count"] == 3  # 3 seeded memories

    async def test_create_snapshot_empty_db(
        self, snapshot_repo: SnapshotRepository, conn: Any
    ) -> None:
        svc = SnapshotService(repository=snapshot_repo, connection=conn)
        snap = await svc.create_snapshot(label="empty")
        assert snap.to_dict()["memory_count"] == 0

    async def test_list_snapshots(self, seeded_service: SnapshotService) -> None:
        await seeded_service.create_snapshot(label="first")
        await seeded_service.create_snapshot(label="second")
        snaps = await seeded_service.list_snapshots()
        assert len(snaps) == 2
        # Newest first
        assert snaps[0]["label"] == "second"

    async def test_get_snapshot(self, seeded_service: SnapshotService) -> None:
        created = await seeded_service.create_snapshot(label="find-me")
        retrieved = await seeded_service.get_snapshot(created.snapshot_id)
        assert retrieved is not None
        assert retrieved.to_dict()["label"] == "find-me"

    async def test_get_snapshot_nonexistent(
        self, seeded_service: SnapshotService
    ) -> None:
        retrieved = await seeded_service.get_snapshot("no-such")
        assert retrieved is None

    async def test_restore_snapshot(
        self, seeded_repo: tuple[MemoryRepository, SQLConnection],
        snapshot_repo: SnapshotRepository,
    ) -> None:
        repo, conn = seeded_repo
        svc = SnapshotService(repository=snapshot_repo, connection=conn)

        # Create a snapshot (3 memories)
        snap = await svc.create_snapshot(label="before-clear")

        # Add more memories after snapshot
        await repo.add(Memory(content="post-snap", memory_id=MemoryId("post")))
        assert await repo.count() == 4

        # Restore
        count = await svc.restore_snapshot(snap.snapshot_id)
        assert count == 3  # back to 3
        assert await repo.count() == 3

        # Post-snap memory should be gone
        assert await repo.get(MemoryId("post")) is None

    async def test_restore_nonexistent_snapshot(
        self, seeded_service: SnapshotService
    ) -> None:
        with pytest.raises(LookupError, match="not found"):
            await seeded_service.restore_snapshot("no-such")

    async def test_delete_snapshot(
        self, seeded_service: SnapshotService
    ) -> None:
        snap = await seeded_service.create_snapshot(label="to-delete")
        assert await seeded_service.count_snapshots() == 1
        await seeded_service.delete_snapshot(snap.snapshot_id)
        assert await seeded_service.count_snapshots() == 0

    async def test_count_snapshots(
        self, seeded_service: SnapshotService
    ) -> None:
        assert await seeded_service.count_snapshots() == 0
        await seeded_service.create_snapshot()
        assert await seeded_service.count_snapshots() == 1


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class _TestEventBus:
    """Simple event bus that records published events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


class TestSnapshotEvents:
    async def test_create_emits_event(
        self, snapshot_repo: SnapshotRepository, conn: Any
    ) -> None:
        bus = _TestEventBus()
        svc = SnapshotService(
            repository=snapshot_repo,
            connection=conn,
            event_bus=bus,
        )
        await svc.create_snapshot(label="event-test")
        assert len(bus.events) == 1
        event = bus.events[0]
        assert event._event_type == "memory.snapshot_created"
        assert event.label == "event-test"

    async def test_restore_emits_event(
        self, snapshot_repo: SnapshotRepository, conn: Any
    ) -> None:
        bus = _TestEventBus()
        svc = SnapshotService(
            repository=snapshot_repo,
            connection=conn,
            event_bus=bus,
        )
        snap = await svc.create_snapshot(label="restore-test")
        await svc.restore_snapshot(snap.snapshot_id)
        assert len(bus.events) == 2
        restore_event = bus.events[1]
        assert restore_event._event_type == "memory.snapshot_restored"
        assert restore_event.snapshot_id == snap.snapshot_id


# ---------------------------------------------------------------------------
# MemoryManager integration
# ---------------------------------------------------------------------------


class TestManagerSnapshotIntegration:
    async def test_snapshots_property_none_by_default(
        self, repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=repo)
        assert mgr.snapshots is None

    async def test_snapshots_property_returns_service(
        self, repo: MemoryRepository, snapshot_repo: SnapshotRepository, conn: Any
    ) -> None:
        svc = SnapshotService(repository=snapshot_repo, connection=conn)
        mgr = MemoryManager(repository=repo, snapshot_service=svc)
        assert mgr.snapshots is svc

    async def test_create_snapshot_through_manager(
        self, repo: MemoryRepository, snapshot_repo: SnapshotRepository, conn: Any
    ) -> None:
        svc = SnapshotService(repository=snapshot_repo, connection=conn)
        mgr = MemoryManager(repository=repo, snapshot_service=svc)

        # Create a memory through the manager
        m = Memory(content="manager memory", importance=0.8)
        await mgr.create(m)

        snap = await mgr.create_snapshot(label="via manager")
        assert snap.to_dict()["memory_count"] >= 1
        assert snap.to_dict()["label"] == "via manager"

    async def test_list_snapshots_through_manager(
        self, repo: MemoryRepository, snapshot_repo: SnapshotRepository, conn: Any
    ) -> None:
        svc = SnapshotService(repository=snapshot_repo, connection=conn)
        mgr = MemoryManager(repository=repo, snapshot_service=svc)

        await mgr.create_snapshot(label="s1")
        await mgr.create_snapshot(label="s2")
        snaps = await mgr.list_snapshots()
        assert len(snaps) == 2

    async def test_raise_when_no_snapshot_service(
        self, repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=repo)
        with pytest.raises(RuntimeError, match="not configured"):
            await mgr.create_snapshot()

    async def test_list_empty_when_no_snapshot_service(
        self, repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=repo)
        snaps = await mgr.list_snapshots()
        assert snaps == []

    async def test_snapshot_through_manager_restores_correctly(
        self, repo: MemoryRepository, snapshot_repo: SnapshotRepository, conn: Any
    ) -> None:
        svc = SnapshotService(repository=snapshot_repo, connection=conn, logger=None)
        mgr = MemoryManager(repository=repo, snapshot_service=svc)

        # Create and snapshot
        m1 = Memory(content="original", importance=0.9, memory_id=MemoryId("m1"))
        await mgr.create(m1)
        snap = await mgr.create_snapshot(label="checkpoint")

        # Modify after snapshot
        await mgr.delete(MemoryId("m1"))

        # Restore
        count = await svc.restore_snapshot(snap.snapshot_id)
        assert count == 1
        restored = await repo.get(MemoryId("m1"))
        assert restored is not None
        assert restored.content == "original"
