"""Tests for EventRepository.

Verifies domain-specific query methods on top of base CRUD.
"""

from __future__ import annotations

import pytest

from app.storage.connection.sqlite import SQLiteConnection
from app.storage.migration.manager import SqliteMigrationManager
from app.storage.migration.sqlite import V001_InitialSchema
from app.storage.repository.event_repo import EventRecord, EventRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def repo() -> EventRepository:
    conn = SQLiteConnection(":memory:")
    manager = SqliteMigrationManager()
    await manager.apply_all(conn, [V001_InitialSchema()])
    yield EventRepository(connection=conn)
    await conn.close()


class TestEventRepository:
    async def test_add_and_get(self, repo: EventRepository) -> None:
        record = EventRecord(
            id="evt-1",
            event_type="test.event",
            source="test",
            correlation_id="corr-1",
            timestamp="2026-01-01T00:00:00",
            payload='{"msg": "hello"}',
        )
        await repo.add(record)
        retrieved = await repo.get("evt-1")
        assert retrieved is not None
        assert retrieved.event_type == "test.event"

    async def test_find_by_type(self, repo: EventRepository) -> None:
        await repo._connection.execute(
            "INSERT INTO event_store (id, event_type, source, correlation_id, timestamp, payload, metadata) VALUES "
            "('a', 'type.x', 'src', 'c1', '2026-01-01', '{}', '{}'), "
            "('b', 'type.y', 'src', 'c2', '2026-01-02', '{}', '{}'), "
            "('c', 'type.x', 'src', 'c3', '2026-01-03', '{}', '{}')"
        )
        page = await repo.find_by_type("type.x")
        assert page.total == 2

    async def test_find_by_correlation(self, repo: EventRepository) -> None:
        await repo._connection.execute(
            "INSERT INTO event_store (id, event_type, source, correlation_id, timestamp, payload, metadata) VALUES "
            "('a', 't1', 'src', 'abc', '2026-01-01', '{}', '{}'), "
            "('b', 't2', 'src', 'abc', '2026-01-02', '{}', '{}')"
        )
        page = await repo.find_by_correlation("abc")
        assert page.total == 2

    async def test_find_recent(self, repo: EventRepository) -> None:
        await repo._connection.execute(
            "INSERT INTO event_store (id, event_type, source, correlation_id, timestamp, payload, metadata) VALUES "
            "('a', 't1', 'src', 'c1', '2026-01-01', '{}', '{}'), "
            "('b', 't2', 'src', 'c2', '2026-01-02', '{}', '{}')"
        )
        page = await repo.find_recent(limit=1)
        assert page.total == 2
        assert len(page.items) == 1

    async def test_count(self, repo: EventRepository) -> None:
        await repo._connection.execute(
            "INSERT INTO event_store (id, event_type, source, correlation_id, timestamp, payload, metadata) VALUES "
            "('a', 't1', 'src', 'c1', '2026-01-01', '{}', '{}')"
        )
        assert await repo.count() == 1
