"""Tests for SqliteEventStore.

Verifies:
- Append and count events
- Stream by type, correlation, source
- Stream by time range
- Replay all returns in chronological order
- Event payload is serialized/deserialized correctly
"""

from __future__ import annotations

from datetime import datetime, timezone

from typing import ClassVar
import pytest

from app.core.events import Event
from app.storage.connection.sqlite import SQLiteConnection
from app.storage.event_store.service import SqliteEventStore
from app.storage.migration.manager import SqliteMigrationManager
from app.storage.migration.sqlite import V001_InitialSchema


# ---------------------------------------------------------------------------
# Test events
# ---------------------------------------------------------------------------


class SomethingHappened(Event):
    _event_type: ClassVar[str] = "something.happened"
    source: str = "test"
    description: str = ""


class AnotherEvent(Event):
    _event_type: ClassVar[str] = "another.event"
    source: str = "test"
    value: int = 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> SqliteEventStore:
    """Create an event store backed by an in-memory SQLite database
    with the V001 migration applied."""
    c = SQLiteConnection(":memory:")
    manager = SqliteMigrationManager()
    await manager.apply_all(c, [V001_InitialSchema()])
    yield SqliteEventStore(connection=c)
    await c.close()


@pytest.fixture
async def populated_store(store: SqliteEventStore) -> SqliteEventStore:
    """Event store with several events already persisted."""
    events = [
        SomethingHappened(description="first", correlation_id="corr-1"),
        SomethingHappened(description="second", correlation_id="corr-1"),
        AnotherEvent(value=42, correlation_id="corr-2"),
    ]
    for ev in events:
        await store.append(ev)
    return store


# ---------------------------------------------------------------------------
# Append / Count
# ---------------------------------------------------------------------------


class TestAppend:
    async def test_append_and_count(self, store: SqliteEventStore) -> None:
        event = SomethingHappened(description="test")
        await store.append(event)
        assert await store.count() == 1

    async def test_append_multiple(self, store: SqliteEventStore) -> None:
        await store.append(SomethingHappened(description="a"))
        await store.append(SomethingHappened(description="b"))
        assert await store.count() == 2


# ---------------------------------------------------------------------------
# Stream by type
# ---------------------------------------------------------------------------


class TestStreamByType:
    async def test_stream_by_type(self, populated_store: SqliteEventStore) -> None:
        events = await populated_store.stream_by_type("something.happened")
        assert len(events) == 2
        assert all(e.event_type == "something.happened" for e in events)

    async def test_stream_by_type_empty(self, store: SqliteEventStore) -> None:
        events = await store.stream_by_type("nonexistent")
        assert events == []


# ---------------------------------------------------------------------------
# Stream by correlation
# ---------------------------------------------------------------------------


class TestStreamByCorrelation:
    async def test_stream_by_correlation(self, populated_store: SqliteEventStore) -> None:
        events = await populated_store.stream_by_correlation("corr-1")
        assert len(events) == 2
        assert all(e.correlation_id == "corr-1" for e in events)

    async def test_stream_by_correlation_empty(self, store: SqliteEventStore) -> None:
        events = await store.stream_by_correlation("nonexistent")
        assert events == []


# ---------------------------------------------------------------------------
# Stream by source
# ---------------------------------------------------------------------------


class TestStreamBySource:
    async def test_stream_by_source(self, populated_store: SqliteEventStore) -> None:
        events = await populated_store.stream_by_source("test")
        assert len(events) == 3


# ---------------------------------------------------------------------------
# Stream by time range
# ---------------------------------------------------------------------------


class TestStreamByTimeRange:
    async def test_time_range_bounds(self, populated_store: SqliteEventStore) -> None:
        # Use a very broad range that covers the events
        events = await populated_store.stream_by_time_range("2020-01-01", "2030-12-31")
        assert len(events) == 3

    async def test_time_range_narrow(self, populated_store: SqliteEventStore) -> None:
        events = await populated_store.stream_by_time_range("2020-01-01", "2020-01-02")
        assert len(events) == 0


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


class TestReplay:
    async def test_replay_all(self, populated_store: SqliteEventStore) -> None:
        events = await populated_store.replay_all()
        assert len(events) == 3
        # Verify chronological order by event_id ordering
        assert events[0].event_type == "something.happened"
        assert events[1].event_type == "something.happened"
        assert events[2].event_type == "another.event"

    async def test_replay_empty(self, store: SqliteEventStore) -> None:
        events = await store.replay_all()
        assert events == []


# ---------------------------------------------------------------------------
# Payload serialization
# ---------------------------------------------------------------------------


class TestPayload:
    async def test_payload_contains_event_data(self, store: SqliteEventStore) -> None:
        event = SomethingHappened(description="hello world")
        await store.append(event)

        stored = await store.replay_all()
        assert len(stored) == 1
        payload = stored[0].payload
        # The payload should contain the description field as JSON
        assert '"hello world"' in payload

    async def test_payload_excludes_envelope(self, store: SqliteEventStore) -> None:
        event = SomethingHappened(description="test")
        await store.append(event)
        stored = await store.replay_all()
        assert len(stored) == 1
        payload = stored[0].payload
        # Envelope fields should NOT appear in the payload
        assert '"event_id"' not in payload
