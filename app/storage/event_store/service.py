"""Event store — persists and replays bus events.

Stores every event as a JSON-serialized row in the ``event_store``
table (created by the V001 migration).  Supports streaming by type,
correlation ID, source, and time range.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.events import Event
from app.storage.interfaces import (
    EventStore,
    SQLConnection,
    StoredEvent,
)


class SqliteEventStore(EventStore):
    """Event store backed by SQLite.

    Events are serialized to JSON for storage and deserialized on read.
    The ``event_store`` table must exist (created by V001 migration).
    """

    def __init__(self, connection: SQLConnection) -> None:
        self._connection = connection

    async def append(self, event: Any) -> None:
        """Persist a single event.

        The event can be any Pydantic model that has the standard ATLAS
        envelope fields (id, event_type, version, source, correlation_id,
        target, timestamp, metadata).
        """
        # Extract the event type string
        event_type = event.event_type() if hasattr(event, "event_type") else type(event).__name__

        # Serialize the event — exclude envelope fields from the payload
        payload_data = self._event_to_payload(event)
        metadata_json = json.dumps(event.metadata) if hasattr(event, "metadata") else "{}"
        timestamp = event.timestamp.isoformat() if hasattr(event, "timestamp") else ""

        await self._connection.execute(
            """
            INSERT INTO event_store
                (id, event_type, version, source, correlation_id, target, timestamp, payload, metadata)
            VALUES
                (:id, :event_type, :version, :source, :correlation_id, :target, :timestamp, :payload, :metadata)
            """,
            {
                "id": event.event_id if hasattr(event, "event_id") else "",
                "event_type": event_type,
                "version": getattr(event, "version", 1),
                "source": getattr(event, "source", ""),
                "correlation_id": getattr(event, "correlation_id", ""),
                "target": getattr(event, "target", "*"),
                "timestamp": timestamp,
                "payload": json.dumps(payload_data),
                "metadata": metadata_json,
            },
        )

    async def stream_by_type(self, event_type: str) -> list[StoredEvent]:
        """Return all events of a given type."""
        rows = await self._connection.fetchall(
            "SELECT * FROM event_store WHERE event_type = :event_type ORDER BY timestamp",
            {"event_type": event_type},
        )
        return [self._row_to_stored(r) for r in rows]

    async def stream_by_correlation(self, correlation_id: str) -> list[StoredEvent]:
        """Return all events sharing a correlation ID."""
        rows = await self._connection.fetchall(
            "SELECT * FROM event_store WHERE correlation_id = :correlation_id ORDER BY timestamp",
            {"correlation_id": correlation_id},
        )
        return [self._row_to_stored(r) for r in rows]

    async def stream_by_source(self, source: str) -> list[StoredEvent]:
        """Return all events from a given source."""
        rows = await self._connection.fetchall(
            "SELECT * FROM event_store WHERE source = :source ORDER BY timestamp",
            {"source": source},
        )
        return [self._row_to_stored(r) for r in rows]

    async def stream_by_time_range(self, start: str, end: str) -> list[StoredEvent]:
        """Return events within an ISO-8601 time range (inclusive)."""
        rows = await self._connection.fetchall(
            "SELECT * FROM event_store WHERE timestamp >= :start AND timestamp <= :end ORDER BY timestamp",
            {"start": start, "end": end},
        )
        return [self._row_to_stored(r) for r in rows]

    async def replay_all(self) -> list[StoredEvent]:
        """Return every stored event in chronological order."""
        rows = await self._connection.fetchall(
            "SELECT * FROM event_store ORDER BY timestamp"
        )
        return [self._row_to_stored(r) for r in rows]

    async def count(self) -> int:
        """Return the total number of persisted events."""
        row = await self._connection.fetchone("SELECT COUNT(*) as cnt FROM event_store")
        return row["cnt"] if row else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _event_to_payload(event: Any) -> dict[str, Any]:
        """Extract payload data from an event.

        Returns all fields except the standard envelope fields.
        """
        event_dict = event.model_dump() if hasattr(event, "model_dump") else vars(event)
        # Remove standard envelope fields from payload
        for field in ("event_id", "version", "timestamp", "correlation_id", "source", "target", "metadata"):
            event_dict.pop(field, None)
        return event_dict

    @staticmethod
    def _row_to_stored(row: dict[str, Any]) -> StoredEvent:
        """Convert a database row to a StoredEvent."""
        return StoredEvent(
            id=row["id"],
            event_type=row["event_type"],
            version=row["version"],
            source=row["source"],
            correlation_id=row["correlation_id"],
            target=row["target"],
            timestamp=row["timestamp"],
            payload=row["payload"],
            metadata=row["metadata"],
        )
