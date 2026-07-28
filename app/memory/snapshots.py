"""Memory snapshots — point-in-time captures of memory state.

Architecture
============

A snapshot serialises **all** active memories and their relationships
into a single JSON blob stored in the ``memory_snapshots`` table.

Snapshots are **immutable** after creation.  They are never mutated.

Restoring a snapshot:

1. Clears all current memories and relationships
2. Re-inserts every memory from the snapshot
3. Re-inserts every relationship from the snapshot

This gives a clean rollback — no merge logic, no stale leftovers.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

from app.core.interfaces import EventBus, Logger, TelemetryService
from app.memory.events import SnapshotCreated, SnapshotRestored
from app.memory.interfaces import MemorySnapshot, MemorySnapshotService
from app.storage.interfaces import SQLConnection


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Repository
# ---------------------------------------------------------------------------


class _SnapshotRow:
    """Lightweight wrapper for a row from the memory_snapshots table."""

    __slots__ = ("id", "label", "created_at", "data")

    def __init__(
        self,
        *,
        id: str = "",
        label: str = "",
        created_at: str = "",
        data: str = "{}",
    ) -> None:
        self.id = id
        self.label = label
        self.created_at = created_at
        self.data = data


class SnapshotRepository:
    """SQLite-backed persistence for memory snapshots."""

    def __init__(self, connection: SQLConnection) -> None:
        self._conn = connection

    async def add(self, snapshot: _SnapshotRow) -> _SnapshotRow:
        await self._conn.execute(
            """
            INSERT INTO memory_snapshots (id, label, created_at, data)
            VALUES (:id, :label, :created_at, :data)
            """,
            {
                "id": snapshot.id,
                "label": snapshot.label,
                "created_at": snapshot.created_at,
                "data": snapshot.data,
            },
        )
        return snapshot

    async def get(self, snapshot_id: str) -> _SnapshotRow | None:
        row = await self._conn.fetchone(
            "SELECT * FROM memory_snapshots WHERE id = :id",
            {"id": snapshot_id},
        )
        return _SnapshotRow(**row) if row else None

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[_SnapshotRow]:
        rows = await self._conn.fetchall(
            "SELECT * FROM memory_snapshots ORDER BY created_at DESC LIMIT :limit OFFSET :offset",
            {"limit": limit, "offset": offset},
        )
        return [_SnapshotRow(**r) for r in rows]

    async def count(self) -> int:
        row = await self._conn.fetchone("SELECT COUNT(*) as cnt FROM memory_snapshots")
        return row["cnt"] if row else 0

    async def delete(self, snapshot_id: str) -> bool:
        await self._conn.execute(
            "DELETE FROM memory_snapshots WHERE id = :id",
            {"id": snapshot_id},
        )
        return True


# ---------------------------------------------------------------------------
# Domain model (implements MemorySnapshot ABC)
# ---------------------------------------------------------------------------

import uuid


class MemorySnapshotImpl(MemorySnapshot):
    """A point-in-time capture of all memories and relationships.

    Once created, the snapshot is immutable — ``_data``, ``_label``,
    and ``_created_at`` are never modified.
    """

    def __init__(
        self,
        *,
        snapshot_id: str | None = None,
        label: str = "",
        data: str = "{}",
        created_at: str | None = None,
        connection: SQLConnection | None = None,
        telemetry: TelemetryService | None = None,
        logger: Logger | None = None,
    ) -> None:
        self._snapshot_id = snapshot_id or str(uuid.uuid4())
        self._label = label
        self._data = data
        self._created_at = created_at
        self._connection = connection
        self._telemetry = telemetry
        self._logger = logger

    # -- MemorySnapshot ABC --------------------------------------------------

    @property
    def snapshot_id(self) -> str:
        """Return the unique snapshot identifier."""
        return self._snapshot_id

    async def restore(self) -> int:
        """Restore all memories to this snapshot's state.

        Replaces **all** current memories and relationships with the
        contents captured in this snapshot.

        Returns the number of memories restored.
        """
        if self._connection is None:
            raise RuntimeError("Cannot restore: no database connection")

        start = time.monotonic()

        payload = json.loads(self._data)
        memories: list[dict[str, Any]] = payload.get("memories", [])
        relationships: list[dict[str, Any]] = payload.get("relationships", [])

        # 1. Clear existing data
        await self._connection.execute("DELETE FROM memory_relationships")
        await self._connection.execute("DELETE FROM memories")

        # 2. Re-insert memories
        for row in memories:
            columns = ", ".join(row.keys())
            placeholders = ", ".join([f":{k}" for k in row])
            await self._connection.execute(
                f"INSERT INTO memories ({columns}) VALUES ({placeholders})",
                row,
            )

        # 3. Re-insert relationships
        for rel in relationships:
            columns = ", ".join(rel.keys())
            placeholders = ", ".join([f":{k}" for k in rel])
            await self._connection.execute(
                f"INSERT INTO memory_relationships ({columns}) VALUES ({placeholders})",
                rel,
            )

        elapsed = (time.monotonic() - start) * 1000
        if self._telemetry:
            self._telemetry.record_module_lifecycle(
                "memory", "snapshot_restore", elapsed, True,
            )

        if self._logger:
            self._logger.info(
                "Snapshot restored | id={sid} label={label} memories={nmem} relationships={nrel}",
                sid=self._snapshot_id,
                label=self._label,
                nmem=len(memories),
                nrel=len(relationships),
            )

        return len(memories)

    def to_dict(self) -> dict[str, Any]:
        """Return snapshot metadata as a dict."""
        return {
            "snapshot_id": self._snapshot_id,
            "label": self._label,
            "created_at": self._created_at or "",
            "memory_count": self._memory_count_from_data(),
        }

    # -- Internal helpers ----------------------------------------------------

    def _memory_count_from_data(self) -> int:
        try:
            payload = json.loads(self._data)
            return len(payload.get("memories", []))
        except (json.JSONDecodeError, TypeError):
            return 0


# ---------------------------------------------------------------------------
# Service (implements MemorySnapshotService ABC)
# ---------------------------------------------------------------------------


class SnapshotService(MemorySnapshotService):
    """Orchestrates snapshot creation, listing, and restoration.

    All snapshot operations go through this class.
    """

    def __init__(
        self,
        repository: SnapshotRepository,
        connection: SQLConnection,
        event_bus: EventBus | None = None,
        telemetry: TelemetryService | None = None,
        logger: Logger | None = None,
    ) -> None:
        self._repo = repository
        self._conn = connection
        self._event_bus = event_bus
        self._telemetry = telemetry
        self._logger = logger

    async def create_snapshot(self, label: str = "") -> MemorySnapshot:
        """Capture a point-in-time snapshot of all memories and relationships.

        1. Reads all memories from the database
        2. Reads all relationships from the database
        3. Serialises them into a JSON blob
        4. Persists the snapshot row
        5. Emits ``SnapshotCreated``
        6. Returns the ``MemorySnapshotImpl``
        """
        start = time.monotonic()

        # 1. Serialise all memories
        memory_rows = await self._conn.fetchall("SELECT * FROM memories")
        memories: list[dict[str, Any]] = []
        for row in memory_rows:
            memory_dict = dict(row)
            memories.append(memory_dict)

        # 2. Serialise all relationships
        rel_rows = await self._conn.fetchall("SELECT * FROM memory_relationships")
        relationships: list[dict[str, Any]] = []
        for row in rel_rows:
            rel_dict = dict(row)
            relationships.append(rel_dict)

        # 3. Build JSON payload
        payload = json.dumps({
            "memories": memories,
            "relationships": relationships,
        })

        # 4. Create and persist
        now_iso = _now_utc().isoformat()
        snapshot = MemorySnapshotImpl(
            label=label,
            data=payload,
            created_at=now_iso,
            connection=self._conn,
            telemetry=self._telemetry,
            logger=self._logger,
        )

        row = _SnapshotRow(
            id=snapshot.snapshot_id,
            label=label,
            created_at=now_iso,
            data=payload,
        )
        await self._repo.add(row)

        # 5. Emit event
        await self._emit_event(
            SnapshotCreated(
                snapshot_id=snapshot.snapshot_id,
                label=label,
                memory_count=len(memories),
                relationship_count=len(relationships),
            ),
        )

        elapsed = (time.monotonic() - start) * 1000
        if self._telemetry:
            self._telemetry.record_module_lifecycle(
                "memory", "snapshot_create", elapsed, True,
            )

        if self._logger:
            self._logger.info(
                "Snapshot created | id={sid} label={label} memories={nmem} relationships={nrel} elapsed_ms={ms}",
                sid=snapshot.snapshot_id,
                label=label,
                nmem=len(memories),
                nrel=len(relationships),
                ms=round(elapsed, 2),
            )

        return snapshot

    async def list_snapshots(self) -> list[dict[str, Any]]:
        """Return metadata for all snapshots, newest first."""
        rows = await self._repo.list(limit=1000)
        result: list[dict[str, Any]] = []
        for row in rows:
            snap = MemorySnapshotImpl(
                snapshot_id=row.id,
                label=row.label,
                data=row.data,
                created_at=row.created_at,
            )
            result.append(snap.to_dict())
        return result

    async def get_snapshot(self, snapshot_id: str) -> MemorySnapshot | None:
        """Retrieve a single snapshot by ID (including its data)."""
        row = await self._repo.get(snapshot_id)
        if row is None:
            return None
        return MemorySnapshotImpl(
            snapshot_id=row.id,
            label=row.label,
            data=row.data,
            created_at=row.created_at,
            connection=self._conn,
            telemetry=self._telemetry,
            logger=self._logger,
        )

    async def restore_snapshot(self, snapshot_id: str) -> int:
        """Restore a snapshot by ID.  Returns the number of memories restored.

        Emits ``SnapshotRestored`` after successful restoration.
        """
        snapshot = await self.get_snapshot(snapshot_id)
        if snapshot is None:
            raise LookupError(f"Snapshot {snapshot_id!r} not found")

        count = await snapshot.restore()

        await self._emit_event(
            SnapshotRestored(
                snapshot_id=snapshot_id,
                label=snapshot.to_dict().get("label", ""),
                memory_count=count,
            ),
        )

        return count

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """Remove a snapshot from the store.

        This does **not** affect the memories themselves — it only removes
        the snapshot record.
        """
        return await self._repo.delete(snapshot_id)

    async def count_snapshots(self) -> int:
        """Return the total number of stored snapshots."""
        return await self._repo.count()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit_event(self, event: Any) -> None:
        if self._event_bus:
            await self._event_bus.publish(event)
