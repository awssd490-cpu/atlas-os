"""Memory graph — SQL-backed relationship management.

Implements the MemoryGraph ABC from ``interfaces.py`` using the
``memory_relationships`` table created by V002_MemorySchema.

Cycle prevention is enforced for tree-like relationship types
(PARENT, CHILD, DEPENDS_ON).  Other types permit cycles.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.interfaces import EventBus, Logger, TelemetryService
from app.memory.events import MemoryRelationshipsUpdated
from app.memory.interfaces import MemoryGraph
from app.memory.memory import Memory, MemoryState
from app.storage.interfaces import SQLConnection


class _RelationshipRow:
    """Lightweight row wrapper for memory_relationships records.

    Avoids coupling to the storage-layer GraphRelationship (which would
    require mapping between two nearly-identical types).
    """

    __slots__ = ("id", "source_id", "target_id", "rel_type", "properties", "created_at")

    def __init__(
        self,
        *,
        id: int = 0,
        source_id: str = "",
        target_id: str = "",
        rel_type: str = "",
        properties: dict[str, Any] | None = None,
        created_at: str = "",
    ) -> None:
        self.id = id
        self.source_id = source_id
        self.target_id = target_id
        self.rel_type = rel_type
        self.properties = properties or {}
        self.created_at = created_at

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "_RelationshipRow":
        import json

        props_raw = row.get("properties", "{}")
        try:
            props = json.loads(props_raw) if isinstance(props_raw, str) else props_raw
        except (json.JSONDecodeError, TypeError):
            props = {}
        return cls(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            rel_type=row["rel_type"],
            properties=props,
            created_at=row.get("created_at", ""),
        )


_TREE_TYPES = frozenset({"parent", "child", "depends_on"})


class MemoryGraphImpl(MemoryGraph):
    """SQLite-backed implementation of the MemoryGraph interface.

    Stores relationships in the ``memory_relationships`` table and
    fetches related memories by joining against the ``memories`` table.
    """

    def __init__(
        self,
        connection: SQLConnection,
        event_bus: EventBus | None = None,
        telemetry: TelemetryService | None = None,
        logger: Logger | None = None,
    ) -> None:
        self._conn = connection
        self._event_bus = event_bus
        self._telemetry = telemetry
        self._logger = logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a relationship between two memories.

        Raises:
            ValueError: If *source_id* equals *target_id*, the
                relationship already exists, or a cycle would be created.
            LookupError: If either memory does not exist.
        """
        start = time.monotonic()

        if source_id == target_id:
            raise ValueError(
                f"Cannot create a self-referencing relationship "
                f"({source_id} -> {target_id}, type={rel_type})"
            )

        # Validate both memories exist
        source_exists = await self._memory_exists(source_id)
        if not source_exists:
            raise LookupError(f"Source memory {source_id!r} not found")
        target_exists = await self._memory_exists(target_id)
        if not target_exists:
            raise LookupError(f"Target memory {target_id!r} not found")

        # Check for duplicate
        existing = await self._find_relationship(source_id, target_id, rel_type)
        if existing is not None:
            raise ValueError(
                f"Relationship already exists: "
                f"({source_id} -> {target_id}, type={rel_type})"
            )

        # Cycle prevention for tree-like types
        if rel_type.lower() in _TREE_TYPES:
            if await self._would_create_cycle(source_id, target_id, rel_type):
                raise ValueError(
                    f"Cannot create {rel_type!r} relationship "
                    f"from {source_id} to {target_id} — it would create a cycle"
                )

        # Insert
        import json

        props_json = json.dumps(properties or {})
        await self._conn.execute(
            """
            INSERT INTO memory_relationships (source_id, target_id, rel_type, properties)
            VALUES (:source_id, :target_id, :rel_type, :properties)
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "rel_type": rel_type,
                "properties": props_json,
            },
        )

        await self._emit_updated(source_id, added=[target_id])

        self._record("graph.add_relationship", start)

    async def get_related(
        self,
        memory_id: str,
        *,
        rel_type: str | None = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[Memory]:
        """Return memories related to *memory_id*.

        Direction:
            ``"outgoing"`` — follow relationships from *memory_id* → target
            ``"incoming"`` — follow relationships to *memory_id* (target → source)
            ``"both"``      — follow both directions
        """
        start = time.monotonic()

        if max_depth < 1:
            self._record("graph.get_related", start)
            return []

        visited: set[str] = set()
        current_level: set[str] = {memory_id}
        results: list[Memory] = []

        for _depth in range(max_depth):
            if not current_level:
                break
            next_level: set[str] = set()
            for mid in current_level:
                if mid in visited:
                    continue
                visited.add(mid)
                neighbours = await self._get_direct_neighbours(
                    mid, rel_type=rel_type, direction=direction
                )
                for nid in neighbours:
                    if nid not in visited and nid not in current_level:
                        next_level.add(nid)

            if next_level:
                memories = await self._fetch_memories(list(next_level))
                results.extend(memories)
            current_level = next_level

        self._record("graph.get_related", start)
        return results

    async def remove_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str | None = None,
    ) -> None:
        """Remove a relationship (or all between two memories)."""
        start = time.monotonic()

        if rel_type:
            await self._conn.execute(
                """
                DELETE FROM memory_relationships
                WHERE source_id = :source_id AND target_id = :target_id AND rel_type = :rel_type
                """,
                {"source_id": source_id, "target_id": target_id, "rel_type": rel_type},
            )
        else:
            await self._conn.execute(
                """
                DELETE FROM memory_relationships
                WHERE source_id = :source_id AND target_id = :target_id
                """,
                {"source_id": source_id, "target_id": target_id},
            )

        await self._emit_updated(source_id, removed=[target_id])

        self._record("graph.remove_relationship", start)

    async def propagate_importance(
        self,
        memory_id: str,
        *,
        decay: float = 0.5,
        max_depth: int = 3,
    ) -> int:
        """Propagate importance to related memories with *decay* per hop.

        Uses BFS, multiplying importance by (1 - decay) at each level.
        Returns the number of memories updated.
        """
        start = time.monotonic()

        if max_depth < 1:
            self._record("graph.propagate_importance", start)
            return 0

        source_memory = await self._fetch_memory(memory_id)
        if source_memory is None:
            self._record("graph.propagate_importance", start)
            return 0

        source_importance = source_memory.importance
        visited: set[str] = {memory_id}
        current_level: set[str] = {memory_id}
        updated_count = 0

        for depth in range(1, max_depth + 1):
            if not current_level:
                break
            next_level: set[str] = set()
            for mid in current_level:
                neighbours = await self._get_direct_neighbours(mid, direction="both")
                for nid in neighbours:
                    if nid not in visited:
                        next_level.add(nid)
                        visited.add(nid)

            if next_level:
                related_memories = await self._fetch_memories(list(next_level))
                factor = max(0.0, 1.0 - decay * depth)
                boost = source_importance * factor
                for mem in related_memories:
                    if mem.importance < boost:
                        mem.promote(boost - mem.importance)
                        await self._persist_memory(mem)
                        updated_count += 1

            current_level = next_level

        if updated_count and self._logger:
            self._logger.info(
                "Importance propagated | source={source} updated={updated} max_depth={depth} | decay={decay}",
                source=memory_id,
                updated=updated_count,
                depth=max_depth,
                decay=decay,
            )

        self._record("graph.propagate_importance", start)
        return updated_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _memory_exists(self, memory_id: str) -> bool:
        """Return True when a memory with *memory_id* exists."""
        row = await self._conn.fetchone(
            "SELECT 1 FROM memories WHERE id = :id",
            {"id": memory_id},
        )
        return row is not None

    async def _find_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
    ) -> _RelationshipRow | None:
        """Return an existing relationship or None."""
        row = await self._conn.fetchone(
            """
            SELECT * FROM memory_relationships
            WHERE source_id = :source_id AND target_id = :target_id AND rel_type = :rel_type
            """,
            {"source_id": source_id, "target_id": target_id, "rel_type": rel_type},
        )
        return _RelationshipRow.from_row(row) if row else None

    async def _would_create_cycle(self, source_id: str, target_id: str, rel_type: str) -> bool:
        """Return True when adding source→target would create a cycle.

        BFS starting from *target_id*, following outgoing relationships
        of matching types.  If we reach *source_id*, a cycle exists.
        """
        normalized_type = rel_type.lower()

        # Determine which direction to follow based on relationship type.
        # parent: source is parent of target → cycles form if target is already an ancestor of source
        # child: source is child of target → cycles form if target is already a descendant of source
        # depends_on: source depends on target → cycles form if target already depends (transitively) on source
        # For all tree types, we follow outgoing from target and check if we reach source.
        visited: set[str] = set()
        queue: list[str] = [target_id]

        while queue:
            current = queue.pop(0)
            if current == source_id:
                return True
            if current in visited:
                continue
            visited.add(current)

            # Follow outgoing relationships of the same type
            rows = await self._conn.fetchall(
                """
                SELECT target_id FROM memory_relationships
                WHERE source_id = :current AND rel_type = :rel_type
                """,
                {"current": current, "rel_type": rel_type},
            )
            for row in rows:
                nid = row["target_id"]
                if nid not in visited:
                    queue.append(nid)

        return False

    async def _get_direct_neighbours(
        self,
        memory_id: str,
        *,
        rel_type: str | None = None,
        direction: str = "both",
    ) -> list[str]:
        """Return immediate neighbour IDs for *memory_id*."""
        seen: set[str] = set()

        if direction in ("outgoing", "both"):
            sql = """
                SELECT target_id AS neighbour FROM memory_relationships
                WHERE source_id = :mid
            """
            params: dict[str, Any] = {"mid": memory_id}
            if rel_type:
                sql += " AND rel_type = :rel_type"
                params["rel_type"] = rel_type
            rows = await self._conn.fetchall(sql, params)
            for row in rows:
                seen.add(row["neighbour"])

        if direction in ("incoming", "both"):
            sql = """
                SELECT source_id AS neighbour FROM memory_relationships
                WHERE target_id = :mid
            """
            params = {"mid": memory_id}
            if rel_type:
                sql += " AND rel_type = :rel_type"
                params["rel_type"] = rel_type
            rows = await self._conn.fetchall(sql, params)
            for row in rows:
                seen.add(row["neighbour"])

        return list(seen)

    async def _fetch_memory(self, memory_id: str) -> Memory | None:
        """Fetch a single Memory by ID string."""
        row = await self._conn.fetchone(
            "SELECT * FROM memories WHERE id = :id",
            {"id": memory_id},
        )
        return Memory.from_dict(row) if row else None

    async def _fetch_memories(self, memory_ids: list[str]) -> list[Memory]:
        """Fetch multiple memories, preserving order where possible."""
        if not memory_ids:
            return []
        placeholders = ", ".join([f":id_{i}" for i in range(len(memory_ids))])
        params: dict[str, Any] = {f"id_{i}": mid for i, mid in enumerate(memory_ids)}
        rows = await self._conn.fetchall(
            f"SELECT * FROM memories WHERE id IN ({placeholders})",
            params,
        )
        return [Memory.from_dict(r) for r in rows]

    async def _persist_memory(self, memory: Memory) -> None:
        """Write a memory's current state to the database."""
        row = memory.to_dict()
        set_clause = ", ".join(
            [f"{k} = :{k}" for k in row if k != "id"]
        )
        await self._conn.execute(
            f"UPDATE memories SET {set_clause} WHERE id = :id",
            {**row, "id": memory.id.value},
        )

    async def _emit_updated(
        self,
        memory_id: str,
        added: list[str] | None = None,
        removed: list[str] | None = None,
    ) -> None:
        """Emit MemoryRelationshipsUpdated on the event bus."""
        if self._event_bus:
            await self._event_bus.publish(
                MemoryRelationshipsUpdated(
                    memory_id=memory_id,
                    added=added or [],
                    removed=removed or [],
                )
            )

    def _record(self, operation: str, start: float) -> None:
        """Record telemetry for an operation."""
        if self._telemetry:
            elapsed = (time.monotonic() - start) * 1000
            self._telemetry.record_event_metrics(operation, elapsed, True)
