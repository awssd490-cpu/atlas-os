"""Memory lifecycle events.

These events are published on the EventBus whenever a memory transitions
to a new state or is accessed.  The EventStore (Phase 2) persists them
automatically if the EventStore subscriber is active.
"""

from __future__ import annotations

from typing import Any, ClassVar

from app.core.events import Event


class MemoryCreated(Event):
    """Published when a new memory is created."""

    _event_type: ClassVar[str] = "memory.created"
    source: str = "memory"
    memory_id: str = ""
    memory_type: str = ""
    namespace: str = "default"
    importance: float = 0.5


class MemoryUpdated(Event):
    """Published when an existing memory's content or metadata changes."""

    _event_type: ClassVar[str] = "memory.updated"
    source: str = "memory"
    memory_id: str = ""
    version: int = 1
    changes: dict[str, Any] = {}
    importance_delta: float = 0.0


class MemoryAccessed(Event):
    """Published when a memory is accessed (read)."""

    _event_type: ClassVar[str] = "memory.accessed"
    source: str = "memory"
    memory_id: str = ""
    via: str = ""


class MemoryStateChanged(Event):
    """Published when a memory moves to a new lifecycle state."""

    _event_type: ClassVar[str] = "memory.state_changed"
    source: str = "memory"
    memory_id: str = ""
    from_state: str = ""
    to_state: str = ""
    reason: str = ""


class MemoryRelationshipsUpdated(Event):
    """Published when a memory's relationships are modified."""

    _event_type: ClassVar[str] = "memory.relationships_updated"
    source: str = "memory"
    memory_id: str = ""
    added: list[str] = []
    removed: list[str] = []


class SnapshotCreated(Event):
    """Published when a memory snapshot is created."""

    _event_type: ClassVar[str] = "memory.snapshot_created"
    source: str = "memory"
    snapshot_id: str = ""
    label: str = ""
    memory_count: int = 0
    relationship_count: int = 0


class SnapshotRestored(Event):
    """Published when a snapshot is restored (memories rolled back)."""

    _event_type: ClassVar[str] = "memory.snapshot_restored"
    source: str = "memory"
    snapshot_id: str = ""
    label: str = ""
    memory_count: int = 0


class MemoriesCompressed(Event):
    """Published after a compression run completes."""

    _event_type: ClassVar[str] = "memory.compressed"
    source: str = "memory"
    original_count: int = 0
    compressed_count: int = 0
    strategy: str = ""
    ratio: float = 1.0
    snapshot_id: str = ""
