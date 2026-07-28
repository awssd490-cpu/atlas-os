"""Memory domain model.

A Memory is the fundamental unit of knowledge in ATLAS. It represents a
single piece of information with its lifecycle, importance, relationships,
and metadata.

Memory is a plain Python class (not a Pydantic model) so it can be used
across all layers without framework coupling. Serialization happens at
the repository boundary.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any


class MemoryState(str, enum.Enum):
    """Lifecycle states for a Memory.

    Active → Archived → Forgotten → Deleted

    Active:   Normal operational state, included in default search.
    Archived: Preserved but excluded from default retrieval.
              Recoverable via explicit request or importance boost.
    Forgotten: Marked for deletion by policy. Enters a grace period
               before hard deletion. Recovery is possible during grace.
    Deleted:  Irreversibly removed. Only trace is in the EventStore.
    """

    ACTIVE = "active"
    ARCHIVED = "archived"
    FORGOTTEN = "forgotten"
    DELETED = "deleted"


class MemoryRelationshipType(str, enum.Enum):
    """Semantic relationship types between memories."""

    PARENT = "parent"
    CHILD = "child"
    RELATED = "related"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    DERIVED_FROM = "derived_from"
    DUPLICATE = "duplicate"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"


class MemoryType(str, enum.Enum):
    """Memory categories.

    Each type defaults to a different policy profile (TTL, decay, capacity)
    configured in MemoryConfig. Types are data, not code — adding a new
    type means adding a policy entry, not a subclass.
    """

    SHORT_TERM = "short_term"
    WORKING = "working"
    LONG_TERM = "long_term"
    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    CONVERSATION = "conversation"
    PROJECT = "project"
    KNOWLEDGE = "knowledge"
    REFERENCE = "reference"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


class MemoryId:
    """Strongly-typed memory identifier.

    Wraps a UUID string with explicit type so method signatures are
    self-documenting.
    """

    def __init__(self, value: str | None = None) -> None:
        self._value = value or str(uuid.uuid4())

    @property
    def value(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"MemoryId({self._value})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MemoryId):
            return self._value == other._value
        if isinstance(other, str):
            return self._value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


class Memory:
    """A single unit of knowledge in the ATLAS memory system.

    All fields are mutable.  The repository is responsible for persisting
    changes.

    Attributes are documented inline below.
    """

    def __init__(
        self,
        *,
        memory_id: MemoryId | None = None,
        memory_type: str = MemoryType.SHORT_TERM.value,
        namespace: str = "default",
        content: str = "",
        importance: float = 0.5,
        confidence: float = 1.0,
        ttl: float | None = None,
        state: MemoryState = MemoryState.ACTIVE,
        source: str = "manual",
        owner: str = "system",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        now = _now_utc()

        self.id = memory_id or MemoryId()
        self.memory_type = memory_type
        self.namespace = namespace
        self.content = content
        self.importance = importance
        self.confidence = confidence
        self.ttl = ttl
        self.state = state
        self.source = source
        self.owner = owner
        self.tags = tags or []
        self.metadata = metadata or {}
        self.correlation_id = correlation_id or ""

        # Timestamps
        self.created_at = now
        self.updated_at = now
        self.accessed_at = now
        self.archived_at: datetime | None = None
        self.forgotten_at: datetime | None = None
        self.deleted_at: datetime | None = None

        # Access tracking
        self.access_count: int = 0

        # Versioning
        self.version: int = 1

        # Relationship references (maintained by MemoryGraph)
        self.relationship_ids: list[str] = []

    # ------------------------------------------------------------------
    # Domain behavior
    # ------------------------------------------------------------------

    def touch(self) -> None:
        """Record an access to this memory."""
        self.accessed_at = _now_utc()
        self.access_count += 1

    def promote(self, amount: float = 0.1) -> None:
        """Increase importance (capped at 1.0)."""
        self.importance = min(1.0, self.importance + amount)
        self.updated_at = _now_utc()

    def decay(self, rate: float = 0.1) -> None:
        """Decrease importance over time (floored at 0.0)."""
        self.importance = max(0.0, self.importance - rate)
        self.updated_at = _now_utc()

    def transition_to(self, target: MemoryState) -> None:
        """Transition to a new state and record the timestamp."""
        now = _now_utc()
        self.state = target
        self.updated_at = now
        if target == MemoryState.ARCHIVED:
            self.archived_at = now
        elif target == MemoryState.FORGOTTEN:
            self.forgotten_at = now
        elif target == MemoryState.DELETED:
            self.deleted_at = now

    @property
    def is_expired(self) -> bool:
        """Return True when the TTL has elapsed."""
        if self.ttl is None or self.ttl <= 0:
            return False
        if self.created_at is None:
            return False
        elapsed = (_now_utc() - self.created_at).total_seconds()
        return elapsed > self.ttl

    @property
    def age_seconds(self) -> float:
        """Seconds since creation."""
        if self.created_at is None:
            return 0.0
        return (_now_utc() - self.created_at).total_seconds()

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dict for repository persistence."""
        return {
            "id": self.id.value,
            "memory_type": self.memory_type,
            "namespace": self.namespace,
            "content": self.content,
            "importance": self.importance,
            "confidence": self.confidence,
            "ttl": self.ttl,
            "state": self.state.value,
            "source": self.source,
            "owner": self.owner,
            "tags": ",".join(self.tags) if self.tags else "",
            "metadata": str(self.metadata) if self.metadata else "{}",
            "correlation_id": self.correlation_id,
            "created_at": self.created_at.isoformat() if self.created_at else "",
            "updated_at": self.updated_at.isoformat() if self.updated_at else "",
            "accessed_at": self.accessed_at.isoformat() if self.accessed_at else "",
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "forgotten_at": self.forgotten_at.isoformat() if self.forgotten_at else None,
            "deleted_at": self.deleted_at.isoformat() if self.deleted_at else None,
            "access_count": self.access_count,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        """Create a Memory from a flat dict (repository read path)."""
        memory = cls(
            memory_id=MemoryId(data.get("id", "")),
            memory_type=data.get("memory_type", MemoryType.SHORT_TERM.value),
            namespace=data.get("namespace", "default"),
            content=data.get("content", ""),
            importance=float(data.get("importance", 0.5)),
            confidence=float(data.get("confidence", 1.0)),
            ttl=data.get("ttl"),
            state=MemoryState(data.get("state", "active")),
            source=data.get("source", "manual"),
            owner=data.get("owner", "system"),
            tags=data.get("tags", "").split(",") if data.get("tags") else [],
            metadata={},
            correlation_id=data.get("correlation_id", ""),
        )

        # Parse metadata from JSON string
        raw_metadata = data.get("metadata", "{}")
        if isinstance(raw_metadata, str):
            import json
            try:
                memory.metadata = json.loads(raw_metadata)
            except (json.JSONDecodeError, TypeError):
                memory.metadata = {}

        # Restore timestamps
        for field in ("created_at", "updated_at", "accessed_at", "archived_at", "forgotten_at", "deleted_at"):
            val = data.get(field)
            if val:
                try:
                    setattr(memory, field, datetime.fromisoformat(val))
                except (ValueError, TypeError):
                    pass

        memory.access_count = int(data.get("access_count", 0))
        memory.version = int(data.get("version", 1))
        memory.relationship_ids = []

        return memory

    def __repr__(self) -> str:
        return (
            f"Memory(id={self.id.value[:8]}..., "
            f"type={self.memory_type}, "
            f"state={self.state.value}, "
            f"importance={self.importance:.2f})"
        )
