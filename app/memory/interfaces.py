"""Memory-layer interfaces and protocols.

All memory abstractions live here.  Implementations reside in subpackages
and import only these interfaces (plus domain models).  Higher layers
import interfaces, never concrete implementations.
"""

from __future__ import annotations

import abc
from typing import Any

from app.memory.memory import Memory, MemoryId, MemoryState
from app.storage.interfaces import Page, PaginationParams, SortField


# ---------------------------------------------------------------------------
# Search / Query types
# ---------------------------------------------------------------------------


class MemoryQuery:
    """Structured query for retrieving memories.

    Every field is optional.  An empty query returns all active memories.
    """

    def __init__(
        self,
        *,
        memory_types: list[str] | None = None,
        namespaces: list[str] | None = None,
        states: list[MemoryState] | None = None,
        tags: list[str] | None = None,
        content_search: str | None = None,
        sources: list[str] | None = None,
        owners: list[str] | None = None,
        min_importance: float | None = None,
        max_importance: float | None = None,
        correlation_id: str | None = None,
        created_after: str | None = None,
        created_before: str | None = None,
        accessed_after: str | None = None,
        only_expired: bool | None = None,
    ) -> None:
        self.memory_types = memory_types
        self.namespaces = namespaces
        self.states = states
        self.tags = tags
        self.content_search = content_search
        self.sources = sources
        self.owners = owners
        self.min_importance = min_importance
        self.max_importance = max_importance
        self.correlation_id = correlation_id
        self.created_after = created_after
        self.created_before = created_before
        self.accessed_after = accessed_after
        self.only_expired = only_expired


# ---------------------------------------------------------------------------
# MemoryService (primary facade)
# ---------------------------------------------------------------------------


class MemoryService(abc.ABC):
    """Primary interface for memory operations.

    Every memory operation goes through this interface.  Implementations
    add caching, event emission, and telemetry around the core logic.
    """

    @abc.abstractmethod
    async def create(self, memory: Memory) -> Memory:
        """Persist a new memory and emit MemoryCreated."""

    @abc.abstractmethod
    async def get(self, memory_id: MemoryId) -> Memory | None:
        """Retrieve a memory by ID (touching access tracking)."""

    @abc.abstractmethod
    async def update(self, memory: Memory) -> Memory:
        """Update an existing memory and emit MemoryUpdated."""

    @abc.abstractmethod
    async def delete(self, memory_id: MemoryId) -> bool:
        """Hard-delete a memory by ID."""

    @abc.abstractmethod
    async def transition_state(
        self,
        memory_id: MemoryId,
        target: MemoryState,
        reason: str = "",
    ) -> Memory | None:
        """Transition a memory to a new state."""

    @abc.abstractmethod
    async def list(
        self,
        *,
        pagination: PaginationParams | None = None,
        sort: list[SortField] | None = None,
    ) -> Page[Memory]:
        """Paginated listing of all active memories."""

    @abc.abstractmethod
    async def count(self, state: MemoryState | None = None) -> int:
        """Count memories, optionally filtered by state."""


# ---------------------------------------------------------------------------
# MemorySearchService
# ---------------------------------------------------------------------------


class MemorySearchService(abc.ABC):
    """Structured retrieval with filtering, ranking, and namespace isolation."""

    @abc.abstractmethod
    async def search(
        self,
        query: MemoryQuery,
        *,
        pagination: PaginationParams | None = None,
    ) -> Page[Memory]:
        """Execute a structured query and return matching memories."""

    @abc.abstractmethod
    async def search_by_importance(
        self,
        *,
        namespace: str | None = None,
        min_importance: float = 0.0,
        limit: int = 10,
    ) -> list[Memory]:
        """Return the highest-importance memories in a namespace."""

    @abc.abstractmethod
    async def search_by_tag(
        self,
        tag: str,
        *,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        """Return memories with a specific tag."""

    @abc.abstractmethod
    async def search_temporal(
        self,
        *,
        after: str | None = None,
        before: str | None = None,
        namespace: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        """Return memories within a time range."""


# ---------------------------------------------------------------------------
# MemoryGarbageCollector
# ---------------------------------------------------------------------------


class GCResult:
    """Result of a garbage collector sweep."""

    def __init__(
        self,
        *,
        archived: int = 0,
        forgotten: int = 0,
        deleted: int = 0,
    ) -> None:
        self.archived = archived
        self.forgotten = forgotten
        self.deleted = deleted

    @property
    def total(self) -> int:
        return self.archived + self.forgotten + self.deleted

    def to_dict(self) -> dict[str, Any]:
        return {
            "archived": self.archived,
            "forgotten": self.forgotten,
            "deleted": self.deleted,
            "total": self.total,
        }


class MemoryGarbageCollector(abc.ABC):
    """Policy-driven memory cleanup: TTL expiration, importance threshold,
    namespace capacity enforcement, and grace-period purge."""

    @abc.abstractmethod
    async def collect(self) -> GCResult:
        """Run one sweep of the garbage collector."""

    @abc.abstractmethod
    async def count_candidates(self) -> int:
        """Return how many memories would be affected by the next sweep."""


# ---------------------------------------------------------------------------
# MemorySnapshotService
# ---------------------------------------------------------------------------


class MemorySnapshot(abc.ABC):
    """A point-in-time snapshot of all memories."""

    @property
    @abc.abstractmethod
    def snapshot_id(self) -> str: ...

    @abc.abstractmethod
    async def restore(self) -> int:
        """Restore all memories to this snapshot's state.  Returns count."""

    @abc.abstractmethod
    def to_dict(self) -> dict[str, Any]: ...


class MemorySnapshotService(abc.ABC):
    """Checkpoint/restore for memory state."""

    @abc.abstractmethod
    async def create_snapshot(self, label: str = "") -> MemorySnapshot:
        """Capture a point-in-time snapshot of all memories."""

    @abc.abstractmethod
    async def list_snapshots(self) -> list[dict[str, Any]]:
        """List all snapshots with metadata."""


# ---------------------------------------------------------------------------
# MemoryCompressor
# ---------------------------------------------------------------------------


class CompressionResult:
    """Result of compressing a list of memories."""

    def __init__(
        self,
        *,
        compressed: list[Memory],
        original_count: int,
        compressed_count: int,
        strategy: str = "",
    ) -> None:
        self.compressed = compressed
        self.original_count = original_count
        self.compressed_count = compressed_count
        self.strategy = strategy

    @property
    def ratio(self) -> float:
        if self.original_count == 0:
            return 1.0
        return self.compressed_count / self.original_count


class MemoryCompressor(abc.ABC):
    """Compression strategies for memory sets.

    Must preserve provenance — each compressed memory tracks which
    original memories contributed to it.
    """

    @abc.abstractmethod
    async def compress(
        self,
        memories: list[Memory],
        *,
        target_count: int | None = None,
        strategy: str = "dedup",
    ) -> CompressionResult:
        """Compress *memories* to *target_count* using *strategy*."""


# ---------------------------------------------------------------------------
# MemoryGraph
# ---------------------------------------------------------------------------


class MemoryGraph(abc.ABC):
    """Relationship management for memories.

    Wraps the Phase 2 GraphStore with memory-specific query methods.
    """

    @abc.abstractmethod
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """Create a relationship between two memories."""

    @abc.abstractmethod
    async def get_related(
        self,
        memory_id: str,
        *,
        rel_type: str | None = None,
        direction: str = "both",
        max_depth: int = 1,
    ) -> list[Memory]:
        """Return memories related to *memory_id*."""

    @abc.abstractmethod
    async def remove_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str | None = None,
    ) -> None:
        """Remove a relationship (or all between two memories)."""

    @abc.abstractmethod
    async def propagate_importance(
        self,
        memory_id: str,
        *,
        decay: float = 0.5,
        max_depth: int = 3,
    ) -> int:
        """Propagate importance to related memories with *decay* per hop.

        Returns the number of memories updated.
        """
