"""Storage-layer interfaces, protocols, and domain types.

All storage abstractions live here.  Implementations reside in their
respective subpackages and import *only* these interfaces (plus the
error hierarchy).  Business logic imports interfaces, never concrete
backends.
"""

from __future__ import annotations

import abc
import enum
from collections.abc import AsyncIterator
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


class SortOrder(str, enum.Enum):
    ASC = "asc"
    DESC = "desc"


class SortField:
    """A single sort field specification."""

    __slots__ = ("field", "order")

    def __init__(self, field: str, order: SortOrder = SortOrder.ASC) -> None:
        self.field = field
        self.order = order


class FilterOperator(str, enum.Enum):
    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    LIKE = "like"
    ILIKE = "ilike"
    IS_NULL = "is_null"
    NOT_NULL = "not_null"


class FilterCondition:
    """A single filter condition on a field."""

    __slots__ = ("field", "operator", "value")

    def __init__(
        self,
        field: str,
        operator: FilterOperator = FilterOperator.EQ,
        value: Any = None,
    ) -> None:
        self.field = field
        self.operator = operator
        self.value = value


class PaginationParams:
    """Offset/limit pagination."""

    __slots__ = ("offset", "limit")

    def __init__(self, offset: int = 0, limit: int = 100) -> None:
        self.offset = max(0, offset)
        self.limit = max(1, min(limit, 1000))


T = TypeVar("T")


class Page(Generic[T]):
    """A single page of results."""

    __slots__ = ("items", "total", "offset", "limit")

    def __init__(
        self,
        items: list[T],
        total: int,
        offset: int = 0,
        limit: int = 100,
    ) -> None:
        self.items = items
        self.total = total
        self.offset = offset
        self.limit = limit

    @property
    def has_next(self) -> bool:
        return (self.offset + self.limit) < self.total

    @property
    def has_previous(self) -> bool:
        return self.offset > 0


# ---------------------------------------------------------------------------
# Connection protocols
# ---------------------------------------------------------------------------


class Row(dict[str, Any]):
    """A single row returned from a query, dict-like with optional attribute access."""

    __slots__ = ()


@runtime_checkable
class SQLConnection(Protocol):
    """Minimal SQL connection interface.

    Every SQL backend (SQLite, PostgreSQL, etc.) implements this protocol.
    Methods map to the fundamental database I/O primitives.
    """

    async def execute(self, sql: str, params: dict[str, Any] | list | None = None) -> None:
        """Execute a SQL statement that does not return rows (INSERT, UPDATE, DELETE, DDL)."""

    async def fetchone(
        self,
        sql: str,
        params: dict[str, Any] | list | None = None,
    ) -> Row | None:
        """Execute a query and return the first row, or ``None``."""

    async def fetchall(
        self,
        sql: str,
        params: dict[str, Any] | list | None = None,
    ) -> list[Row]:
        """Execute a query and return all matching rows."""

    async def executemany(self, sql: str, params: list[dict[str, Any] | list]) -> None:
        """Execute the same statement for every parameter set."""

    async def execute_script(self, sql: str) -> None:
        """Execute a multi-statement script (DDL batches, migrations)."""

    async def close(self) -> None:
        """Release the connection."""


# ---------------------------------------------------------------------------
# Storage engine
# ---------------------------------------------------------------------------


class StorageEngine(abc.ABC):
    """Manages a backend storage system's lifecycle and connection pool."""

    @abc.abstractmethod
    async def connect(self) -> None:
        """Establish the backing connection(s)."""

    @abc.abstractmethod
    async def disconnect(self) -> None:
        """Close all connections and release resources."""

    @abc.abstractmethod
    async def connection(self) -> SQLConnection:
        """Return a connection from the pool.

        The caller *must* close the connection when done (or use as a
        context manager if supported).
        """

    @abc.abstractmethod
    async def is_healthy(self) -> bool:
        """Return ``True`` when the backend is reachable and functional."""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class CacheService(abc.ABC):
    """Best-effort key-value cache.

    CACHE IS NEVER AUTHORITATIVE.  A miss must produce correct results
    from the backing store.  Cache is safe to clear at any time.
    """

    @abc.abstractmethod
    async def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` on miss."""

    @abc.abstractmethod
    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store *value* under *key* with an optional TTL (seconds).

        Args:
            key: Cache key.
            value: Any JSON-serializable value.
            ttl: Time-to-live in seconds.  ``None`` uses the default.
        """

    @abc.abstractmethod
    async def delete(self, key: str) -> None:
        """Remove *key* from the cache (no-op if absent)."""

    @abc.abstractmethod
    async def invalidate_pattern(self, pattern: str) -> None:
        """Remove all keys matching a glob-style *pattern*."""

    @abc.abstractmethod
    async def clear(self) -> None:
        """Remove every entry from the cache."""

    @abc.abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return cache hit/miss/size counters."""


# ---------------------------------------------------------------------------
# Event store
# ---------------------------------------------------------------------------


class StoredEvent:
    """A single persisted event, deserialized from the event store."""

    __slots__ = (
        "id",
        "event_type",
        "version",
        "source",
        "correlation_id",
        "target",
        "timestamp",
        "payload",
        "metadata",
    )

    def __init__(
        self,
        *,
        id: str,
        event_type: str,
        version: int,
        source: str,
        correlation_id: str,
        target: str,
        timestamp: str,
        payload: str,
        metadata: str,
    ) -> None:
        self.id = id
        self.event_type = event_type
        self.version = version
        self.source = source
        self.correlation_id = correlation_id
        self.target = target
        self.timestamp = timestamp
        self.payload = payload
        self.metadata = metadata


class EventStore(abc.ABC):
    """Append-only event persistence store.

    Events are written sequentially and never mutated.  This is the
    foundation for audit, replay, and future event sourcing.
    """

    @abc.abstractmethod
    async def append(self, event: Any) -> None:
        """Persist a single event (serialized to JSON internally)."""

    @abc.abstractmethod
    async def stream_by_type(self, event_type: str) -> list[StoredEvent]:
        """Return all events of a given type, in chronological order."""

    @abc.abstractmethod
    async def stream_by_correlation(self, correlation_id: str) -> list[StoredEvent]:
        """Return all events sharing a correlation ID, in chronological order."""

    @abc.abstractmethod
    async def stream_by_source(self, source: str) -> list[StoredEvent]:
        """Return all events from a given source module, in chronological order."""

    @abc.abstractmethod
    async def stream_by_time_range(
        self,
        start: str,
        end: str,
    ) -> list[StoredEvent]:
        """Return events within an ISO-8601 time range (inclusive)."""

    @abc.abstractmethod
    async def replay_all(self) -> list[StoredEvent]:
        """Return every stored event, in chronological order (for replay)."""

    @abc.abstractmethod
    async def count(self) -> int:
        """Return the total number of persisted events."""


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


class VectorRecord:
    """A single vector with its metadata."""

    __slots__ = ("id", "vector", "metadata", "namespace")

    def __init__(
        self,
        *,
        id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        namespace: str = "default",
    ) -> None:
        self.id = id
        self.vector = vector
        self.metadata = metadata or {}
        self.namespace = namespace


class SearchResult:
    """A vector search result with similarity score."""

    __slots__ = ("id", "score", "metadata")

    def __init__(
        self,
        *,
        id: str,
        score: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.score = score
        self.metadata = metadata or {}


class VectorStore(abc.ABC):
    """Vector storage for embedding-based similarity search.

    No embedding model integration yet — this is purely the storage side.
    """

    @abc.abstractmethod
    async def upsert(self, record: VectorRecord) -> None:
        """Insert or update a vector record."""

    @abc.abstractmethod
    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 10,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Nearest-neighbor search.

        Args:
            vector: The query embedding.
            limit: Maximum results.
            namespace: Optional namespace to restrict search.
            filter: Optional metadata filter (backend-specific).

        Returns:
            Results ordered by similarity (highest score first).
        """

    @abc.abstractmethod
    async def delete(self, id: str, *, namespace: str | None = None) -> None:
        """Remove a vector by ID."""

    @abc.abstractmethod
    async def list_ids(self, *, namespace: str | None = None) -> list[str]:
        """Return all vector IDs in a namespace."""

    @abc.abstractmethod
    async def count(self, *, namespace: str | None = None) -> int:
        """Return the number of vectors in a namespace."""


# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------


class GraphNode:
    """A node in the knowledge graph."""

    __slots__ = ("id", "labels", "properties")

    def __init__(
        self,
        *,
        id: str,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.labels = labels or []
        self.properties = properties or {}


class GraphRelationship:
    """A directed relationship between two nodes."""

    __slots__ = ("id", "type", "source_id", "target_id", "properties")

    def __init__(
        self,
        *,
        id: str,
        type: str,
        source_id: str,
        target_id: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        self.id = id
        self.type = type
        self.source_id = source_id
        self.target_id = target_id
        self.properties = properties or {}


class GraphStore(abc.ABC):
    """Knowledge graph storage.

    Supports node/relationship CRUD, label-based queries, and BFS/DFS
    traversal.  No Cypher/SPARQL implementation yet.
    """

    @abc.abstractmethod
    async def create_node(self, node: GraphNode) -> GraphNode:
        """Create a node.  Raises if ``id`` already exists."""

    @abc.abstractmethod
    async def get_node(self, id: str) -> GraphNode | None:
        """Retrieve a node by ID, or ``None``."""

    @abc.abstractmethod
    async def update_node(self, node: GraphNode) -> GraphNode:
        """Replace properties and labels on an existing node."""

    @abc.abstractmethod
    async def delete_node(self, id: str) -> None:
        """Delete a node and all its relationships."""

    @abc.abstractmethod
    async def find_nodes(
        self,
        *,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
    ) -> list[GraphNode]:
        """Find nodes by label and/or property filter."""

    @abc.abstractmethod
    async def create_relationship(self, rel: GraphRelationship) -> GraphRelationship:
        """Create a relationship between two nodes."""

    @abc.abstractmethod
    async def get_relationships(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        type: str | None = None,
    ) -> list[GraphRelationship]:
        """Query relationships by source, target, and/or type."""

    @abc.abstractmethod
    async def traverse(
        self,
        start_id: str,
        *,
        direction: str = "outgoing",
        max_depth: int = 3,
        relationship_types: list[str] | None = None,
    ) -> list[GraphNode]:
        """Traverse the graph starting from *start_id*.


        Args:
            start_id: Starting node ID.
            direction: ``"outgoing"``, ``"incoming"``, or ``"both"``.
            max_depth: Maximum traversal depth.
            relationship_types: Only follow these relationship types.

        Returns:
            All reachable nodes (excluding the start node).
        """


# ---------------------------------------------------------------------------
# Object store
# ---------------------------------------------------------------------------


class ObjectMetadata:
    """Metadata for a stored object."""

    __slots__ = ("key", "size", "content_type", "checksum_sha256", "created_at", "metadata")

    def __init__(
        self,
        *,
        key: str,
        size: int = 0,
        content_type: str = "application/octet-stream",
        checksum_sha256: str | None = None,
        created_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.key = key
        self.size = size
        self.content_type = content_type
        self.checksum_sha256 = checksum_sha256
        self.created_at = created_at
        self.metadata = metadata or {}


class ObjectStore(abc.ABC):
    """Binary object storage (files, blobs).

    Supports streaming, checksums, and metadata.
    """

    @abc.abstractmethod
    async def upload(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ObjectMetadata:
        """Store a binary object.

        Returns the object's metadata after storage.
        """

    @abc.abstractmethod
    async def download(self, key: str) -> bytes | None:
        """Retrieve a binary object by key, or ``None``."""

    @abc.abstractmethod
    async def download_stream(self, key: str, chunk_size: int = 65536) -> AsyncIterator[bytes]:
        """Stream a binary object in chunks."""
        ...

    @abc.abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete an object.  Returns ``True`` if it existed."""

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        """Return ``True`` when the object exists."""

    @abc.abstractmethod
    async def list_keys(self, prefix: str = "") -> list[str]:
        """Return all object keys with an optional *prefix*."""

    @abc.abstractmethod
    async def get_metadata(self, key: str) -> ObjectMetadata | None:
        """Return an object's metadata without downloading it."""


# ---------------------------------------------------------------------------
# Repository (domain CRUD)
# ---------------------------------------------------------------------------

TModel = TypeVar("TModel")
TId = TypeVar("TId")


class BaseRepository(abc.ABC, Generic[TModel, TId]):
    """Generic CRUD repository.

    Domain repositories extend this and add domain-specific query methods.
    """

    @abc.abstractmethod
    async def add(self, model: TModel) -> TModel:
        """Insert a new record and return it (with generated fields populated)."""

    @abc.abstractmethod
    async def get(self, id: TId) -> TModel | None:
        """Retrieve by primary key, or ``None``."""

    @abc.abstractmethod
    async def update(self, model: TModel) -> TModel:
        """Update an existing record and return the updated version."""

    @abc.abstractmethod
    async def delete(self, id: TId) -> bool:
        """Delete by primary key.  Returns ``True`` if a row was removed."""

    @abc.abstractmethod
    async def list(
        self,
        *,
        pagination: PaginationParams | None = None,
        sort: list[SortField] | None = None,
        filters: list[FilterCondition] | None = None,
    ) -> Page[TModel]:
        """Paginated listing with sorting and filtering."""

    @abc.abstractmethod
    async def count(self, filters: list[FilterCondition] | None = None) -> int:
        """Count records matching optional filters."""

    @abc.abstractmethod
    async def add_batch(self, models: list[TModel]) -> list[TModel]:
        """Insert multiple records in a single operation."""


# ---------------------------------------------------------------------------
# Transaction / Unit of Work
# ---------------------------------------------------------------------------


class UnitOfWork(abc.ABC):
    """Transaction boundary.

    Multiple repository operations share a single UoW.  Commit persists
    all changes atomically; rollback discards them.
    """

    @abc.abstractmethod
    async def commit(self) -> None:
        """Persist all changes made within this UoW."""

    @abc.abstractmethod
    async def rollback(self) -> None:
        """Discard all changes made within this UoW."""

    @abc.abstractmethod
    async def flush(self) -> None:
        """Emit pending SQL without committing (useful for getting generated keys)."""

    @abc.abstractmethod
    async def __aenter__(self) -> "UnitOfWork": ...

    @abc.abstractmethod
    async def __aexit__(self, *args: Any) -> None: ...


class UnitOfWorkFactory(abc.ABC):
    """Creates scoped UnitOfWork instances."""

    @abc.abstractmethod
    async def create(self) -> UnitOfWork:
        """Create a new UnitOfWork backed by a fresh connection."""


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------


class Migration(abc.ABC):
    """A single, reversible database migration."""

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Unique version identifier (e.g. ``"V001"``, ``"V002"``)."""

    @abc.abstractmethod
    async def up(self, connection: SQLConnection) -> None:
        """Apply the migration (create tables, add columns, etc.)."""

    @abc.abstractmethod
    async def down(self, connection: SQLConnection) -> None:
        """Revert the migration (DROP tables, remove columns, etc.)."""


class MigrationManager(abc.ABC):
    """Orchestrates schema migrations."""

    @abc.abstractmethod
    async def initialize(self, connection: SQLConnection) -> None:
        """Create the migration tracking table if it doesn't exist."""

    @abc.abstractmethod
    async def apply(self, connection: SQLConnection, migration: Migration) -> None:
        """Apply *migration* and record it in the history."""

    @abc.abstractmethod
    async def rollback(self, connection: SQLConnection, migration: Migration) -> None:
        """Rollback *migration* and remove it from the history."""

    @abc.abstractmethod
    async def has_been_applied(self, connection: SQLConnection, version: str) -> bool:
        """Return ``True`` when the migration has already been applied."""

    @abc.abstractmethod
    async def history(self, connection: SQLConnection) -> list[dict[str, Any]]:
        """Return all applied migrations (ordered by version)."""

    @abc.abstractmethod
    async def pending(
        self,
        connection: SQLConnection,
        migrations: list[Migration],
    ) -> list[Migration]:
        """Return migrations from *migrations* that have not been applied, in version order."""
