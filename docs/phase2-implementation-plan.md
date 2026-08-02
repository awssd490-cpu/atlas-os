# Phase 2 Implementation Plan — Persistence & Memory Foundation

## Folder Structure

```text
app/
├── storage/
│   ├── __init__.py
│   ├── interfaces.py          ← All storage protocols/ABCs
│   ├── errors.py              ← Storage-specific errors
│   │
│   ├── engine/
│   │   ├── __init__.py
│   │   ├── base.py            ← StorageEngine ABC
│   │   └── sqlite.py          ← SQLiteStorageEngine
│   │
│   ├── connection/
│   │   ├── __init__.py
│   │   ├── protocol.py        ← Connection protocols
│   │   └── sqlite.py          ← SQLiteConnection
│   │
│   ├── repository/
│   │   ├── __init__.py
│   │   ├── base.py            ← BaseRepository, BaseRepositoryImpl
│   │   ├── types.py           ← Pagination, Sort, Filter types
│   │   └── sqlite.py          ← SQLiteRepositoryMixin
│   │
│   ├── transaction/
│   │   ├── __init__.py
│   │   ├── unit_of_work.py    ← UnitOfWork interface + impl
│   │   └── sqlite.py          ← SQLiteUnitOfWork
│   │
│   ├── migration/
│   │   ├── __init__.py
│   │   ├── manager.py         ← MigrationManager
│   │   └── sqlite.py          ← SQLiteMigrationEngine
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   ├── service.py         ← CacheService interface
│   │   └── memory.py          ← MemoryCache
│   │
│   ├── event_store/
│   │   ├── __init__.py
│   │   ├── service.py         ← EventStore interface + impl
│   │   └── subscriber.py      ← EventBus subscriber
│   │
│   ├── vector/
│   │   ├── __init__.py
│   │   ├── interfaces.py      ← VectorStore interface
│   │   └── memory.py          ← InMemoryVectorStore
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── interfaces.py      ← GraphStore interface
│   │   └── memory.py          ← InMemoryGraphStore
│   │
│   ├── object_store/
│   │   ├── __init__.py
│   │   ├── interfaces.py      ← ObjectStore interface
│   │   └── local.py           ← LocalFileObjectStore
│   │
│   ├── versions/
│   │   ├── __init__.py
│   │   └── service.py         ← OptimisticLocking, VersionedEntity
│   │
│   └── module.py              ← StorageModule (kernel module)
│
└── config/
    └── settings.py            ← Extended with StorageConfig
```

## Build Order (16 steps)

Dependencies flow downward. Each step produces a file + its unit tests.

### Step 1: Storage config (`app/config/settings.py`)

- Add `StorageConfig` model to `AtlasSettings`
- SQLite path, cache TTL defaults, vector dimension defaults

### Step 2: Storage errors (`app/storage/errors.py`)

- `StorageError`, `ConnectionError`, `MigrationError`, `RecordNotFoundError`, `VersionConflictError`, `CacheError`

### Step 3: Storage interfaces (`app/storage/interfaces.py`)

- ALL protocol definitions in one file (dependencies stay minimal)
- `Connection`, `SQLConnection`, `KVConnection`
- `StorageEngine`, `CacheService`
- `EventStore`, `VectorStore`, `GraphStore`, `ObjectStore`
- `UnitOfWork`, `UnitOfWorkFactory`
- `MigrationManager`, `Migration`
- Domain types: `Page`, `SortOrder`, `FilterCondition`

### Step 4: Cache — MemoryCache (`app/storage/cache/memory.py`)

- In-memory dict + asyncio TTL sweeper
- `MemoryCache(CacheService)`
- Tests: get/set/delete, TTL expiration, pattern invalidation, hit/miss counting

### Step 5: Connection protocol + SQLite connection (`app/storage/connection/`)

- `app/storage/connection/protocol.py`: `SQLConnection` protocol with `execute`, `fetchone`, `fetchall`, `executemany`, `execute_script`
- `app/storage/connection/sqlite.py`: `SQLiteConnection` wrapping stdlib sqlite3 via thread pool
- Tests: connect, execute, fetch, parameterized queries, context manager

### Step 6: StorageEngine — SQLite (`app/storage/engine/`)

- `app/storage/engine/base.py`: `StorageEngine` ABC — `connect()`, `disconnect()`, `is_healthy`
- `app/storage/engine/sqlite.py`: `SQLiteStorageEngine`
- Tests: lifecycle, connection reuse, connection isolation

### Step 7: Repository — Base implementation (`app/storage/repository/`)

- `app/storage/repository/base.py`: `BaseRepository[TModel, TId]` with full CRUD
- `app/storage/repository/types.py`: `Page[T]`, `SortField`, `Filter`, `PaginationParams`
- `app/storage/repository/sqlite.py`: `SQLiteRepositoryMixin` — SQL generation helpers for pagination, filter, sort
- Concrete: `SqliteBaseRepository(TModel, TId, SQLConnection)` that implements BaseRepository using the mixin
- Tests: add, get, update, delete, list with pagination, soft delete, optimistic locking

### Step 8: Migration system (`app/storage/migration/`)

- `Migration` ABC: `version`, `up(conn)`, `down(conn)`
- `MigrationManager`: tracking table, ordered execution, rollback, history
- Initial migration: `V001_initial_schema` (event_store, migration_history)
- Tests: apply, rollback, idempotent re-apply, failure handling

### Step 9: Unit of Work (`app/storage/transaction/`)

- `app/storage/transaction/unit_of_work.py`: `UnitOfWork[TConnection]` with commit/rollback/flush
- `SqliteUnitOfWork` using SQLiteConnection
- `UnitOfWorkFactory` for UoW-per-operation scoping
- Tests: commit persists, rollback discards, context manager, nested error handling

### Step 10: Event Store (`app/storage/event_store/`)

- `EventStoreService`: append, stream_by_{type,correlation,source}, stream_by_time_range, replay_all
- `EventBusSubscriber`: subscribes to ALL Event types on bus, persists via EventStoreService
- SQLite implementation using the V001 schema
- Tests: append, stream, replay, time range, correlation stream

### Step 11: Vector Store — In-memory (`app/storage/vector/`)

- `VectorStore` interface: upsert_vectors, search, delete, list_namespaces
- `InMemoryVectorStore` using numpy-free cosine similarity (pure math)
- Tests: insert, search returns nearest, metadata filtering, namespace isolation

### Step 12: Graph Store — In-memory (`app/storage/graph/`)

- `GraphStore` interface: create_node, create_relationship, query, traversal
- `InMemoryGraphStore` with adjacency-list-like structure
- Tests: nodes, edges, property queries, traversal, label filtering

### Step 13: Object Store — Local filesystem (`app/storage/object_store/`)

- `ObjectStore` interface: upload, download, delete, exists, list, get_metadata
- `LocalFileObjectStore` — file-backed with checksum (SHA-256)
- Tests: store/retrieve binary, streaming, checksum verification

### Step 14: Storage Module (`app/storage/module.py`)

- `StorageModule(Module)` implementing the kernel lifecycle
- `initialize()` — read config, create engine
- `start()` — connect, run migrations, register event subscriber
- `health()` — check engine, run integrity check
- Declares capabilities: `"storage.sql"`, `"storage.kv"`, `"storage.event_store"`
- Tests: boot with kernel, health reporting, shutdown cleanup

### Step 15: Repository implementations for Phase 1 types

- `EventRepository` — CRUD for persisted events (backed by EventStore)
- Demonstrate how domain repositories integrate with the storage system
- Tests: CRUD, pagination, correlation lookup

### Step 16: Integration tests

- Full-stack test: kernel → storage module → SQLite → event store → replay
- Multi-repository UoW transaction test
- Migration up/down round-trip test
- Concurrent read/write test
- Performance regression guard (queries under threshold)

## Test Standards

- Unit tests: pytest-asyncio, `asyncio_mode=auto`
- Repository tests: use in-memory connection (SQLite `:memory:`)
- No external services required for any test
- Every storage interface tested against at least one implementation
- Mutation tests for optimistic locking, soft delete, TTL cache

## Explicitly Deferred

| Feature | Deferred to |
|---------|-------------|
| PostgreSQL backend | Phase 3 |
| Redis KV backend | Phase 3 |
| Redis cache backend | Phase 3 |
| Qdrant backend | Phase 4 |
| Neo4j backend | Phase 5 |
| S3 blob backend | Phase 6 |
| Distributed transactions | Phase 10 |
| Full-text search index | Phase 3 |
| Connection pooling (pool_size > 1) | PostgreSQL Phase 3 |
