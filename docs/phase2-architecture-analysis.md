# Phase 2 Architecture Analysis: Persistence & Memory Foundation

## 1. Integration with Phase 1

### What Phase 2 consumes

| Phase 1 Component | How Phase 2 Uses It |
|---|---|
| **KernelContext** | StorageModule receives context via `initialize()`. Storage services are resolved via `context.resolve()`. |
| **DI Container** | All storage implementations registered as singletons: `StorageEngine`, `CacheService`, `EventStore`, domain repositories. |
| **ConfigService** | Storage configuration via `config.get("storage.sqlite.path")`. Storage config schema added to `AtlasSettings`. |
| **TelemetryService** | Every storage operation instruments latency, errors. Cache hits/misses recorded. Transaction duration tracked. |
| **EventBus** | EventStore subscribes to ALL events and persists them. Storage lifecycle events emitted. |
| **Module Lifecycle** | `StorageModule` uses `initialize(ctx)`, `start()` (connect + migrate), `shutdown()` (disconnect). |
| **Capability Registry** | StorageModule declares: `"storage.sql"`, `"storage.kv"`, `"storage.event_store"`, etc. |
| **LoggingService** | Every query and migration step logged with structured context. |
| **Health API** | StorageModule returns connection pool status, replication lag, migration state via `health()`. |

### What Phase 2 extends

- **`AtlasSettings`**: Adds `storage:` section with SQLite path, pool size, etc.
- **`KernelContext`**: *Not* extended (storage accessed via `context.resolve(StorageEngine)` to keep the context interface small).

## 2. Architectural Evaluation

### Design Domain: Layered Storage

```
 Domain Code
    ↕  (domain operations, never SQL)
 Domain Repositories
    ↕  (generic CRUD, pagination, filtering)
 UnitOfWork  ←  Transaction Management
    ↕
 StorageEngine  ←  Connection Pool, Lifecycle
    ↕
 SQLite / PostgreSQL / Redis  ←  Backend
```

### Rejected Alternatives

| Alternative | Rejected Because |
|---|---|
| **ORM-first (SQLAlchemy model inheritance)** | Violates Clean Architecture — domain inherits from infrastructure. Every model is coupled to SQLAlchemy. Backend swap requires migration. |
| **Active Record pattern** | Each entity knows how to save itself. Violates SRP. Testing requires real DB. |
| **Generic Repository with query builder** | Produces leaky abstractions that try to be all databases and fail at all of them. Complex queries require stringly-typed DSLs. |
| **Repository-per-backend** | Every domain concept needs N implementations. Only justified when backends are truly different (SQL vs DynamoDB). |

### Chosen: Repository with explicit query methods + base CRUD

```python
class MemoryRepository(BaseRepository[Memory, UUID]):
    async def find_by_agent(self, agent_id: UUID) -> list[Memory]: ...
    async def search_similar(self, embedding: list[float], limit: int) -> list[Memory]: ...
```

**Rationale**: Specific query methods are IDE-findable, type-safe, optimizable per backend, and avoid building a mini-ORM. The base class handles the 80% case (CRUD, pagination, soft delete).

## 3. Key Design Decisions

### 3.1 Session Abstraction over Raw DB-API

The `Connection` protocol exposes `execute()`, `fetchone()`, `fetchall()` — the fundamental primitives every database operation reduces to. Repositories build on these. This is NOT a leak; it's the minimal surface that guarantees backend replaceability.

A driver that can implement `execute(sql, params) → rows` can implement the Connection protocol. SQLite, PostgreSQL (asyncpg), Redis (via command interface) — all map to this.

### 3.2 Async SQLite via Thread Pool (not aiosqlite)

Using `asyncio.to_thread()` with stdlib `sqlite3` instead of `aiosqlite`. Rationale: zero dependencies. SQLite is serialized by design (single-writer), so threading adds no contention that wouldn't exist anyway. When PostgreSQL arrives, `asyncpg` provides true async I/O behind the same interface.

### 3.3 Event Store as Bus Subscriber, Not Coupled Bus

The EventStore is a subscriber on the EventBus, not a modification to the bus itself. This preserves the lightweight in-process bus and makes EventStore replaceable independently. Replay reads persisted events and re-publishes through the bus.

### 3.4 Migration as Python Methods (not SQL files)

Migrations are Python classes that implement `up(conn)` and `down(conn)`. This lets them use the typed `Connection` interface, access config, and embed logic (data migration, backfills, integrity checks). SQL-file support can be added in Phase 3+ without breaking existing migrations.

### 3.5 Cache as Separate Concern

Cache has a distinct interface (`get`, `set`, `delete`, `invalidate`) with different guarantees (best-effort, TTL-bound) than storage (durable, transactional). Conflating them causes the "stale cache in a transaction" problem.

## 4. What's Deferred (Phase 3+)

| Feature | Justification |
|---|---|
| Full-text search | Requires indexing engine; Phase 3 search needs |
| Document store | KV + blob covers Phase 2; dedicated doc store when needed |
| Distributed transactions | Requires consensus protocol (2PC, Saga); Phase 10+ |
| Connection pooling | SQLite doesn't benefit; PostgreSQL Phase 3 |
| Query builder DSL | Premature abstraction; add when repository count > 20 |
| S3 blob store | Phase 3 when distributed storage needed |
