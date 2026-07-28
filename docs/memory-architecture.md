# Phase 3 Architecture: Knowledge & Memory Engine

## Critical Design Evaluation

Before presenting the architecture, I challenge several aspects of the spec:

### 1. 39 components → 13 files (merged, not cut)

The spec lists ~20 service classes. Many are responsibilities, not classes:
- **MemoryManager** and **MemoryService** are the same thing → `manager.py`
- **MemoryStatistics** is a method, not a service → `manager.stats()`
- **MemoryVersionManager** and **MemorySnapshotService** overlap → `snapshots.py`
- **MemoryIndexer** is a lifecycle hook, not a service → called by `start()`
- **MemoryLifecycleManager** is `manager.py`'s `transition_state()` method
- **MemoryConsolidationService** is a policy run by the garbage collector

**Result: 13 files vs 39. Zero lost functionality.**

### 2. 10 memory types → 1 `memory_type` field + `namespace`

ShortTermMemory, WorkingMemory, EpisodicMemory, etc., have NO behavioral difference. They are tags with different TTL/default importance/project associations. A `memory_type` column + `namespace` + `Policy` object handles this without 10 subclasses.

**What differs between memory types:** default TTL, importance decay rate, max count, compression threshold. All are policy parameters, not class hierarchies.

Prefer `memory_type` overinheritance. New memory types are data, not code.

### 3. 9 lifecycle states → 5

The spec's chain: Created → Validated → Indexed → Active → Referenced → Strengthened → Archived → Forgotten → Deleted

"Validated" and "Indexed" are internal lifecycle steps, not meaningful states for consumers. "Strengthened" and "Referenced" are metadata updates.

**Collapsed to:** Active, Archived, Forgotten, Deleted. Plus the transient "Created" before first persistence.

### 4. Importance scoring is a policy function, not a tree

The spec asks for "frequency, recency, pinning, user preference, relevance, semantic importance, manual weighting, model evaluation." This is a `float score(memory) → float` function, not a scorer class hierarchy. Multiple scoring strategies exist, but they compose via the policy object.

### 5. Forgetting is policy, not a service

"TTL expiration, decay, LRU, LFU, importance threshold, manual deletion, archive instead of delete" are all selection strategies for the garbage collector. The GC reads the policy, finds candidates, acts. No ForgettingService needed.

### 6. Compression without LLMs remains deterministic

Summarization requires a model (Phase 6+). Phase 3 implements: deduplication, chunk merging, reference replacement, hierarchical summaries through length-based truncation. The interface supports pluggable summarizers later.

### 7. Relationships reuse Phase 2 GraphStore

Memory relationships (parent, child, contradicts, supports, etc.) map directly to GraphNode/GraphRelationship. Phase 2 already implements this. The MemoryEngine wraps GraphStore with memory-specific query methods.

---

## Architecture

```
LLM / App Code
    ↓
MemoryManager  ←── primary facade, carries context
    ├── MemorySearchService      ← queries + ranking
    ├── RetrievalPipeline        ← multi-stage retrieval
    ├── ContextBuilder           ← assembles LLM context
    ├── MemoryGarbageCollector   ← policy-driven cleanup
    ├── MemorySnapshotService    ← versioning + snapshots
    ├── MemoryGraph              ← relationship queries
    └── MemoryCompressor         ← dedup, merge, future LLM
            ↓
      Phase 2 infrastructure:
      SQLConnection (SqliteRepository)
      CacheService
      VectorStore (for future embeddings)
      GraphStore (for relationships)
      EventStore (for memory events)
      StorageEngine (for lifecycle)
```

### Integration with Phase 1

| Phase 1 Contract | How Phase 3 Uses It |
|---|---|
| `Module.initialize(ctx)` | Creates MemoryManager from config |
| `Module.start()` | Runs V002 migration, seeds default policies |
| `Module.health()` | Reports memory counts, GC status, cache stats |
| `KernelContext` | Resolves storage services: `context.resolve(SQLConnection)` |
| `EventBus` | Publishes `MemoryCreated`, `MemoryArchived`, `MemoryForgotten` |
| `CapabilityRegistry` | Declares `"memory.store"`, `"memory.search"`, `"memory.relationships"` |
| `TelemetryService` | Records retrieval latency, cache hit rates, GC runs |
| `ConfigService` | Reads memory policy from `config.get("memory.*")` |
| `Logger` | Structured logging per operation |

### Integration with Phase 2

| Phase 2 Contract | How Phase 3 Uses It |
|---|---|
| `SQLConnection` | Persists memories, relationships, snapshots via SQLite |
| `SqliteRepository[Memory, str]` | CRUD for memory records |
| `CacheService` | Caches frequent retrievals, ranking results |
| `VectorStore` | Future embedding storage (Phase 6) |
| `GraphStore` | Stores memory relationships as labeled edges |
| `UnitOfWork` | Atomic multi-table operations |
| `MigrationManager` | Runs V002 migration for memory tables |
| `EventStore` | Persists memory lifecycle events |

---

## Folder Structure

```
app/memory/
├── __init__.py
├── memory.py             ← Memory domain model
├── interfaces.py         ← All service protocols/ABCs
├── manager.py            ← MemoryManager (primary facade)
├── policies.py           ← MemoryPolicy, retention rules
├── scoring.py            ← Importance scoring
├── retrieval.py          ← RetrievalPipeline + ContextBuilder
├── relationships.py      ← MemoryGraph (GraphStore wrapper)
├── compression.py        ← MemoryCompressor
├── collector.py          ← MemoryGarbageCollector
├── snapshots.py          ← MemorySnapshotService
├── search.py             ← MemorySearchService
├── events.py             ← Memory lifecycle events
├── module.py             ← MemoryModule (kernel module)
└── migrations.py         ← V002: memories, relationships, snapshots

app/config/
└── settings.py           ← Extended with MemoryConfig
```

---

## Build Order (14 steps)

### Step 1: Memory config
- `AtlasSettings` gets `MemoryConfig` with policy defaults

### Step 2: Memory events
- `MemoryCreated`, `MemoryUpdated`, `MemoryArchived`, `MemoryForgotten`, `MemoryRestored`

### Step 3: Memory domain model
- `Memory` class with all fields (id, type, namespace, importance, tags, content, version, TTL, state, ...)
- `MemoryState` enum, `MemoryRelationship` enum

### Step 4: Memory interfaces
- `MemoryService` — primary operations
- `MemorySearchService` — query, rank, filter
- `MemoryGarbageCollector` — policy scan
- `MemorySnapshotService` — checkpoint/restore
- `MemoryCompressor` — compress strategy
- `MemoryGraph` — relationship queries

### Step 5: V002 migration
- `memories`, `memory_relationships`, `memory_snapshots` tables + indexes

### Step 6: MemoryManager (core CRUD)
- `MemoryManager(MemoryService)` — create, get, update, delete, list, count, transition_state
- Caching wrapper around all read operations
- Unit tests: CRUD, state transitions, pagination, cache, telemetry

### Step 7: Search service
- `SqliteMemorySearchService(MemorySearchService)` — search by type, namespace, tags, content (LIKE), temporal, importance ranking
- Combined queries with sorted merge
- Unit tests: each search mode, pagination, empty results

### Step 8: Relationship management
- `MemoryGraphImpl(MemoryGraph)` wraps GraphStore
- Creates memory nodes on memory creation, edges on relationship assignment
- Traversal for context building
- Unit tests: assign, query, traverse

### Step 9: Policy engine
- `MemoryPolicy` config object
- `PolicyEvaluator` — evaluates retention, archive, delete decisions
- `ImportanceScorer` — base + override strategies
- Unit tests: policy application, scorer composition, decay

### Step 10: Retrieval pipeline
- `RetrievalPipeline` — multi-stage: filter → rank → expand → dedup → limit
- Pipeline stages as a list of callables
- `ContextBuilder` — takes retrieved memories → deduplicates → sorts → compresses → respects token budget
- Unit tests: pipeline assembly, stage isolation, token budgeting

### Step 11: Garbage collector
- `MemoryGarbageCollector` — sweep by TTL, importance threshold, max count per namespace
- Policy-driven: archive first, then delete
- Telemetry: memories collected, bytes freed
- Unit tests: TTL expiry, importance threshold, archive vs delete, namespace isolation

### Step 12: Snapshot service
- `MemorySnapshotService` — create/restore/list/delete snapshots
- Serializes all memories, relationships, and state
- Unit tests: snapshot round-trip, diff

### Step 13: Compression
- `MemoryCompressor` — dedup, chunk merge, reference replacement
- Truncation-based summarization (LLM summarization deferred)
- Unit tests: dedup, merge, truncation

### Step 14: MemoryModule
- Registers all services in DI
- Runs V002 migration
- Declares capabilities: `"memory.store"`, `"memory.search"`, `"memory.relationships"`
- Reports health: memory counts, GC status, cache performance
- Integration tests: full cycle with kernel

---

## Explicitly Deferred

| Feature | Deferred to |
|---|---|
| LLM-based summarization | Phase 6 (Model Engine) |
| Embedding-based semantic search | Phase 6 |
| Hybrid search (vector + keyword) | Phase 6 |
| Federation across kernel instances | Phase 10 |
| Memory diff/merge with CRDTs | Phase 10 |
| Multi-user access control | Phase 9 (API Gateway) |
