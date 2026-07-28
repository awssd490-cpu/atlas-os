# ADR-007: Storage Abstraction Layer

## Status

Accepted

## Context

ATLAS Phase 2 must support multiple storage backends (SQLite now, PostgreSQL/Redis/Qdrant/Neo4j/S3 later). Business logic must never depend on a specific backend. The abstraction must handle:

- **SQL databases** (SQLite, PostgreSQL) — schema, migrations, transactions, row-level CRUD
- **Key-Value stores** (Redis) — TTL, namespace, best-effort cache vs durable KV
- **Vector stores** (Qdrant) — embedding storage, nearest-neighbor search
- **Graph stores** (Neo4j) — nodes, edges, property traversal
- **Blob stores** (S3) — streaming, checksums, metadata

## Decision

We define a **Connection protocol** as the lowest abstraction. All backends implement this protocol. Repositories build on connections.

```
Connection (protocol)
  ├── SQLConnection    — execute, fetchone, fetchall, execute_batch
  ├── KVConnection     — get, set, delete, scan
  ├── VectorConnection — upsert, search, delete
  ├── GraphConnection — create_node, create_rel, query
  └── BlobConnection   — upload, download, delete, exists
```

A **StorageEngine** manages connections: create, pool, lifecycle. A **StorageModule** registers with the kernel, boots connections, runs migrations, and provides health.

## Consequences

### Positive
- Business code imports only `Connection` protocols, never a db driver
- Backend swap = write a new connection class
- Testing = implement the protocol with in-memory structures
- Clear separation: Connection handles I/O; Repository handles domain operations

### Negative
- Protocol-based abstraction cannot express every DB-specific feature (e.g., PostgreSQL's `ON CONFLICT` vs SQLite's `INSERT OR REPLACE`) — these surface as explicit Repository methods
- Five protocols instead of one means more surface to maintain

## Alternatives

| Alternative | Assessment |
|---|---|
| Single unified Connection | Lowest-common-denominator loses too much. Vector search through a `fetchall()` interface is absurd. |
| Repository directly on DB driver | Every domain repository hard-coupled to a specific driver. Backend swap rewrites every repository. |
| Connection protocol + per-kind sub-protocols (chosen) | Each backend kind has a natural primitive set. Repositories pick the right protocol. |
