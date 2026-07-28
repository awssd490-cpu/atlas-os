# ADR-009: SQLite via Thread Pool (stdlib sqlite3)

## Status

Accepted

## Context

Phase 2 needs a local SQL backend. Options: `sqlite3` (stdlib), `aiosqlite`, `sqlite-utils`, or `duckdb`.

## Decision

Use `sqlite3` from the stdlib, dispatched through `asyncio.to_thread()`.

**Why not aiosqlite**: aiosqlite wraps the same C library with equivalent serialization semantics. It adds a dependency with no performance or correctness benefit — SQLite is serialized at the connection level anyway. `asyncio.to_thread()` gives us the same "not blocking the event loop" guarantee with zero dependencies.

**Why not DuckDB**: excellent for analytics, but SQLite is more universal for application storage (nested transactions, WAL mode, broader tooling support, replaces PostgreSQL eventually).

## Consequences

### Positive
- Zero new dependencies for SQLite
- Thread pool naturally isolates long-running queries
- Same code path works with `asyncpg` when PostgreSQL arrives (real async I/O)

### Negative
- Thread overhead per query (minimal — SQLite queries are fast)
- No `async` context manager for connections (handled manually with `loop.run_in_executor`)
