# ADR-006: Async-First Design

## Status

Accepted

## Context

ATLAS is fundamentally I/O-bound: model API calls, database queries, tool executions, event dispatching. The concurrency model must:

- Handle thousands of concurrent operations
- Avoid thread-safety complexity
- Integrate with FastAPI (async-native)
- Support streaming responses

## Decision

**Everything is async by default.** Synchronous code is the exception and must be justified.

### Rules

1. All I/O operations are `async` (database, HTTP, file operations where meaningful)
2. Module lifecycle hooks are async: `async def boot()`, `async def shutdown()`
3. Event handlers are async: `async def handle(event)`
4. CPU-bound work is offloaded: `asyncio.to_thread()` or process pools
5. Never block the event loop: no `time.sleep()`, no sync HTTP clients, no sync DB drivers

### Technology Alignment

| Component | Async Support |
|-----------|--------------|
| FastAPI | Native |
| SQLAlchemy 2 | Native (asyncpg driver) |
| httpx | Native |
| Redis | redis-py asyncio |
| Loguru | `enqueue=True` for non-blocking |

## Consequences

### Positive

- Single-threaded concurrency — no locks, no race conditions on shared state
- Scales to thousands of concurrent I/O operations per process
- Natural fit with the entire chosen stack
- Streaming support built-in

### Negative

- Async is contagious — all callers must be async
- CPU-bound work needs explicit offloading
- Debugging async stack traces is harder (mitigated with structured logging and correlation IDs)
- Some libraries lack async support (must be wrapped with `asyncio.to_thread`)

### Alternatives Considered

| Alternative | Assessment |
|-------------|------------|
| Threading | GIL limits parallelism, locks introduce complexity, harder to reason about |
| Multiprocessing | High memory overhead, IPC costs, not needed for I/O-bound work |
| Sync + workers (Celery) | Deployment complexity, unnecessary for Phase 1 |
| AsyncIO (chosen) | Native fit for I/O-bound AI workloads |
