# ADR-010: Cache Abstraction

## Status

Accepted

## Context

Multiple ATLAS services need caching: repository lookups, embedding search results, auth tokens, configuration. Cache backends differ in guarantees: in-memory dict is fast but unshared; Redis is shared but adds latency and failure modes.

## Decision

Define a `CacheService` interface with the minimal surface:

```python
class CacheService(ABC):
    async def get(self, key: str) -> Any | None: ...
    async def set(self, key: str, value: Any, ttl: float | None = None) -> None: ...
    async def delete(self, key: str) -> None: ...
    async def invalidate_pattern(self, pattern: str) -> None: ...
```

Implement two backends:
1. **MemoryCache** — `dict`-backed, TTL via `asyncio.Event`, for development/testing
2. **RedisCache** — Redis-backed, for production (Phase 3)

## Design Rule

Cache is NEVER authoritative. A miss must produce correct results from the backing store. This means:
- Cache is safe to clear at any time
- `set()` is always optional; callers must handle misses
- Write-through vs write-behind is a deployment policy, not a code concern

## Consequences

### Positive
- Zero-dependency cache for development
- One-line config switch to Redis in production
- Cache instrumentation: hit/miss counters feed into TelemetryService

### Negative
- Serialization responsibility sits with the caller (we recommend JSON for portability)
- `invalidate_pattern` behavior varies by backend (Redis supports natively; memory requires scan)
