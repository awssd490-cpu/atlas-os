# ADR-015: Forgetting as Garbage Collection

## Status

Accepted

## Context

The specification lists 8 forgetting mechanisms: TTL expiration, decay, LRU, LFU, importance threshold, manual deletion, project cleanup, archive-before-delete. A "ForgettingService" or "ForgettingEngine" class seems like the natural shape.

## Decision

**Forgetting is NOT a service. It is a garbage collector driven by policy.**

```python
class MemoryGarbageCollector:
    def __init__(self, repo, policy, clock):
        ...

    async def collect(self) -> GCResult:
        # 1. Expire by TTL
        # 2. Expire by importance threshold
        # 3. Evict by namespace capacity
        # 4. Purge FORGOTTEN past grace period
```

Each forgetting mechanism is a query + action, not a strategy class:

| Mechanism | Query | Action |
|---|---|---|
| TTL expiration | `WHERE ttl > 0 AND created_at + ttl < NOW()` | → ARCHIVED (if archive_before_delete) or → FORGOTTEN |
| Importance threshold | `WHERE importance < threshold AND state = ACTIVE` | → ARCHIVED (if archive_before_delete) or → FORGOTTEN |
| LRU eviction | `WHERE namespace = X AND state = ACTIVE ORDER BY accessed_at LIMIT N` | → FORGOTTEN |
| LFU eviction | `WHERE namespace = X ORDER BY access_count LIMIT N` | → FORGOTTEN |
| Manual deletion | Specific ID | → DELETED (hard) |
| Project cleanup | `WHERE namespace LIKE 'project:%'` | → FORGOTTEN (batch) |

## Why Not Service Classes

1. **All mechanisms share the same output**: a state transition. They differ only in the SELECT query finding candidates.
2. **Composing services**: would need a "forgetting orchestrator" that calls each service. That's the GC.
3. **Testing**: testing 8 strategies is 8 test functions on one class, not 8 test files across 8 classes.

## Separation

The GC doesn't decide *what* forgetting means — the policy does:

```python
class MemoryPolicy:
    archive_before_forget: bool = True   # Active → Archived before → Forgotten
    grace_period_seconds: float = 604800  # 7 days in Forgotten before hard delete
    ttl_policy: TTLPolicy | None
    importance_policy: ImportancePolicy | None
    capacity_policy: dict[str, int]       # namespace → max memories
```

The GC reads policy, selects candidates, acts. Policy changes = behavior changes. No code changes.
