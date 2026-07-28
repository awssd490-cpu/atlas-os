# ADR-013: Memory Lifecycle Simplification

## Status

Accepted

## Context

The specification proposes a 9-state lifecycle: Created → Validated → Indexed → Active → Referenced → Strengthened → Archived → Forgotten → Deleted.

Each transition should be configurable.

## Decision

**Collapse to 5 states: Active, Archived, Forgotten, Deleted (plus Created).**

```python
class MemoryState(enum.Enum):
    ACTIVE = "active"       # Normal operational state
    ARCHIVED = "archived"   # Preserved but excluded from default search
    FORGOTTEN = "forgotten" # Marked for deletion by policy (grace period)
    DELETED = "deleted"     # Irreversibly removed
```

**Created** is not a state stored in the database — it's the in-memory state before the first `save()`.

## Why Collapse

- **Validated** and **Indexed** are internal processing steps that complete within milliseconds of creation. Making them persistent states adds a write + state transition for zero consumer value.
- **Referenced** and **Strengthened** are counters/metadata, not state transitions. `reference_count += 1` and `importance *= 1.1` are mutations of existing fields.
- Every additional state doubles the transition matrix for policy authoring.

## Configurable Transitions

Instead of a state machine, transitions are policy-evaluated by the garbage collector:

```python
class RetentionPolicy:
    ttl_seconds: float | None
    importance_threshold: float | None
    max_count_per_namespace: int | None
    archive_before_delete: bool = True
    grace_period_seconds: float = 86400 * 7  # 7 days in FORGOTTEN
```

The GC:
1. Scans for TTL-expired memories → mark FORGOTTEN
2. Scans for below-threshold importance → mark FORGOTTEN
3. Scans for namespace over capacity → mark oldest/lowest-importance FORGOTTEN
4. Scans FORGOTTEN past grace period → DELETE

## Consequences

- Fewer states = simpler query logic (no "Active" vs "Referenced" vs "Strengthened" in WHERE clauses)
- Policy governs *when* transitions happen, not *what* states exist
- Grace period between FORGOTTEN and DELETED enables recovery
