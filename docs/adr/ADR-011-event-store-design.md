# ADR-011: Event Store as Write-Ahead Log

## Status

Accepted

## Context

ATLAS events must be persisted for: replay, audit, debugging, event sourcing, and downstream analytics. The Event Bus is in-memory and volatile. The Event Store bridges the gap: persist what the bus publishes, and replay what persists.

## Decision

The EventStore is a **subscriber** on the EventBus with a **generic append-only table**:

```sql
CREATE TABLE event_store (
    id         TEXT PRIMARY KEY,         -- ULID from the event
    event_type TEXT NOT NULL,
    version    INTEGER NOT NULL,
    source     TEXT NOT NULL,
    correlation_id TEXT NOT NULL,
    target     TEXT NOT NULL DEFAULT '*',
            timestamp   TEXT NOT NULL,            -- ISO-8601
    payload    TEXT NOT NULL,             -- JSON serialized
    metadata   TEXT NOT NULL DEFAULT '{}' -- JSON
);

CREATE INDEX idx_event_store_correlation ON event_store(correlation_id);
CREATE INDEX idx_event_store_source ON event_store(source);
CREATE INDEX idx_event_store_type ON event_store(event_type);
CREATE INDEX idx_event_store_timestamp ON event_store(timestamp);
```

Key design choices:

1. **Append-only**: Events written in order and never mutated. Immutability is the foundation of event sourcing.
2. **No deserialization on write**: The event is serialized once to JSON by the subscriber. Handlers replaying events deserialize on read.
3. **Separation of concerns**: The bus emits, the store persists. They share the `Event` type but not implementation.

## Consequences

### Positive
- Every event is durably recorded
- Replay picks up any subset (by type, correlation_id, time range)
- Append-only = no UPDATE locks, friendly to replication
- Event sourcing built on this: write `EventSourcedAggregateRoot` that records events

### Negative
- Storage grows unboundedly — mitigated by retention policy (configurable TTL or count-based compaction)
- Deserialization overhead on replay — mitigated by bulk reads and lazy loading
- JSON representation loses type information — mitigated by `event_type` column that selects correct Pydantic model
