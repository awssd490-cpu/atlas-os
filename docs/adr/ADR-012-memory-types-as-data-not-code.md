# ADR-012: Memory Types as Data, Not Code

## Status

Accepted

## Context

The specification lists 10 memory types: ShortTermMemory, WorkingMemory, LongTermMemory, SemanticMemory, EpisodicMemory, ProceduralMemory, ConversationMemory, ProjectMemory, KnowledgeMemory, ReferenceMemory. Future memory types should be pluggable.

Should each type be a subclass?

## Decision

**Memory types are a field on the Memory model, not a class hierarchy.**

A single `memory_type` string field distinguishes types. Each type is configured by a `MemoryTypePolicy` that sets defaults for TTL, importance decay rate, max count, compression threshold, and allowed state transitions.

```python
class Memory:
    memory_type: str  # "short_term", "working", "episodic", ...
    namespace: str    # "conversation:abc", "project:xyz", ...
```

The `MemoryTypePolicy` registry maps type → policy parameters.

```python
policies = {
    "short_term": MemoryTypePolicy(ttl=3600, max_count=100, decay_rate=0.5),
    "long_term": MemoryTypePolicy(ttl=0, max_count=10000, decay_rate=0.01),
    "working": MemoryTypePolicy(ttl=300, max_count=20, decay_rate=0.8),
}
```

## Why Not Subclasses

Subclassing memory types creates several problems:

1. **Serialization complexity**: Each subclass needs a discriminator, type mapper, and registry to deserialize from SQL rows.
2. **Query complexity**: "Find all memories in namespace X" must UNION across type tables or use single-table inheritance with a `type` column — at which point subclasses add zero storage benefit.
3. **Feature interaction**: A compression strategy that works on all "long_term" memories needs to know every subclass.
4. **Pluggability illusion**: Adding a memory type requires deploying new Python code. With policy data, operators add types by writing config.

## What Differs Between Types

| Dimension | ShortTerm | LongTerm | Working | Episodic |
|---|---|---|---|---|
| Default TTL | 1 hour | None | 5 min | 24 hours |
| Max count | 100 | 10,000 | 20 | 500 |
| Decay rate | 0.5/hr | 0.01/day | 0.8/min | 0.1/hr |
| Compress by default | No | Yes | No | Yes |

Every dimension is a policy parameter. Not one requires a different code path.

## Alternatives

| Alternative | Rejected Because |
|---|---|
| 10 subclasses | Serialization overhead, query complexity, false pluggability |
| Single table with type+policy (chosen) | One query path, no type mapper, policy-driven behavior |
| Separate tables per type | Migration cost for every type, JOIN queries for cross-type operations |
