# ADR-014: Retrieval Pipeline as Stage List

## Status

Accepted

## Context

The specification requires a retrieval pipeline where each stage is independently replaceable. Stages include filtering, permission checks, namespace scoping, metadata matching, importance ranking, recency, semantic search, relationship expansion, ranking, compression, and context assembly.

A naive implementation would hard-code this sequence.

## Decision

**The pipeline is a list of callables (stages).**

```python
class RetrievalPipeline:
    def __init__(self, stages: list[RetrievalStage]):
        self._stages = stages

    async def retrieve(self, query: MemoryQuery) -> RetrievalResult:
        ctx = PipelineContext(query=query)
        for stage in self._stages:
            ctx = await stage.process(ctx)
            if ctx.aborted:
                break
        return ctx.to_result()
```

Each stage implements:

```python
class RetrievalStage(Protocol):
    async def process(self, ctx: PipelineContext) -> PipelineContext: ...
```

## Why Not a Configurable Graph

A DAG of stages (this filter→that ranker→this expander) would be more flexible but introduces:
1. **Topological ordering** — must compute and validate every query shape
2. **Stage discovery** — query must know which DAG to use
3. **Testing complexity** — permutations explode with branching

A linear list covers 95% of retrieval shapes. When the pipeline needs branching, a single "router" stage at the front splits the flow.

## Default Pipeline

```python
DEFAULT_PIPELINE = [
    NamespaceFilter(),       # scope to requested namespaces
    TypeFilter(),            # filter by memory type(s)
    MetadataFilter(),        # tag, owner, source matching
    SearchExecutor(),        # execute against DB (exact/LIKE/temporal)
    ImportanceRanker(),      # sort by (importance * recency * frequency)
    RelationshipExpander(),  # walk relationships, append related
    Deduplicator(),          # remove duplicates by memory_id
    ContextAssembler(),      # apply token budget, produce result
]
```

## Consequences

- Stages are independently testable
- Pipeline can be assembled by policy (different shapes for different query types)
- New stages added by appending to the list — no inheritance, no registration
- Debugging is linear: trace through the stage list
