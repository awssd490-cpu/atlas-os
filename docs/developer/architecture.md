# Architecture

## High-level architecture

Atlas follows a layered architecture with clear separation of concerns:

```mermaid
graph TB
    subgraph "Application Layer"
        APP[Application / Agent Runtime]
    end

    subgraph "Pipeline Layer"
        PP[KnowledgePipeline]
        PB[PipelineBuilder]
    end

    subgraph "Knowledge Layer"
        KB[KnowledgeBase]
        CTX[KnowledgeContextBuilder]
        RET[KnowledgeRetriever]
    end

    subgraph "Provider Layer"
        CH[ChunkingEngine]
        EM[EmbeddingProvider]
        VS[VectorStore]
        HY[HybridRetriever]
        RR[Reranker]
    end

    subgraph "Persistence Layer"
        PS[PersistenceBackend]
    end

    subgraph "Evaluation Layer"
        EV[EvaluationRunner]
        BM[BenchmarkRunner]
        PF[PerformanceProfiler]
        MT[RetrievalMetrics]
    end

    subgraph "Core Infrastructure"
        CFG[ConfigLoader]
        LOG[AtlasLogger]
        HLTH[HealthMonitor]
        CONC[ConcurrencyLimiter]
        REL[RetryExecutor]
    end

    APP --> PP
    PP --> PB
    PP --> KB
    PP --> CTX
    PP --> PS
    KB --> CH
    KB --> EM
    KB --> VS
    KB --> RR
    CTX --> RET
    CTX --> HY
    CTX --> RR
    HY --> EM
    HY --> VS
    HY --> RET
    EV --> PP
    EV --> MT
    EV --> BM
    EV --> PF
    APP --> CFG
    APP --> LOG
    APP --> HLTH
    APP --> CONC
    APP --> REL
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `app.core.config` | Centralised configuration from dict/JSON/env |
| `app.core.log` | Structured JSON logging to stderr |
| `app.core.health` | Pluggable health check execution |
| `app.core.concurrency` | Async semaphore and resource lifecycle |
| `app.core.reliability` | Exponential backoff retry |
| `app.rag.models` | Shared domain models (documents, chunks, queries) |
| `app.rag.errors` | Shared error hierarchy |
| `app.rag.knowledge_base` | Central document registry with embedding/indexing |
| `app.rag.retriever` | Keyword-based retrieval |
| `app.rag.context` | Context builder merging retrieval + reranking |
| `app.rag.chunking` | Document splitting strategies |
| `app.rag.embeddings` | Vector embedding providers |
| `app.rag.vectorstore` | Vector storage and similarity search |
| `app.rag.hybrid` | Fused keyword + semantic retrieval |
| `app.rag.rerank` | Result reordering |
| `app.rag.pipeline` | End-to-end ingest/search orchestration |
| `app.rag.persistence` | Save/load KB state to durable storage |
| `app.rag.evaluation` | Quality metrics, benchmarking, profiling |

## Design principles

### 1. Frozen dataclasses for all models

Every data type is a `@dataclass(frozen=True)`. This ensures:
- Immutability by default — no accidental mutation
- Hashable when all fields are hashable
- Equality by value, not identity
- Default factory patterns for optional fields

### 2. ABCs for all pluggable interfaces

Every subsystem that supports multiple implementations defines an ABC:

- `EmbeddingProvider`, `VectorStore`, `HybridRetriever`, `Reranker`
- `KnowledgePipeline`, `PersistenceBackend`, `EvaluationRunner`

New implementations subclass the ABC and register with the global registry.

### 3. Global registries for type discovery

Each pluggable subsystem has a global registry mapping `name → class`:

```
embeddings:  register_provider("openai", OpenAIProvider)
rerank:      register_reranker("cross_encoder", CrossEncoderReranker)
pipeline:    register("custom", CustomPipeline)
persistence: register("sqlite", SQLiteBackend)
evaluation:  register("retrieval", RetrievalRunner)
```

Registries store **classes**, not instances. Instantiation happens at the call site.

### 4. No external dependencies for core functionality

Core infrastructure (config, logging, health, concurrency, retry) and the built-in providers (deterministic embeddings, memory vector store, heuristic reranker) use **stdlib only**. External ML models or databases are optional extensions.

### 5. Favor composition over inheritance

The pipeline composes providers rather than inheriting from them:

```
DefaultKnowledgePipeline
    └── has a → ChunkingEngine
    └── has a → KnowledgeBase (which has providers)
    └── has a → KnowledgeContextBuilder (which has retrievers)
```

### 6. Deterministic by default

All serialization uses `sort_keys=True`, `ensure_ascii=False`, and sorted field ordering. This means:
- Same input → same output every time
- Diffable snapshot files
- Reproducible test assertions

### 7. All I/O is async

All potentially blocking operations (embedding, vector search, file I/O, network calls) are `async def`. Synchronous callables are wrapped (e.g. `RetryExecutor` and `PerformanceProfiler` auto-detect and await where needed).

## Ingestion lifecycle

```mermaid
sequenceDiagram
    participant Pipe as Pipeline
    participant Loader as Loader
    participant Chunker as ChunkingEngine
    participant KB as KnowledgeBase
    participant EP as EmbeddingProvider
    participant VS as VectorStore

    Pipe->>Loader: ingest(path)
    Loader-->>Pipe: list[KnowledgeDocument]
    loop for each document
        Pipe->>Chunker: chunk(content)
        Chunker-->>Pipe: ChunkResult
        Pipe->>KB: register(document)
        alt auto_embed=True
            Pipe->>EP: embed_batch(chunks)
            EP-->>Pipe: EmbeddingResult
            Pipe->>KB: store embeddings
            alt auto_index=True
                Pipe->>VS: add(chunk_id, vector)
            end
        end
    end
    Pipe->>Pipe: update stats
    Pipe-->>User: PipelineResult
```

## Search lifecycle

```mermaid
sequenceDiagram
    participant Pipe as Pipeline
    participant CB as ContextBuilder
    participant KB as KnowledgeBase
    participant RR as Reranker

    Pipe->>CB: build(query)
    alt has hybrid_retriever
        CB->>KB: hybrid_retriever.retrieve()
        KB-->>CB: HybridResult
    else
        CB->>KB: keyword retriever.retrieve()
        KB-->>CB: KnowledgeResult
    end
    alt reranker configured and enabled
        CB->>RR: rerank(query, results)
        RR-->>CB: RerankResponse
    end
    CB-->>Pipe: KnowledgeContext
    Pipe->>Pipe: increment search counter
    Pipe-->>User: PipelineResult
```

## Error propagation

```mermaid
flowchart TD
    A[Operation] --> B{Success?}
    B -->|Yes| C[Return result]
    B -->|No| D{Error type?}
    D -->|Configuration| E[InvalidConfiguration]
    D -->|Missing resource| F[ResourceNotFound / DocumentNotFound / PipelineNotFound]
    D -->|Provider failure| G[EmbeddingProviderError / FusionError]
    D -->|Validation| H[InvalidPipelineConfiguration / InvalidRetryPolicy]
    D -->|Duplicate| I[DuplicateDocumentError / DuplicateResource]
    E --> J[Wrapped as KnowledgeError or AtlasError]
    F --> J
    G --> J
    H --> J
    I --> J
    J --> K[Caller handles via except / to_dict()]
```

## Configuration flow

```mermaid
flowchart LR
    A[JSON file] --> C[ConfigLoader.from_json]
    B[dict] --> D[ConfigLoader.from_dict]
    E[Environment vars] --> F[ConfigLoader.from_env]
    C --> G[AtlasConfig]
    D --> G
    F --> G
    G --> H[.validate]
    H --> I{Valid?}
    I -->|Yes| J[Immutable config object]
    I -->|No| K[InvalidConfiguration]
```
