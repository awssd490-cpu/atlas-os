# Project Layout

## Repository tree

```
atlas/
├── app/                              # Application source
│   ├── core/                         # Core infrastructure
│   │   ├── __init__.py
│   │   ├── errors.py                 # AtlasError base hierarchy
│   │   ├── interfaces.py             # IKernel, IModule, IProvider
│   │   ├── events.py                 # Event bus
│   │   ├── manifest.py               # Module manifest
│   │   ├── config/                   # Configuration management
│   │   │   ├── __init__.py
│   │   │   ├── errors.py
│   │   │   ├── models.py             # AtlasConfig
│   │   │   ├── loader.py             # ConfigLoader
│   │   ├── log/                      # Structured logging
│   │   │   ├── __init__.py
│   │   │   ├── errors.py
│   │   │   ├── models.py             # LogRecord
│   │   │   ├── logger.py             # AtlasLogger
│   │   │   ├── formatter.py          # JsonFormatter
│   │   ├── health/                   # Health monitoring
│   │   │   ├── __init__.py
│   │   │   ├── errors.py
│   │   │   ├── models.py             # HealthStatus, HealthCheck
│   │   │   ├── registry.py           # Check registry
│   │   │   ├── monitor.py            # HealthMonitor
│   │   ├── concurrency/              # Concurrency utilities
│   │   │   ├── __init__.py
│   │   │   ├── errors.py
│   │   │   ├── models.py             # ResourceState, ManagedResource
│   │   │   ├── limiter.py            # ConcurrencyLimiter
│   │   │   ├── resources.py          # ResourceManager
│   │   └── reliability/              # Retry utilities
│   │       ├── __init__.py
│   │       ├── errors.py
│   │       ├── models.py             # RetryPolicy, RetryResult
│   │       ├── retry.py              # RetryExecutor
│   │
│   └── rag/                          # Knowledge Layer (RAG)
│       ├── __init__.py
│       ├── models.py                 # Shared domain models
│       ├── errors.py                 # KnowledgeError hierarchy
│       ├── knowledge_base.py         # KnowledgeBase
│       ├── retriever.py              # KnowledgeRetriever
│       ├── context.py                # KnowledgeContextBuilder
│       ├── chunking/                 # Document chunking
│       │   ├── __init__.py
│       │   ├── base.py               # ChunkResult, ChunkingStrategy
│       │   ├── config.py             # ChunkingConfig
│       │   ├── chunker.py            # ChunkingEngine
│       │   ├── strategies.py         # Built-in strategies
│       │   ├── metadata.py           # ChunkMetadata
│       │   ├── errors.py
│       ├── embeddings/               # Vector embeddings
│       │   ├── __init__.py
│       │   ├── base.py               # EmbeddingProvider ABC
│       │   ├── config.py             # EmbeddingConfig
│       │   ├── models.py             # EmbeddingResult, EmbeddingVector
│       │   ├── registry.py           # Provider registry
│       │   ├── errors.py
│       │   └── providers/
│       │       ├── __init__.py
│       │       ├── deterministic.py  # DeterministicEmbeddingProvider
│       │       └── mock.py           # MockEmbeddingProvider
│       ├── vectorstore/              # Vector storage
│       │   ├── __init__.py
│       │   ├── base.py               # VectorStore ABC
│       │   ├── config.py             # VectorStoreConfig
│       │   ├── models.py             # SearchResult
│       │   ├── metrics.py            # SimilarityMetric, compute_similarity
│       │   ├── memory.py             # MemoryVectorStore
│       │   ├── errors.py
│       ├── hybrid/                   # Hybrid retrieval
│       │   ├── __init__.py
│       │   ├── base.py               # HybridRetriever ABC
│       │   ├── config.py             # HybridConfig
│       │   ├── models.py             # HybridResult, RetrievalScore
│       │   ├── fusion.py             # Fusion algorithms
│       │   ├── retriever.py          # DefaultHybridRetriever
│       │   ├── errors.py
│       ├── rerank/                   # Result reranking
│       │   ├── __init__.py
│       │   ├── base.py               # Reranker ABC
│       │   ├── config.py             # RerankConfig
│       │   ├── models.py             # RerankResponse, RerankedResult
│       │   ├── registry.py           # Reranker registry
│       │   ├── default.py            # DefaultReranker
│       │   ├── errors.py
│       ├── pipeline/                 # Pipeline orchestration
│       │   ├── __init__.py
│       │   ├── base.py               # KnowledgePipeline ABC
│       │   ├── config.py             # PipelineConfig
│       │   ├── models.py             # PipelineResult, PipelineStats
│       │   ├── errors.py
│       │   ├── registry.py           # Pipeline type registry
│       │   ├── builder.py            # PipelineBuilder
│       │   └── default.py            # DefaultKnowledgePipeline
│       ├── persistence/              # State persistence
│       │   ├── __init__.py
│       │   ├── base.py               # PersistenceBackend ABC
│       │   ├── config.py             # PersistenceConfig
│       │   ├── models.py             # PersistenceResult, PersistenceStats
│       │   ├── errors.py
│       │   ├── registry.py           # Backend registry
│       │   └── json_backend.py       # JsonPersistenceBackend
│       └── evaluation/               # Quality evaluation
│           ├── __init__.py
│           ├── base.py               # EvaluationRunner ABC
│           ├── config.py             # EvaluationConfig
│           ├── models.py             # EvaluationResult, BenchmarkResult
│           ├── errors.py
│           ├── registry.py           # Runner registry
│           ├── retrieval_metrics.py  # RetrievalMetrics
│           ├── benchmark.py          # BenchmarkRunner
│           ├── datasets.py           # EvaluationDataset, DatasetLoader
│           └── profiler.py           # PerformanceProfiler
│
├── tests/                            # Test suite
│   ├── unit/
│   │   ├── core/
│   │   │   ├── config/test_config.py
│   │   │   ├── log/test_log.py
│   │   │   ├── health/test_health.py
│   │   │   ├── concurrency/test_concurrency.py
│   │   │   └── reliability/test_reliability.py
│   │   ├── rag/
│   │   │   ├── test_models.py
│   │   │   ├── test_retriever.py
│   │   │   ├── test_context.py
│   │   │   ├── test_knowledge_base.py
│   │   │   ├── chunking/test_chunking.py
│   │   │   ├── embeddings/test_embeddings.py
│   │   │   ├── vectorstore/test_vectorstore.py
│   │   │   ├── hybrid/test_hybrid.py
│   │   │   ├── rerank/test_rerank.py
│   │   │   ├── pipeline/test_pipeline.py
│   │   │   ├── persistence/test_persistence.py
│   │   │   └── evaluation/test_evaluation.py
│   │   └── memory/
│   │       └── test_snapshots.py
│   ├── integration/                  # Integration tests
│   └── conftest.py                   # Shared fixtures
│
├── docs/                             # Documentation
│   ├── api/
│   │   ├── overview.md
│   │   ├── core.md
│   │   ├── providers.md
│   │   ├── rag.md
│   │   ├── persistence.md
│   │   └── evaluation.md
│   ├── tutorials/
│   │   ├── getting_started.md
│   │   ├── rag_pipeline.md
│   │   ├── persistence.md
│   │   ├── evaluation.md
│   │   └── advanced.md
│   └── developer/
│       ├── architecture.md
│       ├── contributing.md
│       ├── testing.md
│       ├── style_guide.md
│       └── project_layout.md
│
├── examples/                          # Runnable examples
│   ├── basic_rag.py
│   ├── custom_provider.py
│   ├── persistence_demo.py
│   ├── benchmark_demo.py
│   └── retry_demo.py
│
├── pyproject.toml                    # Project configuration
├── CLAUDE.md                         # Claude Code project config
├── README.md
└── .claude/                          # Claude Code settings
    └── settings.json
```

## Package overview

| Directory | Package | Responsibility |
|---|---|---|
| `app.core` | Core infrastructure | Config, logging, health, concurrency, retry |
| `app.rag` | Knowledge Layer | RAG models, chunking, embeddings, retrieval, reranking |
| `app.rag.pipeline` | Pipeline | Ingest/search orchestration, builder |
| `app.rag.persistence` | Persistence | Save/load KB state to/from JSON |
| `app.rag.evaluation` | Evaluation | Metrics, benchmark, profiling, datasets |
| `tests` | Tests | All unit and integration tests |

## Where to add new modules

| If you want to... | Directory |
|---|---|
| Add a new core utility | `app/core/<name>/` |
| Add a new chunking strategy | `app/rag/chunking/strategies.py` + register in `__init__.py` |
| Add a new embedding provider | `app/rag/embeddings/providers/<name>.py` |
| Add a new vector store backend | `app/rag/vectorstore/<name>.py` (subclass `VectorStore`) |
| Add a new reranker | `app/rag/rerank/<name>.py` (subclass `Reranker`) |
| Add a new persistence backend | `app/rag/persistence/<name>.py` (subclass `PersistenceBackend`) |
| Add a new pipeline variant | `app/rag/pipeline/<name>.py` (subclass `KnowledgePipeline`) |
| Add a new evaluation runner | `app/rag/evaluation/<name>.py` (subclass `EvaluationRunner`) |

## Where to add tests

Tests mirror the source layout exactly:

| Source | Tests |
|---|---|
| `app/core/config/` | `tests/unit/core/config/` |
| `app/rag/pipeline/` | `tests/unit/rag/pipeline/` |
| `app/rag/evaluation/` | `tests/unit/rag/evaluation/` |

Each source module has a corresponding test module named `test_<module>.py`.

## Documentation organization

```
docs/
├── api/               # API reference (per-package)
│   ├── overview.md    # Cross-reference and architecture
│   ├── core.md        # app.core.*
│   ├── providers.md   # app.rag chunking/embeddings/vectorstore/hybrid/rerank
│   ├── rag.md         # app.rag models/knowledge_base/context/pipeline
│   ├── persistence.md # app.rag.persistence
│   └── evaluation.md  # app.rag.evaluation
│
├── tutorials/         # Hands-on guides
│   ├── getting_started.md
│   ├── rag_pipeline.md
│   ├── persistence.md
│   ├── evaluation.md
│   └── advanced.md
│
└── developer/         # Contributor-facing
    ├── architecture.md
    ├── contributing.md
    ├── testing.md
    ├── style_guide.md
    └── project_layout.md   # this file
```
