# Atlas API Overview

## Purpose

Atlas is a modular, provider-independent framework for building Retrieval-Augmented Generation (RAG) applications. It provides a complete stack — from document loading and chunking through embedding, retrieval, reranking, evaluation, persistence, and core infrastructure — that can be composed into custom pipelines.

## Package dependency diagram

```
app/core/                     (infrastructure layer)
├── config/                   ← no deps
├── log/                      ← depends on config (log levels)
├── health/                   ← no deps
├── concurrency/              ← no deps
└── reliability/              ← no deps

app/rag/                      (knowledge layer)
├── chunking/                 ← depends on rag/models
├── embeddings/               ← depends on rag/models, rag/errors
├── vectorstore/              ← depends on rag/models, rag/errors
├── retriever.py              ← depends on knowledge_base, models
├── hybrid/                   ← depends on embeddings, vectorstore, retriever
├── rerank/                   ← depends on rag/models, rag/errors
├── knowledge_base.py         ← depends on chunking, embeddings, vectorstore
├── context.py                ← depends on knowledge_base, retriever, rerank
├── pipeline/                 ← depends on knowledge_base, chunking, context, embeddings, vectorstore
├── persistence/              ← depends on knowledge_base, embeddings, vectorstore
└── evaluation/               ← depends on pipeline, persistence (no hard deps)
```

## Lifecycle diagram

```
Configuration  ──→  Logging  ──→  Application
    │
    ├── Document Loader ──→ Chunking ──→ Embeddings
    │                                              │
    │                                              ▼
    │                                   Vector Store ←── Index
    │                                              │
    ▼                                              ▼
Knowledge Base ←── Registration         Hybrid / Keyword Retrieval
    │                                              │
    │                                              ▼
    │                                           Reranker
    │                                              │
    │                                              ▼
    │                                    Knowledge Context Builder
    │                                              │
    └──── Pipeline ─── ingest() ──────── search()

Persistence: save() / load() / update()
Evaluation:  evaluate() / benchmark() / profile()
Health:      check() / check_all()
Reliability: execute() with retry
```

## Module relationships

| If you want to... | Use this package |
|---|---|
| Load documents from files | A loader callable → `DefaultKnowledgePipeline` |
| Split documents into chunks | `ChunkingEngine` with a strategy |
| Generate vector embeddings | `EmbeddingProvider` subclass |
| Store and search vectors | `VectorStore` subclass |
| Combine keyword + semantic search | `DefaultHybridRetriever` |
| Improve retrieval ordering | `Reranker` subclass / `DefaultReranker` |
| Orchestrate the full RAG flow | `DefaultKnowledgePipeline` / `PipelineBuilder` |
| Save/load pipeline state | `JsonPersistenceBackend` |
| Measure retrieval quality | `RetrievalMetrics` |
| Benchmark performance | `BenchmarkRunner` |
| Profile execution | `PerformanceProfiler` |
| Manage configuration | `AtlasConfig` / `ConfigLoader` |
| Write structured logs | `AtlasLogger` |
| Monitor component health | `HealthMonitor` |
| Limit concurrent access | `ConcurrencyLimiter` |
| Track resource lifecycle | `ResourceManager` |
| Retry transient failures | `RetryExecutor` |

## Best practices

- **Always validate configuration** by calling `AtlasConfig.validate()` after loading.
- **Use the builder** (`PipelineBuilder`) rather than constructing pipelines manually — it validates required components.
- **Prefer hybrid retrieval** when both embeddings and a vector store are configured — it fuses keyword precision with semantic breadth.
- **Enable reranking** on the knowledge base to improve result ordering at a small latency cost.
- **Batch embeddings** during ingestion — the pipeline respects `PipelineConfig.batch_size`.
- **Use `auto_embed=True` with `auto_index=False`** when you want to keep embeddings in memory but not in the vector store.
- **Run warmup iterations** before benchmarking to stabilise JIT and caching effects.
- **Set `overwrite=False`** (the default) on `PersistenceConfig` to guard against accidental snapshot overwrites.

## Common mistakes

- **Forgetting to set `overwrite=True`** when re-saving to an existing path — `save()` will raise `PersistenceError`.
- **Registering the same document ID twice** — the pipeline silently skips duplicates during `ingest_documents()`.
- **Using `time.time()` for benchmarks** — always use `time.perf_counter()` which is monotonic and high-resolution.
- **Sharing a `KnowledgeBase` across pipelines** — each pipeline should own its own KB unless you explicitly want shared state.
- **Forgetting `ensure_ascii=False`** when writing JSON with unicode content — all Atlas JSON serialisers handle this correctly.

## Thread safety notes

- `AtlasLogger` is **not** thread-safe — each thread should create its own logger.
- `ConcurrencyLimiter` is **async-only** — it wraps `asyncio.Semaphore`, not `threading.Semaphore`.
- `ResourceManager` is **not** thread-safe — intended for single-threaded async use.
- All frozen dataclasses (`AtlasConfig`, `LogRecord`, `HealthCheck`, `ManagedResource`, `RetryPolicy`, `RetryResult`) are safe to share across threads.
- Global registries (`register_*`, `get_*`) are **not** thread-safe — register at startup.

## Async usage notes

- All pipeline methods (`ingest`, `search`, `clear`) are async.
- `HealthMonitor.check()` and `check_all()` are async but accept sync or async check functions.
- `RetryExecutor.execute()` is async but accepts sync or async callables.
- `PerformanceProfiler.profile()` is async but accepts sync or async callables.
- `BenchmarkRunner.run()` requires an async component callable.
- Persistence operations (`save`, `load`, `update`) are async.
