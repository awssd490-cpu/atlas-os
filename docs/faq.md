# Frequently Asked Questions

## Architecture

### What is Atlas?

Atlas is a modular, provider-independent framework for building Retrieval-Augmented Generation (RAG) applications. It provides a complete stack from document loading through chunking, embedding, retrieval, reranking, and evaluation — with pluggable providers at every layer.

### How does Atlas compare to LangChain or LlamaIndex?

Atlas is designed for **composability and minimal dependencies**. Unlike LangChain which wraps many third-party libraries, Atlas's built-in providers use only the Python stdlib (deterministic embeddings, in-memory vector store, heuristic reranker). This makes Atlas ideal for:
- **Development and prototyping** without external API keys or ML models
- **Testing retrieval quality** before committing to an external provider
- **Custom provider workflows** where you bring your own embedding or reranking model

### Does Atlas support LLMs?

Atlas is a **RAG framework**, not an LLM provider. It handles document storage, retrieval, and context building — the "retrieval" and "augmented" parts of RAG. The retrieved context is designed to be injected into any LLM provider's prompt.

### What are the layers?

```
Core infrastructure (config, logging, health, concurrency, retry)
    ↓
Provider layer (chunking, embeddings, vector store, hybrid, rerank)
    ↓
Knowledge layer (knowledge base, retriever, context builder)
    ↓
Pipeline layer (orchestration, persistence, evaluation)
```

## Async

### Why is everything async?

All I/O-bound operations (embedding, vector search, file persistence, health checks) are `async def` to avoid blocking the event loop. This allows efficient concurrent execution — for example, embedding multiple chunks in parallel.

### Can I use Atlas with sync code?

Yes. Atlas accepts sync callables in several places:
- `HealthMonitor.register()` accepts sync or async check functions
- `RetryExecutor.execute()` accepts sync or async callables
- `PerformanceProfiler.profile()` accepts sync or async callables
- Pipeline loader functions are typically sync

The entry point for any async operation must be `asyncio.run(main())`.

### Do I need an async web framework?

No — Atlas is framework-agnostic. It works with FastAPI, aiohttp, Sanic, or any Python application.

## Providers

### How do I add a custom embedding provider?

```python
from app.rag.embeddings import EmbeddingProvider, register_provider

class MyProvider(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    async def embed(self, text: str) -> EmbeddingResult:
        # Your logic here
        ...

    async def embed_batch(self, texts) -> EmbeddingResult:
        # Your batch logic here
        ...

register_provider("my_provider", MyProvider)
```

See the tutorial at `docs/tutorials/advanced.md` and the template at `templates/custom_provider/`.

### How do I add a custom vector store?

Subclass `VectorStore` from `app.rag.vectorstore.base` and implement all abstract methods. The in-memory implementation at `app.rag.vectorstore.memory.MemoryVectorStore` is a good reference.

### Can I use OpenAI embeddings?

Yes, by implementing a custom `EmbeddingProvider` that calls the OpenAI API. The `DeterministicEmbeddingProvider` shows the required interface. See `templates/custom_provider/` for the pattern.

### How do I use a reranker?

```python
from app.rag.rerank import DefaultReranker

# Create and attach to KnowledgeBase
reranker = DefaultReranker()
kb = KnowledgeBase(reranker=reranker)

# Or use the pipeline builder
pipeline = PipelineBuilder().reranker(reranker)...build()
```

The reranker is applied automatically during context building — no additional code needed.

## Persistence

### How often should I save?

That depends on your durability requirements:
- Save after each ingestion batch if data is valuable
- Use `update()` for incremental saves — it's faster than a full re-save
- Call `stats()` periodically to monitor snapshot size

### Can I have multiple snapshots?

Yes. Each `save()` call writes to a distinct path. You can maintain daily snapshots:

```python
import datetime
path = f"snapshot-{datetime.date.today().isoformat()}.json"
backend = JsonPersistenceBackend(PersistenceConfig(overwrite=True))
await backend.save(path, kb)
```

### Can I load a snapshot into a different KnowledgeBase structure?

The JSON format is tied to the data model — loaded documents, chunks, and embeddings are deserialized into the same Python types. If you've added custom fields, they won't be in the snapshot.

## Evaluation

### What metrics should I use?

| Goal | Metric |
|---|---|
| Are my top results relevant? | `precision_at_k` |
| Did I find all relevant documents? | `recall_at_k` |
| Balance of precision and recall | `f1_at_k` |
| Is the first result relevant? | `mean_reciprocal_rank` |
| Overall ranking quality | `average_precision` |
| Position-weighted ranking | `normalized_dcg` |

### How many queries do I need for reliable evaluation?

- **Small-scale**: 10–50 queries for development iteration
- **Medium**: 50–200 queries for feature evaluation
- **Production**: 200–1000+ queries from user logs for statistical significance

### Why is my benchmark throughput zero?

The component completed too quickly for `time.perf_counter()` to measure. Either:
- Increase `benchmark_runs` (100+)
- Use a slower component
- Add `asyncio.sleep(0.001)` to simulate real-workload latency

## Performance

### How can I improve ingestion speed?

```python
cfg = PipelineConfig(
    auto_embed=True,
    auto_index=True,
    batch_size=64,        # larger batches for embedding
)
```

- Increase `batch_size` for more throughput
- Use a faster embedding provider
- Disable embedding/indexing during initial load (`auto_embed=False`), then batch-embed later

### How can I improve search latency?

- Reduce `max_chunks` in search
- Disable reranking during latency-sensitive requests
- Use keyword-only retrieval if semantic search is overkill (`KnowledgeRetriever` only)
- Prefer an indexed vector store over `MemoryVectorStore` for >10K vectors

## Testing

### How do I run a single test?

```bash
python -m pytest tests/unit/rag/pipeline/ -k test_build_returns_pipeline -v
```

### How do I add a new test?

See the [testing guide](developer/testing.md) for test structure, fixtures, and async patterns.

### Why do some registry tests need `clear_*()` calls?

Global registries persist across tests. If one test registers a provider, a later test that checks for an empty registry will fail. Use `clear_*()` in an `autouse` fixture:

```python
@pytest.fixture(autouse=True)
def setup(self):
    clear_providers()
    yield
    clear_providers()
```

## Contribution

### How do I submit a pull request?

See [contribute.md](developer/contribute.md) for the full PR workflow.

### What should I include in a PR?

See the [PR checklist](developer/contribute.md#pull-request-checklist) — 11 items covering tests, documentation, and code style.

### How are errors structured?

```python
class MyError(AtlasError):
    def __init__(self, message, *, code="MY_ERROR", details=None):
        super().__init__(message, code=code, details=details)
```

Every error has:
- `message` — human-readable text
- `code` — machine-readable identifier (SCREAMING_SNAKE_CASE)
- `details` — dict with structured context
- `to_dict()` — serialization for API responses
