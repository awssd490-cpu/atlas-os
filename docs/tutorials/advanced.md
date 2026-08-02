# Advanced Atlas Usage

## Goal

Explore advanced patterns: error handling, retry policies, health monitoring, concurrency control, and custom providers.

## Prerequisites

- All preceding tutorials
- Familiarity with `app.core.*` packages

---

## Error handling

### Hierarchy

All Atlas errors inherit from `AtlasError`:

```text
AtlasError
├── ConfigurationError
│   └── InvalidConfiguration
├── LoggingError
│   └── InvalidLogLevel
├── HealthError
│   ├── HealthCheckNotFound
│   └── DuplicateHealthCheck
├── ConcurrencyError
│   ├── ResourceNotFound
│   └── DuplicateResource
├── ReliabilityError
│   └── InvalidRetryPolicy
└── KnowledgeError (app.rag)
    ├── DuplicateDocumentError
    ├── DocumentNotFoundError
    ├── PipelineError
    │   ├── InvalidPipelineConfiguration
    │   └── PipelineNotFound
    ├── PersistenceError
    │   ├── InvalidPersistenceConfiguration
    │   └── PersistenceNotFound
    ├── EvaluationError
    │   ├── InvalidEvaluationConfiguration
    │   └── EvaluationNotFound
    ├── EmbeddingError
    │   ├── InvalidEmbeddingConfiguration
    │   ├── EmbeddingProviderError
    │   └── UnsupportedEmbeddingProvider
    ├── RerankError
    │   ├── InvalidRerankConfiguration
    │   └── RerankerNotFound
    ├── ChunkingError
    │   ├── ChunkingConfigError
    │   ├── ChunkingEngineError
    │   ├── ChunkingStrategyError
    │   └── UnsupportedStrategyError
    ├── VectorStoreError
    │   ├── InvalidVectorStoreConfiguration
    │   ├── VectorStoreFullError
    │   ├── VectorDimensionMismatchError
    │   └── VectorNotFoundError
    └── HybridError
        ├── InvalidHybridConfiguration
        └── FusionError
```

Every error has:

- `message` — human-readable description
- `code` — machine-readable identifier (e.g. `"INVALID_PIPELINE_CONFIGURATION"`)
- `details` — structured dict with context
- `to_dict()` — serialization for API responses

### Best practice: catch specific errors

```python
from app.rag.errors import DuplicateDocumentError, DocumentNotFoundError
from app.rag.persistence.errors import PersistenceError

try:
    kb.register(doc)
except DuplicateDocumentError as exc:
    log.warning("Document already exists", document_id=exc.details.get("document_id"))
except DocumentNotFoundError as exc:
    log.error("Document not found", document_id=exc.details.get("document_id"))
```

### Use `to_dict()` for structured error output

```python
try:
    pipeline.ingest("data")
except Exception as exc:
    if hasattr(exc, "to_dict"):
        print(exc.to_dict())  # {"code": "...", "message": "...", "details": {...}}
```

---

## Retry policies

Use `RetryExecutor` to handle transient failures (network timeouts, rate limits):

```python
from app.core.reliability import RetryExecutor, RetryPolicy

retry = RetryExecutor(
    RetryPolicy(
        max_attempts=5,
        initial_delay_ms=100.0,
        backoff_multiplier=2.0,
        max_delay_ms=10_000.0,
        retry_exceptions=(ConnectionError, TimeoutError),
    )
)

async def fetch_data() -> list[str]:
    # Simulated network call
    ...

result = await retry.execute(fetch_data)
if result.success:
    print(f"Succeeded on attempt {result.attempts}")
else:
    print(f"Failed after {result.attempts} attempts")
```

**Policies by use case:**

| Use case | max_attempts | initial_delay_ms | retry_exceptions |
|---|---|---|---|
| Network calls | 3 | 100 | `ConnectionError, TimeoutError` |
| Rate-limited API | 5 | 1000 | `RateLimitError` |
| Database | 2 | 50 | `OperationalError, InterfaceError` |
| File I/O | 3 | 10 | `OSError, PermissionError` |

---

## Health monitoring

Register health checks for your application components:

```python
from app.core.health import HealthMonitor, HealthStatus
from app.rag.knowledge_base import KnowledgeBase

# Sync check — returns a status
HealthMonitor.register("knowledge_base", lambda: (
    HealthStatus.HEALTHY if kb.count() > 0 else HealthStatus.DEGRADED,
    f"{kb.count()} documents loaded",
))

# Async check — checks an external dependency
async def check_vector_store() -> HealthStatus:
    count = vector_store.count()
    return HealthStatus.HEALTHY if count > 0 else HealthStatus.UNHEALTHY

HealthMonitor.register("vector_store", check_vector_store)

# Run all checks
results = await HealthMonitor.check_all()
for check in results:
    print(f"{check.name}: {check.status.name} ({check.duration_ms:.1f}ms)")
```

---

## Concurrency control

Limit concurrent operations when accessing shared resources:

```python
from app.core.concurrency import ConcurrencyLimiter

# Allow at most 5 concurrent embedding calls
limiter = ConcurrencyLimiter(max_concurrent=5)

async def safe_embed(text: str):
    async with limiter:
        return await provider.embed(text)

# 20 concurrent calls, but only 5 run at once
tasks = [safe_embed(t) for t in texts]
results = await asyncio.gather(*tasks)
```

Track resource lifecycle:

```python
from app.core.concurrency import ResourceManager

mgr = ResourceManager()
mgr.register("vector_store", vector_store)
mgr.open("vector_store")

# ... use the store ...

mgr.close("vector_store")

# Emergency cleanup
records = mgr.close_all()
```

---

## Custom embedding provider

Implement a custom `EmbeddingProvider` by subclassing the ABC:

```python
from app.rag.embeddings import EmbeddingProvider, EmbeddingConfig
from app.rag.embeddings.models import EmbeddingResult, EmbeddingVector

class MyEmbeddingProvider(EmbeddingProvider):
    @property
    def name(self) -> str:
        return "my_provider"

    async def embed(self, text: str) -> EmbeddingResult:
        vector = self._generate(text)
        return EmbeddingResult(
            embeddings=(EmbeddingVector(vector=vector, dimensions=len(vector)),),
            provider=self.name,
        )

    async def embed_batch(self, texts: list[str]) -> EmbeddingResult:
        vectors = tuple(self._generate(t) for t in texts)
        return EmbeddingResult(embeddings=vectors, provider=self.name)

    def _generate(self, text: str) -> tuple[float, ...]:
        # Your custom embedding logic here
        return (0.1, 0.2, 0.3, 0.4)
```

Register it globally:

```python
from app.rag.embeddings import register_provider
register_provider("my_provider", MyEmbeddingProvider)
```

---

## Custom reranker

```python
from app.rag.rerank import Reranker, RerankConfig
from app.rag.rerank.models import RerankResponse, RerankedResult

class MyReranker(Reranker):
    async def rerank(self, query: str, results: list[tuple[str, float]]) -> RerankResponse:
        reranked = []
        for chunk_id, score in results:
            final_score = score * 1.1
            reranked.append(RerankedResult(
                chunk_id=chunk_id,
                original_score=score,
                final_score=final_score,
            ))
        reranked.sort(key=lambda r: r.final_score, reverse=True)
        return RerankResponse(results=tuple(reranked))
```

---

## Structured logging in production

```python
from app.core.log import AtlasLogger

log = AtlasLogger("my-app", level="INFO")

# Log with structured metadata
log.info("Request started", method="GET", path="/search", query_id="abc-123")
log.info("Search completed", results=5, latency_ms=12.3)

# Log exceptions with trace
try:
    risky_operation()
except ValueError as exc:
    log.exception("Operation failed", exception=exc, component="search")
```

---

## Complete examples

See these files for full runnable examples:

- `examples/custom_provider.py` — Custom embedding and reranker
- `examples/retry_demo.py` — Retry policies in action
- `examples/basic_rag.py` — Full pipeline example
- `examples/persistence_demo.py` — Save/load/update
- `examples/benchmark_demo.py` — Metrics, benchmarks, profiling
