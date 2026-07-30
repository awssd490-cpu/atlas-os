# Provider Layer (`app.rag.*`)

## Overview

The provider layer comprises the pluggable subsystems that KnowledgeBase and Pipeline orchestrate: chunking, embeddings, vector stores, hybrid retrieval, and reranking. Each follows the same pattern — an abstract base class defining the interface, a global registry, frozen dataclass models, and concrete implementations.

---

## `app.rag.chunking`

### Purpose

Splits document text into `KnowledgeChunk` objects using configurable strategies. The `ChunkingEngine` is the entry point; strategies are registered callables.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `ChunkingEngine` | Class | Entry point — calls a named strategy |
| `ChunkingConfig` | Frozen dataclass | Strategy selection, chunk size, overlap, etc. |
| `ChunkResult` | Frozen dataclass | Contains `tuple[KnowledgeChunk]` and metadata |
| `ChunkMetadata` | Frozen dataclass | Per-chunk metadata |
| `ChunkingStrategy` | Protocol | Callable signature for strategies |
| `STRATEGY_FIXED_SIZE` | str | `"fixed_size"` |
| `STRATEGY_PARAGRAPH` | str | `"paragraph"` |
| `STRATEGY_RECURSIVE` | str | `"recursive"` |
| `STRATEGY_SENTENCE` | str | `"sentence"` |
| `STRATEGY_SLIDING_WINDOW` | str | `"sliding_window"` |
| `STRATEGY_WHOLE_DOCUMENT` | str | `"whole_document"` |
| `ChunkingError` | Exception | Base chunking error |
| `ChunkingConfigError` | Exception | Invalid configuration |
| `ChunkingEngineError` | Exception | Engine execution failure |
| `ChunkingStrategyError` | Exception | Strategy execution failure |
| `UnsupportedStrategyError` | Exception | Unknown strategy name |

### `ChunkingEngine`

```python
class ChunkingEngine:
    def __init__(self, config: ChunkingConfig | None = None): ...
    def chunk(self, text: str, config: ChunkingConfig | None = None,
              *, document_id: str = "") -> ChunkResult: ...
    def register_strategy(self, name: str, strategy: ChunkingStrategy) -> None: ...
    def available_strategies(self) -> list[str]: ...
    def reset(self) -> None: ...
```

**Example:**

```python
engine = ChunkingEngine(ChunkingConfig(strategy="sentence"))
result = engine.chunk("Long document text...", document_id="doc_1")
for chunk in result.chunks:
    print(chunk.chunk_id, chunk.content[:50])
```

---

## `app.rag.embeddings`

### Purpose

Generates vector embeddings from text. Provides an abstract `EmbeddingProvider` base class, a global registry, and two built-in implementations.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `EmbeddingProvider` | ABC | Abstract base for embedding providers |
| `EmbeddingConfig` | Frozen dataclass | Provider name, dimensions, batch size, etc. |
| `EmbeddingResult` | Frozen dataclass | Result with `tuple[EmbeddingVector]` |
| `EmbeddingVector` | Frozen dataclass | Single vector with metadata |
| `DeterministicEmbeddingProvider` | Class | SHA-256-based deterministic provider |
| `MockEmbeddingProvider` | Class | Random vector provider (for testing) |
| `EmbeddingError` | Exception | Base embedding error |
| `EmbeddingProviderError` | Exception | Provider failure |
| `InvalidEmbeddingConfiguration` | Exception | Invalid config values |
| `UnsupportedEmbeddingProvider` | Exception | Unknown provider name in registry |
| `register_provider()` | Function | Register a provider class |
| `get_provider()` | Function | Look up a provider class |
| `list_providers()` | Function | List registered provider names |
| `clear_providers()` | Function | Clear registry (tests) |

### `EmbeddingProvider`

```python
class EmbeddingProvider(ABC):
    def __init__(self, config: EmbeddingConfig): ...
    @property
    def config(self) -> EmbeddingConfig: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    async def embed(self, text: str) -> EmbeddingResult: ...
    @abstractmethod
    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult: ...
```

**Built-in providers:** `DeterministicEmbeddingProvider` (deterministic, SHA-256-based, no ML deps) and `MockEmbeddingProvider` (random vectors, testing only).

**Example:**

```python
config = EmbeddingConfig(provider_name="deterministic", dimensions=768)
provider = DeterministicEmbeddingProvider(config)
result = await provider.embed("What is the capital of France?")
vector = result.embeddings[0].vector
```

---

## `app.rag.vectorstore`

### Purpose

Stores and searches embedding vectors. Provides an abstract `VectorStore` base class and an in-memory implementation.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `VectorStore` | ABC | Abstract base for vector stores |
| `VectorStoreConfig` | Frozen dataclass | Metric, max vectors, dimension validation |
| `SearchResult` | Frozen dataclass | Result with chunk_id, score, vector |
| `SimilarityMetric` | Enum | COSINE, DOT_PRODUCT, NEGATIVE_EUCLIDEAN |
| `MemoryVectorStore` | Class | In-memory dict-backed implementation |
| `VectorStoreError` | Exception | Base vector store error |
| `VectorStoreFullError` | Exception | Capacity limit reached |
| `VectorDimensionMismatchError` | Exception | Wrong vector dimensions |
| `VectorNotFoundError` | Exception | Unknown chunk_id |
| `InvalidVectorStoreConfiguration` | Exception | Invalid config |
| `compute_similarity()` | Function | Compute similarity between two vectors |

### `VectorStore`

```python
class VectorStore(ABC):
    def __init__(self, config: VectorStoreConfig | None = None): ...
    @abstractmethod
    def add(self, chunk_id: str, vector: tuple[float, ...]) -> None: ...
    @abstractmethod
    def add_batch(self, items: Sequence[tuple[str, tuple[float, ...]]]) -> None: ...
    @abstractmethod
    def remove(self, chunk_id: str) -> bool: ...
    @abstractmethod
    def clear(self) -> None: ...
    @abstractmethod
    def get(self, chunk_id: str) -> tuple[float, ...] | None: ...
    @abstractmethod
    def contains(self, chunk_id: str) -> bool: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def search(self, query_vector: tuple[float, ...], top_k: int = 5) -> list[SearchResult]: ...
```

**Example:**

```python
store = MemoryVectorStore()
store.add("chunk_1", (0.1, 0.2, 0.3))
results = store.search((0.1, 0.2, 0.3), top_k=5)
for r in results:
    print(r.chunk_id, r.score)
```

---

## `app.rag.hybrid`

### Purpose

Combines keyword (lexical) and semantic (vector) retrieval results using configurable fusion strategies. The `DefaultHybridRetriever` orchestrates both retrievers.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `HybridRetriever` | ABC | Abstract base for hybrid retrievers |
| `DefaultHybridRetriever` | Class | Concrete hybrid implementation |
| `HybridConfig` | Frozen dataclass | Fusion strategy, weights, max candidates |
| `HybridResult` | Frozen dataclass | Ranked results with scores |
| `RetrievalScore` | Frozen dataclass | Per-result scores (keyword, semantic, final) |
| `FusionStrategy` | Enum | WEIGHTED_SUM, RECIPROCAL_RANK_FUSION |
| `HybridError` | Exception | Base hybrid error |
| `InvalidHybridConfiguration` | Exception | Invalid config |
| `FusionError` | Exception | Fusion failure |
| `reciprocal_rank_fusion()` | Function | RRF fusion algorithm |
| `weighted_sum()` | Function | Weighted sum fusion algorithm |

### `DefaultHybridRetriever`

```python
class DefaultHybridRetriever(HybridRetriever):
    def __init__(self, knowledge_base, keyword_retriever,
                 config: HybridConfig | None = None): ...
    async def retrieve(self, query: str, top_k: int = 5) -> HybridResult: ...
```

**Behaviour:**

1. Executes keyword retrieval via `KnowledgeRetriever`
2. If an embedding provider and vector store exist, executes semantic retrieval
3. Fuses results using the configured strategy (weighted sum or RRF)
4. Returns a ranked `HybridResult` with per-result scores

**Example:**

```python
from app.rag.hybrid import DefaultHybridRetriever
from app.rag.retriever import KnowledgeRetriever

retriever = DefaultHybridRetriever(knowledge_base, KnowledgeRetriever(knowledge_base))
result = await retriever.retrieve("capital of France", top_k=10)
for r in result.results:
    print(r.chunk_id, r.final_score)
```

---

## `app.rag.rerank`

### Purpose

Re-orders retrieval results using a secondary scoring model. The `DefaultReranker` uses lightweight text heuristics (no external models).

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `Reranker` | ABC | Abstract base for rerankers |
| `DefaultReranker` | Class | Heuristic-based deterministic reranker |
| `RerankConfig` | Frozen dataclass | enabled, top_k, score_threshold |
| `RerankResponse` | Frozen dataclass | List of `RerankedResult` + metadata |
| `RerankedResult` | Frozen dataclass | chunk_id, original_score, rerank_score, final_score |
| `RerankError` | Exception | Base rerank error |
| `InvalidRerankConfiguration` | Exception | Invalid config |
| `RerankerNotFound` | Exception | Unknown name in registry |
| `register_reranker()` | Function | Register a reranker class |
| `get_reranker()` | Function | Look up a reranker class |
| `list_rerankers()` | Function | List registered names |
| `clear_rerankers()` | Function | Clear registry (tests) |

### `DefaultReranker`

```python
class DefaultReranker(Reranker):
    def __init__(self, config: RerankConfig | None = None, *,
                 content_provider: Callable[[str], str | None] | None = None,
                 rerank_weight: float = 1.0,
                 length_penalty_exponent: float = 0.3,
                 phrase_bonus: float = 0.5): ...
    async def rerank(self, query: str, results: list[tuple[str, float]]) -> RerankResponse: ...
    def score(self, query: str, chunk_content: str,
              original_score: float = 0.0) -> RerankedResult: ...
```

**Scoring heuristics:**
- Lexical overlap — fraction of query terms present in the chunk
- Length penalty — chunks far from 200 chars score lower
- Exact phrase bonus — bonus if the exact query appears as substring
- `final_score = original_score + rerank_weight × rerank_score`

**Example:**

```python
reranker = DefaultReranker()
results = [("chunk_a", 0.9), ("chunk_b", 0.8), ("chunk_c", 0.5)]
response = await reranker.rerank("capital of France", results)
for r in response.results:
    print(r.chunk_id, r.final_score)
```
