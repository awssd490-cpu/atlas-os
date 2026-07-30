# RAG Layer (`app.rag`)

## Overview

The RAG layer provides the core knowledge retrieval abstractions: domain models, knowledge base, context builder, and the orchestration pipeline. It depends on the provider layer (chunking, embeddings, vector store, hybrid, reranking).

## Module relationships

```
app.rag
├── models.py                 Shared domain models (KnowledgeDocument, etc.)
├── errors.py                 Shared error hierarchy
├── knowledge_base.py         Central document registry
├── retriever.py              Keyword retriever
├── context.py                Merges retrieval into provider context
└── pipeline/
    ├── base.py               KnowledgePipeline ABC
    ├── config.py             PipelineConfig
    ├── models.py             PipelineResult, PipelineStats
    ├── errors.py             PipelineError hierarchy
    ├── registry.py           Pipeline type registry
    ├── builder.py            PipelineBuilder (fluent)
    └── default.py            DefaultKnowledgePipeline
```

---

## `app.rag.models`

### Purpose

Canonical, immutable data types shared across all RAG subsystems.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `KnowledgeDocument` | Frozen dataclass | A document with ID, title, content, chunks, metadata |
| `KnowledgeChunk` | Frozen dataclass | A single chunk with ID, content, index, metadata |
| `KnowledgeMetadata` | Frozen dataclass | Metadata (source, author, tags, etc.) |
| `KnowledgeSource` | Frozen dataclass | Provenance for a retrieved chunk |
| `KnowledgeQuery` | Frozen dataclass | Query with max_results, min_score, filters |
| `KnowledgeResult` | Frozen dataclass | Retrieval result with chunks, sources, timing |
| `KnowledgeContext` | Frozen dataclass | Formatted context text for provider injection |

### `KnowledgeDocument`

```python
@dataclass(frozen=True)
class KnowledgeDocument:
    document_id: str = ""
    title: str = ""
    content: str = ""
    chunks: tuple[KnowledgeChunk, ...] = ()
    metadata: KnowledgeMetadata = field(default_factory=KnowledgeMetadata)

    @property
    def chunk_count(self) -> int: ...
    @property
    def content_length(self) -> int: ...
    def to_dict(self) -> dict[str, Any]: ...
```

### `KnowledgeChunk`

```python
@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str = ""
    document_id: str = ""
    content: str = ""
    index: int = 0
    metadata: KnowledgeMetadata = field(default_factory=KnowledgeMetadata)
```

---

## `app.rag.errors`

### Purpose

Shared error hierarchy for all RAG subsystems.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `KnowledgeError` | Exception | Base, inherits `AtlasError` |
| `DuplicateDocumentError` | Exception | Document ID already registered |
| `DocumentNotFoundError` | Exception | Document ID not found |

Both errors support `code` ("KNOWLEDGE_DUPLICATE_DOCUMENT", "KNOWLEDGE_DOCUMENT_NOT_FOUND") and `details` dict.

---

## `app.rag.knowledge_base`

### Purpose

Central registry for `KnowledgeDocument` objects. Supports registration (with or without automatic chunking), removal, and lookup. Integrates with `EmbeddingProvider` and `VectorStore` for automatic embedding/indexing.

### Public API

```python
class KnowledgeBase:
    def __init__(self, chunking_config=None, embedding_provider=None,
                 vector_store=None, reranker=None): ...

    # Properties
    @property
    def chunking_config(self) -> ChunkingConfig: ...
    @property
    def embedding_provider(self) -> EmbeddingProvider | None: ...
    @property
    def vector_store(self) -> VectorStore | None: ...
    @property
    def reranker(self) -> Reranker | None: ...
    @property
    def hybrid_retriever(self) -> DefaultHybridRetriever | None: ...

    # Registration
    def register(self, document: KnowledgeDocument) -> KnowledgeDocument: ...
    def add_document(self, document: KnowledgeDocument, *, config=None) -> KnowledgeDocument: ...
    def remove(self, document_id: str) -> bool: ...

    # Lookup
    def get(self, document_id: str) -> KnowledgeDocument | None: ...
    def exists(self, document_id: str) -> bool: ...
    def get_chunk(self, chunk_id: str) -> KnowledgeChunk | None: ...
    def get_embedding(self, chunk_id: str) -> EmbeddingVector | None: ...

    # Enumeration
    def list_documents(self) -> list[KnowledgeDocument]: ...
    def list_chunks(self) -> list[KnowledgeChunk]: ...
    def list_embeddings(self) -> list[EmbeddingVector]: ...
    def count(self) -> int: ...

    # Lifecycle
    def clear(self) -> None: ...
```

**Behaviour:**

- `register()` stores a document as-is with its existing chunks.
- `add_document()` automatically chunks content through the `ChunkingEngine`, generates embeddings (if configured), and inserts into the vector store (if configured).
- `hybrid_retriever` is lazily constructed — it returns `None` unless both an embedding provider and vector store are configured.

**Example:**

```python
kb = KnowledgeBase(
    embedding_provider=provider,
    vector_store=store,
)
doc = KnowledgeDocument(document_id="doc_1", title="Paris", content="Paris is...")
kb.add_document(doc)
found = kb.get("doc_1")
```

---

## `app.rag.context`

### Purpose

`KnowledgeContextBuilder` retrieves knowledge and formats it for provider injection. It auto-detects hybrid retrieval when the knowledge base has both embeddings and a vector store, and applies a configured reranker automatically.

### Public API

```python
class KnowledgeContextBuilder:
    def __init__(self, knowledge_base=None, retriever=None): ...
    async def build(self, query="", *, max_chunks=10, min_score=0.0,
                    format_as="text") -> KnowledgeContext: ...
```

---

## `app.rag.pipeline`

### Purpose

Orchestrates the end-to-end knowledge flow: ingestion (load → chunk → register → embed → index) and search (query → retrieve → rerank → format). Provides an abstract base, a concrete default implementation, and a fluent builder.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `KnowledgePipeline` | ABC | Abstract base with ingest/search/clear/stats |
| `DefaultKnowledgePipeline` | Class | Concrete pipeline |
| `PipelineBuilder` | Class | Fluent builder |
| `PipelineConfig` | Frozen dataclass | auto_embed, auto_index, auto_rerank, batch_size |
| `PipelineResult` | Frozen dataclass | context + metadata dict |
| `PipelineStats` | Frozen dataclass | documents, chunks, vectors, searches |
| `PipelineError` | Exception | Base pipeline error |
| `InvalidPipelineConfiguration` | Exception | Configuration validation |
| `PipelineNotFound` | Exception | Unknown name in registry |
| `register()` / `get()` / `list_pipelines()` / `clear_pipelines()` | Functions | Pipeline type registry |

### `DefaultKnowledgePipeline`

```python
class DefaultKnowledgePipeline(KnowledgePipeline):
    def __init__(self, loader, chunker, knowledge_base,
                 embedding_provider=None, vector_store=None,
                 config=None): ...
    async def ingest(self, path: str, **kwargs) -> int: ...
    async def ingest_documents(self, documents: list[KnowledgeDocument], **kwargs) -> PipelineResult: ...
    async def search(self, query: str, **kwargs) -> PipelineResult: ...
    async def clear(self, **kwargs) -> None: ...
    async def stats(self, **kwargs) -> PipelineStats: ...
```

### `PipelineBuilder`

```python
class PipelineBuilder:
    def loader(self, loader) -> PipelineBuilder: ...
    def chunker(self, chunker: ChunkingEngine) -> PipelineBuilder: ...
    def knowledge_base(self, knowledge_base: KnowledgeBase) -> PipelineBuilder: ...
    def embedding_provider(self, provider) -> PipelineBuilder: ...
    def vector_store(self, store) -> PipelineBuilder: ...
    def reranker(self, reranker) -> PipelineBuilder: ...
    def config(self, config: PipelineConfig) -> PipelineBuilder: ...
    def build(self) -> DefaultKnowledgePipeline: ...
```

**Raises:**

- `InvalidPipelineConfiguration` if `build()` is called without `loader`, `chunker`, or `knowledge_base`.

**Example:**

```python
pipeline = (
    PipelineBuilder()
    .loader(my_loader)
    .chunker(ChunkingEngine())
    .knowledge_base(KnowledgeBase())
    .embedding_provider(provider)
    .vector_store(store)
    .config(PipelineConfig(auto_embed=True, batch_size=16))
    .build()
)

count = await pipeline.ingest("/path/to/docs")
result = await pipeline.search("capital of France")
print(result.context)
```
