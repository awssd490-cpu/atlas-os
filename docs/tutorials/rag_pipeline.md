# Building a RAG Pipeline

## Goal

Create a `DefaultKnowledgePipeline` with a custom loader, configure hybrid retrieval with reranking, and run searches.

## Prerequisites

- Understanding of [Getting Started](getting_started.md)
- Familiarity with the provider layer concepts (chunking, embeddings, vector store)

## Step-by-step guide

### 1. Create a loader

A loader is any callable that accepts a path string and returns `list[KnowledgeDocument]`:

```python
from app.rag.models import KnowledgeDocument, KnowledgeChunk

def my_loader(path: str) -> list[KnowledgeDocument]:
    """Load documents from a directory or file."""
    documents = []
    if path == "demo":
        documents = [
            KnowledgeDocument(
                document_id="paris",
                title="Paris",
                content="Paris is the capital of France. The Eiffel Tower is a famous landmark.",
            ),
            KnowledgeDocument(
                document_id="london",
                title="London",
                content="London is the capital of the UK. The Thames flows through London.",
            ),
            KnowledgeDocument(
                document_id="tokyo",
                title="Tokyo",
                content="Tokyo is the capital of Japan. It is a bustling metropolis.",
            ),
        ]
    return documents
```

### 2. Configure chunking

```python
from app.rag.chunking import ChunkingEngine, ChunkingConfig

chunk_config = ChunkingConfig(
    strategy="sentence",           # split on sentence boundaries
    min_chunk_size=1,
)
chunker = ChunkingEngine(config=chunk_config)
```

### 3. Configure embedding and vector store

```python
from app.rag.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingConfig,
)
from app.rag.vectorstore import MemoryVectorStore

embedding_config = EmbeddingConfig(
    provider_name="det",
    dimensions=4,
    normalize_embeddings=True,
)
embedding_provider = DeterministicEmbeddingProvider(embedding_config)
vector_store = MemoryVectorStore()
```

### 4. Build the pipeline

Use the fluent `PipelineBuilder`:

```python
from app.rag.pipeline import PipelineBuilder, PipelineConfig
from app.rag.knowledge_base import KnowledgeBase

pipeline = (
    PipelineBuilder()
    .loader(my_loader)
    .chunker(chunker)
    .knowledge_base(KnowledgeBase())
    .embedding_provider(embedding_provider)
    .vector_store(vector_store)
    .config(PipelineConfig(
        auto_embed=True,
        auto_index=True,
        batch_size=10,
    ))
    .build()
)
```

### 5. Ingest documents

```python
import asyncio

async def main():
    count = await pipeline.ingest("demo")
    print(f"Ingested {count} documents")

    stats = await pipeline.stats()
    print(f"Stats: {stats}")

asyncio.run(main())
```

### 6. Add a reranker

Reranking improves result ordering. Attach a reranker to the KnowledgeBase:

```python
from app.rag.rerank import DefaultReranker, RerankConfig

reranker = DefaultReranker(
    config=RerankConfig(enabled=True, top_k=5),
    rerank_weight=1.0,
)

# Attach via the builder
pipeline = (
    PipelineBuilder()
    .loader(my_loader)
    .chunker(chunker)
    .knowledge_base(KnowledgeBase())
    .embedding_provider(embedding_provider)
    .vector_store(vector_store)
    .reranker(reranker)                   # <-- added
    .config(PipelineConfig(auto_embed=True, auto_index=True))
    .build()
)
```

### 7. Search with reranking

```python
result = await pipeline.search("capital of France")
print(f"Mode: {result.metadata['retrieval_mode']}")
print(f"Reranking: {result.metadata['reranking_enabled']}")
print(result.context)
```

The pipeline automatically selects:

- **Hybrid retrieval** when both embedding provider and vector store are present
- **Keyword-only retrieval** when embeddings are not configured
- **Reranking** when a reranker is attached to the knowledge base

### 8. Inspect statistics

```python
stats = await pipeline.stats()
print(f"Documents: {stats.documents}")
print(f"Chunks: {stats.chunks}")
print(f"Vectors: {stats.vectors}")
print(f"Searches: {stats.searches}")
```

## Complete example

See `examples/basic_rag.py` for the full runnable example.

## Expected output

```text
Ingested 3 documents
Mode: hybrid
Reranking: True
Relevant knowledge:
- Paris is the capital of France. The Eiffel Tower is a famous landmark.
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `retrieval_mode` shows `keyword` | Embeddings not configured | Check `embedding_provider` is set |
| Duplicate document error | Same ID ingested twice | Pipeline skips duplicates silently |
| No reranking | Reranker not attached | Check `reranker` parameter in builder |

## Next steps

- [Save and reload your pipeline](persistence.md)
- [Benchmark pipeline performance](evaluation.md)
- [Add custom error handling](advanced.md)
