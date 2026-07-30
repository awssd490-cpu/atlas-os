# Getting Started with Atlas

## Goal

Install Atlas, understand the architecture, and run a minimal retrieval example end-to-end.

## Prerequisites

- Python 3.12+
- `pip`

No external ML dependencies required — Atlas uses the built-in `DeterministicEmbeddingProvider` for development.

## Step-by-step guide

### 1. Install Atlas

Atlas is structured as a Python application. Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/atlas.git
cd atlas
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### 2. Configure logging

Atlas uses structured JSON logging via `AtlasLogger`:

```python
from app.core.log import AtlasLogger

log = AtlasLogger("getting-started", level="INFO")
log.info("Atlas initialized", version="1.0")
```

Output (stderr):
```json
{"level":"INFO","logger":"getting-started","message":"Atlas initialized","metadata":{"version":"1.0"},"timestamp":"2026-07-30T12:00:00+00:00"}
```

### 3. Create a KnowledgeBase

A `KnowledgeBase` is the central document registry:

```python
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeDocument, KnowledgeChunk

kb = KnowledgeBase()

doc = KnowledgeDocument(
    document_id="doc_1",
    title="Paris",
    content="Paris is the capital of France. It is known for the Eiffel Tower.",
    chunks=(
        KnowledgeChunk(
            chunk_id="doc_1:0",
            document_id="doc_1",
            content="Paris is the capital of France.",
            index=0,
        ),
        KnowledgeChunk(
            chunk_id="doc_1:1",
            document_id="doc_1",
            content="It is known for the Eiffel Tower.",
            index=1,
        ),
    ),
)
kb.register(doc)
print(f"Registered {kb.count()} document")
```

### 4. Set up embedding and vector store

Atlas ships with a deterministic embedding provider ideal for development:

```python
from app.rag.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingConfig,
)
from app.rag.vectorstore import MemoryVectorStore

embed_config = EmbeddingConfig(
    provider_name="deterministic",
    dimensions=4,
    normalize_embeddings=True,
)
provider = DeterministicEmbeddingProvider(embed_config)
store = MemoryVectorStore()

# Embed every chunk
import asyncio

async def index_chunks(kb, provider, store):
    for chunk in kb.list_chunks():
        result = await provider.embed(chunk.content)
        vec = result.embeddings[0]
        kb._embeddings[chunk.chunk_id] = vec  # store in KB
        store.add(chunk.chunk_id, vec.vector)

asyncio.run(index_chunks(kb, provider, store))
print(f"Indexed {store.count()} vectors")
```

### 5. Retrieve relevant content

```python
from app.rag.retriever import KnowledgeRetriever
from app.rag.models import KnowledgeQuery

retriever = KnowledgeRetriever(kb)
query = KnowledgeQuery(query="capital of France", max_results=5)
result = asyncio.run(retriever.retrieve(query))

print(f"Found {result.total} matches:")
for chunk in result.chunks:
    print(f"  - {chunk.content}")
```

### 6. Format for provider context

```python
from app.rag.context import KnowledgeContextBuilder

builder = KnowledgeContextBuilder(kb, retriever)
context = asyncio.run(builder.build("capital of France"))
print(context.text)
```

## Complete example

The complete example is available in `examples/basic_rag.py`. Run it with:

```bash
python examples/basic_rag.py
```

## Expected output

```
Registered 1 document
Indexed 2 vectors
Found 1 matches:
  - Paris is the capital of France.
Relevant knowledge:
- Paris is the capital of France.
- It is known for the Eiffel Tower.
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Package not installed | `pip install -e .` |
| No matches found | Empty knowledge base | Check `kb.count()` |
| Chunks not indexed | Embedding not called | Call `provider.embed()` for each chunk |

## Next steps

- [Build a full RAG pipeline](rag_pipeline.md)
- [Save and load your knowledge base](persistence.md)
- [Evaluate retrieval quality](evaluation.md)
