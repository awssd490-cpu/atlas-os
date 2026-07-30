# Saving and Loading Pipelines

## Goal

Persist a knowledge base to a JSON file, reload it later, and incrementally update the snapshot.

## Prerequisites

- A working pipeline from [Building a RAG Pipeline](rag_pipeline.md)
- Understanding of `JsonPersistenceBackend`

## Step-by-step guide

### 1. Create a backend

```python
from app.rag.persistence import (
    JsonPersistenceBackend,
    PersistenceConfig,
)

config = PersistenceConfig(
    overwrite=True,            # allow overwriting existing files
    include_embeddings=True,   # include embedding vectors
    include_vectors=True,      # include vector store entries
)
backend = JsonPersistenceBackend(config)
```

### 2. Save a knowledge base

```python
import asyncio

async def demo():
    # Assume kb is a populated KnowledgeBase
    result = await backend.save("snapshot.json", kb)
    print(f"Saved: {result.metadata['size_bytes']} bytes")
    print(f"  Documents: {result.metadata['documents']}")
    print(f"  Chunks: {result.metadata['chunks']}")
    print(f"  Embeddings: {result.metadata['embeddings']}")
    print(f"  Vectors: {result.metadata['vectors']}")

asyncio.run(demo())
```

### 3. Load a knowledge base

```python
result = await backend.load("snapshot.json")
kb2 = result.metadata["knowledge_base"]

print(f"Loaded {kb2.count()} documents")
doc = kb2.get("paris")
if doc:
    print(f"  Title: {doc.title}")
    print(f"  Chunks: {doc.chunk_count}")
```

### 4. Check for a snapshot before loading

```python
if await backend.exists("snapshot.json"):
    stats = await backend.stats("snapshot.json")
    print(f"Snapshot: {stats.documents} docs, {stats.size_bytes} bytes")
    result = await backend.load("snapshot.json")
    kb = result.metadata["knowledge_base"]
else:
    print("No snapshot found, creating new KB")
    kb = KnowledgeBase()
```

### 5. Incremental update

When you've added or removed documents, use `update()` instead of a full save:

```python
# Add a new document
new_doc = KnowledgeDocument(document_id="berlin", title="Berlin",
    content="Berlin is the capital of Germany.")
kb.add_document(new_doc)

result = await backend.update("snapshot.json", kb)
changes = result.metadata["changes"]
print(f"Added: {changes['added_documents']}")
print(f"Updated: {changes['updated_documents']}")
print(f"Removed: {changes['removed_documents']}")
```

### 6. Delete a snapshot

```python
result = await backend.delete("snapshot.json")
print(f"Deleted: {result.success}")
```

## JSON file structure

The saved file is a deterministic JSON document:

```json
{
  "version": 1,
  "documents": [...],
  "chunks": [...],
  "embeddings": [...],
  "vectors": [...],
  "metadata": {
    "saved_at": "...",
    "document_count": 1,
    "chunk_count": 3,
    "embedding_count": 3,
    "vector_count": 3
  }
}
```

## Overwrite protection

By default, `overwrite=False`. Calling `save()` on an existing path raises `PersistenceError`:

```python
backend = JsonPersistenceBackend(PersistenceConfig(overwrite=False))
try:
    await backend.save("existing.json", kb)
except app.rag.persistence.errors.PersistenceError as exc:
    print(f"Target already exists: {exc}")
    # Re-save with overwrite=True or use update()
```

## Complete example

See `examples/persistence_demo.py` for the full runnable example.

## Expected output

```
Saved: 2048 bytes
Loaded 3 documents
Snapshot: 3 docs, 2048 bytes
Added: 1
Updated: 0
Removed: 0
Deleted: True
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `PERSISTENCE_TARGET_EXISTS` | File exists, `overwrite=False` | Set `overwrite=True` or use `update()` |
| `PERSISTENCE_PATH_NOT_FOUND` | `load()` on missing file | Call `exists()` first |
| Corrupted JSON after load | Version mismatch | Create fresh snapshot |

## Next steps

- [Evaluate retrieval quality](evaluation.md)
- [Advanced patterns](advanced.md)
