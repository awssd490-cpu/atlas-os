# Persistence Layer (`app.rag.persistence`)

## Overview

Provides a pluggable persistence framework for saving and loading the state of a `KnowledgeBase` (documents, chunks, embeddings, vectors). The architecture mirrors the other RAG subsystems: an abstract base class, a concrete JSON implementation, a global registry, and frozen dataclass models.

## Architecture

```
PersistenceBackend (ABC)
    │
    ├── save(path, data) → PersistenceResult
    ├── load(path) → PersistenceResult      (KB in metadata)
    ├── update(path, data) → PersistenceResult  (incremental)
    ├── exists(path) → bool
    ├── delete(path) → PersistenceResult
    └── stats(path) → PersistenceStats
            │
            └── JsonPersistenceBackend  (concrete implementation)
```

## Public API

| Symbol | Kind | Description |
|---|---|---|
| `PersistenceBackend` | ABC | Abstract base |
| `JsonPersistenceBackend` | Class | JSON file backend |
| `PersistenceConfig` | Frozen dataclass | compress, overwrite, include_embeddings, include_vectors |
| `PersistenceResult` | Frozen dataclass | success + metadata dict |
| `PersistenceStats` | Frozen dataclass | documents, chunks, embeddings, vectors, size_bytes |
| `PersistenceError` | Exception | Base persistence error |
| `InvalidPersistenceConfiguration` | Exception | Config validation |
| `PersistenceNotFound` | Exception | Unknown backend name |
| `register()` / `get()` / `list_backends()` / `clear_backends()` | Functions | Backend type registry |

---

## `JsonPersistenceBackend`

### Purpose

Serialises a `KnowledgeBase` to/from a deterministic JSON file. No external dependencies — uses stdlib `json` only.

### JSON format

```json
{
  "version": 1,
  "documents": [
    {
      "document_id": "doc_1",
      "title": "Paris",
      "content": "Paris is the capital of France.",
      "content_length": 35,
      "chunk_count": 2,
      "metadata": { "source": "wiki", "tags": ["geography"] }
    }
  ],
  "chunks": [
    {
      "chunk_id": "doc_1:0",
      "document_id": "doc_1",
      "content": "Paris is...",
      "index": 0,
      "metadata": { "source": "wiki", "tags": ["geography"] }
    }
  ],
  "embeddings": [
    {
      "chunk_id": "doc_1:0",
      "vector": [0.1, 0.2, ...],
      "dimensions": 768,
      "provider": "openai",
      "metadata": {}
    }
  ],
  "vectors": [
    {
      "chunk_id": "doc_1:0",
      "vector": [0.1, 0.2, ...]
    }
  ],
  "metadata": {
    "saved_at": "2026-07-30T12:00:00+00:00",
    "document_count": 1,
    "chunk_count": 2,
    "embedding_count": 1,
    "vector_count": 1
  }
}
```

Written with `indent=2`, `sort_keys=True`, `ensure_ascii=False`.

### Methods

```python
class JsonPersistenceBackend(PersistenceBackend):
    CURRENT_VERSION: int = 1
    SUPPORTED_VERSIONS: tuple[int, ...] = (1,)

    async def save(self, path: str, data: object, **kwargs) -> PersistenceResult: ...
    async def load(self, path: str, **kwargs) -> PersistenceResult: ...
    async def update(self, path: str, data: object, **kwargs) -> PersistenceResult: ...
    async def exists(self, path: str, **kwargs) -> bool: ...
    async def delete(self, path: str, **kwargs) -> PersistenceResult: ...
    async def stats(self, path: str, **kwargs) -> PersistenceStats: ...
```

### `save()` behaviour

- Validates `data` is a `KnowledgeBase` instance
- Checks `overwrite` flag — raises `PersistenceError` if `overwrite=False` and file exists
- Serialises documents (sorted by `document_id`), chunks (sorted by `document_id` then `index`), embeddings (sorted by `chunk_id`), and vectors (sorted by `chunk_id`)
- Writes atomically via temp file + `os.replace()`
- Returns `PersistenceResult` with `size_bytes`, `elapsed_time`, and item counts

### `load()` behaviour

- Validates file exists, JSON is well-formed, root is an object
- Validates version is in `SUPPORTED_VERSIONS`
- Validates `documents` and `chunks` fields exist and are lists
- Validates no duplicate `document_id` or `chunk_id` values
- Validates embeddings/vectors reference known chunks
- Reconstructs `KnowledgeBase` with all documents, chunks, embeddings, and vector store
- Returns `PersistenceResult` with `knowledge_base` in `metadata`

### `update()` behaviour

- If the snapshot doesn't exist, falls back to `save()`
- Otherwise, loads existing snapshot, diffs document IDs
- Reports `added_documents`, `removed_documents`, `updated_documents` (by title or content change)
- Writes the full current KB state atomically (the diff is informational, not incremental storage)

### Validation errors (load)

| Condition | Error |
|---|---|
| File does not exist | `PersistenceError("File does not exist...")` |
| Invalid JSON | `PersistenceError("Failed to parse...")` |
| Root is not a dict | `PersistenceError("JSON root must be an object")` |
| Missing version | `PersistenceError("Missing required field: version")` |
| Unsupported version | `PersistenceError("Unsupported snapshot version...")` |
| Missing documents | `PersistenceError("Missing required field: documents")` |
| Missing chunks | `PersistenceError("Missing required field: chunks")` |
| Duplicate document_id | `PersistenceError("Duplicate document_id...")` |
| Duplicate chunk_id | `PersistenceError("Duplicate chunk_id...")` |
| Embedding references unknown chunk | `PersistenceError("references unknown chunk...")` |
| Vector references unknown chunk | `PersistenceError("references unknown chunk...")` |

### Example

```python
backend = JsonPersistenceBackend(PersistenceConfig(overwrite=True))

# Save
result = await backend.save("snapshot.json", knowledge_base)
print(f"Saved {result.metadata['size_bytes']} bytes")

# Load
result = await backend.load("snapshot.json")
kb = result.metadata["knowledge_base"]

# Incremental update
result = await backend.update("snapshot.json", updated_kb)
print(result.metadata["changes"])  # {"added_documents": 1, "removed_documents": 0, ...}

# Stats
stats = await backend.stats("snapshot.json")
print(f"{stats.documents} documents, {stats.size_bytes} bytes")
```

---

## Best practices

- **Always set `overwrite=True`** when calling `save()` multiple times on the same path, or handle the `PersistenceError` for the first call.
- **Use `update()` for incremental changes** — it detects added/removed/modified documents and writes atomically.
- **Call `stats()` before `load()`** to check whether a snapshot exists and how large it is.
- **Set `include_embeddings=False` or `include_vectors=False`** when you don't need them to reduce file size.
- **The `version` field enables future migrations** — bump `CURRENT_VERSION` and add upgrade logic in `load()`.
