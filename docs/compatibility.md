# Compatibility Reference

## Supported Python versions

| Python | Status | Notes |
|---|---|---|
| 3.12 | ✅ Full support | Primary development target |
| 3.13 | ✅ Full support | All tests passing |
| 3.11 | ⚠️ Untested | May work — no guarantees |
| < 3.11 | ❌ Not supported | Requires `X | Y` union syntax, `datetime.UTC` |

## Operating systems

| OS | Status | Notes |
|---|---|---|
| Linux (x86_64) | ✅ Full support | CI-tested |
| Windows (AMD64) | ✅ Full support | CI-tested; atomic writes use `os.replace()` |
| macOS (ARM64) | ✅ Full support | Developer machines |
| WSL2 | ✅ Full support | Ubuntu on WSL2 |

### Platform notes

| Feature | Linux | Windows | macOS |
|---|---|---|---|
| `os.replace()` atomic write | ✅ Yes | ✅ Yes | ✅ Yes |
| `time.perf_counter()` | ✅ Monotonic | ✅ Monotonic (QueryPerformanceCounter) | ✅ Monotonic |
| `tracemalloc` | ✅ Full | ✅ Full | ✅ Full |
| `asyncio` event loop | `SelectorEventLoop` | `ProactorEventLoop` (default) | `SelectorEventLoop` |
| `pathlib` | ✅ | ✅ (careful with backslashes) | ✅ |
| JSON filenames | `snapshot.json` | `snapshot.json` | `snapshot.json` |
| Temp file paths | Uses system tmp | Uses `%TEMP%` | Uses system tmp |

## Filesystem considerations

### Path separators

All Atlas code uses `pathlib.Path` for cross-platform path handling. Hardcoded forward slashes in string paths may fail on Windows.

```python
# Correct — cross-platform
from pathlib import Path
path = Path("data") / "documents" / "file.txt"

# Avoid — Windows-incompatible
path = "data/documents/file.txt"
```

### File permissions

- `JsonPersistenceBackend.save()` writes with default file permissions (umask-respected on POSIX)
- Configuration files should be readable by the running user
- Snapshot files default to `overwrite=False` — remove or rename manually if needed

### Unicode filenames

All Atlas I/O uses UTF-8 encoding:
- `open(..., encoding="utf-8")` for text files
- `json.dumps(ensure_ascii=False)` for JSON output
- `pathlib.Path.read_bytes()/write_bytes()` for raw data

## Optional dependencies

Atlas has **zero required runtime dependencies** beyond Python 3.12+. The following are optional and only needed for specific use cases:

| Package | Used for | When to install |
|---|---|---|
| `coverage` | Test coverage reports | `pip install coverage` |
| `pytest` | Running the test suite | Included in dev install |
| `pytest-asyncio` | Async test support | Included in dev install |

## Unicode support

| Component | Support | Notes |
|---|---|---|
| `KnowledgeDocument.content` | ✅ Full | UTF-8 stored as Python str |
| `KnowledgeChunk.content` | ✅ Full | Same as above |
| `JsonPersistenceBackend` | ✅ Full | `ensure_ascii=False` |
| `AtlasLogger` | ✅ Full | Unicode messages and metadata |
| `DatasetLoader` | ✅ Full | Unicode queries and IDs |
| `RetrievalMetrics` | ✅ Full | Operates on any hashable IDs |
| `ConfigLoader` | ✅ Full | UTF-8 JSON files |
| `HealthMonitor` | ✅ Limited | Check names are ASCII-safe (best practice) |
| `ResourceManager` | ✅ Limited | Resource names are ASCII-safe (best practice) |

### Unicode best practices

```python
# All of these work correctly
doc = KnowledgeDocument(
    document_id="doc_1",                              # ASCII
    title="東京の首都",                                # Unicode
    content="東京は日本の首都です。",                   # Unicode
)

# JSON round-trip preserves unicode
await backend.save("snapshot.json", kb)
result = await backend.load("snapshot.json")
loaded = result.metadata["knowledge_base"]
assert "東京" in loaded.get("doc_1").title  # True
```

## Compatibility matrix: Key features

| Feature | app.core.config | app.core.log | app.core.health | app.core.concurrency | app.core.reliability |
|---|---|---|---|---|---|
| Requires network | No | No | No | No | No |
| Requires filesystem | Read (JSON) | Write (stderr) | No | No | No |
| Requires threading | No | No | No | No | No |
| Thread-safe | Yes* | Yes* | No | No | Yes* |
| Async-safe | N/A | N/A | Yes | Yes | Yes |
| Stdlib only | Yes | Yes | Yes | Yes | Yes |

> *: Frozen dataclass instances are thread-safe; mutating methods are not.

| Feature | Pipeline | Persistence | Evaluation |
|---|---|---|---|
| Requires network | No | No | No |
| Requires filesystem | Via loader | Yes (JSON files) | Yes (datasets) |
| Stdlib only | Yes | Yes | Yes |
| Async-safe | Yes | Yes | Yes |

## Backward compatibility promises

| Surface | Stable in MAJOR | Notes |
|---|---|---|
| `__all__` exports | ✅ | Removal = MAJOR bump |
| ABC method signatures | ✅ | New abstract methods = MINOR, must be implemented |
| Frozen dataclass fields | ✅ | New fields with defaults = MINOR |
| Error `code` strings | ✅ | Code values are stable within MAJOR |
| JSON snapshot format | ✅ | Version field enables migration |
| CLI commands | ✅ | New commands = MINOR; removal = MAJOR |
| Registry function names | ✅ | `register`/`unregister`/`get`/`list`/`clear` |

## Environment variable reference

| Variable | Used by | Default | Example |
|---|---|---|---|
| `ATLAS_ENVIRONMENT` | `ConfigLoader.from_env()` | `"development"` | `"production"` |
| `ATLAS_DEBUG` | `ConfigLoader.from_env()` | `False` | `"true"` |
| `ATLAS_LOG_LEVEL` | `ConfigLoader.from_env()` | `"INFO"` | `"DEBUG"` |
| `ATLAS_RANDOM_SEED` | `ConfigLoader.from_env()` | `42` | `"7"` |
