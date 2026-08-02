# Troubleshooting Guide

## Quick diagnostic

Before diving into specific issues, run the diagnostic tool:

```bash
cd /path/to/atlas
PYTHONPATH=. python tools/atlas_cli.py doctor
```

This checks:

- Python version >= 3.12
- Core package importability
- Pytest availability
- Test directory existence

## Troubleshooting decision tree

```mermaid
flowchart TD
    A[Encountered an error?] --> B{What type?}
    B -->|Import| C[Import guide]
    B -->|Configuration| D[Config guide]
    B -->|Runtime| E[Runtime guide]
    B -->|Persistence| F[Persistence guide]
    B -->|Performance| G[Performance guide]
    C --> C1{Module not found?}
    C1 -->|"app.*"| C2[Run from repo root]
    C1 -->|Third-party| C3[Install dependency]
    D --> D1{Running validate-config?}
    D1 -->|Invalid value| D2[Check environment / log_level]
    D1 -->|File error| D3[Check path and permissions]
    E --> E1{Async error?}
    E1 -->|"got Future"| E2[Use await / run in event loop]
    E1 -->|"not awaitable"| E3[Component must be async or sync-wrapped]
    F --> F1{Overwrite error?}
    F1 -->|PERSISTENCE_TARGET_EXISTS| F2[Set overwrite=True or use update()]
```

## Installation issues

### ModuleNotFoundError: No module named 'app'

**Symptoms:** `ModuleNotFoundError: No module named 'app'`

**Causes:**

- Running Python from outside the repository root
- Package not installed in development mode

**Diagnostic:**

```bash
pwd                    # should be /path/to/atlas
python -c "import app" # should succeed
```

**Solutions:**

```bash
cd /path/to/atlas
pip install -e .
```

### ModuleNotFoundError: No module named 'app.core.log'

**Symptoms:** ImportError when importing a core subpackage.

**Causes:** The `app.core.log` package name shadows the stdlib `logging` module when not imported correctly.

**Solutions:**

```python
# Correct — import through the package
from app.core.log import AtlasLogger

# Also correct
from app.core.log.logger import AtlasLogger
```

## Import errors

### Symbol not found in `__init__.py`

**Symptoms:** `ImportError: cannot import name 'X' from 'app.rag.X'`

**Causes:**

- The symbol was added in a later version
- The symbol was removed or renamed

**Solutions:**

```bash
# Check available exports
python -c "from app.rag.pipeline import __all__; print(__all__)"
```

## Configuration errors

### InvalidConfiguration: Invalid environment

**Symptoms:**

```text
InvalidConfiguration: Invalid environment: 'invalid'. Must be one of ('development', 'testing', 'staging', 'production')
```

**Causes:** The `environment` field was set to a value outside the allowed set.

**Solution:** Use one of: `development`, `testing`, `staging`, `production`.

### InvalidConfiguration: Invalid log_level

**Symptoms:**

```text
InvalidConfiguration: Invalid log_level: 'TRACE'. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL
```

**Causes:** The `log_level` field was set to an unrecognised level.

**Solution:** Use one of: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.

## Provider registration failures

### UnsupportedEmbeddingProvider

**Symptoms:** `UnsupportedEmbeddingProvider: 'my_provider'`

**Causes:** The provider name is not registered in the global registry.

**Diagnostic:**

```bash
PYTHONPATH=. python -c "
from app.rag.embeddings import list_providers
print('Available:', list_providers())
"
```

**Solutions:**

```python
# Register before use
from app.rag.embeddings import register_provider
register_provider("my_provider", MyProviderClass)
```

### DuplicateHealthCheck

**Symptoms:** `DuplicateHealthCheck: 'my_check' is already registered`

**Causes:** A health check with the same name was registered twice.

**Solution:** Use `clear_checks()` or `unregister()` before re-registering:

```python
from app.core.health import unregister, register
try:
    unregister("my_check")
except:
    pass
register("my_check", my_fn)
```

## Async errors

### "RuntimeError: asyncio.run() cannot be called from a running event loop"

**Symptoms:** Error when calling `asyncio.run()` from inside a running event loop (e.g., a Jupyter notebook).

**Causes:** `asyncio.run()` must only be called from the main thread.

**Solutions:**

In scripts:

```python
async def main():
    result = await pipeline.search("query")

if __name__ == "__main__":
    asyncio.run(main())
```

In Jupyter / REPL:

```python
result = await pipeline.search("query")
```

### "TypeError: 'coroutine' object is not callable"

**Symptoms:** TypeError when calling an async function without await.

**Causes:** An async function was called but not awaited.

**Solution:**

```python
# Wrong
result = my_async_fn()

# Correct
result = await my_async_fn()
```

## Persistence failures

### PERSISTENCE_TARGET_EXISTS

**Symptoms:**

```text
PersistenceError: Target path already exists: snapshot.json
```

**Causes:** `save()` was called on an existing path with default `overwrite=False`.

**Solutions:**

```python
# Option 1: Allow overwrite
backend = JsonPersistenceBackend(PersistenceConfig(overwrite=True))

# Option 2: Use incremental update
result = await backend.update("snapshot.json", kb)

# Option 3: Delete first
await backend.delete("snapshot.json")
await backend.save("snapshot.json", kb)
```

### PERSISTENCE_PATH_NOT_FOUND

**Symptoms:**

```text
PersistenceError: Path does not exist: snapshot.json
```

**Causes:** `load()` was called on a non-existent file.

**Solution:**

```python
if await backend.exists("snapshot.json"):
    result = await backend.load("snapshot.json")
else:
    print("No snapshot found")
```

### Corrupted snapshot

**Symptoms:** `PersistenceError: Failed to parse JSON snapshot` or `Missing required field: version`

**Causes:** The file was truncated, hand-edited incorrectly, or produced by a different version.

**Solutions:**

```bash
# Check JSON validity
python -m json.tool snapshot.json

# Re-create the snapshot from source
await backend.save("snapshot.json", kb, overwrite=True)
```

## Benchmark failures

### Benchmark returns zero throughput

**Symptoms:** `throughput_qps = 0.0`

**Causes:** The total benchmark duration was 0 or very close to 0 (too few queries or too fast).

**Solutions:**

- Increase `benchmark_runs`
- Use a slower component (e.g., add `asyncio.sleep(0.001)`)
- Increase dataset size

### Latency seems too high

**Symptoms:** Latency is higher than expected.

**Causes:**

- No warmup runs — caches not populated
- Debug logging enabled — I/O overhead
- System under load

**Solutions:**

```python
# Add warmup runs
result = await runner.run(component, dataset, warmup_runs=10, benchmark_runs=20)
```

## Logging issues

### No output to stderr

**Symptoms:** No JSON log lines visible.

**Causes:**

- Log level is too restrictive
- Output is captured by the test runner

**Solutions:**

```python
# Set level appropriately
log = AtlasLogger("my-logger", level="DEBUG")

# In pytest, use -s to show stderr
```

### Log metadata not showing

**Symptoms:** Metadata dict is empty in output.

**Causes:** Metadata is passed as part of the message string, not as keyword arguments.

**Solutions:**

```python
# Wrong — metadata embedded in message
log.info(f"Request from {ip} on port {port}")

# Correct — structured metadata
log.info("Request received", remote_ip=ip, port=port)
```

## Performance issues

| Symptom | Likely cause | Fix |
|---|---|---|
| Slow ingestion | Embedding batching too small | Increase `batch_size` in `PipelineConfig` |
| Slow search | Vector store has too many vectors | Use a more efficient vector store backend |
| High memory usage | All chunks loaded at once | Process in batches; use `batch_size` |
| Slow benchmark | No warmup | Set `warmup_runs >= 5` |
| High latency | Reranking enabled | Disable or reduce `top_k` |

## Error lookup table

| Error code | Module | Most likely cause |
|---|---|---|
| `INVALID_CONFIGURATION` | config | Invalid environment or log_level |
| `INVALID_LOG_LEVEL` | log | Unknown level name |
| `HEALTH_CHECK_NOT_FOUND` | health | Unknown check name in `get()` or `unregister()` |
| `DUPLICATE_HEALTH_CHECK` | health | Duplicate `register()` |
| `RESOURCE_NOT_FOUND` | concurrency | Unknown resource name |
| `DUPLICATE_RESOURCE` | concurrency | Duplicate `register()` |
| `INVALID_RETRY_POLICY` | reliability | Policy validation failed |
| `KNOWLEDGE_DUPLICATE_DOCUMENT` | rag | Document ID already registered |
| `KNOWLEDGE_DOCUMENT_NOT_FOUND` | rag | Document ID not found |
| `PIPELINE_NOT_FOUND` | pipeline | Unknown pipeline name in registry |
| `INVALID_PIPELINE_CONFIGURATION` | pipeline | Missing required component in builder |
| `PERSISTENCE_TARGET_EXISTS` | persistence | `save()` on existing path, `overwrite=False` |
| `PERSISTENCE_PATH_NOT_FOUND` | persistence | `load()` or `stats()` on non-existent path |
| `EMBEDDING_PROVIDER_ERROR` | embeddings | Provider failure during embed |
| `UNSUPPORTED_EMBEDDING_PROVIDER` | embeddings | Unknown provider name |
| `RERANKER_NOT_FOUND` | rerank | Unknown reranker name |
| `VECTOR_DIMENSION_MISMATCH` | vectorstore | Wrong vector dimensionality |
| `VECTOR_STORE_FULL` | vectorstore | At capacity (`max_vectors` limit) |
| `VECTOR_NOT_FOUND` | vectorstore | Unknown chunk_id |
| `INVALID_HYBRID_CONFIGURATION` | hybrid | Invalid fusion configuration |
| `EVALUATION_NOT_FOUND` | evaluation | Unknown runner name |
| `INVALID_EVALUATION_CONFIGURATION` | evaluation | Config validation failure |
