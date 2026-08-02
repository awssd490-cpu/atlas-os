# Contributing to Atlas

## Development setup

### Prerequisites

- Python 3.12+
- Git
- `pip`

### One-time setup

```bash
# Clone the repository (currently private; use the Tekvora-provided access)
git clone https://github.com/awssd490-cpu/atlas-os.git
cd atlas-os

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .

# Verify installation
python -c "from app.core.errors import AtlasError; print('OK:', AtlasError)"
```

### Verify the test suite runs

```bash
python -m pytest tests/ -q
```

All ~2400+ tests should pass before you begin working.

## Branching strategy

- `main` — stable, release-ready
- `phase-<N>-<name>` — active phase branches (e.g. `phase-7-agent-runtime`)
- Feature branches branch from `main` or the relevant phase branch

**Naming conventions:**

| Branch type | Pattern | Example |
|---|---|---|
| Phase branch | `phase-<N>-<name>` | `phase-7-agent-runtime` |
| Feature | `feat/<short-name>` | `feat/custom-reranker` |
| Bug fix | `fix/<issue-or-desc>` | `fix/duplicate-chunk-bug` |
| Documentation | `docs/<topic>` | `docs/update-readme` |
| Refactoring | `refactor/<module>` | `refactor/registry-pattern` |

## Commit message conventions

Atlas follows conventional commits:

```
type(scope): brief description

Body (optional, wrap at 72 characters).

Co-Authored-By: Claude <noreply@anthropic.com>
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

**Examples:**

```
feat(rag): implement DefaultReranker with lexical overlap scoring
```

```
fix(pipeline): skip duplicate documents during ingest_documents
```

```
test(persistence): add round-trip tests for save -> load -> save
```

For phase checkpoint commits, include the checkpoint number:

```
feat(rag): finalize hybrid retrieval  [Checkpoint 5]
```

## Pull request checklist

Before submitting a PR:

- [ ] All existing tests pass (`python -m pytest tests/`)
- [ ] New tests cover the change (aim for >=90% line coverage on new code)
- [ ] `py_compile.compile()` passes on all new `.py` files
- [ ] All new modules have `__init__.py` with `__all__`
- [ ] All new errors follow the existing hierarchy
- [ ] All new functions have docstrings (Args, Returns, Raises)
- [ ] All new dataclasses are `frozen=True`
- [ ] All new JSON serialization uses `sort_keys=True` and `ensure_ascii=False`
- [ ] No `print()` calls in production code (use `AtlasLogger` instead)
- [ ] No `time.time()` for measurements (use `time.perf_counter()`)
- [ ] No new external dependencies added to stdlib-only modules

### PR description template

```markdown
## Summary

Brief description of what this PR does.

## Changes

- `path/to/file.py`: what changed and why
- `path/to/tests/`: what was added

## Testing

Ran `python -m pytest tests/` -- all 2400+ tests pass.

## Documentation

- [ ] API docs updated in `docs/api/`
- [ ] Tutorials updated in `docs/tutorials/`
- [ ] Developer docs updated in `docs/developer/`
```

## Review expectations

- **For maintainers:** Review within 2 business days
- **For contributors:** Respond to feedback within 5 business days
- Each PR requires at least one approval from a maintainer
- CI must be green before merging
- Squash-merge is preferred to keep history clean

## FAQ

### How do I add a new embedding provider?

1. Create `app/rag/embeddings/providers/my_provider.py`
2. Subclass `EmbeddingProvider` and implement `name`, `embed()`, `embed_batch()`
3. Register in `app/rag/embeddings/providers/__init__.py`
4. Add tests in `tests/unit/rag/embeddings/`
5. Document in `docs/api/providers.md`

### How do I add a new pipeline?

1. Subclass `KnowledgePipeline` from `app/rag/pipeline/base.py`
2. Implement `ingest()`, `search()`, `clear()`, `stats()`
3. Register with `pipeline_registry.register("my_pipeline", MyPipeline)`
4. Add tests in `tests/unit/rag/pipeline/`

### How do I add a new persistence backend?

1. Subclass `PersistenceBackend` from `app/rag/persistence/base.py`
2. Implement `save()`, `load()`, `exists()`, `delete()`, `stats()`
3. Register with `persistence_registry.register("my_backend", MyBackend)`
4. Add tests in `tests/unit/rag/persistence/`

### The test suite is slow. Can I run a subset?

```bash
# Single test file
python -m pytest tests/unit/rag/pipeline/

# Single test class
python -m pytest tests/unit/rag/pipeline/ -k TestPipelineBuilder

# Single test
python -m pytest tests/unit/rag/pipeline/ -k test_build_returns_pipeline

# Exclude a module
python -m pytest tests/ --ignore=tests/unit/memory/
```

### How do I debug a failing test?

```bash
# Verbose output
python -m pytest tests/path/to/test.py -v

# Show print() output
python -m pytest tests/path/to/test.py -s

# Stop on first failure
python -m pytest tests/path/to/test.py -x

# Run with full traceback
python -m pytest tests/path/to/test.py --tb=long
```

### Why are all dataclasses frozen?

Immutability prevents accidental mutation of shared state. A `KnowledgeDocument` passed to a pipeline should never be modified by the pipeline. If you need a modified copy, use `dataclasses.replace()`:

```python
from dataclasses import replace
new_doc = replace(doc, title="Updated Title")
```

### Why is there no `__init__` in some test packages?

Test packages need `__init__.py` to be importable by pytest, but the file can be empty. Some test packages have empty `__init__.py` — that is intentional and correct.
