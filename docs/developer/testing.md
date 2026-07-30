# Testing Guide

## Running tests

### Run the full suite

```bash
# From the repository root
python -m pytest tests/
```

### Run a subsystem

```bash
# All RAG tests
python -m pytest tests/unit/rag/

# All core infrastructure tests
python -m pytest tests/unit/core/

# A specific package
python -m pytest tests/unit/rag/pipeline/
python -m pytest tests/unit/rag/persistence/
python -m pytest tests/unit/rag/evaluation/
```

### Run a single test

```bash
python -m pytest tests/unit/rag/pipeline/test_pipeline.py -v
python -m pytest tests/unit/rag/pipeline/ -k test_build_returns_pipeline
```

### Useful flags

| Flag | Purpose |
|---|---|
| `-v` | Verbose output (test names) |
| `-q` | Quiet (dots only) |
| `-x` | Stop on first failure |
| `-s` | Show stdout/stderr (for debugging) |
| `--tb=long` | Full traceback |
| `--tb=short` | Concise traceback |
| `-k "pattern"` | Run tests matching pattern |
| `--ignore=path` | Skip a directory |

## Writing tests

### Template for a new test module

```python
"""Tests for the <module> module."""

from __future__ import annotations

from typing import Any

import pytest

# Import via the public API (__init__.py), not internal modules
from app.rag.mymodule import MyClass, MyConfig, MyError
from app.rag.mymodule import internal_implementation as Internal_Impl


# ======================================================================
# Imports
# ======================================================================

class TestImports:
    """Verify public API symbols resolve correctly."""

    def test_my_class_imported(self) -> None:
        assert MyClass is MyClass_Impl
```

### Test structure

Atlas tests follow a consistent structure:

1. **Import verification** — verify public API symbols via identity checks
2. **Config/model tests** — verify frozen dataclasses (defaults, customs, immutability)
3. **ABC tests** — verify abstract methods and construction
4. **Registry tests** — verify register/get/duplicate/unregister/list/clear
5. **Error tests** — verify error messages, codes, `to_dict()`, hierarchy
6. **Behavior tests** — verify actual business logic
7. **Edge case tests** — verify empty inputs, unicode, exceptions, determinism

### Naming conventions

| Element | Convention | Example |
|---|---|---|
| Test class | `Test<Component>` | `TestKnowledgeBase` |
| Test method | `test_<scenario>` | `test_register_duplicate_raises` |
| Fixture | Descriptive noun | `def populated_kb()`, `def tmp_path()` |
| Helper class | Leading underscore | `class _AsyncDelayed` |

### Fixture guidelines

- Use `@pytest.fixture` for shared setup
- Use `@pytest.fixture(autouse=True)` for cleanup that must run before/after every test
- Use `tmp_path` (pytest's built-in) or custom temp file fixtures
- Name fixtures clearly: `def populated_kb()`, `def empty_kb()`, `def pipeline()`
- Place shared fixtures in `conftest.py` or define them in the test class
- If a fixture is only used by one test class, define it inside that class

### Async testing

Atlas uses `pytest-asyncio` with auto mode:

```python
@pytest.mark.asyncio
async def test_async_operation() -> None:
    result = await some_async_function()
    assert result.success is True
```

No special setup needed — `pytest-asyncio` is configured in `pyproject.toml` with `mode = "auto"`.

### Pattern: test helpers for async callables

When testing benchmark or retry code that needs controlled async callables:

```python
class _AsyncDelayed:
    """Async callable that simulates a fixed delay."""

    def __init__(self, delay_s: float = 0.01) -> None:
        self._delay = delay_s

    async def __call__(self, query: str) -> str:
        await asyncio.sleep(self._delay)
        return query
```

### Pattern: test helpers for flaky callables

```python
class _AsyncFailing:
    """Async callable that raises for a specific query."""

    def __init__(self, fail_on: str = "fail") -> None:
        self._fail_on = fail_on

    async def __call__(self, query: str) -> str:
        if query == self._fail_on:
            raise ValueError(f"Failed on: {query}")
        return query
```

### Pattern: environment variable cleanup

```python
class TestConfigLoaderFromEnv:
    @pytest.fixture(autouse=True)
    def cleanup_env(self) -> None:
        for key in ("ENVIRONMENT", "DEBUG", "LOG_LEVEL", "RANDOM_SEED"):
            os.environ.pop(f"ATLAS_{key}", None)
        yield
        for key in ("ENVIRONMENT", "DEBUG", "LOG_LEVEL", "RANDOM_SEED"):
            os.environ.pop(f"ATLAS_{key}", None)
```

## Snapshot testing

Persistence tests use temporary files for snapshot round-trips:

```python
@pytest.fixture
def tmp_path() -> str:
    """Return a temporary file path for JSON output."""
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)
```

## Coverage expectations

| Area | Target | Notes |
|---|---|---|
| Core infrastructure | ≥95% | Config, logging, health, concurrency, retry |
| RAG models | 100% | Frozen dataclass tests (defaults, customs, immutability) |
| Pipeline | ≥90% | Architecture + behavior tests |
| Persistence | ≥90% | Save, load, update, round-trip, error cases |
| Evaluation | ≥90% | Metrics, benchmark, profiler, datasets, edge cases |
| Providers | ≥85% | Default implementations + ABC tests |

To check coverage:

```bash
pip install coverage
coverage run -m pytest tests/unit/rag/pipeline/
coverage report -m
```

## Common testing mistakes

| Mistake | Why it's wrong | Correct approach |
|---|---|---|
| Importing internal modules directly | Bypasses public API | Import through `__init__.py` |
| Using `time.time()` in tests | Not monotonic, can produce negative durations | Use `time.perf_counter()` |
| Forgetting `@pytest.mark.asyncio` | Test won't run in event loop | Add the decorator |
| Testing frozen dataclass mutation | AttributeError is expected | Use `pytest.raises(AttributeError)` |
| Not cleaning up global registry | Tests leak state | Use `autouse` fixtures with `clear_*()` |
| Hard-coding file paths | Platform-incompatible | Use `tmp_path` fixture |
| Asserting `==` on floats | Floating-point comparison | Use `pytest.approx()` |
| Using `assert False` for expected failures | Confusing when it passes | Use explicit assertions |
