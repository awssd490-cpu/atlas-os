# Style Guide

## Naming conventions

| Element | Convention | Example |
|---|---|---|
| Packages | `lowercase_with_underscores` | `app.core.config`, `app.rag.vectorstore` |
| Modules | `lowercase_with_underscores` | `knowledge_base.py`, `json_backend.py` |
| Classes | `PascalCase` | `DefaultKnowledgePipeline`, `MemoryVectorStore` |
| Functions/Methods | `lowercase_with_underscores` | `register_provider()`, `precision_at_k()` |
| Variables | `lowercase_with_underscores` | `document_id`, `max_chunks`, `tmp_path` |
| Constants | `UPPERCASE_WITH_UNDERSCORES` | `STRATEGY_FIXED_SIZE`, `CURRENT_VERSION` |
| Private attributes | `_leading_underscore` | `self._config`, `self._kb` |
| Type aliases | `PascalCase` | `Loader`, `HealthCheckFn` |
| Exceptions | `PascalCase` ending with `Error` | `InvalidConfiguration`, `PipelineNotFound` |
| Enums | `PascalCase` | `ResourceState`, `FusionStrategy` |
| Enum values | `UPPERCASE` | `HealthStatus.HEALTHY`, `ResourceState.OPEN` |
| Test classes | `Test<PascalCase>` | `TestKnowledgeBase`, `TestRetryExecutor` |
| Test methods | `test_<lowercase_underscore>` | `test_register_duplicate_raises` |
| Fixtures | `lowercase_with_underscores` | `tmp_path`, `populated_kb` |
| `__all__` entries | Alphabetically sorted | See any `__init__.py` |

## Typing

### Rules

- All function signatures must have type annotations
- All class attributes must have type annotations
- Use `from __future__ import annotations` at the top of every file
- Use `Mapping` for read-only dict-like parameters (not `dict`)
- Use `Sequence` for read-only list-like parameters (not `list`)
- Use `Any` sparingly — prefer `object` or a type variable
- Use `TYPE_CHECKING` guard for imports only needed in type annotations

```python
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.rag.embeddings.base import EmbeddingProvider


class MyClass:
    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self._provider = provider

    def process(self, items: Sequence[str]) -> dict[str, int]:
        return {item: len(item) for item in items}
```

### Union syntax

Prefer `X | Y` syntax over `Union[X, Y]` (Python 3.10+):

```python
# Correct
def load(self, path: str | None = None) -> object: ...

# Avoid (pre-3.10 compatibility not needed)
from typing import Optional, Union
def load(self, path: Optional[str] = None) -> object: ...
```

## Docstrings

### Rules

- Every public function, method, class, and module must have a docstring
- Follow Google-style with `Args:`, `Returns:`, `Raises:` sections
- Use single-line docstrings for trivial properties
- Use triple-quoted multi-line docstrings for everything else

```python
def precision_at_k(
    retrieved_ids: Sequence[Any],
    relevant_ids: set[Any],
    k: int,
) -> float:
    """Compute precision at k (P@k).

    ``precision@k = |retrieved[:k] ∩ relevant| / k``

    Args:
        retrieved_ids: Ordered list of retrieved IDs.
        relevant_ids: Set of ground-truth relevant IDs.
        k: The cutoff rank.

    Returns:
        Precision at k in ``[0.0, 1.0]``.  Returns ``0.0`` when
        *k* ≤ 0 or the retrieved list is empty.
    """
    ...
```

### Class docstrings

```python
class DefaultReranker(Reranker):
    """Deterministic reranker using lightweight text heuristics.

    Computes a ``rerank_score`` for each result based on:
    - **Lexical overlap**: fraction of query terms present
    - **Query term coverage**: proportion of query terms matched

    Usage::

        reranker = DefaultReranker()
        response = await reranker.rerank("query", results)
    """
```

## Dataclasses

### Rules

- All model dataclasses must be `frozen=True`
- Use `field(default_factory=...)` for mutable defaults
- Every field must have a docstring in the class docstring (Attributes: section)
- Use `Mapping` for read-only metadata, `dict` for mutable internals

```python
@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry behaviour.

    Attributes:
        max_attempts: Maximum number of execution attempts (>= 1).
        initial_delay_ms: Delay before first retry (>= 0).
        backoff_multiplier: Exponential factor (>= 1.0).
        retry_exceptions: Exception types that trigger retry.
    """

    max_attempts: int = 3
    initial_delay_ms: float = 100.0
    backoff_multiplier: float = 2.0
    retry_exceptions: tuple[type[Exception], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

## Error hierarchy

### Rules

- Every subsystem defines its own base error inheriting from the parent subsystem error
- Errors use `code` (SCREAMING_SNAKE_CASE) and `details` (dict) patterns
- `to_dict()` serialises errors for API responses

```python
class PipelineError(KnowledgeError):
    """Base class for all pipeline errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "PIPELINE_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class PipelineNotFound(PipelineError):
    """Raised when a requested pipeline is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Pipeline {name!r} not found" if name else "Pipeline not found"
        super().__init__(msg, code="PIPELINE_NOT_FOUND", details=details)
```

## Logging

### Rules

- Use `AtlasLogger`, never `print()`
- Always pass structured metadata as keyword arguments, not f-strings
- Log at the appropriate level (DEBUG for details, INFO for normal ops, WARNING for recoverable issues, ERROR for failures)

```python
from app.core.log import AtlasLogger

log = AtlasLogger("my.component")

# Correct
log.info("Document ingested", document_id=doc_id, chunk_count=len(chunks))

# Incorrect
log.info(f"Document {doc_id} ingested with {len(chunks)} chunks")
```

## Async guidelines

### Rules

- All potentially blocking operations are `async def`
- Use `asyncio` primitives (`Semaphore`, `gather`, `sleep`)
- Use `time.perf_counter()` for timing (not `time.time()`)
- Denote async test functions with `@pytest.mark.asyncio`
- Accept both sync and async callables where ergonomic (e.g. `RetryExecutor`, `PerformanceProfiler`, `HealthMonitor`)

```python
async def execute(self, fn, *args, **kwargs):
    val = fn(*args, **kwargs)
    if isinstance(val, Awaitable):
        val = await val
    return val
```

## Import order

Within each file, group imports in this order:

1. `from __future__ import annotations`
2. Standard library
3. Third-party libraries (if any)
4. Application modules

Each group separated by a blank line:

```python
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from app.rag.models import KnowledgeDocument, KnowledgeChunk
from app.rag.pipeline.errors import PipelineError
```

## File conventions

- Every Python package must have an `__init__.py` that exports via `__all__`
- `__all__` entries are alphabetically sorted
- Include `__future__ import annotations` as the first line of every `.py` file
- Maximum line length: 100 characters (informal — use judgement)
- Use 4-space indentation (no tabs)
