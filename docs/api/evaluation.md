# Evaluation Layer (`app.rag.evaluation`)

## Overview

Provides a pluggable framework for measuring the quality and performance of RAG components. Includes standard IR metrics, a benchmark runner with warmup/measurement separation, dataset support, and a lightweight profiler.

## Architecture

```
EvaluationRunner (ABC)
    │
    ├── evaluate(component, dataset) → EvaluationResult
    ├── benchmark(component, dataset) → BenchmarkResult
    └── profile(component, dataset) → BenchmarkResult
            │
            └── (concrete runners not yet implemented)

RetrievalMetrics (static)
    precision_at_k / recall_at_k / f1_at_k
    mean_reciprocal_rank / average_precision / normalized_dcg

BenchmarkRunner (standalone)
    run(component, dataset) → BenchmarkResult
    # warmup + measurement phases

PerformanceProfiler (standalone)
    profile(component, *args, **kwargs) → PerformanceProfile
    # time.perf_counter + tracemalloc

DatasetLoader (static)
    from_dict / from_json / to_dict / to_json

EvaluationDataset (frozen)
    name, samples, metadata
```

## Public API

| Symbol | Kind | Description |
|---|---|---|
| `EvaluationRunner` | ABC | Abstract base (evaluate/benchmark/profile) |
| `EvaluationConfig` | Frozen | enabled, metrics, warmup_runs, benchmark_runs, random_seed |
| `EvaluationResult` | Frozen | score, metrics dict, metadata, duration |
| `BenchmarkResult` | Frozen | latency variants (3), throughput (2 aliases), total, memory, metadata |
| `BenchmarkRunner` | Class | Generic benchmark engine |
| `RetrievalMetrics` | Static | 6 standard IR metrics |
| `EvaluationSample` | Frozen | query + relevant_ids (frozenset) + metadata |
| `EvaluationDataset` | Frozen | named collection of samples |
| `DatasetLoader` | Static | load/save datasets from dict/JSON |
| `PerformanceProfile` | Frozen | execution_time_ms, peak/current memory, metadata |
| `PerformanceProfiler` | Class | Profile sync/async callables |
| `EvaluationError` | Exception | Base evaluation error |
| `InvalidEvaluationConfiguration` | Exception | Config validation |
| `EvaluationNotFound` | Exception | Unknown runner name |
| `register()` / `get()` / `list_runners()` / `clear_runners()` | Functions | Runner type registry |

---

## RetrievalMetrics

### Purpose

Six standard information retrieval metrics, all returning `float` in `[0.0, 1.0]`.

### Methods

```python
class RetrievalMetrics:
    @staticmethod
    def precision_at_k(retrieved_ids, relevant_ids, k) -> float: ...
    @staticmethod
    def recall_at_k(retrieved_ids, relevant_ids, k) -> float: ...
    @staticmethod
    def f1_at_k(retrieved_ids, relevant_ids, k) -> float: ...
    @staticmethod
    def mean_reciprocal_rank(retrieved_ids, relevant_ids) -> float: ...
    @staticmethod
    def average_precision(retrieved_ids, relevant_ids) -> float: ...
    @staticmethod
    def normalized_dcg(retrieved_ids, relevant_ids, k) -> float: ...
```

### Formulas

| Metric | Formula | Returns |
|---|---|---|
| P@k | `|R_k ∩ rel| / k` | 0 when k≤0 or empty retrieved |
| R@k | `|R_k ∩ rel| / |rel|` | 1 when relevant set empty; 0 when k≤0 |
| F1@k | `2 × P@k × R@k / (P@k + R@k)` | 0 when both are 0 |
| MRR | `1 / rank_of_first_relevant` | 0 when none found |
| AP | `Σ P@i for each relevant / |rel|` | 1 when relevant set empty |
| nDCG@k | `DCG@k / IDCG@k` | 1 when relevant set empty; 0 when k≤0 |

### Edge cases

| Condition | P@k | R@k | F1@k | MRR | AP | nDCG |
|---|---|---|---|---|---|---|
| k ≤ 0 | 0.0 | 0.0 | 0.0 | — | — | 0.0 |
| Empty retrieved | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Empty relevant | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 1.0 |

### Example

```python
metrics = RetrievalMetrics()
retrieved = ["a", "b", "c", "d", "e"]
relevant = {"a", "c", "e"}

p5 = metrics.precision_at_k(retrieved, relevant, k=5)
r5 = metrics.recall_at_k(retrieved, relevant, k=5)
ap = metrics.average_precision(retrieved, relevant)
```

---

## BenchmarkRunner

### Purpose

Measures latency and throughput of an async callable over a dataset with separate warmup and measurement phases.

### Methods

```python
class BenchmarkRunner:
    def __init__(self, config: EvaluationConfig | None = None): ...
    async def run(self, component, dataset, *,
                  warmup_runs=None, benchmark_runs=None) -> BenchmarkResult: ...
    @property
    def config(self) -> EvaluationConfig: ...
```

### Timing

- Uses `time.perf_counter()` for all measurements.
- Warmup iterations execute but their timing is discarded.
- Per-operation latency is recorded for each query × benchmark run.
- Throughput is `total_queries / total_wall_time`.

### BenchmarkResult fields

| Field | Description |
|---|---|
| `average_latency_ms` | Mean latency across all operations |
| `min_latency_ms` | Minimum observed latency |
| `max_latency_ms` | Maximum observed latency |
| `throughput_qps` | Queries per second |
| `total_queries` | `len(dataset) × benchmark_runs` |
| `total_duration` | Wall-clock time in ms |
| `latency_ms` | Alias for `average_latency_ms` |
| `throughput` | Alias for `throughput_qps` |

### Example

```python
runner = BenchmarkRunner()
result = await runner.run(
    component=pipeline.search,
    dataset=["capital of France", "capital of Japan"],
    warmup_runs=5,
    benchmark_runs=20,
)
print(f"Avg latency: {result.average_latency_ms:.1f}ms")
print(f"Throughput: {result.throughput_qps:.0f} qps")
```

---

## PerformanceProfiler

### Purpose

Profiles a sync or async callable, measuring execution time (`time.perf_counter()`) and memory usage (`tracemalloc`). No external dependencies.

### Methods

```python
class PerformanceProfiler:
    async def profile(self, component, *args, **kwargs) -> PerformanceProfile: ...
    @staticmethod
    def start_tracing() -> None: ...
    @staticmethod
    def stop_tracing() -> None: ...
    @staticmethod
    def get_traced_memory() -> tuple[int, int]: ...
```

### Measurement

- Execution time: `time.perf_counter()` before/after call
- Memory: `tracemalloc.get_traced_memory()` delta (peak and current)
- `tracemalloc.start()` called automatically on first `profile()` if not already active
- Memory values clamped to `max(0, delta)`

### Example

```python
profiler = PerformanceProfiler()

# Profile a sync function
profile = await profiler.profile(my_slow_function, arg1, arg2)
print(f"Took {profile.execution_time_ms:.1f}ms, peak memory {profile.peak_memory_bytes} bytes")

# Profile an async function
profile = await profiler.profile(async_fetch, url="https://example.com")
```

---

## Dataset support

### EvaluationSample

```python
@dataclass(frozen=True)
class EvaluationSample:
    query: str = ""
    relevant_ids: frozenset[str] = field(default_factory=frozenset)
    metadata: Mapping[str, Any] = field(default_factory=dict)
```

### EvaluationDataset

```python
@dataclass(frozen=True)
class EvaluationDataset:
    name: str = ""
    samples: tuple[EvaluationSample, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> int: ...
    @property
    def is_empty(self) -> bool: ...
    def queries(self) -> list[str]: ...
    def relevant_sets(self) -> list[frozenset[str]]: ...
    def sample(self, index: int) -> EvaluationSample: ...
```

### DatasetLoader

```python
class DatasetLoader:
    @staticmethod
    def from_dict(data: dict) -> EvaluationDataset: ...
    @staticmethod
    def from_json(path: str) -> EvaluationDataset: ...
    @staticmethod
    def to_dict(dataset: EvaluationDataset) -> dict: ...
    @staticmethod
    def to_json(dataset: EvaluationDataset, path: str) -> None: ...
```

### JSON format

```json
{
  "name": "retrieval-dataset",
  "metadata": { "version": "1" },
  "samples": [
    {
      "query": "capital of France",
      "relevant_ids": ["doc1", "doc2"],
      "metadata": { "difficulty": "easy" }
    }
  ]
}
```

### Example

```python
loader = DatasetLoader()
ds = loader.from_json("dataset.json")

for query, relevant in zip(ds.queries(), ds.relevant_sets()):
    retrieved = await pipeline.search(query)
    p10 = RetrievalMetrics.precision_at_k(retrieved_ids, relevant, k=10)
    print(f"P@10 for '{query}': {p10:.3f}")
```

---

## Best practices

- **Always warm up** before benchmarking to stabilise caches and JIT compilation.
- **Use `min_score` and `max_chunks`** in search to control evaluation cost.
- **Use frozenset for `relevant_ids`** — it makes set operations explicit and prevents accidental mutation.
- **Run metrics on multiple queries** and average the results — single-query metrics have high variance.
- **Clear `tracemalloc` traces between profiling runs** with `PerformanceProfiler.stop_tracing()` / `start_tracing()` for clean measurements.
