# Evaluating Retrieval Quality

## Goal

Measure retrieval quality using standard IR metrics, benchmark pipeline latency, and profile execution.

## Prerequisites

- A working pipeline from [Building a RAG Pipeline](rag_pipeline.md)
- Understanding of `RetrievalMetrics`, `BenchmarkRunner`, and `PerformanceProfiler`

## Step-by-step guide

### 1. Prepare a dataset

A dataset is a collection of queries with their expected relevant document IDs:

```python
from app.rag.evaluation import DatasetLoader

dataset = DatasetLoader.from_dict({
    "name": "capital-cities",
    "samples": [
        {"query": "capital of France", "relevant_ids": ["paris"]},
        {"query": "capital of Japan", "relevant_ids": ["tokyo"]},
        {"query": "capital of the UK", "relevant_ids": ["london"]},
    ],
})
```

Datasets can also be loaded from JSON files:

```python
dataset = DatasetLoader.from_json("evaluation_dataset.json")
```

### 2. Run queries and collect results

```python
import asyncio

async def evaluate(dataset, pipeline):
    results = []
    for sample in dataset.samples:
        result = await pipeline.search(sample.query)
        # Extract retrieved chunk IDs from the context
        retrieved_ids = extract_ids(result.context)
        results.append((retrieved_ids, sample.relevant_ids))
    return results

# Simple heuristic: extract document IDs from context text
def extract_ids(context: str) -> list[str]:
    ids = []
    for line in context.split("\n"):
        if line.startswith("- "):
            # Assume the first word after "- " is a doc identifier hint
            pass
    return ids
```

### 3. Compute precision and recall

```python
from app.rag.evaluation import RetrievalMetrics

metrics = RetrievalMetrics()

for retrieved, relevant in results:
    p5 = metrics.precision_at_k(retrieved, set(relevant), k=5)
    r5 = metrics.recall_at_k(retrieved, set(relevant), k=5)
    f1 = metrics.f1_at_k(retrieved, set(relevant), k=5)
    mrr = metrics.mean_reciprocal_rank(retrieved, set(relevant))
    ap = metrics.average_precision(retrieved, set(relevant))
    print(f"P@5={p5:.3f} R@5={r5:.3f} F1@5={f1:.3f} MRR={mrr:.3f} AP={ap:.3f}")
```

### 4. Benchmark pipeline performance

```python
from app.rag.evaluation import BenchmarkRunner

async def search_fn(query: str):
    result = await pipeline.search(query)
    return result.context

runner = BenchmarkRunner()
result = await runner.run(
    component=search_fn,
    dataset=dataset.queries(),
    warmup_runs=3,
    benchmark_runs=10,
)

print(f"Average latency: {result.average_latency_ms:.1f} ms")
print(f"Min latency: {result.min_latency_ms:.1f} ms")
print(f"Max latency: {result.max_latency_ms:.1f} ms")
print(f"Throughput: {result.throughput_qps:.0f} qps")
```

### 5. Profile query execution

```python
from app.rag.evaluation import PerformanceProfiler

profiler = PerformanceProfiler()

async def single_query():
    return await pipeline.search("capital of France")

profile = await profiler.profile(single_query)
print(f"Execution time: {profile.execution_time_ms:.2f} ms")
print(f"Peak memory: {profile.peak_memory_bytes / 1024:.1f} KB")
```

## Metrics reference

| Metric | Range | When to use |
|---|---|---|
| `precision_at_k` | [0, 1] | How many retrieved results are relevant |
| `recall_at_k` | [0, 1] | How many relevant documents were found |
| `f1_at_k` | [0, 1] | Harmonic mean of P@k and R@k |
| `mean_reciprocal_rank` | [0, 1] | Rank of the first relevant result |
| `average_precision` | [0, 1] | Overall ranking quality |
| `normalized_dcg` | [0, 1] | Ranking quality with position discount |

## Complete example

See `examples/benchmark_demo.py` for the full runnable example.

## Expected output

```
P@5=1.000 R@5=1.000 F1@5=1.000
Average latency: 12.3 ms
Throughput: 81.3 qps
Execution time: 10.2 ms
Peak memory: 24.0 KB
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `precision_at_k` is always 0 | `relevant_ids` don't match retrieval IDs | Verify document IDs in dataset match KB |
| Benchmark is very slow | No warmup runs | Set `warmup_runs >= 3` |
| Profiler memory is 0 | `tracemalloc` not started | Profiler starts it automatically |

## Next steps

- [Advanced patterns](advanced.md)
- [Custom retry policies](advanced.md#retry-policies)
