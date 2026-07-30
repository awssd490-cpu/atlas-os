#!/usr/bin/env python3
"""
Benchmark and evaluation demo.

Demonstrates:
  - RetrievalMetrics (precision, recall, F1, MRR, AP, nDCG)
  - BenchmarkRunner (latency and throughput)
  - PerformanceProfiler (execution time and memory)
"""

import asyncio

from app.rag.evaluation import (
    BenchmarkRunner,
    DatasetLoader,
    PerformanceProfiler,
    RetrievalMetrics,
)


async def main() -> None:
    print("=" * 60)
    print("Benchmark & Evaluation Demo")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Part 1: Retrieval metrics
    # ------------------------------------------------------------------
    print("\n--- Retrieval Metrics ---")
    metrics = RetrievalMetrics()

    # Simulate a retrieval result
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c", "e"}

    p5 = metrics.precision_at_k(retrieved, relevant, k=5)
    r5 = metrics.recall_at_k(retrieved, relevant, k=5)
    f1 = metrics.f1_at_k(retrieved, relevant, k=5)
    mrr = metrics.mean_reciprocal_rank(retrieved, relevant)
    ap = metrics.average_precision(retrieved, relevant)
    ndcg = metrics.normalized_dcg(retrieved, relevant, k=5)

    print(f"  Results: retrieved={retrieved}, relevant={relevant}")
    print(f"  P@5:     {p5:.3f}")
    print(f"  R@5:     {r5:.3f}")
    print(f"  F1@5:    {f1:.3f}")
    print(f"  MRR:     {mrr:.3f}")
    print(f"  AP:      {ap:.3f}")
    print(f"  nDCG@5:  {ndcg:.3f}")

    # ------------------------------------------------------------------
    # Part 2: Dataset round-trip
    # ------------------------------------------------------------------
    print("\n--- Dataset ---")
    ds = DatasetLoader.from_dict({
        "name": "demo-dataset",
        "samples": [
            {"query": "capital of France", "relevant_ids": ["a"]},
            {"query": "capital of Japan", "relevant_ids": ["c"]},
        ],
    })
    print(f"  Dataset: {ds.name}, {ds.size} sample(s)")
    print(f"  Queries: {ds.queries()}")

    # Round-trip
    data = DatasetLoader.to_dict(ds)
    ds2 = DatasetLoader.from_dict(data)
    assert ds2.size == ds.size
    print(f"  Round-trip OK")

    # ------------------------------------------------------------------
    # Part 3: Benchmark runner
    # ------------------------------------------------------------------
    print("\n--- Benchmark ---")

    async def fast_search(query: str) -> str:
        await asyncio.sleep(0.005)  # simulate 5ms latency
        return f"results for {query}"

    runner = BenchmarkRunner()
    benchmark_result = await runner.run(
        component=fast_search,
        dataset=ds.queries(),
        warmup_runs=2,
        benchmark_runs=5,
    )
    print(f"  Queries: {benchmark_result.total_queries}")
    print(f"  Avg latency: {benchmark_result.average_latency_ms:.1f} ms")
    print(f"  Min latency: {benchmark_result.min_latency_ms:.1f} ms")
    print(f"  Max latency: {benchmark_result.max_latency_ms:.1f} ms")
    print(f"  Throughput: {benchmark_result.throughput_qps:.0f} qps")

    # ------------------------------------------------------------------
    # Part 4: Performance profiler
    # ------------------------------------------------------------------
    print("\n--- Profiler ---")

    def sync_work(n: int = 1000) -> int:
        """Synchronous work to profile."""
        total = 0
        for i in range(n):
            total += i * i
        return total

    profiler = PerformanceProfiler()
    profile = await profiler.profile(sync_work, 2000)
    print(f"  Execution time: {profile.execution_time_ms:.2f} ms")
    print(f"  Peak memory: {profile.peak_memory_bytes / 1024:.1f} KB")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
