# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-02

### Added

- **Atlas v1.0 official stable release** — now distributed as `tekvora-atlas`
- Release engineering subsystem (`app.release`)
- CLI release commands: `release-version`, `release-next`, `release-changelog`, `release-check`
- CI/CD and release automation (GitHub Actions, Dependabot, CODEOWNERS)
- Project templates: minimal, rag_app, custom_provider

### Changed

- Package distribution renamed from `atlas` to `tekvora-atlas` to resolve a PyPI name conflict
- Version bumped from `0.11.0` to `1.0.0` (official stable release)
- Publisher metadata updated to Tekvora
- Development Status classifier promoted from Alpha to Production/Stable
- Snapshot listing falls back to insertion-order `rowid` so "newest first" is deterministic when timestamps tie
- Release workflow publishes stable tags to PyPI via trusted publishing

### Fixed

- Deterministic snapshot ordering (`app/memory/snapshots.py`)
- Stabilized the performance-profiler timing test with an event-loop warmup

## [0.12.0] — 2026-08-02

### Added

- Release engineering subsystem (`app.release`):
  - `Version` model with semantic version parsing and bump levels
  - Changelog construction and rendering (Keep a Changelog compatible)
  - Distribution artifact discovery and SHA-256 validation
  - `ReleaseService` orchestrating version → changelog → artifacts
- CLI release commands: `release-version`, `release-next`, `release-changelog`, `release-check`
- Release engineering documentation (`docs/release-engineering.md`)

### Changed

- Release workflow publishes stable tags to PyPI via trusted publishing
- Snapshot listing now falls back to insertion-order `rowid` so "newest first" is deterministic when timestamps tie

### Fixed

- Deterministic snapshot ordering (`app/memory/snapshots.py`)
- Stabilized the performance-profiler timing test with an event-loop warmup

## [0.11.0] — 2026-07-30

### Added

- API documentation: overview, core, providers, RAG, persistence, evaluation
- Tutorials: getting started, RAG pipeline, persistence, evaluation, advanced
- Developer guide: architecture, contributing, style guide, testing, project layout
- CLI developer tool: info, doctor, version, list-packages, list-providers, list-rerankers, validate-config
- Project templates: minimal, rag_app, custom_provider
- Migration guide, troubleshooting guide, FAQ, compatibility reference
- Packaging: pyproject.toml, MANIFEST.in, LICENSE, CHANGELOG, CONTRIBUTING, SECURITY

## [0.10.0] — 2026-07-30

### Added

- Core infrastructure packages:
  - `app.core.config` — Configuration management (AtlasConfig, ConfigLoader)
  - `app.core.log` — Structured logging (AtlasLogger, JsonFormatter)
  - `app.core.health` — Health monitoring (HealthMonitor, HealthStatus)
  - `app.core.concurrency` — Concurrency limiter, resource manager
  - `app.core.reliability` — Retry executor with exponential backoff

## [0.9.0] — 2026-07-30

### Added

- Evaluation framework:
  - `RetrievalMetrics` — precision@k, recall@k, F1@k, MRR, AP, nDCG
  - `BenchmarkRunner` — warmup/measurement phases, latency/throughput
  - `PerformanceProfiler` — execution time + memory profiling
  - `EvaluationDataset` / `DatasetLoader` — ground-truth dataset support
  - `EvaluationRunner` — ABC for evaluation runners

## [0.8.0] — 2026-07-30

### Added

- Persistence layer:
  - `PersistenceBackend` ABC
  - `JsonPersistenceBackend` — save, load, update, delete, stats
  - Atomic writes via temp-file + rename
  - Incremental snapshot updates with change detection
  - Embedding and vector store serialization

## [0.7.0] — 2026-07-30

### Added

- Pipeline orchestration:
  - `KnowledgePipeline` ABC with ingest/search/clear/stats
  - `DefaultKnowledgePipeline` — concrete implementation
  - `PipelineBuilder` — fluent builder with validation
  - Automatic embedding and indexing during ingestion
  - Keyword, hybrid, and reranked search support

## [0.6.0] — 2026-07-30

### Added

- Reranking subsystem:
  - `Reranker` ABC
  - `DefaultReranker` — heuristic-based (lexical overlap, length penalty, phrase bonus)
  - Reranker registry

## [0.5.0] — 2026-07-30

### Added

- Hybrid retrieval:
  - `HybridRetriever` ABC
  - `DefaultHybridRetriever` — keyword + semantic fusion
  - Fusion strategies: weighted sum, reciprocal rank fusion
  - Retrieval scoring models

## [0.4.0] — 2026-07-30

### Added

- Vector store:
  - `VectorStore` ABC
  - `MemoryVectorStore` — in-memory dict-backed
  - Cosine, dot product, and negative Euclidean similarity
  - VectorStore registry

## [0.3.0] — 2026-07-30

### Added

- Embedding providers:
  - `EmbeddingProvider` ABC
  - `DeterministicEmbeddingProvider` — SHA-256-based (stdlib only)
  - `MockEmbeddingProvider` — testing
  - Embedding provider registry

## [0.2.0] — 2026-07-30

### Added

- Document chunking:
  - `ChunkingEngine` with strategy pattern
  - Strategies: fixed_size, sentence, paragraph, recursive, sliding_window, whole_document
  - Deterministic chunk ID generation

## [0.1.0] — 2026-07-30

### Added

- Initial release
- Core framework: Kernel, DI, Event Bus, Lifecycle Manager
- Module and capability registries
- Telemetry, health API
- Memory subsystem foundations
- Agent runtime foundations
- Provider layer foundations
