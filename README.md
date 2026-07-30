# ATLAS

**AI Operating System** — a modular framework for building intelligent agent systems with RAG, memory, tool orchestration, and workflows.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/tests-2400%2B-brightgreen.svg)](tests/)
[![RAG](https://img.shields.io/badge/RAG-complete-orange.svg)](docs/api/rag.md)
[![Documentation](https://img.shields.io/badge/docs-phase11-blueviolet.svg)](docs/api/overview.md)

---

## Overview

Atlas provides a complete stack for building AI-powered applications:

- **Agent Runtime** — orchestrate intelligent agents with tool use and reasoning
- **Memory System** — persistent conversation and knowledge memory
- **RAG Framework** — document ingestion, chunking, embedding, retrieval, reranking
- **Evaluation** — quality metrics, benchmarking, and performance profiling
- **Core Infrastructure** — configuration, logging, health monitoring, concurrency, retry
- **Persistence** — save/load knowledge base state to JSON

### Architecture

```
Application Layer
    │
    ├── Agent Runtime ─── Memory System
    │
    ├── RAG Pipeline ─── KnowledgeBase ─── Embeddings ─── Vector Store
    │       │                │
    │       └── Reranker ────┘
    │
    ├── Evaluation ─── Metrics / Benchmark / Profiler
    │
    └── Core Infrastructure ─── Config / Logging / Health / Concurrency / Retry
```

## Quick start

```bash
# Install
pip install -e ".[dev]"

# Run the diagnostics
python tools/atlas_cli.py doctor

# Run a basic RAG example
PYTHONPATH=. python examples/basic_rag.py

# Or use a template
cp -r templates/minimal ./my-project
cd my-project
pip install -e .
python app/main.py
```

## Documentation

| Section | Contents |
|---|---|
| [API Reference](docs/api/overview.md) | Package-by-package API documentation |
| [Tutorials](docs/tutorials/getting_started.md) | Step-by-step guides from beginner to advanced |
| [Developer Guide](docs/developer/architecture.md) | Architecture, contributing, testing, style guide |
| [CLI Reference](docs/cli.md) | Developer CLI commands |
| [FAQ](docs/faq.md) | Frequently asked questions |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |
| [Migration Guide](docs/migration.md) | Upgrading between versions |

## Project templates

| Template | Description |
|---|---|
| [minimal/](templates/minimal/) | Smallest runnable Atlas project |
| [rag_app/](templates/rag_app/) | Production RAG application layout |
| [custom_provider/](templates/custom_provider/) | Custom provider extension template |

## Examples

| Example | Description |
|---|---|
| `examples/basic_rag.py` | Basic retrieval flow with KnowledgeBase |
| `examples/benchmark_demo.py` | Retrieval metrics, benchmarking, profiling |
| `examples/custom_provider.py` | Custom embedding and reranker |
| `examples/persistence_demo.py` | Save, load, update JSON snapshots |
| `examples/retry_demo.py` | Exponential backoff retry with policies |

## Development

```bash
git clone https://github.com/your-org/atlas.git
cd atlas
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Run tests
pytest
```

## License

MIT — see [LICENSE](LICENSE).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.
