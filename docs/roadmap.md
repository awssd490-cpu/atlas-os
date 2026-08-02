# ATLAS Roadmap

This roadmap defines the milestones for building ATLAS. Each phase builds on the previous, layering capabilities onto the kernel.

---

## Phase 1: Atlas Kernel ✅ (Current)

**Core operating system infrastructure.**

- [x] Project structure and build system
- [x] Configuration management (Pydantic Settings)
- [x] Structured logging (Loguru)
- [x] Dependency injection container
- [x] Event bus (in-process, async)
- [x] Module registration system
- [x] Lifecycle manager (startup/shutdown)
- [x] Health check endpoints
- [x] Internal API gateway skeleton
- [x] Unit tests for all kernel components
- [ ] ADRs for major architectural decisions

**Delivery:** Core library with no external dependencies beyond Python standard library, FastAPI, Pydantic, and Loguru.

---

## Phase 2: Persistence & Storage Engine

**Data infrastructure for all engines.**

- PostgreSQL connection management (async)
- SQLAlchemy 2.0 async session management
- Repository pattern with generic base
- Migration system (Alembic)
- Redis connection management
- Cache abstraction layer
- Storage engine module
- Integration tests

**Delivery:** All engines can persist and retrieve data through a uniform interface.

---

## Phase 3: Task Engine

**Task decomposition, routing, and execution.**

- Task data model and state machine
- Task repository
- Task router (synchronous and queued)
- Task planner interface
- Execution engine skeleton
- Task lifecycle events
- Task status reporting

**Delivery:** ATLAS can accept, decompose, route, and track tasks.

---

## Phase 4: Memory Engine

**Context, history, and knowledge management.**

- Conversation/session memory
- Document memory
- Entity extraction and storage
- Memory query interface
- Vector store integration (Qdrant)
- Memory compression and summarization
- Retrieval pipeline

**Delivery:** Agents have persistent, queryable memory across sessions.

---

## Phase 5: Tool Engine

**Secure, governed tool execution.**

- Tool registry and discovery
- Tool execution sandbox
- Tool parameter validation
- Tool output handling
- Rate limiting and governance
- Tool chain execution
- Tool authorization

**Delivery:** Agents can safely discover and execute tools through a governed interface.

---

## Phase 6: Model Engine

**Model provider abstraction.**

- Provider interface (OpenAI, Anthropic, Gemini, Ollama)
- Model routing and load balancing
- Prompt template management
- Token tracking and budgeting
- Fallback and retry logic
- Streaming response handling
- Structured output parsing

**Delivery:** Model access is abstracted behind a unified, resilient interface.

---

## Phase 7: Evaluation Engine

**Systematic quality measurement.**

- Evaluation pipeline
- Metric collection and aggregation
- A/B comparison framework
- Automated regression detection
- Human feedback integration
- Evaluation data store
- Dashboard data export

**Delivery:** Every task execution can be evaluated against configurable criteria.

---

## Phase 8: Telemetry Engine

**Full observability suite.**

- OpenTelemetry integration
- Distributed tracing
- Metrics aggregation (Prometheus)
- Structured audit logging
- Performance profiling
- Cost tracking
- Usage analytics
- Grafana dashboards

**Delivery:** Full production observability for all ATLAS operations.

---

## Phase 9: API Gateway & SDK

**External interfaces.**

- REST API for engine operations
- WebSocket for streaming
- Python SDK
- Authentication and authorization
- API key management
- Rate limiting
- API versioning
- Client libraries

**Delivery:** ATLAS is accessible programmatically through multiple interfaces.

---

## Phase 10: Orchestration & Multi-Agent

**Advanced coordination.**

- Multi-agent task decomposition
- Agent communication protocols
- Consensus and voting mechanisms
- Hierarchical agent management
- Agent specialization and discovery
- Workflow engine (DAG-based)
- Human-in-the-loop patterns

**Delivery:** ATLAS orchestrates multiple specialized agents working together on complex tasks.

---

## Phase 11: Production Readiness

**Enterprise capabilities.**

- Horizontal scaling
- Kubernetes operator
- Helm charts
- Monitoring and alerting
- Backup and recovery
- Disaster recovery
- Security audit
- Performance benchmarks
- Documentation site

**Delivery:** ATLAS is production-ready for enterprise deployments.

---

## Phase 12: Ecosystem

**Community and extensibility.**

- Plugin system
- Engine SDK
- Developer documentation
- Contribution guide
- Community templates
- Integration marketplace
- Case studies

**Delivery:** ATLAS has a thriving ecosystem of community-contributed engines and tools.

---

## Phase 12.2: CI/CD & Release Automation

**Continuous integration, linting, packaging, and release tooling.**

- [x] GitHub Actions workflows — tests, lint, build, docs validation
- [x] Dependabot for GitHub Actions and Python dependencies
- [x] CODEOWNERS and repository governance
- [x] Linting & formatting tooling (Ruff, Black, isort, mypy) configured in `pyproject.toml`
- [x] Distribution build and metadata validation (wheel + sdist via `build`, `twine check`)
- [x] GitHub Releases automation on version tags (PyPI publishing deferred)
- [x] Documentation validation (markdownlint, markdown-link-check, example compile checks)
- [x] `.gitignore` hygiene for generated artifacts

**Delivery:** Atlas is continuously tested, linted, and packaged on every push and pull request, and tagged versions produce GitHub Releases.

---

## Current Status

| Phase | Status | Target |
|-------|--------|--------|
| 1: Kernel | **In Progress** | Q3 2026 |
| 2: Persistence | Planned | Q3 2026 |
| 3: Task Engine | Planned | Q4 2026 |
| 4: Memory Engine | Planned | Q4 2026 |
| 5: Tool Engine | Planned | Q1 2027 |
| 6: Model Engine | Planned | Q1 2027 |
| 7: Evaluation Engine | Planned | Q2 2027 |
| 8: Telemetry Engine | Planned | Q2 2027 |
| 9: API & SDK | Planned | Q3 2027 |
| 10: Orchestration | Planned | Q3 2027 |
| 11: Production | Planned | Q4 2027 |
| 12: Ecosystem | Ongoing | 2028+ |
| 12.2: CI/CD & Release Automation | **Complete** | Q3 2026 |
