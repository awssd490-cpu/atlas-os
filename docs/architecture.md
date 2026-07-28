# ATLAS Architecture

## Overview

ATLAS (Autonomous Task-driven Learning Agent System) is an AI Operating System designed to build, orchestrate, monitor, evaluate, and improve production AI systems at scale.

This document describes the system architecture, design principles, and key structural decisions.

---

## Architectural Philosophy

ATLAS follows a **microkernel architecture** with event-driven communication between isolated engines.

```
┌─────────────────────────────────────────────────────┐
│                    API Gateway                       │
│              (FastAPI — thin transport layer)        │
└──────────┬──────────────────────────────────────┬────┘
           │                                      │
┌──────────▼──────────────────────────────────────▼────┐
│                     ATLAS KERNEL                      │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │   Config  │  │    DI    │  │  Module Registry  │    │
│  │  Manager  │  │ Container│  │                   │    │
│  └──────────┘  └──────────┘  └──────────────────┘    │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │ Event Bus│  │ Lifecycle│  │     Logging       │    │
│  │          │  │  Manager │  │    Subsystem      │    │
│  └──────────┘  └──────────┘  └──────────────────┘    │
└──────────┬──────────────────────────────────────┬────┘
           │                                      │
┌──────────▼──────────────────────────────────────▼────┐
│                     Engines                            │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │  Planner  │  │Execution │  │     Memory       │    │
│  │  Engine   │  │  Engine  │  │     Engine       │    │
│  └──────────┘  └──────────┘  └──────────────────┘    │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │
│  │   Tool    │  │  Eval    │  │   Telemetry      │    │
│  │  Engine   │  │  Engine  │  │    Engine        │    │
│  └──────────┘  └──────────┘  └──────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### Layers

1. **Transport Layer** (API Gateway)
   - FastAPI endpoints only
   - Authentication, authorization, rate limiting
   - Request validation (Pydantic)
   - Delegates to kernel services, never contains business logic

2. **Kernel Layer**
   - Application lifecycle management
   - Configuration loading and validation
   - Dependency injection container
   - Module registration and orchestration
   - Internal event bus
   - Structured logging
   - Health monitoring

3. **Engine Layer**
   - Pluggable engines implementing domain logic
   - Each engine is a self-contained module registered with the kernel
   - Communication via the event bus (loose coupling)
   - Each engine has its own lifecycle (register, boot, shutdown)

4. **Infrastructure Layer**
   - PostgreSQL (persistence)
   - Redis (caching, pub-sub, rate limiting)
   - Vector stores (Qdrant)
   - Graph stores (Neo4j)
   - Message queues (future: NATS, RabbitMQ)

---

## Core Design Principles

### 1. Clean Architecture

Dependencies point inward. Domain logic is independent of frameworks, databases, and external services.

```
Frameworks → Interfaces → Application → Domain
```

- **Domain**: Core business entities and rules (pure Python)
- **Application**: Use cases that orchestrate domain logic
- **Interfaces**: Repository interfaces, service abstractions
- **Frameworks**: FastAPI, SQLAlchemy, Redis clients, etc.

### 2. Event-Driven Communication

Engines communicate through the Event Bus, not direct method calls. This ensures:
- Loose coupling between engines
- Full observability (every event is logged)
- Extensibility (new engines react to existing events)
- Resilience (event handlers are isolated)

### 3. Dependency Injection

All dependencies are resolved through a centralized DI container:
- Reduces coupling
- Simplifies testing (mock at container level)
- Allows runtime replacement of implementations
- Explicit dependency graphs

### 4. Everything is Observable

Every important action produces:
- A structured log entry
- An event on the event bus
- A metric record (when the telemetry engine exists)

---

## Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Kernel Pattern | Microkernel | Minimal core, extensible via modules |
| Communication | Event Bus (internal) | Decoupling, observability, resilience |
| DI Container | Custom lightweight | No external dependency, type-safe, explicit |
| Configuration | Pydantic v2 Settings | Validation, env support, type safety |
| Logging | Loguru | Superior to stdlib logging, structured, async |
| API Framework | FastAPI | Async, Pydantic integration, OpenAPI docs |
| ORM | SQLAlchemy 2 (async) | Mature, flexible, async-native |
| Validation | Pydantic v2 | Every layer validates data at boundaries |

---

## Data Flow

### Request Lifecycle

```
1. HTTP Request → API Gateway
2. Gateway validates input (Pydantic)
3. Gateway emits RequestReceived event
4. Task Router picks up event, routes to Planner
5. Planner emits PlanningStarted event
6. Planner decomposes task, emits sub-tasks
7. Execution Engine picks up sub-tasks
8. Execution emits events for each step
9. Memory Engine records execution context
10. Evaluation Engine scores results
11. Telemetry Engine captures metrics
12. Gateway returns response
```

---

## Future Evolution

The architecture anticipates:

1. **Horizontal Scaling** — Stateless engines behind message queues
2. **Multi-Tenancy** — Tenant isolation at the gateway and storage layers
3. **Federated Deployments** — Cross-instance event bridges
4. **Dynamic Plugin Loading** — Hot-reload of module code
5. **Graph-Based Task Representation** — Neo4j for complex dependency resolution
6. **Agentic Loops** — Self-improving execution via evaluation feedback
7. **Multi-Modal Support** — Pluggable model backends (OpenAI, Anthropic, Ollama)
