# ATLAS Module System

## Overview

ATLAS uses a modular architecture. Every capability beyond the kernel is packaged as a **module** (also called an **engine**). Modules register with the kernel at boot time and communicate through the event bus.

---

## Module Lifecycle

```
Registered → Booting → Active → Shutting Down → Stopped
                │                        │
                ▼                        ▼
            Failed                    Error
```

| State | Description |
|-------|-------------|
| `REGISTERED` | Module is known to the kernel, not yet initialized |
| `BOOTING` | Module is initializing (async) |
| `ACTIVE` | Module is running and processing events |
| `SHUTTING_DOWN` | Module is stopping (async) |
| `STOPPED` | Module has stopped cleanly |
| `FAILED` | Boot or runtime error occurred |

---

## Module Contract

Every module must implement the `Module` interface:

```python
class Module(ABC):
    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dependencies(self) -> list[str]: ...

    async def boot(self, kernel: AtlasKernel) -> None: ...

    async def shutdown(self) -> None: ...

    def event_handlers(self) -> dict[type[Event], type[EventHandler]]: ...
```

---

## Standard Modules

### 1. Config Module (built into kernel)
- Loads and validates configuration from environment/files
- Provides typed configuration access

### 2. Logging Module (built into kernel)
- Structured logging via Loguru
- Configurable sinks (console, file, external)
- Structured log context

### 3. Storage Module (Phase 2)
- Database session management
- Repository base classes
- Migration management

### 4. Task Module (Phase 3)
- Task creation, routing, tracking
- Task state machine
- Task lifecycle events

### 5. Memory Module (Phase 4)
- Session context management
- Embedding and vector search
- Entity extraction and graph storage
- Memory retrieval pipelines

### 6. Tool Module (Phase 5)
- Tool registration and discovery
- Tool execution sandboxing
- Rate limiting

### 7. Model Module (Phase 6)
- Provider abstraction
- Model routing
- Token management

### 8. Evaluation Module (Phase 7)
- Evaluation pipelines
- Metric collection
- Regression detection

### 9. Telemetry Module (Phase 8)
- OpenTelemetry integration
- Metrics and tracing
- Audit logging

---

## Module Registration

Modules register at kernel startup:

```python
kernel = AtlasKernel()
kernel.register(ConfigModule())
kernel.register(LoggingModule())
kernel.register(TaskModule())
await kernel.boot()
```

Modules can also be discovered automatically via entry points or filesystem scanning (future).

---

## Inter-Module Communication

Modules never import each other directly.

They communicate through the **Event Bus**:

1. Module A emits an event: `await event_bus.emit(TaskCreated(...))`
2. Module B handles the event: `async def handle(event: TaskCreated) -> None`
3. The Event Bus routes events to all registered handlers

This ensures:
- No direct coupling between modules
- Modules can be added/removed without affecting others
- Full observability of inter-module communication
- Easy testing (mock the event bus)
