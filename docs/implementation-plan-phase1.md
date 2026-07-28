# Phase 1 Implementation Plan — Atlas Kernel

## Scope

Kernel infrastructure only. No AI, no persistence, no business logic.

## Component Build Order

Dependencies flow downward — each step builds only on the ones above it.

### Step 1: Configuration (`app/config/`)
- `settings.py` — Pydantic Settings models (`AtlasSettings`, `AppConfig`, `LoggingConfig`, `ServerConfig`)
- `service.py` — `PydanticConfigService(ConfigService)` implementing dotted-key access, section access, secret-masked dump
- Tests: env var loading, `.env` loading, validation failure, secret masking, dotted-key lookup

### Step 2: Logging (`app/logging/`)
- `service.py` — `LoguruLoggingService(LoggingService)` — configures sinks from `LoggingConfig`, environment-aware formats (colored dev / JSON prod), stdlib logging intercept
- Tests: logger binding, level filtering, sink configuration per environment

### Step 3: DI Container (`app/di/`)
- `container.py` — `Container(DIContainer)` — sync/async factories, singleton caching, `init_singletons()`, `dispose()`, resolution error with registered-keys context
- Tests: register/resolve, singleton identity, transient uniqueness, async factories, unregistered error, dispose calls close/shutdown

### Step 4: Event Bus (`app/events/`)
- `base.py` — `DataEvent(Event, BaseModel)` base with `event_id`, `timestamp`, auto event-type
- `bus.py` — `InProcessEventBus(EventBus)` — typed subscription, concurrent dispatch, handler isolation, stats counters
- `kernel_events.py` — kernel lifecycle events: `KernelBooting`, `KernelBooted`, `KernelShuttingDown`, `ModuleRegistered`, `ModuleBooted`, `ModuleFailed`, `ModuleStopped`
- Tests: publish/subscribe, subtype dispatch, handler isolation (one fails, others run), emit_and_wait error propagation, unsubscribe, stats

### Step 5: Module Registry (`app/modules/`)
- `registry.py` — `ModuleRegistry` — registration, duplicate detection, dependency validation, topological boot ordering (Kahn's algorithm), cycle detection
- Tests: registration, duplicates rejected, dependency order, missing dependency error, cycle detection

### Step 6: Lifecycle Manager (`app/lifecycle/`)
- `manager.py` — `LifecycleManager` — boots modules in dependency order, rollback on failure (shuts down already-booted modules in reverse), graceful shutdown with per-module timeout
- Tests: boot order, failure rollback, shutdown reverse order, shutdown timeout handling

### Step 7: Kernel (`app/kernel/`)
- `kernel.py` — `Kernel(AtlasKernel)` — composes all of the above; state machine enforcement (no double boot); wires core services into DI; runs health checks; emits lifecycle events
- `builder.py` — `KernelBuilder` — fluent construction for tests and entry points
- Tests: full boot/shutdown cycle, state transitions, invalid transitions rejected, health aggregation, uptime

### Step 8: API Layer (`app/api/`)
- `app.py` — FastAPI app factory bound to a kernel via lifespan
- `routes/health.py` — `/health`, `/health/live`, `/health/ready`
- `routes/system.py` — `/system/modules`, `/system/events`, `/system/config`
- `schemas.py` — response models (no business logic in routes; routes read from kernel services only)
- Tests: endpoint contracts via httpx AsyncClient against a booted test kernel

### Step 9: Entry Point
- `main.py` — builds kernel, registers modules, runs uvicorn with lifespan integration
- `.env.example`, updated `pyproject.toml` (pydantic-settings, pytest, pytest-asyncio, dev tooling: ruff, mypy)

## Testing Standards

- pytest + pytest-asyncio (`asyncio_mode = "auto"`)
- Every component: unit tests in `tests/unit/<component>/`
- API: integration tests with in-memory kernel (no network, no DB)
- Coverage target: kernel components ≥ 90%

## Explicitly Deferred (not Phase 1)

| Item | Deferred to |
|------|-------------|
| PostgreSQL / SQLAlchemy / Redis wiring | Phase 2 |
| Distributed event bus | Phase 2+ |
| AuthN/AuthZ | Phase 9 |
| OpenTelemetry export | Phase 8 |
| Dynamic module discovery (entry points) | Phase 12 |
