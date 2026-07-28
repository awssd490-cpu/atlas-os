# ADR-003: Dependency Injection Container

## Status

Accepted

## Context

ATLAS has multiple components that depend on each other — configuration, logging, event bus, database connections, engine services. We need a way to:

- Wire dependencies together without hard-coding instantiation
- Make every component testable by substituting implementations
- Avoid the "service locator" anti-pattern where components fetch dependencies from a global registry
- Support async initialization (common in I/O-bound services)
- Maintain explicit dependency declarations rather than magic auto-wiring

## Decision

We implement a **lightweight, explicit DI container** custom for ATLAS.

### Design

```python
container = DIContainer()

# Register factories
container.register(ConfigService, lambda c: ConfigService(...))
container.register(EventBus, lambda c: EventBus())
container.register_singleton(DatabasePool, lambda c: DatabasePool(c.resolve(ConfigService)))

# Resolve
config = await container.resolve(ConfigService)
```

### Key Properties

1. **Explicit Registration** — No auto-scanning or magic discovery. Dependencies are registered explicitly.

2. **Singleton vs Transient** — Singletons are created once and cached. Transients are created on every resolution.

3. **Async Resolution** — Factories can be async (e.g., database connection pools).

4. **Lazy Initialization** — Nothing is created until resolved. This avoids circular initialization issues.

5. **Lifecycle Awareness** — The container manages singleton lifecycle: `container.init()` and `container.shutdown()`.

6. **Replaceability** — Register a different factory to swap implementations (e.g., mock for test).

### Why Not Existing Libraries?

| Library | Assessment |
|---------|------------|
| pytest-dependency | Test-only, not for production |
| FastAPI Depends | Tied to request scope, not suitable for kernel services |
| python-dependency-injector | Heavy (400KB+), complex DSL, C-extensions, over-engineered for our needs |
| Custom (chosen) | ~100 lines, explicit, typed, no magic, full control |

We avoid heavy dependencies on DI frameworks. A custom container that does just enough is preferable to learning a DI framework's DSL and debugging its edge cases.

## Consequences

### Positive

- Zero external dependency for DI
- Full control over lifecycle and error handling
- Explicit dependency graph — easy to reason about
- Easy to mock in tests (register test doubles)
- Works with async components seamlessly

### Negative

- No auto-discovery — dependencies must be registered manually
- No circular dependency detection (will be caught as `KeyError` at resolution time)

### Alternatives Considered

| Alternative | Assessment |
|-------------|------------|
| Manual DI (constructors) | Satisfies DI principle but scatters wiring across codebase. Violates DRY. |
| Service Locator | Anti-pattern — hides dependencies, makes testing harder. |
| Existing library | Heavy, magic, DSL-heavy. Unnecessary for our needs. |
| Custom container (chosen) | Lightweight, explicit, typed, maintainable. |
