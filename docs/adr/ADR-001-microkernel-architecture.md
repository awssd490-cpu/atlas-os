# ADR-001: Microkernel Architecture

## Status

Accepted

## Context

ATLAS needs an architecture that can grow from a single-process development system to a distributed multi-engine production platform. The architecture must support:

- Independent development of engines by different teams
- Hot-swapping implementations without system downtime
- Clear separation of concerns between infrastructure and domain logic
- Testability of each component in isolation
- Gradual evolution from monolith to distributed system

## Decision

We adopt a **microkernel architecture** where:

1. **The Kernel** is minimal — responsible only for: lifecycle, configuration, dependency injection, module registry, event bus, logging, and health checks.

2. **Engines** are self-contained modules that register with the kernel. They contain all business logic.

3. **Communication** between engines occurs through the event bus, not direct calls.

4. **The API layer** is a thin transport that delegates to kernel services.

## Consequences

### Positive

- Kernel remains stable as the system grows
- Engines can be developed, tested, and deployed independently
- New capabilities (e.g., telemetry) can be added without modifying existing code
- Testing an engine requires only mocking the event bus and DI container
- The architecture can evolve from in-process to distributed by replacing the in-memory event bus with a message queue — no engine code changes needed

### Negative

- Event-driven communication adds indirection compared to direct calls
- Debugging event flows requires good observability tooling (tracing, event logging)
- Module dependency management requires explicit declaration and validation
- Slightly higher initial complexity than a monolithic design

### Trade-offs Considered

| Alternative | Pros | Cons |
|-------------|------|------|
| Monolithic | Simpler initially | Tight coupling, hard to scale, impossible to distribute |
| Plugin-based | Hot-reload, isolation | Complex lifecycle, versioning problems, classloader issues |
| Service-oriented | Full distribution | Network overhead, serialization cost, deployment complexity |
| Microkernel (chosen) | Balance of all concerns | Event bus is an additional abstraction |

## Migration Path

If the microkernel proves too complex for the current scale, we can:
1. Keep the interface contracts intact
2. Compile engines into a single deployment unit
3. Revert to in-process event dispatching

The interface contracts remain the same in all cases. No engine code changes are needed.
