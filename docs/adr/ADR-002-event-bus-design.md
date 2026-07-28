# ADR-002: Event Bus Design

## Status

Accepted

## Context

The microkernel architecture requires decoupled communication between engines. We need a mechanism that:

- Allows asynchronous, decoupled interaction between components
- Supports both fire-and-forget and publish-subscribe patterns
- Is observable by default
- Can be replaced with a distributed message queue when scaling

## Decision

We implement an **in-process, async, typed Event Bus** as a kernel service.

### Design

```python
# Event is a typed Pydantic model
class TaskCreated(Event):
    task_id: str
    task_type: str
    payload: dict

# Handlers are async callables registered by type
@event_bus.on(TaskCreated)
async def handle_task_created(event: TaskCreated) -> None:
    ...

# Emission is async
await event_bus.emit(TaskCreated(task_id="...", ...))
```

### Key Properties

1. **Typed Events** — Every event is a Pydantic model. Handlers receive typed objects, not raw dicts.

2. **Async Dispatch** — Events are dispatched to handlers concurrently via `asyncio.gather()`.

3. **Handler Isolation** — A failing handler does not affect other handlers. Errors are logged and reported.

4. **Observability** — Every emit produces a structured log entry with event type, payload size, and handler count.

5. **No Middleware in Phase 1** — Filtering, transformation, and enrichment will be added as needed, not speculatively.

## Consequences

### Positive

- Fully typed — event contracts are self-documenting
- Async-native — no thread pool, no blocking
- Observable by default — every event is logged
- Easy to test — mock the event bus in unit tests
- Straightforward to replace with Redis Pub/Sub or NATS (same interface, different implementation)

### Negative

- In-process only — not distributed (Phase 2+ will address this with a distributed event bus)
- No guaranteed delivery — if a handler crashes, the event is lost (acceptable for Phase 1; durable events come with the distributed bus)
- No fan-out limiting — broadcast to all handlers (acceptable for Phase 1 scale)

### Alternatives Considered

| Alternative | Assessment |
|-------------|------------|
| Message Queue (RabbitMQ/NATS) | Premature distribution. Adds deployment complexity. Not needed until multi-process. |
| Redis Pub/Sub | Same as above. Also, no persistence — messages are lost on disconnect. |
| Observer Pattern (sync) | Tight coupling, error propagation, hard to trace. |
| Typed Event Bus (chosen) | Right level of abstraction for Phase 1. Replaceable when scaling. |

## Migration Path

To distribute the event bus:
1. Implement `DistributedEventBus` adhering to the same `EventBus` interface
2. Swap in the DI container
3. Engine code does not change

## Open Questions

- Should we add event versioning before Phase 2? Consider if event schemas change frequently across engines.
