# ADR-008: Repository Pattern with Unit of Work

## Status

Accepted

## Context

Storage operations must be transactional, testable, and backend-agnostic. Repositories must not expose SQL or database-specific APIs. Multiple repositories may participate in a single transaction.

## Decision

We implement the **Repository pattern** with a **Unit of Work**.

### Repository

```python
class BaseRepository[TModel, TId](ABC):
    async def add(self, model: TModel) -> TModel: ...
    async def get(self, id: TId) -> TModel | None: ...
    async def update(self, model: TModel) -> TModel: ...
    async def delete(self, id: TId) -> None: ...
    async def list(self, ...) -> list[TModel]: ...
```

Domain-specific repositories extend this:

```python
class EventRepository(BaseRepository[EventRecord, str]):
    async def stream_by_correlation(self, correlation_id: str) -> list[EventRecord]: ...
```

### Unit of Work

```python
class UnitOfWork(ABC):
    @property
    def events(self) -> EventRepository: ...
    @property
    def configs(self) -> ConfigRepository: ...

    async def commit(self) -> None: ...
    async def rollback(self) -> None: ...
    async def flush(self) -> None: ...
```

The UoW is the **transaction boundary**. Multiple repositories inside one UoW share a database connection and transaction. The UoW is injected via DI and scoped per operation.

## Consequences

### Positive
- Repositories are testable with in-memory implementations
- UoW coordinates multi-repository transactions without coupling repositories to each other
- Explicit transaction boundaries (no auto-commit surprises)

### Negative
- UoW requires a factory or scoped DI to create instances per operation
- Python's context manager pattern is the most ergonomic but requires careful `async with` handling

## Default Gone

The **simplest** repository gets CRUD for free. Only override when behavior differs.
