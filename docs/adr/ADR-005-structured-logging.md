# ADR-005: Structured Logging with Loguru

## Status

Accepted

## Context

Every important operation in ATLAS must produce structured logs. Requirements:

- Structured output (JSON in production, human-readable in development)
- Async-safe (no blocking in the event loop)
- Contextual binding (attach task_id, module name, correlation id to log entries)
- Multiple sinks (console, file, future: external aggregators)
- Future OpenTelemetry compatibility

## Decision

We use **Loguru** wrapped behind an ATLAS logging facade.

### Design

```python
# app/logging/service.py — facade over loguru
class LoggingService:
    def configure(self, config: LoggingConfig) -> None: ...
    def get_logger(self, module: str) -> Logger: ...

# Usage in modules — always bound with context
logger = logging_service.get_logger("kernel")
logger.info("Module registered", module_name="tasks", version="1.0.0")
```

### Why a Facade?

Modules never import `loguru` directly. They receive a logger through DI. This means:
- We can replace Loguru without touching module code
- We can enforce structured fields (module, correlation_id)
- Testing can capture log output through the facade

### Sink Configuration

| Environment | Sinks | Format |
|-------------|-------|--------|
| Development | Console | Colored, human-readable |
| Testing | Captured | Plain, minimal |
| Production | Console + File | JSON (serialize=True) |

### OpenTelemetry Readiness

The facade design allows adding an OTel handler as another sink when Phase 8 (Telemetry) arrives — log records will be exported as OTel LogRecords without changing call sites.

## Consequences

### Positive

- Beautiful development experience (colors, formatting)
- JSON structured logs in production
- Context binding (`logger.bind(...)`) for correlation
- No handler configuration boilerplate (unlike stdlib logging)
- Async-safe with `enqueue=True`

### Negative

- Loguru is a global singleton internally — the facade mitigates this by controlling all configuration in one place
- Non-standard compared to stdlib `logging` — mitigated with an intercept handler that routes stdlib logging (from libraries like SQLAlchemy, uvicorn) into Loguru

### Alternatives Considered

| Alternative | Assessment |
|-------------|------------|
| stdlib logging | Verbose configuration, no structured support without extra libs |
| structlog | Excellent structured logging, but more configuration surface; Loguru covers our needs with less setup |
| Loguru (chosen) | Simple, structured, async-safe, great DX |
