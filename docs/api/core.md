# Core Infrastructure (`app.core`)

## Overview

The `app.core` package provides foundational infrastructure used across all Atlas components: configuration management, structured logging, health monitoring, concurrency control, and reliability (retry) utilities.

All core subsystems follow the same architectural conventions: frozen dataclass models, `AtlasError`-derived error hierarchies, and deterministic serialization where applicable.

## Packages

- [`app.core.config`](#appcoreconfig) — Configuration management
- [`app.core.log`](#appcorelog) — Structured logging
- [`app.core.health`](#appcorehealth) — Health monitoring
- [`app.core.concurrency`](#appcoreconcurrency) — Concurrency & resource management
- [`app.core.reliability`](#appcorereliability) — Retry & recovery

---

## `app.core.config`

### Purpose

Centralised configuration for an Atlas application instance. Supports loading from Python dicts, JSON files, and environment variables with validation.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `AtlasConfig` | Frozen dataclass | Top-level settings (environment, debug, log_level, random_seed, metadata) |
| `ConfigLoader` | Static class | Load/save config from dict/JSON/env |
| `ConfigurationError` | Exception | Base configuration error |
| `InvalidConfiguration` | Exception | Validation failure |

### `AtlasConfig`

```python
@dataclass(frozen=True)
class AtlasConfig:
    environment: str = "development"     # one of: development, testing, staging, production
    debug: bool = False
    log_level: str = "INFO"              # one of: DEBUG, INFO, WARNING, ERROR, CRITICAL
    random_seed: int = 42
    metadata: Mapping[str, Any] = {}
```

**Methods:**

- `validate()` — raises `InvalidConfiguration` if:
  - `environment` is not one of the four allowed values
  - `log_level` is not a recognised level name
  - `random_seed` is negative

**Example:**

```python
cfg = AtlasConfig(environment="production", debug=True)
cfg.validate()
```

### `ConfigLoader`

```python
class ConfigLoader:
    @staticmethod
    def from_dict(data: dict[str, Any]) -> AtlasConfig: ...
    @staticmethod
    def from_json(path: str) -> AtlasConfig: ...
    @staticmethod
    def from_env(prefix: str = "ATLAS_") -> AtlasConfig: ...
    @staticmethod
    def to_dict(config: AtlasConfig) -> dict[str, Any]: ...
    @staticmethod
    def to_json(config: AtlasConfig, path: str) -> None: ...
```

**Environment variable mapping** (default prefix `ATLAS_`):

| Variable | Config field |
|---|---|
| `ATLAS_ENVIRONMENT` | `environment` |
| `ATLAS_DEBUG` | `debug` (accepts `true`/`1`/`yes` for `True`) |
| `ATLAS_LOG_LEVEL` | `log_level` |
| `ATLAS_RANDOM_SEED` | `random_seed` (parsed as `int`) |

**Example:**

```python
# From JSON
cfg = ConfigLoader.from_json("config.json")

# From environment
cfg = ConfigLoader.from_env()  # reads ATLAS_* vars

# Round-trip
ConfigLoader.to_json(cfg, "config.json")
loaded = ConfigLoader.from_json("config.json")
assert loaded == cfg
```

**Raises:**

- `InvalidConfiguration` — When input is not a dict, values fail validation
- `ConfigurationError` — When JSON file cannot be read or parsed

---

## `app.core.log`

### Purpose

Structured logging subsystem that produces immutable `LogRecord` objects written as single-line JSON to stderr.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `AtlasLogger` | Class | Structured logger with 6 log methods |
| `LogRecord` | Frozen dataclass | Immutable log record |
| `JsonFormatter` | Class | Formats `LogRecord` to deterministic JSON |
| `LoggingError` | Exception | Base logging error |
| `InvalidLogLevel` | Exception | Invalid level name |

### `LogRecord`

```python
@dataclass(frozen=True)
class LogRecord:
    timestamp: str = ""             # ISO-8601 with UTC
    level: str = ""
    logger: str = ""
    message: str = ""
    metadata: Mapping[str, Any] = {}
```

### `AtlasLogger`

```python
class AtlasLogger:
    def __init__(self, name: str = "atlas", *, level: str = "INFO",
                 formatter: JsonFormatter | None = None): ...

    def debug(self, message: str, **metadata: Any) -> LogRecord: ...
    def info(self, message: str, **metadata: Any) -> LogRecord: ...
    def warning(self, message: str, **metadata: Any) -> LogRecord: ...
    def error(self, message: str, **metadata: Any) -> LogRecord: ...
    def critical(self, message: str, **metadata: Any) -> LogRecord: ...
    def exception(self, message: str, exception: BaseException | None = None,
                  **metadata: Any) -> LogRecord: ...
    def is_enabled_for(self, level: str) -> bool: ...

    @property
    def name(self) -> str: ...
    @property
    def level(self) -> str: ...
```

**Behaviour:**

- Each log method returns a `LogRecord` — never `None`.
- Records are written to **stderr** as single-line JSON only when the log level meets or exceeds the configured threshold.
- When suppressed, a `LogRecord` is still returned but **without a timestamp** (timestamp `""`) and **without writing to stderr**.
- `exception()` automatically captures exception type and message in `metadata.exception`.
- Timestamps are ISO-8601 with UTC timezone.

**Example:**

```python
log = AtlasLogger("my.service", level="DEBUG")
log.info("Server started", port=8080, env="production")

try:
    risky_operation()
except ValueError as exc:
    log.exception("Operation failed", exception=exc, component="worker")
```

### `JsonFormatter`

```python
class JsonFormatter:
    def format(self, record: LogRecord) -> str: ...
```

Output is a single-line JSON string with sorted keys and `ensure_ascii=False`.

**Example output:**

```json
{"level":"INFO","logger":"my.service","message":"Server started","metadata":{"port":8080},"timestamp":"2026-07-30T12:00:00+00:00"}
```

### Errors

- `LoggingError(AtlasError)` — code `"LOGGING_ERROR"`
- `InvalidLogLevel(LoggingError)` — code `"INVALID_LOG_LEVEL"`, raised when an unrecognised level is passed to `AtlasLogger.__init__`

---

## `app.core.health`

### Purpose

Lightweight health monitoring. Register sync or async check functions, execute them individually or in bulk, and receive immutable `HealthCheck` results with timing.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `HealthMonitor` | Static class | Register checks, run check/check_all |
| `HealthStatus` | Enum | UNKNOWN, HEALTHY, DEGRADED, UNHEALTHY |
| `HealthCheck` | Frozen dataclass | Immutable check result |
| `HealthError` | Exception | Base health error |
| `HealthCheckNotFound` | Exception | Unknown check name |
| `DuplicateHealthCheck` | Exception | Duplicate registration |
| `register()` | Function | Global registry — add check |
| `unregister()` | Function | Global registry — remove check |
| `get()` | Function | Global registry — lookup |
| `list_checks()` | Function | Global registry — list names |
| `clear_checks()` | Function | Global registry — clear all |

### `HealthStatus`

```python
class HealthStatus(enum.Enum):
    UNKNOWN = 0
    HEALTHY = 1
    DEGRADED = 2
    UNHEALTHY = 3
```

### `HealthCheck`

```python
@dataclass(frozen=True)
class HealthCheck:
    name: str = ""
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    duration_ms: float = 0.0
    metadata: Mapping[str, Any] = {}
```

### `HealthMonitor`

```python
class HealthMonitor:
    @staticmethod
    def register(name: str, fn: Callable) -> None: ...
    @staticmethod
    def unregister(name: str) -> None: ...
    @staticmethod
    async def check(name: str) -> HealthCheck: ...
    @staticmethod
    async def check_all() -> list[HealthCheck]: ...
    @staticmethod
    def list_checks() -> list[str]: ...
```

**Check function signatures** (any of these):

```python
fn() -> HealthStatus
fn() -> bool                      # True → HEALTHY, False → UNHEALTHY
fn() -> tuple[HealthStatus, str]  # (status, message)
fn() -> tuple[HealthStatus, str, dict]  # (status, message, metadata)
fn() -> str                       # → HEALTHY with message
```

Functions can be sync or async. Exceptions are caught and returned as `UNHEALTHY` with the error message — they never propagate.

**Example:**

```python
async def db_check() -> HealthStatus:
    return HealthStatus.HEALTHY

def disk_check() -> tuple[HealthStatus, str]:
    return HealthStatus.HEALTHY, "disk OK"

HealthMonitor.register("database", db_check)
HealthMonitor.register("disk", disk_check)

results = await HealthMonitor.check_all()
for check in results:
    print(f"{check.name}: {check.status.name} ({check.duration_ms:.1f}ms)")
```

### Errors

- `HealthError(AtlasError)` — code `"HEALTH_ERROR"`
- `HealthCheckNotFound(HealthError)` — code `"HEALTH_CHECK_NOT_FOUND"`
- `DuplicateHealthCheck(HealthError)` — code `"DUPLICATE_HEALTH_CHECK"`

---

## `app.core.concurrency`

### Purpose

Async concurrency limiter (`ConcurrencyLimiter`) and resource lifecycle manager (`ResourceManager`) for controlling access to shared resources and tracking their state.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `ConcurrencyLimiter` | Class | Async semaphore with context manager |
| `ResourceManager` | Class | Named resource registry with state tracking |
| `ResourceState` | Enum | CREATED, OPEN, CLOSED |
| `ManagedResource` | Frozen dataclass | Immutable resource record |
| `ConcurrencyError` | Exception | Base concurrency error |
| `ResourceNotFound` | Exception | Unknown resource name |
| `DuplicateResource` | Exception | Duplicate registration |

### `ConcurrencyLimiter`

```python
class ConcurrencyLimiter:
    def __init__(self, max_concurrent: int = 10): ...
    async def acquire(self) -> None: ...
    def release(self) -> None: ...
    @property
    def available_permits(self) -> int: ...
    async def __aenter__(self) -> ConcurrencyLimiter: ...
    async def __aexit__(self, ...) -> None: ...
```

**Raises:**

- `ValueError` — if `max_concurrent < 1`

**Example:**

```python
limiter = ConcurrencyLimiter(5)

# Context manager
async with limiter:
    await access_shared_resource()

# Manual acquire/release
await limiter.acquire()
try:
    await access_shared_resource()
finally:
    limiter.release()
```

### `ResourceManager`

```python
class ResourceManager:
    def __init__(self): ...
    def register(self, name: str, resource: Any = None) -> ManagedResource: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> Any: ...
    def get_record(self, name: str) -> ManagedResource: ...
    def list_resources(self) -> list[ManagedResource]: ...
    def list_names(self) -> list[str]: ...
    def open(self, name: str) -> ManagedResource: ...
    def close(self, name: str) -> ManagedResource: ...
    def close_all(self) -> list[ManagedResource]: ...
```

**Lifecycle:** `register()` → `CREATED` → `open()` → `OPEN` → `close()` → `CLOSED`

**Example:**

```python
mgr = ResourceManager()
mgr.register("db_pool", connection_pool)
mgr.open("db_pool")

pool = mgr.get("db_pool")
# ... use pool ...

mgr.close("db_pool")
records = mgr.close_all()  # close everything
```

### Errors

- `ConcurrencyError(AtlasError)` — code `"CONCURRENCY_ERROR"`
- `ResourceNotFound(ConcurrencyError)` — code `"RESOURCE_NOT_FOUND"`
- `DuplicateResource(ConcurrencyError)` — code `"DUPLICATE_RESOURCE"`

---

## `app.core.reliability`

### Purpose

Retry utility with exponential backoff. Executes sync or async callables, retrying on configurable exception types with capped exponential delay.

### Public API

| Symbol | Kind | Description |
|---|---|---|
| `RetryExecutor` | Class | Execute with retry |
| `RetryPolicy` | Frozen dataclass | Retry configuration |
| `RetryResult` | Frozen dataclass | Immutable retry outcome |
| `ReliabilityError` | Exception | Base reliability error |
| `InvalidRetryPolicy` | Exception | Invalid policy values |

### `RetryPolicy`

```python
@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3                     # >= 1
    initial_delay_ms: float = 100.0           # >= 0
    backoff_multiplier: float = 2.0           # >= 1.0
    max_delay_ms: float = 5000.0              # >= 0
    retry_exceptions: tuple[type[Exception], ...] = ()  # empty = all
    metadata: Mapping[str, Any] = {}
    def validate(self) -> None: ...
```

### `RetryResult`

```python
@dataclass(frozen=True)
class RetryResult:
    attempts: int = 0
    success: bool = False
    duration_ms: float = 0.0
    metadata: Mapping[str, Any] = {}
```

### `RetryExecutor`

```python
class RetryExecutor:
    def __init__(self, policy: RetryPolicy | None = None): ...
    async def execute(self, fn: Callable, *args, retry_policy: RetryPolicy | None = None,
                      **kwargs) -> RetryResult: ...
    @property
    def policy(self) -> RetryPolicy: ...
```

**Backoff formula:** `delay = min(initial_delay × multiplier^(attempt-1), max_delay)`

**Exception filtering:**

- If `retry_exceptions` is empty (default), **all** exceptions trigger a retry.
- If `retry_exceptions` is non-empty, only matching exception types trigger a retry. Non-matching types fail immediately.
- After the final attempt fails, the executor returns a `RetryResult` with `success=False` — it **does not** raise.

**Example:**

```python
executor = RetryExecutor()

# Use default policy (3 attempts, 100ms → 200ms → 400ms backoff)
result = await executor.execute(maybe_flaky_fn, arg1)

# Custom policy
policy = RetryPolicy(
    max_attempts=5,
    initial_delay_ms=50.0,
    retry_exceptions=(ConnectionError, TimeoutError),
)
result = await executor.execute(flaky_api_call, retry_policy=policy)

if result.success:
    print(f"Succeeded after {result.attempts} attempts")
else:
    print(f"Failed after {result.attempts} attempts: {result.metadata['last_error']}")
```

### Errors

- `ReliabilityError(AtlasError)` — code `"RELIABILITY_ERROR"`
- `InvalidRetryPolicy(ReliabilityError)` — code `"INVALID_RETRY_POLICY"`, raised when `validate()` detects:
  - `max_attempts < 1`
  - `initial_delay_ms < 0`
  - `backoff_multiplier < 1.0`
  - `max_delay_ms < 0`
