# ADR-004: Configuration Management

## Status

Accepted

## Context

ATLAS runs in multiple environments (development, testing, production). Configuration comes from multiple sources: environment variables, `.env` files, and potentially secrets managers. Configuration must be:

- Validated at startup (not silently using defaults)
- Type-safe (typed access, not raw dicts)
- Hierarchical (nested configuration sections)
- Environment-aware (different values per environment)
- Auditable (can dump current config with secrets masked)

## Decision

We use **Pydantic v2 Settings** as the foundation for configuration, wrapped in a `ConfigService` that manages loading, validation, and access.

### Design

```python
class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "atlas"
    username: str = "postgres"
    password: SecretStr = SecretStr("")

class LoggingConfig(BaseModel):
    level: str = "INFO"
    sinks: list[str] = ["console"]

class AppConfig(BaseModel):
    name: str = "atlas"
    version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

class AtlasSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_nested_delimiter="__",
        env_file=".env",
    )

    app: AppConfig = AppConfig()
    database: DatabaseConfig = DatabaseConfig()
    logging: LoggingConfig = LoggingConfig()
```

### Configuration Sources (priority order)

1. Environment variables (highest)
2. `.env` file
3. Default values (lowest)

### Secrets Handling

- All secrets use `SecretStr` to prevent accidental exposure
- Config dump masks secrets automatically
- Future: Vault/Secret Store integration as a source

## Consequences

### Positive

- Schema validation catches misconfiguration at startup, not at runtime
- Full IDE support with type hints on configuration values
- Environment variables map naturally to settings fields
- Secret fields are protected by default

### Negative

- Pydantic Settings is a dependency (already present for FastAPI)
- Nested configuration requires the `env_nested_delimiter` convention
- Hot-reload of configuration is not supported (future: watch `.env` changes)

### Why Not YAML/JSON Config Files?

Environment-based configuration is preferred over config files because:
- Config files vary per environment (dev/staging/prod) and must be managed separately
- Environment variables integrate with container orchestration (Docker, Kubernetes)
- Secrets should never be in config files (env vars can be injected by secret stores)
- 12-Factor App methodology

## Migration Path

To add a secrets manager:
1. Implement a `SecretsSource` that implements `ConfigSource` protocol
2. Register it in the config service as an additional source
3. Configuration consumers don't change
