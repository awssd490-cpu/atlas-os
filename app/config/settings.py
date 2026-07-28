"""ATLAS configuration models.

Uses Pydantic v2 :class:`BaseSettings` to load from environment variables
and ``.env`` files.  Configuration is validated at startup so every
consumer can assume it is valid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseModel):
    """Application-level metadata."""

    name: str = Field(default="atlas", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    environment: str = Field(
        default="development",
        description="Runtime environment (development, testing, production)",
    )
    debug: bool = Field(default=True, description="Enable debug mode")


class ServerConfig(BaseModel):
    """HTTP server configuration."""

    host: str = Field(default="0.0.0.0", description="Bind address")
    port: int = Field(default=8000, ge=1, le=65535, description="Bind port")
    reload: bool = Field(default=True, description="Hot-reload on code changes")
    workers: int = Field(default=1, ge=1, le=64, description="Number of worker processes")
    timeout_keepalive: int = Field(default=30, ge=1, description="Keepalive timeout (seconds)")


class LoggingConfig(BaseModel):
    """Logging subsystem configuration."""

    level: str = Field(default="DEBUG", description="Minimum log level")
    format: str = Field(default="colored", description="Output format: ``colored`` or ``json``")
    sinks: list[str] = Field(default_factory=lambda: ["console"], description="Active log sinks")
    file_path: str | None = Field(default=None, description="Path to log file (when ``file`` sink is used)")
    queue_size: int = Field(default=65536, description="Internal log message queue size")
    serialize: bool = Field(default=False, description="Force JSON serialization")


class DatabaseConfig(BaseModel):
    """Persistence configuration (future — Phase 2)."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    database: str = Field(default="atlas", description="Database name")
    username: str = Field(default="postgres", description="Database user")
    password: SecretStr = Field(default=SecretStr(""), description="Database password")
    pool_min: int = Field(default=2, ge=1, description="Minimum connection pool size")
    pool_max: int = Field(default=16, ge=1, description="Maximum connection pool size")
    echo: bool = Field(default=False, description="Emit SQL statements to logs")


class StorageConfig(BaseModel):
    """Storage subsystem configuration."""

    sqlite_path: str = Field(
        default="data/atlas.db",
        description="Path to the SQLite database file",
    )
    sqlite_pool_size: int = Field(
        default=1,
        ge=1,
        le=16,
        description="SQLite connection pool size",
    )
    cache_ttl_default: int = Field(
        default=300,
        ge=0,
        description="Default cache TTL in seconds (0 = no expiry)",
    )
    cache_max_size: int = Field(
        default=10_000,
        ge=1,
        description="Maximum cache entries (LRU eviction)",
    )
    vector_dimension_default: int = Field(
        default=1536,
        ge=1,
        description="Default embedding dimension",
    )
    vector_namespaces: list[str] = Field(
        default_factory=lambda: ["default"],
        description="Registered vector namespaces",
    )
    object_store_path: str = Field(
        default="data/objects",
        description="Local object store base path",
    )
    event_store_retention_days: int = Field(
        default=90,
        ge=0,
        description="Event store retention in days (0 = indefinite)",
    )


class MemoryConfig(BaseModel):
    """Memory subsystem configuration and policy defaults."""

    default_ttl_seconds: float = Field(
        default=86400.0,
        ge=0,
        description="Default TTL for short-term memories in seconds (0 = no expiry)",
    )
    long_term_ttl_seconds: float = Field(
        default=0.0,
        ge=0,
        description="TTL for long-term memories (0 = no expiry)",
    )
    working_ttl_seconds: float = Field(
        default=300.0,
        ge=0,
        description="TTL for working memories in seconds",
    )
    max_short_term: int = Field(
        default=100,
        ge=1,
        description="Maximum short-term memories per namespace",
    )
    max_working: int = Field(
        default=20,
        ge=1,
        description="Maximum working memories per namespace",
    )
    default_importance: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Default importance for new memories",
    )
    importance_decay_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Importance decay rate per day (0 = no decay)",
    )
    importance_promotion_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Importance above which memories are promoted to long-term",
    )
    archive_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Importance below which memories become archive candidates",
    )
    gc_interval_seconds: float = Field(
        default=3600.0,
        ge=60.0,
        description="Interval between garbage collector sweeps in seconds",
    )
    grace_period_seconds: float = Field(
        default=604800.0,
        ge=0,
        description="Grace period for FORGOTTEN memories before hard delete (7 days)",
    )
    context_token_budget: int = Field(
        default=4096,
        ge=256,
        description="Default token budget for context assembly",
    )
    enable_auto_archive: bool = Field(
        default=True,
        description="Automatically archive low-importance memories",
    )


class AtlasSettings(BaseSettings):
    """Root settings object — merge of defaults, ``.env``, and env vars.

    Source priority (highest wins):

    1. Environment variable (e.g. ``ATLAS_APP__ENVIRONMENT=production``)
    2. ``.env`` file in the project root
    3. Default values above
    """

    model_config = SettingsConfigDict(
        env_prefix="ATLAS_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app: AppConfig = AppConfig()
    server: ServerConfig = ServerConfig()
    logging: LoggingConfig = LoggingConfig()
    database: DatabaseConfig = DatabaseConfig()
    storage: StorageConfig = StorageConfig()
    memory: MemoryConfig = MemoryConfig()
