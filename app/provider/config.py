"""Provider Configuration System.

Every provider receives validated configuration from this subsystem.
No provider should parse environment variables or configuration files
directly.

Supports multiple configuration sources with clear precedence rules.
"""

from __future__ import annotations

import abc
import os
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderCredentials:
    """Provider authentication credentials.

    Supports API key mode and optional organization/project IDs.

    Never exposes secrets in ``__repr__``, ``__str__``, or ``to_dict()``.
    """

    api_key: str = ""
    organization_id: str = ""
    project_id: str = ""

    def __repr__(self) -> str:
        return f"ProviderCredentials(api_key='***{self._suffix}'..., org_id={self.organization_id!r})"

    def __str__(self) -> str:
        return f"ProviderCredentials(api_key=****{self._suffix})"

    def to_dict(self) -> dict[str, str]:
        """Serialize without exposing the full API key."""
        return {
            "api_key": f"***{self._suffix}",
            "organization_id": self.organization_id,
            "project_id": self.project_id,
        }

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    @property
    def _suffix(self) -> str:
        return self.api_key[-4:] if len(self.api_key) >= 4 else ""

    @classmethod
    def empty(cls) -> "ProviderCredentials":
        return cls()


# ---------------------------------------------------------------------------
# Retry / Timeout policies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for request retry behaviour."""

    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    retryable_error_types: tuple[str, ...] = (
        "PROVIDER_RATE_LIMIT_ERROR",
        "PROVIDER_TIMEOUT_ERROR",
        "PROVIDER_UNAVAILABLE_ERROR",
        "PROVIDER_STREAMING_ERROR",
    )

    def is_retryable(self, error_code: str) -> bool:
        return error_code in self.retryable_error_types


@dataclass(frozen=True)
class TimeoutPolicy:
    """Configuration for request timeouts."""

    request_timeout_seconds: float = 60.0
    connect_timeout_seconds: float = 10.0
    stream_timeout_seconds: float = 120.0
    idle_timeout_seconds: float = 30.0


# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationDefaults:
    """Default generation parameters applied to every request.

    These can be overridden per-request by the caller.
    """

    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderEndpoint:
    """Provider API endpoint configuration."""

    base_url: str = ""
    api_path: str = "/v1/messages"
    stream_path: str = "/v1/messages/stream"

    @property
    def request_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.api_path}"

    @property
    def stream_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.stream_path}"

    def is_configured(self) -> bool:
        return bool(self.base_url)


# ---------------------------------------------------------------------------
# Config sources
# ---------------------------------------------------------------------------


class ConfigSource(abc.ABC):
    """Abstract configuration source.

    Implementations read configuration from different origins
    (env vars, dict, files, etc.).
    """

    @abc.abstractmethod
    def load(self) -> dict[str, Any]:
        """Load configuration as a flat dict with dot-delimited keys.

        E.g. ``{"api_key": "...", "timeout.request_timeout_seconds": 60.0}``
        """
        ...

    @property
    @abc.abstractmethod
    def priority(self) -> int:
        """Source priority.  Higher values override lower.

        Convention:
            0 = default values
            10 = constructor arguments
            20 = configuration files
            30 = environment variables
        """
        ...


class DictConfigSource(ConfigSource):
    """Configuration from a Python dictionary (priority 10)."""

    def __init__(self, data: dict[str, Any], priority: int = 10) -> None:
        self._data = data
        self._priority = priority

    def load(self) -> dict[str, Any]:
        return dict(self._data)

    @property
    def priority(self) -> int:
        return self._priority


class EnvConfigSource(ConfigSource):
    """Configuration from environment variables (priority 30).

    Maps prefixed env vars to config keys::

        ATLAS_PROVIDER_API_KEY → api_key
        ATLAS_PROVIDER_MODEL → model
    """

    def __init__(self, prefix: str = "ATLAS_PROVIDER_") -> None:
        self._prefix = prefix

    def load(self) -> dict[str, Any]:
        config: dict[str, Any] = {}
        for key, value in os.environ.items():
            if key.startswith(self._prefix):
                config_key = key[len(self._prefix):].lower()
                config[config_key] = value
        return config

    @property
    def priority(self) -> int:
        return 30


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigValidationResult:
    """Structured result of a configuration validation."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)

    def add_error(self, error: str) -> "ConfigValidationResult":
        return ConfigValidationResult(
            valid=False,
            errors=list(self.errors) + [error],
        )

    @classmethod
    def ok(cls) -> "ConfigValidationResult":
        return cls()

    @classmethod
    def failed(cls, *errors: str) -> "ConfigValidationResult":
        return cls(valid=False, errors=list(errors))


def validate_config(config: dict[str, Any]) -> ConfigValidationResult:
    """Validate a flat provider configuration dictionary.

    Checks required fields, format constraints, and ranges.
    Returns a ``ConfigValidationResult`` with all errors found.
    """
    result = ConfigValidationResult.ok()

    # Required fields
    if not config.get("api_key"):
        result = result.add_error("api_key is required")

    if not config.get("endpoint_base_url") and not config.get("endpoint.base_url"):
        # Not all providers require an endpoint (e.g. Ollama has a default)
        pass

    # Temperature range
    temp = config.get("temperature")
    if temp is not None:
        try:
            t = float(temp)
            if t < 0.0 or t > 2.0:
                result = result.add_error(f"temperature must be in [0.0, 2.0], got {t}")
        except (ValueError, TypeError):
            result = result.add_error(f"temperature must be a number, got {temp!r}")

    # Max tokens
    max_t = config.get("max_tokens")
    if max_t is not None:
        try:
            mt = int(max_t)
            if mt < 1:
                result = result.add_error(f"max_tokens must be >= 1, got {mt}")
        except (ValueError, TypeError):
            result = result.add_error(f"max_tokens must be an integer, got {max_t!r}")

    # Retry limits
    retries = config.get("retry_max_retries") or config.get("retry.max_retries")
    if retries is not None:
        try:
            r = int(retries)
            if r < 0 or r > 20:
                result = result.add_error(f"max_retries must be in [0, 20], got {r}")
        except (ValueError, TypeError):
            result = result.add_error(f"max_retries must be an integer, got {retries!r}")

    return result


# ---------------------------------------------------------------------------
# ProviderConfig — unified configuration model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderConfig:
    """Unified, validated configuration for a provider instance.

    Every provider implementation receives one of these from the factory.
    """

    # Identity
    name: str = ""
    model: str = ""

    # Credentials
    credentials: ProviderCredentials = field(default_factory=ProviderCredentials)

    # Connection
    endpoint: ProviderEndpoint = field(default_factory=ProviderEndpoint)

    # Generation defaults
    generation: GenerationDefaults = field(default_factory=GenerationDefaults)

    # Policies
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    timeout: TimeoutPolicy = field(default_factory=TimeoutPolicy)

    # Extra provider-specific settings
    extra: dict[str, Any] = field(default_factory=dict)

    # Source metadata
    source_priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize without exposing secrets."""
        return {
            "name": self.name,
            "model": self.model,
            "credentials": self.credentials.to_dict(),
            "endpoint": {
                "base_url": self.endpoint.base_url,
                "api_path": self.endpoint.api_path,
            },
            "generation": {
                "model": self.generation.model,
                "max_tokens": self.generation.max_tokens,
                "temperature": self.generation.temperature,
            },
            "retry": {
                "max_retries": self.retry.max_retries,
                "base_delay_seconds": self.retry.base_delay_seconds,
            },
            "timeout": {
                "request_timeout_seconds": self.timeout.request_timeout_seconds,
            },
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        name: str = "",
    ) -> "ProviderConfig":
        """Build a ``ProviderConfig`` from a flat or nested dictionary.

        Accepts both ``{"api_key": "..."}`` and
        ``{"credentials": {"api_key": "..."}}`` formats.
        """
        creds = ProviderCredentials(
            api_key=str(data.get("api_key", data.get("credentials.api_key", ""))),
            organization_id=str(data.get("organization_id", data.get("credentials.organization_id", ""))),
            project_id=str(data.get("project_id", data.get("credentials.project_id", ""))),
        )

        endpoint = ProviderEndpoint(
            base_url=str(data.get("endpoint_base_url", data.get("endpoint.base_url", ""))),
            api_path=str(data.get("endpoint_api_path", data.get("endpoint.api_path", "/v1/messages"))),
            stream_path=str(data.get("endpoint_stream_path", data.get("endpoint.stream_path", "/v1/messages/stream"))),
        )

        generation = GenerationDefaults(
            model=str(data.get("model", "")),
            max_tokens=int(data.get("max_tokens", data.get("generation.max_tokens", 4096))),
            temperature=float(data.get("temperature", data.get("generation.temperature", 0.7))),
            top_p=float(data.get("top_p", data.get("generation.top_p", 1.0))),
            frequency_penalty=float(data.get("frequency_penalty", data.get("generation.frequency_penalty", 0.0))),
            presence_penalty=float(data.get("presence_penalty", data.get("generation.presence_penalty", 0.0))),
        )

        retry = RetryPolicy(
            max_retries=int(data.get("retry_max_retries", data.get("retry.max_retries", 3))),
            base_delay_seconds=float(data.get("retry_base_delay", data.get("retry.base_delay_seconds", 1.0))),
            max_delay_seconds=float(data.get("retry_max_delay", data.get("retry.max_delay_seconds", 60.0))),
            backoff_multiplier=float(data.get("retry_backoff", data.get("retry.backoff_multiplier", 2.0))),
        )

        timeout = TimeoutPolicy(
            request_timeout_seconds=float(data.get("timeout_request", data.get("timeout.request_timeout_seconds", 60.0))),
            connect_timeout_seconds=float(data.get("timeout_connect", data.get("timeout.connect_timeout_seconds", 10.0))),
            stream_timeout_seconds=float(data.get("timeout_stream", data.get("timeout.stream_timeout_seconds", 120.0))),
            idle_timeout_seconds=float(data.get("timeout_idle", data.get("timeout.idle_timeout_seconds", 30.0))),
        )

        # Collect remaining keys as extra
        known_keys = {
            "api_key", "credentials.api_key", "organization_id", "project_id",
            "endpoint_base_url", "endpoint.api_path", "endpoint.stream_path",
            "endpoint.base_url", "endpoint.api_path", "endpoint.stream_path",
            "model", "max_tokens", "temperature", "top_p", "frequency_penalty",
            "presence_penalty",
            "retry_max_retries", "retry_base_delay", "retry_max_delay",
            "retry.backoff_multiplier", "retry.base_delay_seconds",
            "retry.max_delay_seconds",
            "timeout_request", "timeout_connect", "timeout_stream", "timeout_idle",
            "timeout.request_timeout_seconds", "timeout.connect_timeout_seconds",
            "timeout.stream_timeout_seconds", "timeout.idle_timeout_seconds",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        return cls(
            name=name,
            model=generation.model,
            credentials=creds,
            endpoint=endpoint,
            generation=generation,
            retry=retry,
            timeout=timeout,
            extra=extra,
        )


# ---------------------------------------------------------------------------
# Configuration builder (merges sources with precedence)
# ---------------------------------------------------------------------------


class ConfigBuilder:
    """Merges multiple ``ConfigSource`` instances with precedence.

    Usage::

        builder = ConfigBuilder()
        builder.add_source(DictConfigSource({"model": "claude-3"}))
        builder.add_source(EnvConfigSource())
        config = builder.build(name="claude")

    Later sources override earlier ones.
    """

    def __init__(self) -> None:
        self._sources: list[ConfigSource] = []

    def add_source(self, source: ConfigSource) -> "ConfigBuilder":
        """Add a configuration source.

        Sources added later override earlier ones when their
        ``priority`` is higher or equal.
        """
        self._sources.append(source)
        return self

    def build(self, name: str = "") -> ProviderConfig:
        """Merge all sources and build a ``ProviderConfig``.

        Args:
            name: Optional provider name.

        Returns:
            A validated ``ProviderConfig``.
        """
        # Sort by priority (ascending) then reverse-chronological
        self._sources.sort(key=lambda s: s.priority)

        merged: dict[str, Any] = {}
        for source in self._sources:
            data = source.load()
            merged.update(data)

        return ProviderConfig.from_dict(merged, name=name)

    def build_and_validate(self, name: str = "") -> tuple[ProviderConfig, ConfigValidationResult]:
        """Build a config and validate it.

        Returns:
            ``(config, validation_result)`` tuple.
        """
        config = self.build(name=name)
        # Build a flat dict for validation
        flat: dict[str, Any] = {}
        if config.credentials.has_key:
            flat["api_key"] = config.credentials.api_key
        else:
            flat["api_key"] = ""
        flat["temperature"] = config.generation.temperature
        flat["max_tokens"] = config.generation.max_tokens
        flat["retry_max_retries"] = config.retry.max_retries
        validation = validate_config(flat)
        return config, validation
