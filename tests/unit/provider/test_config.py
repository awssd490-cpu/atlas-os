"""Tests for the Provider Configuration System."""

from __future__ import annotations

import os
from typing import Any

import pytest

from app.provider.config import (
    ConfigBuilder,
    ConfigSource,
    ConfigValidationResult,
    DictConfigSource,
    EnvConfigSource,
    GenerationDefaults,
    ProviderConfig,
    ProviderCredentials,
    ProviderEndpoint,
    RetryPolicy,
    TimeoutPolicy,
    validate_config,
)


# ---------------------------------------------------------------------------
# ProviderCredentials
# ---------------------------------------------------------------------------


class TestProviderCredentials:
    def test_empty(self) -> None:
        c = ProviderCredentials.empty()
        assert c.has_key is False

    def test_masks_in_repr(self) -> None:
        c = ProviderCredentials(api_key="sk-abcdefgh1234")
        assert "sk-" not in repr(c)
        assert "***" in repr(c)

    def test_masks_in_str(self) -> None:
        c = ProviderCredentials(api_key="secret-key-1234")
        assert "secret-key" not in str(c)
        assert "****" in str(c)

    def test_to_dict_masks(self) -> None:
        c = ProviderCredentials(api_key="sk-test-key-5678")
        d = c.to_dict()
        assert d["api_key"] == "***5678"
        assert "sk-test" not in d["api_key"]

    def test_has_key(self) -> None:
        c = ProviderCredentials(api_key="key")
        assert c.has_key is True

    def test_immutable(self) -> None:
        c = ProviderCredentials(api_key="k")
        with pytest.raises(AttributeError):
            c.api_key = "other"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ProviderCredentials(api_key="same")
        b = ProviderCredentials(api_key="same")
        assert a == b

    def test_inequality(self) -> None:
        a = ProviderCredentials(api_key="a")
        b = ProviderCredentials(api_key="b")
        assert a != b

    def test_short_key_suffix(self) -> None:
        c = ProviderCredentials(api_key="ab")
        d = c.to_dict()
        assert d["api_key"] == "***"


# ---------------------------------------------------------------------------
# RetryPolicy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    def test_defaults(self) -> None:
        p = RetryPolicy()
        assert p.max_retries == 3
        assert p.base_delay_seconds == 1.0

    def test_is_retryable(self) -> None:
        p = RetryPolicy()
        assert p.is_retryable("PROVIDER_RATE_LIMIT_ERROR") is True
        assert p.is_retryable("PROVIDER_AUTHENTICATION_ERROR") is False

    def test_custom(self) -> None:
        p = RetryPolicy(max_retries=5, base_delay_seconds=2.0)
        assert p.max_retries == 5


# ---------------------------------------------------------------------------
# TimeoutPolicy
# ---------------------------------------------------------------------------


class TestTimeoutPolicy:
    def test_defaults(self) -> None:
        t = TimeoutPolicy()
        assert t.request_timeout_seconds == 60.0

    def test_custom(self) -> None:
        t = TimeoutPolicy(request_timeout_seconds=30.0, stream_timeout_seconds=300.0)
        assert t.request_timeout_seconds == 30.0


# ---------------------------------------------------------------------------
# GenerationDefaults
# ---------------------------------------------------------------------------


class TestGenerationDefaults:
    def test_defaults(self) -> None:
        g = GenerationDefaults()
        assert g.max_tokens == 4096
        assert g.temperature == 0.7

    def test_custom(self) -> None:
        g = GenerationDefaults(model="claude-3", temperature=0.0, max_tokens=100)
        assert g.model == "claude-3"
        assert g.temperature == 0.0


# ---------------------------------------------------------------------------
# ProviderEndpoint
# ---------------------------------------------------------------------------


class TestProviderEndpoint:
    def test_defaults(self) -> None:
        e = ProviderEndpoint()
        assert e.is_configured() is False

    def test_request_url(self) -> None:
        e = ProviderEndpoint(base_url="https://api.example.com")
        assert e.request_url == "https://api.example.com/v1/messages"

    def test_stream_url(self) -> None:
        e = ProviderEndpoint(base_url="https://api.example.com")
        assert e.stream_url == "https://api.example.com/v1/messages/stream"

    def test_trailing_slash(self) -> None:
        e = ProviderEndpoint(base_url="https://api.example.com/")
        assert e.request_url == "https://api.example.com/v1/messages"


# ---------------------------------------------------------------------------
# ConfigSource
# ---------------------------------------------------------------------------


class TestDictConfigSource:
    def test_load(self) -> None:
        source = DictConfigSource({"api_key": "test", "model": "claude"})
        data = source.load()
        assert data["api_key"] == "test"
        assert data["model"] == "claude"

    def test_priority(self) -> None:
        source = DictConfigSource({}, priority=10)
        assert source.priority == 10


class TestEnvConfigSource:
    def test_load_with_prefix(self) -> None:
        os.environ["ATLAS_PROVIDER_API_KEY"] = "env-key"
        os.environ["ATLAS_PROVIDER_MODEL"] = "gpt-4"
        source = EnvConfigSource(prefix="ATLAS_PROVIDER_")
        data = source.load()
        assert data["api_key"] == "env-key"
        assert data["model"] == "gpt-4"
        del os.environ["ATLAS_PROVIDER_API_KEY"]
        del os.environ["ATLAS_PROVIDER_MODEL"]

    def test_empty_prefix_not_found(self) -> None:
        source = EnvConfigSource(prefix="ATLAS_NONEXISTENT_")
        data = source.load()
        assert data == {}


# ---------------------------------------------------------------------------
# ConfigBuilder
# ---------------------------------------------------------------------------


class TestConfigBuilder:
    def test_build_empty(self) -> None:
        builder = ConfigBuilder()
        config = builder.build(name="test")
        assert config.name == "test"
        assert config.credentials.has_key is False

    def test_build_with_source(self) -> None:
        builder = ConfigBuilder()
        builder.add_source(DictConfigSource({"api_key": "test-key", "model": "claude-3"}))
        config = builder.build(name="test")
        assert config.credentials.has_key is True
        assert config.model == "claude-3"

    def test_source_precedence(self) -> None:
        """Later sources with higher or equal priority override earlier."""
        builder = ConfigBuilder()
        builder.add_source(DictConfigSource({"api_key": "low", "model": "old"}, priority=5))
        builder.add_source(DictConfigSource({"api_key": "high"}, priority=10))
        config = builder.build()
        assert config.credentials.api_key == "high"
        # model should still be "old" since second source didn't set it
        # Actually DictConfigSource.load() returns ALL keys, so all get merged
        assert config.generation.model == "old"

    def test_build_and_validate(self) -> None:
        builder = ConfigBuilder()
        builder.add_source(DictConfigSource({"api_key": "valid-key"}))
        config, validation = builder.build_and_validate(name="test")
        assert config.credentials.has_key is True
        assert validation.valid is True


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------


class TestProviderConfig:
    def test_empty(self) -> None:
        pc = ProviderConfig()
        assert pc.name == ""

    def test_from_dict_flat(self) -> None:
        pc = ProviderConfig.from_dict({
            "api_key": "sk-test",
            "model": "gpt-4",
            "temperature": 0.3,
        }, name="gpt")
        assert pc.name == "gpt"
        assert pc.credentials.api_key == "sk-test"
        assert pc.generation.temperature == 0.3
        assert pc.generation.model == "gpt-4"

    def test_from_dict_nested(self) -> None:
        pc = ProviderConfig.from_dict({
            "credentials.api_key": "sk-nested",
            "generation.max_tokens": 2048,
        })
        assert pc.credentials.api_key == "sk-nested"
        assert pc.generation.max_tokens == 2048

    def test_to_dict_masks_key(self) -> None:
        pc = ProviderConfig(credentials=ProviderCredentials(api_key="sk-visible"))
        d = pc.to_dict()
        assert "sk-visible" not in d["credentials"]["api_key"]
        assert "***" in d["credentials"]["api_key"]

    def test_from_dict_extra(self) -> None:
        pc = ProviderConfig.from_dict({
            "api_key": "k",
            "custom_option": "value",
            "another_setting": 42,
        })
        assert pc.extra["custom_option"] == "value"
        assert pc.extra["another_setting"] == 42

    def test_immutable(self) -> None:
        pc = ProviderConfig()
        with pytest.raises(AttributeError):
            pc.name = "other"  # type: ignore[misc]

    def test_from_dict_empty(self) -> None:
        pc = ProviderConfig.from_dict({})
        assert pc.credentials.has_key is False


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_valid(self) -> None:
        result = validate_config({"api_key": "sk-test"})
        assert result.valid is True

    def test_missing_api_key(self) -> None:
        result = validate_config({})
        assert result.valid is False
        assert any("api_key" in e for e in result.errors)

    def test_temperature_out_of_range(self) -> None:
        result = validate_config({"api_key": "k", "temperature": 3.0})
        assert result.valid is False
        assert any("temperature" in e for e in result.errors)

    def test_temperature_string(self) -> None:
        result = validate_config({"api_key": "k", "temperature": "hot"})
        assert result.valid is False

    def test_max_tokens_invalid(self) -> None:
        result = validate_config({"api_key": "k", "max_tokens": -1})
        assert result.valid is False

    def test_retries_invalid(self) -> None:
        result = validate_config({"api_key": "k", "retry_max_retries": 100})
        assert result.valid is False

    def test_valid_full_config(self) -> None:
        result = validate_config({
            "api_key": "sk-test",
            "temperature": 0.5,
            "max_tokens": 2000,
            "retry_max_retries": 3,
        })
        assert result.valid is True


class TestConfigValidationResult:
    def test_ok(self) -> None:
        r = ConfigValidationResult.ok()
        assert r.valid is True
        assert r.errors == []

    def test_failed(self) -> None:
        r = ConfigValidationResult.failed("error 1", "error 2")
        assert r.valid is False
        assert len(r.errors) == 2

    def test_add_error(self) -> None:
        r = ConfigValidationResult.ok().add_error("something broke")
        assert r.valid is False
        assert "something broke" in r.errors


# ---------------------------------------------------------------------------
# Full integration
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_config_builder_to_provider(self) -> None:
        """End-to-end: sources → builder → ProviderConfig."""
        builder = ConfigBuilder()
        builder.add_source(DictConfigSource({
            "api_key": "sk-final",
            "model": "claude-opus-4",
            "temperature": 0.1,
            "max_tokens": 8192,
            "endpoint_base_url": "https://api.anthropic.com",
        }))
        config = builder.build(name="claude")
        assert config.name == "claude"
        assert config.credentials.api_key == "sk-final"
        assert config.generation.model == "claude-opus-4"
        assert config.generation.temperature == 0.1
        assert config.generation.max_tokens == 8192
        assert config.endpoint.base_url == "https://api.anthropic.com"

    def test_credential_never_exposed(self) -> None:
        """Credentials must never appear unmasked in logs or dicts."""
        config = ProviderConfig(
            credentials=ProviderCredentials(api_key="super-secret-value"),
        )
        d = config.to_dict()
        assert "super-secret-value" not in str(d)
        assert "super-secret-value" not in repr(config)
        assert "super-secret-value" not in str(config.credentials)
