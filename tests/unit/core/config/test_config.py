"""Tests for the Atlas configuration system."""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from app.core.config import (
    AtlasConfig,
    ConfigLoader,
    ConfigurationError,
    InvalidConfiguration,
)
from app.core.config.models import AtlasConfig as AtlasConfig_Impl
from app.core.config.loader import ConfigLoader as ConfigLoader_Impl
from app.core.config.errors import ConfigurationError as ConfigurationError_Impl
from app.core.config.errors import InvalidConfiguration as InvalidConfiguration_Impl
from app.core.errors import AtlasError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_atlas_config_imported(self) -> None:
        assert AtlasConfig is AtlasConfig_Impl

    def test_config_loader_imported(self) -> None:
        assert ConfigLoader is ConfigLoader_Impl

    def test_configuration_error_imported(self) -> None:
        assert ConfigurationError is ConfigurationError_Impl

    def test_invalid_configuration_imported(self) -> None:
        assert InvalidConfiguration is InvalidConfiguration_Impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(ConfigurationError, AtlasError)
        assert issubclass(InvalidConfiguration, ConfigurationError)


# ======================================================================
# AtlasConfig
# ======================================================================


class TestAtlasConfig:
    """AtlasConfig frozen dataclass."""

    def test_default_values(self) -> None:
        cfg = AtlasConfig()
        assert cfg.environment == "development"
        assert cfg.debug is False
        assert cfg.log_level == "INFO"
        assert cfg.random_seed == 42
        assert cfg.metadata == {}

    def test_custom_values(self) -> None:
        cfg = AtlasConfig(
            environment="production",
            debug=True,
            log_level="DEBUG",
            random_seed=7,
            metadata={"version": "1.0"},
        )
        assert cfg.environment == "production"
        assert cfg.debug is True
        assert cfg.log_level == "DEBUG"
        assert cfg.random_seed == 7
        assert cfg.metadata == {"version": "1.0"}

    def test_immutable(self) -> None:
        cfg = AtlasConfig()
        with pytest.raises(AttributeError):
            cfg.environment = "production"  # type: ignore[misc]

    def test_validate_passes(self) -> None:
        for env in ("development", "testing", "staging", "production"):
            AtlasConfig(environment=env).validate()

    def test_validate_invalid_environment(self) -> None:
        with pytest.raises(InvalidConfiguration) as exc:
            AtlasConfig(environment="invalid").validate()
        assert "Invalid environment" in str(exc.value)
        assert "invalid" in str(exc.value)

    def test_validate_empty_environment(self) -> None:
        with pytest.raises(InvalidConfiguration):
            AtlasConfig(environment="").validate()

    def test_validate_invalid_log_level(self) -> None:
        with pytest.raises(InvalidConfiguration) as exc:
            AtlasConfig(log_level="TRACE").validate()
        assert "log_level" in str(exc.value).lower()

    def test_validate_negative_random_seed(self) -> None:
        with pytest.raises(InvalidConfiguration) as exc:
            AtlasConfig(random_seed=-1).validate()
        assert "non-negative" in str(exc.value).lower()

    def test_deterministic(self) -> None:
        a = AtlasConfig(environment="staging", debug=True)
        b = AtlasConfig(environment="staging", debug=True)
        assert a == b


# ======================================================================
# ConfigLoader — from_dict
# ======================================================================


class TestConfigLoaderFromDict:
    """ConfigLoader.from_dict() tests."""

    def test_defaults(self) -> None:
        cfg = ConfigLoader.from_dict({})
        assert cfg.environment == "development"
        assert cfg.debug is False

    def test_custom(self) -> None:
        cfg = ConfigLoader.from_dict({
            "environment": "staging",
            "debug": True,
            "log_level": "WARNING",
            "random_seed": 99,
        })
        assert cfg.environment == "staging"
        assert cfg.debug is True
        assert cfg.log_level == "WARNING"
        assert cfg.random_seed == 99

    def test_unknown_keys_ignored(self) -> None:
        cfg = ConfigLoader.from_dict({
            "environment": "production",
            "unknown_key": "should be ignored",
        })
        assert cfg.environment == "production"
        assert not hasattr(cfg, "unknown_key")

    def test_not_a_dict(self) -> None:
        with pytest.raises(InvalidConfiguration) as exc:
            ConfigLoader.from_dict("string")  # type: ignore[arg-type]
        assert "must be a dict" in str(exc.value).lower()

    def test_invalid_environment(self) -> None:
        with pytest.raises(InvalidConfiguration):
            ConfigLoader.from_dict({"environment": "bad"})


# ======================================================================
# ConfigLoader — from_json / to_json
# ======================================================================


class TestConfigLoaderJson:
    """ConfigLoader JSON round-trip tests."""

    def test_round_trip(self, tmp_path: Any) -> None:
        path = str(tmp_path / "config.json")
        cfg = AtlasConfig(environment="testing", debug=True, log_level="DEBUG")
        ConfigLoader.to_json(cfg, path)
        loaded = ConfigLoader.from_json(path)
        assert loaded == cfg

    def test_deterministic_output(self, tmp_path: Any) -> None:
        path = str(tmp_path / "config.json")
        cfg = AtlasConfig(environment="production")
        ConfigLoader.to_json(cfg, path)
        ConfigLoader.to_json(cfg, path)  # overwrite
        with open(path, encoding="utf-8") as f:
            first = json.load(f)
        with open(path, encoding="utf-8") as f:
            second = json.load(f)
        assert first == second

    def test_sorted_keys(self, tmp_path: Any) -> None:
        path = str(tmp_path / "config.json")
        cfg = AtlasConfig(environment="development")
        ConfigLoader.to_json(cfg, path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert list(data.keys()) == sorted(data.keys())

    def test_file_not_found(self) -> None:
        with pytest.raises(ConfigurationError) as exc:
            ConfigLoader.from_json("/nonexistent/config.json")
        assert "does not exist" in str(exc.value)

    def test_invalid_json(self, tmp_path: Any) -> None:
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json")
        with pytest.raises(ConfigurationError) as exc:
            ConfigLoader.from_json(path)
        assert "parse" in str(exc.value).lower()

    def test_root_not_object(self, tmp_path: Any) -> None:
        path = str(tmp_path / "list.json")
        with open(path, "w") as f:
            json.dump([], f)
        with pytest.raises(ConfigurationError) as exc:
            ConfigLoader.from_json(path)
        assert "object" in str(exc.value).lower()

    def test_to_json_write_error(self) -> None:
        cfg = AtlasConfig()
        with pytest.raises(ConfigurationError):
            ConfigLoader.to_json(cfg, "/nonexistent/dir/config.json")


# ======================================================================
# ConfigLoader — from_env
# ======================================================================


class TestConfigLoaderFromEnv:
    """ConfigLoader.from_env() tests."""

    @pytest.fixture(autouse=True)
    def cleanup_env(self) -> None:
        """Remove ATLAS_ env vars before and after each test."""
        for key in ("ENVIRONMENT", "DEBUG", "LOG_LEVEL", "RANDOM_SEED"):
            os.environ.pop(f"ATLAS_{key}", None)
        yield
        for key in ("ENVIRONMENT", "DEBUG", "LOG_LEVEL", "RANDOM_SEED"):
            os.environ.pop(f"ATLAS_{key}", None)

    def test_defaults_when_no_env(self) -> None:
        cfg = ConfigLoader.from_env()
        assert cfg.environment == "development"
        assert cfg.debug is False
        assert cfg.log_level == "INFO"
        assert cfg.random_seed == 42

    def test_environment(self) -> None:
        os.environ["ATLAS_ENVIRONMENT"] = "production"
        cfg = ConfigLoader.from_env()
        assert cfg.environment == "production"

    def test_debug_true(self) -> None:
        os.environ["ATLAS_DEBUG"] = "true"
        cfg = ConfigLoader.from_env()
        assert cfg.debug is True

    def test_debug_1(self) -> None:
        os.environ["ATLAS_DEBUG"] = "1"
        cfg = ConfigLoader.from_env()
        assert cfg.debug is True

    def test_debug_false(self) -> None:
        os.environ["ATLAS_DEBUG"] = "false"
        cfg = ConfigLoader.from_env()
        assert cfg.debug is False

    def test_log_level(self) -> None:
        os.environ["ATLAS_LOG_LEVEL"] = "ERROR"
        cfg = ConfigLoader.from_env()
        assert cfg.log_level == "ERROR"

    def test_random_seed(self) -> None:
        os.environ["ATLAS_RANDOM_SEED"] = "7"
        cfg = ConfigLoader.from_env()
        assert cfg.random_seed == 7

    def test_random_seed_invalid(self) -> None:
        os.environ["ATLAS_RANDOM_SEED"] = "not_a_number"
        with pytest.raises(InvalidConfiguration) as exc:
            ConfigLoader.from_env()
        assert "integer" in str(exc.value).lower()

    def test_partial_env(self) -> None:
        """Only set some env vars — rest should use defaults."""
        os.environ["ATLAS_ENVIRONMENT"] = "staging"
        os.environ["ATLAS_DEBUG"] = "true"
        cfg = ConfigLoader.from_env()
        assert cfg.environment == "staging"
        assert cfg.debug is True
        assert cfg.log_level == "INFO"  # default
        assert cfg.random_seed == 42    # default

    def test_custom_prefix(self) -> None:
        os.environ["MYAPP_ENVIRONMENT"] = "production"
        os.environ["MYAPP_DEBUG"] = "1"
        cfg = ConfigLoader.from_env(prefix="MYAPP_")
        assert cfg.environment == "production"
        assert cfg.debug is True
        assert cfg.log_level == "INFO"

    def test_invalid_env_environment(self) -> None:
        os.environ["ATLAS_ENVIRONMENT"] = "invalid"
        with pytest.raises(InvalidConfiguration):
            ConfigLoader.from_env()

    def test_invalid_env_log_level(self) -> None:
        os.environ["ATLAS_LOG_LEVEL"] = "TRACE"
        with pytest.raises(InvalidConfiguration):
            ConfigLoader.from_env()

    def test_negative_random_seed(self) -> None:
        os.environ["ATLAS_RANDOM_SEED"] = "-1"
        with pytest.raises(InvalidConfiguration):
            ConfigLoader.from_env()


# ======================================================================
# ConfigLoader — to_dict
# ======================================================================


class TestConfigLoaderToDict:
    """ConfigLoader.to_dict() tests."""

    def test_round_trip(self) -> None:
        cfg = AtlasConfig(environment="testing", debug=True)
        data = ConfigLoader.to_dict(cfg)
        assert data["environment"] == "testing"
        assert data["debug"] is True
        assert "log_level" in data
        assert "random_seed" in data

    def test_deterministic(self) -> None:
        cfg = AtlasConfig(environment="production", random_seed=1)
        a = ConfigLoader.to_dict(cfg)
        b = ConfigLoader.to_dict(cfg)
        assert a == b
