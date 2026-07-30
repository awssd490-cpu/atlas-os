"""ConfigLoader — load and save ``AtlasConfig`` from multiple sources.

Supports Python dicts, JSON files, and environment variables.
All output is deterministic and UTF-8.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.core.config.errors import ConfigurationError, InvalidConfiguration
from app.core.config.models import AtlasConfig


class ConfigLoader:
    """Loads and saves ``AtlasConfig`` instances from/to various sources.

    Usage::

        loader = ConfigLoader()

        # From a Python dict
        cfg = loader.from_dict({"environment": "production"})

        # From a JSON file
        cfg = loader.from_json("path/to/config.json")

        # From environment variables
        cfg = loader.from_env(prefix="ATLAS_")

        # Save to JSON
        loader.to_json(cfg, "path/to/config.json")
    """

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(data: dict[str, Any]) -> AtlasConfig:
        """Build an ``AtlasConfig`` from a Python dict.

        Only recognised keys are forwarded to the config constructor.

        Args:
            data: A dict with configuration keys.

        Returns:
            A new validated ``AtlasConfig``.

        Raises:
            InvalidConfiguration: If any value is invalid.
        """
        if not isinstance(data, dict):
            raise InvalidConfiguration(
                "Config data must be a dict",
                details={"received_type": type(data).__name__},
            )

        known_keys = {"environment", "debug", "log_level", "random_seed", "metadata"}
        kwargs: dict[str, Any] = {}

        for key in known_keys:
            if key in data:
                kwargs[key] = data[key]

        cfg = AtlasConfig(**kwargs)
        cfg.validate()
        return cfg

    @staticmethod
    def from_json(path: str) -> AtlasConfig:
        """Load configuration from a JSON file.

        Args:
            path: Path to a JSON configuration file.

        Returns:
            A new validated ``AtlasConfig``.

        Raises:
            ConfigurationError: If the file cannot be read or parsed.
            InvalidConfiguration: If the configuration values are invalid.
        """
        target = Path(path)
        if not target.exists():
            raise ConfigurationError(
                f"Configuration file does not exist: {path}",
                details={"path": path},
            )

        try:
            raw = target.read_bytes()
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            raise ConfigurationError(
                f"Failed to parse configuration file: {exc}",
                details={"path": path},
            ) from exc

        if not isinstance(data, dict):
            raise ConfigurationError(
                "Configuration file must contain a JSON object",
                details={"path": path, "received_type": type(data).__name__},
            )

        return ConfigLoader.from_dict(data)

    @staticmethod
    def from_env(prefix: str = "ATLAS_") -> AtlasConfig:
        """Load configuration from environment variables.

        Mappings::

            {prefix}ENVIRONMENT  → environment
            {prefix}DEBUG        → debug
            {prefix}LOG_LEVEL    → log_level
            {prefix}RANDOM_SEED  → random_seed

        Only set variables are included — missing variables fall back
        to ``AtlasConfig`` defaults.

        Args:
            prefix: Environment variable prefix (default ``"ATLAS_"``).

        Returns:
            A new validated ``AtlasConfig``.
        """
        kwargs: dict[str, Any] = {}

        env_map: dict[str, str] = {
            f"{prefix}ENVIRONMENT": "environment",
            f"{prefix}DEBUG": "debug",
            f"{prefix}LOG_LEVEL": "log_level",
            f"{prefix}RANDOM_SEED": "random_seed",
        }

        for env_key, config_key in env_map.items():
            value = os.environ.get(env_key)
            if value is None:
                continue

            if config_key == "debug":
                kwargs[config_key] = value.lower() in ("1", "true", "yes")
            elif config_key == "random_seed":
                try:
                    kwargs[config_key] = int(value)
                except ValueError:
                    raise InvalidConfiguration(
                        f"Environment variable {env_key} must be an integer, "
                        f"got {value!r}",
                        details={"env_var": env_key, "value": value},
                    ) from None
            else:
                kwargs[config_key] = value

        cfg = AtlasConfig(**kwargs)
        cfg.validate()
        return cfg

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def to_dict(config: AtlasConfig) -> dict[str, Any]:
        """Serialise an ``AtlasConfig`` to a plain Python dict.

        The output is deterministic and ready for ``json.dumps``.

        Args:
            config: The configuration to serialise.

        Returns:
            A dict with configuration keys.
        """
        return {
            "environment": config.environment,
            "debug": config.debug,
            "log_level": config.log_level,
            "random_seed": config.random_seed,
            "metadata": dict(config.metadata),
        }

    @staticmethod
    def to_json(config: AtlasConfig, path: str) -> None:
        """Serialise an ``AtlasConfig`` to a JSON file.

        Output is UTF-8 with pretty-printing and sorted keys for
        deterministic output.

        Args:
            config: The configuration to serialise.
            path: Target file path.

        Raises:
            ConfigurationError: On write failures.
        """
        data = ConfigLoader.to_dict(config)

        try:
            json_bytes = json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Failed to serialise configuration: {exc}",
                details={"path": path},
            ) from exc

        try:
            Path(path).write_bytes(json_bytes)
        except OSError as exc:
            raise ConfigurationError(
                f"Failed to write configuration file: {exc}",
                details={"path": path},
            ) from exc
