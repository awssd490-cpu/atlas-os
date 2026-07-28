"""Configuration service — wraps Pydantic Settings with runtime accessors.

**Responsibility:** load once at boot, validate, then provide typed,
dotted-key read access for the rest of the process lifetime.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, SecretStr

from app.config.settings import AtlasSettings
from app.core.errors import ConfigurationError
from app.core.interfaces import ConfigService


class _Missing:
    """Sentinel type so we can detect ``_MISSING`` with ``isinstance``."""

    _instance: _Missing | None = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover
        return "<MISSING>"


_MISSING = _Missing()


class PydanticConfigService(ConfigService):
    """Configuration service backed by Pydantic :class:`BaseSettings`.

    The settings object is built once (on construction) and served
    read-only thereafter.  Mutating config at runtime is intentionally
    not supported — it should be handled by restarting the process.
    """

    def __init__(self, settings: AtlasSettings | None = None) -> None:
        self._settings = settings or AtlasSettings()

    @property
    def settings(self) -> AtlasSettings:
        """Expose the raw settings object for callers that need typed access."""
        return self._settings

    # ------------------------------------------------------------------
    # ConfigService interface
    # ------------------------------------------------------------------

    def get(self, key: str, default: Any = _MISSING) -> Any:
        """Return the value at a dotted *key* (e.g. ``"database.host"``).

        Raises :class:`ConfigurationError` when the key is not found and
        no *default* is provided.
        """
        parts = key.split(".", 1)
        top = parts[0]

        if top not in type(self._settings).model_fields:
            if not isinstance(default, _Missing):
                return default
            raise ConfigurationError(
                f"Configuration key '{key}' not found",
                details={"key": key, "available": list(type(self._settings).model_fields.keys())},
            )

        section = self.get_section(top)
        if len(parts) == 1:
            return section

        remaining = parts[1]
        current: Any = section
        for part in remaining.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            elif isinstance(current, BaseModel):
                current = getattr(current, part)
            elif isinstance(current, SecretStr):
                # Convert to string for dotted traversal
                current = current.get_secret_value()
            else:
                if not isinstance(default, _Missing):
                    return default
                raise ConfigurationError(
                    f"Configuration key '{key}' not found",
                    details={
                        "key": key,
                        "found_up_to": str(current)[:100],
                    },
                )
        return current

    def get_section(self, prefix: str) -> dict[str, Any]:
        """Return config values whose root key is *prefix*.

        Returns an empty dict when *prefix* does not exist.
        """
        value = getattr(self._settings, prefix, _MISSING)
        if isinstance(value, _Missing):
            return {}
        if isinstance(value, BaseModel):
            return self._model_to_dict(value)
        if isinstance(value, dict):
            return value
        return {prefix: value}

    def dump(self, *, mask_secrets: bool = True) -> dict[str, Any]:
        """Return the entire config as a plain dict.

        Args:
            mask_secrets: When ``True`` (default), ``SecretStr`` values
                are replaced with ``"*****"``.  When ``False``, the raw
                secret value string is included.
        """
        raw = self._model_to_dict(self._settings)
        return self._mask_secrets(raw) if mask_secrets else self._reveal_secrets(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_to_dict(self, model: BaseModel) -> dict[str, Any]:
        """Recursively convert a Pydantic model to a plain dict.

        ``SecretStr`` values are kept as ``SecretStr`` objects so the
        caller can decide to mask or reveal.
        """
        result: dict[str, Any] = {}
        for field_name in type(model).model_fields.keys():
            value = getattr(model, field_name)
            if isinstance(value, BaseModel):
                result[field_name] = self._model_to_dict(value)
            elif isinstance(value, SecretStr):
                result[field_name] = value  # keep as SecretStr
            else:
                result[field_name] = value
        return result

    def _mask_secrets(self, d: dict[str, Any]) -> dict[str, Any]:
        """Replace ``SecretStr`` values with ``"*****"``."""
        out: dict[str, Any] = {}
        for key, value in d.items():
            if isinstance(value, SecretStr):
                out[key] = "*****"
            elif isinstance(value, dict):
                out[key] = self._mask_secrets(value)
            elif isinstance(value, list):
                out[key] = [
                    self._mask_secrets(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                out[key] = value
        return out

    def _reveal_secrets(self, d: dict[str, Any]) -> dict[str, Any]:
        """Convert ``SecretStr`` values to their raw string."""
        out: dict[str, Any] = {}
        for key, value in d.items():
            if isinstance(value, SecretStr):
                out[key] = value.get_secret_value()
            elif isinstance(value, dict):
                out[key] = self._reveal_secrets(value)
            elif isinstance(value, list):
                out[key] = [
                    self._reveal_secrets(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                out[key] = value
        return out

    def _root_keys(self) -> set[str]:
        """Return the top-level setting keys."""
        return set(type(self._settings).model_fields.keys())
