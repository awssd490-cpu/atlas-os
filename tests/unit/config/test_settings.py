"""Tests for configuration models.

Verifies:
- Default values are correct
- Environment variable overrides work (via ``ATLAS_`` prefix)
- Validation constraints are enforced
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from app.config.settings import AtlasSettings


class TestAtlasSettingsDefaults:
    """Settings with no overrides should produce sensible defaults."""

    def test_app_defaults(self) -> None:
        settings = AtlasSettings()
        assert settings.app.name == "atlas"
        assert settings.app.version == "0.1.0"
        assert settings.app.environment == "development"
        assert settings.app.debug is True

    def test_server_defaults(self) -> None:
        settings = AtlasSettings()
        assert settings.server.host == "0.0.0.0"
        assert settings.server.port == 8000
        assert settings.server.reload is True
        assert settings.server.workers == 1
        assert settings.server.timeout_keepalive == 30

    def test_logging_defaults(self) -> None:
        settings = AtlasSettings()
        assert settings.logging.level == "DEBUG"
        assert settings.logging.format == "colored"
        assert settings.logging.sinks == ["console"]
        assert settings.logging.file_path is None
        assert settings.logging.queue_size == 65536
        assert settings.logging.serialize is False

    def test_database_defaults(self) -> None:
        settings = AtlasSettings()
        assert settings.database.host == "localhost"
        assert settings.database.port == 5432
        assert settings.database.database == "atlas"
        assert settings.database.username == "postgres"
        assert isinstance(settings.database.password, SecretStr)
        assert settings.database.password.get_secret_value() == ""
        assert settings.database.pool_min == 2
        assert settings.database.pool_max == 16
        assert settings.database.echo is False

    def test_storage_defaults(self) -> None:
        settings = AtlasSettings()
        assert settings.storage.sqlite_path == "data/atlas.db"
        assert settings.storage.cache_ttl_default == 300
        assert settings.storage.cache_max_size == 10_000
        assert settings.storage.vector_dimension_default == 1536
        assert settings.storage.vector_namespaces == ["default"]
        assert settings.storage.object_store_path == "data/objects"
        assert settings.storage.event_store_retention_days == 90


class TestAtlasSettingsEnvironOverrides:
    """Environment variables with ``ATLAS_`` prefix override defaults."""

    def test_app_name_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_APP__NAME", "custom-atlas")
        settings = AtlasSettings()
        assert settings.app.name == "custom-atlas"

    def test_environment_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_APP__ENVIRONMENT", "production")
        settings = AtlasSettings()
        assert settings.app.environment == "production"
        assert settings.app.debug is True  # still default

    def test_nested_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_DATABASE__HOST", "db.example.com")
        monkeypatch.setenv("ATLAS_DATABASE__PORT", "15432")
        settings = AtlasSettings()
        assert settings.database.host == "db.example.com"
        assert settings.database.port == 15432

    def test_port_validation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ATLAS_SERVER__PORT", "99999")
        with pytest.raises(ValidationError):
            AtlasSettings()

    def test_unknown_env_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Extra='ignore' means unknown env vars don't crash."""
        monkeypatch.setenv("ATLAS_APP__UNKNOWN_FIELD", "something")
        settings = AtlasSettings()
        assert settings.app.name == "atlas"


class TestAtlasSettingsSecretStr:
    """SecretStr fields should protect sensitive values."""

    def test_password_not_in_repr(self) -> None:
        settings = AtlasSettings(database={"password": "s3cret!"})
        rep = repr(settings.database.password)
        assert "s3cret!" not in rep
        assert settings.database.password.get_secret_value() == "s3cret!"
