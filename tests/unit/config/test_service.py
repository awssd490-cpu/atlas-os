"""Tests for PydanticConfigService.

Verifies:
- Dotted-key access resolves nested values
- ``get()`` raises ``ConfigurationError`` on missing keys with no default
- ``get()`` returns the default when a key is missing
- ``get_section()`` returns correct prefix scope
- ``dump(mask_secrets=True)`` masks ``SecretStr`` values
- ``dump(mask_secrets=False)`` reveals them
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from app.config.service import PydanticConfigService
from app.config.settings import AtlasSettings
from app.core.errors import ConfigurationError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> PydanticConfigService:
    """Config service with overridden values to exercise all paths."""
    settings = AtlasSettings(
        app={"name": "test-atlas", "environment": "testing"},
        database={"host": "pg.local", "password": "hunter2"},
    )
    return PydanticConfigService(settings=settings)


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------


class TestGet:
    def test_top_level_key(self, config: PydanticConfigService) -> None:
        result = config.get("app")
        assert isinstance(result, dict)
        assert result["name"] == "test-atlas"

    def test_nested_key(self, config: PydanticConfigService) -> None:
        assert config.get("app.name") == "test-atlas"
        assert config.get("database.host") == "pg.local"

    def test_deeply_nested(self, config: PydanticConfigService) -> None:
        assert config.get("server.host") == "0.0.0.0"

    def test_missing_key_raises(self, config: PydanticConfigService) -> None:
        with pytest.raises(ConfigurationError) as exc:
            config.get("nonexistent.setting")
        assert "not found" in str(exc.value)

    def test_missing_key_with_default(self, config: PydanticConfigService) -> None:
        assert config.get("nonexistent", default=42) == 42

    def test_missing_key_default_none(self, config: PydanticConfigService) -> None:
        assert config.get("nonexistent", default=None) is None


# ---------------------------------------------------------------------------
# get_section()
# ---------------------------------------------------------------------------


class TestGetSection:
    def test_section_returns_dict(self, config: PydanticConfigService) -> None:
        section = config.get_section("app")
        assert section["name"] == "test-atlas"
        assert section["environment"] == "testing"

    def test_section_missing_returns_empty(self, config: PydanticConfigService) -> None:
        assert config.get_section("doesnotexist") == {}


# ---------------------------------------------------------------------------
# dump()
# ---------------------------------------------------------------------------


class TestDump:
    def test_dump_masks_secrets(self, config: PydanticConfigService) -> None:
        dumped = config.dump(mask_secrets=True)
        assert dumped["database"]["password"] == "*****"

    def test_dump_reveals_secrets(self, config: PydanticConfigService) -> None:
        dumped = config.dump(mask_secrets=False)
        assert dumped["database"]["password"] == "hunter2"

    def test_dump_includes_all_sections(self, config: PydanticConfigService) -> None:
        dumped = config.dump(mask_secrets=True)
        for section in ("app", "server", "logging", "database"):
            assert section in dumped, f"Missing section {section}"
