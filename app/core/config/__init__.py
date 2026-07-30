"""Configuration — centralised configuration management for Atlas.

Provides ``AtlasConfig``, an immutable frozen dataclass holding all
top-level application settings, and ``ConfigLoader`` for loading
and saving from Python dicts, JSON files, and environment variables.
"""

from __future__ import annotations

from app.core.config.errors import ConfigurationError, InvalidConfiguration
from app.core.config.loader import ConfigLoader
from app.core.config.models import AtlasConfig

__all__ = [
    "AtlasConfig",
    "ConfigLoader",
    "ConfigurationError",
    "InvalidConfiguration",
]
