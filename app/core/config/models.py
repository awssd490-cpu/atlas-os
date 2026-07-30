"""Configuration domain models.

All models in this module are immutable frozen dataclasses, following
the convention established throughout Atlas.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping


_VALID_ENVIRONMENTS: tuple[str, ...] = (
    "development",
    "testing",
    "staging",
    "production",
)


@dataclass(frozen=True)
class AtlasConfig:
    """Centralised configuration for an Atlas application instance.

    Attributes:
        environment: Deployment environment.  Must be one of
            ``"development"``, ``"testing"``, ``"staging"``, or
            ``"production"``.  Default ``"development"``.
        debug: Enable debug mode.  Default ``False``.
        log_level: Logging verbosity level.  Default ``"INFO"``.
        random_seed: Seed for reproducible random operations.
            Default 42.
        metadata: Optional arbitrary metadata attached to the
            configuration (e.g. version, instance ID, tags).
    """

    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    random_seed: int = 42
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidConfiguration: If any value is out of range
                or invalid.
        """
        from app.core.config.errors import InvalidConfiguration

        if self.environment not in _VALID_ENVIRONMENTS:
            raise InvalidConfiguration(
                f"Invalid environment: {self.environment!r}. "
                f"Must be one of {_VALID_ENVIRONMENTS}",
                details={
                    "environment": self.environment,
                    "valid_environments": list(_VALID_ENVIRONMENTS),
                },
            )

        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise InvalidConfiguration(
                f"Invalid log_level: {self.log_level!r}. "
                "Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL",
                details={"log_level": self.log_level},
            )

        if self.random_seed < 0:
            raise InvalidConfiguration(
                f"random_seed must be non-negative, got {self.random_seed}",
                details={"random_seed": self.random_seed},
            )
