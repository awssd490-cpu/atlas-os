"""Release engineering subsystem for Atlas.

Provides version resolution and bumping, changelog construction, and
distribution artifact validation for planning and publishing releases.
It is used by the CLI (``tools/atlas_cli.py``) and by CI release
workflows.

Architecture::

    ReleaseService
        ├── versioning   (Version, resolve, bump)
        ├── changelog    (Changelog, ChangelogEntry, render)
        ├── artifacts    (ReleaseArtifact, discover, validate)
        └── config       (ReleaseConfig)
"""

from __future__ import annotations

from app.release.artifacts import ReleaseArtifact, classify, discover, validate
from app.release.changelog import Changelog, ChangelogEntry, make_entry, parse_changelog, render_entry
from app.release.config import ReleaseConfig
from app.release.errors import (
    ArtifactError,
    ChangelogError,
    InvalidReleaseConfiguration,
    ReleaseError,
    VersionError,
    VersionNotFound,
)
from app.release.models import ReleaseInfo, Version
from app.release.service import ReleaseService
from app.release.versioning import bump, resolve_version

__all__ = [
    "ArtifactError",
    "Changelog",
    "ChangelogEntry",
    "ChangelogError",
    "InvalidReleaseConfiguration",
    "ReleaseArtifact",
    "ReleaseConfig",
    "ReleaseError",
    "ReleaseInfo",
    "ReleaseService",
    "Version",
    "VersionError",
    "VersionNotFound",
    "bump",
    "classify",
    "discover",
    "make_entry",
    "parse_changelog",
    "render_entry",
    "resolve_version",
    "validate",
]
