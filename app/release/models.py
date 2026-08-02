"""Release engineering data models.

All models are immutable frozen dataclasses, following the convention
established in ``app.rag.models`` and ``app.rag.persistence.config``.
They describe the version, changelog, and artifacts that make up a
release — with no I/O or business logic beyond serialization helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Semantic version: major.minor.patch, optionally with a pre-release
# suffix such as ``-alpha``, ``-beta``, ``-rc.1`` (matches the project's
# tag history: ``v1.0.0``, ``v1.1.0-alpha``, ``v0.7.5-alpha``).
_VERSION_PATTERN = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<prerelease>[0-9A-Za-z.-]+))?$"
)


@dataclass(frozen=True)
class Version:
    """A parsed semantic version.

    Attributes:
        major: Major version component.
        minor: Minor version component.
        patch: Patch version component.
        prerelease: Optional pre-release identifier (``"alpha"``,
            ``"rc.1"``, ...).  Empty string when not a pre-release.
    """

    major: int
    minor: int
    patch: int
    prerelease: str = ""

    @classmethod
    def parse(cls, text: str) -> "Version":
        """Parse a semantic version string.

        Accepts ``X.Y.Z`` and ``X.Y.Z-pre`` forms.  The leading ``v``
        prefix used by git tags is stripped before parsing.

        Args:
            text: The version string to parse.

        Returns:
            The parsed :class:`Version`.

        Raises:
            VersionError: If the string is not a valid semantic version.
        """
        from app.release.errors import VersionError

        stripped = text[1:] if text.startswith("v") else text
        match = _VERSION_PATTERN.match(stripped)
        if match is None:
            raise VersionError(
                f"Invalid version string {text!r}; "
                f"expected X.Y.Z with optional pre-release suffix"
            )
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=match.group("prerelease") or "",
        )

    def __str__(self) -> str:
        """Render the version without a ``v`` prefix."""
        if self.prerelease:
            return f"{self.major}.{self.minor}.{self.patch}-{self.prerelease}"
        return f"{self.major}.{self.minor}.{self.patch}"

    @property
    def tag(self) -> str:
        """Render the version as a git tag (with ``v`` prefix)."""
        return f"v{self}"

    @property
    def is_prerelease(self) -> bool:
        """Whether this is a pre-release (e.g. ``-alpha``)."""
        return bool(self.prerelease)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the version to a plain dict."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "version": str(self),
        }


@dataclass(frozen=True)
class ChangelogEntry:
    """A single changelog entry for one release.

    Attributes:
        version: The version the entry describes.
        title: Optional human-readable title (e.g. ``"Hybrid Retrieval"``).
        date: Release date as an ISO string (``YYYY-MM-DD``), or empty.
        sections: Ordered mapping of section name to list of bullet items.
    """

    version: str
    title: str = ""
    date: str = ""
    sections: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Changelog:
    """An ordered collection of changelog entries (newest first)."""

    entries: tuple[ChangelogEntry, ...] = ()

    def __bool__(self) -> bool:
        """A changelog is truthy when it has at least one entry."""
        return bool(self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the changelog to a plain dict."""
        return {
            "entries": [
                {
                    "version": entry.version,
                    "title": entry.title,
                    "date": entry.date,
                    "sections": {
                        name: list(items) for name, items in entry.sections.items()
                    },
                }
                for entry in self.entries
            ]
        }


@dataclass(frozen=True)
class ReleaseArtifact:
    """A built distribution file produced for a release.

    Attributes:
        path: Absolute or relative path to the artifact file.
        name: Base file name (e.g. ``tekvora_atlas-1.0.0-py3-none-any.whl``).
        kind: Artifact kind — ``"wheel"`` or ``"sdist"``.
        size_bytes: File size in bytes.
        sha256: SHA-256 hex digest of the file contents.
    """

    path: str
    name: str
    kind: str
    size_bytes: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact to a plain dict."""
        return {
            "path": self.path,
            "name": self.name,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ReleaseInfo:
    """A complete, validated release plan.

    Attributes:
        version: The version being released.
        changelog: The changelog entry describing this release.
        artifacts: Built distribution artifacts, if discovered.
    """

    version: Version
    changelog: ChangelogEntry | None = None
    artifacts: tuple[ReleaseArtifact, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the release info to a plain dict."""
        return {
            "version": self.version.to_dict(),
            "tag": self.version.tag,
            "changelog": None
            if self.changelog is None
            else {
                "version": self.changelog.version,
                "title": self.changelog.title,
                "date": self.changelog.date,
                "sections": {
                    name: list(items)
                    for name, items in self.changelog.sections.items()
                },
            },
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }
