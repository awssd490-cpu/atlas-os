"""ReleaseService — orchestration for the release engineering subsystem.

A :class:`ReleaseService` ties together version resolution, changelog
construction, and artifact validation into a small set of operations used
by the CLI and CI:

* resolve the current version,
* plan the next version for a bump level,
* build a changelog entry for the version being released,
* validate the built distribution artifacts.

It is intentionally I/O-light and dependency-free beyond the other
``app.release`` modules so it stays trivially testable.
"""

from __future__ import annotations

from typing import Literal

from app.release.artifacts import discover, validate
from app.release.changelog import make_entry
from app.release.config import ReleaseConfig
from app.release.models import ChangelogEntry, ReleaseArtifact, ReleaseInfo, Version
from app.release.versioning import bump, resolve_version

# Bump levels exposed through the service.
BumpLevel = Literal["major", "minor", "patch", "prerelease"]


class ReleaseService:
    """High-level operations for planning and validating a release.

    Args:
        config: Optional :class:`ReleaseConfig`.  Defaults to
            ``ReleaseConfig()``.
    """

    def __init__(self, config: ReleaseConfig | None = None) -> None:
        self._config = config or ReleaseConfig()

    @property
    def config(self) -> ReleaseConfig:
        """The configuration backing this service."""
        return self._config

    def current_version(self) -> Version:
        """Resolve the current project version.

        Returns:
            The current :class:`Version`.
        """
        return resolve_version(self._config)

    def next_version(self, level: BumpLevel = "patch") -> Version:
        """Resolve the next version for a bump level.

        Args:
            level: The bump level (``major``, ``minor``, ``patch`` or
                ``prerelease``).

        Returns:
            The proposed next :class:`Version`.
        """
        return bump(self.current_version(), level)

    def changelog_entry(
        self,
        version: Version | None = None,
        title: str = "",
        date: str = "",
        sections: dict[str, list[str]] | None = None,
    ) -> ChangelogEntry:
        """Build the changelog entry for a release.

        Args:
            version: The version to describe.  Defaults to the current
                project version.
            title: Optional release title.
            date: Optional ISO release date.
            sections: Optional mapping of section name to bullet items.

        Returns:
            A :class:`ChangelogEntry`.
        """
        target = version or self.current_version()
        return make_entry(str(target), title=title, date=date, sections=sections)

    def artifacts(self, expected_version: str | None = None) -> tuple[ReleaseArtifact, ...]:
        """Discover the built distribution artifacts.

        Args:
            expected_version: Optional version to filter artifacts by.

        Returns:
            A tuple of :class:`ReleaseArtifact`.
        """
        return discover(self._config.dist_dir, expected_version=expected_version)

    def validate_artifacts(
        self,
        artifacts: tuple[ReleaseArtifact, ...] | list[ReleaseArtifact],
    ) -> list[str]:
        """Validate artifacts for release completeness.

        Args:
            artifacts: The artifacts to validate.

        Returns:
            A list of problem strings; empty when the artifacts are valid.
        """
        return validate(artifacts)

    def build_release_info(
        self,
        level: BumpLevel = "patch",
        title: str = "",
        date: str = "",
        sections: dict[str, list[str]] | None = None,
        *,
        include_artifacts: bool = True,
    ) -> ReleaseInfo:
        """Build a complete :class:`ReleaseInfo` for the next release.

        Args:
            level: The bump level for the release version.
            title: Optional release title.
            date: Optional ISO release date.
            sections: Optional changelog sections.
            include_artifacts: Whether to discover and attach the built
                distribution artifacts.  Default ``True``.

        Returns:
            A :class:`ReleaseInfo` describing the planned release.
        """
        version = self.next_version(level)
        entry = self.changelog_entry(
            version=version, title=title, date=date, sections=sections
        )
        artifacts = self.artifacts(str(version)) if include_artifacts else ()
        return ReleaseInfo(version=version, changelog=entry, artifacts=artifacts)
