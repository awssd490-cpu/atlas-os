"""Version resolution and bumping for Atlas releases.

The authoritative project version lives in ``pyproject.toml``
(``[project].version``) and in the installed package metadata.  This
module resolves either source into a :class:`~app.release.models.Version`
and can produce the next release version for a given bump level.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from app.release.config import ReleaseConfig
from app.release.errors import VersionError, VersionNotFound
from app.release.models import Version

# Bump levels supported by :func:`bump`.
BumpLevel = Literal["major", "minor", "patch", "prerelease"]

_PYPROJECT_VERSION_PATTERN = re.compile(r"^version\s*=\s*[\"'](.+?)[\"']\s*$")


def _read_pyproject_version(path: str) -> str:
    """Read the version string from ``pyproject.toml``.

    Args:
        path: Path to ``pyproject.toml``.

    Returns:
        The version string declared under ``[project]``.

    Raises:
        VersionNotFound: If the file or the ``version`` field is missing.
    """
    pyproject = Path(path)
    if not pyproject.exists():
        raise VersionNotFound(f"pyproject.toml not found at {path!r}")
    for line in pyproject.read_text(encoding="utf-8").splitlines():
        match = _PYPROJECT_VERSION_PATTERN.match(line.strip())
        if match is not None:
            return match.group(1)
    raise VersionNotFound(f"no [project].version found in {path!r}")


def resolve_version(config: ReleaseConfig | None = None) -> Version:
    """Resolve the current project version.

    Args:
        config: Optional release configuration controlling the version
            source.  Defaults to ``ReleaseConfig()``.

    Returns:
        The parsed :class:`Version`.

    Raises:
        VersionNotFound: If the version cannot be resolved.
        InvalidReleaseConfiguration: If the version source is invalid.
    """
    from app.release.errors import InvalidReleaseConfiguration

    cfg = config or ReleaseConfig()
    cfg.validate()

    if cfg.version_source == "pyproject":
        raw = _read_pyproject_version(cfg.pyproject_path)
    elif cfg.version_source == "package":
        raw = _read_package_version(cfg.project_name)
    else:
        # Unreachable — validate() already rejected other sources.
        raise InvalidReleaseConfiguration(
            f"Invalid version_source {cfg.version_source!r}"
        )

    try:
        return Version.parse(raw)
    except VersionError as exc:
        raise VersionNotFound(
            f"could not resolve a valid version from {cfg.version_source!r} "
            f"source (got {raw!r})",
            details={"raw": raw},
        ) from exc


def _read_package_version(project_name: str) -> str:
    """Read the installed package version from importlib.metadata.

    Args:
        project_name: The distribution name (e.g. ``"atlas"``).

    Returns:
        The installed version string.

    Raises:
        VersionNotFound: If the distribution is not installed.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(project_name)
    except PackageNotFoundError as exc:
        raise VersionNotFound(
            f"distribution {project_name!r} is not installed; "
            f"run 'pip install -e .' first"
        ) from exc


def bump(version: Version, level: BumpLevel = "patch") -> Version:
    """Return the next version after ``version`` for the given bump level.

    A ``prerelease`` bump advances the pre-release qualifier or promotes a
    stable version to ``-alpha``.  ``major``, ``minor`` and ``patch`` bumps
    always drop any pre-release qualifier, producing a stable version.

    Args:
        version: The current version.
        level: The bump level.

    Returns:
        The bumped :class:`Version`.

    Raises:
        VersionError: If the bump level is unsupported or the version
            cannot be bumped.
    """
    if level == "major":
        return Version(version.major + 1, 0, 0)
    if level == "minor":
        return Version(version.major, version.minor + 1, 0)
    if level == "patch":
        return Version(version.major, version.minor, version.patch + 1)
    if level == "prerelease":
        if version.is_prerelease:
            return Version(
                version.major,
                version.minor,
                version.patch,
                _next_prerelease(version.prerelease),
            )
        return Version(version.major, version.minor, version.patch, "alpha")
    raise VersionError(f"Unsupported bump level {level!r}")


def _next_prerelease(current: str) -> str:
    """Compute the next pre-release qualifier.

    Handles both plain qualifiers (``alpha`` -> ``alpha.1``) and numbered
    qualifiers (``alpha.1`` -> ``alpha.2``).
    """
    parts = current.split(".")
    if len(parts) == 1:
        return f"{current}.1"
    head, tail = ".".join(parts[:-1]), parts[-1]
    if tail.isdigit():
        return f"{head}.{int(tail) + 1}"
    return f"{current}.1"
