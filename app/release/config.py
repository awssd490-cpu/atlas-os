"""Release engineering configuration.

Configuration is an immutable frozen dataclass following the convention
established in ``app.rag.persistence.config``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseConfig:
    """Configuration for release engineering operations.

    Attributes:
        project_name: The package/project name used to resolve the
            installed version.  Default ``"tekvora-atlas"``.
        version_source: Where the authoritative version is read from —
            ``"package"`` (importlib.metadata) or ``"pyproject"``
            (``pyproject.toml``).  Default ``"package"``.
        pyproject_path: Path to ``pyproject.toml`` when
            ``version_source == "pyproject"``.  Default
            ``"pyproject.toml"``.
        changelog_path: Path to the changelog file.  Default
            ``"CHANGELOG.md"``.
        dist_dir: Directory containing built distribution artifacts.
            Default ``"dist"``.
    """

    project_name: str = "tekvora-atlas"
    version_source: str = "package"
    pyproject_path: str = "pyproject.toml"
    changelog_path: str = "CHANGELOG.md"
    dist_dir: str = "dist"

    def validate(self) -> None:
        """Validate configuration values.

        Raises:
            InvalidReleaseConfiguration: If any value is out of range
                or invalid.
        """
        from app.release.errors import InvalidReleaseConfiguration

        if self.version_source not in ("package", "pyproject"):
            raise InvalidReleaseConfiguration(
                f"Invalid version_source {self.version_source!r}; "
                f"expected 'package' or 'pyproject'"
            )
        if not self.project_name.strip():
            raise InvalidReleaseConfiguration("project_name must not be empty")
        if not self.dist_dir.strip():
            raise InvalidReleaseConfiguration("dist_dir must not be empty")
