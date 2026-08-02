"""Tests for the release engineering architecture and ReleaseService."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import pytest

from app.core.errors import AtlasError
from app.release import (
    ArtifactError,
    Changelog,
    ChangelogEntry,
    ChangelogError,
    InvalidReleaseConfiguration,
    ReleaseArtifact,
    ReleaseConfig,
    ReleaseError,
    ReleaseInfo,
    ReleaseService,
    Version,
    VersionError,
    VersionNotFound,
    bump,
    classify,
    discover,
    make_entry,
    parse_changelog,
    render_entry,
    resolve_version,
    validate,
)
from app.release.artifacts import (
    ReleaseArtifact as ReleaseArtifact_Impl,
    discover as discover_impl,
    validate as validate_impl,
)
from app.release.changelog import (
    Changelog as Changelog_Impl,
    ChangelogEntry as ChangelogEntry_Impl,
    make_entry as make_entry_impl,
    parse_changelog as parse_changelog_impl,
    render_entry as render_entry_impl,
)
from app.release.config import ReleaseConfig as ReleaseConfig_Impl
from app.release.errors import (
    ArtifactError as ArtifactError_Impl,
    ChangelogError as ChangelogError_Impl,
    InvalidReleaseConfiguration as InvalidReleaseConfiguration_Impl,
    ReleaseError as ReleaseError_Impl,
    VersionError as VersionError_Impl,
    VersionNotFound as VersionNotFound_Impl,
)
from app.release.models import (
    ReleaseInfo as ReleaseInfo_Impl,
    Version as Version_Impl,
)
from app.release.service import ReleaseService as ReleaseService_Impl
from app.release.versioning import (
    bump as bump_impl,
    resolve_version as resolve_version_impl,
)

# ======================================================================
# Imports
# ======================================================================

class TestImports:
    def test_version_imported(self) -> None:
        assert Version is Version_Impl

    def test_release_info_imported(self) -> None:
        assert ReleaseInfo is ReleaseInfo_Impl

    def test_release_service_imported(self) -> None:
        assert ReleaseService is ReleaseService_Impl

    def test_release_config_imported(self) -> None:
        assert ReleaseConfig is ReleaseConfig_Impl

    def test_changelog_imported(self) -> None:
        assert Changelog is Changelog_Impl

    def test_changelog_entry_imported(self) -> None:
        assert ChangelogEntry is ChangelogEntry_Impl

    def test_release_artifact_imported(self) -> None:
        assert ReleaseArtifact is ReleaseArtifact_Impl

    def test_functions_imported(self) -> None:
        assert bump is bump_impl
        assert resolve_version is resolve_version_impl
        assert make_entry is make_entry_impl
        assert parse_changelog is parse_changelog_impl
        assert render_entry is render_entry_impl
        assert discover is discover_impl
        assert validate is validate_impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(ReleaseError, AtlasError)
        assert issubclass(VersionError, ReleaseError)
        assert issubclass(VersionNotFound, ReleaseError)
        assert issubclass(ChangelogError, ReleaseError)
        assert issubclass(ArtifactError, ReleaseError)
        assert issubclass(InvalidReleaseConfiguration, ReleaseError)

    def test_error_codes(self) -> None:
        assert ReleaseError("x").code == "RELEASE_ERROR"
        assert VersionError("x").code == "VERSION_ERROR"
        assert VersionNotFound("x").code == "VERSION_NOT_FOUND"
        assert ChangelogError("x").code == "CHANGELOG_ERROR"
        assert ArtifactError("x").code == "ARTIFACT_ERROR"
        assert InvalidReleaseConfiguration("x").code == "INVALID_RELEASE_CONFIGURATION"

    def test_error_impl_imported(self) -> None:
        assert ReleaseError is ReleaseError_Impl
        assert VersionError is VersionError_Impl
        assert ArtifactError is ArtifactError_Impl
        assert ChangelogError is ChangelogError_Impl
        assert InvalidReleaseConfiguration is InvalidReleaseConfiguration_Impl


# ======================================================================
# Version
# ======================================================================

class TestVersion:
    def test_defaults(self) -> None:
        v = Version(1, 2, 3)
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == ""
        assert not v.is_prerelease

    def test_frozen(self) -> None:
        v = Version(1, 0, 0)
        with pytest.raises(AttributeError):
            v.major = 2  # type: ignore[misc]

    def test_str(self) -> None:
        assert str(Version(1, 2, 3)) == "1.2.3"
        assert str(Version(1, 2, 3, "alpha")) == "1.2.3-alpha"

    def test_tag(self) -> None:
        assert Version(1, 2, 3).tag == "v1.2.3"
        assert Version(1, 2, 3, "rc.1").tag == "v1.2.3-rc.1"

    def test_parse(self) -> None:
        v = Version.parse("1.0.0")
        assert (v.major, v.minor, v.patch) == (1, 0, 0)
        assert v.prerelease == ""

    def test_parse_v_prefix(self) -> None:
        v = Version.parse("v1.0.0-alpha")
        assert (v.major, v.minor, v.patch) == (1, 0, 0)
        assert v.prerelease == "alpha"

    def test_parse_prerelease(self) -> None:
        v = Version.parse("1.2.3-rc.1")
        assert v.prerelease == "rc.1"
        assert v.is_prerelease

    def test_parse_invalid(self) -> None:
        for bad in ("1.2", "1.2.3.4", "abc", "1.2.x", ""):
            with pytest.raises(VersionError):
                Version.parse(bad)

    def test_to_dict(self) -> None:
        d = Version(1, 2, 3, "alpha").to_dict()
        assert d["version"] == "1.2.3-alpha"
        assert d["major"] == 1
        assert d["prerelease"] == "alpha"


# ======================================================================
# Bump
# ======================================================================

class TestBump:
    def test_patch(self) -> None:
        assert bump(Version(1, 2, 3), "patch") == Version(1, 2, 4)

    def test_minor(self) -> None:
        assert bump(Version(1, 2, 3), "minor") == Version(1, 3, 0)

    def test_major(self) -> None:
        assert bump(Version(1, 2, 3), "major") == Version(2, 0, 0)

    def test_bump_drops_prerelease(self) -> None:
        assert bump(Version(1, 2, 3, "alpha"), "patch") == Version(1, 2, 4)
        assert not bump(Version(1, 2, 3, "alpha"), "minor").is_prerelease

    def test_prerelease_from_stable(self) -> None:
        assert bump(Version(1, 2, 3), "prerelease") == Version(1, 2, 3, "alpha")

    def test_prerelease_increment(self) -> None:
        assert bump(Version(1, 2, 3, "alpha"), "prerelease") == Version(1, 2, 3, "alpha.1")
        assert bump(Version(1, 2, 3, "alpha.1"), "prerelease") == Version(1, 2, 3, "alpha.2")

    def test_unsupported_level(self) -> None:
        with pytest.raises(VersionError):
            bump(Version(1, 0, 0), "bogus")  # type: ignore[arg-type]


# ======================================================================
# ReleaseConfig
# ======================================================================

class TestReleaseConfig:
    def test_defaults(self) -> None:
        cfg = ReleaseConfig()
        assert cfg.project_name == "tekvora-atlas"
        assert cfg.version_source == "package"
        assert cfg.dist_dir == "dist"

    def test_frozen(self) -> None:
        cfg = ReleaseConfig()
        with pytest.raises(AttributeError):
            cfg.project_name = "other"  # type: ignore[misc]

    def test_validate_valid(self) -> None:
        ReleaseConfig().validate()

    def test_validate_invalid_source(self) -> None:
        with pytest.raises(InvalidReleaseConfiguration):
            ReleaseConfig(version_source="invalid").validate()

    def test_validate_empty_project(self) -> None:
        with pytest.raises(InvalidReleaseConfiguration):
            ReleaseConfig(project_name="").validate()


# ======================================================================
# Changelog
# ======================================================================

class TestChangelog:
    def test_entry_defaults(self) -> None:
        entry = ChangelogEntry(version="1.0.0")
        assert entry.title == ""
        assert entry.date == ""
        assert entry.sections == {}

    def test_entry_frozen(self) -> None:
        entry = ChangelogEntry(version="1.0.0")
        with pytest.raises(AttributeError):
            entry.version = "2.0.0"  # type: ignore[misc]

    def test_changelog_bool(self) -> None:
        assert not Changelog()
        assert Changelog((ChangelogEntry(version="1.0.0"),))

    def test_make_entry(self) -> None:
        entry = make_entry(
            "1.0.0",
            title="Launch",
            date="2026-08-02",
            sections={"Added": ["Feature A", "Feature B"]},
        )
        assert entry.version == "1.0.0"
        assert entry.title == "Launch"
        assert entry.date == "2026-08-02"
        assert entry.sections == {"Added": ("Feature A", "Feature B")}

    def test_render_entry(self) -> None:
        entry = make_entry(
            "1.0.0",
            title="Launch",
            sections={"Added": ["Feature A"]},
        )
        rendered = render_entry(entry)
        assert "# 1.0.0" in rendered
        assert "## Added" in rendered
        assert "- Feature A" in rendered

    def test_parse_changelog_missing(self) -> None:
        with pytest.raises(ChangelogError):
            parse_changelog("/nonexistent/CHANGELOG.md")


# ======================================================================
# Artifacts
# ======================================================================

class TestArtifacts:
    def test_classify_wheel(self) -> None:
        assert classify("tekvora_atlas-1.0.0-py3-none-any.whl") == "wheel"
        assert classify("tekvora_atlas-1.0.0-py2.py3-none-any.whl") == "wheel"

    def test_classify_sdist(self) -> None:
        assert classify("tekvora_atlas-1.0.0.tar.gz") == "sdist"
        assert classify("tekvora_atlas-1.0.0.zip") == "sdist"

    def test_classify_other(self) -> None:
        assert classify("README.md") is None
        assert classify("tekvora_atlas-1.0.0.whl") is None

    def test_discover_missing_dir(self) -> None:
        with pytest.raises(ArtifactError):
            discover("/nonexistent/dist")

    def test_discover_empty_dir(self, tmp_path: Path) -> None:
        assert discover(str(tmp_path)) == ()

    def test_discover_filters_version(self, tmp_path: Path) -> None:
        (tmp_path / "tekvora_atlas-1.0.0-py3-none-any.whl").write_bytes(b"wheel-bytes")
        (tmp_path / "tekvora_atlas-1.0.0.tar.gz").write_bytes(b"sdist-bytes")
        (tmp_path / "tekvora_atlas-1.1.0-py3-none-any.whl").write_bytes(b"other")
        (tmp_path / "README.md").write_text("not an artifact")

        all_artifacts = discover(str(tmp_path))
        assert len(all_artifacts) == 3

        filtered = discover(str(tmp_path), expected_version="1.0.0")
        assert len(filtered) == 2
        assert {a.kind for a in filtered} == {"wheel", "sdist"}

    def test_discover_sha256(self, tmp_path: Path) -> None:
        (tmp_path / "tekvora_atlas-1.0.0.tar.gz").write_bytes(b"data")
        artifacts = discover(str(tmp_path))
        assert len(artifacts) == 1
        import hashlib
        assert artifacts[0].sha256 == hashlib.sha256(b"data").hexdigest()

    def test_validate_complete(self) -> None:
        artifacts = (
            ReleaseArtifact("w.whl", "atlas-0.11.0-py3-none-any.whl", "wheel", 10, "a" * 64),
            ReleaseArtifact("s.tar.gz", "atlas-0.11.0.tar.gz", "sdist", 20, "b" * 64),
        )
        assert validate(artifacts) == []

    def test_validate_missing_kinds(self) -> None:
        wheel_only = (ReleaseArtifact("w.whl", "x.whl", "wheel", 10, "a" * 64),)
        assert "no sdist artifact found" in validate(wheel_only)
        assert "no wheel artifact found" in validate((), require_wheel=True)

    def test_validate_empty_file(self) -> None:
        artifacts = (ReleaseArtifact("e.whl", "x.whl", "wheel", 0, "a" * 64),)
        assert any("empty" in p for p in validate(artifacts))


# ======================================================================
# ReleaseService
# ======================================================================

class TestReleaseService:
    def test_default_config(self) -> None:
        service = ReleaseService()
        assert service.config == ReleaseConfig()

    def test_current_version_from_package(self) -> None:
        # The installed tekvora-atlas distribution reports the current version.
        service = ReleaseService()
        version = service.current_version()
        assert version.major >= 0
        assert isinstance(version, Version)

    def test_current_version_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text(
            '[project]\nversion = "3.4.5"\n', encoding="utf-8"
        )
        service = ReleaseService(
            ReleaseConfig(version_source="pyproject", pyproject_path=str(tmp_path / "pyproject.toml"))
        )
        assert service.current_version() == Version(3, 4, 5)

    def test_current_version_missing_pyproject(self, tmp_path: Path) -> None:
        service = ReleaseService(
            ReleaseConfig(
                version_source="pyproject",
                pyproject_path=str(tmp_path / "missing.toml"),
            )
        )
        with pytest.raises(VersionNotFound):
            service.current_version()

    def test_next_version(self) -> None:
        service = ReleaseService()
        assert service.next_version("minor").minor >= service.current_version().minor

    def test_changelog_entry_default_version(self) -> None:
        service = ReleaseService()
        entry = service.changelog_entry()
        assert isinstance(entry, ChangelogEntry)
        assert entry.version == str(service.current_version())

    def test_changelog_entry_with_sections(self) -> None:
        service = ReleaseService()
        entry = service.changelog_entry(
            version=Version(2, 0, 0),
            title="Launch",
            date="2026-08-02",
            sections={"Added": ["Feature"]},
        )
        assert entry.version == "2.0.0"
        assert entry.title == "Launch"
        assert entry.sections == {"Added": ("Feature",)}

    def test_artifacts_discover(self) -> None:
        # dist_test/ contains the Phase 12.2 built artifacts.
        service = ReleaseService(ReleaseConfig(dist_dir="dist_test"))
        artifacts = service.artifacts(expected_version="0.11.0")
        assert artifacts
        assert {a.kind for a in artifacts} == {"wheel", "sdist"}

    def test_artifacts_discover_current_version(self) -> None:
        # dist_test/ contains the Phase 12.2 built artifacts (0.11.0); the
        # discovery filter is version-agnostic and returns whatever matches.
        service = ReleaseService(ReleaseConfig(dist_dir="dist_test"))
        all_artifacts = service.artifacts()
        assert all_artifacts
        assert {a.kind for a in all_artifacts} == {"wheel", "sdist"}

    def test_validate_artifacts(self) -> None:
        service = ReleaseService()
        problems = service.validate_artifacts(())
        assert "no wheel artifact found" in problems

    def test_build_release_info(self) -> None:
        service = ReleaseService(ReleaseConfig(dist_dir="dist_test"))
        info = service.build_release_info(
            level="patch", title="Next", sections={"Added": ["X"]}
        )
        assert isinstance(info, ReleaseInfo)
        assert info.version == service.next_version("patch")
        assert info.changelog is not None
        assert info.changelog.title == "Next"

    def test_build_release_info_no_artifacts(self) -> None:
        service = ReleaseService(ReleaseConfig(dist_dir="dist_test"))
        info = service.build_release_info(include_artifacts=False)
        assert info.artifacts == ()


# ======================================================================
# Changelog parsing
# ======================================================================

class TestChangelogParsing:
    def test_parse_changelog_basic(self, tmp_path: Path) -> None:
        path = tmp_path / "CHANGELOG.md"
        path.write_text(
            "# Changelog\n\n"
            "## [2.0.0]\n\n"
            "### Added\n\n"
            "- Feature A\n"
            "- Feature B\n\n"
            "### Fixed\n\n"
            "- Bug C\n\n"
            "## [1.0.0]\n\n"
            "### Added\n\n"
            "- Initial\n",
            encoding="utf-8",
        )
        changelog = parse_changelog(str(path))
        assert len(changelog.entries) == 2
        first = changelog.entries[0]
        assert first.version == "2.0.0"
        assert first.sections["Added"] == ("Feature A", "Feature B")
        assert first.sections["Fixed"] == ("Bug C",)
        assert changelog.entries[1].version == "1.0.0"

    def test_parse_changelog_no_sections(self, tmp_path: Path) -> None:
        path = tmp_path / "CHANGELOG.md"
        path.write_text("# Changelog\n\n## [1.0.0]\n\n- Loose bullet\n", encoding="utf-8")
        changelog = parse_changelog(str(path))
        assert len(changelog.entries) == 1
        assert changelog.entries[0].sections == {}

    def test_parse_changelog_to_dict(self, tmp_path: Path) -> None:
        path = tmp_path / "CHANGELOG.md"
        path.write_text("# Changelog\n\n## [1.0.0]\n\n### Added\n\n- A\n", encoding="utf-8")
        d = parse_changelog(str(path)).to_dict()
        assert d["entries"][0]["version"] == "1.0.0"


# ======================================================================
# Version parsing details
# ======================================================================

class TestVersionDetails:
    def test_parse_empty_raises(self) -> None:
        with pytest.raises(VersionError):
            Version.parse("")

    def test_bracket_version_in_changelog(self) -> None:
        # Keep a Changelog headings may wrap the version in brackets.
        import tempfile
        from app.release import parse_changelog

        with tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8"
        ) as handle:
            handle.write("# Changelog\n\n## [2.0.0]\n\n### Added\n\n- A\n")
            path = handle.name
        try:
            changelog = parse_changelog(path)
            assert changelog.entries[0].version == "2.0.0"
        finally:
            os.unlink(path)


# ======================================================================
# ReleaseInfo
# ======================================================================

class TestReleaseInfo:
    def test_to_dict(self) -> None:
        info = ReleaseInfo(
            version=Version(1, 2, 3),
            changelog=ChangelogEntry(version="1.2.3"),
        )
        d = info.to_dict()
        assert d["version"]["version"] == "1.2.3"
        assert d["tag"] == "v1.2.3"
        assert d["changelog"]["version"] == "1.2.3"

    def test_empty_changelog(self) -> None:
        info = ReleaseInfo(version=Version(1, 0, 0))
        assert info.to_dict()["changelog"] is None
