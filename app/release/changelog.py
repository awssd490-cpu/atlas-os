"""Changelog construction and rendering for Atlas releases.

The project's changelog follows Keep a Changelog: a top-level ``##``
heading per version, optional sub-sections (``### Added``,
``### Changed``, ...), and a bullet list under each.  This module parses
that structure into :class:`~app.release.models.Changelog` objects and
renders :class:`~app.release.models.ChangelogEntry` objects back into
markdown fragments used for GitHub release notes.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.release.errors import ChangelogError
from app.release.models import Changelog, ChangelogEntry

_VERSION_HEADING = re.compile(r"^##\s+(.+?)\s*$")
_BRACKETS = re.compile(r"^\[(.+?)\]")
_FIRST_TOKEN = re.compile(r"^(\S+)")
_SECTION_HEADING = re.compile(r"^###\s+(.+?)\s*$")
_BULLET = re.compile(r"^[-*]\s+(.+?)\s*$")


def parse_changelog(path: str) -> Changelog:
    """Parse a Keep-a-Changelog style file.

    Args:
        path: Path to the changelog file.

    Returns:
        An ordered :class:`Changelog` (newest first).

    Raises:
        ChangelogError: If the file cannot be read.
    """
    changelog = Path(path)
    if not changelog.exists():
        raise ChangelogError(f"changelog not found at {path!r}")
    try:
        lines = changelog.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ChangelogError(
            f"could not read changelog at {path!r}", details={"error": str(exc)}
        ) from exc

    entries: list[ChangelogEntry] = []
    current: ChangelogEntry | None = None
    current_section: str | None = None
    sections: dict[str, list[str]] = {}

    for line in lines:
        version_match = _VERSION_HEADING.match(line.strip())
        if version_match is not None:
            # Commit the previous entry.
            if current is not None:
                entries.append(_finalize(current, sections))
            version = version_match.group(1)
            # Keep a Changelog headings are often `## [2.0.0]` or
            # `## [2.0.0] — 2026-08-02`; extract the version token so the
            # entry version is the bare version string.
            bracket_match = _BRACKETS.match(version)
            if bracket_match is not None:
                version = bracket_match.group(1)
            else:
                token_match = _FIRST_TOKEN.match(version)
                if token_match is not None:
                    version = token_match.group(1)
            current = ChangelogEntry(version=version)
            sections = {}
            current_section = None
            continue

        section_match = _SECTION_HEADING.match(line.strip())
        if section_match is not None:
            if current is None:
                continue
            current_section = section_match.group(1)
            sections.setdefault(current_section, [])
            continue

        bullet_match = _BULLET.match(line.strip())
        if bullet_match is not None and current is not None and current_section:
            sections.setdefault(current_section, []).append(bullet_match.group(1))

    if current is not None:
        entries.append(_finalize(current, sections))

    return Changelog(entries=tuple(entries))


def _finalize(entry: ChangelogEntry, sections: dict[str, list[str]]) -> ChangelogEntry:
    """Materialize a partially built entry into an immutable one."""
    return ChangelogEntry(
        version=entry.version,
        title=entry.title,
        date=entry.date,
        sections={name: tuple(items) for name, items in sections.items()},
    )


def render_entry(entry: ChangelogEntry) -> str:
    """Render a changelog entry as a markdown fragment.

    Args:
        entry: The entry to render.

    Returns:
        A markdown string suitable for GitHub release notes.
    """
    lines = [f"# {entry.version}"]
    if entry.title:
        lines.append("")
        lines.append(entry.title)
    for section, items in entry.sections.items():
        lines.append("")
        lines.append(f"## {section}")
        for item in items:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def make_entry(
    version: str,
    title: str = "",
    date: str = "",
    sections: dict[str, list[str]] | None = None,
) -> ChangelogEntry:
    """Build a changelog entry from plain values.

    Args:
        version: The version string the entry describes.
        title: Optional release title.
        date: Optional ISO release date.
        sections: Optional mapping of section name to bullet items.

    Returns:
        A :class:`ChangelogEntry`.
    """
    return ChangelogEntry(
        version=version,
        title=title,
        date=date,
        sections={
            name: tuple(items) for name, items in (sections or {}).items()
        },
    )


def to_dict(changelog: Changelog) -> dict[str, Any]:
    """Serialize a changelog to a plain dict."""
    return changelog.to_dict()
