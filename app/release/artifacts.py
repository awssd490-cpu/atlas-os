"""Release artifact discovery and validation.

Distribution artifacts are the wheel and source distribution produced by
``python -m build``.  This module discovers the files in a distribution
directory, classifies them (wheel vs sdist), and computes the SHA-256
digests used for release integrity checks.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.release.errors import ArtifactError
from app.release.models import ReleaseArtifact

# ``atlas-0.11.0-py3-none-any.whl`` (wheel) and ``atlas-0.11.0.tar.gz`` (sdist).
# A wheel always carries a Python tag (``-py3``, ``-py3-none-any``), which
# distinguishes it from an sdist or a generic ``*.whl``-named file.
_WHEEL_PATTERN = re.compile(r"^.+?-\d[^/]*-py[0-9]+[^/]*\.whl$")
_SDIST_PATTERN = re.compile(r"^.+?-\d[^/]*\.(?:tar\.gz|zip)$")


def _sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(name: str) -> str | None:
    """Classify a file name as a wheel, sdist, or neither.

    Args:
        name: The base file name.

    Returns:
        ``"wheel"``, ``"sdist"``, or ``None`` if the name does not look
        like a distribution artifact.
    """
    if _WHEEL_PATTERN.match(name):
        return "wheel"
    if _SDIST_PATTERN.match(name):
        return "sdist"
    return None


def discover(dist_dir: str, expected_version: str | None = None) -> tuple[ReleaseArtifact, ...]:
    """Discover distribution artifacts in a directory.

    Args:
        dist_dir: Directory containing built artifacts.
        expected_version: Optional version string; when provided, only
            artifacts whose name contains the version are returned.

    Returns:
        A tuple of :class:`ReleaseArtifact`, sorted by name.

    Raises:
        ArtifactError: If the directory does not exist.
    """
    directory = Path(dist_dir)
    if not directory.exists():
        raise ArtifactError(f"distribution directory not found at {dist_dir!r}")

    artifacts: list[ReleaseArtifact] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        kind = classify(path.name)
        if kind is None:
            continue
        if expected_version is not None and expected_version not in path.name:
            continue
        artifacts.append(
            ReleaseArtifact(
                path=str(path),
                name=path.name,
                kind=kind,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    return tuple(artifacts)


def validate(
    artifacts: tuple[ReleaseArtifact, ...] | list[ReleaseArtifact],
    *,
    require_wheel: bool = True,
    require_sdist: bool = True,
) -> list[str]:
    """Validate a set of artifacts for release completeness.

    Args:
        artifacts: The discovered artifacts.
        require_wheel: Require at least one wheel.  Default ``True``.
        require_sdist: Require at least one sdist.  Default ``True``.

    Returns:
        A list of validation problem strings; empty when valid.
    """
    problems: list[str] = []
    kinds = {artifact.kind for artifact in artifacts}
    if require_wheel and "wheel" not in kinds:
        problems.append("no wheel artifact found")
    if require_sdist and "sdist" not in kinds:
        problems.append("no sdist artifact found")
    for artifact in artifacts:
        if artifact.size_bytes <= 0:
            problems.append(f"artifact {artifact.name!r} is empty")
    return problems
