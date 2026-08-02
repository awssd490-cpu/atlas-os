# Atlas Release Engineering

## Purpose

Release engineering ties together versioning, changelog construction, artifact
validation, and publishing so that Atlas releases are consistent, documented,
and verifiable. The `app.release` subsystem provides the building blocks; the
CLI (`tools/atlas_cli.py`) and CI workflows orchestrate them.

## Versioning

The authoritative project version lives in `pyproject.toml`
(`[project].version`) and the installed package metadata. The
`app.release.versioning` module resolves either source into a
`Version` frozen dataclass and can produce the next version for a bump level.

### Version model

A `Version` has four components:

| Component | Meaning | Example |
|---|---|---|
| `major` | Breaking changes | `0` |
| `minor` | Backward-compatible features | `11` |
| `patch` | Backward-compatible fixes | `0` |
| `prerelease` | Optional pre-release qualifier | `alpha`, `rc.1` |

Tags use the `v` prefix to match project history: `v1.0.0`, `v1.1.0-alpha`.

### Bump levels

| Level | Effect | Example (`1.0.0`) |
|---|---|---|
| `patch` | Increment patch, drop prerelease | `1.0.1` |
| `minor` | Increment minor, reset patch | `1.1.0` |
| `major` | Increment major, reset minor+patch | `2.0.0` |
| `prerelease` | Promote to `-alpha` or advance the qualifier | `1.0.0-alpha`, `1.0.0-alpha.1` |

## Changelog

Changelogs follow Keep a Changelog: a top-level `##` heading per version,
optional `###` sub-sections, and bullet lists. The `app.release.changelog`
module parses `CHANGELOG.md` into `Changelog`/`ChangelogEntry` objects and
renders entries back into markdown fragments for GitHub release notes.

## Artifact validation

Built distributions are discovered from the `dist/` directory and classified
as wheels or sdists. Each artifact records its file size and SHA-256 digest.
Validation checks that a release contains at least one wheel and one sdist and
that no artifact is empty.

## CLI usage

The CLI is installed as the `atlas` console script when `tekvora-atlas`
is installed:

```bash
# Show the current project version and its git tag
atlas release-version

# Show the next version for a bump level (default: patch)
atlas release-next minor

# Show the changelog entry for the current version
atlas release-changelog

# Show the changelog entry for a specific version
atlas release-changelog --version 2.0.0

# Validate built distribution artifacts in dist/
atlas release-check
```

The same commands are available from the repository source via the
`tools/atlas_cli.py` shim (run from the repository root so `app` is
importable, or set `PYTHONPATH=.`): `python tools/atlas_cli.py
release-version`, etc.

## Publishing

Stable version tags (`v1.0.0`, no prerelease suffix) trigger the release
workflow, which:

1. Runs the full test suite.
2. Builds the wheel and source distribution (`python -m build`).
3. Validates package metadata (`twine check`).
4. Creates a GitHub Release with the artifacts attached.
5. Publishes to PyPI using trusted publishing (OIDC) — **stable tags only**.

Pre-release tags (`v1.1.0-alpha`) skip PyPI publishing so release candidates
stay out of the stable index.

> **Note:** PyPI publishing requires a trusted publisher configured for this
> repository on the PyPI project. See the [trusted publishing
> guide](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).
