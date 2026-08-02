# Atlas CLI Reference

## Overview

The Atlas CLI (`tools/atlas_cli.py`) provides developer-focused commands for inspecting the Atlas installation, diagnosing setup issues, validating configuration files, and planning releases.

## Installation

Run the CLI directly from the repository root:

```bash
python tools/atlas_cli.py <command>
```

Or make it executable:

```bash
chmod +x tools/atlas_cli.py
./tools/atlas_cli.py <command>
```

## Usage

```bash
python tools/atlas_cli.py --help
```

Output:

```text
usage: atlas_cli.py [-h] {info,doctor,version,list-packages,list-providers,list-rerankers,list-vectorstores,validate-config,release-version,release-next,release-changelog,release-check} ...

Atlas CLI — developer tooling for the Atlas RAG framework.

options:
  -h, --help            show this help message and exit

Commands:
  {info,doctor,version,list-packages,list-providers,list-rerankers,list-vectorstores,validate-config,release-version,release-next,release-changelog,release-check}
    info                Show system and package information
    doctor              Diagnose common setup issues
    version             Show version information
    list-packages       List all registered Atlas packages
    list-providers      List available embedding providers
    list-rerankers      List available reranker implementations
    list-vectorstores   List available vector store implementations
    validate-config     Validate a JSON configuration file
    release-version     Show the current project version
    release-next        Show the next version for a bump level
    release-changelog   Show the changelog entry for a release
    release-check       Validate built distribution artifacts in dist/
```

## Commands

### `info`

Show system and package information.

```bash
python tools/atlas_cli.py info
```

Example output:

```text
============================================================
Atlas System Information
============================================================
  Python:       3.12.0
  Platform:     win32
  Executable:   /path/to/python

  Installed packages:
    Configuration                  ✅
    Logging                        ✅
    Health Monitoring              ✅
    Concurrency                    ✅
    Reliability                    ✅
    RAG Core                       ✅
    ...
```

### `doctor`

Diagnose common setup issues. Checks Python version, importability, and test directory.

```bash
python tools/atlas_cli.py doctor
```

Example output:

```text
============================================================
Atlas Doctor — Setup Diagnostics
============================================================
  ✅ Python version >= 3.12
  ✅ app.core.errors importable
  ✅ app.rag.models importable
  ✅ app.rag.pipeline importable
  ✅ Pytest available
  ✅ Tests directory exists

  All checks passed!
```

### `version`

Show version information.

```bash
python tools/atlas_cli.py version
```

Example output:

```text
Atlas version dev
Python 3.12.0 on win32
```

### `list-packages`

List all registered Atlas packages with status and description.

```bash
python tools/atlas_cli.py list-packages
```

Example output:

```text
Package                            Status     Description
---------------------------------------------------------------------------
app.core.config                   ✅         Configuration management
app.core.log                      ✅         Structured logging
app.core.health                   ✅         Health monitoring
app.core.concurrency              ✅         Concurrency & resources
app.core.reliability              ✅         Retry & reliability
app.rag                           ✅         RAG domain models & core
app.rag.chunking                  ✅         Document chunking engine
app.rag.embeddings                ✅         Embedding providers
app.rag.vectorstore               ✅         Vector storage & search
app.rag.hybrid                    ✅         Hybrid retrieval
app.rag.rerank                    ✅         Result reranking
app.rag.pipeline                  ✅         Pipeline orchestration
app.rag.persistence               ✅         State persistence
app.rag.evaluation                ✅         Quality evaluation
```

### `list-providers`

List available embedding providers by querying the provider registry.

```bash
python tools/atlas_cli.py list-providers
```

Example output:

```text
============================================================
Embedding Providers
============================================================
  - deterministic
  - mock
```

### `list-rerankers`

List available reranker implementations by querying the reranker registry.

```bash
python tools/atlas_cli.py list-rerankers
```

Example output:

```text
============================================================
Reranker Implementations
============================================================
  (No rerankers registered — instantiate directly)
```

### `list-vectorstores`

List available vector store implementations.

```bash
python tools/atlas_cli.py list-vectorstores
```

Example output:

```text
============================================================
Vector Store Implementations
============================================================
  - MemoryVectorStore (built-in)
```

### `validate-config`

Validate a JSON configuration file using `ConfigLoader`.

```bash
python tools/atlas_cli.py validate-config ./config.json
```

Example output:

```text
Validating: ./config.json

✅ Configuration is valid!

  Parsed values:
    environment: development
    debug:       True
    log_level:   DEBUG
    random_seed: 42
```

On failure:

```text
Validating: ./config.json

❌ Validation failed: Invalid environment: 'invalid'. Must be one of ('development', 'testing', 'staging', 'production')
```

### `release-version`

Show the current project version, its git tag, and whether it is a pre-release.

```bash
python tools/atlas_cli.py release-version
```

Example output:

```text
Atlas version 1.0.0
Tag:          v1.0.0
```

### `release-next`

Show the next version for a bump level. The level is optional and defaults to `patch`.

```bash
python tools/atlas_cli.py release-next minor
```

Example output:

```text
Current: 1.0.0
Bump:    minor
Next:    1.1.0
```

### `release-changelog`

Show the changelog entry for a release. Without `--version`, uses the current project version.

```bash
python tools/atlas_cli.py release-changelog --version 2.0.0
```

Example output:

```text
# 2.0.0
```

### `release-check`

Validate the built distribution artifacts in `dist/`. Reports the kind, size, and SHA-256 of each artifact and fails if no artifacts are found or a required artifact kind is missing.

```bash
python tools/atlas_cli.py release-check
```

Example output:

```text
Release artifacts:
  wheel  tekvora_atlas-1.0.0-py3-none-any.whl (310925 bytes)
         sha256: 4dffd3e31925dcd6...
  sdist  tekvora_atlas-1.0.0.tar.gz (340945 bytes)
         sha256: 16d13adf6d812143...

[ok] All release artifacts validated.
```

## Extending the CLI

To add a new command:

1. Define a function in `tools/atlas_cli.py` with signature `cmd_<name>(args)`
2. Add the command name to the `commands` dict in `main()`
3. Add a subparser with help text

```python
def cmd_my_command() -> None:
    """My custom command."""
    print("Running my command...")
```

```python
subparsers.add_parser("my-command", help="Description of my command")
```

```python
commands = {
    ...
    "my-command": cmd_my_command,
}
```

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'app'` | Not running from repo root | Run `cd /path/to/atlas` first |
| Provider list is empty | Providers not imported yet | `import app.rag.embeddings.providers` |
| `validate-config` shows wrong values | File has invalid JSON | Use `python -m json.tool config.json` |
