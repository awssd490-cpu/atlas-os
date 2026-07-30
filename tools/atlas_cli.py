#!/usr/bin/env python3
"""
Atlas CLI — developer tooling for the Atlas RAG framework.

Usage:
    atlas info              Show system and package information
    atlas doctor            Diagnose common setup issues
    atlas version           Show version information
    atlas list-packages     List all registered Atlas packages
    atlas list-providers    List available embedding providers
    atlas list-rerankers    List available reranker implementations
    atlas list-vectorstores List available vector store implementations
    atlas validate-config   Validate a JSON configuration file
"""

import argparse
import importlib
import json
import os
import sys
import textwrap
from pathlib import Path


def main() -> None:
    """Entry point for the Atlas CLI."""
    parser = argparse.ArgumentParser(
        description="Atlas CLI — developer tooling for the Atlas RAG framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              atlas info
              atlas doctor
              atlas version
              atlas list-packages
              atlas list-providers
              atlas list-rerankers
              atlas list-vectorstores
              atlas validate-config ./config.json
        """),
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    subparsers.add_parser("info", help="Show system and package information")

    # doctor
    subparsers.add_parser("doctor", help="Diagnose common setup issues")

    # version
    subparsers.add_parser("version", help="Show version information")

    # list-packages
    subparsers.add_parser("list-packages", help="List all registered Atlas packages")

    # list-providers
    subparsers.add_parser("list-providers", help="List available embedding providers")

    # list-rerankers
    subparsers.add_parser("list-rerankers", help="List available reranker implementations")

    # list-vectorstores
    subparsers.add_parser("list-vectorstores", help="List available vector store implementations")

    # validate-config
    validate_parser = subparsers.add_parser(
        "validate-config", help="Validate a JSON configuration file"
    )
    validate_parser.add_argument("file", help="Path to a JSON configuration file")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "info": cmd_info,
        "doctor": cmd_doctor,
        "version": cmd_version,
        "list-packages": cmd_list_packages,
        "list-providers": cmd_list_providers,
        "list-rerankers": cmd_list_rerankers,
        "list-vectorstores": cmd_list_vectorstores,
        "validate-config": cmd_validate_config,
    }

    cmd = commands[args.command]
    if args.command == "validate-config":
        cmd(args.file)
    else:
        cmd()


# ---------------------------------------------------------------------------
# Command: info
# ---------------------------------------------------------------------------

def cmd_info() -> None:
    """Show system and package information."""
    print("=" * 60)
    print("Atlas System Information")
    print("=" * 60)
    print(f"  Python:       {sys.version.split()[0]}")
    print(f"  Platform:     {sys.platform}")
    print(f"  Executable:   {sys.executable}")
    print()

    # Check if atlas packages are importable
    packages = [
        ("app.core.config", "Configuration"),
        ("app.core.log", "Logging"),
        ("app.core.health", "Health Monitoring"),
        ("app.core.concurrency", "Concurrency"),
        ("app.core.reliability", "Reliability"),
        ("app.rag", "RAG Core"),
        ("app.rag.chunking", "Chunking"),
        ("app.rag.embeddings", "Embeddings"),
        ("app.rag.vectorstore", "Vector Store"),
        ("app.rag.hybrid", "Hybrid Retrieval"),
        ("app.rag.rerank", "Reranking"),
        ("app.rag.pipeline", "Pipeline"),
        ("app.rag.persistence", "Persistence"),
        ("app.rag.evaluation", "Evaluation"),
    ]

    max_name = max(len(name) for _, name in packages)
    print("  Installed packages:")
    for module, name in packages:
        status = _check_import(module)
        print(f"    {name:<{max_name}}  {status}")


def _check_import(module: str) -> str:
    """Check if a module can be imported, returning a status string."""
    try:
        importlib.import_module(module)
        return "[ok]"
    except ImportError:
        return "[!!]"


# ---------------------------------------------------------------------------
# Command: doctor
# ---------------------------------------------------------------------------

def cmd_doctor() -> None:
    """Diagnose common setup issues."""
    print("=" * 60)
    print("Atlas Doctor — Setup Diagnostics")
    print("=" * 60)

    checks = [
        ("Python version >= 3.12", _check_python_version()),
        ("app.core.errors importable", _check_import_bool("app.core.errors")),
        ("app.rag.models importable", _check_import_bool("app.rag.models")),
        ("app.rag.pipeline importable", _check_import_bool("app.rag.pipeline")),
        ("Pytest available", _check_import_bool("pytest")),
        ("Tests directory exists", os.path.isdir("tests") if os.path.exists("tests") else False),
    ]

    all_pass = True
    for desc, passed in checks:
        status = "[ok]" if passed else "[!!]"
        if not passed:
            all_pass = False
        print(f"  {status} {desc}")

    print()
    if all_pass:
        print("  All checks passed!")
    else:
        print("  Some checks failed. See diagnostics above.")
        print("  Run: pip install -e .")


def _check_python_version() -> bool:
    """Check Python version is 3.12 or higher."""
    return sys.version_info >= (3, 12)


def _check_import_bool(module: str) -> bool:
    """Check if a module is importable, returning a boolean."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Command: version
# ---------------------------------------------------------------------------

def cmd_version() -> None:
    """Show version information."""
    version = _get_version()
    print(f"Atlas version {version}")
    print(f"Python {sys.version.split()[0]} on {sys.platform}")


def _get_version() -> str:
    """Try to get version from package metadata, fallback to 'dev'."""
    try:
        from importlib.metadata import version
        return version("atlas")
    except (ImportError, Exception):
        return "dev"


# ---------------------------------------------------------------------------
# Command: list-packages
# ---------------------------------------------------------------------------

def cmd_list_packages() -> None:
    """List all registered Atlas packages."""
    packages = [
        ("app.core.config", "Configuration management"),
        ("app.core.log", "Structured logging"),
        ("app.core.health", "Health monitoring"),
        ("app.core.concurrency", "Concurrency & resources"),
        ("app.core.reliability", "Retry & reliability"),
        ("app.rag", "RAG domain models & core"),
        ("app.rag.chunking", "Document chunking engine"),
        ("app.rag.embeddings", "Embedding providers"),
        ("app.rag.vectorstore", "Vector storage & search"),
        ("app.rag.hybrid", "Hybrid retrieval"),
        ("app.rag.rerank", "Result reranking"),
        ("app.rag.pipeline", "Pipeline orchestration"),
        ("app.rag.persistence", "State persistence"),
        ("app.rag.evaluation", "Quality evaluation"),
    ]

    print(f"{'Package':<35} {'Status':<10}  Description")
    print("-" * 75)
    for pkg, desc in packages:
        status = _check_import(pkg)
        print(f"{pkg:<35} {status:<10}  {desc}")


# ---------------------------------------------------------------------------
# Command: list-providers
# ---------------------------------------------------------------------------

def cmd_list_providers() -> None:
    """List available embedding providers."""
    print("=" * 60)
    print("Embedding Providers")
    print("=" * 60)

    try:
        from app.rag.embeddings import list_providers
        providers = list_providers()
        if providers:
            for name in sorted(providers):
                print(f"  - {name}")
        else:
            print("  No providers registered.")
    except ImportError:
        print("  (app.rag.embeddings not available — install the atlas package)")


# ---------------------------------------------------------------------------
# Command: list-rerankers
# ---------------------------------------------------------------------------

def cmd_list_rerankers() -> None:
    """List available reranker implementations."""
    print("=" * 60)
    print("Reranker Implementations")
    print("=" * 60)

    try:
        from app.rag.rerank import list_rerankers
        rerankers = list_rerankers()
        if rerankers:
            for name in sorted(rerankers):
                print(f"  - {name}")
        else:
            print("  No rerankers registered.")
    except ImportError:
        print("  (app.rag.rerank not available — install the atlas package)")


# ---------------------------------------------------------------------------
# Command: list-vectorstores
# ---------------------------------------------------------------------------

def cmd_list_vectorstores() -> None:
    """List available vector store implementations."""
    print("=" * 60)
    print("Vector Store Implementations")
    print("=" * 60)

    try:
        from app.rag.rerank import list_rerankers
        # Vector stores don't have a registry yet, list known implementations
        stores = ["MemoryVectorStore (built-in)"]
        for name in stores:
            print(f"  - {name}")
        print()
        print("  Vector stores implement the VectorStore ABC.")
        print("  Register via app.rag.vectorstore base class.")
    except ImportError:
        print("  (app.rag.vectorstore not available)")


# ---------------------------------------------------------------------------
# Command: validate-config
# ---------------------------------------------------------------------------

def cmd_validate_config(file_path: str) -> None:
    """Validate a JSON configuration file."""
    path = Path(file_path)

    if not path.exists():
        print(f"[!!] File not found: {file_path}")
        sys.exit(1)

    try:
        from app.core.config import ConfigLoader
    except ImportError:
        print("[!!] app.core.config not available -- install the atlas package")
        sys.exit(1)

    print(f"Validating: {file_path}")
    print()

    try:
        cfg = ConfigLoader.from_json(str(path))
        print("[ok] Configuration is valid!")
        print()
        print("  Parsed values:")
        print(f"    environment: {cfg.environment}")
        print(f"    debug:       {cfg.debug}")
        print(f"    log_level:   {cfg.log_level}")
        print(f"    random_seed: {cfg.random_seed}")
    except Exception as exc:
        print(f"[!!] Validation failed: {exc}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
