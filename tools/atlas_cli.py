#!/usr/bin/env python3
"""Atlas CLI — developer tooling for the Atlas framework.

This is a thin backward-compatible shim over the installed package entry
point in :mod:`app.cli.main`.  The CLI is installed as the ``atlas``
console script; this module exists so existing ``python tools/atlas_cli.py
<command>`` invocations (docs, CI) keep working unchanged.

Usage:
    atlas info              Show system and package information
    atlas doctor            Diagnose common setup issues
    atlas version           Show version information
    atlas list-packages     List all registered Atlas packages
    atlas list-providers    List available embedding providers
    atlas list-rerankers    List available reranker implementations
    atlas list-vectorstores List available vector store implementations
    atlas validate-config   Validate a JSON configuration file
    atlas release-version   Show the current project version
    atlas release-next      Show the next version for a bump level
    atlas release-changelog Show the changelog entry for a release
    atlas release-check     Validate built distribution artifacts
"""

import sys

from app.cli.main import main

if __name__ == "__main__":
    sys.exit(main())