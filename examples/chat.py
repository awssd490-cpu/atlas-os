#!/usr/bin/env python3
"""Atlas Chat — Interactive chat with Claude via the Atlas stack.

Usage::

    # Set your API key
    export ATLAS_PROVIDER_API_KEY=sk-ant-...

    # Run with defaults
    python examples/chat.py

    # Run with options
    python examples/chat.py --model claude-sonnet-4-20250514 --verbose

    # Override API key inline
    python examples/chat.py --api-key sk-ant-...
"""

from __future__ import annotations

import sys
import os

# Add project root to path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.cli.chat import build_arg_parser, main as _main

if __name__ == "__main__":
    _main()
