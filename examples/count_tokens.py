#!/usr/bin/env python3
"""Atlas Token Count Demo — count tokens via Claude's API.

Usage::

    export ATLAS_PROVIDER_API_KEY=sk-ant-...
    python examples/count_tokens.py
    python examples/count_tokens.py --text "Hello, world!"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.provider.claude import ClaudeProvider
from app.provider.config import ConfigBuilder, DictConfigSource, EnvConfigSource
from app.provider.models import ProviderMessage, ProviderRequest, Role


async def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas Token Count Demo")
    parser.add_argument("--text", default="Hello, how are you today?")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--api-key", default="")
    args = parser.parse_args()

    # Build config
    builder = ConfigBuilder()
    builder.add_source(EnvConfigSource(prefix="ATLAS_PROVIDER_"))
    builder.add_source(DictConfigSource({
        "model": args.model,
        "api_key": args.api_key,
    }, priority=10))
    config = builder.build(name="claude")

    if not config.credentials.has_key:
        print("Error: No API key found. Set ATLAS_PROVIDER_API_KEY or pass --api-key")
        sys.exit(1)

    # Create provider
    provider = ClaudeProvider(config=config)
    await provider.initialize()

    try:
        request = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content=args.text)],
            metadata={"model": args.model},
        )
        count = await provider.count_tokens(request)
        print(f"Text:        {args.text}")
        print(f"Characters:  {len(args.text)}")
        print(f"Token count: {count}")

    finally:
        await provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
