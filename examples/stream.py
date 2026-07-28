#!/usr/bin/env python3
"""Atlas Streaming Demo — Stream tokens from Claude in real time.

Usage::

    export ATLAS_PROVIDER_API_KEY=sk-ant-...
    python examples/stream.py
    python examples/stream.py --prompt "Write a haiku about AI"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.provider.claude import ClaudeProvider
from app.provider.config import ConfigBuilder, DictConfigSource, EnvConfigSource
from app.provider.models import ProviderMessage, ProviderRequest, Role


async def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas Streaming Demo")
    parser.add_argument("--prompt", default="Write a short poem about artificial intelligence.")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--system", default="You are a helpful assistant.")
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
        # Build request
        request = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content=args.prompt)],
            system=args.system,
            metadata={"model": args.model},
        )

        print(f"\nPrompt: {args.prompt}")
        print("Streaming response:")
        print("-" * 40)
        sys.stdout.flush()

        # Stream
        t0 = time.monotonic()
        async for chunk in provider.stream(request):
            if chunk.content:
                print(chunk.content, end="", flush=True)
            if chunk.stop_reason:
                print()
                print("-" * 40)
                elapsed = time.monotonic() - t0
                print(f"Stop reason: {chunk.stop_reason.value}")
                if chunk.usage:
                    print(f"Tokens: {chunk.usage.total_tokens} ({elapsed:.1f}s)")
                else:
                    print(f"Elapsed: {elapsed:.1f}s")
        print()

    finally:
        await provider.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
