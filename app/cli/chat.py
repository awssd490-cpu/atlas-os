"""Atlas CLI — interactive chat with any registered provider.

Usage::
    python -m app.cli.chat
    python -m app.cli.chat --provider claude --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

from app.provider.claude import ClaudeProvider
from app.provider.config import ConfigBuilder, DictConfigSource, EnvConfigSource
from app.provider.errors import (
    AuthenticationError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.provider.factory import ProviderFactory
from app.provider.models import ProviderMessage, ProviderRequest, Role
from app.provider.registry import ProviderRegistry


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Atlas CLI — AI chat")
    parser.add_argument("--provider", default="claude", help="Provider name (default: claude)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model name")
    parser.add_argument("--system", default="You are a helpful assistant.", help="System prompt")
    parser.add_argument("--verbose", action="store_true", help="Show provider, model, timing")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens per response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--api-key", default="", help="API key (overrides ATLAS_PROVIDER_API_KEY)")
    return parser


def build_chat_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build a config dict from CLI args."""
    config: dict[str, Any] = {
        "model": args.model,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.api_key:
        config["api_key"] = args.api_key
    return config


class ChatCLI:
    """Interactive chat loop using the Atlas provider stack."""

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._provider: Any = None
        self._history: list[ProviderMessage] = []

    async def run(self) -> None:
        """Run the chat loop."""
        try:
            await self._initialize()
            await self._loop()
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
        except AuthenticationError:
            print(f"\nError: Authentication failed. Check your API key.")
            print(f"  Set ATLAS_PROVIDER_API_KEY or pass --api-key")
            sys.exit(1)
        except ProviderUnavailableError as exc:
            print(f"\nError: Provider unavailable: {exc}")
            sys.exit(1)
        except Exception as exc:
            print(f"\nUnexpected error: {exc}")
            if self._args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
        finally:
            await self._shutdown()

    async def _initialize(self) -> None:
        """Build config, create provider via factory, initialize."""
        args = self._args

        # Build config from env vars + CLI args
        builder = ConfigBuilder()
        builder.add_source(EnvConfigSource(prefix="ATLAS_PROVIDER_"))
        builder.add_source(DictConfigSource(build_chat_config(args), priority=10))
        provider_config = builder.build(name=args.provider)

        # Ensure api_key is present
        if not provider_config.credentials.has_key:
            print("Error: No API key found.")
            print("  Set ATLAS_PROVIDER_API_KEY environment variable")
            print("  or pass --api-key <key>")
            sys.exit(1)

        # Register Claude provider
        registry = ProviderRegistry()
        factory = ProviderFactory(registry)
        factory.register_constructor("claude", ClaudeProvider)

        # Create and initialize
        if args.verbose:
            print(f"Configuring provider: {args.provider}")
            print(f"Model: {provider_config.model or args.model}")
            print()

        self._provider = await factory.create_and_initialize(
            args.provider,
            provider_config=provider_config,
            register=True,
            set_default=True,
        )

    async def _shutdown(self) -> None:
        if self._provider is not None:
            await self._provider.shutdown()

    async def _loop(self) -> None:
        """Main REPL loop with conversation history."""
        args = self._args
        print(f"\nAtlas v0.5.1-alpha")
        print(f"Provider: {args.provider}")
        print(f"Model: {args.model or 'default'}")
        print("-" * 40)
        print("Commands: /exit, /clear, /help")
        print()

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    break
                continue

            await self._chat_turn(user_input)

    def _handle_command(self, command: str) -> bool:
        """Handle slash commands.  Returns True to exit."""
        cmd = command.lower()

        if cmd in ("/exit", "/quit"):
            print("Goodbye!")
            return True
        if cmd in ("/clear", "/new"):
            self._history.clear()
            print("Conversation reset.")
            return False
        if cmd in ("/help", "/?"):
            print("Commands:")
            print("  /exit       Exit the chat")
            print("  /clear      Clear conversation history")
            print("  /help       Show this help")
            return False
        print(f"Unknown command: {command}")
        return False

    async def _chat_turn(self, user_input: str) -> None:
        """Process one turn of conversation."""
        args = self._args

        # Add user message to history
        user_msg = ProviderMessage(role=Role.USER, content=user_input)
        self._history.append(user_msg)

        # Build request with conversation history
        request = ProviderRequest(
            messages=list(self._history),
            system=args.system,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            metadata={"model": args.model},
        )

        # Send
        t0 = time.monotonic()
        try:
            response = await self._provider.generate(request)
        except RateLimitError:
            print("Assistant: (Rate limited. Please wait a moment.)")
            return
        except AuthenticationError:
            print("Assistant: (Authentication failed. Check your API key.)")
            return
        except ProviderUnavailableError:
            print("Assistant: (Provider unavailable. Try again later.)")
            return

        elapsed = time.monotonic() - t0

        # Display
        print(f"Assistant: {response.content}")
        if args.verbose and response.usage:
            print(f"  [{response.usage.prompt_tokens} prompt → "
                  f"{response.usage.completion_tokens} completion, "
                  f"{elapsed:.1f}s]")
        print()

        # Add assistant response to history
        assistant_msg = ProviderMessage(
            role=Role.ASSISTANT,
            content=response.content,
            tool_calls=response.tool_calls,
        )
        self._history.append(assistant_msg)


def main() -> None:
    """Entry point for ``python -m app.cli.chat``."""
    parser = build_arg_parser()
    args = parser.parse_args()
    cli = ChatCLI(args)
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()
