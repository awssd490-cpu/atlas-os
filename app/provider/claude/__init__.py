"""Claude (Anthropic) provider adapter.

This is the reference implementation for all future Atlas providers.
It uses ONLY Atlas abstractions — Provider, ProviderRequest, ProviderResponse,
HttpTransport, ProviderConfig.

No httpx calls exist here.  No environment variable parsing exists here.
"""

from __future__ import annotations

from app.provider.claude.provider import ClaudeProvider

__all__ = ["ClaudeProvider"]
