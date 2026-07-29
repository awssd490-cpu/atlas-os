"""Universal OpenAI-Compatible Provider.

Communicates with ANY server implementing the OpenAI Chat Completions API,
including OpenAI, OpenRouter, Groq, Together AI, Fireworks AI, DeepInfra,
Nebius, SambaNova, LM Studio, LocalAI, vLLM, LiteLLM, and Ollama.

Usage::

    config = {
        "api_key": "sk-...",
        "model": "gpt-4",
        "endpoint.base_url": "https://api.openai.com/v1",
        "endpoint.api_path": "/chat/completions",
    }
    provider = OpenAICompatibleProvider(ProviderConfig.from_dict(config))
    await provider.initialize()
    response = await provider.generate(request)
"""

from __future__ import annotations

from app.provider.openai.provider import OpenAICompatibleProvider

__all__ = ["OpenAICompatibleProvider"]
