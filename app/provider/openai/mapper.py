"""Request and response mappers for the OpenAI-Compatible provider.

Maps between Atlas domain models and the OpenAI Chat Completions API format.
Works with ANY OpenAI-compatible endpoint (OpenAI, OpenRouter, Groq, etc.).
"""

from __future__ import annotations

from typing import Any

from app.provider.models import (
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Role,
    StopReason,
    StreamingChunk,
    ToolCallRequest,
)
from app.provider.openai.models import map_stop_reason


# ---------------------------------------------------------------------------
# Request mapping
# ---------------------------------------------------------------------------


class OpenAIRequestMapper:
    """Maps ``ProviderRequest`` → OpenAI Chat Completions request body.

    The resulting dict is compatible with any OpenAI-compatible API.
    """

    def to_dict(
        self,
        request: ProviderRequest,
        *,
        model: str = "",
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        n: int = 1,
    ) -> dict[str, Any]:
        """Convert *request* to an OpenAI Chat Completions payload.

        Args:
            request: The provider-agnostic request.
            model: The model name to use (from config or metadata).
            frequency_penalty: Default frequency penalty.
            presence_penalty: Default presence penalty.
            n: Number of completions (choices) to generate.

        Returns:
            A dict ready for JSON serialization and HTTP POST.
        """
        resolved_model = (
            request.metadata.get("model", model) or model or "gpt-4"
        )

        body: dict[str, Any] = {
            "model": resolved_model,
            "messages": self._map_messages(request),
            "max_tokens": request.max_tokens,
            "temperature": request.metadata.get("temperature", request.temperature),
            "top_p": request.metadata.get("top_p", request.top_p),
        }

        # Include frequency_penalty if explicitly set via metadata or non-default
        fp = request.metadata.get("frequency_penalty", frequency_penalty)
        if fp != 0.0:
            body["frequency_penalty"] = fp

        # Include presence_penalty if explicitly set via metadata or non-default
        pp = request.metadata.get("presence_penalty", presence_penalty)
        if pp != 0.0:
            body["presence_penalty"] = pp

        # Stop sequences
        if request.stop_sequences:
            body["stop"] = request.stop_sequences

        # Number of choices
        choice_count = request.metadata.get("n", n)
        if choice_count != 1:
            body["n"] = choice_count

        # Response format (JSON mode)
        response_format = request.metadata.get("response_format")
        if response_format is not None:
            body["response_format"] = response_format

        # User label / metadata passthrough
        user = request.metadata.get("user")
        if user:
            body["user"] = user

        # Seed for deterministic sampling
        seed = request.metadata.get("seed")
        if seed is not None:
            body["seed"] = seed

        return body

    def _map_messages(self, request: ProviderRequest) -> list[dict[str, Any]]:
        """Map messages to OpenAI Chat Completions format.

        Supports system, user, assistant, and tool roles.
        The system prompt from ``request.system`` is prepended as
        a system-role message if present.
        """
        messages: list[dict[str, Any]] = []

        # System prompt as a system message
        if request.system:
            messages.append({"role": "system", "content": request.system})

        for msg in request.messages:
            mapped = self._map_message(msg)
            if mapped is not None:
                messages.append(mapped)

        return messages

    @staticmethod
    def _map_message(msg: ProviderMessage) -> dict[str, Any] | None:
        """Map a single ``ProviderMessage`` to OpenAI format."""
        role = msg.role.value if isinstance(msg.role, Role) else str(msg.role)

        entry: dict[str, Any] = {"role": role, "content": msg.content}

        if msg.name:
            entry["name"] = msg.name

        # Map tool call requests (in assistant messages)
        if msg.tool_calls and role == "assistant":
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        # Map tool call ID (in tool role messages)
        if msg.tool_call_id and role == "tool":
            entry["tool_call_id"] = msg.tool_call_id

        return entry


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------


class OpenAIResponseMapper:
    """Maps OpenAI Chat Completions API responses → ``ProviderResponse``."""

    def to_response(self, data: dict[str, Any]) -> ProviderResponse:
        """Convert an OpenAI Chat Completions API response to ``ProviderResponse``.

        Args:
            data: The parsed JSON response from an OpenAI-compatible API.

        Returns:
            A ``ProviderResponse`` with the generated content.
        """
        choices = data.get("choices", [])
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {}) if first_choice else {}

        content = message.get("content", "") or ""
        finish_reason = first_choice.get("finish_reason", "")
        stop_reason = map_stop_reason(finish_reason)
        usage = self._map_usage(data.get("usage", {}))
        tool_calls = self._extract_tool_calls(message)

        return ProviderResponse(
            content=content,
            message=ProviderMessage(
                role=Role.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            ),
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=tool_calls,
            metadata={
                "raw_finish_reason": finish_reason,
                "id": data.get("id", ""),
                "model": data.get("model", ""),
            },
        )

    def to_chunk(self, data: dict[str, Any]) -> StreamingChunk | None:
        """Convert an OpenAI streaming chunk to a ``StreamingChunk``.

        Args:
            data: The parsed JSON from one SSE ``data:`` line.

        Returns:
            A ``StreamingChunk`` or ``None`` for chunks with no content.
        """
        choices = data.get("choices", [])

        if not choices:
            # Usage-only chunk at end of stream
            usage = self._map_usage(data.get("usage", {}))
            if usage.total_tokens > 0:
                return StreamingChunk(content="", usage=usage)
            return None

        choice = choices[0]
        delta = choice.get("delta", {})

        content = delta.get("content", "") or ""
        finish_reason = choice.get("finish_reason")
        stop_reason = map_stop_reason(finish_reason) if finish_reason else None
        index = choice.get("index", 0)
        tool_call = self._extract_delta_tool_call(delta)

        return StreamingChunk(
            content=content,
            stop_reason=stop_reason if stop_reason != StopReason.UNKNOWN else None,
            tool_call=tool_call,
            index=index,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _map_usage(usage: dict[str, Any]) -> ProviderUsage:
        """Map OpenAI usage object to ``ProviderUsage``."""
        return ProviderUsage(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    @staticmethod
    def _extract_tool_calls(message: dict[str, Any]) -> list[ToolCallRequest]:
        """Extract tool calls from an OpenAI message object."""
        calls: list[ToolCallRequest] = []
        raw_calls = message.get("tool_calls", [])
        if not raw_calls:
            return calls

        for tc in raw_calls:
            function = tc.get("function", {})
            arguments = function.get("arguments", "{}")
            # OpenAI returns arguments as JSON string; parse if possible
            if isinstance(arguments, str):
                import json
                try:
                    parsed_args: dict[str, Any] = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_args = {"raw": arguments}
            else:
                parsed_args = dict(arguments)

            calls.append(ToolCallRequest(
                id=tc.get("id", ""),
                name=function.get("name", ""),
                arguments=parsed_args,
            ))
        return calls

    @staticmethod
    def _extract_delta_tool_call(delta: dict[str, Any]) -> ToolCallRequest | None:
        """Extract a partial tool call from a streaming delta.

        OpenAI streaming sends tool calls incrementally across multiple chunks.
        This extracts the first tool call delta if present.

        Note: Full streaming tool call aggregation requires accumulating
        across chunks. This extracts a single chunk's contribution.
        """
        raw_calls = delta.get("tool_calls", [])
        if not raw_calls:
            return None

        tc = raw_calls[0]
        function = tc.get("function", {})
        return ToolCallRequest(
            id=tc.get("id", ""),
            name=function.get("name", ""),
            arguments={"delta": function.get("arguments", "")},
        )
