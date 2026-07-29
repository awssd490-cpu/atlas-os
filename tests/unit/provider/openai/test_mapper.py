"""Tests for OpenAI-Compatible request and response mappers."""

from __future__ import annotations

from app.provider.models import (
    ProviderMessage,
    ProviderRequest,
    Role,
    StopReason,
    ToolCallRequest,
)
from app.provider.openai.mapper import OpenAIRequestMapper, OpenAIResponseMapper
from app.provider.openai.models import map_stop_reason


class TestOpenAIRequestMapper:
    def setup_method(self) -> None:
        self.mapper = OpenAIRequestMapper()

    def test_basic_request(self) -> None:
        """Map a basic request with system prompt and user message."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hello")],
            system="Be helpful.",
            max_tokens=500,
            temperature=0.5,
        )
        body = self.mapper.to_dict(req, model="gpt-4")
        assert body["model"] == "gpt-4"
        assert body["max_tokens"] == 500
        assert body["temperature"] == 0.5
        assert body["top_p"] == 1.0
        assert len(body["messages"]) == 2  # system + user
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "Be helpful."
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "Hello"

    def test_no_system_prompt(self) -> None:
        """Without system prompt, only user message is included."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
        )
        body = self.mapper.to_dict(req, model="gpt-3.5-turbo")
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_model_from_metadata(self) -> None:
        """Model from request metadata overrides config model."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"model": "gpt-4-turbo"},
        )
        body = self.mapper.to_dict(req, model="gpt-3.5-turbo")
        assert body["model"] == "gpt-4-turbo"

    def test_default_model_fallback(self) -> None:
        """Fall back to empty-string fallback when nothing specified."""
        req = ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
        body = self.mapper.to_dict(req)
        assert body["model"] == "gpt-4"  # hardcoded fallback

    def test_temperature_top_p_from_metadata(self) -> None:
        """Temperature and top_p from metadata override request defaults."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"temperature": 0.2, "top_p": 0.5},
        )
        body = self.mapper.to_dict(req)
        assert body["temperature"] == 0.2
        assert body["top_p"] == 0.5

    def test_stop_sequences(self) -> None:
        """Stop sequences mapped to 'stop' key."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            stop_sequences=["END", "STOP"],
        )
        body = self.mapper.to_dict(req)
        assert body["stop"] == ["END", "STOP"]

    def test_frequency_penalty(self) -> None:
        """Frequency penalty included when non-zero."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"frequency_penalty": 0.5},
        )
        body = self.mapper.to_dict(req)
        assert body["frequency_penalty"] == 0.5

    def test_presence_penalty(self) -> None:
        """Presence penalty included when non-zero."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"presence_penalty": 0.3},
        )
        body = self.mapper.to_dict(req)
        assert body["presence_penalty"] == 0.3

    def test_zero_penalties_omitted(self) -> None:
        """Zero penalties omitted from body (default behavior)."""
        req = ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
        body = self.mapper.to_dict(req)
        assert "frequency_penalty" not in body
        assert "presence_penalty" not in body

    def test_multiple_messages(self) -> None:
        """Multiple messages in correct order."""
        req = ProviderRequest(messages=[
            ProviderMessage(role=Role.USER, content="Hello"),
            ProviderMessage(role=Role.ASSISTANT, content="Hi there!"),
            ProviderMessage(role=Role.USER, content="How are you?"),
        ])
        body = self.mapper.to_dict(req)
        assert len(body["messages"]) == 3
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][1]["role"] == "assistant"
        assert body["messages"][2]["role"] == "user"

    def test_message_name(self) -> None:
        """Name field included when present."""
        req = ProviderRequest(messages=[
            ProviderMessage(role=Role.USER, content="Hello", name="user_1"),
        ])
        body = self.mapper.to_dict(req)
        assert body["messages"][0]["name"] == "user_1"

    def test_json_mode(self) -> None:
        """Response format for JSON mode."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Return JSON")],
            metadata={"response_format": {"type": "json_object"}},
        )
        body = self.mapper.to_dict(req)
        assert body["response_format"] == {"type": "json_object"}

    def test_json_schema(self) -> None:
        """Response format for JSON schema."""
        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Return schema")],
            metadata={"response_format": schema},
        )
        body = self.mapper.to_dict(req)
        assert body["response_format"]["type"] == "json_schema"

    def test_user_metadata(self) -> None:
        """User metadata passed through."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"user": "abc123"},
        )
        body = self.mapper.to_dict(req)
        assert body["user"] == "abc123"

    def test_seed(self) -> None:
        """Seed for deterministic sampling."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"seed": 42},
        )
        body = self.mapper.to_dict(req)
        assert body["seed"] == 42

    def test_number_of_choices(self) -> None:
        """Number of choices (n) from metadata."""
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"n": 3},
        )
        body = self.mapper.to_dict(req)
        assert body["n"] == 3

    def test_default_n_omitted(self) -> None:
        """Default n=1 omitted from body."""
        req = ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
        body = self.mapper.to_dict(req)
        assert "n" not in body

    def test_tool_messages(self) -> None:
        """Tool calls in assistant and tool role messages."""
        req = ProviderRequest(messages=[
            ProviderMessage(
                role=Role.ASSISTANT,
                content="Let me check",
                tool_calls=[
                    ToolCallRequest(
                        id="call_1",
                        name="get_weather",
                        arguments={"city": "SF"},
                    ),
                ],
            ),
            ProviderMessage(
                role=Role.TOOL,
                content='{"temp": 72}',
                tool_call_id="call_1",
            ),
        ])
        body = self.mapper.to_dict(req)
        assert len(body["messages"]) == 2

        # Assistant message with tool_calls
        assert body["messages"][0]["role"] == "assistant"
        assert body["messages"][0]["tool_calls"][0]["id"] == "call_1"
        assert body["messages"][0]["tool_calls"][0]["function"]["name"] == "get_weather"
        assert body["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"city": "SF"}

        # Tool result message
        assert body["messages"][1]["role"] == "tool"
        assert body["messages"][1]["tool_call_id"] == "call_1"


class TestOpenAIResponseMapper:
    def setup_method(self) -> None:
        self.mapper = OpenAIResponseMapper()

    def test_basic_response(self) -> None:
        """Map a standard OpenAI Chat Completions response."""
        data = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-4",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                },
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        response = self.mapper.to_response(data)
        assert response.content == "Hello! How can I help?"
        assert response.stop_reason == StopReason.STOP
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    def test_length_stop_reason(self) -> None:
        """Max tokens / length finish reason."""
        data = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Partial"},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105},
        }
        response = self.mapper.to_response(data)
        assert response.stop_reason == StopReason.LENGTH
        assert response.usage.total_tokens == 105

    def test_max_tokens_stop_reason(self) -> None:
        """'max_tokens' finish_reason also maps to LENGTH."""
        data = {
            "id": "chatcmpl-789",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Partial"},
                "finish_reason": "max_tokens",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.stop_reason == StopReason.LENGTH

    def test_content_filter_stop_reason(self) -> None:
        """Content filter finish reason."""
        data = {
            "id": "chatcmpl-789",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "content_filter",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.stop_reason == StopReason.CONTENT_FILTER

    def test_tool_calls_stop_reason(self) -> None:
        """Tool calls finish reason."""
        data = {
            "id": "chatcmpl-101",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Let me look that up.",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "San Francisco"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.stop_reason == StopReason.TOOL_CALL
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "San Francisco"}
        assert response.tool_calls[0].id == "call_abc"

    def test_tool_calls_with_dict_arguments(self) -> None:
        """Tool calls where arguments is already a dict."""
        data = {
            "id": "chatcmpl-102",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_def",
                            "type": "function",
                            "function": {
                                "name": "search",
                                "arguments": {"q": "hello"},
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {"q": "hello"}

    def test_tool_calls_invalid_json_arguments(self) -> None:
        """Tool calls with unparseable JSON arguments string."""
        data = {
            "id": "chatcmpl-103",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_ghi",
                            "type": "function",
                            "function": {
                                "name": "broken",
                                "arguments": "not-json-at-all",
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].arguments == {"raw": "not-json-at-all"}

    def test_empty_content(self) -> None:
        """Empty content string."""
        data = {
            "id": "chatcmpl-empty",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.content == ""

    def test_none_content(self) -> None:
        """Null/None content handled gracefully."""
        data = {
            "id": "chatcmpl-none",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": None},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.content == ""

    def test_empty_choices(self) -> None:
        """No choices in response."""
        data = {
            "id": "chatcmpl-nope",
            "choices": [],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.content == ""
        assert response.stop_reason == StopReason.UNKNOWN

    def test_unknown_finish_reason(self) -> None:
        """Unknown finish reason mapped to UNKNOWN."""
        data = {
            "id": "chatcmpl-unk",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "hmm"},
                "finish_reason": "weird_reason",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.stop_reason == StopReason.UNKNOWN

    def test_null_finish_reason(self) -> None:
        """Null finish reason mapped to UNKNOWN."""
        reason = map_stop_reason(None)
        assert reason == StopReason.UNKNOWN

    def test_empty_finish_reason(self) -> None:
        """Empty finish reason mapped to UNKNOWN."""
        reason = map_stop_reason("")
        assert reason == StopReason.UNKNOWN

    def test_usage_empty(self) -> None:
        """Empty usage object defaults to zeros."""
        usage = self.mapper._map_usage({})
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_metadata_contains_id_and_model(self) -> None:
        """Response metadata includes id and model."""
        data = {
            "id": "chatcmpl-xyz",
            "model": "gpt-4o",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }],
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.metadata["id"] == "chatcmpl-xyz"
        assert response.metadata["model"] == "gpt-4o"
        assert response.metadata["raw_finish_reason"] == "stop"


class TestOpenAIStreamingMapper:
    def setup_method(self) -> None:
        self.mapper = OpenAIResponseMapper()

    def test_content_chunk(self) -> None:
        """Typical content delta chunk."""
        chunk = self.mapper.to_chunk({
            "choices": [{
                "index": 0,
                "delta": {"content": "Hello"},
            }],
        })
        assert chunk is not None
        assert chunk.content == "Hello"
        assert chunk.stop_reason is None
        assert chunk.index == 0

    def test_empty_delta(self) -> None:
        """Chunk with empty delta content."""
        chunk = self.mapper.to_chunk({
            "choices": [{
                "index": 0,
                "delta": {},
            }],
        })
        assert chunk is not None
        assert chunk.content == ""

    def test_finish_reason_chunk(self) -> None:
        """Chunk with finish_reason."""
        chunk = self.mapper.to_chunk({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        })
        assert chunk is not None
        assert chunk.stop_reason == StopReason.STOP

    def test_finish_reason_length(self) -> None:
        """Chunk with length finish_reason."""
        chunk = self.mapper.to_chunk({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "length",
            }],
        })
        assert chunk is not None
        assert chunk.stop_reason == StopReason.LENGTH

    def test_usage_chunk(self) -> None:
        """Final chunk with usage but no choices."""
        chunk = self.mapper.to_chunk({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        })
        assert chunk is not None
        assert chunk.content == ""
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 15

    def test_usage_chunk_zero_tokens(self) -> None:
        """Usage chunk with zero total tokens returns None."""
        chunk = self.mapper.to_chunk({
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        })
        assert chunk is None

    def test_empty_data(self) -> None:
        """Completely empty data with no choices or usage."""
        chunk = self.mapper.to_chunk({})
        assert chunk is None

    def test_tool_call_delta(self) -> None:
        """Streaming tool call delta chunk."""
        chunk = self.mapper.to_chunk({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "',
                            },
                        },
                    ],
                },
            }],
        })
        assert chunk is not None
        assert chunk.tool_call is not None
        assert chunk.tool_call.id == "call_abc"
        assert chunk.tool_call.name == "get_weather"
        assert chunk.tool_call.arguments == {"delta": '{"city": "'}

    def test_multiple_choices_uses_first(self) -> None:
        """Multiple choices uses first index."""
        chunk = self.mapper.to_chunk({
            "choices": [
                {"index": 0, "delta": {"content": "First"}},
                {"index": 1, "delta": {"content": "Second"}},
            ],
        })
        assert chunk is not None
        assert chunk.content == "First"
        assert chunk.index == 0

    def test_stop_reason_over_unknown(self) -> None:
        """'stop' finish_reason in streaming."""
        chunk = self.mapper.to_chunk({
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        })
        assert chunk is not None
        assert chunk.stop_reason == StopReason.STOP


class TestOpenAIStopReasonMap:
    def test_all_mapped_reasons(self) -> None:
        """Verify all standard OpenAI finish reasons map correctly."""
        assert map_stop_reason("stop") == StopReason.STOP
        assert map_stop_reason("length") == StopReason.LENGTH
        assert map_stop_reason("max_tokens") == StopReason.LENGTH
        assert map_stop_reason("tool_calls") == StopReason.TOOL_CALL
        assert map_stop_reason("content_filter") == StopReason.CONTENT_FILTER
        assert map_stop_reason("error") == StopReason.ERROR
        assert map_stop_reason("timeout") == StopReason.TIMEOUT
        assert map_stop_reason("cancelled") == StopReason.CANCELLED

    def test_unknown_reasons(self) -> None:
        """Unrecognized reasons map to UNKNOWN."""
        assert map_stop_reason("nobody_knows") == StopReason.UNKNOWN
        assert map_stop_reason("") == StopReason.UNKNOWN
        assert map_stop_reason(None) == StopReason.UNKNOWN
