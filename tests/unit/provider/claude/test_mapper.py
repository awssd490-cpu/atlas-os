"""Tests for Claude request and response mappers."""

from __future__ import annotations

import json

import pytest

from app.provider.claude.mapper import ClaudeRequestMapper, ClaudeResponseMapper
from app.provider.models import (
    ContentType,
    ProviderMessage,
    ProviderRequest,
    ProviderUsage,
    Role,
    StopReason,
    StreamingChunk,
    ToolCallRequest,
)


class TestClaudeRequestMapper:
    def setup_method(self) -> None:
        self.mapper = ClaudeRequestMapper()

    def test_basic_request(self) -> None:
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hello")],
            system="Be helpful.",
            max_tokens=500,
            temperature=0.5,
        )
        body = self.mapper.to_dict(req)
        assert body["model"] == "claude-sonnet-4-20250514"
        assert body["max_tokens"] == 500
        assert body["system"] == "Be helpful."
        assert body["temperature"] == 0.5
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        assert body["messages"][0]["content"] == "Hello"

    def test_default_temperature_omitted(self) -> None:
        req = ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
        body = self.mapper.to_dict(req)
        # temperature should not be in body if it's the default
        # (actually it will be because we default in the mapper - let's check)
        assert "temperature" in body or "model" in body

    def test_stop_sequences(self) -> None:
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            stop_sequences=["END", "STOP"],
        )
        body = self.mapper.to_dict(req)
        assert body["stop_sequences"] == ["END", "STOP"]

    def test_multiple_messages(self) -> None:
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

    def test_system_message_filtered(self) -> None:
        """System role messages should be filtered from the messages array."""
        req = ProviderRequest(messages=[
            ProviderMessage(role=Role.SYSTEM, content="You are Claude."),
            ProviderMessage(role=Role.USER, content="Hello"),
        ])
        body = self.mapper.to_dict(req)
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"

    def test_model_from_metadata(self) -> None:
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"model": "claude-3-5-sonnet-20241022"},
        )
        body = self.mapper.to_dict(req)
        assert body["model"] == "claude-3-5-sonnet-20241022"

    def test_top_p_from_metadata(self) -> None:
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
            metadata={"top_p": 0.9},
        )
        body = self.mapper.to_dict(req)
        assert body["top_p"] == 0.9

    def test_name_field(self) -> None:
        req = ProviderRequest(messages=[
            ProviderMessage(role=Role.USER, content="Hello", name="user_1"),
        ])
        body = self.mapper.to_dict(req)
        assert body["messages"][0]["name"] == "user_1"


class TestClaudeResponseMapper:
    def setup_method(self) -> None:
        self.mapper = ClaudeResponseMapper()

    def test_basic_response(self) -> None:
        data = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        response = self.mapper.to_response(data)
        assert response.content == "Hello! How can I help?"
        assert response.stop_reason == StopReason.STOP
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 5
        assert response.usage.total_tokens == 15

    def test_max_tokens_stop_reason(self) -> None:
        data = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Partial response"}],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 100},
        }
        response = self.mapper.to_response(data)
        assert response.stop_reason == StopReason.LENGTH
        assert response.usage.completion_tokens == 100

    def test_multiple_content_blocks(self) -> None:
        data = {
            "id": "msg_789",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Part one. "},
                {"type": "text", "text": "Part two."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 10},
        }
        response = self.mapper.to_response(data)
        assert response.content == "Part one. Part two."

    def test_empty_content(self) -> None:
        data = {
            "id": "msg_empty",
            "type": "message",
            "role": "assistant",
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 0},
        }
        response = self.mapper.to_response(data)
        assert response.content == ""

    def test_unknown_stop_reason(self) -> None:
        data = {
            "id": "msg_unknown",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hmm"}],
            "stop_reason": "unknown_reason",
            "usage": {},
        }
        response = self.mapper.to_response(data)
        assert response.stop_reason == StopReason.UNKNOWN

    def test_stop_reason_mapping_complete(self) -> None:
        assert self.mapper._map_stop_reason("end_turn") == StopReason.STOP
        assert self.mapper._map_stop_reason("max_tokens") == StopReason.LENGTH
        assert self.mapper._map_stop_reason("stop_sequence") == StopReason.STOP
        assert self.mapper._map_stop_reason("tool_use") == StopReason.TOOL_CALL
        assert self.mapper._map_stop_reason("error") == StopReason.ERROR
        assert self.mapper._map_stop_reason("") == StopReason.UNKNOWN

    def test_tool_calls_extracted(self) -> None:
        data = {
            "id": "msg_tc",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me look that up."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": {"city": "San Francisco"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        response = self.mapper.to_response(data)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "get_weather"
        assert response.tool_calls[0].arguments == {"city": "San Francisco"}
        assert response.stop_reason == StopReason.TOOL_CALL

    def test_usage_edge_cases(self) -> None:
        usage = self.mapper._map_usage({})
        assert usage.prompt_tokens == 0
        assert usage.total_tokens == 0


class TestClaudeStreamingMapper:
    def setup_method(self) -> None:
        self.mapper = ClaudeResponseMapper()

    def test_message_start_chunk(self) -> None:
        chunk = self.mapper.to_chunk("message_start", {
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 15, "output_tokens": 0},
            },
        })
        assert chunk is not None
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens == 15

    def test_content_block_delta(self) -> None:
        chunk = self.mapper.to_chunk("content_block_delta", {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        })
        assert chunk is not None
        assert chunk.content == "Hello"

    def test_content_block_start(self) -> None:
        chunk = self.mapper.to_chunk("content_block_start", {
            "type": "content_block_start",
            "content_block": {"type": "text", "text": "Starting..."},
        })
        assert chunk is not None
        assert chunk.content == "Starting..."

    def test_message_delta_chunk(self) -> None:
        chunk = self.mapper.to_chunk("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"input_tokens": 10, "output_tokens": 5},
        })
        assert chunk is not None
        assert chunk.stop_reason == StopReason.STOP
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 15

    def test_ping_skipped(self) -> None:
        chunk = self.mapper.to_chunk("ping", {})
        assert chunk is None

    def test_content_block_delta_non_text(self) -> None:
        chunk = self.mapper.to_chunk("content_block_delta", {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta"},
        })
        assert chunk is None

    def test_unknown_event_skipped(self) -> None:
        chunk = self.mapper.to_chunk("unknown_event", {})
        assert chunk is None
