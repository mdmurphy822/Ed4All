"""Tests for the LLMBackend abstraction (Wave 7)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from MCP.orchestrator.llm_backend import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicBackend,
    BackendSpec,
    LLMBackend,
    LocalBackend,
    MockBackend,
    OpenAIBackend,
    build_backend,
)


class TestProtocolConformance:
    def test_mock_backend_is_llm_backend(self):
        backend = MockBackend(responses=["x"])
        assert isinstance(backend, LLMBackend)

    def test_local_backend_is_llm_backend(self):
        backend = LocalBackend()
        assert isinstance(backend, LLMBackend)

    def test_openai_backend_is_llm_backend(self):
        backend = OpenAIBackend(api_key="key")
        assert isinstance(backend, LLMBackend)

    def test_anthropic_backend_is_llm_backend(self):
        backend = AnthropicBackend(api_key="key")
        assert isinstance(backend, LLMBackend)


class TestMockBackend:
    def test_fifo_responses(self):
        backend = MockBackend(responses=["a", "b", "c"])
        assert backend.complete_sync("sys", "u1") == "a"
        assert backend.complete_sync("sys", "u2") == "b"
        assert backend.complete_sync("sys", "u3") == "c"

    def test_records_calls(self):
        backend = MockBackend(responses=["ok"])
        backend.complete_sync("sys", "hello", model="m1", max_tokens=123)
        assert len(backend.calls) == 1
        call = backend.calls[0]
        assert call.system == "sys"
        assert call.user == "hello"
        assert call.model == "m1"
        assert call.max_tokens == 123

    @pytest.mark.asyncio
    async def test_complete_async(self):
        backend = MockBackend(responses=["async-ok"])
        result = await backend.complete("sys", "user")
        assert result == "async-ok"

    @pytest.mark.asyncio
    async def test_streaming_rejected(self):
        backend = MockBackend(responses=["x"])
        with pytest.raises(NotImplementedError):
            await backend.complete("sys", "user", stream=True)

    def test_default_response(self):
        backend = MockBackend(default_response="fallback")
        assert backend.complete_sync("sys", "u") == "fallback"

    def test_response_fn(self):
        backend = MockBackend(response_fn=lambda s, u: f"echo:{u}")
        assert backend.complete_sync("sys", "hello") == "echo:hello"

    def test_fixture_dir(self, tmp_path: Path):
        # Write a fixture keyed by hash of "sys\nuser"
        key = MockBackend._fixture_key("sys", "user")
        (tmp_path / f"{key}.json").write_text(json.dumps({"text": "fx-response"}))
        backend = MockBackend(fixture_dir=tmp_path)
        assert backend.complete_sync("sys", "user") == "fx-response"

    def test_fixture_miss_falls_back(self, tmp_path: Path):
        backend = MockBackend(fixture_dir=tmp_path, default_response="def")
        # No fixture file => should return default_response
        assert backend.complete_sync("sys", "nope") == "def"


class TestLocalBackend:
    def test_complete_sync_raises(self):
        backend = LocalBackend()
        with pytest.raises(NotImplementedError, match="LocalDispatcher"):
            backend.complete_sync("sys", "user")

    @pytest.mark.asyncio
    async def test_complete_async_raises(self):
        backend = LocalBackend()
        with pytest.raises(NotImplementedError):
            await backend.complete("sys", "user")


class TestOpenAIBackend:
    def test_stub_raises(self):
        backend = OpenAIBackend(api_key="k")
        with pytest.raises(NotImplementedError, match="stub"):
            backend.complete_sync("sys", "user")


class TestAnthropicBackend:
    def test_requires_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="API key"):
            AnthropicBackend()

    def test_default_model(self):
        backend = AnthropicBackend(api_key="k")
        assert backend.default_model == DEFAULT_ANTHROPIC_MODEL

    @pytest.mark.asyncio
    async def test_stream_not_implemented(self):
        backend = AnthropicBackend(api_key="k")
        with pytest.raises(NotImplementedError, match="O3"):
            await backend.complete("sys", "u", stream=True)

    def test_complete_sync_invokes_sdk(self):
        """Wire a mocked anthropic client and ensure the call shape is right."""
        backend = AnthropicBackend(api_key="k", default_model="test-model")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_content = MagicMock()
        mock_content.text = "sdk-output"
        mock_response.content = [mock_content]
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client  # inject

        result = backend.complete_sync(
            system="sys",
            user="hello",
            max_tokens=100,
            temperature=0.5,
        )
        assert result == "sdk-output"

        # Verify the SDK was called with the expected shape
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["system"] == "sys"
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "hello"

    def test_complete_sync_with_images(self):
        """Image blocks are attached correctly for vision calls."""
        backend = AnthropicBackend(api_key="k")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="alt text")]
        mock_client.messages.create.return_value = mock_response
        backend._client = mock_client

        result = backend.complete_sync(
            system="",
            user="describe",
            images=[{"media_type": "image/png", "data": "b64data"}],
        )
        assert result == "alt text"

        call_kwargs = mock_client.messages.create.call_args.kwargs
        content = call_kwargs["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "image"
        assert content[0]["source"]["media_type"] == "image/png"
        assert content[1]["type"] == "text"
        assert content[1]["text"] == "describe"


class TestBuildBackend:
    def test_local_mode_default(self, monkeypatch):
        monkeypatch.delenv("LLM_MODE", raising=False)
        backend = build_backend()
        assert isinstance(backend, LocalBackend)

    def test_api_mode_anthropic(self):
        spec = BackendSpec(mode="api", provider="anthropic")
        backend = build_backend(spec, api_key="test-key")
        assert isinstance(backend, AnthropicBackend)

    def test_api_mode_mock(self):
        spec = BackendSpec(mode="api", provider="mock", mock_responses=["x"])
        backend = build_backend(spec)
        assert isinstance(backend, MockBackend)

    def test_api_mode_openai_stub(self):
        spec = BackendSpec(mode="api", provider="openai")
        backend = build_backend(spec, api_key="k")
        assert isinstance(backend, OpenAIBackend)

    def test_unknown_provider_raises(self):
        spec = BackendSpec(mode="api", provider="nonsense")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            build_backend(spec)

    def test_env_vars_picked_up(self, monkeypatch):
        monkeypatch.setenv("LLM_MODE", "local")
        backend = build_backend()
        assert isinstance(backend, LocalBackend)
