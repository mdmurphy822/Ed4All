"""Wave W-D13 T13.2 — vision-mode regression tests for the
``OpenAICompatibleBackend`` and the registry resolver.

The W-D12 backend raised ``NotImplementedError`` on every ``images=``
call; W-D13 promotes vision to a per-provider opt-in driven by the
``vision_capable`` registry flag. These tests exercise the new
contract:

- A vision-capable backend forwards ``images=`` through to the
  underlying ``OpenAICompatibleClient.chat_completion`` (which
  translates to OpenAI's content-block shape on the wire).
- A non-vision-capable backend raises ``RuntimeError`` BEFORE the
  HTTP round-trip with a message pointing at every escape hatch.
- The registry seeds ``together-vision`` with ``vision_capable=True``;
  ``local`` and ``together`` (text) default to ``False``.
- ``LOCAL_VISION_CAPABLE=true`` flips the resolved local-entry
  backend's flag without editing the registry source file.
- A ``LOCAL_SYNTHESIS_MODEL`` containing ``vision`` / ``llava`` /
  ``-vl`` substring also flips the flag (heuristic for operators
  who set the model identifier without remembering the env hatch).
- The ``provider="anthropic"`` vision path is unchanged — that's the
  ``AnthropicBackend._build_messages`` content-block shape, exercised
  by the broader suite.

Capture-rationale interpolation of the dynamic provider name is
covered separately in ``lib/tests/test_llm_backend.py`` — these tests
focus on vision-payload routing and registry resolution.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.orchestrator.llm_backend import (  # noqa: E402
    _OPENAI_COMPATIBLE_PROVIDERS,
    OpenAICompatibleBackend,
    resolve_openai_compatible_backend,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeChatCompletion:
    """Records each ``chat_completion`` call so tests assert on the args.

    Mirrors the shape used by ``lib/tests/test_llm_backend.py`` so the
    fake stays interchangeable across test files.
    """

    def __init__(self, response_text: str = "ok"):
        self.response_text = response_text
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int,
        temperature: float,
        decision_metadata: Dict[str, Any] | None = None,
        extra_payload: Dict[str, Any] | None = None,
        images: List[Dict[str, Any]] | None = None,
    ) -> str:
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "decision_metadata": decision_metadata,
                "extra_payload": extra_payload,
                "images": images,
            }
        )
        return self.response_text


def _wire_fake_client(
    backend: OpenAICompatibleBackend, response_text: str = "ok"
) -> _FakeChatCompletion:
    fake = _FakeChatCompletion(response_text=response_text)
    stub = MagicMock()
    stub.chat_completion = fake
    stub.model = backend.default_model
    stub._model = backend.default_model
    backend._client = stub
    return fake


# ---------------------------------------------------------------------------
# Constructor flag
# ---------------------------------------------------------------------------


def test_default_vision_capable_is_false():
    """W-D13 default: backends are NOT vision-capable unless flagged."""

    backend = OpenAICompatibleBackend(
        provider_name="local",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5",
    )
    assert backend.vision_capable is False


def test_explicit_vision_capable_true_round_trips():
    backend = OpenAICompatibleBackend(
        provider_name="together-vision",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        api_key="sk-test",
        vision_capable=True,
    )
    assert backend.vision_capable is True


# ---------------------------------------------------------------------------
# Vision-capable happy path
# ---------------------------------------------------------------------------


def test_vision_capable_backend_forwards_images_to_client():
    """W-D13: vision-capable backend passes images= through verbatim."""

    backend = OpenAICompatibleBackend(
        provider_name="together-vision",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        api_key="sk-test",
        vision_capable=True,
    )
    fake = _wire_fake_client(backend, response_text="An image of a cat.")

    out = backend.complete_sync(
        "describe images",
        "what is in this image?",
        images=[{"media_type": "image/jpeg", "data": "AAA="}],
    )
    assert out == "An image of a cat."
    assert len(fake.calls) == 1
    # The backend forwards the images list verbatim — the underlying
    # client owns the content-block translation.
    assert fake.calls[0]["images"] == [
        {"media_type": "image/jpeg", "data": "AAA="}
    ]


def test_vision_capable_backend_no_images_keeps_text_only_path():
    """vision_capable=True but no images= → still text-only on the wire."""

    backend = OpenAICompatibleBackend(
        provider_name="together-vision",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        api_key="sk-test",
        vision_capable=True,
    )
    fake = _wire_fake_client(backend)
    backend.complete_sync("sys", "user")
    assert fake.calls[0]["images"] is None


# ---------------------------------------------------------------------------
# Non-vision-capable failure path
# ---------------------------------------------------------------------------


def test_non_vision_capable_backend_raises_runtime_error_on_images():
    """W-D13: ``images=`` against vision_capable=False raises RuntimeError.

    The error message must point at every escape hatch so an operator
    sees the fix inline (registry flag, vision_capable_env, sibling
    vision providers).
    """

    backend = OpenAICompatibleBackend(
        provider_name="local",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5",
        vision_capable=False,
    )
    # Wire the fake so the test fails LOUDLY if the gate check is
    # missing and the call reaches the wire instead of raising.
    _wire_fake_client(backend)

    with pytest.raises(RuntimeError) as excinfo:
        backend.complete_sync(
            "sys",
            "user",
            images=[{"media_type": "image/png", "data": "AAA="}],
        )
    # The error message points at every escape hatch.
    msg = str(excinfo.value)
    assert "local" in msg
    assert "vision_capable" in msg


def test_non_vision_capable_backend_text_calls_unaffected():
    """Non-vision backend with no images= still works — gate is image-only."""

    backend = OpenAICompatibleBackend(
        provider_name="local",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5",
        vision_capable=False,
    )
    fake = _wire_fake_client(backend, response_text="text reply")
    out = backend.complete_sync("sys", "user")
    assert out == "text reply"
    assert fake.calls[0]["images"] is None


# ---------------------------------------------------------------------------
# Registry resolver — vision flag
# ---------------------------------------------------------------------------


def test_registry_seeds_local_with_vision_capable_false():
    """Default ``local`` is text-only — Qwen 14B isn't a vision model."""

    assert _OPENAI_COMPATIBLE_PROVIDERS["local"]["vision_capable"] is False


def test_registry_seeds_together_with_vision_capable_false():
    """``together`` text endpoint stays vision-incapable.

    The text default (Llama-3.3-70B) is not a vision model; flipping
    the entry's flag would silently route every Together text call
    through a vision-capable model. The sibling ``together-vision``
    entry is the explicit opt-in path.
    """

    assert _OPENAI_COMPATIBLE_PROVIDERS["together"]["vision_capable"] is False


def test_registry_seeds_together_vision_with_vision_capable_true():
    entry = _OPENAI_COMPATIBLE_PROVIDERS["together-vision"]
    assert entry["vision_capable"] is True
    assert entry["api_key_env"] == "TOGETHER_API_KEY"
    assert "Vision" in entry["model_default"]


def test_resolver_flips_vision_capable_via_env_var(monkeypatch):
    """``LOCAL_VISION_CAPABLE=true`` flips the resolved backend's flag."""

    monkeypatch.setenv("LOCAL_VISION_CAPABLE", "true")
    # Avoid env coupling on the local synthesis vars.
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "llama3.2-vision")
    backend = resolve_openai_compatible_backend("local")
    assert backend.vision_capable is True


def test_resolver_treats_empty_env_as_off(monkeypatch):
    """Empty ``LOCAL_VISION_CAPABLE`` falls through to the registry default."""

    monkeypatch.setenv("LOCAL_VISION_CAPABLE", "")
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    backend = resolve_openai_compatible_backend("local")
    assert backend.vision_capable is False


def test_resolver_heuristic_flips_on_vision_substring_in_model(monkeypatch):
    """A vision-substring model flips the flag without explicit env opt-in."""

    monkeypatch.delenv("LOCAL_VISION_CAPABLE", raising=False)
    monkeypatch.setenv(
        "LOCAL_SYNTHESIS_MODEL", "qwen2.5-vl:7b-instruct-q4_K_M"
    )
    backend = resolve_openai_compatible_backend("local")
    assert backend.vision_capable is True


def test_resolver_heuristic_flips_on_llava_substring(monkeypatch):
    monkeypatch.delenv("LOCAL_VISION_CAPABLE", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "llava:13b")
    backend = resolve_openai_compatible_backend("local")
    assert backend.vision_capable is True


def test_resolver_text_model_keeps_default(monkeypatch):
    """Plain text model name + no env override → vision_capable=False."""

    monkeypatch.delenv("LOCAL_VISION_CAPABLE", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    backend = resolve_openai_compatible_backend("local")
    assert backend.vision_capable is False


def test_resolver_together_vision_resolves_with_vision_flag(monkeypatch):
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")
    backend = resolve_openai_compatible_backend("together-vision")
    assert backend.vision_capable is True
    assert backend.provider_name == "together-vision"


# ---------------------------------------------------------------------------
# Anthropic vision path remains unchanged
# ---------------------------------------------------------------------------


def test_anthropic_backend_vision_path_unaffected():
    """W-D13 changes don't perturb AnthropicBackend's vision content-block.

    AnthropicBackend builds its own ``image`` blocks via
    ``_build_messages`` (Anthropic's vision shape, not OpenAI's). Make
    sure the existing import surface still works — the sibling test
    suite (``lib/tests/test_llm_backend.py``) carries the full
    behavioural assertion.
    """

    from MCP.orchestrator.llm_backend import AnthropicBackend

    backend = AnthropicBackend(api_key="sk-test")
    msgs = backend._build_messages(
        "describe",
        images=[{"media_type": "image/jpeg", "data": "AAA="}],
    )
    # Anthropic's content-block shape, not OpenAI's image_url shape.
    assert msgs[0]["content"][0]["type"] == "image"
    assert msgs[0]["content"][0]["source"]["type"] == "base64"
