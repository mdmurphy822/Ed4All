"""Follow-up — per-endpoint ``extra_body`` forwarding through the GENERAL
``OpenAICompatibleBackend``.

``config/endpoints.yaml`` carries an ``nvidia-deepseek`` row with
``extra_body: {chat_template_kwargs: {thinking: false}}`` so a REASONING
model suppresses chain-of-thought. SemantiK has its own
``SEMANTIK_SPECIALIST_DISABLE_THINKING`` path; these tests cover the
GENERAL Ed4All backend that every non-SemantiK consumer (Courseforge /
Trainforge synthesis providers, objective-review, the block planner)
reaches via the shared client.

Contract under test:

- Resolving / using the backend for ``nvidia-deepseek`` forwards the
  endpoint's ``extra_body`` keys verbatim into the request body sent to
  the underlying client (``chat_completion(extra_payload=...)``).
- An endpoint WITHOUT ``extra_body`` (e.g. ``local``) sends NO such key
  — byte-identical to the legacy body.
- A caller-supplied ``extra_payload`` MERGES with the endpoint
  ``extra_body``, with caller keys WINNING on conflict.
- Generic: the forwarding is keyed on the row carrying ``extra_body``,
  not hardcoded to ``nvidia-deepseek``.

Everything is mocked — NO network, NO live NVIDIA call.
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
    OpenAICompatibleBackend,
    resolve_openai_compatible_backend,
)


# ---------------------------------------------------------------------------
# Fake client — records the chat_completion args (incl. extra_payload)
# ---------------------------------------------------------------------------


class _FakeChatCompletion:
    """Records each ``chat_completion`` call so tests assert on the args."""

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


_THINKING_OFF = {"chat_template_kwargs": {"thinking": False}}


# ---------------------------------------------------------------------------
# (a) nvidia-deepseek → request body carries chat_template_kwargs
# ---------------------------------------------------------------------------


def test_nvidia_deepseek_backend_forwards_extra_body(monkeypatch):
    """The resolved ``nvidia-deepseek`` backend forwards thinking-off."""

    monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
    backend = resolve_openai_compatible_backend("nvidia-deepseek")
    assert backend.extra_body == _THINKING_OFF

    fake = _wire_fake_client(backend)
    backend.complete_sync("sys", "user")

    assert len(fake.calls) == 1
    sent = fake.calls[0]["extra_payload"]
    assert sent is not None
    assert sent["chat_template_kwargs"] == {"thinking": False}


def test_direct_backend_with_extra_body_forwards_it():
    """Constructing the backend directly with ``extra_body`` forwards it.

    Generic path — nothing keys on the provider NAME; ANY backend
    carrying ``extra_body`` forwards it.
    """

    backend = OpenAICompatibleBackend(
        provider_name="any-reasoning-endpoint",
        base_url="https://example.invalid/v1",
        default_model="some-reasoning-model",
        api_key="sk-test",
        extra_body=_THINKING_OFF,
    )
    fake = _wire_fake_client(backend)
    backend.complete_sync("sys", "user")

    assert fake.calls[0]["extra_payload"] == _THINKING_OFF


# ---------------------------------------------------------------------------
# (b) endpoint WITHOUT extra_body → no key (byte-stable)
# ---------------------------------------------------------------------------


def test_local_backend_without_extra_body_sends_none(monkeypatch):
    """An endpoint without ``extra_body`` forwards ``None`` (byte-stable)."""

    monkeypatch.delenv("LOCAL_VISION_CAPABLE", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    backend = resolve_openai_compatible_backend("local")
    assert backend.extra_body is None

    fake = _wire_fake_client(backend)
    backend.complete_sync("sys", "user")

    # No extra_payload key was synthesized — the request body is the
    # legacy shape (model / messages / temperature / max_tokens only).
    assert fake.calls[0]["extra_payload"] is None


def test_direct_backend_without_extra_body_sends_none():
    backend = OpenAICompatibleBackend(
        provider_name="local",
        base_url="http://localhost:11434/v1",
        default_model="qwen2.5",
    )
    assert backend.extra_body is None
    fake = _wire_fake_client(backend)
    backend.complete_sync("sys", "user")
    assert fake.calls[0]["extra_payload"] is None


def test_empty_extra_body_dict_normalizes_to_none():
    """An empty ``extra_body`` mapping → ``None`` (no empty key in body)."""

    backend = OpenAICompatibleBackend(
        provider_name="x",
        base_url="https://example.invalid/v1",
        default_model="m",
        api_key="sk-test",
        extra_body={},
    )
    assert backend.extra_body is None
    fake = _wire_fake_client(backend)
    backend.complete_sync("sys", "user")
    assert fake.calls[0]["extra_payload"] is None


# ---------------------------------------------------------------------------
# (c) caller-supplied extra_payload merges; caller wins on conflict
# ---------------------------------------------------------------------------


def test_caller_extra_payload_merges_with_endpoint_extra_body():
    """A caller's ``extra_payload`` MERGES with the endpoint ``extra_body``."""

    backend = OpenAICompatibleBackend(
        provider_name="any-reasoning-endpoint",
        base_url="https://example.invalid/v1",
        default_model="m",
        api_key="sk-test",
        extra_body=_THINKING_OFF,
    )
    fake = _wire_fake_client(backend)
    backend.complete_sync(
        "sys", "user", extra_payload={"top_p": 0.9}
    )

    sent = fake.calls[0]["extra_payload"]
    # Endpoint default present...
    assert sent["chat_template_kwargs"] == {"thinking": False}
    # ...AND the caller's own key.
    assert sent["top_p"] == 0.9


def test_caller_extra_payload_wins_on_key_conflict():
    """On a key conflict the CALLER's value overrides the endpoint default."""

    backend = OpenAICompatibleBackend(
        provider_name="any-reasoning-endpoint",
        base_url="https://example.invalid/v1",
        default_model="m",
        api_key="sk-test",
        extra_body=_THINKING_OFF,
    )
    fake = _wire_fake_client(backend)
    backend.complete_sync(
        "sys",
        "user",
        extra_payload={"chat_template_kwargs": {"thinking": True}},
    )

    sent = fake.calls[0]["extra_payload"]
    # Caller wins — thinking is re-enabled for this call.
    assert sent["chat_template_kwargs"] == {"thinking": True}


def test_caller_extra_payload_without_endpoint_extra_body():
    """Caller ``extra_payload`` alone forwards when the endpoint has none."""

    backend = OpenAICompatibleBackend(
        provider_name="local",
        base_url="http://localhost:11434/v1",
        default_model="m",
    )
    assert backend.extra_body is None
    fake = _wire_fake_client(backend)
    backend.complete_sync("sys", "user", extra_payload={"stop": ["</s>"]})
    assert fake.calls[0]["extra_payload"] == {"stop": ["</s>"]}


# ---------------------------------------------------------------------------
# resolve_endpoint surfaces extra_body on the ResolvedEndpoint dataclass
# ---------------------------------------------------------------------------


def test_resolve_endpoint_surfaces_extra_body(monkeypatch):
    from lib.llm.endpoints import resolve_endpoint

    monkeypatch.setenv("NVIDIA_API_KEY", "sk-test")
    resolved = resolve_endpoint("nvidia-deepseek")
    assert resolved.extra_body == _THINKING_OFF

    resolved_local = resolve_endpoint("local")
    assert resolved_local.extra_body is None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
