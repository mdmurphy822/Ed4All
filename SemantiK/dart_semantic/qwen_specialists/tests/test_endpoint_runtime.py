"""Unit + live-smoke tests for the hosted-endpoint specialist runtime.

CPU-pinned, no GPU, no local GGUF. The unit tests MOCK ``requests.post``;
the optional live smoke fires only when ``NVIDIA_API_KEY`` is in the env
(sourced by the caller — the key value is NEVER asserted on or printed).

Run:
  ED4ALL_NLI_DEVICE=cpu ED4ALL_EMBEDDING_DEVICE=cpu \
    python -m pytest dart_semantic/qwen_specialists/tests/test_endpoint_runtime.py -q
"""

from __future__ import annotations

import os
from unittest import mock

import pytest

from dart_semantic.qwen_specialists.endpoint_runtime import (
    EndpointRuntimeError,
    OpenAICompatibleRuntime,
    split_specialist_prompt,
)
from dart_semantic.qwen_specialists.runtime import (
    MockRuntime,
    make_runtime,
    resolve_specialist_provider,
    specialist_provider_is_endpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json_body = json_body
        self.text = text

    def json(self):
        if self._json_body is None:
            raise ValueError("no json body")
        return self._json_body


def _ok_body(content: str = "<math><mi>x</mi></math>") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


# ---------------------------------------------------------------------------
# generate() — happy path (mocked requests.post)
# ---------------------------------------------------------------------------


def test_generate_posts_to_chat_completions_and_returns_text():
    rt = OpenAICompatibleRuntime(
        base_url="https://endpoint.example/v1",
        api_key="UNIT-TEST-KEY",
        model="meta/llama-3.3-70b-instruct",
        timeout=30.0,
    )
    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(200, _ok_body("<math><mi>y</mi></math>"))

    # The runtime lazy-imports ``requests`` inside the method, so patch the
    # real module's ``post``.
    with mock.patch("requests.post", side_effect=_fake_post):
        out = rt.generate(
            "SYSTEM: math role\nUSER: {\"kind\": \"math\", \"src_text\": \"x\"}",
            n=2,
            temperature=0.6,
            top_p=0.95,
            max_tokens=256,
        )

    # n completions returned.
    assert out == ["<math><mi>y</mi></math>", "<math><mi>y</mi></math>"]
    # URL is base + /chat/completions.
    assert captured["url"] == "https://endpoint.example/v1/chat/completions"
    # Model + messages in payload; the region text reached the user turn.
    assert captured["json"]["model"] == "meta/llama-3.3-70b-instruct"
    msgs = captured["json"]["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "src_text" in msgs[1]["content"]
    # Bearer header PRESENT — never assert the key VALUE here.
    assert "Authorization" in captured["headers"]
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert captured["timeout"] == 30.0


def test_generate_one_post_per_candidate_with_seed_offset():
    rt = OpenAICompatibleRuntime(
        base_url="https://x/v1", api_key="K", model="m", timeout=10.0
    )
    seeds = []

    def _fake_post(url, *, json, headers, timeout):
        seeds_local = json.get("seed")
        seeds.append(seeds_local)
        return _FakeResponse(200, _ok_body("frag"))

    with mock.patch("requests.post", side_effect=_fake_post):
        out = rt.generate("SYSTEM: r\nUSER: {}", n=3, temperature=0.6, top_p=0.95, max_tokens=64, seed=100)

    assert len(out) == 3
    # one POST per candidate, seed offset by candidate index.
    assert seeds == [100, 101, 102]


# ---------------------------------------------------------------------------
# generate() — fail-loud arms
# ---------------------------------------------------------------------------


def test_missing_api_key_raises_clear_error(monkeypatch):
    # No explicit key, and no env key.
    monkeypatch.delenv("SEMANTIK_SPECIALIST_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", model="m")
    with pytest.raises(EndpointRuntimeError) as exc:
        rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)
    assert "api key" in str(exc.value).lower()


def test_missing_base_url_raises_clear_error(monkeypatch):
    # Clear env so the explicit empty base_url can't be backfilled from
    # NVIDIA_BASE_URL when the live key is sourced into the env.
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BASE_URL", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    rt = OpenAICompatibleRuntime(base_url="", api_key="K", model="m")
    with pytest.raises(EndpointRuntimeError) as exc:
        rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)
    assert "base url" in str(exc.value).lower()


def test_non_200_raises():
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    def _fake_post(url, *, json, headers, timeout):
        return _FakeResponse(503, json_body=None, text="upstream unavailable")

    with mock.patch("requests.post", side_effect=_fake_post):
        with pytest.raises(EndpointRuntimeError) as exc:
            rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)
    assert "503" in str(exc.value)


def test_timeout_raises():
    import requests

    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    def _fake_post(url, *, json, headers, timeout):
        raise requests.exceptions.Timeout("timed out")

    with mock.patch("requests.post", side_effect=_fake_post):
        with pytest.raises(EndpointRuntimeError) as exc:
            rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)
    assert "failed" in str(exc.value).lower()


def test_malformed_json_raises():
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    def _fake_post(url, *, json, headers, timeout):
        return _FakeResponse(200, json_body={"unexpected": "shape"})

    with mock.patch("requests.post", side_effect=_fake_post):
        with pytest.raises(EndpointRuntimeError) as exc:
            rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)
    assert "malformed" in str(exc.value).lower()


# ---------------------------------------------------------------------------
# load / free are no-ops
# ---------------------------------------------------------------------------


def test_load_free_are_noops():
    from pathlib import Path

    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")
    rt.load(Path("/does/not/matter.gguf"))  # must NOT raise / bind weights
    rt.free()
    assert rt.load_calls == [Path("/does/not/matter.gguf")]
    assert rt.free_calls == 1


# ---------------------------------------------------------------------------
# Prompt splitting / envelope
# ---------------------------------------------------------------------------


def test_split_specialist_prompt_extracts_system_and_user():
    system, user = split_specialist_prompt(
        'SYSTEM: You convert math.\nUSER: {"kind": "math"}'
    )
    assert "Specialist role: You convert math." in system
    assert user == '{"kind": "math"}'
    # Envelope directive is always present so the 70B honours the contract.
    assert "fragment ONLY" in system


def test_split_specialist_prompt_passthrough_on_unexpected_shape():
    system, user = split_specialist_prompt("just some text")
    assert user == "just some text"
    assert "fragment ONLY" in system


# ---------------------------------------------------------------------------
# Provider selector + make_runtime
# ---------------------------------------------------------------------------


def test_provider_local_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    assert resolve_specialist_provider() == "local"
    assert specialist_provider_is_endpoint() is False


@pytest.mark.parametrize("val", ["local", "LOCAL", "", "gguf"])
def test_provider_local_aliases(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", val)
    assert resolve_specialist_provider() == "local"
    assert specialist_provider_is_endpoint() is False


@pytest.mark.parametrize("val", ["nvidia", "local-openai", "endpoint", "vllm"])
def test_provider_endpoint_values(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", val)
    assert resolve_specialist_provider() == val.lower()
    assert specialist_provider_is_endpoint() is True


def test_make_runtime_mock_unaffected(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    # mock stays mock even when an endpoint provider is set.
    assert isinstance(make_runtime("mock"), MockRuntime)


def test_make_runtime_real_routes_to_endpoint_when_provider_set(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    # No NVIDIA_API_KEY needed at construction — only at generate().
    rt = make_runtime("real")
    assert isinstance(rt, OpenAICompatibleRuntime)


def test_make_runtime_explicit_endpoint_mode(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    rt = make_runtime("endpoint")
    assert isinstance(rt, OpenAICompatibleRuntime)


def test_make_runtime_real_local_still_strict(monkeypatch):
    # Local arm (no endpoint provider) keeps the strict GGUF check — with
    # the v1 null-adapter config it must raise rather than silently mock.
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    with pytest.raises(RuntimeError) as exc:
        make_runtime("real")
    assert "no Qwen LoRA adapters" in str(exc.value) or "MISSING" in str(exc.value)


# ---------------------------------------------------------------------------
# runtime_mode='real' under endpoint mode (cascade derivation parity)
# ---------------------------------------------------------------------------


def test_cascade_runtime_mode_real_under_endpoint(monkeypatch):
    # The cascade derives runtime_mode from the specialist provider: an
    # endpoint provider forces "real" even with no local adapter loaded.
    from dart_semantic.v2_config import V2Config

    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    cfg = V2Config()  # no local qwen adapters configured
    assert cfg.is_qwen_specialist_loaded() is False
    derived = "real" if (specialist_provider_is_endpoint() or cfg.is_qwen_specialist_loaded()) else "mock"
    assert derived == "real"


def test_cascade_runtime_mode_mock_without_endpoint(monkeypatch):
    from dart_semantic.v2_config import V2Config

    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    cfg = V2Config()
    derived = "real" if (specialist_provider_is_endpoint() or cfg.is_qwen_specialist_loaded()) else "mock"
    assert derived == "mock"


# ---------------------------------------------------------------------------
# OPTIONAL live 70B smoke — only when NVIDIA_API_KEY is present in env.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("NVIDIA_API_KEY") or not os.environ.get("NVIDIA_BASE_URL"),
    reason="NVIDIA_API_KEY / NVIDIA_BASE_URL not set; skipping live 70B smoke",
)
def test_live_70b_math_smoke():
    """Single live generate() against the real endpoint. NEVER prints the
    key. Asserts a non-empty MathML-ish response for a tiny math prompt."""
    rt = make_runtime("endpoint")
    out = rt.generate(
        'SYSTEM: You convert one math expression into MathML 4.0 with an '
        'alttext attribute.\nUSER: {"kind": "math", "src_text": "x^2 + 1", '
        '"display": false}',
        n=1,
        temperature=0.2,
        top_p=0.95,
        max_tokens=512,
    )
    assert len(out) == 1
    text = out[0]
    assert text and text.strip(), "live endpoint returned empty content"
    # MathML-ish: a <math element or at least HTML/MathML tag content.
    low = text.lower()
    assert "<math" in low or "<m" in low or "mathml" in low, (
        f"live response did not look like MathML (head): {text[:200]!r}"
    )
