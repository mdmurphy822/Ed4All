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
    EndpointBatchItemError,
    EndpointRuntimeError,
    OpenAICompatibleRuntime,
    parse_batch_envelope,
    resolve_disable_thinking,
    resolve_specialist_batch_regions,
    resolve_specialist_max_retries,
    split_specialist_prompt,
)
from dart_semantic.qwen_specialists.runtime import (
    LlamaCppRuntime,
    MockRuntime,
    make_runtime,
    resolve_specialist_provider,
    resolve_structure_review_model,
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
# Transient/permanent classification + bounded retry
# ---------------------------------------------------------------------------


def test_max_retries_resolver_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_MAX_RETRIES", raising=False)
    assert resolve_specialist_max_retries() == 2
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "3")
    assert resolve_specialist_max_retries() == 3
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "0")
    assert resolve_specialist_max_retries() == 0  # honoured: no retry
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "-1")
    assert resolve_specialist_max_retries() == 2  # negative -> default
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "garbage")
    assert resolve_specialist_max_retries() == 2


# ---------------------------------------------------------------------------
# SEMANTIK_SPECIALIST_DISABLE_THINKING — resolver + body injection
# ---------------------------------------------------------------------------


def test_disable_thinking_resolver_parse_with_fallback(monkeypatch):
    # Unset -> False (default off, byte-stable for non-reasoning seats).
    monkeypatch.delenv("SEMANTIK_SPECIALIST_DISABLE_THINKING", raising=False)
    assert resolve_disable_thinking() is False
    # Falsey / garbage -> False.
    for falsey in ("0", "false", "no", "off", "garbage", "", "  "):
        monkeypatch.setenv("SEMANTIK_SPECIALIST_DISABLE_THINKING", falsey)
        assert resolve_disable_thinking() is False, falsey
    # Truthy (case-insensitive) -> True.
    for truthy in ("1", "true", "TRUE", "yes", "Yes", "on", "ON", " true "):
        monkeypatch.setenv("SEMANTIK_SPECIALIST_DISABLE_THINKING", truthy)
        assert resolve_disable_thinking() is True, truthy


def test_disable_thinking_on_injects_chat_template_kwargs(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPECIALIST_DISABLE_THINKING", "1")
    rt = OpenAICompatibleRuntime(
        base_url="https://x/v1", api_key="K", model="deepseek-ai/deepseek-v4-pro"
    )
    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(200, _ok_body("frag"))

    with mock.patch("requests.post", side_effect=_fake_post):
        rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)

    assert captured["json"]["chat_template_kwargs"] == {"thinking": False}


def test_disable_thinking_off_omits_chat_template_kwargs(monkeypatch):
    # Default OFF -> body is byte-stable (no chat_template_kwargs key).
    monkeypatch.delenv("SEMANTIK_SPECIALIST_DISABLE_THINKING", raising=False)
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")
    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(200, _ok_body("frag"))

    with mock.patch("requests.post", side_effect=_fake_post):
        rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)

    assert "chat_template_kwargs" not in captured["json"]


def test_timeout_is_marked_transient():
    import requests

    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    def _fake_post(url, *, json, headers, timeout):
        raise requests.exceptions.ReadTimeout("read timeout=120.0")

    with mock.patch("requests.post", side_effect=_fake_post):
        with pytest.raises(EndpointRuntimeError) as exc:
            rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)
    assert exc.value.transient is True


def test_5xx_transient_4xx_permanent():
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    def _post_503(url, *, json, headers, timeout):
        return _FakeResponse(503, json_body=None, text="upstream unavailable")

    with mock.patch("requests.post", side_effect=_post_503):
        with pytest.raises(EndpointRuntimeError) as exc:
            rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)
    assert exc.value.transient is True

    def _post_400(url, *, json, headers, timeout):
        return _FakeResponse(400, json_body=None, text="bad request")

    with mock.patch("requests.post", side_effect=_post_400):
        with pytest.raises(EndpointRuntimeError) as exc:
            rt.generate("SYSTEM: r\nUSER: {}", n=1, temperature=0.6, top_p=0.95, max_tokens=64)
    assert exc.value.transient is False


def test_transient_error_retried_then_succeeds(monkeypatch):
    """A transient timeout on the first attempt is retried and succeeds —
    the region's completion is recovered, not lost. (Test scenario (d).)"""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "2")
    # Zero out the back-off so the test is fast.
    monkeypatch.setattr(
        "dart_semantic.qwen_specialists.endpoint_runtime._RETRY_BACKOFF_BASE_SECONDS",
        0.0,
    )
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    import requests

    calls = {"n": 0}

    def _fake_post(url, *, json, headers, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("transient")
        return _FakeResponse(200, _ok_body("<p>recovered</p>"))

    with mock.patch("requests.post", side_effect=_fake_post):
        # generate_batch routes through the retry wrapper.
        out = rt.generate_batch(["SYSTEM: r\nUSER: {}"], max_tokens=64)
    assert out == ["<p>recovered</p>"]
    assert calls["n"] == 2  # one failed attempt + one successful retry


def test_permanent_error_not_retried(monkeypatch):
    """A 400 is NOT retried — only one POST is fired."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "5")
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    calls = {"n": 0}

    def _fake_post(url, *, json, headers, timeout):
        calls["n"] += 1
        return _FakeResponse(400, json_body=None, text="bad request")

    with mock.patch("requests.post", side_effect=_fake_post):
        with pytest.raises(EndpointBatchItemError):
            rt.generate_batch(["SYSTEM: r\nUSER: {}"], max_tokens=64)
    assert calls["n"] == 1  # NOT retried


def test_generate_batch_fail_soft_returns_none_sentinel(monkeypatch):
    """fail_soft=True: a failed item is the None sentinel; others succeed."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "0")
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    def _fake_post(url, *, json, headers, timeout):
        user = json["messages"][1]["content"]
        if user == "item-1":
            return _FakeResponse(500, json_body=None, text="boom")
        return _FakeResponse(200, _ok_body(f"<p>{user}</p>"))

    prompts = [f"SYSTEM: r\nUSER: item-{i}" for i in range(3)]
    with mock.patch("requests.post", side_effect=_fake_post):
        out = rt.generate_batch(prompts, max_tokens=32, fail_soft=True)
    assert out == ["<p>item-0</p>", None, "<p>item-2</p>"]


def test_generate_batch_default_still_raises(monkeypatch):
    """fail_soft default (False) keeps the byte-identical legacy raise."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "0")
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    def _fake_post(url, *, json, headers, timeout):
        user = json["messages"][1]["content"]
        if user == "item-1":
            return _FakeResponse(500, json_body=None, text="boom")
        return _FakeResponse(200, _ok_body(f"<p>{user}</p>"))

    prompts = [f"SYSTEM: r\nUSER: item-{i}" for i in range(3)]
    with mock.patch("requests.post", side_effect=_fake_post):
        with pytest.raises(EndpointBatchItemError) as ei:
            rt.generate_batch(prompts, max_tokens=32)
    assert ei.value.index == 1


# ---------------------------------------------------------------------------
# generate_multi — batched multi-region primitive (the 429 defeat)
# ---------------------------------------------------------------------------


def _batch_block(rid: str, frag: str) -> str:
    return f'<<<DART_REGION id="{rid}">>>\n{frag}\n<<<DART_REGION_END id="{rid}">>>'


def test_batch_regions_resolver_parse_with_fallback(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_BATCH_REGIONS", raising=False)
    assert resolve_specialist_batch_regions() == 12
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH_REGIONS", "20")
    assert resolve_specialist_batch_regions() == 20
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH_REGIONS", "0")
    assert resolve_specialist_batch_regions() == 12
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BATCH_REGIONS", "garbage")
    assert resolve_specialist_batch_regions() == 12


def test_parse_batch_envelope_well_formed_and_missing():
    body = "\n\n".join(
        [_batch_block("r0", "<p>zero</p>"), _batch_block("r2", "<p>two</p>")]
    )
    out = parse_batch_envelope(body, ["r0", "r1", "r2"])
    assert out == {"r0": "<p>zero</p>", "r1": None, "r2": "<p>two</p>"}


def test_parse_batch_envelope_tolerates_fences_and_missing_end_tag():
    # r1's END tag is absent -> None; fences + commentary around the rest are
    # stripped/ignored.
    body = (
        "```html\n"
        "Sure, here you go:\n"
        + _batch_block("r0", "<p>zero</p>")
        + '\n<<<DART_REGION id="r1">>>\n<p>orphan, no end tag</p>\n'
        + _batch_block("r2", "<math><mi>x</mi></math>")
        + "\n```"
    )
    out = parse_batch_envelope(body, ["r0", "r1", "r2"])
    assert out["r0"] == "<p>zero</p>"
    assert out["r1"] is None  # missing END tag
    assert out["r2"] == "<math><mi>x</mi></math>"


def test_generate_multi_one_post_splits_by_id(monkeypatch):
    rt = OpenAICompatibleRuntime(
        base_url="https://x/v1", api_key="K", model="m", timeout=10.0
    )
    posts = {"n": 0}

    def _fake_post(url, *, json, headers, timeout):
        posts["n"] += 1
        # Echo a well-formed multi-region response for r0/r1 (drop r2 to
        # exercise the None sentinel).
        content = "\n\n".join(
            [_batch_block("r0", "<p>A</p>"), _batch_block("r1", "<p>B</p>")]
        )
        return _FakeResponse(200, _ok_body(content))

    items = [("r0", "payload-0"), ("r1", "payload-1"), ("r2", "payload-2")]
    with mock.patch("requests.post", side_effect=_fake_post):
        out = rt.generate_multi(items, max_tokens=1024)

    # EXACTLY ONE POST for THREE regions (the rate-limit win).
    assert posts["n"] == 1
    assert out == {"r0": "<p>A</p>", "r1": "<p>B</p>", "r2": None}


def test_generate_multi_packs_payloads_into_single_user_turn(monkeypatch):
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")
    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        content = _batch_block("r0", "<p>A</p>")
        return _FakeResponse(200, _ok_body(content))

    with mock.patch("requests.post", side_effect=_fake_post):
        rt.generate_multi([("r0", "GROUNDING-PAYLOAD-VERBATIM")], max_tokens=512)

    user = captured["json"]["messages"][1]["content"]
    # Grounding payload travels verbatim, wrapped in the region delimiters.
    assert "GROUNDING-PAYLOAD-VERBATIM" in user
    assert '<<<DART_REGION id="r0">>>' in user
    assert '<<<DART_REGION_END id="r0">>>' in user
    # System turn carries the batch envelope directive.
    sys_turn = captured["json"]["messages"][0]["content"]
    assert "DART_REGION" in sys_turn


def test_generate_multi_caps_max_tokens_at_ceiling(monkeypatch):
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")
    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(200, _ok_body(_batch_block("r0", "<p>A</p>")))

    with mock.patch("requests.post", side_effect=_fake_post):
        # Ask for an absurd summed budget; the runtime caps at ceiling - margin.
        rt.generate_multi(
            [("r0", "p")], max_tokens=999999, output_token_ceiling=16384
        )
    # 16384 - 512 safety margin.
    assert captured["json"]["max_tokens"] == 16384 - 512


def test_generate_multi_refine_embeds_drafts_once(monkeypatch):
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")
    captured = {}

    def _fake_post(url, *, json, headers, timeout):
        captured["json"] = json
        return _FakeResponse(200, _ok_body(_batch_block("r0", "<p>better</p>")))

    with mock.patch("requests.post", side_effect=_fake_post):
        out = rt.generate_multi(
            [("r0", "payload-0")],
            max_tokens=512,
            drafts={"r0": "<p>local draft</p>"},
        )
    assert out == {"r0": "<p>better</p>"}
    user = captured["json"]["messages"][1]["content"]
    assert "DRAFT_FRAGMENT" in user
    assert "<p>local draft</p>" in user
    # The refine directive is stated ONCE in the SYSTEM turn, not per region.
    sys_turn = captured["json"]["messages"][0]["content"]
    assert "DRAFT_FRAGMENT" in sys_turn or "draft" in sys_turn.lower()


def test_generate_multi_whole_batch_failure_returns_all_none(monkeypatch):
    """A batch POST that fails after retries degrades EVERY region in the
    batch to None (fail-soft) — never raises, never aborts the document."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MAX_RETRIES", "0")
    rt = OpenAICompatibleRuntime(base_url="https://x/v1", api_key="K", model="m")

    def _fake_post(url, *, json, headers, timeout):
        return _FakeResponse(429, json_body=None, text="rate limited")

    items = [("r0", "p0"), ("r1", "p1")]
    with mock.patch("requests.post", side_effect=_fake_post):
        out = rt.generate_multi(items, max_tokens=512)
    assert out == {"r0": None, "r1": None}


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


def test_make_runtime_force_local_suppresses_endpoint_short_circuit(monkeypatch):
    """force_local=True keeps mode='real' on the LOCAL GGUF arm even when an
    endpoint PROVIDER is configured (the dereliction-fix escape hatch).

    It must NEVER return the endpoint runtime; depending on whether GGUFs are
    on disk it either returns the local LlamaCppRuntime or raises the strict
    'no Qwen LoRA adapters' GGUF error — both prove it took the local arm."""
    monkeypatch.setenv("SEMANTIK_SPECIALIST_PROVIDER", "nvidia")
    try:
        rt = make_runtime("real", force_local=True)
    except RuntimeError as exc:
        # Local arm's strict GGUF presence check (no adapters on disk).
        assert "no Qwen LoRA adapters" in str(exc) or "MISSING" in str(exc)
    else:
        assert not isinstance(rt, OpenAICompatibleRuntime)
        assert isinstance(rt, LlamaCppRuntime)


def test_make_runtime_explicit_endpoint_mode_ignores_force_local(monkeypatch):
    """An explicit mode='endpoint' (reviewer / resegment seats) is NEVER
    forced local — force_local only suppresses the mode='real' short-circuit."""
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    rt = make_runtime("endpoint", force_local=True)
    assert isinstance(rt, OpenAICompatibleRuntime)


def test_make_runtime_real_local_still_strict(monkeypatch):
    # Local arm (no endpoint provider) keeps the strict GGUF check — with
    # the v1 null-adapter config it must raise rather than silently mock.
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    with pytest.raises(RuntimeError) as exc:
        make_runtime("real")
    assert "no Qwen LoRA adapters" in str(exc.value) or "MISSING" in str(exc.value)


# ---------------------------------------------------------------------------
# Stage-5d structure-reviewer model seat (SEMANTIK_STRUCTURE_REVIEW_MODEL)
#
# The reviewer must honour its OWN model id (resolved by
# resolve_structure_review_model) when set, falling back to the shared
# SEMANTIK_SPECIALIST_MODEL otherwise. This is the precedence CLAUDE.md
# documents; these tests pin that the documented knob actually ROUTES the
# reviewer runtime's model (not just an audit string).
# ---------------------------------------------------------------------------


def test_make_runtime_endpoint_model_override(monkeypatch):
    # An explicit model= (how the cascade passes the reviewer seat) wins
    # over the env-resolved specialist model.
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MODEL", "specialist/model-x")
    rt = make_runtime("endpoint", model="reviewer/model-y")
    assert isinstance(rt, OpenAICompatibleRuntime)
    assert rt._model == "reviewer/model-y"


def test_make_runtime_endpoint_model_none_uses_specialist(monkeypatch):
    # Stage-6 specialist path passes no model → byte-stable env resolution
    # (SEMANTIK_SPECIALIST_MODEL). Confirms the new kwarg is opt-in only.
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MODEL", "specialist/model-x")
    monkeypatch.delenv("NVIDIA_LARGE_MODEL", raising=False)
    rt = make_runtime("endpoint")
    assert isinstance(rt, OpenAICompatibleRuntime)
    assert rt._model == "specialist/model-x"


def test_reviewer_runtime_resolves_structure_review_model(monkeypatch):
    # The reviewer seat (model=resolve_structure_review_model()) honours
    # SEMANTIK_STRUCTURE_REVIEW_MODEL when set — the documented dedicated seat.
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", "reviewer/dedicated-70b")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MODEL", "specialist/model-x")
    rt = make_runtime("endpoint", model=resolve_structure_review_model())
    assert rt._model == "reviewer/dedicated-70b"


def test_reviewer_runtime_falls_back_to_specialist_model(monkeypatch):
    # Reviewer model unset → falls back to SEMANTIK_SPECIALIST_MODEL (the
    # shared seat), matching the legacy behaviour byte-for-byte.
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", raising=False)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_MODEL", "specialist/model-x")
    rt = make_runtime("endpoint", model=resolve_structure_review_model())
    assert rt._model == "specialist/model-x"


def test_make_runtime_mock_ignores_model(monkeypatch):
    # The model override is endpoint-only; the mock arm ignores it (no
    # hosted model to pin) and stays a MockRuntime.
    rt = make_runtime("mock", model="reviewer/model-y")
    assert isinstance(rt, MockRuntime)


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
