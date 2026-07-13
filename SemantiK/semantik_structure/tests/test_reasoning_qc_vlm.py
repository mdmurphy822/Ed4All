"""Unit tests for the reasoning-QC TEXT client (``reasoning_qc_vlm``).

**2026-07-12 text-only pivot.** The QC pass reasons over the ASSEMBLED
accessible-HTML block text (block RECORDS: type / role / level / page / text),
NOT over page-image rasters. These tests pin the post-pivot transport contract:

  (a) seat resolution is the OWN reasoning chain (explicit reasoning-QC seat >
      specialist text seat > legacy VLM seat with a deprecation warning), with a
      fail-loud ``require_ready`` on a non-loopback keyless hosted seat;
  (b) NO page render — the module has no ``render_page`` and no request carries
      image bytes (the new image-free regression guard);
  (c) the composed body carries a TEXT-ONLY user turn (block list, no
      ``image_url`` part), ``temperature=0``, the JSON-mode fields, and
      ``chat_template_kwargs`` (thinking-off) iff the QC-specific
      ``SEMANTIK_REASONING_QC_DISABLE_THINKING`` is set — INDEPENDENT of the
      extraction flag ``SEMANTIK_VLM_DISABLE_THINKING``;
  (d) the JSON parse tolerates fenced / embedded output;
  (e) a ``VlmExtractError`` degrades to an EMPTY verdict (fail-soft, no raise);
  (f) the reasoning-preserving SPLIT LADDER never emits a thinking-off request.
"""
from __future__ import annotations

import logging

import pytest

from semantik_structure import reasoning_qc_vlm, vlm_extract
from semantik_structure.extract_shared import VLMSeat, VLMSeatError


# ---------------------------------------------------------------------------
# Block-record helper (the text-only atomic unit after the pivot).
# ---------------------------------------------------------------------------
def _blk(text: str, *, type: str = "paragraph", page: int | None = 1, level=None, role=None) -> dict:
    return {"type": type, "role": role or type, "level": level, "page": page, "text": text}


def _clear_seat_envs(monkeypatch) -> None:
    for env in (
        "SEMANTIK_REASONING_QC_BASE_URL",
        "SEMANTIK_REASONING_QC_API_KEY",
        "SEMANTIK_REASONING_QC_MODEL",
        "SEMANTIK_SPECIALIST_BASE_URL",
        "SEMANTIK_SPECIALIST_API_KEY",
        "SEMANTIK_STRUCTURE_REVIEW_MODEL",
        "SEMANTIK_SPECIALIST_MODEL",
        "NVIDIA_LARGE_MODEL",
    ):
        monkeypatch.delenv(env, raising=False)


# ---------------------------------------------------------------------------
# (a) seat resolution — the OWN reasoning-seat chain + fail-loud require_ready.
# ---------------------------------------------------------------------------
def test_qc_model_override(monkeypatch):
    monkeypatch.setenv("SEMANTIK_REASONING_QC_MODEL", "nemotron-reasoner:120b")
    assert reasoning_qc_vlm.resolve_reasoning_qc_model(default_model="qwen2.5:7b") == "nemotron-reasoner:120b"


def test_qc_model_defaults_to_supplied_model(monkeypatch):
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MODEL", raising=False)
    assert reasoning_qc_vlm.resolve_reasoning_qc_model(default_model="qwen2.5:7b") == "qwen2.5:7b"


def test_seat_chain_explicit_wins(monkeypatch):
    """Tier 1: SEMANTIK_REASONING_QC_BASE_URL wins over everything."""
    _clear_seat_envs(monkeypatch)
    monkeypatch.setenv("SEMANTIK_REASONING_QC_BASE_URL", "http://localhost:9911/v1")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_MODEL", "explicit-reasoner")
    # A specialist seat is ALSO set — the explicit reasoning-QC seat must win.
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BASE_URL", "http://localhost:8822/v1")
    seat = reasoning_qc_vlm.resolve_reasoning_qc_seat()
    assert seat.base_url == "http://localhost:9911/v1"
    assert seat.model == "explicit-reasoner"
    assert seat.provider == "reasoning-qc"


def test_seat_chain_specialist_second(monkeypatch):
    """Tier 2: with no explicit QC seat, the SPECIALIST text seat is used (the
    'reasoning model, not VLM' endpoint); model = the specialist review model."""
    _clear_seat_envs(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BASE_URL", "http://localhost:8822/v1")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_API_KEY", "spec-key")
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", "qwen2.5-7b-16k:latest")
    seat = reasoning_qc_vlm.resolve_reasoning_qc_seat()
    assert seat.base_url == "http://localhost:8822/v1"
    assert seat.api_key == "spec-key"
    assert seat.model == "qwen2.5-7b-16k:latest"
    assert seat.provider == "specialist"


def test_seat_chain_qc_model_override_beats_specialist_model(monkeypatch):
    """On the specialist tier the QC model override still wins the model id."""
    _clear_seat_envs(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BASE_URL", "http://localhost:8822/v1")
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", "spec-model")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_MODEL", "qc-override")
    seat = reasoning_qc_vlm.resolve_reasoning_qc_seat()
    assert seat.model == "qc-override"


def test_seat_chain_legacy_vlm_last_with_warning(monkeypatch, caplog):
    """Tier 3: no reasoning/specialist seat → legacy VLM seat WITH a warning."""
    _clear_seat_envs(monkeypatch)
    monkeypatch.setattr(
        reasoning_qc_vlm,
        "resolve_vlm_seat",
        lambda: VLMSeat(provider="local", base_url="http://localhost:11434/v1", api_key=None, model="qwen2.5vl:7b"),
    )
    with caplog.at_level(logging.WARNING):
        seat = reasoning_qc_vlm.resolve_reasoning_qc_seat()
    assert seat.base_url == "http://localhost:11434/v1"
    assert any("LEGACY VLM seat" in r.getMessage() for r in caplog.records)


def test_seat_fail_loud_on_keyless_hosted_legacy(monkeypatch):
    """The tier-3 legacy fallback still fail-louds on a keyless hosted seat."""
    _clear_seat_envs(monkeypatch)
    monkeypatch.setattr(
        reasoning_qc_vlm,
        "resolve_vlm_seat",
        lambda: VLMSeat(provider="nvidia", base_url="https://api.nvidia.example/v1", api_key=None, model="m"),
    )
    with pytest.raises(VLMSeatError):
        reasoning_qc_vlm.resolve_reasoning_qc_seat()


def test_seat_fail_loud_on_keyless_hosted_specialist(monkeypatch):
    """A hosted specialist seat with no key fail-louds (never a silent stub)."""
    _clear_seat_envs(monkeypatch)
    monkeypatch.setenv("SEMANTIK_SPECIALIST_BASE_URL", "https://api.together.example/v1")
    with pytest.raises(VLMSeatError):
        reasoning_qc_vlm.resolve_reasoning_qc_seat()


def test_no_render_page_symbol():
    """The page-render entry point is GONE after the text-only pivot."""
    assert not hasattr(reasoning_qc_vlm, "render_page")


# ---------------------------------------------------------------------------
# (c) composed body shape — TEXT-ONLY, no image_url.
# ---------------------------------------------------------------------------
class _CapturingResp:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _CapturingRequests:
    """Records the last POST url + body; returns a canned QC JSON completion."""

    def __init__(self, content='{"reading_order": [], "phantom_headings": []}'):
        self.last_body = None
        self.last_url = None
        self._content = content

    def post(self, url, json=None, headers=None, timeout=None):
        self.last_url = url
        self.last_body = json
        return _CapturingResp({"choices": [{"message": {"content": self._content}}]})


def _user_text(body) -> str:
    user = body["messages"][1]["content"]
    return next(p for p in user if p["type"] == "text")["text"]


def _assert_no_image_bytes(body) -> None:
    """IMAGE-FREE regression guard: no user-turn part may be an image_url."""
    user = body["messages"][1]["content"]
    for part in user:
        assert part.get("type") != "image_url", body
        # Belt: no data-URI image smuggled inside a text part.
        assert "data:image" not in str(part.get("text", "")), body


def _assert_body_thinking_on(body) -> None:
    """Thinking-ON regression guard: the body must carry NO thinking-off key.

    chat_template_kwargs is ALLOWED (the Nemotron-3 reasoning_budget rides it on
    thinking-on bodies) — what must never appear is thinking / enable_thinking."""
    ctk = body.get("chat_template_kwargs") or {}
    assert "thinking" not in ctk, body
    assert "enable_thinking" not in ctk, body


def test_post_body_shape_text_only(monkeypatch):
    monkeypatch.delenv("SEMANTIK_VLM_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    req = _CapturingRequests()
    out = reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="qwen2.5:7b",
        blocks=[_blk("First block", type="heading", level=2, page=12), _blk("Second block", page=12)],
        timeout=30.0,
        requests_module=req,
    )
    assert '"reading_order"' in out
    body = req.last_body
    # Thinking-ON (default) → reasoning sampling: temp 0.6 / top_p 0.95, pinned
    # max_tokens 16384 (greedy 0 loops an unterminated <think> on Nemotron).
    assert body["temperature"] == 0.6
    assert body["top_p"] == 0.95
    assert body["max_tokens"] == 16384
    assert body["model"] == "qwen2.5:7b"
    # JSON-mode fields present.
    assert body["response_format"] == {"type": "json_object"}
    assert body["format"] == "json"
    # TEXT-ONLY user content — no image_url part.
    user = body["messages"][1]["content"]
    assert {part["type"] for part in user} == {"text"}
    _assert_no_image_bytes(body)
    # The rendered block list carries index, type, level, page, and text.
    text = _user_text(body)
    assert "[0]" in text and "First block" in text
    assert "heading" in text and "h2" in text
    assert "[p.12]" in text
    # Thinking-ON default → the ONLY chat_template_kwargs content is the
    # Nemotron-3 TRAINED reasoning budget (never a thinking-off key).
    assert body["chat_template_kwargs"] == {"reasoning_budget": 4096}


def test_post_body_middle_truncates_long_block(monkeypatch):
    """A very long block body keeps head + tail with a middle-omitted marker."""
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    long_text = "HEAD" + ("x" * 2000) + "TAIL"
    req = _CapturingRequests()
    reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="m",
        blocks=[_blk(long_text)],
        timeout=30.0,
        requests_module=req,
    )
    text = _user_text(req.last_body)
    assert "HEAD" in text and "TAIL" in text
    assert "middle" in text and "omitted" in text
    # The full 2008-char body is NOT dumped whole.
    assert long_text not in text


def test_post_body_thinking_off(monkeypatch):
    """The QC-specific SEMANTIK_REASONING_QC_DISABLE_THINKING adds the thinking-off block."""
    monkeypatch.delenv("SEMANTIK_VLM_DISABLE_THINKING", raising=False)
    monkeypatch.setenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", "1")
    req = _CapturingRequests()
    reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="m",
        blocks=[_blk("b")],
        timeout=30.0,
        requests_module=req,
    )
    assert req.last_body["chat_template_kwargs"] == {"thinking": False, "enable_thinking": False}
    # Thinking-OFF NEVER carries the reasoning budget (budget is thinking-on only).
    assert "reasoning_budget" not in req.last_body["chat_template_kwargs"]
    # Explicit thinking-OFF keeps GREEDY sampling (temp 0, NO top_p override) —
    # greedy is correct for a non-reasoning transcription request.
    assert req.last_body["temperature"] == 0
    assert "top_p" not in req.last_body
    assert req.last_body["max_tokens"] == 16384  # still pinned


def test_qc_reasoning_budget_env_override(monkeypatch):
    """SEMANTIK_REASONING_QC_REASONING_BUDGET overrides the default 4096."""
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.setenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", "2048")
    req = _CapturingRequests()
    reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="m",
        blocks=[_blk("b")],
        timeout=30.0,
        requests_module=req,
    )
    assert req.last_body["chat_template_kwargs"] == {"reasoning_budget": 2048}


def test_qc_reasoning_budget_zero_disables_kwarg(monkeypatch):
    """Budget 0 (or a falsey token) → NO reasoning_budget kwarg at all —
    the escape hatch for a non-Nemotron seat whose template lacks it."""
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    for off_val in ("0", "off", "false", "no"):
        monkeypatch.setenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", off_val)
        req = _CapturingRequests()
        reasoning_qc_vlm._post_qc_completion(
            base_url="http://localhost:11434/v1",
            api_key=None,
            model="m",
            blocks=[_blk("b")],
            timeout=30.0,
            requests_module=req,
        )
        assert "chat_template_kwargs" not in req.last_body, off_val


@pytest.mark.parametrize(
    "val,expected",
    [
        (None, 4096),        # unset → default
        ("", 4096),          # blank → default
        ("garbage", 4096),   # non-int → default
        ("-1", 4096),        # negative → default
        ("0", 0),            # explicit 0 → disabled
        ("off", 0),          # falsey token → disabled
        ("false", 0),
        ("no", 0),
        ("2048", 2048),      # valid int → honoured
        ("8192", 8192),
    ],
)
def test_qc_reasoning_budget_resolver(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", val)
    assert reasoning_qc_vlm.resolve_reasoning_qc_reasoning_budget() == expected


def test_qc_sampling_dict_carries_budget(monkeypatch):
    """resolve_reasoning_qc_sampling folds the budget in (thinking-mode aware),
    so the POST body AND the sidecar fingerprint pick it up from ONE source."""
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", raising=False)
    s = reasoning_qc_vlm.resolve_reasoning_qc_sampling()
    assert s["reasoning_budget"] == 4096
    # Disabled budget → None in the dict (no kwarg on the body).
    monkeypatch.setenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", "0")
    assert reasoning_qc_vlm.resolve_reasoning_qc_sampling()["reasoning_budget"] is None
    # Thinking-off → None regardless of the env (budget is thinking-on only).
    monkeypatch.setenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", "2048")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", "1")
    assert reasoning_qc_vlm.resolve_reasoning_qc_sampling()["reasoning_budget"] is None


def test_qc_sampling_env_overrides(monkeypatch):
    """Thinking-ON sampling params honour their envs (parse-with-fallback)."""
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.setenv("SEMANTIK_REASONING_QC_TEMPERATURE", "0.3")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_TOP_P", "0.8")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_MAX_TOKENS", "4096")
    req = _CapturingRequests()
    reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="m",
        blocks=[_blk("b")],
        timeout=30.0,
        requests_module=req,
    )
    assert req.last_body["temperature"] == 0.3
    assert req.last_body["top_p"] == 0.8
    assert req.last_body["max_tokens"] == 4096


@pytest.mark.parametrize(
    "env,val,resolver,expected",
    [
        ("SEMANTIK_REASONING_QC_TEMPERATURE", None, "resolve_reasoning_qc_temperature", 0.6),
        ("SEMANTIK_REASONING_QC_TEMPERATURE", "garbage", "resolve_reasoning_qc_temperature", 0.6),
        ("SEMANTIK_REASONING_QC_TEMPERATURE", "-1", "resolve_reasoning_qc_temperature", 0.6),
        ("SEMANTIK_REASONING_QC_TEMPERATURE", "0", "resolve_reasoning_qc_temperature", 0.0),
        ("SEMANTIK_REASONING_QC_TOP_P", None, "resolve_reasoning_qc_top_p", 0.95),
        ("SEMANTIK_REASONING_QC_TOP_P", "1.5", "resolve_reasoning_qc_top_p", 0.95),
        ("SEMANTIK_REASONING_QC_TOP_P", "0.5", "resolve_reasoning_qc_top_p", 0.5),
        ("SEMANTIK_REASONING_QC_MAX_TOKENS", None, "resolve_reasoning_qc_max_tokens", 16384),
        ("SEMANTIK_REASONING_QC_MAX_TOKENS", "garbage", "resolve_reasoning_qc_max_tokens", 16384),
        ("SEMANTIK_REASONING_QC_MAX_TOKENS", "0", "resolve_reasoning_qc_max_tokens", 16384),
        ("SEMANTIK_REASONING_QC_MAX_TOKENS", "2048", "resolve_reasoning_qc_max_tokens", 2048),
    ],
)
def test_qc_sampling_resolvers(monkeypatch, env, val, resolver, expected):
    if val is None:
        monkeypatch.delenv(env, raising=False)
    else:
        monkeypatch.setenv(env, val)
    assert getattr(reasoning_qc_vlm, resolver)() == expected


def test_qc_directive_structural_contract():
    """v2 terse-directive regression lock — STRUCTURAL properties, not prose:
    (a) all five output-contract keys are named (downstream parse/stitch depends
    on them); (b) the never-rewrite + never-solve-math guards are present;
    (c) the v1 rumination fuel (step-by-step coaching, the ignored soft word
    cap, the 'strong reasoning' framing) is gone — deliberation length is the
    reasoning_budget chat kwarg's job now."""
    d = reasoning_qc_vlm._QC_SYSTEM_DIRECTIVE
    # (a) the exact JSON output contract keys.
    for key in (
        '"reading_order"',
        '"phantom_headings"',
        '"apparatus_retype"',
        '"misordered"',
        '"confidence"',
    ):
        assert key in d, key
    assert '{"index": int, "reason": str}' in d
    assert '{"run": [int, ...], "reason": str}' in d
    # (b) the guard sentences.
    assert "NEVER rewrite" in d
    assert "never solve, verify, or re-compute" in d
    assert "ONLY the block indices" in d
    assert "every list empty" in d
    assert "ONLY the JSON object" in d
    # (c) the rumination fuel is cut.
    lowered = d.lower()
    assert "step by step" not in lowered
    assert "1000 words" not in lowered
    assert "strong reasoning" not in lowered
    # The prompt-contract version was bumped for this directive change.
    assert reasoning_qc_vlm.QC_PROMPT_VERSION >= 2


def test_qc_thinking_independent_of_extraction_flag(monkeypatch):
    """SPLIT GUARD: SEMANTIK_VLM_DISABLE_THINKING=1 alone (the extraction flag)
    must NOT disable QC thinking — QC reads its OWN flag."""
    monkeypatch.setenv("SEMANTIK_VLM_DISABLE_THINKING", "1")
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    req = _CapturingRequests()
    reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="m",
        blocks=[_blk("b")],
        timeout=30.0,
        requests_module=req,
    )
    _assert_body_thinking_on(req.last_body)


def test_qc_thinking_on_by_default(monkeypatch):
    """DEFAULT (no thinking envs) → QC body has NO thinking-off block (thinking ON)."""
    monkeypatch.delenv("SEMANTIK_VLM_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    req = _CapturingRequests()
    reasoning_qc_vlm._post_qc_completion(
        base_url="http://localhost:11434/v1",
        api_key=None,
        model="m",
        blocks=[_blk("b")],
        timeout=30.0,
        requests_module=req,
    )
    _assert_body_thinking_on(req.last_body)


def test_request_goes_to_resolved_base_url(monkeypatch):
    """A QC request is POSTed to the RESOLVED reasoning-seat base_url."""
    _clear_seat_envs(monkeypatch)
    monkeypatch.setenv("SEMANTIK_REASONING_QC_BASE_URL", "http://localhost:7788/v1")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_MODEL", "reasoner")
    seat = reasoning_qc_vlm.resolve_reasoning_qc_seat()
    req = _CapturingRequests()
    reasoning_qc_vlm.run_qc_judgment(seat, "/tmp/x.pdf", 1, [_blk("a")], requests_module=req)
    assert req.last_url.startswith("http://localhost:7788/v1")


# ---------------------------------------------------------------------------
# (d) tolerant JSON parse.
# ---------------------------------------------------------------------------
def test_parse_qc_json_variants():
    assert reasoning_qc_vlm.parse_qc_json('{"a": 1}') == {"a": 1}
    assert reasoning_qc_vlm.parse_qc_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert reasoning_qc_vlm.parse_qc_json('noise {"a": 3} tail') == {"a": 3}
    assert reasoning_qc_vlm.parse_qc_json("") == {}
    assert reasoning_qc_vlm.parse_qc_json("not json") == {}
    assert reasoning_qc_vlm.parse_qc_json("[1,2,3]") == {}  # a list is not a verdict dict


# ---------------------------------------------------------------------------
# (e) run_qc_judgment fail-soft.
# ---------------------------------------------------------------------------
def _seat():
    return VLMSeat(provider="local", base_url="http://localhost:11434/v1", api_key=None, model="m")


def test_run_qc_judgment_happy(monkeypatch):
    monkeypatch.setattr(
        reasoning_qc_vlm,
        "_post_with_retry",
        lambda **k: '{"reading_order": [1, 0], "phantom_headings": []}',
    )
    out = reasoning_qc_vlm.run_qc_judgment(_seat(), "/tmp/x.pdf", 1, [_blk("a"), _blk("b")], requests_module=object())
    assert out == {"reading_order": [1, 0], "phantom_headings": []}


def test_run_qc_judgment_failsoft_on_error(monkeypatch):
    def _boom(**k):
        raise vlm_extract.VlmExtractError("permanent boom", transient=False)

    monkeypatch.setattr(reasoning_qc_vlm, "_post_with_retry", _boom)
    out = reasoning_qc_vlm.run_qc_judgment(_seat(), "/tmp/x.pdf", 1, [_blk("a")], requests_module=object())
    assert out == {}  # fail-soft empty verdict, no raise


def test_post_with_retry_is_mockable_module_level():
    assert callable(reasoning_qc_vlm._post_with_retry)


# ===========================================================================
# (f) Reasoning-preserving SPLIT LADDER (text-window is now the reasoning
# surface). OWNER DIRECTIVE: never a thinking-off verdict — the only lever is a
# SMALLER text window, re-judged thinking-on; exhaustion → qc_incomplete.
# ===========================================================================
class ReadTimeout(Exception):
    """A stand-in whose class name carries 'Timeout' (the QC timeout sniff)."""


def _blocks_in(body) -> int:
    """Count the ``[N] ...`` block-list lines in a composed QC request body."""
    return sum(1 for line in _user_text(body).splitlines() if line.startswith("["))


class _PolicyRequests:
    """Drives each POST by its block-count via ``policy(nblocks)``.

    ``policy`` returns ``"null"`` (content=None), ``"timeout"`` (raise a
    ReadTimeout), or a JSON string (a valid completion). Records EVERY request
    body so a test can assert NO body carries a thinking-off / image part.
    """

    def __init__(self, policy):
        self.bodies: list = []
        self.calls = 0
        self._policy = policy

    def post(self, url, json=None, headers=None, timeout=None):
        self.bodies.append(json)
        self.calls += 1
        outcome = self._policy(_blocks_in(json))
        if outcome == "timeout":
            raise ReadTimeout("simulated read timeout")
        content = None if outcome == "null" else outcome
        return _CapturingResp({"choices": [{"message": {"content": content}}]})


def _assert_no_thinking_off(bodies) -> None:
    for b in bodies:
        _assert_body_thinking_on(b)


def _assert_no_image(bodies) -> None:
    for b in bodies:
        _assert_no_image_bytes(b)


def _blocks(n):
    return [_blk(f"b{i}") for i in range(n)]


def test_null_content_enters_split_ladder(monkeypatch):
    """A null on the FULL window splits into TWO half-window thinking-on calls
    (never a thinking-off retry); the halves' verdicts merge with correct remap."""
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MAX_SPLIT_DEPTH", raising=False)  # default 2
    valid = '{"phantom_headings": [{"index": 0, "reason": "x"}], "confidence": 0.9}'
    req = _PolicyRequests(lambda n: "null" if n == 8 else valid)
    out = reasoning_qc_vlm.run_qc_judgment(_seat(), "/tmp/x.pdf", 7, _blocks(8), requests_module=req)
    assert req.calls == 3
    assert _blocks_in(req.bodies[0]) == 8
    assert _blocks_in(req.bodies[1]) == 4 and _blocks_in(req.bodies[2]) == 4
    _assert_no_thinking_off(req.bodies)
    _assert_no_image(req.bodies)
    idxs = sorted(item["index"] for item in out.get("phantom_headings") or [])
    assert idxs == [0, 4]
    assert out.get("confidence") == 0.9
    assert "_qc_incomplete" not in out


def test_null_min_size_exhausted_qc_incomplete(monkeypatch):
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MAX_SPLIT_DEPTH", raising=False)
    req = _PolicyRequests(lambda n: "null")
    out = reasoning_qc_vlm.run_qc_judgment(_seat(), "/tmp/x.pdf", 7, _blocks(8), requests_module=req)
    assert req.calls == 3
    assert out == {"_qc_incomplete": [0, 1, 2, 3, 4, 5, 6, 7]}
    _assert_no_thinking_off(req.bodies)


def test_null_below_min_immediate_qc_incomplete(monkeypatch):
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    req = _PolicyRequests(lambda n: "null")
    out = reasoning_qc_vlm.run_qc_judgment(_seat(), "/tmp/x.pdf", 7, _blocks(5), requests_module=req)
    assert req.calls == 1
    assert out == {"_qc_incomplete": [0, 1, 2, 3, 4]}
    _assert_no_thinking_off(req.bodies)


def test_timeout_enters_split_ladder(monkeypatch):
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MAX_SPLIT_DEPTH", raising=False)
    monkeypatch.setattr(reasoning_qc_vlm, "_QC_MAX_RETRIES", 0)
    valid = '{"phantom_headings": [], "confidence": 0.8}'
    req = _PolicyRequests(lambda n: "timeout" if n == 8 else valid)
    out = reasoning_qc_vlm.run_qc_judgment(_seat(), "/tmp/x.pdf", 7, _blocks(8), requests_module=req)
    assert req.calls == 3
    assert _blocks_in(req.bodies[0]) == 8
    assert _blocks_in(req.bodies[1]) == 4 and _blocks_in(req.bodies[2]) == 4
    _assert_no_thinking_off(req.bodies)
    _assert_no_image(req.bodies)
    assert out.get("confidence") == 0.8


def test_null_content_error_is_vlm_extract_subclass():
    err = reasoning_qc_vlm.QCNullContentError("null", transient=False)
    assert isinstance(err, vlm_extract.VlmExtractError)
    assert err.transient is False


def test_timeout_error_is_transient_vlm_extract_subclass():
    err = reasoning_qc_vlm.QCTimeoutError("t", transient=True)
    assert isinstance(err, vlm_extract.VlmExtractError)
    assert err.transient is True


# ---------------------------------------------------------------------------
# (g) resolvers — max split depth + QC POST timeout.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "val,expected",
    [(None, 2), ("", 2), ("garbage", 2), ("-1", 2), ("0", 0), ("1", 1), ("3", 3)],
)
def test_resolve_max_split_depth(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_MAX_SPLIT_DEPTH", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_MAX_SPLIT_DEPTH", val)
    assert reasoning_qc_vlm.resolve_reasoning_qc_max_split_depth() == expected


@pytest.mark.parametrize(
    "val,expected",
    [
        (None, 1200.0), ("", 1200.0), ("garbage", 1200.0), ("nan", 1200.0),
        ("inf", 1200.0), ("0", 1200.0), ("-5", 1200.0), ("600", 600.0), ("90.5", 90.5),
    ],
)
def test_resolve_qc_timeout(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_TIMEOUT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_TIMEOUT_SECONDS", val)
    assert reasoning_qc_vlm.resolve_reasoning_qc_timeout() == expected
