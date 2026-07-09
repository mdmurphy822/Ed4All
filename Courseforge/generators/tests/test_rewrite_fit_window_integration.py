"""Integration tests for the rewrite-tier fit-window wiring + the input-
truncation tripwire + the backstop-no-op contract (rewrite-overflow-fix-
2026-06). These exercise a real ``RewriteProvider`` over an httpx
MockTransport, not the pure helpers."""
from __future__ import annotations

from typing import Callable, List

import httpx
import pytest

from Courseforge.generators._rewrite_provider import (
    RewriteProvider,
    _REWRITE_SYSTEM_PROMPT,
    _REWRITE_SYSTEM_PROMPT_TRIMMED,
)
from Courseforge.scripts.blocks import Block


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _success_body(content: str, *, prompt_tokens: int = 8800) -> dict:
    return {
        "id": "cmpl",
        "model": "test",
        "choices": [
            {"index": 0,
             "message": {"role": "assistant", "content": content},
             "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 50,
                  "total_tokens": prompt_tokens + 50},
    }


def _block(
    *, block_type: str = "concept", chunk_ids=None, curies=None,
) -> Block:
    return Block(
        block_id=f"page#{block_type}_x_0",
        block_type=block_type,
        page_id="page",
        sequence=0,
        content={
            "key_claims": [
                {"text": "The central concept is X.",
                 "source_chunk_ids": list(chunk_ids or [])},
            ],
            "curies": list(curies or []),
            "source_refs": ["dart:slug#blk1"],
            "objective_refs": ["TO-01"],
        },
    )


# ---------------------------------------------------------------------------
# Fit-window OFF: system prompt is the untrimmed constant; chunks untrimmed.
# ---------------------------------------------------------------------------
def test_off_uses_untrimmed_system_prompt(monkeypatch):
    monkeypatch.delenv("ED4ALL_REWRITE_FIT_WINDOW", raising=False)
    seen: List[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read())
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    assert p._fit_window is False
    assert p._system_prompt == _REWRITE_SYSTEM_PROMPT
    p.generate_rewrite(_block())
    body = seen[0].decode("utf-8")
    # The full system prompt's block-specific STYLED COMPONENTS segment is
    # present in the OFF wire body (it lives in the system message).
    assert "STYLED COMPONENTS" in body


# ---------------------------------------------------------------------------
# Fit-window ON: trimmed system prompt + relocated per-type contract.
# ---------------------------------------------------------------------------
def test_on_uses_trimmed_system_prompt_and_relocates(monkeypatch):
    monkeypatch.setenv("ED4ALL_REWRITE_FIT_WINDOW", "1")
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "16384")
    seen: List[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read())
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    assert p._fit_window is True
    assert p._system_prompt == _REWRITE_SYSTEM_PROMPT_TRIMMED
    assert p._rewrite_num_ctx == 16384
    # The trimmed system prompt is materially shorter.
    assert len(p._system_prompt) < len(_REWRITE_SYSTEM_PROMPT)
    p.generate_rewrite(_block(block_type="concept"))
    body = seen[0].decode("utf-8")
    # The relocated STYLED COMPONENTS guidance now rides in the USER prompt
    # (the per-type contract), not the system prompt — but it's still on the
    # wire so the block keeps its rules.
    assert "STYLED COMPONENTS" in body


# ---------------------------------------------------------------------------
# Fit-window ON: chunk-window budget trims grounding to fit a tiny num_ctx.
# ---------------------------------------------------------------------------
def test_on_chunk_window_drops_trailing_chunks(monkeypatch):
    monkeypatch.setenv("ED4ALL_REWRITE_FIT_WINDOW", "1")
    # Tiny window forces dropping most chunks.
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "8192")
    seen: List[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read())
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    big_chunks = [
        {"id": f"c{i}", "text": "word " * 400} for i in range(20)
    ]
    p.generate_rewrite(_block(), source_chunks=big_chunks)
    body = seen[0].decode("utf-8")
    # Not every chunk id survived (the budget dropped the trailing ones).
    present = sum(1 for i in range(20) if f"[c{i}]" in body)
    assert 1 <= present < 20


# ---------------------------------------------------------------------------
# Truncation tripwire: a large reported shortfall stamps the marker.
# ---------------------------------------------------------------------------
def test_tripwire_stamps_input_prompt_truncated(monkeypatch):
    monkeypatch.delenv("ED4ALL_REWRITE_FIT_WINDOW", raising=False)
    # Server reports a tiny prompt_tokens vs the big (OFF) system prompt →
    # head truncated.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_success_body("<p>fabricated</p>", prompt_tokens=100)
        )

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_block())
    assert out.escalation_marker == "input_prompt_truncated"


def test_tripwire_noops_when_disabled(monkeypatch):
    monkeypatch.setenv("ED4ALL_REWRITE_TRUNCATION_TRIPWIRE", "off")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_success_body("<p>ok</p>", prompt_tokens=100)
        )

    p = RewriteProvider(provider="local", client=_make_client(handler))
    assert p._truncation_tripwire is False
    out = p.generate_rewrite(_block())
    # No marker — the tripwire is off.
    assert out.escalation_marker is None


def test_tripwire_noops_when_usage_realistic(monkeypatch):
    # Reported tokens close to the estimate → no truncation → clean emit.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_success_body("<p>ok</p>", prompt_tokens=8800)
        )

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_block())
    assert out.escalation_marker is None
    assert "ok" in out.content


# ---------------------------------------------------------------------------
# Backstops stay wired but no-op on a clean attr-emitting block.
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Defect 1 — the request pins the SERVED window (options.num_ctx == the value
# the budget + tripwire assume). Asserted at the client boundary via a mock.
# ---------------------------------------------------------------------------
def test_request_carries_resolved_num_ctx_on_local_lane(monkeypatch):
    import json

    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "16384")
    seen: List[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.read().decode("utf-8")))
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    p.generate_rewrite(_block())
    # The single source of truth: the served-window option on the wire equals
    # the value the budget + tripwire resolved (resolve_rewrite_num_ctx).
    assert seen[0]["options"]["num_ctx"] == 16384
    assert p._rewrite_num_ctx == 16384


def test_batch_request_carries_resolved_num_ctx(monkeypatch):
    import json

    from Courseforge.scripts.blocks import Block  # noqa: PLC0415

    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "16384")
    seen: List[dict] = []

    def _cf_block(html: str, ids) -> str:
        parts = []
        for bid in ids:
            parts.append(f'<<<CF_BLOCK id="{bid}">>>{html}<<<CF_BLOCK_END id="{bid}">>>')
        return "\n".join(parts)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read().decode("utf-8"))
        seen.append(body)
        ids = [b for b in ("page#concept_a_0", "page#concept_b_0")]
        return httpx.Response(200, json=_success_body(_cf_block("<p>ok</p>", ids)))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    blocks = [
        Block(block_id="page#concept_a_0", block_type="concept", page_id="page",
              sequence=0, content={"key_claims": [], "curies": [],
                                   "source_refs": [], "objective_refs": []}),
        Block(block_id="page#concept_b_0", block_type="concept", page_id="page",
              sequence=1, content={"key_claims": [], "curies": [],
                                   "source_refs": [], "objective_refs": []}),
    ]
    p.generate_rewrite_batch(blocks)
    assert seen and seen[0]["options"]["num_ctx"] == 16384


# ---------------------------------------------------------------------------
# Defect 2 — the ESCALATED render path applies the whole-prompt chunk budget.
# ---------------------------------------------------------------------------
def test_escalated_path_applies_chunk_budget(monkeypatch):
    monkeypatch.setenv("ED4ALL_REWRITE_FIT_WINDOW", "1")
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "8192")
    seen: List[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read())
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    esc = _block()
    esc = esc.__class__(
        **{**esc.__dict__, "escalation_marker": "outline_budget_exhausted"}
    )
    big_chunks = [{"id": f"c{i}", "text": "word " * 400} for i in range(20)]
    p.generate_rewrite(esc, source_chunks=big_chunks)
    body = seen[0].decode("utf-8")
    # The escalated prompt is on the wire (marker echoed) AND the chunk budget
    # dropped the trailing chunks — NOT every chunk id survived.
    assert "ESCALATED REWRITE" in body
    present = sum(1 for i in range(20) if f"[c{i}]" in body)
    assert 1 <= present < 20


# ---------------------------------------------------------------------------
# Defect 2/3 — an oversized non-chunk scaffold is a LOUD escalation, NO dispatch.
# ---------------------------------------------------------------------------
def test_oversized_scaffold_escalates_without_dispatch(monkeypatch):
    monkeypatch.setenv("ED4ALL_REWRITE_FIT_WINDOW", "1")
    # A tiny window the trimmed system prompt + scaffold alone can't fit.
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "6000")
    seen: List[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read())
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_block(), source_chunks=[{"id": "c0", "text": "x"}])
    # No POST was ever made — the prompt that cannot fit is never dispatched.
    assert seen == []
    assert out.escalation_marker == "rewrite_scaffold_overflow"


def test_huge_outline_scaffold_escalates_at_realistic_window(monkeypatch):
    monkeypatch.setenv("ED4ALL_REWRITE_FIT_WINDOW", "1")
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "8192")
    seen: List[bytes] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.read())
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    # A giant outline payload blows the window even though chunks are tiny —
    # the budget bounds the WHOLE prompt, not just the chunks.
    huge = _block()
    huge = huge.__class__(**{
        **huge.__dict__,
        "content": {
            "key_claims": [
                {"text": "concept " * 4000, "source_chunk_ids": []},
            ],
            "curies": [], "source_refs": [], "objective_refs": ["TO-01"],
        },
    })
    out = p.generate_rewrite(huge, source_chunks=[{"id": "c0", "text": "x"}])
    assert seen == []
    assert out.escalation_marker == "rewrite_scaffold_overflow"


# ---------------------------------------------------------------------------
# Defect 3 — the mid-size post-dispatch tripwire catches the 8192-cap case.
# ---------------------------------------------------------------------------
def test_midsize_tripwire_stamps_when_reported_at_window_cap(monkeypatch):
    # fit-window OFF so no budgeting; num_ctx assumed 8192 (matches server).
    monkeypatch.delenv("ED4ALL_REWRITE_FIT_WINDOW", raising=False)
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "8192")

    # Server truncated to its 8192 window and reports ~8194 — ABOVE
    # estimate/2 (a big OFF prompt estimates ~15k). The severe /2 arm would
    # pass; the mid-size arm (reported ~= 8192 cap AND materially below the
    # estimate) trips it.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_success_body("<p>fabricated</p>", prompt_tokens=8194)
        )

    p = RewriteProvider(provider="local", client=_make_client(handler))
    # A big grounding block pushes the local estimate well past ~11k so the
    # 8194 report is materially below it (< estimate*0.75).
    out = p.generate_rewrite(
        _block(), source_chunks=[{"id": "c0", "text": "word " * 2500}]
    )
    assert out.escalation_marker == "input_prompt_truncated"


# ---------------------------------------------------------------------------
# Recalibration (2026-07-09) — the wired PRE-dispatch refusal
# (_check_input_fits_predispatch → check_prompt_fits_window) uses the
# OPTIMISTIC 3.8-c/t divisor, so a prompt whose 2.5-c/t UPPER-bound estimate
# is high but whose REAL token count fits the window DISPATCHES instead of
# being refused (the 397/659-blocks-refused bug).
# ---------------------------------------------------------------------------
def _prompt_for_upper_est(provider, target_total_est: int) -> str:
    """Build a user prompt so _estimate_tokens(sys)+_estimate_tokens(user)
    ≈ target_total_est (the 2.5-c/t upper bound the pre-dispatch check sees)."""
    from lib.retrieval._prompts import estimate_tokens as _est

    sys_tok = _est(provider._system_prompt)
    user_tok = max(0, target_total_est - sys_tok)
    return "x" * int(user_tok * 2.5)


def test_predispatch_dispatches_when_upper_bound_high_but_real_fits(monkeypatch):
    """(a) est ~20000 (2.5-c/t upper bound), num_ctx 16384 → NO refusal:
    optimistic 20000*2.5/3.8 ≈ 13157 < 16384, so the prompt whose real token
    count fits dispatches (does not raise / stamp input_prompt_truncated)."""
    monkeypatch.setenv("ED4ALL_REWRITE_FIT_WINDOW", "1")
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "16384")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    assert p._fit_window is True and p._truncation_tripwire is True
    user = _prompt_for_upper_est(p, 20000)
    # No raise — the calibrated optimistic reading fits the window.
    p._check_input_fits_predispatch(user)


def test_predispatch_refuses_when_optimistic_reading_overflows(monkeypatch):
    """(b) chars/3.8 > num_ctx STILL refuses: est ~26000 → optimistic
    26000*2.5/3.8 ≈ 17105 > 16384 → the wired pre-dispatch check raises
    PromptTruncatedError (a genuine cannot-fit prompt)."""
    from lib.retrieval.answer_backend import PromptTruncatedError

    monkeypatch.setenv("ED4ALL_REWRITE_FIT_WINDOW", "1")
    monkeypatch.setenv("ED4ALL_REWRITE_NUM_CTX", "16384")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body("<p>ok</p>"))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    user = _prompt_for_upper_est(p, 26000)
    with pytest.raises(PromptTruncatedError):
        p._check_input_fits_predispatch(user)


def test_backstops_noop_on_clean_block(monkeypatch):
    """A rewrite that already emits the CURIE token + clean HTML makes the
    str-backstop / force-inject idempotent (no extra hidden span added)."""
    from Courseforge.generators._rewrite_provider import (
        html_has_forced_curie_marker,
    )

    monkeypatch.delenv("ED4ALL_REWRITE_FIT_WINDOW", raising=False)
    # The model emits the CURIE token verbatim in TEXT content already.
    clean_html = (
        '<section data-cf-source-ids="dart:slug#blk1" '
        'data-cf-block-id="page#concept_x_0" '
        'data-cf-objective-id="TO-01">'
        "<h2>The sh:NodeShape concept</h2>"
        "<p>A sh:NodeShape declares a node constraint.</p></section>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(clean_html))

    p = RewriteProvider(provider="local", client=_make_client(handler))
    out = p.generate_rewrite(_block(curies=["sh:NodeShape"]))
    # CURIE already present in prose → no forced-curie hidden span injected
    # (the backstop ran but was a no-op).
    assert not html_has_forced_curie_marker(out.content)
    # The clean attrs survived untouched (backstops didn't rewrite them).
    assert 'data-cf-objective-id="TO-01"' in out.content
    assert "sh:NodeShape" in out.content
