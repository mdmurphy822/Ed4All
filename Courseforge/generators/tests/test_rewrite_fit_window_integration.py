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
