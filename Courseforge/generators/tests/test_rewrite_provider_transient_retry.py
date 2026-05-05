"""Worker W6 — transient-retry budget tests for ``RewriteProvider``.

A transient Ollama 503 / connection reset / read timeout MUST NOT burn
the rewrite-tier parse-retry budget (``MAX_PARSE_RETRIES``); a permanent
error (authentication, bad request) MUST surface immediately; once the
per-block transient-retry budget (``_TRANSIENT_RETRY_BUDGET``) is
exhausted, the failure MUST surface as a distinct
``RewriteProviderError(code="rewrite_transient_exhausted")``.

Mirrors the helper conventions in
``Courseforge/generators/tests/test_rewrite_provider.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._rewrite_provider import (  # noqa: E402
    MAX_PARSE_RETRIES,
    RewriteProvider,
    RewriteProviderError,
    _TRANSIENT_RETRY_BUDGET,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirror test_rewrite_provider.py)
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-rewrite") -> dict:
    return {
        "id": "cmpl-rewrite-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 200,
            "completion_tokens": 80,
            "total_tokens": 280,
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _outline_block(
    *,
    block_type: str = "concept",
    curies: List[str] | None = None,
    escalation_marker: str | None = None,
) -> Block:
    return Block(
        block_id="page#concept_intro_0",
        block_type=block_type,
        page_id="page",
        sequence=0,
        content={
            "key_claims": ["The central concept is X."],
            "curies": list(curies or []),
            "source_refs": ["dart:slug#blk1"],
            "objective_refs": ["TO-01"],
        },
        escalation_marker=escalation_marker,
    )


# ---------------------------------------------------------------------------
# Worker W6 contract — transient retries do NOT burn the parse budget.
# ---------------------------------------------------------------------------


def test_transient_does_not_burn_parse_budget(monkeypatch):
    """``_dispatch_call`` raises a TRANSIENT-classified exception twice
    then returns valid HTML. The transient retries MUST NOT advance the
    ``MAX_PARSE_RETRIES`` parse-budget counter; the third dispatch must
    succeed and return a Block."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    success_html = (
        "<section data-cf-source-ids=\"dart:slug#blk1\">"
        "<p>The central concept is X.</p></section>"
    )
    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(success_html))
        ),
    )

    call_count = {"n": 0}
    real_dispatch = p._dispatch_call

    def fake_dispatch(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            # ``connection reset`` matches the classifier's TRANSIENT
            # regex (``connection (refused|reset|closed)``).
            raise ConnectionError("connection reset by peer")
        return real_dispatch(*args, **kwargs)

    monkeypatch.setattr(p, "_dispatch_call", fake_dispatch)

    # No CURIEs → the post-dispatch CURIE-preservation gate accepts on
    # first emit, so the test isolates the transient-retry behavior.
    block = _outline_block(curies=[])
    out = p.generate_rewrite(block)

    assert isinstance(out, Block), "third dispatch should succeed"
    # Two transient retries + one successful dispatch = 3 total calls.
    assert call_count["n"] == 3


def test_permanent_surfaces_immediately(monkeypatch):
    """A PERMANENT-classified error (e.g. ``401 unauthorized``) MUST
    re-raise on its first occurrence — no retry, no parse-budget
    advance, no swallow."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("<p>ok</p>"))
        ),
    )

    call_count = {"n": 0}

    def fake_dispatch(*args, **kwargs):
        call_count["n"] += 1
        raise PermissionError("401 unauthorized: invalid api key")

    monkeypatch.setattr(p, "_dispatch_call", fake_dispatch)

    block = _outline_block(curies=[])
    with pytest.raises(PermissionError):
        p.generate_rewrite(block)
    # Single dispatch — no retry on permanent.
    assert call_count["n"] == 1


def test_transient_budget_exhaustion_yields_distinct_code(monkeypatch):
    """``_TRANSIENT_RETRY_BUDGET + 1`` consecutive transient failures MUST
    surface as ``RewriteProviderError(code="rewrite_transient_exhausted")``,
    distinct from the CURIE-drop ``rewrite_curie_drop`` code so the
    router can branch on the failure mode."""
    monkeypatch.delenv("COURSEFORGE_REWRITE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    p = RewriteProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("<p>ok</p>"))
        ),
    )

    call_count = {"n": 0}

    def fake_dispatch(*args, **kwargs):
        call_count["n"] += 1
        raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(p, "_dispatch_call", fake_dispatch)

    block = _outline_block(curies=[])
    with pytest.raises(RewriteProviderError) as excinfo:
        p.generate_rewrite(block)
    assert excinfo.value.code == "rewrite_transient_exhausted"
    assert excinfo.value.code != "rewrite_curie_drop"
    # Up to ``_TRANSIENT_RETRY_BUDGET`` retries on top of the initial
    # dispatch attempt; the (budget+1)-th transient triggers the raise.
    assert call_count["n"] == _TRANSIENT_RETRY_BUDGET + 1


def test_transient_budget_constant_is_three():
    """Sanity check: the per-block transient-retry budget is the
    Worker-W6-specified value (3). If the constant changes the test
    above must follow."""
    assert _TRANSIENT_RETRY_BUDGET == 3
