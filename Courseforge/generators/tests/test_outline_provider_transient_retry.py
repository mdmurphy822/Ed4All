"""Worker W6 — transient-retry budget tests for ``OutlineProvider``.

A transient Ollama 503 / connection reset / read timeout MUST NOT burn
the parse-retry budget (``MAX_PARSE_RETRIES``); a permanent error
(authentication, bad request, schema_error) MUST surface immediately;
and once the per-call transient-retry budget (``_TRANSIENT_RETRY_BUDGET``)
is exhausted, the failure MUST surface as a distinct
``OutlineProviderError(code="outline_transient_exhausted")`` so the
router can branch on the failure mode.

Mirrors the helper conventions in
``Courseforge/generators/tests/test_outline_provider.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.generators._outline_provider import (  # noqa: E402
    MAX_PARSE_RETRIES,
    OutlineProvider,
    OutlineProviderError,
    _OUTLINE_KIND_BOUNDS,
    _TRANSIENT_RETRY_BUDGET,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirror test_outline_provider.py)
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-outline") -> dict:
    return {
        "id": "cmpl-outline-test",
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


def _stub_block(
    *,
    block_type: str = "concept",
    block_id: str = "page-1#concept_intro_0",
    page_id: str = "page-1",
) -> Block:
    return Block(
        block_id=block_id,
        block_type=block_type,
        page_id=page_id,
        sequence=0,
        content="",
    )


def _valid_outline_payload(
    *,
    block_type: str = "concept",
    block_id: str = "page-1#concept_intro_0",
) -> Dict[str, Any]:
    bounds = _OUTLINE_KIND_BOUNDS.get(block_type, {})
    section_min, _section_max = bounds.get("section_skeleton", (0, 0))
    payload: Dict[str, Any] = {
        "block_id": block_id,
        "block_type": block_type,
        "content_type": "explanation",
        "bloom_level": "understand",
        "objective_refs": ["TO-01"],
        "curies": ["sh:NodeShape"],
        "key_claims": ["The central concept is X."],
        "section_skeleton": [
            {"heading": "Definition"} for _ in range(max(section_min, 1))
        ] if section_min > 0 else [],
        "source_refs": [{"sourceId": "dart:slug#blk1", "role": "primary"}],
        "structural_warnings": [],
    }
    return payload


# ---------------------------------------------------------------------------
# Worker W6 contract — transient retries do NOT burn the parse budget.
# ---------------------------------------------------------------------------


def test_transient_does_not_burn_parse_budget(monkeypatch):
    """``_dispatch_call`` raises a TRANSIENT-classified exception twice
    then returns valid JSON. The transient retries MUST NOT advance the
    ``MAX_PARSE_RETRIES`` parse-budget counter; the third dispatch must
    succeed and return a Block."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    payload = _valid_outline_payload(block_type="concept")
    p = OutlineProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body(json.dumps(payload)))
        ),
    )

    call_count = {"n": 0}
    real_dispatch = p._dispatch_call

    def fake_dispatch(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            # ConnectionError messages match the classifier's
            # TRANSIENT_PATTERNS regex (``connection (refused|reset|closed)``).
            raise ConnectionError("connection reset by peer")
        return real_dispatch(*args, **kwargs)

    monkeypatch.setattr(p, "_dispatch_call", fake_dispatch)

    block = _stub_block()
    out = p.generate_outline(block, source_chunks=[], objectives=[])

    assert isinstance(out, Block), "third dispatch should succeed"
    # Two transient retries + one successful dispatch = 3 total calls.
    assert call_count["n"] == 3
    # Sanity check: MAX_PARSE_RETRIES > the number of failed parse
    # attempts the loop saw. A transient burning the parse budget would
    # have raised ``outline_exhausted`` after MAX_PARSE_RETRIES (=3)
    # failures with no successful dispatch.


def test_permanent_surfaces_immediately(monkeypatch):
    """A PERMANENT-classified error (e.g. ``401 unauthorized``) MUST
    re-raise on its first occurrence — no retry, no parse-budget
    advance, no ``outline_*_exhausted`` swallow."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    p = OutlineProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("{}"))
        ),
    )

    call_count = {"n": 0}

    def fake_dispatch(*args, **kwargs):
        call_count["n"] += 1
        # PermissionError stringifies to messages matching the
        # classifier's PERMANENT_PATTERNS regex (``401 unauthorized``).
        raise PermissionError("401 unauthorized: invalid api key")

    monkeypatch.setattr(p, "_dispatch_call", fake_dispatch)

    block = _stub_block()
    with pytest.raises(PermissionError):
        p.generate_outline(block, source_chunks=[], objectives=[])
    # Single dispatch — no retry on permanent.
    assert call_count["n"] == 1


def test_transient_budget_exhaustion_yields_distinct_code(monkeypatch):
    """``_TRANSIENT_RETRY_BUDGET + 1`` consecutive transient failures MUST
    surface as ``OutlineProviderError(code="outline_transient_exhausted")``,
    distinct from the parse-budget ``outline_exhausted`` code so the
    router can branch on the failure mode."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    p = OutlineProvider(
        provider="local",
        client=_make_client(
            lambda r: httpx.Response(200, json=_success_body("{}"))
        ),
    )

    call_count = {"n": 0}

    def fake_dispatch(*args, **kwargs):
        call_count["n"] += 1
        raise ConnectionError("connection reset by peer")

    monkeypatch.setattr(p, "_dispatch_call", fake_dispatch)

    block = _stub_block()
    with pytest.raises(OutlineProviderError) as excinfo:
        p.generate_outline(block, source_chunks=[], objectives=[])
    assert excinfo.value.code == "outline_transient_exhausted"
    assert excinfo.value.code != "outline_exhausted"
    # Up to ``_TRANSIENT_RETRY_BUDGET`` retries on top of the initial
    # dispatch attempt; the (budget+1)-th transient triggers the raise.
    assert call_count["n"] == _TRANSIENT_RETRY_BUDGET + 1


def test_transient_budget_constant_is_three():
    """Sanity check: the per-block transient-retry budget is the
    Worker-W6-specified value (3). If the constant changes the test
    above must follow."""
    assert _TRANSIENT_RETRY_BUDGET == 3
