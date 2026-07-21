"""outline-overflow-fix-2026-07 regression tests for ``OutlineProvider``.

Two guards defend the outline tier against silent head-truncation on a
small-ctx served model:

1. **Chunk-count cap** (``COURSEFORGE_OUTLINE_MAX_CHUNKS``, default 8):
   ``_render_user_prompt`` renders at most K source chunks (head-K over the
   already-ordered list), on top of the pre-existing per-chunk 1200-char cap.

2. **Input-truncation tripwire** (``COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE``,
   default ON): after each dispatch, the server-reported
   ``usage.prompt_tokens`` is compared against the local estimate; a large
   shortfall (the served window silently dropped the prompt HEAD) fails the
   call with ``OutlineProviderError(code="outline_input_truncated")`` instead
   of returning a silent-success wrong-topic outline. Off → passes through.

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
    DEFAULT_MAX_CHUNKS,
    OutlineProvider,
    OutlineProviderError,
    _OUTLINE_KIND_BOUNDS,
    _resolve_outline_max_chunks,
    _resolve_outline_truncation_tripwire,
)
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirror test_outline_provider.py)
# ---------------------------------------------------------------------------


def _success_body(
    content: str,
    *,
    model: str = "test-outline",
    prompt_tokens: int = 8000,
) -> dict:
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 80,
            "total_tokens": prompt_tokens + 80,
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
        "source_refs": [{"sourceId": "semantik:slug#blk1", "role": "primary"}],
        "structural_warnings": [],
    }
    return payload


def _chunks(n: int) -> List[Dict[str, Any]]:
    """N distinct source chunks with ordered, greppable ids + bodies."""
    return [
        {"id": f"chunk-{i:02d}", "body": f"Body of chunk {i:02d}. " * 3}
        for i in range(n)
    ]


def _provider(monkeypatch, **kwargs: Any) -> OutlineProvider:
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv(
        "LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1"
    )
    kwargs.setdefault(
        "client",
        _make_client(
            lambda r: httpx.Response(
                200, json=_success_body(json.dumps(_valid_outline_payload()))
            )
        ),
    )
    return OutlineProvider(provider="local", **kwargs)


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def test_max_chunks_default_is_eight():
    assert DEFAULT_MAX_CHUNKS == 8
    assert _resolve_outline_max_chunks({}) == 8


def test_max_chunks_env_override_and_garbage_fallback():
    assert _resolve_outline_max_chunks({"COURSEFORGE_OUTLINE_MAX_CHUNKS": "3"}) == 3
    # Garbage / non-positive → the default (never disables the guard).
    assert _resolve_outline_max_chunks({"COURSEFORGE_OUTLINE_MAX_CHUNKS": "0"}) == 8
    assert _resolve_outline_max_chunks({"COURSEFORGE_OUTLINE_MAX_CHUNKS": "x"}) == 8


def test_truncation_tripwire_default_on_escape_hatch():
    assert _resolve_outline_truncation_tripwire({}) is True
    assert _resolve_outline_truncation_tripwire(
        {"COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE": "off"}
    ) is False
    assert _resolve_outline_truncation_tripwire(
        {"COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE": "0"}
    ) is False
    # Garbage / truthy → on.
    assert _resolve_outline_truncation_tripwire(
        {"COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE": "garbage"}
    ) is True


# ---------------------------------------------------------------------------
# (b) Chunk-count cap
# ---------------------------------------------------------------------------


def test_render_caps_chunks_to_default_k_preserving_order(monkeypatch):
    """>K chunks in → exactly K rendered, first-K order preserved."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MAX_CHUNKS", raising=False)
    p = _provider(monkeypatch)
    chunks = _chunks(12)  # 12 > default 8
    prompt = p._render_user_prompt(
        block=_stub_block(), source_chunks=chunks, objectives=[]
    )
    rendered = [c["id"] for c in chunks if f"[{c['id']}]" in prompt]
    assert rendered == [f"chunk-{i:02d}" for i in range(DEFAULT_MAX_CHUNKS)]
    # The dropped chunks (9..12) must NOT appear.
    for i in range(DEFAULT_MAX_CHUNKS, 12):
        assert f"[chunk-{i:02d}]" not in prompt


def test_render_cap_env_override(monkeypatch):
    """``COURSEFORGE_OUTLINE_MAX_CHUNKS`` overrides the default K."""
    monkeypatch.setenv("COURSEFORGE_OUTLINE_MAX_CHUNKS", "3")
    p = _provider(monkeypatch)
    chunks = _chunks(10)
    prompt = p._render_user_prompt(
        block=_stub_block(), source_chunks=chunks, objectives=[]
    )
    rendered = [c["id"] for c in chunks if f"[{c['id']}]" in prompt]
    assert rendered == ["chunk-00", "chunk-01", "chunk-02"]


def test_render_below_cap_renders_all(monkeypatch):
    """Fewer than K chunks → all rendered (no drop)."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MAX_CHUNKS", raising=False)
    p = _provider(monkeypatch)
    chunks = _chunks(4)
    prompt = p._render_user_prompt(
        block=_stub_block(), source_chunks=chunks, objectives=[]
    )
    rendered = [c["id"] for c in chunks if f"[{c['id']}]" in prompt]
    assert rendered == [f"chunk-{i:02d}" for i in range(4)]


def test_per_chunk_char_cap_preserved(monkeypatch):
    """The pre-existing per-chunk 1200-char body cap is unchanged."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_MAX_CHUNKS", raising=False)
    p = _provider(monkeypatch)
    long_chunk = [{"id": "chunk-long", "body": "x" * 5000}]
    prompt = p._render_user_prompt(
        block=_stub_block(), source_chunks=long_chunk, objectives=[]
    )
    # Body truncated at 1197 + "..." per the existing 1200-char cap.
    assert "x" * 1197 + "..." in prompt
    assert "x" * 1201 not in prompt


# ---------------------------------------------------------------------------
# (c) Input-truncation tripwire
# ---------------------------------------------------------------------------


def test_tripwire_on_fails_call_on_reported_shortfall(monkeypatch):
    """usage.prompt_tokens far below the local estimate → the call FAILS
    with ``OutlineProviderError(code="outline_input_truncated")`` (not a
    silent-success stub)."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE", raising=False)
    # Server reports a tiny prompt-token count → simulated head-truncation.
    client = _make_client(
        lambda r: httpx.Response(
            200,
            json=_success_body(
                json.dumps(_valid_outline_payload()), prompt_tokens=50
            ),
        )
    )
    p = _provider(monkeypatch, client=client)
    with pytest.raises(OutlineProviderError) as excinfo:
        p.generate_outline(
            _stub_block(), source_chunks=_chunks(2), objectives=[]
        )
    assert excinfo.value.code == "outline_input_truncated"


def test_tripwire_off_passes_through(monkeypatch):
    """Tripwire off → the same reported shortfall does NOT fail the call;
    a valid payload parses and returns a Block."""
    monkeypatch.setenv("COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE", "off")
    client = _make_client(
        lambda r: httpx.Response(
            200,
            json=_success_body(
                json.dumps(_valid_outline_payload()), prompt_tokens=50
            ),
        )
    )
    p = _provider(monkeypatch, client=client)
    out = p.generate_outline(
        _stub_block(), source_chunks=_chunks(2), objectives=[]
    )
    assert isinstance(out, Block)
    assert out.escalation_marker is None


def test_tripwire_fail_open_on_realistic_usage(monkeypatch):
    """A realistic reported prompt-token count (no shortfall) passes even
    with the tripwire ON — the guard is conservative, not a hair-trigger."""
    monkeypatch.delenv("COURSEFORGE_OUTLINE_TRUNCATION_TRIPWIRE", raising=False)
    client = _make_client(
        lambda r: httpx.Response(
            200,
            json=_success_body(
                json.dumps(_valid_outline_payload()), prompt_tokens=8000
            ),
        )
    )
    p = _provider(monkeypatch, client=client)
    out = p.generate_outline(
        _stub_block(), source_chunks=_chunks(2), objectives=[]
    )
    assert isinstance(out, Block)
    assert out.escalation_marker is None
