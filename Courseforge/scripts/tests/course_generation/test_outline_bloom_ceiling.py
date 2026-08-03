"""Bloom-ladder WI-07 — outline-tier ceiling clamp.

``OutlineProvider.generate_outline`` already LIFTS an emitted
``bloom_level`` to the max of (emitted, ``block.target_bloom``,
objective-declared floor) so a lazy 7B emit never ships below the
objective's own declared Bloom level (the pre-existing "bloom-diversity
floor" repair). WI-07 adds the SYMMETRIC clamp-down, gated on
``ED4ALL_BLOOM_LADDER`` (``lib.generation.bloom_ladder_blocks.resolve_bloom_ladder``):
a post-lift ``bloom_level`` that ranks ABOVE the objective's declared
ceiling is clamped back down to that ceiling instead of shipping above
it.

Coverage:

- Flag OFF (default/unset/garbage): byte-identical to pre-WI-07 —
  the floor-lift alone can push ``bloom_level`` above the objective's
  own declared level (e.g. via a high ``block.target_bloom``) and no
  clamp fires; exactly one decision-capture event (``block_outline_call``).
- Flag ON: an emit that survives the floor-lift ABOVE the objective
  ceiling is clamped back down to the ceiling; a second decision-capture
  event (``bloom_ladder_ceiling_enforcement``) fires with a dynamic,
  ``>=20``-char rationale interpolating the block id / emitted level /
  ceiling.
- Flag ON, no-op case: an emit at or below the objective ceiling is left
  untouched — no clamp event, ``bloom_level`` unchanged.
- Truthy-token parsing mirrors the shared ``_TRUTHY`` contract
  (``1``/``true``/``yes``/``on``; anything else off).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from Courseforge.generators.outline._outline_provider import (  # noqa: E402
    OutlineProvider,
    _OUTLINE_KIND_BOUNDS,
)
from lib.generation.bloom_ladder_blocks import ENV_BLOOM_LADDER  # noqa: E402
from blocks import Block  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers (mirrors Courseforge/generators/tests/test_outline_provider.py)
# ---------------------------------------------------------------------------


def _success_body(content: str, *, model: str = "test-outline") -> dict:
    return {
        "id": "cmpl-outline-ceiling-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            # A realistic served prompt-token count so the default-ON
            # input-truncation tripwire never false-fires on this stub.
            "prompt_tokens": 8000,
            "completion_tokens": 80,
            "total_tokens": 8080,
        },
    }


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response]
) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def _stub_block(*, target_bloom: str = None) -> Block:
    return Block(
        block_id="page-1#concept_intro_0",
        block_type="concept",
        page_id="page-1",
        sequence=0,
        content="",
        target_bloom=target_bloom,
    )


def _valid_outline_payload(
    *, block_type: str = "concept", bloom_level: str = "understand"
) -> Dict[str, Any]:
    bounds = _OUTLINE_KIND_BOUNDS.get(block_type, {})
    section_min, _section_max = bounds.get("section_skeleton", (0, 0))
    return {
        "block_id": "page-1#concept_intro_0",
        "block_type": block_type,
        "content_type": "explanation",
        "bloom_level": bloom_level,
        "objective_refs": ["TO-01"],
        "curies": ["sh:NodeShape"],
        "key_claims": ["The central concept is X."],
        "section_skeleton": (
            [{"heading": "Definition"} for _ in range(max(section_min, 1))]
            if section_min > 0
            else []
        ),
        "source_refs": [{"sourceId": "semantik:slug#blk1", "role": "primary"}],
        "structural_warnings": [],
    }


class _FakeCapture:
    """Capture stub mirroring the production ``DecisionCapture.events`` shape."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def _make_provider(
    *, capture: _FakeCapture, payload: Dict[str, Any], monkeypatch
) -> OutlineProvider:
    monkeypatch.delenv("COURSEFORGE_OUTLINE_PROVIDER", raising=False)
    monkeypatch.delenv("LOCAL_SYNTHESIS_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_BASE_URL", "http://localhost:11434/v1")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_success_body(json.dumps(payload)))

    return OutlineProvider(
        provider="local", capture=capture, client=_make_client(handler)
    )


# ---------------------------------------------------------------------------
# Flag OFF — byte-identical to pre-WI-07 behavior
# ---------------------------------------------------------------------------


def test_flag_off_ceiling_not_enforced_default(monkeypatch):
    """``ED4ALL_BLOOM_LADDER`` unset: the floor-lift can push ``bloom_level``
    above the objective's declared ceiling (here via a high
    ``block.target_bloom``) and no clamp fires — exactly one capture event
    (``block_outline_call``), and the shipped level is the LIFTED (not
    clamped) value."""
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    capture = _FakeCapture()
    payload = _valid_outline_payload(bloom_level="understand")
    p = _make_provider(capture=capture, payload=payload, monkeypatch=monkeypatch)

    block = _stub_block(target_bloom="create")
    out = p.generate_outline(
        block,
        source_chunks=[],
        objectives=[{"bloom_level": "apply"}],
    )

    # Floor-lift alone takes bloom_level to "create" (max of emitted
    # "understand" / target "create" / objective-floor "apply"); with the
    # ladder gate off, nothing clamps it back down.
    assert out.content["bloom_level"] == "create"
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == "block_outline_call"


def test_flag_off_garbage_token_also_no_clamp(monkeypatch):
    """A garbage / non-truthy token for ``ED4ALL_BLOOM_LADDER`` resolves
    to off — same as unset."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "definitely-not-truthy")
    capture = _FakeCapture()
    payload = _valid_outline_payload(bloom_level="understand")
    p = _make_provider(capture=capture, payload=payload, monkeypatch=monkeypatch)

    block = _stub_block(target_bloom="create")
    out = p.generate_outline(
        block,
        source_chunks=[],
        objectives=[{"bloom_level": "apply"}],
    )

    assert out.content["bloom_level"] == "create"
    assert len(capture.events) == 1


# ---------------------------------------------------------------------------
# Flag ON — ceiling clamp fires
# ---------------------------------------------------------------------------


def test_flag_on_clamps_above_objective_ceiling(monkeypatch):
    """``ED4ALL_BLOOM_LADDER=true``: a floor-lifted ``bloom_level`` that
    ranks ABOVE the objective's declared ceiling is clamped back down to
    the ceiling, and a ``bloom_ladder_ceiling_enforcement`` decision fires
    with a dynamic, >=20-char rationale."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    capture = _FakeCapture()
    payload = _valid_outline_payload(bloom_level="understand")
    p = _make_provider(capture=capture, payload=payload, monkeypatch=monkeypatch)

    block = _stub_block(target_bloom="create")
    out = p.generate_outline(
        block,
        source_chunks=[],
        objectives=[{"bloom_level": "apply"}],
    )

    # Clamped DOWN to the objective ceiling ("apply"), not left at the
    # lifted "create".
    assert out.content["bloom_level"] == "apply"

    assert len(capture.events) == 2
    decision_types = [e["decision_type"] for e in capture.events]
    assert "block_outline_call" in decision_types
    assert "bloom_ladder_ceiling_enforcement" in decision_types

    clamp_event = next(
        e for e in capture.events
        if e["decision_type"] == "bloom_ladder_ceiling_enforcement"
    )
    rationale = clamp_event["rationale"]
    assert len(rationale) >= 20
    # Dynamic interpolation: block id + emitted/ceiling levels, not a
    # static boilerplate string.
    assert block.block_id in rationale
    assert "create" in rationale
    assert "apply" in rationale
    assert clamp_event["decision"]
    assert block.block_id in clamp_event["decision"]


def test_flag_on_truthy_tokens_all_enforce(monkeypatch):
    """Every canonical truthy token (``1``/``true``/``yes``/``on``,
    case-insensitive) enables the clamp identically."""
    for token in ("1", "TRUE", "yes", "On"):
        monkeypatch.setenv(ENV_BLOOM_LADDER, token)
        capture = _FakeCapture()
        payload = _valid_outline_payload(bloom_level="understand")
        p = _make_provider(
            capture=capture, payload=payload, monkeypatch=monkeypatch
        )
        block = _stub_block(target_bloom="create")
        out = p.generate_outline(
            block, source_chunks=[], objectives=[{"bloom_level": "apply"}]
        )
        assert out.content["bloom_level"] == "apply", f"token={token!r}"
        assert len(capture.events) == 2, f"token={token!r}"


# ---------------------------------------------------------------------------
# Flag ON — no-op when already within ceiling
# ---------------------------------------------------------------------------


def test_flag_on_noop_when_within_ceiling(monkeypatch):
    """``ED4ALL_BLOOM_LADDER=true`` but the floor-lifted ``bloom_level``
    never exceeds the objective ceiling (no ``target_bloom`` override
    above it): no clamp fires, only the one ``block_outline_call``
    event."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    capture = _FakeCapture()
    payload = _valid_outline_payload(bloom_level="understand")
    p = _make_provider(capture=capture, payload=payload, monkeypatch=monkeypatch)

    block = _stub_block(target_bloom=None)
    out = p.generate_outline(
        block,
        source_chunks=[],
        objectives=[{"bloom_level": "apply"}],
    )

    # Floor-lift takes "understand" -> "apply" (the objective floor);
    # "apply" is exactly the ceiling, not above it, so no clamp.
    assert out.content["bloom_level"] == "apply"
    assert len(capture.events) == 1
    assert capture.events[0]["decision_type"] == "block_outline_call"


def test_flag_on_no_objectives_no_ceiling_to_clamp_to(monkeypatch):
    """No objectives supplied -> no declared ceiling exists; the clamp
    guard (``objective_declared_floor is not None``) prevents it from
    firing against a fabricated ceiling."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "true")
    capture = _FakeCapture()
    payload = _valid_outline_payload(bloom_level="create")
    p = _make_provider(capture=capture, payload=payload, monkeypatch=monkeypatch)

    block = _stub_block(target_bloom=None)
    out = p.generate_outline(block, source_chunks=[], objectives=[])

    assert out.content["bloom_level"] == "create"
    assert len(capture.events) == 1
