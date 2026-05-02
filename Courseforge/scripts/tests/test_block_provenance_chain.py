"""Phase 2 Subtask 34 — provenance audit test.

Synthesises a 3-tier touch chain on a single Block (outline →
validation → rewrite) and asserts the audit invariants:

  * ``len(b.touched_by) == 3``
  * Strictly monotonically increasing timestamps
  * Every ``Touch.decision_capture_id`` is non-empty
    (Wave 112 invariant — the constructor enforces this; we
    re-assert it on the rendered chain to keep the audit explicit)
  * Tier sequence is exactly ``["outline", "validation", "rewrite"]``
  * The ``touchedBy`` JSON-LD projection is a 3-element list whose
    entries carry the right ``tier`` / ``provider`` / ``model``
    fields per ``Touch.to_jsonld``

This test does NOT need ``COURSEFORGE_EMIT_BLOCKS`` — the touch chain
operates on the in-memory Block + Touch dataclasses, independent of
the emit-time HTML/JSON-LD plumbing. Phase 3's regeneration router
will be the producer of the chain in production; Phase 2 lands the
audit-level invariants.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from blocks import Block, Touch  # noqa: E402


def _isoformat(dt: datetime) -> str:
    """ISO-8601 with explicit 'Z' suffix mirroring the Phase-2 emit
    convention (see ``Courseforge/scripts/blocks.py::Touch``).
    """
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_chain_three_tiers() -> Block:
    """Synthesise the canonical 3-tier touch chain on a section block.

    The base Block carries an ``explanation`` block_type so it traverses
    the section-jsonld emit path; touches are spaced by 1 second so the
    timestamp monotonicity assertion is unambiguous.
    """
    base = Block(
        block_id=Block.stable_id("page_audit", "explanation", "intro", 0),
        block_type="explanation",
        page_id="page_audit",
        sequence=0,
        content="Section heading for audit",
        content_type_label="explanation",
        key_terms=("term_one",),
    )
    t0 = datetime(2026, 5, 2, 12, 0, 0, tzinfo=timezone.utc)
    outline = Touch(
        model="qwen2.5-14b",
        provider="local",
        tier="outline",
        timestamp=_isoformat(t0),
        decision_capture_id="decisions_20260502_120000.jsonl:0",
        purpose="draft",
    )
    validation = Touch(
        model="qwen2.5-14b",
        provider="local",
        tier="validation",
        timestamp=_isoformat(t0 + timedelta(seconds=1)),
        decision_capture_id="decisions_20260502_120000.jsonl:1",
        purpose="validate",
    )
    rewrite = Touch(
        model="claude-sonnet-4",
        provider="anthropic",
        tier="rewrite",
        timestamp=_isoformat(t0 + timedelta(seconds=2)),
        decision_capture_id="decisions_20260502_120000.jsonl:2",
        purpose="rewrite",
    )
    return base.with_touch(outline).with_touch(validation).with_touch(rewrite)


def test_three_tier_provenance_chain_audit_invariants() -> None:
    """All five audit invariants on a synthesised 3-tier chain."""
    b = _make_chain_three_tiers()

    # 1. Chain length is exactly three.
    assert len(b.touched_by) == 3

    # 2. Timestamps are strictly monotonically increasing.
    timestamps = [t.timestamp for t in b.touched_by]
    parsed = [datetime.fromisoformat(ts.replace("Z", "+00:00")) for ts in timestamps]
    for i in range(1, len(parsed)):
        assert parsed[i] > parsed[i - 1], (
            f"Touch timestamps must be strictly monotonically "
            f"increasing; chain[{i}]={timestamps[i]} not > "
            f"chain[{i - 1}]={timestamps[i - 1]}"
        )

    # 3. Every decision_capture_id is non-empty (Wave 112 invariant).
    #    The constructor already raises on empty values, but we re-
    #    assert on the rendered chain so an audit replay catches any
    #    future regression that bypasses the constructor.
    for i, t in enumerate(b.touched_by):
        assert t.decision_capture_id, (
            f"Touch[{i}].decision_capture_id violates Wave 112 invariant "
            "(must be non-empty)"
        )

    # 4. Tier sequence is exactly the canonical Phase 3 cascade.
    tiers = [t.tier for t in b.touched_by]
    assert tiers == ["outline", "validation", "rewrite"], (
        f"Tier sequence drift: got {tiers!r}"
    )

    # 5. JSON-LD projection: 3 entries, right tier/provider/model.
    entry = b.to_jsonld_entry()
    # Section-shape entries don't carry ``touchedBy`` natively (only
    # the Phase-2 minimal shape does), so we assert directly on the
    # ``Touch.to_jsonld()`` projection of the chain.
    rendered = [t.to_jsonld() for t in b.touched_by]
    assert len(rendered) == 3
    assert rendered[0]["tier"] == "outline"
    assert rendered[0]["provider"] == "local"
    assert rendered[0]["model"] == "qwen2.5-14b"
    assert rendered[1]["tier"] == "validation"
    assert rendered[1]["provider"] == "local"
    assert rendered[2]["tier"] == "rewrite"
    assert rendered[2]["provider"] == "anthropic"
    assert rendered[2]["model"] == "claude-sonnet-4"
    # Wire keys are camelCase (Phase 2 convention).
    for r in rendered:
        assert "decisionCaptureId" in r
        assert r["decisionCaptureId"]  # non-empty
        assert "decision_capture_id" not in r  # snake_case must NOT leak

    # And the section-shape JSON-LD entry must still validate even
    # though it doesn't surface ``touchedBy``: we want a non-empty
    # heading (audit chain doesn't corrupt downstream emit).
    assert entry.get("heading") == "Section heading for audit"


def test_phase2_minimal_block_jsonld_surfaces_touched_by_chain() -> None:
    """Phase-2 minimal block_type entries (e.g. ``self_check_question``)
    DO surface the ``touchedBy`` chain on their JSON-LD entry — the
    canonical audit path for non-section blocks. Round-trip the chain
    through that surface.
    """
    base = Block(
        block_id="page_audit#self_check_question_q1_0",
        block_type="self_check_question",
        page_id="page_audit",
        sequence=0,
        content={"question": "?"},
        bloom_level="apply",
    )
    t0 = datetime(2026, 5, 2, 12, 0, 0, tzinfo=timezone.utc)
    chain = base.with_touch(
        Touch(
            model="qwen2.5-14b",
            provider="local",
            tier="outline",
            timestamp=_isoformat(t0),
            decision_capture_id="d:0",
            purpose="draft",
        )
    ).with_touch(
        Touch(
            model="qwen2.5-14b",
            provider="local",
            tier="validation",
            timestamp=_isoformat(t0 + timedelta(seconds=1)),
            decision_capture_id="d:1",
            purpose="validate",
        )
    ).with_touch(
        Touch(
            model="claude-sonnet-4",
            provider="anthropic",
            tier="rewrite",
            timestamp=_isoformat(t0 + timedelta(seconds=2)),
            decision_capture_id="d:2",
            purpose="rewrite",
        )
    )

    entry = chain.to_jsonld_entry()
    # Phase-2 minimal shape carries the chain inline.
    assert "touchedBy" in entry, entry
    assert isinstance(entry["touchedBy"], list)
    assert len(entry["touchedBy"]) == 3
    assert [t["tier"] for t in entry["touchedBy"]] == [
        "outline",
        "validation",
        "rewrite",
    ]
    assert entry["touchedBy"][0]["provider"] == "local"
    assert entry["touchedBy"][2]["provider"] == "anthropic"
    assert entry["touchedBy"][2]["model"] == "claude-sonnet-4"
