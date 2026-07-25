"""W2 §6 — block↔objective alignment invariants (hermetic fake embed)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block  # noqa: E402
from lib.objectives.block_alignment import (  # noqa: E402
    _NO_ALIGNMENT_MARKER,
    align_blocks_to_objectives,
)
from lib.objectives.tests._fakes import FakeEmbed  # noqa: E402


def _block(block_id, content):
    return Block(
        block_id=block_id,
        block_type="concept",
        page_id="week_01_content_01",
        sequence=0,
        content=content,
    )


def _content(key_claims, objective_refs=None):
    return {
        "block_id": "b",
        "block_type": "concept",
        "key_claims": key_claims,
        "objective_refs": objective_refs or [],
        "source_refs": [],
        "structural_warnings": [],
    }


class _RecordingCapture:
    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def test_alignment_adds_missed_objective_ref():
    """A block whose claims match CO-02 but omitted the ref gets it ADDED.

    Also the capture-quality regression for the ``block_objective_alignment``
    decision event: it must fire with a real ``inputs_ref`` (the block) and
    genuine ``alternatives_considered`` (the cosine-scored-but-rejected
    objective candidates) so the row rates proficient instead of developing.
    """
    objectives = [
        {"id": "CO-01", "statement": "Explain the basics of medieval heraldry."},
        {"id": "CO-02", "statement": "Compute the area of a triangle from base and height."},
    ]
    block = _block(
        "b1",
        _content(
            ["The area of a triangle equals half base times height computed."],
            objective_refs=[],
        ),
    )
    capture = _RecordingCapture()
    # threshold=0.4 sits between the FakeEmbed cosines (CO-01 ~0.30 vs
    # CO-02 ~0.59) so exactly one candidate is added and the other is a
    # genuinely scored-and-rejected alternative.
    out = align_blocks_to_objectives(
        [block], objectives, embed=FakeEmbed(), threshold=0.4,
        capture=capture,
    )
    refs = out[0].content["objective_refs"]
    assert "CO-02" in refs
    assert "CO-01" not in refs

    events = [
        c for c in capture.calls
        if c.get("decision_type") == "block_objective_alignment"
    ]
    assert len(events) == 1, "alignment capture must fire once per block"
    event = events[0]
    assert {"source_type": "block", "path_or_id": "b1"} in event["inputs_ref"]
    alternatives = event["alternatives_considered"]
    # CO-01 was cosine-scored against the block and rejected below the
    # threshold — a genuinely weighed alternative, listed with its score.
    assert alternatives
    assert any("CO-01" in a and "cosine" in a for a in alternatives)
    assert "no_objective_alignment" not in (
        out[0].content.get("structural_warnings") or []
    )


def test_unaligned_block_stamps_warning_no_fabrication():
    """A block matching nothing → empty refs + the warning (gate's job)."""
    objectives = [
        {"id": "CO-01", "statement": "Photosynthesis biology chloroplast sunlight glucose."},
    ]
    block = _block(
        "b2",
        _content(
            ["Completely unrelated administrative scheduling logistics paperwork."],
            objective_refs=[],
        ),
    )
    out = align_blocks_to_objectives(
        [block], objectives, embed=FakeEmbed(), threshold=0.9,
    )
    refs = out[0].content["objective_refs"]
    assert refs == []  # NEVER fabricated to dodge the gate
    assert _NO_ALIGNMENT_MARKER in out[0].content["structural_warnings"]


def test_existing_ref_is_authoritative_and_preserved():
    """The model's explicit ref is kept even if cosine is below threshold."""
    objectives = [
        {"id": "CO-01", "statement": "Some objective statement."},
    ]
    block = _block(
        "b3",
        _content(["Disjoint text zzz qqq."], objective_refs=["CO-01"]),
    )
    out = align_blocks_to_objectives(
        [block], objectives, embed=FakeEmbed(), threshold=0.99,
    )
    assert "CO-01" in out[0].content["objective_refs"]


def test_embeddings_absent_passthrough():
    """embed=None → pass-through; outline-tier refs unchanged."""
    objectives = [{"id": "CO-01", "statement": "x"}]
    block = _block("b4", _content(["claim"], objective_refs=["CO-99"]))
    out = align_blocks_to_objectives([block], objectives, embed=None)
    # Unchanged — same object, refs untouched.
    assert out[0].content["objective_refs"] == ["CO-99"]
