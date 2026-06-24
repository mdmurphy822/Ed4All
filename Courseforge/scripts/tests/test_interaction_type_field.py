"""FR-PLAN-01 — Block.interaction_type field: hash-exclusion + byte-stability.

``interaction_type`` (and the reused IB5 ``fade_state``) must be EXCLUDED from
``compute_content_hash`` so an interaction-type retro-fit never drifts an
existing block hash, and the default-None state must keep legacy blocks
byte-identical.
"""
from __future__ import annotations

from Courseforge.scripts.blocks import Block


def _block(**kw):
    base = dict(
        block_id="p#concept_x_0", block_type="concept", page_id="p",
        sequence=0, content="body",
    )
    base.update(kw)
    return Block(**base)


def test_interaction_type_defaults_none():
    b = _block()
    assert b.interaction_type is None
    assert b.fade_state is None


def test_interaction_type_excluded_from_hash():
    h = _block().compute_content_hash()
    assert _block(interaction_type="numeric").compute_content_hash() == h
    assert _block(interaction_type="drag_drop").compute_content_hash() == h


def test_fade_state_excluded_from_hash():
    h = _block().compute_content_hash()
    assert _block(fade_state="completion").compute_content_hash() == h


def test_interaction_type_accepts_arbitrary_token():
    # the field is permissive (no enum on the dataclass; the planner is the
    # source of valid tokens) — mirrors the IB5 fields' posture.
    b = _block(block_type="assessment_item", interaction_type="multiple_choice")
    assert b.interaction_type == "multiple_choice"
