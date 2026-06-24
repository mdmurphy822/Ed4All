"""FR-INT-04 — B10 three-move discussion protocol.

CPU-pinned, no-GPU, no-model unit tests for the B10ProtocolValidator + the
discussion Block protocol fields + the three-move render scaffold.

PRIMARY gate: byte-stability with ED4ALL_NEW_BLOCK_TYPES unset — the validator
no-ops (passed=True + info issue) and the render scaffold returns "".
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from Courseforge.scripts.blocks import Block  # noqa: E402
from lib.validators.b10_protocol import B10ProtocolValidator  # noqa: E402


def _disc(content="Debate the topic.", **kw) -> Block:
    return Block(
        block_id="w1_disc#discussion_prompt_x_0",
        block_type="discussion_prompt",
        page_id="w1_disc",
        sequence=0,
        content=content,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Block fields
# --------------------------------------------------------------------------- #
def test_discussion_protocol_fields_hash_excluded():
    """discussion_protocol / discussion_bloom_verb must not drift the hash."""
    base = _disc()
    enriched = _disc(
        discussion_protocol={"post": "p", "respond": "r", "synthesize": "s"},
        discussion_bloom_verb="create",
    )
    assert base.compute_content_hash() == enriched.compute_content_hash()


# --------------------------------------------------------------------------- #
# Validator — flag off (byte-stable no-op)
# --------------------------------------------------------------------------- #
def test_b10_noop_when_flag_off():
    r = B10ProtocolValidator().validate(
        {"blocks": [_disc()], "new_block_types_enabled": False}
    )
    assert r.passed is True
    assert r.issues[0].code == "B10_PROTOCOL_DISABLED"
    assert r.metadata["b10_protocol"]["enabled"] is False


# --------------------------------------------------------------------------- #
# Validator — flag on, collapsed discussion fires
# --------------------------------------------------------------------------- #
def test_b10_fires_on_collapsed_single_prompt():
    r = B10ProtocolValidator().validate(
        {"blocks": [_disc()], "new_block_types_enabled": True}
    )
    assert r.passed is True  # warning-day-1
    codes = [i.code for i in r.issues]
    assert "DISCUSSION_PROTOCOL_COLLAPSED" in codes
    assert r.action == "regenerate"
    assert r.issues[0].severity == "warning"


def test_b10_fires_when_missing_synthesize_move():
    # has post + respond, but no synthesize -> collapsed.
    blk = _disc(
        content="Post your view. Respond to two classmates with your reasoning."
    )
    r = B10ProtocolValidator().validate(
        {"blocks": [blk], "new_block_types_enabled": True}
    )
    assert "DISCUSSION_PROTOCOL_COLLAPSED" in [i.code for i in r.issues]


# --------------------------------------------------------------------------- #
# Validator — flag on, full three-move protocol passes clean
# --------------------------------------------------------------------------- #
def test_b10_clean_with_structured_protocol():
    blk = _disc(
        discussion_protocol={
            "post": "Post your initial position.",
            "respond": "Reply to two peers.",
            "synthesize": "Synthesize the thread into a revised position.",
        }
    )
    r = B10ProtocolValidator().validate(
        {"blocks": [blk], "new_block_types_enabled": True}
    )
    assert r.issues == []
    assert r.action is None


def test_b10_clean_from_html_three_moves():
    html = (
        "<p>Post your position.</p>"
        "<p>Respond to at least two classmates.</p>"
        "<p>Synthesize the discussion thread into a consolidated view.</p>"
    )
    r = B10ProtocolValidator().validate(
        {"blocks": [_disc(content=html)], "new_block_types_enabled": True}
    )
    assert r.issues == []


def test_b10_non_discussion_blocks_ignored():
    other = Block(
        block_id="w1#concept_a_0",
        block_type="concept",
        page_id="w1",
        sequence=0,
        content="A concept.",
    )
    r = B10ProtocolValidator().validate(
        {"blocks": [other], "new_block_types_enabled": True}
    )
    assert r.metadata["b10_protocol"]["discussion_blocks"] == 0
    assert r.issues == []


# --------------------------------------------------------------------------- #
# Render scaffold — byte-stable off / emits three moves on
# --------------------------------------------------------------------------- #
def test_discussion_protocol_render_byte_stable_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_NEW_BLOCK_TYPES", raising=False)
    import generate_course as gc

    out = gc._render_discussion_protocol(
        {"prompt": "q", "discussion_protocol": {"post": "p", "respond": "r",
                                                "synthesize": "s"}}
    )
    assert out == ""


def test_discussion_protocol_render_emits_three_moves_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_NEW_BLOCK_TYPES", "1")
    import generate_course as gc

    out = gc._render_discussion_protocol(
        {"prompt": "q", "discussion_protocol": {"post": "PostMove",
                                                "respond": "RespondMove",
                                                "synthesize": "SynthMove"}}
    )
    assert 'data-move="post"' in out
    assert 'data-move="respond"' in out
    assert 'data-move="synthesize"' in out
    assert "PostMove" in out and "RespondMove" in out and "SynthMove" in out


# --------------------------------------------------------------------------- #
# Decision capture fires
# --------------------------------------------------------------------------- #
def test_b10_emits_decision_capture():
    events = []

    class _Cap:
        def log_decision(self, **kw):
            events.append(kw)

    B10ProtocolValidator().validate(
        {"blocks": [_disc()], "new_block_types_enabled": True,
         "decision_capture": _Cap()}
    )
    assert len(events) == 1
    assert events[0]["decision_type"] == "content_structure_check"
    assert len(events[0]["rationale"]) >= 20
