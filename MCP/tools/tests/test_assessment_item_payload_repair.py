"""Unit tests for the assessment_item parsed-but-structurally-empty repair
(rewrite-overflow-fix-2026-06, component 7).

A real 7B ``assessment_item`` that authored non-empty HTML but omitted the
canonical ``<li data-cf-distractor-index="N">`` option markup would FAIL
the CRITICAL ``rewrite_assessment_item_payload`` gate and BLOCK the run.
``_assessment_item_payload_structurally_empty`` detects that case so the
emit loop routes it to the deterministic MCQ-completion fallback (which
DOES emit the canonical markup) BEFORE the gate sees it.
"""
from __future__ import annotations

from MCP.tools.pipeline_tools import (
    _assessment_item_payload_structurally_empty,
    _render_block_fallback_html,
)
from lib.validators.assessment_item_payload import (
    BlockAssessmentItemPayloadValidator,
)

try:  # blocks.py lives under Courseforge/scripts
    from Courseforge.scripts.blocks import Block
except ImportError:  # pragma: no cover
    from blocks import Block  # type: ignore


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
def test_detector_flags_missing_option_markup():
    # Non-empty HTML, but NO <li data-cf-distractor-index> siblings → would
    # fail the gate → flagged.
    html = (
        "<section><p>Which is correct?</p>"
        "<ul><li>2/3</li><li>4/6</li></ul>"
        "<p>Correct answer: 2/3</p></section>"
    )
    assert _assessment_item_payload_structurally_empty(html) is True


def test_detector_passes_valid_mcq_markup():
    html = (
        "<section><p>Q?</p><ol>"
        '<li data-cf-distractor-index="0" data-cf-correct="true">a</li>'
        '<li data-cf-distractor-index="1">b</li>'
        "</ol></section>"
    )
    assert _assessment_item_payload_structurally_empty(html) is False


def test_detector_passes_on_empty_content():
    # Empty content is handled by the escalation path, not this branch.
    assert _assessment_item_payload_structurally_empty("") is False
    assert _assessment_item_payload_structurally_empty("   ") is False


def test_detector_flags_single_option():
    # Exactly one option < 2 → flagged (the gate floor is >= 2).
    html = '<ol><li data-cf-distractor-index="0">a</li></ol>'
    assert _assessment_item_payload_structurally_empty(html) is True


# ---------------------------------------------------------------------------
# The fallback the repair routes to produces GATE-VALID MCQ markup.
# ---------------------------------------------------------------------------
def test_fallback_produces_gate_valid_assessment_item():
    block = Block(
        block_id="page#assessment_item_x_0",
        block_type="assessment_item",
        page_id="page",
        sequence=0,
        content={
            "key_claims": [
                "What is 20/30 in simplest form?",
                "The answer is 2/3.",
            ],
        },
        objective_ids=("CO-01",),
        source_ids=("semantik:slug#blk1",),
    )
    fallback_html = _render_block_fallback_html(
        block,
        objective_statements={"CO-01": "Simplify fractions."},
        source_ids=["semantik:slug#blk1"],
    )
    # The fallback HTML must NOT itself look structurally-empty (it carries
    # the canonical <li data-cf-distractor-index> siblings).
    assert _assessment_item_payload_structurally_empty(fallback_html) is False

    # And it passes the CRITICAL gate as a rewrite-tier (str-content) block.
    repaired = Block(
        block_id=block.block_id,
        block_type="assessment_item",
        page_id="page",
        sequence=0,
        content=fallback_html,
        objective_ids=("CO-01",),
        source_ids=("semantik:slug#blk1",),
    )
    validator = BlockAssessmentItemPayloadValidator()
    result = validator.validate({"blocks": [repaired]})
    assert result.passed is True, [i.code for i in result.issues]
