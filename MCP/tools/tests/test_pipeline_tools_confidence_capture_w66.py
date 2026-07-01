"""W6.6 — confidence-on-graded-checks emit wiring (ED4ALL_REFLECTION_CALIBRATION).

Regression for the assessment_item render wiring in
``MCP/tools/pipeline_tools.py``: a content-tier ``assessment_item`` (B14) block
must, when ``ED4ALL_REFLECTION_CALIBRATION`` is on, get a default
``confidence_prompt`` stamped and render the confidence-capture control PLUS the
graded calibration-comparison reveal (delegating to
``generate_course._render_confidence_capture``). Default (flag off) =>
byte-identical (no capture).

Pure-deterministic — no model, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.tools import pipeline_tools as _pt  # noqa: E402
from lib.generation.reflection_calibration import (  # noqa: E402
    ENV_REFLECTION_CALIBRATION,
)


def _ai() -> Block:
    """A content-tier assessment_item Block with outline key_claims so the
    deterministic fallback renderer can author a gate-valid MCQ."""
    return Block(
        block_id="week_01_self_check#assessment_item_co-01_0",
        block_type="assessment_item",
        page_id="week_01_self_check",
        sequence=0,
        content={"key_claims": [
            {"claim": "Which statement about concept A is correct?"},
            {"claim": "Concept A is foundational."},
            {"claim": "Concept A is optional."},
        ]},
        objective_ids=("CO-01",),
        target_bloom="apply",
    )


def _concept() -> Block:
    return Block(
        block_id="week_01_content_01#concept_a_0",
        block_type="concept",
        page_id="week_01_content_01",
        sequence=0,
        content={"key_claims": [{"claim": "Concept A is foundational."}]},
    )


# --------------------------------------------------------------------------- #
# _assessment_confidence_capture_html — the shared stamp+render helper.
# --------------------------------------------------------------------------- #

def test_helper_off_returns_empty(monkeypatch):
    monkeypatch.delenv(ENV_REFLECTION_CALIBRATION, raising=False)
    assert _pt._assessment_confidence_capture_html(_ai()) == ""


def test_helper_on_stamps_and_renders(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    out = _pt._assessment_confidence_capture_html(_ai())
    assert "confidence-capture" in out
    assert 'class="calibration-comparison"' in out
    assert "Compare your confidence with your result" in out


def test_helper_noop_for_non_assessment_block(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    assert _pt._assessment_confidence_capture_html(_concept()) == ""


# --------------------------------------------------------------------------- #
# _render_block_fallback_html — the assessment_item render branch.
# --------------------------------------------------------------------------- #

def test_fallback_render_on_includes_capture(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    out = _pt._render_block_fallback_html(_ai())
    # MCQ payload still present (gate-valid) AND the confidence capture rides on.
    assert 'data-cf-distractor-index="0"' in out
    assert "confidence-capture" in out
    assert 'class="calibration-comparison"' in out


def test_fallback_render_off_byte_identical(monkeypatch):
    # Flag OFF must be byte-identical to a render with no capture wiring at all.
    monkeypatch.delenv(ENV_REFLECTION_CALIBRATION, raising=False)
    out = _pt._render_block_fallback_html(_ai())
    assert "confidence-capture" not in out
    assert "calibration-comparison" not in out
    # And the MCQ payload is unaffected (byte-stable off).
    assert 'data-cf-distractor-index="0"' in out
