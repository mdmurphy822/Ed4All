"""W6.4 _render_faq_section + W6.6 confidence-capture assessment_item extension.

Pure-deterministic HTML-render assertions — no model, no GPU.
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
from lib.generation.reflection_calibration import (  # noqa: E402
    ENV_REFLECTION_CALIBRATION,
)

_ENTRIES = [
    {
        "question": "Is it true that you can divide by zero?",
        "answer": "Dividing by zero is undefined.",
        "source_link": "/api/learn/source/course-alpha?item_path=ch1/div.html",
    },
    {"question": "What is a fraction?", "answer": "A fraction is a part of a whole."},
]


# --------------------------------------------------------------------------- #
# W6.4 — _render_faq_section
# --------------------------------------------------------------------------- #

def test_faq_section_empty():
    import generate_course as gc

    assert gc._render_faq_section([]) == ""


def test_faq_section_renders_cards():
    import generate_course as gc

    out = gc._render_faq_section(_ENTRIES, page_id="week_01_faq")
    assert '<section class="faq" data-cf-content-type="faq">' in out
    assert out.count('class="callout faq-card"') == 2
    assert "Is it true that you can divide by zero?" in out
    assert "A fraction is a part of a whole." in out
    # first card has a source link, second does not.
    assert out.count("View source") == 1


def test_faq_section_accepts_camel_source_link_key():
    import generate_course as gc

    out = gc._render_faq_section(
        [{"question": "Q?", "answer": "A.", "sourceLink": "/api/learn/source/x"}]
    )
    assert "View source" in out


def test_faq_section_carries_block_id(monkeypatch):
    # data-cf-block-id is gated behind COURSEFORGE_EMIT_BLOCKS (byte-stable
    # posture); the block-attr splice mirrors _render_key_terms_section.
    monkeypatch.setenv("COURSEFORGE_EMIT_BLOCKS", "1")
    import generate_course as gc

    out = gc._render_faq_section(_ENTRIES, page_id="week_01_faq")
    assert "data-cf-block-id" in out


def test_faq_section_block_attrs_noop_when_emit_off(monkeypatch):
    # Byte-stable posture: no block-id attr when COURSEFORGE_EMIT_BLOCKS is off.
    monkeypatch.delenv("COURSEFORGE_EMIT_BLOCKS", raising=False)
    import generate_course as gc

    out = gc._render_faq_section(_ENTRIES, page_id="week_01_faq")
    assert "data-cf-block-id" not in out


def test_faq_section_escapes_html():
    import generate_course as gc

    out = gc._render_faq_section([{"question": "<x> & y?", "answer": "z<b>"}])
    assert "&lt;x&gt;" in out
    assert "&amp; y" in out


# --------------------------------------------------------------------------- #
# W6.6 — confidence capture: assessment_item calibration-comparison note
# --------------------------------------------------------------------------- #

def _ai(**kw) -> Block:
    return Block(
        block_id="w1_a#assessment_item_q_0",
        block_type="assessment_item",
        page_id="w1_a",
        sequence=0,
        content="Which option is correct?",
        **kw,
    )


def _sc(**kw) -> Block:
    return Block(
        block_id="w1_s#self_check_question_q_0",
        block_type="self_check_question",
        page_id="w1_s",
        sequence=0,
        content="Which option is correct?",
        **kw,
    )


def test_confidence_off_byte_stable_assessment(monkeypatch):
    monkeypatch.delenv(ENV_REFLECTION_CALIBRATION, raising=False)
    import generate_course as gc

    assert gc._render_confidence_capture(_ai(confidence_prompt="Sure?")) == ""


def test_assessment_item_gets_calibration_note_on(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    out = gc._render_confidence_capture(_ai(confidence_prompt="How sure?"))
    assert "confidence-capture" in out
    assert 'class="calibration-comparison"' in out
    assert "Compare your confidence with your result" in out


def test_self_check_has_no_calibration_note(monkeypatch):
    # W6.6 must NOT regress B07 self-check: no calibration-comparison note.
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    out = gc._render_confidence_capture(_sc(confidence_prompt="How sure?"))
    assert "confidence-capture" in out
    assert "calibration-comparison" not in out


def test_assessment_item_no_note_without_prompt_even_on(monkeypatch):
    monkeypatch.setenv(ENV_REFLECTION_CALIBRATION, "1")
    import generate_course as gc

    assert gc._render_confidence_capture(_ai()) == ""
