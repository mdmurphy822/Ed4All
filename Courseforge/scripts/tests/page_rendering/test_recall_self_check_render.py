"""recall_self_check — structured render branch + Block hash-exclusion tests."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from blocks import Block  # noqa: E402
from generate_course import _render_self_check  # noqa: E402


_Q = {
    "question": "The property letting a(b+c)=ab+ac is the ___ property.",
    "options": [
        {"text": "distributive", "correct": True, "feedback": "Right."},
        {"text": "commutative", "correct": False, "feedback": "No."},
    ],
    "bloom_level": "remember",
}


def test_off_byte_identical_to_legacy(monkeypatch):
    monkeypatch.delenv("ED4ALL_RECALL_SELF_CHECK", raising=False)
    # A question carrying a recall_format is IGNORED when the flag is off ->
    # legacy radio markup, byte-identical.
    q_marked = dict(_Q, recall_format="cloze")
    baseline = _render_self_check([dict(_Q)])
    with_marker_off = _render_self_check([q_marked])
    assert with_marker_off == baseline
    assert '<input type="radio"' in baseline


def test_on_free_recall_renders_text_input_and_reveal(monkeypatch):
    monkeypatch.setenv("ED4ALL_RECALL_SELF_CHECK", "1")
    q = dict(_Q, recall_format="free_recall")
    html = _render_self_check([q])
    # No radio enumeration; a text input + a <details> reveal of the correct
    # option text (no fabricated answer; answer not in the visible prompt body).
    assert '<input type="radio"' not in html
    assert '<input type="text"' in html
    assert "<details" in html
    assert "distributive" in html  # reused correct-option text in the reveal


def test_on_cloze_renders_blank(monkeypatch):
    monkeypatch.setenv("ED4ALL_RECALL_SELF_CHECK", "1")
    q = dict(_Q, recall_format="cloze")
    html = _render_self_check([q])
    assert '<input type="radio"' not in html
    assert "____" in html
    assert "<details" in html


def test_block_recall_format_excluded_from_content_hash():
    base = Block(
        block_id="p#sc_0",
        block_type="self_check_question",
        page_id="p",
        sequence=0,
        content={"question": "Q", "options": []},
    )
    marked = Block(
        block_id="p#sc_0",
        block_type="self_check_question",
        page_id="p",
        sequence=0,
        content={"question": "Q", "options": []},
        recall_format="cloze",
    )
    assert base.compute_content_hash() == marked.compute_content_hash()
