"""Wave #22 Tier-1 — reading-order flow annotation + unit-skeleton heading
re-derivation (synthetic block IR only).
"""
from __future__ import annotations

import re

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _annotate_reading_order_flow,
    _rederive_unit_headings,
    normalize_cascade_to_ed4all,
)
from lib.semantik.heading_classifier import is_emphasis_label_heading


class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = 0.9
        self.flags = []
        self.lane_used = "fast"
        self.lang = "en"


def _opener(text, idx, role):
    return _AdapterBlock(
        html="", region_kind="heading", raw_block_index=idx, heading_level=4,
        raw_text=text, heading_text=text, block_role=role,
    )


def _para(text, idx):
    return _AdapterBlock(
        html=f"<p>{text}</p>", region_kind="paragraph", raw_block_index=idx,
        raw_text=text, heading_text=None,
    )


def _heading(text, idx):
    return _AdapterBlock(
        html="", region_kind="heading", raw_block_index=idx, heading_level=3,
        raw_text=text, heading_text=text,
    )


# ---------------------------------------------------------------------------
# Flow annotation.
# ---------------------------------------------------------------------------


def test_flow_statement_and_solution_steps():
    blocks = [
        _opener("Example 9.1", 0, "worked_example"),
        _para("Simplify the radical.", 1),
        _opener("Solution", 2, "solution"),
        _para("Step one.", 3),
        _para("Step two.", 4),
    ]
    ch = _AdapterChapter(title="Roots", blocks=blocks)
    _annotate_reading_order_flow([ch])
    assert blocks[1].flow == "statement"
    assert blocks[3].flow == "solution-steps"
    assert blocks[4].flow == "solution-steps"
    # the opener headings themselves are never flow-annotated
    assert blocks[0].flow is None and blocks[2].flow is None


def test_flow_lone_example_has_no_statement():
    # No following solution -> no "between", so the body gets no statement flow.
    blocks = [_opener("Example 1.1", 0, "worked_example"), _para("body", 1)]
    ch = _AdapterChapter(title="X", blocks=blocks)
    _annotate_reading_order_flow([ch])
    assert blocks[1].flow is None


def test_flow_procedure_steps():
    blocks = [
        _opener("How To", 0, "how_to"),
        _para("First do this.", 1),
        _para("Then do that.", 2),
    ]
    ch = _AdapterChapter(title="P", blocks=blocks)
    _annotate_reading_order_flow([ch])
    assert blocks[1].flow == "procedure-steps"
    assert blocks[2].flow == "procedure-steps"


def test_flow_does_not_cross_section_heading():
    blocks = [
        _opener("Example 1.1", 0, "worked_example"),
        _para("stmt", 1),
        _heading("9.2 New Section", 2),
        _opener("Solution", 3, "solution"),
        _para("after", 4),
    ]
    ch = _AdapterChapter(title="X", blocks=blocks)
    _annotate_reading_order_flow([ch])
    # The heading splits the segment: the example has no solution in its segment.
    assert blocks[1].flow is None
    # The solution's own segment still marks its steps.
    assert blocks[4].flow == "solution-steps"


def test_flow_emitted_as_attribute():
    ch = _AdapterChapter(
        title="Roots",
        blocks=[
            _opener("Example 9.1", 0, "worked_example"),
            _para("Simplify.", 1),
            _opener("Solution", 2, "solution"),
            _para("Steps.", 3),
        ],
    )
    html = normalize_cascade_to_ed4all(_Result([ch]), pdf_stem="s_ch09")["html"]
    assert 'data-dart-flow="statement"' in html
    assert 'data-dart-flow="solution-steps"' in html


# ---------------------------------------------------------------------------
# Heading re-derivation.
# ---------------------------------------------------------------------------


def test_emphasis_label_predicate():
    assert is_emphasis_label_heading(r"\textbf{Square of a Number}")
    assert is_emphasis_label_heading(r"\textit{Square Root Notation}")
    # A real heading is never wholly emphasis-wrapped.
    assert not is_emphasis_label_heading("Square of a Number")
    assert not is_emphasis_label_heading(r"9.1 Simplify \textbf{Square} Roots")
    assert not is_emphasis_label_heading("")


def test_rederive_demotes_emphasis_label_heading():
    blocks = [_heading(r"\textbf{Square of a Number}", 0), _para("def prose", 1)]
    ch = _AdapterChapter(title="Roots", blocks=blocks)
    n = _rederive_unit_headings([ch])
    assert n == 1
    assert blocks[0].heading_text is None  # demoted out of the heading stream
    assert blocks[0].region_kind == "paragraph"
    # The bold semantics survive as <strong> in the demoted body.
    assert "<strong>Square of a Number</strong>" in blocks[0].html


def test_rederive_demotes_severing_heading():
    # Example ... <stray heading> ... Solution: the heading severs a worked
    # example and is demoted so the unit can re-form.
    blocks = [
        _opener("Example 9.1", 0, "worked_example"),
        _para("stmt", 1),
        _heading("A Stray Label", 2),
        _opener("Solution", 3, "solution"),
        _para("steps", 4),
    ]
    ch = _AdapterChapter(title="X", blocks=blocks)
    n = _rederive_unit_headings([ch])
    assert n == 1
    assert blocks[2].heading_text is None


def test_rederive_keeps_numbered_section_title():
    # A numbered section title between example and solution is a REAL section.
    blocks = [
        _opener("Example 9.1", 0, "worked_example"),
        _para("stmt", 1),
        _heading("9.2 Add Square Roots", 2),
        _opener("Solution", 3, "solution"),
        _para("steps", 4),
    ]
    ch = _AdapterChapter(title="X", blocks=blocks)
    n = _rederive_unit_headings([ch])
    assert n == 0
    assert blocks[2].heading_text == "9.2 Add Square Roots"


def test_rederive_keeps_normal_section_heading():
    # A plain section heading NOT sandwiched inside a worked example is kept.
    blocks = [_heading("Simplify Square Roots", 0), _para("prose", 1)]
    ch = _AdapterChapter(title="X", blocks=blocks)
    assert _rederive_unit_headings([ch]) == 0
    assert blocks[0].heading_text == "Simplify Square Roots"


def test_releveling_count_in_result():
    ch = _AdapterChapter(
        title="Roots",
        blocks=[_heading(r"\textbf{Square of a Number}", 0), _para("prose", 1)],
    )
    res = normalize_cascade_to_ed4all(_Result([ch]), pdf_stem="s_ch09")
    assert res["heading_releveling_count"] == 1
    assert "<strong>Square of a Number</strong>" in res["html"]
