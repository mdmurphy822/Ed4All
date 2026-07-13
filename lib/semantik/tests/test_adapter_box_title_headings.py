"""SEMANTIK_BOX_TITLE_HEADINGS — assembly-time presentational callout box titles
on the page-arranger scan lane.

The arranger DEMOTES a worked example's / solution's / definition box's leading
label out of heading typing (anti-poisoning), typing it as a ``paragraph``
Region with a ``pedagogy-example`` / ``pedagogy-solution`` / ``definition-box``
css class exported to the provenance as ``pedagogy_class``. This opt-in re-emits
those labels at assembly as ``data-semantik-box-title`` <h4>/<h5> headings that
structure derivation (chapter/section extraction + the chunker boundaries)
UNCONDITIONALLY skips.

All IR built inline — no course-data path, no model, no cascade run.
"""
from __future__ import annotations

import re

import pytest

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)


class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = 0.9
        self.flags = []
        self.lane_used = "arranger"
        self.lang = "en"


def _section(text, idx):
    return _AdapterBlock(
        html="",
        region_kind="heading",
        raw_block_index=idx,
        heading_level=3,
        raw_text=text,
        heading_text=text,
        pages=[idx + 1],
    )


def _callout(text, idx, pedagogy_class):
    return _AdapterBlock(
        html=f"<p>{text}</p>",
        region_kind="paragraph",
        raw_block_index=idx,
        raw_text=text,
        heading_text=None,
        pages=[idx + 1],
        pedagogy_class=pedagogy_class,
    )


def _chapter():
    return _AdapterChapter(
        title="Whole Numbers",
        blocks=[
            _section("1.1 Introduction to Whole Numbers", 0),
            _callout(
                "EXAMPLE 1.108 Divide: 27 by 3.", 1, "pedagogy-example"
            ),
            _callout(
                "Solution The quotient is 9 because 9 times 3 is 27.",
                2,
                "pedagogy-solution",
            ),
            _callout(
                "Variable\nA variable is a letter that represents a number.",
                3,
                "definition-box",
            ),
        ],
    )


def _render(chapters):
    return normalize_cascade_to_ed4all(_Result(chapters), pdf_stem="ch01")["html"]


# ---------------------------------------------------------------------------
# (a) flag OFF → byte-identical assembly of a callout-bearing fixture.
# ---------------------------------------------------------------------------
def test_flag_off_no_box_title_and_field_is_inert(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BOX_TITLE_HEADINGS", raising=False)
    html_off = _render([_chapter()])
    assert "data-semantik-box-title" not in html_off

    # The pedagogy_class field is INERT when the flag is off: rendering the same
    # chapter with the callout css class STRIPPED is byte-identical.
    stripped = _AdapterChapter(
        title="Whole Numbers",
        blocks=[
            _section("1.1 Introduction to Whole Numbers", 0),
            _callout("EXAMPLE 1.108 Divide: 27 by 3.", 1, None),
            _callout(
                "Solution The quotient is 9 because 9 times 3 is 27.", 2, None
            ),
            _callout(
                "Variable\nA variable is a letter that represents a number.",
                3,
                None,
            ),
        ],
    )
    assert _render([stripped]) == html_off


# ---------------------------------------------------------------------------
# (b) flag ON → h4/h5 emitted with data-semantik-box-title + aria-labelledby +
#     label carved from body.
# ---------------------------------------------------------------------------
def test_flag_on_emits_box_titles(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BOX_TITLE_HEADINGS", "1")
    html = _render([_chapter()])

    # Example / definition openers -> <h4 data-semantik-box-title>.
    assert re.search(
        r'<h4 data-semantik-box-title="1" id="[^"]+">Example 1.108</h4>', html
    ), html
    assert re.search(
        r'<h4 data-semantik-box-title="1" id="[^"]+">Variable</h4>', html
    ), html
    # Solution -> <h5> (vendor parity, one deeper).
    m = re.search(
        r'<h5 data-semantik-box-title="1" id="([^"]+)">Solution</h5>', html
    )
    assert m is not None, html
    solution_hid = m.group(1)

    # The enclosing callout <section> is named via aria-labelledby -> box-title id.
    assert f'aria-labelledby="{solution_hid}"' in html

    # The carved label is REMOVED from the body (no duplication): the solution
    # body prose survives, the leading "Solution" label does not lead the <p>.
    assert "<p>The quotient is 9 because 9 times 3 is 27.</p>" in html
    assert "<p>Solution" not in html
    # The example label is out of its <p> body too.
    assert "<p>Divide: 27 by 3.</p>" in html
    assert "<p>EXAMPLE 1.108" not in html


# ---------------------------------------------------------------------------
# (c) no-marker block → no heading invented.
# ---------------------------------------------------------------------------
def test_flag_on_no_marker_invents_nothing(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BOX_TITLE_HEADINGS", "1")
    ch = _AdapterChapter(
        title="Whole Numbers",
        blocks=[
            _section("1.1 Introduction to Whole Numbers", 0),
            _callout(
                "This paragraph carries no pedagogical marker and merely "
                "describes a general property of numbers.",
                1,
                "pedagogy-example",
            ),
            _callout(
                "another lowercase definition body with no leading term line",
                2,
                "definition-box",
            ),
        ],
    )
    html = _render([ch])
    assert "data-semantik-box-title" not in html
    # The body prose is untouched (no carve).
    assert "This paragraph carries no pedagogical marker" in html
