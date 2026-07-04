"""Callout grouping + pedagogical-marker label-residue scrub at the adapter
seam (end-user-HTML audit, ch02 shots). Synthetic block IR only.

  * A promoted opener heading (How To / Example / Solution / Try It / …) and the
    content blocks that FOLLOW it are wrapped in ONE ``data-dart-opener-group``
    box container, so ``dart_content.css`` encloses the whole unit (the "How To
    box wraps only the label, steps spill outside" defect).
  * ":: " / ": : " colon-run + stray ">"/"|" gutter residue is folded out of
    block text before the opener split.
"""
from __future__ import annotations

import re

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _render_chapters,
    _scrub_marker_artifacts,
    normalize_cascade_to_ed4all,
)


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
        html="",
        region_kind="heading",
        raw_block_index=idx,
        heading_level=4,
        raw_text=text,
        heading_text=text,
        block_role=role,
    )


def _para(text, idx):
    return _AdapterBlock(
        html=f"<p>{text}</p>",
        region_kind="paragraph",
        raw_block_index=idx,
        raw_text=text,
        heading_text=None,
    )


def _section(text, idx):
    return _AdapterBlock(
        html="",
        region_kind="heading",
        raw_block_index=idx,
        heading_level=3,
        raw_text=text,
        heading_text=text,
    )


# ---------------------------------------------------------------------------
# Callout grouping (_render_chapters).
# ---------------------------------------------------------------------------
def test_opener_and_following_content_share_one_box():
    ch = _AdapterChapter(
        title="Solving",
        blocks=[
            _opener("How To", 0, "how_to"),
            _para("Step 1. Simplify each side.", 1),
            _para("Step 2. Collect the variable terms.", 2),
        ],
    )
    html = _render_chapters([ch])
    m = re.search(
        r'<div class="dart-callout-group" data-dart-opener-group="how_to">'
        r"(.*?)</div>",
        html,
        re.DOTALL,
    )
    assert m is not None, "no how_to group container emitted"
    box = m.group(1)
    assert "How To" in box
    assert "Step 1. Simplify each side." in box  # content INSIDE the box
    assert "Step 2. Collect the variable terms." in box


def test_genuine_heading_closes_the_group():
    ch = _AdapterChapter(
        title="Solving",
        blocks=[
            _opener("How To", 0, "how_to"),
            _para("Step 1.", 1),
            _section("Next Section", 2),
            _para("Body after section.", 3),
        ],
    )
    html = _render_chapters([ch])
    m = re.search(
        r'data-dart-opener-group="how_to">(.*?)</div>', html, re.DOTALL
    )
    box = m.group(1)
    assert "Step 1." in box
    # A genuine <h3> boundary is OUTSIDE the box, and its following body too.
    assert "Next Section" not in box
    assert "Body after section." not in box
    assert "<h3" in html and "Next Section" in html


def test_new_opener_starts_a_fresh_group():
    ch = _AdapterChapter(
        title="Ex",
        blocks=[
            _opener("Example 2.1", 0, "worked_example"),
            _para("worked content", 1),
            _opener("Solution", 2, "solution"),
            _para("solution content", 3),
        ],
    )
    html = _render_chapters([ch])
    assert html.count("data-dart-opener-group=") == 2
    ex = re.search(
        r'data-dart-opener-group="worked_example">(.*?)</div>', html, re.DOTALL
    ).group(1)
    sol = re.search(
        r'data-dart-opener-group="solution">(.*?)</div>', html, re.DOTALL
    ).group(1)
    assert "worked content" in ex and "solution content" not in ex
    assert "solution content" in sol and "worked content" not in sol


def test_plain_prose_without_opener_is_not_wrapped():
    ch = _AdapterChapter(title="Prose", blocks=[_para("just prose", 0)])
    html = _render_chapters([ch])
    assert "dart-callout-group" not in html
    assert "just prose" in html


# ---------------------------------------------------------------------------
# Marker-residue scrub (_scrub_marker_artifacts).
# ---------------------------------------------------------------------------
def test_double_colon_marker_folds():
    assert (
        _scrub_marker_artifacts("TRY IT :: 2.1 Solve x.", html=False)
        == "TRY IT 2.1 Solve x."
    )


def test_spaced_colons_and_trailing_gutter_fold():
    assert (
        _scrub_marker_artifacts("TRY IT : : 9.129 >", html=False)
        == "TRY IT 9.129"
    )


def test_leading_marker_residue_stripped():
    assert (
        _scrub_marker_artifacts(":: GENERAL STRATEGY.", html=False)
        == "GENERAL STRATEGY."
    )


def test_single_colon_label_preserved():
    # A lone label colon is NOT a marker artifact — leave it.
    txt = "Solving Applications with Formulas: read carefully."
    assert _scrub_marker_artifacts(txt, html=False) == txt


def test_math_run_colons_preserved():
    txt = r"ratio $a::b$ and \(x::y\) stay"
    assert _scrub_marker_artifacts(txt, html=False) == txt


def test_html_tag_gt_not_touched():
    html = "<p>a :: b</p>"
    got = _scrub_marker_artifacts(html, html=True)
    assert got == "<p>a b</p>"  # colon-run folded, the <p> tag intact


def test_scrub_promotes_de_doubled_try_it_end_to_end():
    # After the scrub the numbered marker reads "TRY IT 2.1", which the opener
    # promotion turns into a boxed try_it group in the rendered document.
    ch = _AdapterChapter(
        title="Solving",
        blocks=[_para("TRY IT :: 2.1 Solve: x + 1 = 2. and more.", 0)],
    )
    html = normalize_cascade_to_ed4all(
        _Result([ch]), pdf_stem="synthetic_ch02"
    )["html"]
    assert "::" not in html
    assert 'data-dart-opener-group="try_it"' in html
