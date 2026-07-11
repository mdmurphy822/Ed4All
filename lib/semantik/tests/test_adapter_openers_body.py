"""A7 opener promotion + B3 body-LaTeX sanitation + B4 label removal, at the
adapter seam. Synthetic block IR only (no corpus files).
"""
from __future__ import annotations

import re

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)
from lib.semantik.math_fold import sanitize_body_latex


class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = 0.9
        self.flags = []
        self.lane_used = "fast"
        self.lang = "en"


def _para(text, idx, *, kind="paragraph"):
    return _AdapterBlock(
        html=f"<p>{text}</p>",
        region_kind=kind,
        raw_block_index=idx,
        raw_text=text,
        heading_text=None,
    )


def _section(text, idx):
    """A genuine <h3> section heading block (opener nests one level under it)."""
    return _AdapterBlock(
        html="",
        region_kind="heading",
        raw_block_index=idx,
        raw_text=text,
        heading_text=text,
    )


def _render(chapters, **kw):
    return normalize_cascade_to_ed4all(
        _Result(chapters), pdf_stem="synthetic_ch09", **kw
    )["html"]


# ---------------------------------------------------------------------------
# B3 — body LaTeX/markdown sanitation (pure function).
# ---------------------------------------------------------------------------


def test_body_textbf_to_strong_html():
    got = sanitize_body_latex(r"A \textbf{Square Root of a Number} is defined.")
    assert got == "A <strong>Square Root of a Number</strong> is defined."


def test_body_textit_to_em_html():
    assert sanitize_body_latex(r"the \textit{square root} of m") == (
        "the <em>square root</em> of m"
    )


def test_body_plain_mode_drops_markup_to_bare_word():
    got = sanitize_body_latex(r"\textbf{Rule} and \textit{term}", html=False)
    assert got == "Rule and term"


def test_body_protects_math_runs():
    # $...$ math is MathJax's — never touched, even a \textbf inside it.
    src = r"A number $\sqrt{m}$ with \textbf{root} $\textbf{x}$ end"
    got = sanitize_body_latex(src)
    assert r"$\sqrt{m}$" in got
    assert r"$\textbf{x}$" in got  # inside math → untouched
    assert "<strong>root</strong>" in got  # outside math → converted


def test_body_drops_tabular_and_md_sep_and_checkmark():
    src = r"before \begin{tabular}{|c|c|}a & b\end{tabular} | --- | --- | \checkmark after"
    got = sanitize_body_latex(src)
    assert "tabular" not in got
    assert "---" not in got
    assert "checkmark" not in got
    assert "before" in got and "after" in got


def test_body_noop_on_plain_prose():
    plain = "The order of operations is important here."
    assert sanitize_body_latex(plain) == plain


# ---------------------------------------------------------------------------
# B3 — integrated through the adapter render.
# ---------------------------------------------------------------------------


def test_render_strips_body_latex_no_leak():
    ch = _AdapterChapter(
        title="Chapter 9 Roots and Radicals",
        blocks=[
            _AdapterBlock(
                html=r"<p>A number is called a \textit{square root}. \textbf{Square Root of a Number}</p>",
                region_kind="paragraph",
                raw_block_index=0,
                raw_text=r"A number is called a \textit{square root}.",
                heading_text=None,
            ),
        ],
    )
    html = _render([ch])
    assert "\\textbf" not in html
    assert "\\textit" not in html
    assert "<em>square root</em>" in html
    assert "<strong>Square Root of a Number</strong>" in html


# ---------------------------------------------------------------------------
# A7 — opener promotion + data-semantik-opener; A5 — nearest-ancestor heading level.
# ---------------------------------------------------------------------------


def test_standalone_example_under_section_promoted_to_h4_opener():
    # Opener nested under a genuine <h3> section → <h4> (section_level + 1).
    ch = _AdapterChapter(
        title="Chapter 9 Roots and Radicals",
        blocks=[_section("9.1 Simplify Square Roots", 0), _para("EXAMPLE 9.1", 1)],
    )
    html = _render([ch])
    assert re.search(r'data-semantik-opener="worked_example"', html)
    assert re.search(r"<h4 id=\"[^\"]+\">Example 9\.1</h4>", html)


def test_standalone_example_directly_under_chapter_is_h3_no_skip():
    # A5 — opener directly under the chapter <h2> (no <h3> section yet) renders
    # at <h3>, NOT <h4>, so there is no h2->h4 level skip.
    ch = _AdapterChapter(
        title="Chapter 9 Roots and Radicals",
        blocks=[_para("EXAMPLE 9.1", 0)],
    )
    html = _render([ch])
    assert re.search(r'data-semantik-opener="worked_example"', html)
    assert re.search(r"<h3 id=\"[^\"]+\">Example 9\.1</h3>", html)
    assert "<h4" not in html  # no skipped level


def test_try_it_and_be_prepared_promoted():
    ch = _AdapterChapter(
        title="Chapter 9 Roots and Radicals",
        blocks=[
            _section("9.1 Simplify Square Roots", 0),
            _para("TRY IT 9.7", 1),
            _para("BE PREPARED 9.1", 2),
        ],
    )
    html = _render([ch])
    assert 'data-semantik-opener="try_it"' in html
    assert ">Try It 9.7</h4>" in html
    assert 'data-semantik-opener="readiness_check"' in html
    assert ">Be Prepared 9.1</h4>" in html


def test_leading_objectives_split_emits_heading_plus_ul():
    # A fused "Learning Objectives By the end ... : item\nitem" paragraph splits
    # into an <h4> opener + a <ul> of bullets (newline-delimited → list).
    fused = (
        "Learning Objectives By the end of this section you will be able to:\n"
        "Simplify expressions with square roots\nEstimate square roots"
    )
    ch = _AdapterChapter(
        title="Chapter 9 Roots and Radicals",
        blocks=[_section("9.1 Simplify Square Roots", 0), _para(fused, 1)],
    )
    html = _render([ch])
    assert 'data-semantik-opener="objectives"' in html
    assert ">Learning Objectives</h4>" in html
    assert "<ul>" in html
    assert "<li>Simplify expressions with square roots</li>" in html
    assert "<li>Estimate square roots</li>" in html


def test_objectives_flat_prose_stays_paragraph_no_fabricated_bullets():
    # Flat run-on objectives (no delimiter) must NOT fabricate <li> boundaries.
    fused = (
        "Learning Objectives By the end of this section you will be able to: "
        "Simplify expressions with square roots Estimate square roots"
    )
    ch = _AdapterChapter(
        title="Chapter 9 Roots and Radicals",
        blocks=[_section("9.1 Simplify Square Roots", 0), _para(fused, 1)],
    )
    html = _render([ch])
    assert ">Learning Objectives</h4>" in html
    # No <ul> minted from run-on prose (conservative anti-fabrication).
    assert "<ul>" not in html


def test_prose_example_not_promoted():
    # A paragraph merely starting with "Example" (no number) is untouched.
    ch = _AdapterChapter(
        title="Chapter 9 Roots and Radicals",
        blocks=[_para("Examples are common when learning algebra.", 0)],
    )
    html = _render([ch])
    assert "data-semantik-opener" not in html
    assert "<h4" not in html


# ---------------------------------------------------------------------------
# Round-3 Defect 3 — a table-declared block's trailing fused openers surface.
# ---------------------------------------------------------------------------
def test_table_block_trailing_openers_surface():
    # The ch02 s320 shape: a "table"-role block whose pipe run spilled trailing
    # TRY IT / EXAMPLE markers PAST the last pipe. The trailing markers become
    # real opener headings; the pipe-cell content is de-fused.
    fused = (
        "| Check: | 14 - 23 | TRY IT 2.35 | Solve: $a$. "
        "TRY IT 2.36 Solve: $b$. EXAMPLE 2.19 Solve: $c$."
    )
    blk = _para(fused, 1, kind="table")
    blk.block_role = "table"
    ch = _AdapterChapter(
        title="Chapter 2 Solving Linear Equations and Inequalities",
        blocks=[_section("2.4 Solve Equations", 0), blk],
    )
    html = _render([ch])
    assert ">Try It 2.36</h4>" in html
    assert ">Example 2.19</h4>" in html
    assert 'data-semantik-opener="try_it"' in html
