r"""Round-5 visual-convergence ledger regression tests (2026-07-04).

Covers the six residual-defect items cleared in round 5 of the SemantiK
visual-convergence loop (``plans/enduser-html-improvements-2026-07.md`` round-5
ledger):

* ITEM 1 — pure gutter + garbled-marker DEBRIS blocks (``>| TRYIT::``) collapse
  to suppressed furniture, with strict anti-fabrication guards.
* ITEM 2 — ``::`` colon-runs fused INTO / abutting a math span (``$ :: N & $``,
  ``\text{ TRY IT }:: 9.19``, ``\textbf{TRY IT ::}``) are folded while real math
  and single label colons survive.
* ITEM 3 — the ``trvit`` OCR-confusable normalizes to ``TRY IT`` (number-guarded).
* ITEM 4 — a LABEL-ONLY stacked-opener block ("EXAMPLE 2.3 Solution") emits the
  stacked opener headings (empty groups) instead of shipping as prose.
* ITEM 5 — the cross-block severed-math NEUTRALIZE-AND-ANNOTATE arm (documented
  as the safe disposition; a merge would fabricate prose-in-math). This test
  pins the current behavior so a future "merge" regression is caught.
* ITEM 6 — a bare exercise-number lead-in ("430.") before a nested exercise list
  folds into the list's ``start`` attribute, not a ``<p>430</p>`` debris paragraph.

All fixtures are inline synthetic HTML/text — NO reference to ``inputs/`` or any
course-data path.
"""

from __future__ import annotations

import pytest

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _CONFUSABLE_TRYIT_RE,  # noqa: F401 (imported to assert existence)
    _is_marker_debris,
    _normalize_confusable_marker_text,
    _scrub_marker_artifacts,
    normalize_cascade_to_ed4all,
)
from lib.semantik.opener_classifier import (
    ROLE_SOLUTION,
    ROLE_WORKED_EXAMPLE,
    split_label_only_openers,
)
from lib.semantik.structure_emit import parse_list


def _result(chapters):
    r = type("R", (), {})()
    r.exit_action = "ship_with_confidence"
    r.wcag_status = "passed"
    r.lang = "en"
    r.chapters = chapters
    return r


# ---------------------------------------------------------------------------
# ITEM 1 — marker-debris block collapse
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        ">| TRYIT::",
        ">| TRYIT:: >| TRYIT::",
        "&gt;| TRYIT",
        "> TRYIT |",
    ],
)
def test_item1_pure_marker_debris_is_debris(text):
    assert _is_marker_debris(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "> TRY IT :: 9.174 >",  # has a number -> promotable, never debris
        "Try It",  # gutter-free -> opener-promotion territory, not debris
        r">| TRY IT $\sqrt{2}$",  # math-bearing -> content
        ">| TRY IT to solve this",  # >= 3 word tokens -> real content
        ">| something else",  # no recognised marker token
        "",
    ],
)
def test_item1_guards_refuse_non_debris(text):
    assert _is_marker_debris(text) is False


def test_item1_debris_block_collapses_content_survives():
    """A pure ``>| TRYIT::`` block is suppressed; a numbered / content-bearing
    TRY-IT block and real prose are untouched."""
    chapters = [
        _AdapterChapter(
            title="Chapter 2 Solving Equations",
            blocks=[
                _AdapterBlock(
                    html="<p>Real content sentence stays put here.</p>",
                    region_kind="paragraph",
                    raw_block_index=0,
                    raw_text="Real content sentence stays put here.",
                    pages=[1],
                    block_role="paragraph",
                ),
                _AdapterBlock(
                    html="<p>&gt;| TRYIT::</p>",
                    region_kind="code_block",
                    raw_block_index=1,
                    raw_text=">| TRYIT::",
                    pages=[1],
                    block_role="code_block",
                ),
                _AdapterBlock(
                    html="<p>&gt;| TRYIT:: >| TRYIT::</p>",
                    region_kind="paragraph",
                    raw_block_index=2,
                    raw_text=">| TRYIT:: >| TRYIT::",
                    pages=[1],
                    block_role="paragraph",
                ),
                _AdapterBlock(
                    html="<p>&gt;| TRYIT:: real answer content follows here</p>",
                    region_kind="paragraph",
                    raw_block_index=3,
                    raw_text=">| TRYIT:: real answer content follows here",
                    pages=[1],
                    block_role="paragraph",
                ),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_result(chapters), pdf_stem="algebra_ch2")
    html = out["html"]
    assert "Real content sentence stays put here." in html
    assert "real answer content follows here" in html  # content-bearing survives
    # The two PURE-debris blocks (``>| TRYIT::`` and its double) are gone; only
    # the ONE content-bearing block keeps its ``TRYIT`` prefix (it is not debris).
    assert html.count("TRYIT") == 1
    sidecar = [s["data"]["text"] for s in out["synthesized_sidecar"]["sections"]]
    # Exactly the content-bearing section survives among the TRYIT-bearing blocks.
    assert sum("TRYIT" in t for t in sidecar) == 1
    assert any("real answer content follows here" in t for t in sidecar)


# ---------------------------------------------------------------------------
# ITEM 2 — ``::`` fused into / abutting math
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expect_absent",
    [
        (r"$ :: 9.19 & $", "::"),  # trapped inside a spurious span
        (r"$\text { TRY IT }$ $ :: 9.19 & $", "::"),  # abuts the open delim
        (r"$\begin{array}{ll}\text { TRY IT }:: 9.19 & $", "::"),  # after a brace
        (r"$\textbf{TRY IT ::} 2.24$", "::"),  # before a brace
        (r"$x=2$ :: TRY IT", "::"),  # boundary-adjacent (ledger case A)
        (r":: $y$", "::"),  # boundary-adjacent (ledger case B)
        (r"$\textbf{TRY IT :: 1.173}$", "::"),  # LETTER::DIGIT interior to a span
        (r"$\begin{array}\text{TRY IT :: 4.11}$", "::"),  # marker :: number in math
    ],
)
def test_item2_math_adjacent_colon_runs_folded(text, expect_absent):
    out = _scrub_marker_artifacts(text, html=False)
    assert expect_absent not in out


@pytest.mark.parametrize(
    "text",
    [
        r"$a :: b$ interior proportion stays",  # not brace/$-adjacent
        "Solving Applications with Formulas: keep the single colon",
        r"the type $f :: A \to B$ stays",  # Haskell type-sig in math (letter after)
        r"the ratio $1::2$ stays",  # in-math proportion (digit before ::)
    ],
)
def test_item2_real_math_and_single_colon_preserved(text):
    out = _scrub_marker_artifacts(text, html=False)
    assert out == text  # a no-op — genuine content untouched


# ---------------------------------------------------------------------------
# ITEM 3 — ``trvit`` OCR-confusable normalization
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("trvit:: 26: see 8 th +12", "TRY IT:: 26: see 8 th +12"),
        ("TRVIT 26", "TRY IT 26"),
        ("[>] trvit 9.5 Simplify", "[>] TRY IT 9.5 Simplify"),
    ],
)
def test_item3_confusable_normalized_when_numbered(text, expected):
    assert _normalize_confusable_marker_text(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "trvit with no number is left alone",  # number-guard: not followed by a digit
        "the word private has vit in it",  # not a whole-word confusable
        "",
    ],
)
def test_item3_confusable_number_guard(text):
    assert _normalize_confusable_marker_text(text) == (text or text)


def test_item3_confusable_surfaces_marker_in_render():
    """A fused ``trvit 26`` heading blob surfaces the ``TRY IT`` marker (no garble)."""
    blob = (
        "| $x = 2$ | TRY IT :: 2.61 Solve: $12j = -4j$. [>] trvit:: 26: see 8 th"
    )
    chapters = [
        _AdapterChapter(
            title="Chapter 2",
            blocks=[
                _AdapterBlock(
                    html="",
                    region_kind="heading",
                    raw_block_index=0,
                    raw_text=blob,
                    heading_text=blob,
                    pages=[1],
                    heading_level=3,
                )
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_result(chapters), pdf_stem="algebra_ch2")
    html = out["html"]
    assert "trvit" not in html.lower()
    assert "TRY IT 26" in html  # de-garbled + de-doubled marker surfaced


# ---------------------------------------------------------------------------
# ITEM 4 — label-only stacked openers
# ---------------------------------------------------------------------------
def test_item4_split_label_only_openers():
    assert split_label_only_openers("EXAMPLE 2.3 Solution") == [
        ("Example 2.3", ROLE_WORKED_EXAMPLE),
        ("Solution", ROLE_SOLUTION),
    ]
    # A single label is the standalone case; trailing prose is the leading case.
    assert split_label_only_openers("Learning Objectives") is None
    assert split_label_only_openers("EXAMPLE 2.3 Simplify the expression") is None
    assert split_label_only_openers("example 2.3 solution") is None  # lowercase


def test_item4_label_only_block_promotes_stacked_headings():
    chapters = [
        _AdapterChapter(
            title="Chapter 2",
            blocks=[
                _AdapterBlock(
                    html="<p>EXAMPLE 2.3 Solution</p>",
                    region_kind="paragraph",
                    raw_block_index=0,
                    raw_text="EXAMPLE 2.3 Solution",
                    pages=[5],
                    block_role="paragraph",
                )
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_result(chapters), pdf_stem="algebra_ch2")
    html = out["html"]
    # The example number is a real opener heading (empty group).
    assert 'data-semantik-opener="worked_example"' in html
    assert "Example 2.3</h" in html
    # The Solution marker is honestly surfaced (opener role), never dropped, and
    # the raw run-on "EXAMPLE 2.3 Solution" prose is gone.
    assert 'data-semantik-opener="solution"' in html or "pedagogy-solution" in html
    assert "EXAMPLE 2.3 Solution" not in html


# ---------------------------------------------------------------------------
# ITEM 5 — cross-block severed math: neutralize-and-annotate (documented safe arm)
# ---------------------------------------------------------------------------
def test_item5_severed_display_math_neutralized_not_merged():
    """The OCR splits a display span across a block boundary AND fuses PROSE into
    the math region (``\\[ x = 5 Then we were left with five.``). The safe
    disposition is the per-block NEUTRALIZE arm: the orphan ``\\[`` is dropped so
    the prose stays readable prose — a cross-block MERGE would re-create the
    ``\\[ x = 5 Then we were left with five. \\]`` prose-in-math garbage. This
    test pins that behavior so a future merge attempt is caught."""
    chapters = [
        _AdapterChapter(
            title="Chapter 2",
            blocks=[
                _AdapterBlock(
                    html=r"<p>from each side. \[ x = 5 Then we were left with five.</p>",
                    region_kind="paragraph",
                    raw_block_index=0,
                    raw_text=r"from each side. \[ x = 5 Then we were left with five.",
                    pages=[1],
                    block_role="paragraph",
                ),
                _AdapterBlock(
                    html=r"<p>\] Check: five plus three does equal eight.</p>",
                    region_kind="paragraph",
                    raw_block_index=1,
                    raw_text=r"\] Check: five plus three does equal eight.",
                    pages=[1],
                    block_role="paragraph",
                ),
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_result(chapters), pdf_stem="algebra_ch2")
    html = out["html"]
    # The prose survives as readable text; NO display span wraps the prose.
    assert "Then we were left with five" in html
    assert "Check: five plus three does equal eight" in html
    assert r"\[ x = 5 Then we were left with five. \]" not in html


# ---------------------------------------------------------------------------
# ITEM 6 — bare exercise-number lead-in -> list ``start`` attribute
# ---------------------------------------------------------------------------
def test_item6_bare_number_lead_folds_into_exercise_list_start():
    text = (
        "430. (a) $x < -2$ (b) $x > -1$ (c) $x < 0$ "
        "(a) $x > 1$ (b) $x < -2$ (c) $x < 4$"
    )
    html = parse_list(text)
    assert html is not None
    assert 'class="semantik-exercise-list" start="430"' in html
    # The bare-number debris paragraph is gone.
    assert "<p>430</p>" not in html


def test_item6_single_cycle_alpha_list_keeps_number_as_prose():
    """A single-cycle ``<ol type="a">`` (alpha parts, no numeric wrapper) keeps a
    bare-number lead as prose — a numeric ``start`` on an alpha list is
    nonsensical, so the fold is scoped to the nested exercise list only."""
    text = "267. (a) $7(-2)$ (b) $\\sqrt{7}$"
    html = parse_list(text)
    assert html is not None
    assert "start=" not in html
    assert '<ol type="a">' in html
