"""Unit tests — OCR heading-candidate classifier + adapter normalization pass.

Drives the 2026-07-03 scan-audit fixes on the ADAPTER (re-render + conversion)
path: decorated "Solution" run-in labels, leading/trailing/repeat running
headers, and scanner-watermark garbage headings. CPU-only, no models.
"""

from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)
from lib.semantik.heading_classifier import (
    is_decorated_solution_label,
    is_running_header,
    is_standalone_folio,
    is_watermark_garbage_heading,
    repeated_running_header_indices,
    strip_trailing_running_header,
)


# ---------------------------------------------------------------------------
# (a) decorated "Solution" run-in labels.
# ---------------------------------------------------------------------------


def test_decorated_solution_labels_detected():
    for t in [") Solution", "™ Solution", "“ Solution", "Solution", "  Solution "]:
        assert is_decorated_solution_label(t), t


def test_real_solution_headings_not_run_in_labels():
    # A real section heading that merely contains "Solution(s)" is not a
    # standalone run-in label (anchored $).
    for t in ["Find Solutions to a Linear Equation", "Solution set of the system"]:
        assert not is_decorated_solution_label(t), t


# ---------------------------------------------------------------------------
# (b) running headers — leading / trailing / repeat.
# ---------------------------------------------------------------------------


def test_running_header_leading_and_trailing_page():
    assert is_running_header("188 Chapter 1 Foundations")
    assert is_running_header("686 Chapter 6 Polynomials")
    assert is_running_header("Chapter 9 Roots and Radicals 1039")


def test_bare_chapter_opener_not_a_running_header():
    # No page number → not furniture on its own (the FIRST opener is real).
    assert not is_running_header("Chapter 4 Graphs")
    assert not is_running_header("Chapter 9 Roots and Radicals")


def test_repeat_count_keeps_first_drops_rest():
    titles = ["Chapter 4 Graphs"] * 5 + ["Introduction"]
    furniture = repeated_running_header_indices(titles)
    assert furniture == {1, 2, 3, 4}  # index 0 kept as the real opener


def test_repeat_count_below_threshold_keeps_all():
    titles = ["Chapter 4 Graphs", "Chapter 4 Graphs", "Chapter 4 Graphs"]
    assert repeated_running_header_indices(titles) == set()


# ---------------------------------------------------------------------------
# (c) scanner-watermark garbage.
# ---------------------------------------------------------------------------


def test_watermark_garbage_demoted():
    for t in ["Lexar 2", "Lexar 117 rr", "Ivo", "Niw", "AIS", "Nolo"]:
        assert is_watermark_garbage_heading(t), t


def test_legit_short_headings_survive_watermark_gate():
    for t in [
        "9.3 Add and Subtract Square Roots",  # section number
        "Key Concepts",                        # whitelist + content word
        "Key Terms",
        "Practice Test",
        "Graphs",                              # common short chapter word
        "Roots and Radicals",                  # content words
        "Polynomials",                         # long real word
        "Foundations",
        "Introduction",
    ]:
        assert not is_watermark_garbage_heading(t), t


# ---------------------------------------------------------------------------
# Adapter integration — the normalization pass runs inside
# normalize_cascade_to_ed4all (the shared conversion + re-render chokepoint).
# ---------------------------------------------------------------------------


def _heading_block(text: str, idx: int) -> _AdapterBlock:
    return _AdapterBlock(
        html="", region_kind="heading", raw_block_index=idx,
        raw_text=text, heading_text=text,
    )


def _render(chapters, **kw):
    result = type("R", (), {})()
    result.chapters = chapters
    result.exit_action = "ship_with_confidence"
    result.wcag_status = "passed"
    result.lang = "en"
    return normalize_cascade_to_ed4all(result, pdf_stem="scan-ch04", **kw)


def test_adapter_demotes_decorated_solution_block():
    ch = _AdapterChapter(
        title="Chapter 4 Graphs",
        blocks=[_heading_block("“ Solution", 5)],
    )
    html = _render([ch])["html"]
    # No spurious <h3> heading; the label survives as a pedagogy-solution <p>.
    assert "<h3" not in html or "Solution</h3>" not in html
    assert 'class="pedagogy-solution"' in html
    assert "Solution" in html


def test_adapter_drops_running_header_block():
    ch = _AdapterChapter(
        title="Chapter 4 Graphs",
        blocks=[_heading_block("686 Chapter 6 Polynomials", 7)],
    )
    html = _render([ch])["html"]
    # Furniture running header block is dropped entirely.
    assert "686 Chapter 6 Polynomials" not in html


def test_adapter_demotes_watermark_block():
    ch = _AdapterChapter(
        title="Chapter 4 Graphs", blocks=[_heading_block("Lexar 117 rr", 9)]
    )
    html = _render([ch])["html"]
    assert "<h3" not in html or "Lexar 117 rr</h3>" not in html
    assert "Lexar 117 rr" in html  # preserved as paragraph text


def test_adapter_neutralizes_repeated_chapter_titles():
    chapters = [
        _AdapterChapter(title="Chapter 4 Graphs", blocks=[_heading_block("1.1 Real", 1)])
        for _ in range(5)
    ]
    html = _render(chapters)["html"]
    # The running-header title survives as an <h2> exactly ONCE (the first);
    # the rest are demoted out of the heading stream (presentation div).
    assert html.count("<h2>Chapter 4 Graphs</h2>") == 1
    assert html.count("semantik-continuation") == 4


def test_adapter_neutralizes_watermark_chapter_title():
    chapters = [
        _AdapterChapter(title="Chapter 4 Graphs", blocks=[_heading_block("1.1 Real", 1)]),
        _AdapterChapter(title="Nolo", blocks=[_heading_block("1.2 Real", 2)]),
    ]
    html = _render(chapters)["html"]
    assert "<h2>Nolo</h2>" not in html


def test_adapter_title_override_wins():
    ch = _AdapterChapter(title="188 Chapter 1 Foundations", blocks=[_heading_block("1.1 X", 1)])
    out = _render([ch], title_override="Elementary Algebra 2e — Chapter 1")
    assert "<h1>Elementary Algebra 2e — Chapter 1</h1>" in out["html"]
    assert "<title>Elementary Algebra 2e — Chapter 1</title>" in out["html"]


def test_adapter_real_section_heading_survives():
    ch = _AdapterChapter(
        title="Chapter 4 Graphs",
        blocks=[_heading_block("4.1 Use the Rectangular Coordinate System", 3)],
    )
    html = _render([ch])["html"]
    assert "Use the Rectangular Coordinate System</h3>" in html


# ---------------------------------------------------------------------------
# (j) Round-3 Defect 2 — mid-body folio leakage.
# ---------------------------------------------------------------------------
def test_is_standalone_folio():
    assert is_standalone_folio("233")
    assert is_standalone_folio("  1074  ")
    # an exercise number followed by content is NOT a bare folio
    assert not is_standalone_folio("243. Simplify the expression")
    assert not is_standalone_folio("")
    assert not is_standalone_folio("12345")  # 5 digits — not a folio shape


def test_strip_trailing_running_header():
    assert (
        strip_trailing_running_header(
            "$= 2(7m - 11)$ Chapter 2 Solving Linear Equations and Inequalities 251"
        )
        == "$= 2(7m - 11)$"
    )
    # preserves a closing HTML tag
    assert (
        strip_trailing_running_header("<p>foo Chapter 9 Roots and Radicals 1063</p>")
        == "<p>foo</p>"
    )
    # a real answer / cross-reference is NOT stripped
    assert strip_trailing_running_header("the difference is 107.") is None
    assert strip_trailing_running_header("see Chapter 2 provides examples") is None
    assert strip_trailing_running_header("wait 15 minutes") is None


def _body_block(text: str, idx: int, kind: str = "paragraph") -> _AdapterBlock:
    return _AdapterBlock(
        html=f"<p>{text}</p>", region_kind=kind, raw_block_index=idx,
        raw_text=text,
    )


def test_adapter_drops_standalone_body_folio():
    ch = _AdapterChapter(
        title="Chapter 9 Roots and Radicals",
        blocks=[
            _heading_block("9.1 Simplify Radicals", 1),
            _body_block("1074", 2),
            _body_block("Real content follows here.", 3),
        ],
    )
    html = _render([ch])["html"]
    assert "<p>1074</p>" not in html
    assert "Real content follows here." in html


def test_adapter_strips_trailing_running_header_from_body():
    ch = _AdapterChapter(
        title="Chapter 2 Solving Linear Equations and Inequalities",
        blocks=[
            _heading_block("2.1 Solve Equations", 1),
            _body_block(
                "The result is x = 3 Chapter 2 Solving Linear Equations "
                "and Inequalities 251",
                2,
            ),
        ],
    )
    html = _render([ch])["html"]
    assert "The result is x = 3" in html
    assert "251" not in html
    assert "Solving Linear Equations and Inequalities 251" not in html


# ---------------------------------------------------------------------------
# (e) Package 3 — numbered per-section apparatus banner demotion.
# ---------------------------------------------------------------------------


def test_is_numbered_apparatus_heading_predicate():
    from lib.semantik.heading_classifier import is_numbered_apparatus_heading as nah

    # Ordinal-prefixed apparatus display names → demote.
    assert nah("1.4 Review Exercises")
    assert nah("10.3 Practice Test")
    assert nah("1.4 Key Terms")
    assert nah("9.2 Chapter Review")
    # A real numbered section title strips to a non-apparatus name → keep.
    assert not nah("1.3 Add and Subtract Integers")
    assert not nah("2.1 Foundations")
    # Bare (unnumbered) apparatus name is NOT the numbered-banner case — the
    # standalone apparatus path owns it (kept as a real EOC section heading).
    assert not nah("Review Exercises")
    assert not nah("Exercises")
    assert not nah("")


def test_numbered_bare_exercises_demoted_apparatus_unchanged():
    # FIX 2 — the lexicon apparatus display names carry no bare "Exercises", so
    # "N.M Exercises" banners used to survive as headings. The separate
    # numbered_apparatus_names lexicon key demotes the NUMBERED form ONLY, while
    # is_apparatus_heading('Exercises') (the bare standalone) is UNCHANGED.
    from lib.semantik.heading_classifier import (
        is_apparatus_heading as ah,
        is_numbered_apparatus_heading as nah,
    )

    # The bug the census found: numbered "N.M Exercises" now demotes.
    assert nah("2.1 Exercises")
    assert nah("10.3 Exercises")
    # An existing apparatus display name is still caught (regression guard).
    assert nah("3.4 Review Exercises")
    # A real numbered section title is never demoted.
    assert not nah("2.1 Real Title")
    # CRITICAL: bare unnumbered "Exercises" is NOT a numbered banner, and the
    # standalone is_apparatus_heading behavior for it is unchanged (still False,
    # since "Exercises" is deliberately absent from apparatus_sections).
    assert not nah("Exercises")
    assert ah("Exercises") is False
    # Sanity: a real bare apparatus display name still promotes via ah.
    assert ah("Review Exercises") is True


def test_ocr_fused_numbered_banner_demoted():
    # FIX 1 — OCR fused the section-exercise banner with its "Practice Makes
    # Perfect" subhead into one heading, so the ordinal-stripped remainder
    # ("EXERCISES Practice Makes Perfect") is not an EXACT lexicon entry. The
    # compositional rule (numbered-word prefix + apparatus tail) demotes it.
    from lib.semantik.heading_classifier import is_numbered_apparatus_heading as nah

    assert nah("2.7 EXERCISES Practice Makes Perfect")
    assert nah("4.1 EXERCISES Practice Makes Perfect")
    # Case-insensitive; the title-case fused form also demotes.
    assert nah("2.7 Exercises Practice Makes Perfect")
    # The bare "Practice Makes Perfect" numbered banner also demotes now.
    assert nah("2.7 Practice Makes Perfect")
    # REGRESSION LOCK: a real "N.M Exercises in <topic>" section TITLE whose
    # tail is NOT apparatus vocabulary must NEVER demote.
    assert not nah("3.4 Exercises in Measure Theory")
    assert not nah("2.1 Exercises of the Chapter Reviewed")
    # Standalone (unnumbered) fused text is not the numbered-banner case.
    assert not nah("EXERCISES Practice Makes Perfect")


def test_adapter_demotes_numbered_apparatus_banner():
    ch = _AdapterChapter(
        title="Chapter 1 Foundations",
        blocks=[
            _heading_block("1.1 Introduction to Whole Numbers", 0),
            _heading_block("1.4 Review Exercises", 4),
            _body_block("In the following exercises, simplify.", 5),
        ],
    )
    html = _render([ch])["html"]
    # The numbered apparatus banner does not survive as a heading — no
    # per-section boundary is minted from it. Content still renders as prose.
    import re

    assert not re.search(
        r"<h[1-6][^>]*>[^<]*1\.4 Review Exercises", html
    ), "numbered apparatus banner must not survive as a heading"
    assert "1.4 Review Exercises" in html  # text preserved as prose
    # The real numbered section heading is untouched.
    assert "1.1 Introduction to Whole Numbers" in html


def test_adapter_keeps_real_numbered_section_heading():
    ch = _AdapterChapter(
        title="Chapter 1 Foundations",
        blocks=[_heading_block("1.3 Add and Subtract Integers", 0)],
    )
    html = _render([ch])["html"]
    assert "1.3 Add and Subtract Integers" in html
    # A real section title stays a heading (not demoted to a bare <p>).
    assert "<h" in html
