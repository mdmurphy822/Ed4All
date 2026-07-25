"""Chapter-ladder reconcile (SEMANTIK_CHAPTER_LADDER_RECONCILE) — adapter side.

Reproduces the whole-book reference-corpus shape with a SYNTHETIC,
anonymized fixture (invented placeholder text, NO real course content):

* a level-1 ``doc_title`` book heading;
* a preface numbering its own sections ``0.M`` (front matter);
* explicit level-1 "Chapter N" openers for chapters 1, 2 and 4 — chapter 3's
  opener was MISSED by detection (only its ``3.1`` section survives), and
  chapter 4 has an opener but NO numbered sections;

i.e. the explicit opener series and the implicit ``N.M`` major series each
miss DIFFERENT members. Asserts:

1. build_chapters_ir UNIONS the two series into ONE complete chapter set;
2. front matter (the ``0.M`` preface) does NOT mint a "Chapter 0" root;
3. an opener heading is consumed as its chapter's title, never duplicated as
   an in-chapter block (the double-ladder kill);
4. the rendered document carries the BOOK title as ``<h1>``, one ``<h2>`` per
   chapter, and no phantom "Chapter 0" anywhere;
5. the explicit-falsey flag restores the legacy behavior byte-for-byte at the
   IR level (the revert lever).

Run:
  python -m pytest lib/semantik/tests/test_cascade_ir_chapter_ladder.py -q
"""

from __future__ import annotations

import re

import pytest

from lib.semantik.adapter import _select_document_title, normalize_cascade_to_ed4all
from lib.semantik.cascade_ir import build_chapters_ir


def _heading(idx: int, text: str, level: int, page: int) -> dict:
    return {
        "region_index": idx,
        "region_kind": "heading",
        "role": "heading",
        "confidence": 0.9,
        "wcag_status": "passed",
        "first_raw_block_index": idx,
        "pages": [page],
        "heading_text": text,
        "level": level,
        "figure_alt": None,
        "raw_text": text,
    }


def _para(idx: int, text: str, page: int) -> dict:
    return {
        "region_index": idx,
        "region_kind": "paragraph",
        "role": None,
        "confidence": 0.9,
        "wcag_status": "passed",
        "first_raw_block_index": idx,
        "pages": [page],
        "heading_text": None,
        "level": None,
        "figure_alt": None,
        "raw_text": text,
    }


def _whole_book_provenance() -> list[dict]:
    """The reconciled (post-transform) whole-book shape: openers at level 1."""
    return [
        _heading(0, "SAMPLE WIDGET COMPENDIUM", 1, 1),
        _para(1, "A placeholder preface paragraph about widgets.", 1),
        # front matter: the preface numbers its own sections 0.M
        _heading(2, "0.1 Who should read this guide", 2, 2),
        _para(3, "Placeholder audience prose for the preface pages.", 2),
        # chapter 1: explicit opener + numbered section
        _heading(4, "Chapter 1", 1, 3),
        _heading(5, "1.1 Alpha Widgets", 2, 3),
        _para(6, "Placeholder prose describing alpha widgets at length.", 3),
        # chapter 2: explicit opener + numbered section
        _heading(7, "Chapter 2", 1, 4),
        _heading(8, "2.1 Beta Widgets", 2, 4),
        _para(9, "Placeholder prose describing beta widgets at length.", 4),
        # chapter 3: opener MISSED by detection — only the section survives
        _heading(10, "3.1 Gamma Widgets", 2, 5),
        _para(11, "Placeholder prose describing gamma widgets at length.", 5),
        # chapter 4: explicit opener, NO numbered sections
        _heading(12, "Chapter 4", 1, 6),
        _heading(13, "Delta Widgets", 3, 6),
        _para(14, "Placeholder prose describing delta widgets at length.", 6),
    ]


def _book_result() -> dict:
    prov = _whole_book_provenance()
    return {
        "region_provenance": prov,
        "heading_tree": [
            [p["level"], p["heading_text"]]
            for p in prov
            if p["region_kind"] == "heading"
        ],
        "wcag_status": "passed",
        "exit_action": "ship_with_confidence",
        "theta_score": 0.9,
        "flags": [],
    }


def test_union_yields_one_complete_chapter_ladder():
    chapters = build_chapters_ir(_book_result())
    titles = [ch.title for ch in chapters]
    # leading front-matter chapter carries the book title; chapters 1..4
    # complete — chapter 3 derived from its section major, no "Chapter 0".
    assert titles == [
        "SAMPLE WIDGET COMPENDIUM",
        "Chapter 1",
        "Chapter 2",
        "Chapter 3",
        "Chapter 4",
    ]


def test_front_matter_does_not_mint_chapter_zero():
    chapters = build_chapters_ir(_book_result())
    assert not any(ch.title == "Chapter 0" for ch in chapters)
    # the 0.1 preface section stays inside the leading front-matter chapter
    leading = chapters[0]
    preface = [
        b for b in leading.blocks
        if str(getattr(b, "heading_text", "") or "").startswith("0.1")
    ]
    assert preface, "preface 0.M section must stay in the leading chapter"


def test_opener_heading_not_duplicated_as_block():
    """The double-ladder kill: a consumed opener never re-renders in-chapter."""
    chapters = build_chapters_ir(_book_result())
    for ch in chapters:
        for block in ch.blocks:
            text = str(getattr(block, "heading_text", "") or "")
            assert not re.fullmatch(r"Chapter \d+", text), (
                f"opener {text!r} duplicated as a block inside {ch.title!r}"
            )


def test_document_title_is_book_title_not_a_chapter():
    chapters = build_chapters_ir(_book_result())
    assert _select_document_title(chapters, "stem") == "SAMPLE WIDGET COMPENDIUM"


def test_rendered_html_single_ladder_and_h1():
    result = _book_result()
    result["chapters"] = build_chapters_ir(result)
    out = normalize_cascade_to_ed4all(result, pdf_stem="sample-widget-compendium")
    html = out["html"]
    assert "<h1>SAMPLE WIDGET COMPENDIUM</h1>" in html
    assert "Chapter 0" not in html
    # each chapter ordinal appears as EXACTLY ONE heading element (one ladder)
    headings = re.findall(r"<h(\d)[^>]*>([^<]*)</h\1>", html)
    for ordinal in ("1", "2", "3", "4"):
        chapter_heads = [
            (lvl, txt) for lvl, txt in headings
            if txt.strip() == f"Chapter {ordinal}"
        ]
        assert len(chapter_heads) == 1, (ordinal, chapter_heads)
        assert chapter_heads[0][0] == "2", chapter_heads  # the article <h2>


def test_off_flag_restores_legacy_defect_shape(monkeypatch):
    """The revert lever: with the flag explicitly falsey the guards never run
    — the preface's 0.M numbering mints the phantom "Chapter 0" root again
    and legacy title selection picks it as the document title."""
    monkeypatch.setenv("SEMANTIK_CHAPTER_LADDER_RECONCILE", "0")
    chapters = build_chapters_ir(_book_result())
    titles = [ch.title for ch in chapters]
    assert "Chapter 0" in titles  # the legacy front-matter chapter root
    # legacy title selection: the earliest bare "Chapter N" wins → Chapter 0
    assert _select_document_title(chapters, "stem") == "Chapter 0"


def test_single_chapter_corpus_unchanged():
    """A per-chapter corpus (one opener, its own sections) — the historical
    shape — must not take the union path or change titles."""
    prov = [
        _heading(0, "Chapter 3: Gamma Widgets", 1, 1),
        _heading(1, "3.1 Gamma Basics", 2, 1),
        _para(2, "Placeholder prose describing gamma widget basics.", 1),
        _heading(3, "3.2 Gamma Advanced", 2, 2),
        _para(4, "Placeholder prose describing advanced gamma widgets.", 2),
    ]
    chapters = build_chapters_ir(prov)
    assert [ch.title for ch in chapters] == ["Chapter 3: Gamma Widgets"]
    assert (
        _select_document_title(chapters, "stem") == "Chapter 3: Gamma Widgets"
    )


# ── Judged / layout heading-LEVEL differentiation ───────────────────────────
# The SEMANTIK_HEADING_JUDGE re-levels pending sub-chapter headings (L2 direct
# section vs L3 subsection) and writes the corrected ``level`` back onto the
# layout heading blocks. The render must map that layout level relative to the
# chapter root (chapter opener -> <h2>, L2 -> <h3>, L3 -> <h4>) instead of the
# pre-fix flatten that defaulted EVERY heading to <h3> and nullified the judge.
# Both the fresh in-lane conversion and the standalone judge re-render funnel
# through this same ``build_chapters_ir`` -> ``normalize_cascade_to_ed4all``
# seam, so a fix here covers both.
def _rendered_headings(html: str) -> list[tuple[str, str]]:
    """(level, text) for every ``<hN>…</hN>`` in the rendered HTML."""
    return re.findall(r"<h(\d)[^>]*>([^<]*)</h\1>", html)


def _level_of(html: str, text: str) -> str | None:
    for lvl, txt in _rendered_headings(html):
        if txt.strip() == text:
            return lvl
    return None


def _render_book(result: dict) -> str:
    result["chapters"] = build_chapters_ir(result)
    return normalize_cascade_to_ed4all(result, pdf_stem="sample")["html"]


def test_judged_heading_levels_render_h2_h3_h4():
    """Chapter opener -> <h2>, L2 direct section -> <h3>, L3 subsection -> <h4>.

    The pre-fix flatten (the L3 subsection rendered as <h3>) is asserted as the
    FAILURE mode — the whole point of honoring the judge's L2-vs-L3 verdict.
    """
    html = _render_book(_book_result())
    # chapter opener (level 1) is consumed as the chapter <h2> title.
    assert _level_of(html, "Chapter 1") == "2", html
    # L2 direct section (the N.M spine) -> <h3>.
    assert _level_of(html, "1.1 Alpha Widgets") == "3", html
    # L3 subsection ("Delta Widgets", level 3) -> <h4>, NOT the pre-fix <h3>.
    assert _level_of(html, "Delta Widgets") == "4", html
    assert _level_of(html, "Delta Widgets") != "3", (
        "pre-fix flatten regression: an L3 subsection must not render <h3>"
    )


def _uniform_l2_provenance() -> list[dict]:
    """The whole-book shape but with every sub-chapter heading at L2 — the
    'judge levelled every pending heading to the same level' case."""
    prov = _whole_book_provenance()
    for p in prov:
        # every genuine section heading (not the level-1 openers / book title)
        # sits at L2 — a uniform hierarchy below the chapter root.
        if p["region_kind"] == "heading" and (p["level"] or 0) >= 2:
            p["level"] = 2
    return prov


def test_uniform_judged_levels_are_flat_h3_byte_identical():
    """When the judge left every sub-chapter heading at ONE level (all L2),
    the render is byte-identical to the flat legacy shape: every section
    heading is <h3> and NO <h4>/<h5>/<h6> is minted."""
    prov = _uniform_l2_provenance()
    result = {
        "region_provenance": prov,
        "heading_tree": [
            [p["level"], p["heading_text"]]
            for p in prov
            if p["region_kind"] == "heading"
        ],
        "wcag_status": "passed",
        "exit_action": "ship_with_confidence",
        "theta_score": 0.9,
        "flags": [],
    }
    html = _render_book(result)
    section_levels = {
        lvl
        for lvl, txt in _rendered_headings(html)
        if not re.fullmatch(r"Chapter \d+", txt.strip())
        and txt.strip() != "SAMPLE WIDGET COMPENDIUM"
    }
    # every genuine section heading renders <h3>; nothing deeper appears.
    assert section_levels <= {"3"}, (section_levels, html)
    assert "<h4" not in html and "<h5" not in html and "<h6" not in html


def test_off_flag_flattens_heading_levels_to_h3(monkeypatch):
    """The reconcile ``=0`` revert lever restores the flat-<h3> legacy render:
    the L3 subsection flattens back to <h3> (no differentiation)."""
    monkeypatch.setenv("SEMANTIK_CHAPTER_LADDER_RECONCILE", "0")
    html = _render_book(_book_result())
    assert _level_of(html, "Delta Widgets") == "3", html
    assert "<h4" not in html
