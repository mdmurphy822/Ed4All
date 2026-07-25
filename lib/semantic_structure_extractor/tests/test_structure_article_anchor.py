"""Package 2c — article-boundary chapter anchoring on the outline-anchored path.

Regression net for three defects observed on a live whole-book conversion, all
inside ``_build_chapters_outline_anchored`` (reached only when BOTH
``ED4ALL_STRUCTURE_EXTRACT_GUARDS`` and ``ED4ALL_STRUCTURE_OUTLINE_ANCHOR`` are
on):

1. **Title-only chapter dropped.** The chapter spine was keyed purely on the
   declared ``N.M`` ordinal MAJORS, so a chapter that carries a title but no
   numbered section minted no chapter node at all — its whole body was demoted
   into the PREVIOUS chapter's last section (which is why those sections went
   grotesquely fat). Under ``ED4ALL_TO_CHAPTER_ANCHOR`` a missing chapter node
   means that topic gets no terminal objective.
2. **Phantom "Chapter 0".** A preface numbering its own sections ``0.M`` minted
   a ``ch0`` chapter, inflating ``chapter_count`` and the autoscaled week count.
3. **Bare ``Chapter N`` titles.** A chapter whose opener heading is the
   information-free label ``Chapter N`` now adopts the article's own unnumbered
   title heading when it has one.

Every fixture is SYNTHETIC (invented chapter/section titles with the shapes
observed in the corpus) — no course slug, publisher name, or verbatim book
title.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.semantic_structure_extractor import SemanticStructureExtractor  # noqa: E402


_GUARDS_ENV = "ED4ALL_STRUCTURE_EXTRACT_GUARDS"
_ANCHOR_ENV = "ED4ALL_STRUCTURE_OUTLINE_ANCHOR"


@pytest.fixture(autouse=True)
def _guards_on(monkeypatch):
    monkeypatch.setenv(_GUARDS_ENV, "1")
    monkeypatch.delenv(_ANCHOR_ENV, raising=False)


def _extract(html: str):
    return SemanticStructureExtractor().extract(html)


def _chapters(res):
    return res["chapters"]


def _by_id(res):
    return {c["id"]: c for c in res["chapters"]}


def _all_block_text(node) -> list:
    """Every content-block string under a chapter/section dict, depth-first."""
    out = [b.get("content", "") for b in node.get("contentBlocks", [])]
    for child in node.get("sections", []) or node.get("subsections", []):
        out.extend(_all_block_text(child))
    return out


def _chapter_text_blob(chapter) -> str:
    return " ".join(_all_block_text(chapter))


def _document_blocks(res) -> list:
    out: list = []
    for chapter in res["chapters"]:
        out.extend(_all_block_text(chapter))
    return out


def _article_anchor(res):
    return res["structureDiagnostics"]["guards"]["outline_anchor"]["article_anchor"]


# ---------------------------------------------------------------------------
# Mixed fixture: a front-matter article, two numbered-section chapters, and a
# TITLE-ONLY chapter (a real chapter that declares no ``N.M`` section) between
# them — the exact shape that used to lose a chapter.
# ---------------------------------------------------------------------------

_MIXED_HTML = """
<html lang="en"><body><main>
  <h1>Networked Systems</h1>
  <article role="doc-chapter" id="fm">
    <h2>Networked Systems</h2>
    <section><h3>0.1 Who this book is for</h3><p>FRONTMATTER-MARKER preface prose.</p></section>
  </article>
  <article role="doc-chapter" id="a1">
    <h2>Chapter 1</h2>
    <section><h3>Groundwork</h3><p>Opening prose for the first chapter.</p></section>
    <section><h3>1.1 Message Framing</h3><p>Body of one dot one.</p></section>
    <section><h3>1.2 Ordering Guarantees</h3><p>TAIL-OF-CH1-MARKER body of one dot two.</p></section>
  </article>
  <article role="doc-chapter" id="a2">
    <h2>Chapter 2</h2>
    <section><h3>Endpoint Lookup</h3><p>TITLE-ONLY-MARKER prose that belongs to chapter two.</p></section>
    <section><p>More chapter-two prose under no heading at all.</p></section>
  </article>
  <article role="doc-chapter" id="a3">
    <h2>Chapter 3</h2>
    <section><h3>3.1 Backpressure</h3><p>Body of three dot one.</p></section>
    <section><h3>3.2 Shedding</h3><p>Body of three dot two.</p></section>
  </article>
</main></body></html>
"""


# ---------------------------------------------------------------------------
# Defect 1 — a title-only chapter mints its own node and is NOT absorbed.
# ---------------------------------------------------------------------------


def test_title_only_chapter_mints_its_own_chapter_node():
    res = _extract(_MIXED_HTML)
    by_id = _by_id(res)
    assert "ch2" in by_id, [c["id"] for c in _chapters(res)]
    assert "TITLE-ONLY-MARKER" in _chapter_text_blob(by_id["ch2"])


def test_title_only_chapter_is_not_absorbed_into_previous_chapter():
    res = _extract(_MIXED_HTML)
    by_id = _by_id(res)
    ch1_blob = _chapter_text_blob(by_id["ch1"])
    # The previous chapter still owns its own tail...
    assert "TAIL-OF-CH1-MARKER" in ch1_blob
    # ...and none of the title-only chapter's body.
    assert "TITLE-ONLY-MARKER" not in ch1_blob


def test_title_only_chapter_keeps_its_headingless_body():
    # The trailing headingless <section> of the title-only article must stay in
    # ITS chapter (it used to bleed into the previous chapter's last section).
    res = _extract(_MIXED_HTML)
    assert "More chapter-two prose" in _chapter_text_blob(_by_id(res)["ch2"])


def test_article_anchor_diagnostics_report_the_residual_mint():
    aa = _article_anchor(_extract(_MIXED_HTML))
    assert aa["articles"] == 4
    assert aa["chapters_emitted"] == 3
    assert aa["residual_chapters"] == 1
    assert aa["front_matter_articles"] == 1


# ---------------------------------------------------------------------------
# Defect 2 — the front-matter article never mints a chapter.
# ---------------------------------------------------------------------------


def test_front_matter_article_mints_no_phantom_chapter():
    res = _extract(_MIXED_HTML)
    ids = [c["id"] for c in _chapters(res)]
    assert "ch0" not in ids
    titles = [c["headingText"] for c in _chapters(res)]
    assert "Chapter 0" not in titles
    assert _article_anchor(res)["front_matter_majors"] == [0]


def test_front_matter_prose_is_not_dropped():
    # Suppressing the phantom chapter must never lose content: the front-matter
    # blocks lead the first real chapter instead.
    res = _extract(_MIXED_HTML)
    assert "FRONTMATTER-MARKER" in _chapter_text_blob(_by_id(res)["ch1"])


def test_chapter_count_equals_content_article_count():
    res = _extract(_MIXED_HTML)
    # 4 doc-chapter articles, 1 of them front matter -> 3 content chapters.
    assert len(_chapters(res)) == 3
    assert [c["id"] for c in _chapters(res)] == ["ch1", "ch2", "ch3"]


def test_front_matter_only_document_still_yields_its_structure():
    # Fail-safe: when major 0 is ALL the document declares, it is kept rather
    # than emptying the book.
    html = (
        '<html lang="en"><body><main><h1>Notes</h1>'
        '<article role="doc-chapter" id="fm">'
        "<h2>Preamble</h2>"
        '<section><h3>0.1 Scope</h3><p>Scope prose.</p></section>'
        '<section><h3>0.2 Audience</h3><p>Audience prose.</p></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    assert len(_chapters(res)) == 1
    assert "Scope prose." in _chapter_text_blob(_chapters(res)[0])


# ---------------------------------------------------------------------------
# Regression — numbered-section chapters are unchanged.
# ---------------------------------------------------------------------------


_NUMBERED_ONLY_HTML = """
<html lang="en"><body><main>
  <h1>Numbered Only</h1>
  <article role="doc-chapter" id="n1">
    <h2>Chapter 1</h2>
    <section><h3>1.1 Alpha</h3><p>Alpha prose.</p></section>
    <section><h3>1.2 Beta</h3><p>Beta prose.</p></section>
  </article>
  <article role="doc-chapter" id="n2">
    <h2>Chapter 2</h2>
    <section><h3>2.1 Gamma</h3><p>Gamma prose.</p></section>
  </article>
</main></body></html>
"""


def test_numbered_section_chapters_group_by_declared_major():
    res = _extract(_NUMBERED_ONLY_HTML)
    assert [c["id"] for c in _chapters(res)] == ["ch1", "ch2"]
    headings = {
        c["id"]: [s["headingText"] for s in c["sections"]] for c in _chapters(res)
    }
    assert headings == {
        "ch1": ["1.1 Alpha", "1.2 Beta"],
        "ch2": ["2.1 Gamma"],
    }
    # No residual chapter, no front matter: the pure declared-spine path.
    aa = _article_anchor(res)
    assert aa["residual_chapters"] == 0 and aa["front_matter_articles"] == 0


def test_numbered_sections_keep_their_body_prose():
    res = _extract(_NUMBERED_ONLY_HTML)
    by_id = _by_id(res)
    sections = {s["headingText"]: s for s in by_id["ch1"]["sections"]}
    assert "Alpha prose." in " ".join(_all_block_text(sections["1.1 Alpha"]))
    assert "Beta prose." in " ".join(_all_block_text(sections["1.2 Beta"]))


def test_multi_article_same_major_still_collapses_to_one_chapter():
    # Two articles both carrying sections of major 1 must remain ONE chapter —
    # the declared spine still owns section identity across article boundaries.
    html = (
        '<html lang="en"><body><main><h1>Split</h1>'
        '<article role="doc-chapter" id="p1"><h2>Chapter 1</h2>'
        '<section><h3>1.1 Alpha</h3><p>Alpha prose.</p></section></article>'
        '<article role="doc-chapter" id="p2"><h2>Chapter 1</h2>'
        '<section><h3>1.2 Beta</h3><p>Beta prose.</p></section></article>'
        "</main></body></html>"
    )
    res = _extract(html)
    assert [c["id"] for c in _chapters(res)] == ["ch1"]
    assert len(_chapters(res)[0]["sections"]) == 2


def test_headless_continuation_article_still_merges():
    # Package-1 rule 1a preserved: a HEADLESS article is a continuation of the
    # open chapter (this is what stops an OCR-inflated per-page-wrapped scan
    # corpus from re-exploding into one chapter per page).
    html = (
        '<html lang="en"><body><main><h1>Continued</h1>'
        '<article role="doc-chapter" id="c1"><h2>Chapter 1</h2>'
        '<section><h3>1.1 Alpha</h3><p>Alpha prose.</p></section></article>'
        '<article role="doc-chapter" id="c2">'
        '<section><p>CONTINUATION-MARKER spillover page.</p></section></article>'
        "</main></body></html>"
    )
    res = _extract(html)
    assert len(_chapters(res)) == 1
    assert "CONTINUATION-MARKER" in _chapter_text_blob(_chapters(res)[0])
    assert _article_anchor(res)["residual_chapters"] == 0


# ---------------------------------------------------------------------------
# Nothing is ever dropped — block conservation across the regrouping.
# ---------------------------------------------------------------------------


def test_no_content_block_is_lost_by_the_article_anchoring(monkeypatch):
    after = sorted(_document_blocks(_extract(_MIXED_HTML)))
    monkeypatch.setenv(_ANCHOR_ENV, "0")
    before = sorted(_document_blocks(_extract(_MIXED_HTML)))
    # Same multiset of blocks, only regrouped.
    assert after == before


# ---------------------------------------------------------------------------
# Defect 3a — a BARE ``Chapter N`` opener adopts the article's own title.
# ---------------------------------------------------------------------------


def test_bare_chapter_label_adopts_the_article_title_heading():
    res = _extract(_MIXED_HTML)
    by_id = _by_id(res)
    assert by_id["ch1"]["headingText"] == "Groundwork"
    assert by_id["ch2"]["headingText"] == "Endpoint Lookup"
    assert _article_anchor(res)["titles_derived"] == 2


def test_bare_chapter_label_kept_when_the_article_has_no_title_heading():
    # Chapter 3 opens straight into its numbered spine — there is no title to
    # adopt, and one must never be invented from a numbered section.
    assert _by_id(_extract(_MIXED_HTML))["ch3"]["headingText"] == "Chapter 3"


def test_deep_ordinal_heading_is_never_adopted_as_a_chapter_title():
    # An article whose first heading is a SUB-section ordinal ("4.1.3 …")
    # still opens the numbered spine — adopting it would title the chapter
    # with a sub-sub-section.
    html = (
        '<html lang="en"><body><main><h1>Deep</h1>'
        '<article role="doc-chapter" id="d1"><h2>Chapter 1</h2>'
        '<section><h3>1.1 Alpha</h3><p>Alpha prose.</p></section></article>'
        '<article role="doc-chapter" id="d2"><h2>Chapter 2</h2>'
        '<section><h3>2.1.3 Deep Subtopic</h3><p>Deep prose.</p></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    by_id = _by_id(res)
    assert "ch2" in by_id
    assert by_id["ch2"]["headingText"] == "Chapter 2"
    assert "Deep prose." in _chapter_text_blob(by_id["ch2"])


def test_leading_page_mislabelled_with_a_later_chapters_ordinal_is_front_matter():
    # A copyright / ISBN front page whose running header the converter mis-read
    # as "Chapter 3" must NOT mint a chapter: it anchors nothing of its own and
    # duplicates an ordinal a later article legitimately anchors.
    html = (
        '<html lang="en"><body><main><h1>Some Book</h1>'
        '<article role="doc-chapter" id="f1"><h2>Chapter 3</h2>'
        '<section><h4>Copyright c 1999 by A Publisher</h4>'
        "<p>ISBN-MARKER printing history.</p></section></article>"
        '<article role="doc-chapter" id="f2"><h2>Chapter 1</h2>'
        '<section><h3>1.1 Alpha</h3><p>Alpha prose.</p></section></article>'
        '<article role="doc-chapter" id="f3"><h2>Chapter 3</h2>'
        '<section><h3>3.1 Gamma</h3><p>Gamma prose.</p></section></article>'
        "</main></body></html>"
    )
    res = _extract(html)
    assert [c["id"] for c in _chapters(res)] == ["ch1", "ch3"]
    assert _article_anchor(res)["front_matter_articles"] == 1
    # Still never dropped.
    assert "ISBN-MARKER" in _chapter_text_blob(_by_id(res)["ch1"])


def test_real_chapter_title_is_never_overwritten():
    # A single declared major takes its title from the file h1; the adoption
    # pass must not clobber it.
    html = (
        '<html lang="en"><body><main><h1>Foundations</h1>'
        '<article role="doc-chapter" id="s1"><h2>Chapter 1</h2>'
        '<section><h3>Overview</h3><p>Overview prose.</p></section>'
        '<section><h3>1.1 Alpha</h3><p>Alpha prose.</p></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    assert _chapters(res)[0]["headingText"] == "Foundations"


def test_adopted_title_still_captures_the_whole_chapter_prose():
    # The adoption stamps an id on the chapter's own opener heading so
    # ``chapter_text`` keeps the CHAPTER scope rather than collapsing onto the
    # (much narrower) adopted heading's own scope.
    by_id = _by_id(_extract(_MIXED_HTML))
    text = by_id["ch1"].get("chapter_text", "")
    assert "Body of one dot one." in text
    assert "TAIL-OF-CH1-MARKER" in text


# ---------------------------------------------------------------------------
# Flag semantics — the opt-out levers still revert to the prior paths.
# ---------------------------------------------------------------------------


def test_anchor_off_reverts_to_the_package1_article_path(monkeypatch):
    monkeypatch.setenv(_ANCHOR_ENV, "0")
    res = _extract(_MIXED_HTML)
    guards = res["structureDiagnostics"]["guards"]
    assert "outline_anchor" not in guards
    # Package-1 keeps every article as its own chapter (front matter included).
    assert len(_chapters(res)) == 4


def test_guards_off_never_reaches_the_article_anchor(monkeypatch):
    monkeypatch.delenv(_GUARDS_ENV, raising=False)
    monkeypatch.setenv(_ANCHOR_ENV, "1")
    res = _extract(_MIXED_HTML)
    assert "guards" not in res.get("structureDiagnostics", {})


def test_extraction_is_deterministic():
    def _payload(res):
        out = copy.deepcopy(res)
        out.pop("documentInfo", None)
        return json.dumps(out, sort_keys=True)

    assert _payload(_extract(_MIXED_HTML)) == _payload(_extract(_MIXED_HTML))
