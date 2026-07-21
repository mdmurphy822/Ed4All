"""Package 2 — SemantiK outline-anchored section alignment.

Exercises ``ED4ALL_STRUCTURE_OUTLINE_ANCHOR`` (default ON when the Package-1
guards are on) on SYNTHETIC ``<article role="doc-chapter">`` HTML — no
course-data path, no model. The satellite refines the Package-1 guarded
assembly onto the document's OWN declared ``N.M Title`` spine. Covers:

- outline harvest from clean heading list AND from a fused outline paragraph;
- a heading-bearing section survives iff it matches a declared entry (ordinal
  prefix stripped, OCR-fuzzed title tolerated via difflib);
- an Example label / non-declared heading demotes to content (never a section);
- a body opener is preferred over an answer-key reprint, but the reprint IS
  used (with ``matchedZone`` provenance) when it is the only occurrence;
- chapters regroup by ordinal major (single- and multi-chapter files);
- a file that declares no outline falls through byte-identically to the
  Package-1 guarded behavior;
- ``structureDiagnostics.guards.outline_anchor`` + the
  STRUCTURE_DECLARED_SECTION_MISSING warning;
- flag semantics (guards on + anchor off == Package-1 behavior).
"""

from __future__ import annotations

import copy
import json
import logging
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
    """Package 2 only runs under the Package-1 guards; default-ON anchor."""
    monkeypatch.setenv(_GUARDS_ENV, "1")
    monkeypatch.delenv(_ANCHOR_ENV, raising=False)


def _extract(html: str):
    return SemanticStructureExtractor().extract(html)


def _sections(res):
    return [s for c in res["chapters"] for s in c["sections"]]


# ---------------------------------------------------------------------------
# Primary fixture: outline article (clean headings + fused paragraph), a body
# zone (with an OCR-fuzzed opener + an Example label), and a trailing review /
# answer-key zone (reprints).
# ---------------------------------------------------------------------------

_ANCHOR_HTML = """
<html lang="en"><body><main>
  <h1>Foundations</h1>
  <article role="doc-chapter" id="chap-1">
    <section data-dart-block-id="chapter-outline"><h3>Chapter Outline</h3></section>
    <section aria-labelledby="o11"><h3>1.1 Whole Numbers</h3></section>
    <section data-semantik-demoted-role="list"><p>1.2 Use the Language of Algebra 1.3 Add and Subtract Integers</p></section>
    <section aria-labelledby="o14"><h3>1.4 Multiply and Divide</h3></section>
  </article>
  <article role="doc-chapter" id="chap-2">
    <header><h2>Introduction</h2></header>
    <section><h3 id="b11">1.1 Whole Numhers</h3><p>Real body opener prose for 1.1.</p></section>
    <section><h3>EXAMPLE 1.5</h3><p>Worked example, not a section.</p></section>
    <section><h3 id="b14">1.4 Multiply and Divide</h3><p>Real body opener for 1.4.</p></section>
    <section><h3>1.3 Exercises</h3><p>Drill banner for 1.3 — admits the ordinal, donates no title.</p></section>
  </article>
  <article role="doc-chapter" id="chap-3">
    <header><h2>Chapter 1 Review</h2></header>
    <section><h3>1.1 Whole Numbers</h3><p>Answer-key reprint of 1.1.</p></section>
    <section><h3>1.2 Use the Language of Algebra</h3><p>Reprint of the 1.2 opener.</p></section>
  </article>
</main></body></html>
"""


def _by_ordinal(res):
    return {s["headingText"].split()[0]: s for s in _sections(res)}


# ---------------------------------------------------------------------------
# Outline harvest — clean headings + fused paragraph.
# ---------------------------------------------------------------------------


def test_outline_harvest_recovers_clean_and_fused_entries():
    res = _extract(_ANCHOR_HTML)
    oa = res["structureDiagnostics"]["guards"]["outline_anchor"]
    # 1.1 + 1.4 admitted by their body headings; 1.2 by its answer-key heading;
    # 1.3 by its apparatus banner ("1.3 Exercises"). The fused outline paragraph
    # DONATES the 1.2 / 1.3 titles but never admits an ordinal on its own.
    assert oa["declared_entries"] == 4
    ords = sorted(_by_ordinal(res))
    assert ords == ["1.1", "1.2", "1.3", "1.4"]


def test_declared_section_estimate_improved_by_fused_recovery():
    # Package-1's cheap estimate saw 0 (the chapter-outline block carries no
    # ordinals and there's no nav.toc); the anchor recovers the fused entries
    # and lifts the estimate to the true 4.
    diag = _extract(_ANCHOR_HTML)["structureDiagnostics"]["guards"]
    assert diag["declared_section_estimate"] == 4


def test_nav_toc_fallback_when_no_outline_article():
    # No chapter-outline block -> the nav is the only outline-zone title source.
    # Package 2b: the nav fused text DONATES titles but does not ADMIT; both
    # ordinals are admitted by their (structural) body headings.
    html = (
        '<html lang="en"><body><main><h1>Graphs</h1>'
        '<nav class="toc"><p>4.2 Slope 4.3 Intercepts</p></nav>'
        '<article role="doc-chapter" id="c1"><header><h2>Introduction</h2></header>'
        '<section><h3 id="h42">4.2 Slope</h3><p>Body of 4.2.</p></section>'
        '<section><h3 id="h43">4.3 Intercepts</h3><p>Body of 4.3.</p></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    ords = sorted(_by_ordinal(res))
    assert ords == ["4.2", "4.3"]


# ---------------------------------------------------------------------------
# Section survival + fuzzy / ordinal-prefix matching.
# ---------------------------------------------------------------------------


def test_section_survives_on_ocr_fuzzed_body_match():
    # Declared "1.1 Whole Numbers"; body heading carries the ordinal prefix
    # and an OCR typo ("Numhers"). It still matches (difflib >= 0.8) and the
    # BODY occurrence wins over the outline stub + review reprint.
    res = _extract(_ANCHOR_HTML)
    sec = _by_ordinal(res)["1.1"]
    assert sec["matchedZone"] == "body"
    assert sec["headingId"] == "b11"


def test_example_label_demotes_and_is_never_a_section():
    res = _extract(_ANCHOR_HTML)
    titles = [s["headingText"] for s in _sections(res)]
    assert not any("EXAMPLE" in t.upper() for t in titles)
    # Its content is preserved (demoted), not dropped.
    assert "Worked example" in json.dumps(res["chapters"])
    assert res["structureDiagnostics"]["guards"]["outline_anchor"][
        "demoted_headings"
    ] >= 1


# ---------------------------------------------------------------------------
# Zone preference — body over answer-key; reprint used only as last resort.
# ---------------------------------------------------------------------------


def test_body_opener_preferred_over_answer_key_reprint():
    # 1.1 has outline + body + answer-key occurrences -> body chosen.
    res = _extract(_ANCHOR_HTML)
    assert _by_ordinal(res)["1.1"]["matchedZone"] == "body"


def test_answer_key_reprint_used_when_only_occurrence():
    # 1.2 exists only as a heading in the Chapter Review zone -> the reprint IS
    # used (better than losing the section) and stamped answer_key.
    res = _extract(_ANCHOR_HTML)
    assert _by_ordinal(res)["1.2"]["matchedZone"] == "answer_key"


def test_declared_only_entry_survives_as_stub():
    # 1.3 is admitted by its apparatus banner but has no TITLE-matching heading
    # occurrence anywhere (the banner title "Exercises" never matches the fused
    # "Add and Subtract Integers") -> still emitted as a section, provenance
    # declared_only, headingId None.
    res = _extract(_ANCHOR_HTML)
    sec = _by_ordinal(res)["1.3"]
    assert sec["matchedZone"] == "declared_only"
    assert sec["headingId"] is None


def test_outline_only_entry_stamped_outline():
    # 1.4 has an outline heading + a body opener -> body wins. Craft an
    # outline-only entry to pin the ``outline`` provenance.
    html = (
        '<html lang="en"><body><main><h1>Foundations</h1>'
        '<article role="doc-chapter" id="chap-1">'
        '<section data-dart-block-id="chapter-outline"><h3>Chapter Outline</h3></section>'
        '<section><h3>1.5 Visualize Fractions</h3></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    assert _by_ordinal(res)["1.5"]["matchedZone"] == "outline"


# ---------------------------------------------------------------------------
# Chapter regrouping by ordinal major.
# ---------------------------------------------------------------------------


def test_single_major_regroups_into_one_chapter_titled_by_h1():
    res = _extract(_ANCHOR_HTML)
    assert len(res["chapters"]) == 1
    ch = res["chapters"][0]
    assert ch["headingText"] == "Foundations"  # from the file h1
    assert len(ch["sections"]) == 4


def test_multi_major_regroups_into_multiple_chapters():
    html = (
        '<html lang="en"><body><main>'
        '<article role="doc-chapter" id="chap-1">'
        '<section data-dart-block-id="chapter-outline"><h3>Chapter Outline</h3></section>'
        '<section><h3>1.1 Whole Numbers</h3></section>'
        '<section><h3>1.2 Integers</h3></section>'
        '<section><h3>2.1 Solve Equations</h3></section>'
        "</article></main></body></html>"
    )
    res = _extract(html)
    assert len(res["chapters"]) == 2
    titles = [c["headingText"] for c in res["chapters"]]
    # Multi-major -> synthesized per-major titles (no single h1 to claim).
    assert titles == ["Chapter 1", "Chapter 2"]
    counts = {c["headingText"]: len(c["sections"]) for c in res["chapters"]}
    assert counts == {"Chapter 1": 2, "Chapter 2": 1}


# ---------------------------------------------------------------------------
# Diagnostics + warning.
# ---------------------------------------------------------------------------


def test_unmatched_declared_diagnostics_lists_no_body_entries():
    res = _extract(_ANCHOR_HTML)
    oa = res["structureDiagnostics"]["guards"]["outline_anchor"]
    assert oa["declared_entries"] == 4
    # 1.1/1.2/1.4 matched a heading occurrence; 1.3 is fused-only (declared_only).
    assert oa["matched_sections"] == 3
    unmatched = {u["ordinal"] for u in oa["unmatched_declared"]}
    # 1.1 + 1.4 have body openers; 1.2 (answer_key) + 1.3 (declared_only) do not.
    assert unmatched == {"1.2", "1.3"}
    zones = {u["ordinal"]: u["found_zone"] for u in oa["unmatched_declared"]}
    assert zones == {"1.2": "answer_key", "1.3": "declared_only"}


def test_declared_section_missing_warning_fires(caplog):
    with caplog.at_level(logging.WARNING):
        _extract(_ANCHOR_HTML)
    assert "STRUCTURE_DECLARED_SECTION_MISSING" in caplog.text
    assert "1.2" in caplog.text and "1.3" in caplog.text


def test_no_warning_when_every_section_has_body_opener(caplog):
    html = (
        '<html lang="en"><body><main><h1>Foundations</h1>'
        '<article role="doc-chapter" id="chap-1">'
        '<section data-dart-block-id="chapter-outline"><h3>Chapter Outline</h3></section>'
        '<section><h3>1.1 Whole Numbers</h3></section>'
        "</article>"
        '<article role="doc-chapter" id="chap-2"><header><h2>Introduction</h2></header>'
        '<section><h3>1.1 Whole Numbers</h3><p>Body opener.</p></section>'
        "</article></main></body></html>"
    )
    with caplog.at_level(logging.WARNING):
        res = _extract(html)
    assert _by_ordinal(res)["1.1"]["matchedZone"] == "body"
    assert "STRUCTURE_DECLARED_SECTION_MISSING" not in caplog.text


# ---------------------------------------------------------------------------
# Wide-net fall-through — undeclared corpora unchanged vs Package-1.
# ---------------------------------------------------------------------------


# Two articles, the second headless — NO chapter-outline block, NO nav.toc.
# Nothing to anchor on, so the anchor must fall through to Package-1 (which
# merges the headless continuation into the previous chapter).
_UNDECLARED_HTML = """
<html lang="en"><body><main>
  <article role="doc-chapter" id="chap-1">
    <header><h2>Foundations</h2></header>
    <section class="semantik-section"><h3>Intro Topic</h3><p>Prose.</p></section>
  </article>
  <article role="doc-chapter" id="chap-2">
    <section class="semantik-section"><p>Continuation body block.</p></section>
  </article>
</main></body></html>
"""


def _stable_payload(res):
    payload = copy.deepcopy(res)
    payload.pop("documentInfo", None)
    return json.dumps(payload, sort_keys=True)


def test_undeclared_file_falls_through_to_package1_byte_identical(monkeypatch):
    # Anchor ON (default) vs anchor OFF (Package-1) must be byte-identical when
    # the book declares no outline — the anchor returned None and never ran.
    monkeypatch.delenv(_ANCHOR_ENV, raising=False)
    anchor_on = _stable_payload(_extract(_UNDECLARED_HTML))
    monkeypatch.setenv(_ANCHOR_ENV, "0")
    anchor_off = _stable_payload(_extract(_UNDECLARED_HTML))
    assert anchor_on == anchor_off
    # And no outline_anchor diagnostics were attached.
    res = _extract(_UNDECLARED_HTML)  # anchor still off
    assert "outline_anchor" not in res["structureDiagnostics"]["guards"]


# ---------------------------------------------------------------------------
# Flag semantics — guards on + anchor off == Package-1 behavior.
# ---------------------------------------------------------------------------


def test_anchor_off_is_package1_not_outline_anchored(monkeypatch):
    monkeypatch.setenv(_ANCHOR_ENV, "0")
    res = _extract(_ANCHOR_HTML)
    # Package-1 keeps the article boundaries (chap-2 "Introduction" + the
    # review article are their own chapters) -> more than one chapter, and no
    # outline_anchor diagnostics.
    assert len(res["chapters"]) > 1
    assert "outline_anchor" not in res["structureDiagnostics"]["guards"]


@pytest.mark.parametrize("val", ["0", "false", "off", "no"])
def test_anchor_falsey_values_disable(monkeypatch, val):
    monkeypatch.setenv(_ANCHOR_ENV, val)
    res = _extract(_ANCHOR_HTML)
    assert "outline_anchor" not in res["structureDiagnostics"]["guards"]


def test_guards_off_never_reaches_anchor(monkeypatch):
    # Guards master gate off -> legacy path, anchor unreachable even if its env
    # is truthy.
    monkeypatch.delenv(_GUARDS_ENV, raising=False)
    monkeypatch.setenv(_ANCHOR_ENV, "1")
    res = _extract(_ANCHOR_HTML)
    assert "guards" not in res.get("structureDiagnostics", {})


# ===========================================================================
# Package 2b — multi-source ordinal-UNION harvest.
#
# The load-bearing change: an ordinal is ADMITTED as a declared section iff it
# has STRUCTURAL evidence — it opens a heading element (source d) or forms an
# apparatus banner (source e). Ordinals seen ONLY in fused raw text (outline
# paragraphs, nav, body paragraphs) are TITLE DONORS, never admitters. Title
# backfill is priority-wins: body heading > answer-key heading > outline/nav
# fused split > body-paragraph fused split. All fixtures are SYNTHETIC (invented
# section titles with the same shapes as the observed corpus) — no course slug,
# publisher name, or verbatim book title.
# ===========================================================================


def _oa(res):
    return res["structureDiagnostics"]["guards"]["outline_anchor"]


# (i) ch07-shape — the stamped outline article holds ONLY a figure-caption
# paragraph (a raw-text donor) + an empty outline heading; the real numbered
# openers live in the body article. Under 2b the caption never admits an
# ordinal, so every section is harvested from the body with a clean title (no
# caption text leaks in as a title).
_CH07_SHAPE_HTML = """
<html lang="en"><body><main><h1>Factoring</h1>
  <article role="doc-chapter" id="outline">
    <section data-dart-block-id="chapter-outline"><h3>Chapter Outline</h3></section>
    <section data-semantik-demoted-role="caption"><p>Figure 7.1 A tiling diagram illustrating factoring by area.</p></section>
    <section aria-labelledby="empty"><h3></h3></section>
  </article>
  <article role="doc-chapter" id="body">
    <header><h2>Introduction</h2></header>
    <section><h3>7.1 Common Factors and Grouping</h3><p>Body opener 7.1.</p></section>
    <section><h3>7.2 Trinomials of a Simple Form</h3><p>Body opener 7.2.</p></section>
    <section><h3>7.3 Trinomials of a General Form</h3><p>Body opener 7.3.</p></section>
    <section><h3>7.4 Special Product Patterns</h3><p>Body opener 7.4.</p></section>
    <section><h3>7.5 A Factoring Strategy</h3><p>Body opener 7.5.</p></section>
    <section><h3>7.6 Solving by Factoring</h3><p>Body opener 7.6.</p></section>
  </article>
</main></body></html>
"""


def test_ch07_shape_body_harvest_ignores_figure_caption():
    res = _extract(_CH07_SHAPE_HTML)
    sections = _sections(res)
    ords = sorted(_by_ordinal(res))
    assert ords == ["7.1", "7.2", "7.3", "7.4", "7.5", "7.6"]
    # Every entry anchored to a real body opener, none to the caption stub.
    assert all(s["matchedZone"] == "body" for s in sections)
    # The figure-caption text never became a title.
    titles = " ".join(s["headingText"] for s in sections).lower()
    assert "figure" not in titles and "tiling" not in titles
    assert _oa(res)["title_sources"]["7.1"] == "body_heading"


# (ii) ch04-shape — NO stamped outline article; a junky synthesized nav (partial
# ordinals, OCR-mangled titles) + a body paragraph that fused the printed
# outline. The union admits every ordinal (body headings + apparatus banners)
# and backfills UNPOLLUTED titles: a clean body heading beats the junky nav,
# and ordinals absent from the nav recover their title from the fused body
# paragraph.
_CH04_SHAPE_HTML = """
<html lang="en"><body><main><h1>Graphing Lines</h1>
  <nav class="toc"><p>4.1 Coordnate Plne 4.4 Slpe of a Lne</p></nav>
  <article role="doc-chapter" id="body">
    <header><h2>Introduction</h2></header>
    <section><h3>4.1 The Coordinate Plane</h3><p>Body opener 4.1.</p></section>
    <section data-semantik-demoted-role="list"><p>4.2 Graph Linear Equations 4.3 Graph with Intercepts</p></section>
    <section><h3>4.2 Exercises</h3><p>Drill for 4.2.</p></section>
    <section><h3>4.3 Exercises</h3><p>Drill for 4.3.</p></section>
    <section><h3>4.4 Slope of a Line</h3><p>Body opener 4.4.</p></section>
  </article>
</main></body></html>
"""


def test_ch04_shape_union_recovers_all_ordinals_unpolluted():
    res = _extract(_CH04_SHAPE_HTML)
    by = _by_ordinal(res)
    assert sorted(by) == ["4.1", "4.2", "4.3", "4.4"]
    titles = {o: s["headingText"] for o, s in by.items()}
    assert titles == {
        "4.1": "4.1 The Coordinate Plane",       # clean body heading beats nav junk
        "4.2": "4.2 Graph Linear Equations",     # fused body paragraph (nav absent)
        "4.3": "4.3 Graph with Intercepts",      # fused body paragraph (nav absent)
        "4.4": "4.4 Slope of a Line",            # clean body heading beats nav junk
    }
    # The OCR-mangled nav titles never leaked into a section title.
    joined = " ".join(titles.values()).lower()
    assert "coordnate" not in joined and "slpe" not in joined
    ts = _oa(res)["title_sources"]
    assert ts["4.1"] == "body_heading" and ts["4.4"] == "body_heading"
    assert ts["4.2"] == "body_fused" and ts["4.3"] == "body_fused"


# (iii) small-minor Try-It / Example / Figure ordinals appearing ONLY in fused
# raw text are title donors, NEVER admitters — the load-bearing guard against
# the observed leak class.
_FUSED_ONLY_LEAK_HTML = """
<html lang="en"><body><main><h1>Systems</h1>
  <article role="doc-chapter" id="body">
    <header><h2>Introduction</h2></header>
    <section><h3>5.1 Solve by Graphing</h3><p>Body opener 5.1.</p></section>
    <section><h3>5.2 Solve by Substitution</h3><p>Body opener 5.2.</p></section>
    <section data-semantik-demoted-role="list"><p>5.11 Try It Warmup 5.20 Example Set 5.5 Figure Caption</p></section>
  </article>
</main></body></html>
"""


def test_fused_only_small_minor_ordinals_not_admitted():
    res = _extract(_FUSED_ONLY_LEAK_HTML)
    ords = sorted(_by_ordinal(res))
    # Only the two ordinals that open a real body heading survive.
    assert ords == ["5.1", "5.2"]
    # The fused Try-It / Example / Figure ordinals (minor <= 30, so NOT killed
    # by the minor ceiling) are excluded purely by the structural-admission rule.
    entries = set(_oa(res)["title_sources"])
    assert "5.11" not in entries and "5.20" not in entries and "5.5" not in entries


# (iv) anchor-off byte-stability — with the anchor flag OFF the 2b sources are
# never consulted; the guards-only path is byte-identical regardless of the
# document's numbered headings / banners.
def test_anchor_off_byte_identical_on_2b_rich_fixture(monkeypatch):
    monkeypatch.setenv(_ANCHOR_ENV, "0")
    first = _stable_payload(_extract(_CH04_SHAPE_HTML))
    second = _stable_payload(_extract(_CH04_SHAPE_HTML))
    assert first == second
    res = _extract(_CH04_SHAPE_HTML)  # still off
    assert "outline_anchor" not in res["structureDiagnostics"]["guards"]


# (v) wide-net fall-through — a corpus with NO structural N.M evidence (a bare
# decimal in prose is not a section) harvests nothing and falls through to the
# guards-only path byte-identically.
_NO_ORDINAL_HTML = """
<html lang="en"><body><main><h1>Measurement</h1>
  <article role="doc-chapter" id="c1">
    <header><h2>Overview</h2></header>
    <section><h3>Rounding Decimals</h3><p>The value of pi is about 3.14 in most work.</p></section>
  </article>
</main></body></html>
"""


def test_wide_net_fall_through_when_no_structural_ordinals(monkeypatch):
    res_on = _extract(_NO_ORDINAL_HTML)  # anchor default-on
    assert "outline_anchor" not in res_on["structureDiagnostics"]["guards"]
    on_payload = _stable_payload(res_on)
    monkeypatch.setenv(_ANCHOR_ENV, "0")
    off_payload = _stable_payload(_extract(_NO_ORDINAL_HTML))
    assert on_payload == off_payload


# (vi) title_source recorded per entry in the diagnostics.
def test_title_source_recorded_in_diagnostics():
    ts = _oa(_extract(_ANCHOR_HTML))["title_sources"]
    assert set(ts) == {"1.1", "1.2", "1.3", "1.4"}
    assert ts["1.1"] == "body_heading"        # body opener wins (OCR typo and all)
    assert ts["1.2"] == "answer_key_heading"  # only heading is the review reprint
    assert ts["1.3"] == "outline_fused"       # banner-admitted, title from the fused list


# (vii) an apparatus-tail heading ("N.M Exercises") admits the ordinal but
# donates NO title — with no other donor the section is emitted title-less
# (heading text is the bare ordinal) with title_source == apparatus_only.
_APPARATUS_TAIL_HTML = """
<html lang="en"><body><main><h1>Polynomials</h1>
  <article role="doc-chapter" id="body">
    <header><h2>Introduction</h2></header>
    <section><h3>6.1 Add and Subtract</h3><p>Body opener 6.1.</p></section>
    <section><h3>6.2 Exercises</h3><p>Drill for 6.2 — apparatus tail, no title donor.</p></section>
  </article>
</main></body></html>
"""


def test_apparatus_tail_heading_admits_ordinal_but_donates_no_title():
    res = _extract(_APPARATUS_TAIL_HTML)
    by = _by_ordinal(res)
    assert sorted(by) == ["6.1", "6.2"]
    ts = _oa(res)["title_sources"]
    assert ts["6.1"] == "body_heading"
    # 6.2 admitted by its apparatus banner; no donor -> title-less, bare ordinal.
    assert ts["6.2"] == "apparatus_only"
    assert by["6.2"]["headingText"] == "6.2"
    assert "exercise" not in by["6.2"]["headingText"].lower()
