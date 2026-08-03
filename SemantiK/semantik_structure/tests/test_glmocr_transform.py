"""Unit tests for the GLM-OCR deterministic transform + region classification.

Synthetic fixtures only — a tiny FAKE PP-DocLayoutV3 layout JSON with invented
placeholder text (NO real course content), NO GPU, NO seat.
"""

from __future__ import annotations

import pytest

from semantik_structure.glmocr import region_map as rm
from semantik_structure.glmocr import math_normalize as mn
from semantik_structure.glmocr.transform import GlmPage, transform_document


def _region(index, native_label, content, bbox=None):
    return {
        "index": index,
        "native_label": native_label,
        "content": content,
        "bbox_2d": bbox or [100, 100 + index * 50, 900, 140 + index * 50],
    }


def _kinds(prov):
    return [p["region_kind"] for p in prov]


# ── region_map classification ───────────────────────────────────────────────


def test_classify_native_label_buckets():
    assert rm.classify_native_label("doc_title") == "heading"
    assert rm.classify_native_label("paragraph_title") == "heading"
    assert rm.classify_native_label("display_formula") == "math"
    assert rm.classify_native_label("inline_formula") == "math"
    assert rm.classify_native_label("table") == "table"
    assert rm.classify_native_label("image") == "figure"
    assert rm.classify_native_label("chart") == "figure"
    assert rm.classify_native_label("figure_title") == "caption"
    assert rm.classify_native_label("footnote") == "footnote"
    assert rm.classify_native_label("aside_text") == "aside"
    assert rm.classify_native_label("reference") == "aside"
    assert rm.classify_native_label("text") == "paragraph"
    for furniture in ("header", "footer", "number", "seal", "formula_number"):
        assert rm.classify_native_label(furniture) == "metadata_drop"


def test_apparatus_pedagogy_class_data_driven():
    assert rm.apparatus_pedagogy_class("EXAMPLE 3") == "pedagogy-example"
    assert rm.apparatus_pedagogy_class("Try It 1.42") == "pedagogy-try-it"
    assert rm.apparatus_pedagogy_class("Solution") == "pedagogy-solution"
    assert rm.apparatus_pedagogy_class("How To multiply") == "pedagogy-how-to"
    # word-boundary guard: "stepping" is not "Step"
    assert rm.apparatus_pedagogy_class("Stepping stones of algebra") is None
    assert rm.apparatus_pedagogy_class("A perfectly ordinary sentence.") is None


def test_heading_level_for_title_numbered_vs_ambiguous():
    lvl, ambiguous = rm.heading_level_for_title("1.4 Multiply Whole Numbers")
    assert lvl == 2 and ambiguous is False
    lvl, ambiguous = rm.heading_level_for_title("Introduction")
    assert lvl == 3 and ambiguous is True


# ── furniture / headings ─────────────────────────────────────────────────────


def test_furniture_dropped_and_headings_leveled():
    page = GlmPage(page_no=1, regions=[
        _region(0, "header", "Chapter 1 Foundations 55"),
        _region(1, "doc_title", "# Foundations"),
        _region(2, "paragraph_title", "1.1 Introduce Widgets"),
        _region(3, "paragraph_title", "Overview"),
        _region(4, "text", "Body prose about widgets that is clearly a sentence."),
    ])
    tr = transform_document([page])
    prov = tr.region_provenance
    assert prov[0]["region_kind"] == "metadata_drop"  # header kept as drop
    assert prov[1]["region_kind"] == "heading" and prov[1]["level"] == 1
    assert prov[2]["region_kind"] == "heading" and prov[2]["level"] == 2
    assert prov[3]["region_kind"] == "heading" and prov[3]["level"] == 3
    assert prov[3]["heading_level_pending"] is True
    # the ambiguous title emits a heading_level_pending escalation
    assert any(e["reason"] == "heading_level_pending" for e in tr.escalations)
    # heading_tree captured
    assert (1, "Foundations") in tr.heading_tree


def test_apparatus_title_demoted_from_heading_track():
    page = GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", "EXAMPLE 2"),
        _region(1, "text", "The worked-out solution body sentence goes here."),
    ])
    tr = transform_document([page])
    ex = tr.region_provenance[0]
    assert ex["region_kind"] == "paragraph"
    assert ex["pedagogy_class"] == "pedagogy-example"
    assert ex["role"] == "apparatus"


def test_box_title_promotion():
    # ≥3-word box title still promotes to a heading (the D3.3 ≤2-word demotion
    # constraint does not apply here).
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "Widget Assembly Overview"),
        _region(1, "text", "A body paragraph with at least four words present."),
    ])
    tr = transform_document([page])
    assert tr.region_provenance[0]["region_kind"] == "heading"
    assert tr.region_provenance[0]["heading_text"] == "Widget Assembly Overview"


# ── tables ───────────────────────────────────────────────────────────────────


def test_table_md_pipe_to_cell_grid():
    md = "| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |"
    page = GlmPage(page_no=1, regions=[_region(0, "table", md)])
    prov = transform_document([page]).region_provenance
    assert prov[0]["region_kind"] == "table"
    assert prov[0]["data_table_mode"] == "md-pipe"
    assert prov[0]["cell_grid"] == [["A", "B"], ["1", "2"], ["3", "4"]]


def test_table_html_to_cell_grid():
    html = "<table><tr><td>X</td><td>Y</td></tr><tr><td>1</td><td>2</td></tr></table>"
    page = GlmPage(page_no=1, regions=[_region(0, "table", html)])
    prov = transform_document([page]).region_provenance
    assert prov[0]["data_table_mode"] == "html"
    assert prov[0]["cell_grid"] == [["X", "Y"], ["1", "2"]]


# ── math ─────────────────────────────────────────────────────────────────────


def test_math_wraps_bare_latex_for_mathml():
    page = GlmPage(page_no=1, regions=[
        _region(0, "display_formula", "\\frac{1}{2}"),
        _region(1, "inline_formula", "x^2"),
    ])
    prov = transform_document([page]).region_provenance
    assert prov[0]["region_kind"] == "math"
    assert prov[0]["raw_text"].startswith("$$") and prov[0]["raw_text"].endswith("$$")
    assert prov[1]["raw_text"] == "$x^2$"


def test_math_keeps_already_delimited():
    page = GlmPage(page_no=1, regions=[_region(0, "display_formula", "$$a+b$$")])
    prov = transform_document([page]).region_provenance
    assert prov[0]["raw_text"] == "$$a+b$$"


# ── figures + captions ───────────────────────────────────────────────────────


def test_figure_with_extracted_caption():
    page = GlmPage(page_no=1, regions=[
        _region(0, "image", ""),
        _region(1, "figure_title", "Figure 1: a widget diagram"),
    ])
    prov = transform_document([page]).region_provenance
    assert len(prov) == 1  # caption folded into the figure
    assert prov[0]["region_kind"] == "figure"
    assert prov[0]["caption_text"] == "Figure 1: a widget diagram"
    assert prov[0]["caption_source"] == "extracted"
    assert "bbox" in prov[0]


def test_caption_less_figure_escalated():
    page = GlmPage(page_no=1, regions=[_region(0, "image", "")])
    tr = transform_document([page])
    assert any(e["reason"] == "caption_less_figure" for e in tr.escalations)


# ── exercise pairing / ordinals ──────────────────────────────────────────────


def test_exercise_ordinal_paired_with_formula():
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "12.", bbox=[100, 100, 140, 130]),
        _region(1, "display_formula", "3+4", bbox=[160, 100, 400, 130]),
    ])
    tr = transform_document([page])
    prov = tr.region_provenance
    assert len(prov) == 1  # formula consumed into the exercise unit
    assert prov[0]["role"] == "exercise"
    assert prov[0]["raw_text"].startswith("12. $$")


def test_unpaired_ordinal_escalated():
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "12.", bbox=[100, 100, 140, 130]),
        _region(1, "text", "Some paired-less filler prose here.", bbox=[100, 200, 900, 240]),
    ])
    tr = transform_document([page])
    assert any(e["reason"] == "unpaired_ordinal" for e in tr.escalations)


def test_prose_ordinal_preserved_as_exercise():
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "12.", bbox=[100, 100, 140, 130]),
        _region(1, "display_formula", "1", bbox=[160, 100, 300, 130]),
        _region(2, "text", "413. Simplify the expression fully.",
                 bbox=[100, 300, 900, 340]),
    ])
    prov = transform_document([page]).region_provenance
    prose = [p for p in prov if p.get("role") == "exercise"
             and p["raw_text"].startswith("413.")]
    assert prose and prose[0]["raw_text"].startswith("413.")


# ── aside_text / reference kept ─────────────────────────────────────────────


def test_aside_and_reference_kept():
    page = GlmPage(page_no=1, regions=[
        _region(0, "aside_text", "A margin aside note."),
        _region(1, "reference", "Author, Title, Year."),
    ])
    prov = transform_document([page]).region_provenance
    assert prov[0]["role"] == "aside" and prov[0]["raw_text"]
    assert prov[1]["role"] == "reference" and prov[1]["raw_text"]


# ── reading order + determinism ─────────────────────────────────────────────


def test_reading_order_indices_monotonic_across_pages():
    pages = [
        GlmPage(page_no=1, regions=[_region(0, "text", "First page sentence one.")]),
        GlmPage(page_no=2, regions=[_region(0, "text", "Second page sentence two.")]),
    ]
    prov = transform_document(pages).region_provenance
    idxs = [p["first_raw_block_index"] for p in prov]
    assert idxs == sorted(idxs) and len(set(idxs)) == len(idxs)
    assert prov[0]["source_page"] == 1 and prov[1]["source_page"] == 2


# ── cross-page continuation stitch ───────────────────────────────────────────


def test_cross_page_continuation_stitched():
    pages = [
        GlmPage(page_no=1, regions=[
            _region(0, "text", "The sentence runs off the bottom of the")]),
        GlmPage(page_no=2, regions=[
            _region(0, "text", "page and continues onto the next one here.")]),
    ]
    tr = transform_document(pages)
    assert len(tr.region_provenance) == 1
    assert "continues onto the next one" in tr.region_provenance[0]["raw_text"]
    assert tr.region_provenance[0]["pages"] == [1, 2]
    assert any(e["reason"] == "cross_page_stitch" for e in tr.escalations)


# ── sdk error page ───────────────────────────────────────────────────────────


def test_sdk_error_page_escalated():
    pages = [GlmPage(page_no=1, regions=[], error="seat timeout")]
    tr = transform_document(pages)
    assert tr.region_provenance == []
    assert any(e["reason"] == "sdk_error" for e in tr.escalations)


# ── declared-section anchoring + chapter-opener synthesis ───────────────────


def _toc_body_pages():
    """ToC page declaring 3 sections + body pages where 4.2's opener lost its
    number to a badge image plus a review reprint."""
    toc = GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", "## Chapter Outline"),
        _region(1, "text", "4. 1 Alpha Topic"),
        _region(2, "text", "4. 2 Beta Topic"),
        _region(3, "text", "4. 3 Gamma Topic"),
        _region(4, "paragraph_title", "## 4.1 Alpha Topic"),
        _region(5, "text", "Alpha body prose long enough to be a paragraph."),
    ])
    body = GlmPage(page_no=2, regions=[
        _region(0, "image", ""),  # the number badge
        _region(1, "paragraph_title", "## Beta Topic"),  # number lost
        _region(2, "text", "Beta body prose long enough to be a paragraph."),
    ])
    review = GlmPage(page_no=3, regions=[
        _region(0, "paragraph_title", "## 4.2 Beta Topic"),  # review reprint
        _region(1, "text", "Beta review recap prose."),
    ])
    return [toc, body, review]


def test_declared_section_number_recovery():
    tr = transform_document(_toc_body_pages())
    heads = [p for p in tr.region_provenance if p["region_kind"] == "heading"]
    beta = [h for h in heads if "Beta Topic" in str(h.get("heading_text"))]
    # the body opener (page 2) is renumbered + promoted; the review reprint
    # (page 3) keeps its native numbered form and is NOT double-rewritten
    assert beta[0]["heading_text"] == "4.2 Beta Topic"
    assert beta[0]["level"] == 2
    assert beta[0].get("section_number_recovered") is True
    assert not beta[0].get("heading_level_pending")
    reasons = [e["reason"] for e in tr.escalations]
    assert "section_number_recovered" in reasons
    # 4.3 Gamma Topic is declared but matches nothing → loud escalation
    assert "declared_section_unanchored" in reasons
    # the stale pending escalation for the promoted heading is dropped
    pend = [e for e in tr.escalations if e["reason"] == "heading_level_pending"
            and "Beta" in str(e.get("text"))]
    assert not pend


def test_natively_numbered_opener_untouched():
    tr = transform_document(_toc_body_pages())
    heads = [p for p in tr.region_provenance if p["region_kind"] == "heading"]
    alpha = [h for h in heads if "Alpha Topic" in str(h.get("heading_text"))]
    assert alpha[0]["heading_text"] == "4.1 Alpha Topic"
    assert alpha[0]["level"] == 2
    assert not alpha[0].get("section_number_recovered")


def test_chapter_opener_synthesized_from_toc_major():
    tr = transform_document(_toc_body_pages())
    first_head = next(p for p in tr.region_provenance
                      if p["region_kind"] == "heading")
    assert first_head["heading_text"] == "Chapter 4"
    assert first_head["level"] == 1
    assert first_head.get("chapter_title_synthesized") is True
    assert tr.heading_tree[0] == (1, "Chapter 4")
    assert any(e["reason"] == "chapter_title_synthesized"
               for e in tr.escalations)


def test_chapter_opener_not_synthesized_when_doc_title_present():
    pages = _toc_body_pages()
    pages[0].regions.insert(0, _region(9, "doc_title", "Chapter 4 Placeholder"))
    tr = transform_document(pages)
    synth = [p for p in tr.region_provenance
             if p.get("chapter_title_synthesized")]
    assert not synth


def test_mid_document_doc_title_not_level_1():
    """A ``doc_title`` on an interior page is a page-banner section opener,
    not the document title — it must not become level 1 (level-1 suppresses
    the chapter-opener synthesis)."""
    pages = _toc_body_pages()
    pages[1].regions.insert(1, _region(9, "doc_title", "# Gamma Topic"))
    tr = transform_document(pages)
    heads = [p for p in tr.region_provenance if p["region_kind"] == "heading"]
    gammas = [h for h in heads if "Gamma Topic" in str(h.get("heading_text"))]
    assert gammas, "mid-doc doc_title should still emit a heading"
    # the ToC declares 4.3 Gamma Topic → the recovery pass renumbers it
    assert gammas[0]["level"] == 2
    assert gammas[0]["heading_text"] == "4.3 Gamma Topic"
    # the synthesized chapter opener is NOT suppressed
    assert tr.heading_tree[0] == (1, "Chapter 4")


# ── D2: deterministic math/text normalizer ──────────────────────────────────


def test_normalize_math_collapses_digit_split_inside_delimiters():
    # exact audit-evidence strings
    assert mn.normalize_math_text("$\\sqrt {5 0}$") == "$\\sqrt {50}$"
    assert mn.normalize_math_text("$4 2 m n$") == "$42 m n$"  # digit-letter never joined
    assert mn.normalize_math_text("$\\frac{1 1}{2 8}$") == "$\\frac{11}{28}$"
    # fixpoint: "1 2 3" → "123"
    assert mn.normalize_math_text("$1 2 3$") == "$123$"


def test_normalize_math_collapses_letter_spacing_in_text_command():
    assert mn.normalize_math_text("$\\mathrm {S i m p l i f y}$") == "$\\mathrm {Simplify}$"
    # a real multi-word \text arg is NEVER collapsed
    assert mn.normalize_math_text("$\\text{the answer}$") == "$\\text{the answer}$"


def test_normalize_math_never_joins_across_operator_or_outside_math():
    assert mn.normalize_math_text("$x + 1 0$") == "$x + 10$"
    assert mn.normalize_math_text("$1 + 0$") == "$1 + 0$"  # operator gap
    # digits OUTSIDE any math delimiter are untouched
    assert mn.normalize_math_text("chapter 1 0 review") == "chapter 1 0 review"


def test_fix_spaced_ordinals_joins_section_number():
    assert mn.fix_spaced_ordinals("1. 2 Use the Language") == "1.2 Use the Language"


def test_fix_spaced_ordinals_never_touches_list_item_or_decimal():
    # a numbered list item (no second bare digit) is untouched
    assert mn.fix_spaced_ordinals("1. Simplify the expression") == "1. Simplify the expression"
    # an already-joined decimal is untouched
    assert mn.fix_spaced_ordinals("1.2 Use the Language") == "1.2 Use the Language"


def test_normalizer_wired_into_transform_default_on():
    page = GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", "1. 2 Use the Language of Algebra"),
        _region(1, "display_formula", "\\frac{1 1}{2 8}"),
        _region(2, "text", "Body prose that clearly forms a sentence here."),
    ])
    prov = transform_document([page]).region_provenance
    head = next(p for p in prov if p["region_kind"] == "heading")
    assert head["heading_text"] == "1.2 Use the Language of Algebra"
    math = next(p for p in prov if p["region_kind"] == "math")
    assert "\\frac{11}{28}" in math["raw_text"]


def test_normalizer_flag_falsey_is_byte_identical(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", "0")
    page = GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", "1. 2 Use the Language of Algebra"),
        _region(1, "display_formula", "\\frac{1 1}{2 8}"),
        _region(2, "text", "Body prose that clearly forms a sentence here."),
    ])
    prov = transform_document([page]).region_provenance
    head = next(p for p in prov if p["region_kind"] == "heading")
    assert head["heading_text"] == "1. 2 Use the Language of Algebra"  # untouched
    math = next(p for p in prov if p["region_kind"] == "math")
    assert "\\frac{1 1}{2 8}" in math["raw_text"]


def test_resolve_math_normalize_mode_default_on_falsey_set(monkeypatch):
    monkeypatch.delenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", raising=False)
    assert mn.resolve_math_normalize_mode() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", off)
        assert mn.resolve_math_normalize_mode() is False
    for on in ("1", "true", "yes", "garbage", ""):
        monkeypatch.setenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", on)
        assert mn.resolve_math_normalize_mode() is True


# ── D3: caption-label demotion / end-matter promotion / box-title constraint ─


@pytest.mark.parametrize("label,expected_role", [
    ("Figure 1.2", "caption"),
    ("Table 1.4", "caption"),
    ("Example 1.3", "apparatus"),
    ("Try It 1.5", "apparatus"),
])
def test_numbered_caption_label_never_becomes_heading(label, expected_role):
    page = GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", label),
        _region(1, "text", "Some body prose that follows the label region here."),
    ])
    prov = transform_document([page]).region_provenance
    lab = prov[0]
    assert lab["region_kind"] == "paragraph"
    assert lab.get("role") == expected_role


@pytest.mark.parametrize("marker", [
    "KEY TERMS", "Key Concepts", "Chapter Review", "REVIEW EXERCISES", "PRACTICE TEST",
])
def test_endmatter_apparatus_promoted_to_level_2_heading(marker):
    page = GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", marker),
        _region(1, "text", "End-matter apparatus body prose sentence goes here."),
    ])
    prov = transform_document([page]).region_provenance
    head = prov[0]
    assert head["region_kind"] == "heading"
    assert head["level"] == 2
    assert not head.get("heading_level_pending")


def test_endmatter_with_leading_chapter_ordinal_promoted():
    page = GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", "1.5 Key Terms"),
        _region(1, "text", "End-matter apparatus body prose sentence goes here."),
    ])
    prov = transform_document([page]).region_provenance
    assert prov[0]["region_kind"] == "heading" and prov[0]["level"] == 2


def test_endmatter_apparatus_promoted_from_paragraph_typed_region():
    # a bare marker GLM-OCR mistyped as body text is still promoted
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "Key Terms"),
        _region(1, "text", "End-matter apparatus body prose sentence goes here."),
    ])
    prov = transform_document([page]).region_provenance
    assert prov[0]["region_kind"] == "heading" and prov[0]["level"] == 2


def test_prose_mentioning_key_terms_not_promoted():
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "The key terms in this chapter build on prior work."),
    ])
    prov = transform_document([page]).region_provenance
    assert prov[0]["region_kind"] == "paragraph"


def test_two_word_box_title_not_promoted_to_heading():
    for word in ("Equation", "Opposite"):
        page = GlmPage(page_no=1, regions=[
            _region(0, "text", word),
            _region(1, "text", "A body paragraph with at least four words present."),
        ])
        prov = transform_document([page]).region_provenance
        assert prov[0]["region_kind"] == "paragraph"
        assert prov[0]["pedagogy_class"] == "definition-box"


def test_three_word_box_title_still_promoted():
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "Widget Assembly Guide"),
        _region(1, "text", "A body paragraph with at least four words present."),
    ])
    prov = transform_document([page]).region_provenance
    assert prov[0]["region_kind"] == "heading"


def test_two_word_box_title_flag_off_keeps_heading(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", "0")
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "Opposite"),
        _region(1, "text", "A body paragraph with at least four words present."),
    ])
    prov = transform_document([page]).region_provenance
    assert prov[0]["region_kind"] == "heading"  # legacy behavior preserved


def test_toc_harvested_from_single_blob_region():
    """A ToC OCR'd as ONE multi-line text region (ch06 live failure) must
    still qualify as a declaration run."""
    toc_blob = "7. 1 Alpha Topic\n\n7. 2 Beta Topic\n\n7. 3 Gamma Topic"
    pages = [
        GlmPage(page_no=1, regions=[
            _region(0, "paragraph_title", "## Chapter Outline"),
            _region(1, "text", toc_blob),
        ]),
        GlmPage(page_no=2, regions=[
            _region(0, "image", ""),
            _region(1, "paragraph_title", "## Beta Topic"),
            _region(2, "text", "Beta body prose long enough to be a paragraph."),
        ]),
    ]
    tr = transform_document(pages)
    assert tr.heading_tree[0] == (1, "Chapter 7")
    heads = [p for p in tr.region_provenance if p["region_kind"] == "heading"]
    beta = [h for h in heads if "Beta Topic" in str(h.get("heading_text"))]
    assert beta[0]["heading_text"] == "7.2 Beta Topic"
    assert beta[0]["level"] == 2


# ── Chapter-ladder reconcile (SEMANTIK_CHAPTER_LADDER_RECONCILE) ────────────
# Anonymized whole-book shape: printed "Chapter N" openers + N.M sections.
# NOT real course content.


def _whole_book_pages():
    """A tiny multi-chapter book: preface (0.M), two chapters with openers."""
    front = GlmPage(page_no=1, regions=[
        _region(0, "doc_title", "SAMPLE WIDGET COMPENDIUM"),
        _region(1, "text", "A placeholder preface paragraph about widgets."),
        _region(2, "paragraph_title", "0.1 Who should read this guide"),
        _region(3, "text", "Placeholder audience prose for the preface."),
    ])
    ch1 = GlmPage(page_no=2, regions=[
        _region(0, "paragraph_title", "Chapter 1"),
        _region(1, "paragraph_title", "Widget Basics"),
        _region(2, "paragraph_title", "1.1 Alpha Widgets"),
        _region(3, "text", "Placeholder prose describing alpha widgets."),
    ])
    ch2 = GlmPage(page_no=3, regions=[
        _region(0, "paragraph_title", "Chapter 2"),
        _region(1, "paragraph_title", "2.1 Beta Widgets"),
        _region(2, "text", "Placeholder prose describing beta widgets."),
    ])
    return [front, ch1, ch2]


def _chapter_heads(tr, text):
    return [p for p in tr.region_provenance
            if p["region_kind"] == "heading"
            and str(p.get("heading_text")) == text]


def test_chapter_ladder_promotes_ascending_opener_series():
    tr = transform_document(_whole_book_pages())
    for text in ("Chapter 1", "Chapter 2"):
        head = _chapter_heads(tr, text)[0]
        assert head["level"] == 1, text
        assert head.get("chapter_ladder_promoted") is True
        assert not head.get("heading_level_pending")
    # N.M sections stay the level-2 spine; the preface's 0.1 too.
    for text in ("1.1 Alpha Widgets", "2.1 Beta Widgets",
                 "0.1 Who should read this guide"):
        assert _chapter_heads(tr, text)[0]["level"] == 2, text
    reasons = [e["reason"] for e in tr.escalations]
    assert "chapter_ladder_promoted" in reasons
    # promoted openers must not leave stale pending escalations behind
    pend = [e for e in tr.escalations
            if e["reason"] == "heading_level_pending"
            and str(e.get("text", "")).startswith("Chapter ")]
    assert not pend
    assert (1, "Chapter 1") in tr.heading_tree
    assert (1, "Chapter 2") in tr.heading_tree


def test_chapter_ladder_single_opener_corroborated_by_major():
    pages = [GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", "Chapter 3"),
        _region(1, "paragraph_title", "3.1 Gamma Widgets"),
        _region(2, "text", "Placeholder prose describing gamma widgets."),
    ])]
    tr = transform_document(pages)
    assert _chapter_heads(tr, "Chapter 3")[0]["level"] == 1


def test_chapter_ladder_lone_uncorroborated_mention_not_promoted():
    pages = [GlmPage(page_no=1, regions=[
        _region(0, "paragraph_title", "Chapter 5"),
        _region(1, "paragraph_title", "2.1 Beta Widgets"),
        _region(2, "text", "Placeholder prose describing beta widgets."),
    ])]
    tr = transform_document(pages)
    head = _chapter_heads(tr, "Chapter 5")[0]
    assert head["level"] == 3
    assert head.get("heading_level_pending") is True
    assert not head.get("chapter_ladder_promoted")


def test_chapter_ladder_apparatus_heading_never_promoted():
    pages = _whole_book_pages()
    pages.append(GlmPage(page_no=4, regions=[
        _region(0, "paragraph_title", "Chapter 2 Review"),
        _region(1, "text", "Placeholder review prose for chapter two."),
    ]))
    tr = transform_document(pages)
    review = _chapter_heads(tr, "Chapter 2 Review")[0]
    # an apparatus qualifier tail is never a chapter root — the heading keeps
    # its non-root level (pending, for the judge) and is never promoted
    assert review["level"] != 1
    assert not review.get("chapter_ladder_promoted")


def test_chapter_ladder_off_flag_byte_identical(monkeypatch):
    monkeypatch.setenv("SEMANTIK_CHAPTER_LADDER_RECONCILE", "0")
    tr = transform_document(_whole_book_pages())
    for text in ("Chapter 1", "Chapter 2"):
        head = _chapter_heads(tr, text)[0]
        assert head["level"] == 3, text
        assert head.get("heading_level_pending") is True
        assert "chapter_ladder_promoted" not in head
    assert not any(e["reason"] == "chapter_ladder_promoted"
                   for e in tr.escalations)


def test_chapter_ladder_suppresses_modal_major_opener_synthesis():
    """Promoted openers satisfy has_chapter_heading, so the anchor pass must
    NOT stack a synthesized modal-major opener on top of a real ladder."""
    toc = GlmPage(page_no=1, regions=[
        _region(0, "text", "1.1 Alpha Widgets 5\n1.2 Beta Widgets 9\n"
                           "1.3 Gamma Widgets 13"),
    ])
    body = GlmPage(page_no=2, regions=[
        _region(0, "paragraph_title", "Chapter 1"),
        _region(1, "paragraph_title", "1.1 Alpha Widgets"),
        _region(2, "text", "Placeholder prose describing alpha widgets."),
        _region(3, "paragraph_title", "Chapter 2"),
        _region(4, "paragraph_title", "2.1 Delta Widgets"),
        _region(5, "text", "Placeholder prose describing delta widgets."),
    ])
    tr = transform_document([toc, body])
    assert not any(p.get("chapter_title_synthesized")
                   for p in tr.region_provenance)
    assert not any(e["reason"] == "chapter_title_synthesized"
                   for e in tr.escalations)
