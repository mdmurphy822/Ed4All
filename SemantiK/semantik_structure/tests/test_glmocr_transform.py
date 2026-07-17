"""Unit tests for the GLM-OCR deterministic transform + region classification.

Synthetic fixtures only — a tiny FAKE PP-DocLayoutV3 layout JSON with invented
placeholder text (NO real course content), NO GPU, NO seat.
"""

from __future__ import annotations

from semantik_structure.glmocr import region_map as rm
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
    page = GlmPage(page_no=1, regions=[
        _region(0, "text", "Widget Assembly"),
        _region(1, "text", "A body paragraph with at least four words present."),
    ])
    tr = transform_document([page])
    assert tr.region_provenance[0]["region_kind"] == "heading"
    assert tr.region_provenance[0]["heading_text"] == "Widget Assembly"


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
