"""Unit tests — deterministic Stage-5d structure correction (no GPU, mocked).

Covers each sub-pass of ``clean_structure`` (front-matter / phantom-TOC / OCR
drop, pedagogical demotion, gate / pass-through) + the invariant contract
(verbatim text, FB-partition immutability, token-conservation, fail-closed).
All CPU; no model load.
"""

from __future__ import annotations

import dataclasses

import pytest

from dart_semantic.qwen_specialists.deterministic_structure import (
    clean_structure,
    resolve_structure_clean_mode,
)
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _fb(text: str, page: int = 1) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        page_width=100.0,
        page_height=100.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _heading(text: str, idx: int, level: int = 3) -> Region:
    return Region(
        kind="heading",
        feature_block_indices=(idx,),
        payload={"text": text, "level_hint": level},
        source_region_id=idx,
    )


def _para(text: str, idx: int) -> Region:
    return Region(
        kind="paragraph",
        feature_block_indices=(idx,),
        payload={"text": text},
        source_region_id=idx,
    )


def _build(specs):
    """specs: list of (kind, text, page[, level]) -> (regions, feature_blocks)."""
    regions, fbs = [], []
    for i, spec in enumerate(specs):
        kind, text, page = spec[0], spec[1], spec[2]
        level = spec[3] if len(spec) > 3 else 3
        fbs.append(_fb(text, page))
        if kind == "heading":
            regions.append(_heading(text, i, level=level))
        else:
            regions.append(_para(text, i))
    return regions, fbs


# ---------------------------------------------------------------------------
# Gate / pass-through.
# ---------------------------------------------------------------------------


def test_gate_default_on():
    assert resolve_structure_clean_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF", "False"])
def test_gate_falsey_disables(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", val)
    assert resolve_structure_clean_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "garbage"])
def test_gate_truthy_or_garbage_enabled(monkeypatch, val):
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", val)
    assert resolve_structure_clean_mode() is True


def test_off_is_byte_identical_passthrough(monkeypatch):
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", "off")
    regions, fbs = _build([
        ("heading", "EXAMPLE 1", 5, 4),
        ("heading", "Chapter 5: Systems", 9, 1),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out == regions  # same objects, untouched
    assert diag["front_matter_dropped"] == 0
    assert diag["pedagogical_demoted"] == 0


def test_empty_input():
    out, diag = clean_structure([], [])
    assert out == []
    assert diag["headings_before"] == 0


# ---------------------------------------------------------------------------
# (B) Pedagogical demotion.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label",
    [
        "EXAMPLE 1",
        "EXAMPLE 1.2",
        "EXAMPLE",
        "Solution",
        "Try It",
        "How To",
        "Step 3",
        "Learning Objectives",
        "Be Prepared",
        "Practice Makes Perfect",
    ],
)
def test_pedagogical_labels_demoted(label):
    # Anchor with a real body section so the zone anchor protects the labels'
    # position (labels are AFTER the body anchor, so never front-matter-dropped).
    regions, fbs = _build([
        ("heading", "1.1 Introduction to Whole Numbers", 30, 2),
        ("heading", label, 31, 4),
        ("paragraph", "body text under the label", 31),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "paragraph"            # demoted
    assert out[0].kind == "heading"              # real section kept
    assert diag["pedagogical_demoted"] == 1
    # text preserved verbatim.
    assert out[1].payload["text"] == label


# ---------------------------------------------------------------------------
# (B') Pedagogical demotion — semantic CSS class hint.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected_class",
    [
        ("EXAMPLE 1", "pedagogy-example"),
        ("EXAMPLE 1.2", "pedagogy-example"),
        ("EXAMPLE", "pedagogy-example"),
        ("Solution", "pedagogy-solution"),
        ("Try It", "pedagogy-try-it"),
        ("How To", "pedagogy-how-to"),
        ("Step 3", "pedagogy-step"),
        ("Learning Objectives", "pedagogy-objectives"),
        ("Be Prepared", "pedagogy-be-prepared"),
        ("Practice Makes Perfect", "pedagogy-practice"),
    ],
)
def test_demoted_pedagogical_carries_class_hint(label, expected_class):
    regions, fbs = _build([
        ("heading", "1.1 Introduction to Whole Numbers", 30, 2),
        ("heading", label, 31, 4),
        ("paragraph", "body text under the label", 31),
    ])
    out, diag = clean_structure(regions, fbs)
    # demoted to paragraph AND carries the mapped semantic class hint.
    assert out[1].kind == "paragraph"
    assert out[1].payload.get("css_class") == expected_class
    # text still verbatim, FB partition immutable.
    assert out[1].payload["text"] == label
    assert out[1].feature_block_indices == regions[1].feature_block_indices
    assert out[1].source_region_id == regions[1].source_region_id
    # diagnostics count + per-class breakdown.
    assert diag["pedagogical_classed"] == 1
    assert diag["pedagogical_class_counts"] == {expected_class: 1}
    assert diag["text_changes"] == 0


def test_kept_real_heading_has_no_css_class():
    # A real numbered section heading is NEITHER demoted NOR classed.
    regions, fbs = _build([
        ("heading", "1.1 Anchor Section", 30, 2),
        ("heading", "2.3 Add and Subtract Fractions", 35, 2),
        ("paragraph", "content", 35),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "heading"
    assert "css_class" not in (out[1].payload or {})
    assert diag["pedagogical_classed"] == 0
    assert diag["pedagogical_class_counts"] == {}


def test_front_matter_drop_has_no_css_class():
    # A non-pedagogical demotion (front-matter metadata_drop) carries NO
    # css_class — the class hint is strictly for pedagogical demotions.
    specs = [
        ("heading", "Chapter 1: Foundations", 9, 1),
        ("heading", "Chapter 5: Systems", 9, 1),
        ("heading", "Chapter 7: Factoring", 10, 1),
        ("heading", "Chapter 8: Rational", 10, 1),
        ("heading", "Chapter 9: Roots", 10, 1),
        ("heading", "1.1 Introduction to Whole Numbers", 30, 2),
        ("paragraph", "real body content", 30),
    ]
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    for i in range(5):
        assert out[i].kind == "metadata_drop"
        assert "css_class" not in (out[i].payload or {})
    assert diag["pedagogical_classed"] == 0


def test_mixed_pedagogical_class_counts():
    # Two EXAMPLE + one Solution + one Try It -> per-class breakdown.
    specs = [
        ("heading", "1.1 Anchor", 30, 2),
        ("heading", "EXAMPLE 1", 31, 4),
        ("paragraph", "p1", 31),
        ("heading", "Solution", 31, 5),
        ("paragraph", "p2", 31),
        ("heading", "EXAMPLE 2", 32, 4),
        ("paragraph", "p3", 32),
        ("heading", "Try It", 33, 4),
        ("paragraph", "p4", 33),
    ]
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    assert diag["pedagogical_demoted"] == 4
    assert diag["pedagogical_classed"] == 4
    assert diag["pedagogical_class_counts"] == {
        "pedagogy-example": 2,
        "pedagogy-solution": 1,
        "pedagogy-try-it": 1,
    }
    # every demoted block carries a class hint.
    classed = [r.payload.get("css_class") for r in out if r.kind == "paragraph"
               and r.payload.get("css_class")]
    assert sorted(classed) == [
        "pedagogy-example", "pedagogy-example", "pedagogy-solution",
        "pedagogy-try-it",
    ]


def test_class_hint_off_when_pass_disabled(monkeypatch):
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", "off")
    regions, fbs = _build([
        ("heading", "1.1 Anchor", 30, 2),
        ("heading", "EXAMPLE 1", 31, 4),
        ("paragraph", "p", 31),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out == regions  # untouched, no css_class stamped
    assert diag["pedagogical_classed"] == 0


@pytest.mark.parametrize(
    "real_heading",
    [
        "1.1 Introduction to Whole Numbers",
        "Chapter 1: Foundations",
        "2.3 Add and Subtract Fractions",
    ],
)
def test_real_headings_not_demoted(real_heading):
    regions, fbs = _build([
        ("heading", "1.1 Anchor Section", 30, 2),
        ("heading", real_heading, 35, 2),
        ("paragraph", "content", 35),
    ])
    out, _ = clean_structure(regions, fbs)
    assert out[1].kind == "heading"


# ---------------------------------------------------------------------------
# (A) Front-matter / phantom-TOC / OCR drop.
# ---------------------------------------------------------------------------


def test_phantom_chapter_cluster_dropped():
    # 5 "Chapter N" headings packed onto pages 9-10 (front-matter), then a real
    # body section opener much later — the cluster is phantom, the body kept.
    specs = [
        ("heading", "Chapter 1: Foundations", 9, 1),
        ("heading", "Chapter 5: Systems", 9, 1),
        ("heading", "Chapter 7: Factoring", 10, 1),
        ("heading", "Chapter 8: Rational", 10, 1),
        ("heading", "Chapter 9: Roots", 10, 1),
        ("heading", "1.1 Introduction to Whole Numbers", 30, 2),
        ("paragraph", "real body content", 30),
    ]
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    # all 5 phantom Chapter headings dropped.
    for i in range(5):
        assert out[i].kind == "metadata_drop", f"region {i} not dropped"
    # real body section + its content untouched.
    assert out[5].kind == "heading"
    assert out[6].kind == "paragraph"
    assert diag["front_matter_dropped"] == 5


def test_toc_line_with_trailing_pagenum_dropped():
    specs = [
        ("heading", "Systems of Linear Equations 577", 6, 2),
        ("heading", "Polynomials 645", 6, 2),
        ("heading", "1.1 Real Body Section", 30, 2),
        ("paragraph", "content", 30),
    ]
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "metadata_drop"
    assert out[1].kind == "metadata_drop"
    assert out[2].kind == "heading"  # real body kept


@pytest.mark.parametrize("noise", ["O PEN S TAX", "R ICE U NIVERSITY"])
def test_ocr_titlepage_noise_dropped(noise):
    specs = [
        ("heading", noise, 5, 4),
        ("heading", "1.1 Real Body Section", 30, 2),
        ("paragraph", "content", 30),
    ]
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "metadata_drop"
    assert diag["front_matter_dropped"] >= 1


def test_real_allcaps_title_not_ocr_noise():
    # A genuine all-caps heading (one stray initial at most) is NOT OCR noise.
    specs = [
        ("heading", "LEARNING OBJECTIVES", 5, 4),
        ("heading", "1.1 Real Body Section", 30, 2),
        ("paragraph", "content", 30),
    ]
    regions, fbs = _build(specs)
    out, _ = clean_structure(regions, fbs)
    # "LEARNING OBJECTIVES" is a pedagogical label -> demoted to paragraph,
    # NOT dropped as metadata (it carries real teaching-section text).
    assert out[0].kind == "paragraph"


def test_real_body_chapter_not_dropped():
    # A single real "Chapter 1" opener on a body page with content directly
    # behind it (page-sparse, not a dense front-matter cluster) is KEPT.
    specs = [
        ("heading", "1.1 Anchor", 30, 2),
        ("paragraph", "anchor content", 30),
        ("heading", "Chapter 2: Solving Equations", 80, 1),
        ("paragraph", "chapter 2 intro content", 80),
    ]
    regions, fbs = _build(specs)
    out, _ = clean_structure(regions, fbs)
    assert out[2].kind == "heading"


# ---------------------------------------------------------------------------
# (A') Fused-TOC heading drop + (B') bullet/list-item heading demote.
# ---------------------------------------------------------------------------


_FUSED_TOC_TEXT = (
    "1.1 Introduction to Whole Numbers 1.2 Use the Language of Algebra "
    "1.3 Add and Subtract Integers 1.4 Multiply and Divide Integers"
)
_BULLET_TEXT = (
    "◦ Yes–add 1 to the digit in the given place value. "
    "◦ No–do not change the digit in the given place value."
)


def test_fused_toc_heading_dropped():
    # A heading carrying >=2 "N.M" tokens is a fused TOC run -> metadata_drop.
    regions, fbs = _build([
        ("heading", _FUSED_TOC_TEXT, 6, 2),
        ("heading", "1.1 Real Body Section", 30, 2),
        ("paragraph", "content", 30),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "metadata_drop"
    assert out[1].kind == "heading"          # real single-N.M section kept
    assert diag["fused_toc_dropped"] == 1
    # text preserved verbatim on the dropped region.
    assert out[0].payload["text"] == _FUSED_TOC_TEXT


def test_single_section_number_heading_not_fused_toc():
    # Exactly ONE "N.M" -> a legitimate section heading, never demoted/dropped.
    regions, fbs = _build([
        ("heading", "1.1 Anchor Section", 30, 2),
        ("heading", "1.2 Use the Language of Algebra", 35, 2),
        ("paragraph", "content", 35),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "heading"
    assert out[1].kind == "heading"
    assert diag["fused_toc_dropped"] == 0


def test_bullet_heading_demoted():
    # A bullet/list-item heading -> paragraph (NOT a heading), text verbatim.
    regions, fbs = _build([
        ("heading", "1.1 Anchor Section", 30, 2),
        ("heading", _BULLET_TEXT, 31, 4),
        ("paragraph", "body", 31),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "paragraph"
    assert out[0].kind == "heading"          # real section kept
    assert diag["list_item_demoted"] == 1
    assert out[1].payload["text"] == _BULLET_TEXT
    # not a pedagogical demote -> no css_class hint.
    assert "css_class" not in (out[1].payload or {})


@pytest.mark.parametrize(
    "bullet_text",
    [
        "• a single top-level bullet item here",
        "‣ triangle-bullet list content here",
        "▪ square-bullet list content here",
        "– en-dash bullet item content here",
        "- hyphen bullet item content here",
    ],
)
def test_various_bullet_markers_demoted(bullet_text):
    regions, fbs = _build([
        ("heading", "1.1 Anchor Section", 30, 2),
        ("heading", bullet_text, 31, 4),
        ("paragraph", "body", 31),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "paragraph"
    assert diag["list_item_demoted"] == 1


def test_hyphen_hugging_operand_not_a_bullet():
    # "-40 below zero" is a minus hugging its operand, NOT a list bullet
    # (no space after the dash) -> the heading is untouched by the bullet rule.
    regions, fbs = _build([
        ("heading", "1.1 Anchor Section", 30, 2),
        ("heading", "-40 degrees and falling", 35, 2),
        ("paragraph", "content", 35),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "heading"
    assert diag["list_item_demoted"] == 0


def test_real_heading_not_fused_or_bullet_demoted():
    # A plain real heading trips neither new rule.
    regions, fbs = _build([
        ("heading", "1.1 Anchor Section", 30, 2),
        ("heading", "Chapter 2: Solving Equations", 80, 1),
        ("paragraph", "content", 80),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "heading"
    assert diag["fused_toc_dropped"] == 0
    assert diag["list_item_demoted"] == 0


def test_new_detectors_byte_stable_when_off(monkeypatch):
    # Flag off -> fused-TOC + bullet headings are byte-identical pass-through.
    monkeypatch.setenv("SEMANTIK_STRUCTURE_CLEAN", "off")
    regions, fbs = _build([
        ("heading", _FUSED_TOC_TEXT, 6, 2),
        ("heading", _BULLET_TEXT, 31, 4),
        ("paragraph", "body", 31),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out == regions
    assert diag["fused_toc_dropped"] == 0
    assert diag["list_item_demoted"] == 0


# ---------------------------------------------------------------------------
# Invariants — verbatim text, FB partition, token-conservation, audit.
# ---------------------------------------------------------------------------


def test_fb_partition_and_text_immutable():
    specs = [
        ("heading", "1.1 Anchor", 30, 2),
        ("heading", "EXAMPLE 1", 31, 4),
        ("paragraph", "p", 31),
    ]
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    for before, after in zip(regions, out):
        assert after.feature_block_indices == before.feature_block_indices
        assert after.source_region_id == before.source_region_id
        assert after.payload["text"] == before.payload["text"]  # verbatim
    assert diag["text_changes"] == 0


def test_retag_uses_dataclasses_replace_not_mutation():
    # The original frozen Region must be untouched (replace returns a new obj).
    specs = [("heading", "1.1 Anchor", 30, 2), ("heading", "Solution", 31, 4),
             ("paragraph", "p", 31)]
    regions, fbs = _build(specs)
    original_kind = regions[1].kind
    out, _ = clean_structure(regions, fbs)
    assert regions[1].kind == original_kind  # input unchanged
    assert out[1].kind == "paragraph"
    # frozen dataclass: replace round-trips without raising.
    assert dataclasses.replace(out[1], kind="paragraph") is not out[1]


def test_diagnostics_shape():
    specs = [
        ("heading", "Chapter 5: Systems", 9, 1),
        ("heading", "Chapter 7: Factoring", 9, 1),
        ("heading", "Chapter 8: Rational", 10, 1),
        ("heading", "Chapter 9: Roots", 10, 1),
        ("heading", "1.1 Real Body", 30, 2),
        ("heading", "EXAMPLE 1", 31, 4),
        ("paragraph", "p", 31),
    ]
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    assert set(diag) >= {
        "front_matter_dropped",
        "pedagogical_demoted",
        "pedagogical_classed",
        "pedagogical_class_counts",
        "headings_before",
        "headings_after",
        "headings_before_by_level",
        "headings_after_by_level",
        "text_changes",
    }
    assert diag["headings_after"] < diag["headings_before"]
    assert diag["front_matter_dropped"] == 4
    assert diag["pedagogical_demoted"] == 1
    assert diag["pedagogical_classed"] == 1
    assert diag["pedagogical_class_counts"] == {"pedagogy-example": 1}


def test_pedagogical_label_not_dropped_as_toc_in_frontmatter_zone():
    """A TOC-SHAPED pedagogical label ("EXAMPLE 1.16") must NOT be metadata_dropped
    even when the whole doc is the front-matter zone (a mid-chapter SLICE with no
    chapter anchor) — it survives as a pedagogy-example paragraph (-> worked_example
    box). Regression for the slice worked-example mis-grouping: "EXAMPLE 1.16"
    matches the "title + trailing-1.16-page-number" TOC-drop shape and was dropped,
    so the example lost its box."""
    regions, fbs = _build([
        ("heading", "EXAMPLE 1.16", 1),
        ("paragraph", "Solution", 1),
        ("paragraph", "Multiply first.", 1),
        ("heading", "EXAMPLE 1.17", 2),
        ("paragraph", "Solution", 2),
    ])
    out, _diag = clean_structure(regions, fbs)
    ex = [r for r in out if (r.payload or {}).get("text", "").strip().startswith("EXAMPLE")]
    assert len(ex) == 2, "EXAMPLE labels were dropped"
    for r in ex:
        assert r.kind != "metadata_drop", f"pedagogical label dropped as metadata: {r.payload}"
        assert (r.payload or {}).get("css_class") == "pedagogy-example"


# ---------------------------------------------------------------------------
# Defect 3(a) — running-header text backstop.
# ---------------------------------------------------------------------------


def test_running_header_with_page_number_dropped_in_body():
    """A 'Chapter N <words> <3-4 digit page>' heading that escaped the
    FB-position running-header detector is page furniture ANYWHERE in the doc
    and is re-tagged metadata_drop (defect 3a — 39 such strings became bogus
    <h2> on the EA2e scan)."""
    regions, fbs = _build([
        ("heading", "9.1 Simplify Expressions with Roots", 30, 2),  # real anchor
        ("paragraph", "The nth root generalizes the square root.", 30),
        ("heading", "Chapter 9 Roots and Radicals 1039", 31, 2),     # furniture
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "heading"               # real section survives
    assert out[2].kind == "metadata_drop"         # running header dropped
    assert diag["running_header_dropped"] == 1


def test_real_chapter_title_without_page_number_kept():
    """A real chapter title with NO trailing page number is never dropped by
    the running-header backstop (anti-FP)."""
    regions, fbs = _build([
        ("heading", "Chapter 9 Roots and Radicals", 30, 1),
        ("paragraph", "Body content.", 30),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "heading"
    assert diag["running_header_dropped"] == 0


# ---------------------------------------------------------------------------
# Defect 3(b) — OCR-garbled pedagogical labels routed to the pedagogy path.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected_class",
    [
        ("| EXAMPLE9.9 | PLE 9.9", "pedagogy-example"),
        ("[y] EXAMPLE 3", "pedagogy-example"),
        ("TRYIT::", "pedagogy-try-it"),
        ("| TRY IT 9.5", "pedagogy-try-it"),
    ],
)
def test_ocr_garbled_pedagogical_labels_demoted(label, expected_class):
    """OCR-garbled EXAMPLE / TRY IT labels (leading gutter glyphs, fused words)
    the council mis-promoted to <h2> are demoted to a pedagogy-* paragraph, not
    left as section headings (defect 3b — 20 garbled EXAMPLE labels became
    <h2>)."""
    regions, fbs = _build([
        ("heading", "9.1 Introduction", 30, 2),   # real anchor (protects zone)
        ("heading", label, 31, 4),
        ("paragraph", "body under the label", 31),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "paragraph"                       # demoted
    assert out[1].payload.get("css_class") == expected_class
    assert out[1].payload["text"] == label                  # text verbatim


# ---------------------------------------------------------------------------
# Defect 3(c) — gated inline N.M section-heading promotion.
# ---------------------------------------------------------------------------


def test_section_promotion_off_by_default():
    """Without SEMANTIK_PROMOTE_SECTION_HEADINGS, a mis-typed 'N.M Title'
    paragraph stays a paragraph (byte-identical default path)."""
    regions, fbs = _build([
        ("heading", "Chapter 9 Roots and Radicals", 30, 1),
        ("paragraph", "9.2 Simplify Square Roots", 31),
        ("paragraph", "9.3 Add and Subtract Square Roots", 32),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "paragraph"
    assert out[2].kind == "paragraph"
    assert diag["section_promoted"] == 0


def test_section_promotion_gated_on_promotes_dominant_chapter_sections(monkeypatch):
    """With the flag on, a STANDALONE 'N.M Title-Case' paragraph whose N equals
    the document's dominant chapter number is promoted paragraph→heading
    (defect 3c — real section headings '9.2 …' / '9.3 …' were demoted to <p>)."""
    monkeypatch.setenv("SEMANTIK_PROMOTE_SECTION_HEADINGS", "on")
    regions, fbs = _build([
        ("heading", "Chapter 9 Roots and Radicals", 30, 1),
        ("paragraph", "9.2 Simplify Square Roots", 31),
        ("paragraph", "9.3 Add and Subtract Square Roots", 32),
        ("paragraph", "This is ordinary body prose that is not a section title.", 33),
        ("paragraph", "3.1 A Section From A Different Chapter", 34),  # N != dominant
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "heading"                       # 9.2 promoted
    assert out[1].payload["level_hint"] == 2
    assert out[2].kind == "heading"                       # 9.3 promoted
    assert out[3].kind == "paragraph"                     # body prose untouched
    assert out[4].kind == "paragraph"                     # 3.1 (wrong chapter) untouched
    assert diag["section_promoted"] == 2
    # Text preserved verbatim on the promoted regions.
    assert out[1].payload["text"] == "9.2 Simplify Square Roots"


# ---------------------------------------------------------------------------
# 2026-07-03 scan audit — conversion-path (Region-level) heading fixes.
# ---------------------------------------------------------------------------


def test_decorated_solution_label_demoted():
    # OCR-decorated run-in "Solution" labels (defect 3a) demote to a
    # pedagogy-solution paragraph instead of surviving as spurious <h3>.
    for decorated in [") Solution", "™ Solution", "“ Solution"]:
        regions, fbs = _build([
            ("heading", "1.1 Introduction to Whole Numbers", 30, 2),
            ("heading", decorated, 31, 3),
            ("paragraph", "steps under the label", 31),
        ])
        out, diag = clean_structure(regions, fbs)
        assert out[1].kind == "paragraph", decorated
        assert out[1].payload.get("css_class") == "pedagogy-solution"
        assert out[1].payload["text"] == decorated  # verbatim


def test_running_header_leading_page_dropped():
    # Leading-page-number running header (defect 3b) dropped anywhere in doc.
    regions, fbs = _build([
        ("heading", "1.1 Anchor Section", 30, 2),
        ("heading", "188 Chapter 1 Foundations", 31, 2),
        ("paragraph", "body", 31),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "metadata_drop"
    assert diag["running_header_dropped"] >= 1


def test_repeat_count_running_header_keeps_first_drops_rest():
    # A bare "Chapter 4 Graphs" recurring >3× is per-page furniture: keep the
    # FIRST (real opener), drop the rest (defect 3b repeat-count rule).
    specs = [("heading", "1.1 Anchor", 30, 2)]
    for pg in range(31, 37):  # 6 repeats
        specs.append(("heading", "Chapter 4 Graphs", pg, 2))
        specs.append(("paragraph", f"body {pg}", pg))
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    header_kinds = [out[i].kind for i in range(1, len(out), 2)]
    assert header_kinds[0] == "heading"          # first kept
    assert all(k == "metadata_drop" for k in header_kinds[1:])  # rest dropped
    assert diag["repeated_furniture_dropped"] == 5


def test_repeat_count_below_threshold_keeps_all():
    # 3 repeats (== threshold) → all kept (no furniture drop).
    specs = [("heading", "1.1 Anchor", 30, 2)]
    for pg in range(31, 34):
        specs.append(("heading", "Chapter 4 Graphs", pg, 2))
    regions, fbs = _build(specs)
    out, diag = clean_structure(regions, fbs)
    assert diag["repeated_furniture_dropped"] == 0


# ---------------------------------------------------------------------------
# P2 — VLM structural hints (NON-AUTHORITATIVE): sub-pass E audit breadcrumb +
# demotion immunity. The promotable SET is invariant to hints (set-equality);
# a hint can only add an audit breadcrumb and can never veto a demotion.
# ---------------------------------------------------------------------------


def _vlm_heading_hint(coverage: str = "whole_block", level: int = 2) -> dict:
    return {"kind": "heading", "level": level, "marker": None, "coverage": coverage}


def test_section_promotion_set_equality_with_vlm_hints(monkeypatch):
    """The promoted index SET is identical with VLM hints on vs off — the
    deterministic _STANDALONE_SECTION_RE + dominant-ordinal is the SOLE
    trigger; a hint never widens (or narrows) the set."""
    monkeypatch.setenv("SEMANTIK_PROMOTE_SECTION_HEADINGS", "on")
    specs = [
        ("heading", "Chapter 9 Roots and Radicals", 30, 1),
        ("paragraph", "9.2 Simplify Square Roots", 31),
        ("paragraph", "This is ordinary body prose, not a section title.", 32),
        ("paragraph", "9.3 Add and Subtract Square Roots", 33),
    ]

    # Hints OFF.
    regions, fbs = _build(specs)
    out_off, diag_off = clean_structure(regions, fbs)
    promoted_off = {i for i, r in enumerate(out_off) if r.kind == "heading" and i != 0}

    # Hints ON — put a heading hint even on the BODY-PROSE fb (index 2), which
    # must NOT be promoted (proves a hint can't widen the set).
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", "1")
    monkeypatch.setenv("SEMANTIK_VLM_STRUCT_HINTS", "1")
    regions2, fbs2 = _build(specs)
    for fb in fbs2:
        fb.vlm_hint = _vlm_heading_hint()
    out_on, diag_on = clean_structure(regions2, fbs2)
    promoted_on = {i for i, r in enumerate(out_on) if r.kind == "heading" and i != 0}

    assert promoted_off == promoted_on == {1, 3}
    assert diag_off["section_promoted"] == diag_on["section_promoted"] == 2


def test_section_promotion_vlm_breadcrumb_audit_only(monkeypatch):
    """A promoted section whose seed FB carries a whole-block VLM heading hint
    gets an audit breadcrumb (structure_clean.vlm_corroborated); without a hint
    the breadcrumb is absent. The kind decision is invariant either way."""
    monkeypatch.setenv("SEMANTIK_PROMOTE_SECTION_HEADINGS", "on")
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", "1")
    monkeypatch.setenv("SEMANTIK_VLM_STRUCT_HINTS", "1")
    specs = [
        ("heading", "Chapter 9 Roots and Radicals", 30, 1),
        ("paragraph", "9.2 Simplify Square Roots", 31),
        ("paragraph", "9.3 Add and Subtract Square Roots", 32),
    ]
    regions, fbs = _build(specs)
    fbs[1].vlm_hint = _vlm_heading_hint()          # corroborated
    # fbs[2] gets a PREFIX-coverage hint -> must NOT corroborate.
    fbs[2].vlm_hint = _vlm_heading_hint(coverage="prefix")
    out, _ = clean_structure(regions, fbs)
    assert out[1].kind == "heading"
    assert out[1].payload["structure_clean"].get("vlm_corroborated") is True
    assert out[2].kind == "heading"
    assert "vlm_corroborated" not in out[2].payload.get("structure_clean", {})


def test_demotion_immunity_pedagogical_label_with_vlm_hint(monkeypatch):
    """A pedagogical label ('EXAMPLE 2') carrying a VLM heading hint is STILL
    demoted by sub-pass B — a hint can never veto a deterministic demotion
    (VLM markdown reliably '##'s pedagogical labels, exactly the class
    clean_structure demotes)."""
    monkeypatch.setenv("SEMANTIK_VLM_EXTRACT", "1")
    monkeypatch.setenv("SEMANTIK_VLM_STRUCT_HINTS", "1")
    regions, fbs = _build([
        ("heading", "1.1 Introduction to Whole Numbers", 30, 2),
        ("heading", "EXAMPLE 2", 31, 3),
        ("paragraph", "steps under the label", 31),
    ])
    fbs[1].vlm_hint = _vlm_heading_hint()          # a heading hint on the label
    out, _ = clean_structure(regions, fbs)
    assert out[1].kind == "paragraph"              # demoted despite the hint
    assert out[1].payload.get("css_class") == "pedagogy-example"


# ---------------------------------------------------------------------------
# Defect 3 — obviously-fused heading demoted to paragraph (refuse-as-heading).
# ---------------------------------------------------------------------------


def test_fused_heading_multiple_math_runs_demoted():
    # A page-top mega-heading swallowing exercise math (>=2 $...$ runs).
    regions, fbs = _build([
        ("heading", r"Chapter 9 $\sqrt[4]{9c^8}$ Denise wants $\sqrt{81}$ tiles", 3, 2),
        ("paragraph", "Ordinary body prose about square roots follows here.", 3),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "paragraph"  # demoted
    assert diag["fused_heading_demoted"] == 1
    # Text preserved verbatim.
    assert out[0].payload["text"] == regions[0].payload["text"]


def test_fused_heading_sentence_prose_demoted():
    regions, fbs = _build([
        ("heading", "This heading actually swallowed a whole sentence of real body prose text.", 4, 2),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "paragraph"
    assert diag["fused_heading_demoted"] == 1


def test_real_heading_not_demoted():
    regions, fbs = _build([
        ("heading", "9.3 Add and Subtract Square Roots", 5, 2),
        ("heading", "Chapter 9 Roots and Radicals", 5, 1),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "heading"
    assert out[1].kind == "heading"
    assert diag["fused_heading_demoted"] == 0


# ---------------------------------------------------------------------------
# Defect 4 — apparatus paragraph promoted to a heading (gated).
# ---------------------------------------------------------------------------


def test_apparatus_paragraph_promoted_when_flag_on(monkeypatch):
    monkeypatch.setenv("SEMANTIK_PROMOTE_SECTION_HEADINGS", "1")
    regions, fbs = _build([
        ("heading", "9.1 Simplify Square Roots", 5, 2),  # anchors dominant ord=9
        ("paragraph", "PRACTICE TEST", 40),
        ("paragraph", "Review Exercises", 41),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[1].kind == "heading"  # PRACTICE TEST promoted
    assert out[2].kind == "heading"  # Review Exercises promoted
    assert diag["section_promoted"] == 2


def test_apparatus_paragraph_untouched_when_flag_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_PROMOTE_SECTION_HEADINGS", raising=False)
    regions, fbs = _build([
        ("paragraph", "PRACTICE TEST", 40),
    ])
    out, diag = clean_structure(regions, fbs)
    assert out[0].kind == "paragraph"
    assert diag["section_promoted"] == 0
