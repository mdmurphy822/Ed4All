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
