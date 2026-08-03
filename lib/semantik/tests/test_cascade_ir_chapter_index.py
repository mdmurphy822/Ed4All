"""Hermetic chapter-index regression tests for the SemantiK bridge seam.

The fixture models the structural signals behind a past chapter-IR defect
without loading an operator corpus: a metadata block, a page-numbered contents
run, a contiguous page-number-less chapter-index cluster, and numbered content
sections with no level-one opener. The chapter-index entries must be dropped,
while the numbered sections must derive a compact, valid chapter.
"""

from __future__ import annotations

import re

import pytest


def _region(index: int, text: str, *, level: int | None, role: str) -> dict:
    if role == "metadata_drop":
        region_kind = "metadata_drop"
    else:
        region_kind = "heading" if level is not None else "paragraph"
    return {
        "region_index": index,
        "region_kind": region_kind,
        "role": role,
        "confidence": 0.9,
        "wcag_status": "passed",
        "first_raw_block_index": index,
        "pages": [max(1, index // 4 + 1)],
        "heading_text": text if level is not None else None,
        "level": level,
        "figure_alt": None,
        "raw_text": text,
    }


def _synthetic_bridge() -> dict:
    index_titles = [
        "Chapter 1: Core Concepts",
        "Chapter 2: Applied Concepts",
        "Chapter 3: Extended Concepts",
        "Chapter 5: Systems",
        "Chapter 7: Analysis",
        "Chapter 10: Synthesis",
    ]
    sections = [
        "1.1 First Concept",
        "1.2 Second Concept",
        "1.4 Fourth Concept",
        "1.7 Seventh Concept",
        "1.8 Eighth Concept",
        "1.9 Ninth Concept",
    ]
    provenance = [
        _region(0, "Document metadata", level=None, role="metadata_drop"),
        _region(1, "Table of Contents", level=1, role="heading"),
        _region(2, "Overview 4", level=2, role="heading"),
        _region(3, "Reference 12", level=2, role="heading"),
        _region(4, "Practice 28", level=2, role="heading"),
        _region(5, "Review 46", level=2, role="heading"),
    ]
    provenance.extend(
        _region(6 + offset, title, level=1, role="heading")
        for offset, title in enumerate(index_titles)
    )
    next_index = len(provenance)
    for offset, title in enumerate(sections):
        heading_index = next_index + offset * 2
        provenance.extend(
            [
                _region(heading_index, title, level=2, role="heading"),
                _region(
                    heading_index + 1,
                    f"Neutral instructional content for section {offset + 1}.",
                    level=None,
                    role="body",
                ),
            ]
        )
    return {"region_provenance": provenance, "heading_tree": []}


@pytest.fixture()
def chapters(monkeypatch):
    monkeypatch.setenv("SEMANTIK_DROP_FRONTMATTER_TOC", "on")
    from lib.semantik.cascade_ir import from_bridge_json

    return from_bridge_json(_synthetic_bridge())


# ---------------------------------------------------------------------------
# (1) Chapter count is SMALL and sensible — NOT the 75-garbage-chapter pre-fix
#     blow-up, and no "(cont.) (cont.)" chains.
# ---------------------------------------------------------------------------


def test_chapter_count_is_small_and_sensible(chapters):
    assert len(chapters) == 1, [c.title for c in chapters]

    # No "(cont.) (cont.)" continuation chains — the pre-fix overflow signature.
    for ch in chapters:
        assert "(cont.) (cont.)" not in ch.title, (
            f"continuation-chain garbage title survived: {ch.title!r}"
        )


def test_chapter_index_entries_do_not_become_chapters(chapters):
    titles = [c.title for c in chapters]
    for phantom in (
        "Chapter 5:",
        "Chapter 7:",
        "Chapter 8:",
        "Chapter 9:",
        "Chapter 10:",
    ):
        assert not any(phantom in t for t in titles), (
            f"phantom chapter-index entry survived as a chapter: {titles}"
        )


def test_numbered_sections_survive_under_derived_chapter(chapters):
    all_headings = [
        str(b.heading_text or "")
        for ch in chapters
        for b in ch.blocks
    ]
    joined = "\n".join(all_headings)
    for sec in ("1.1 ", "1.2 ", "1.4 ", "1.7 ", "1.8 ", "1.9 "):
        assert sec in joined, f"section heading {sec!r} missing from IR"


# ---------------------------------------------------------------------------
# (2) Chain → adapter → SemantiKMarkersValidator passes; phantom titles never
#     reach the HTML; the structure extractor sees the single derived chapter.
# ---------------------------------------------------------------------------


def _adapt(chapters):
    from lib.semantik.adapter import normalize_cascade_to_ed4all

    class _Res:
        exit_action = "ship_with_confidence"
        wcag_status = "passed"
        theta_score = 0.9
        flags: list = []
        lane_used = "fast-lane"
        lang = "en"

    res = _Res()
    res.chapters = chapters
    return normalize_cascade_to_ed4all(res, pdf_stem="synthetic_chapter_index")


def test_chain_semantik_markers_validator_passes(chapters):
    from lib.validators.semantik_markers import SemantiKMarkersValidator

    out = _adapt(chapters)
    vres = SemantiKMarkersValidator().validate({"html_content": out["html"]})
    critical = [i for i in vres.issues if i.severity == "critical"]
    assert vres.passed, f"semantik_markers failed: {[i.code for i in critical]}"
    assert not critical

    # No phantom chapter-index titles leaked into the rendered HTML.
    for phantom in ("Chapter 5:", "Chapter 7:", "Chapter 10:"):
        assert phantom not in out["html"], f"phantom in HTML: {phantom}"

    assert "1.1 First Concept" in out["html"]


def test_structure_extractor_finds_only_the_derived_chapter(chapters):
    from lib.semantic_structure_extractor.semantic_structure_extractor import (
        SemanticStructureExtractor,
    )

    out = _adapt(chapters)
    structure = SemanticStructureExtractor().extract(out["html"])
    chapters = structure["chapters"]
    assert len(chapters) == 1
