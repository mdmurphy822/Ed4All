"""Unit tests — whole-document heading-contiguity normalization.

Closes the gap where a Stage-6 specialist (e.g. the hosted 70B prose
seat) emits ``<hN>`` tags inside a non-heading region body, bypassing
the per-region heading-tree normalization and producing a level skip the
Stage-10 ``heading_tree`` gate rejects.
"""

from __future__ import annotations

import re

from semantik_structure.assembler.heading_contiguity import (
    normalize_document_heading_levels,
)
from semantik_structure.gates.hard_document import _check_heading_hierarchy


def _levels(html: str) -> list[int]:
    return [int(m.group(1)) for m in re.finditer(r"<h([1-6])\b", html, re.I)]


def _texts(html: str) -> list[str]:
    return re.findall(r"<h[1-6][^>]*>(.*?)</h[1-6]>", html, re.I | re.S)


def test_repairs_embedded_h2_h6_skip():
    # The exact shape the 70B prose seat produced on the mini slice.
    html = (
        "<h1>Identify Multiples</h1>"
        "<h2>12 Chapter 1 Foundations</h2>"
        "<h6>EXAMPLE 1.6</h6>"
        "<p>body</p>"
    )
    out = normalize_document_heading_levels(html)
    assert _levels(out) == [1, 2, 3]
    assert _check_heading_hierarchy(out).passed is True


def test_preserves_heading_text_and_attrs():
    html = (
        '<h1 id="a">Alpha</h1>'
        '<h4 class="x" data-y="z">Beta</h4>'
    )
    out = normalize_document_heading_levels(html)
    # h1 -> h1, h4 -> h2 (demote-forward never-skip).
    assert _levels(out) == [1, 2]
    assert _texts(out) == ["Alpha", "Beta"]
    # Attributes preserved on the demoted heading.
    assert 'class="x"' in out and 'data-y="z"' in out
    assert 'id="a"' in out


def test_idempotent_on_contiguous_doc():
    html = "<h1>A</h1><h2>B</h2><h3>C</h3><h2>D</h2>"
    out = normalize_document_heading_levels(html)
    assert out == html  # byte-identical


def test_promote_first_heading_to_h1():
    html = "<h3>First</h3><h4>Second</h4>"
    out = normalize_document_heading_levels(html)
    assert _levels(out) == [1, 2]


def test_no_headings_unchanged():
    html = "<p>no headings here</p>"
    assert normalize_document_heading_levels(html) == html


def test_empty_unchanged():
    assert normalize_document_heading_levels("") == ""


def test_close_tag_level_rewritten():
    # The matching close tag must be rewritten too, else html5 balance
    # would see <h6>...</h6> mismatch after the open became <h3>.
    html = "<h1>A</h1><h2>B</h2><h6>C</h6>"
    out = normalize_document_heading_levels(html)
    assert "<h3>C</h3>" in out
    assert "<h6>" not in out and "</h6>" not in out


def test_multiple_h1_with_skip_only_fixes_skip():
    # Multiple h1 is allowed by the gate (no skip); a trailing skip is
    # the only thing that needs repair.
    html = "<h1>A</h1><h1>B</h1><h1>C</h1><h2>D</h2><h6>E</h6>"
    out = normalize_document_heading_levels(html)
    assert _levels(out) == [1, 1, 1, 2, 3]
    assert _check_heading_hierarchy(out).passed is True


# ---------------------------------------------------------------------------
# assemble_document wiring (_finalize_heading_contiguity)
# ---------------------------------------------------------------------------


def _make_doc(html: str):
    from semantik_structure.assembler.types import AssembledDoc

    return AssembledDoc(
        html=html,
        gaps_found=[],
        gaps_resolved=[],
        gaps_fallback=[],
        heading_tree=[(2, "12 Chapter 1 Foundations"), (6, "EXAMPLE 1.6")],
        landmarks={"main": 1},
        anchors={},
        region_provenance=[],
        sub_task_log={},
    )


def test_finalize_updates_html_and_tree():
    from semantik_structure.assembler.api import _finalize_heading_contiguity

    doc = _make_doc(
        "<h1>Identify Multiples</h1>"
        "<h2>12 Chapter 1 Foundations</h2>"
        "<h6>EXAMPLE 1.6</h6>"
    )
    out = _finalize_heading_contiguity(doc)
    assert _levels(out.html) == [1, 2, 3]
    # heading_tree re-derived to match the emitted levels (all three).
    assert out.heading_tree == [
        (1, "Identify Multiples"),
        (2, "12 Chapter 1 Foundations"),
        (3, "EXAMPLE 1.6"),
    ]
    assert out.sub_task_log.get("heading_contiguity")


def test_finalize_idempotent_byte_stable():
    from semantik_structure.assembler.api import _finalize_heading_contiguity

    clean = "<h1>A</h1><h2>B</h2><h3>C</h3>"
    doc = _make_doc(clean)
    original_tree = list(doc.heading_tree)
    out = _finalize_heading_contiguity(doc)
    # No rewrite -> html byte-identical AND the (stale) heading_tree is
    # left untouched (the pass only re-derives when it changed html).
    assert out.html == clean
    assert out.heading_tree == original_tree
    assert "heading_contiguity" not in out.sub_task_log


# ---------------------------------------------------------------------------
# Phase 9 — container-internal headings + the outer-section grouping must not
# introduce a level skip; the downstream contiguity re-level covers them.
# ---------------------------------------------------------------------------


def test_heading_contiguity_holds_with_container_headings(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    from semantik_structure.assembler.api import AssemblerConfig, assemble_document
    from semantik_structure.qwen_specialists.types import Candidate
    from semantik_structure.soft_reranker.types import RankedCandidate
    from semantik_structure.structure_graph import Region
    from semantik_structure.types import FeatureBlock, RawBlock

    def _fb(text: str) -> FeatureBlock:
        raw = RawBlock(
            text=text, page=1, bbox=(0.0, 0.0, 10.0, 10.0),
            page_width=100.0, page_height=100.0,
        )
        return FeatureBlock(
            raw=raw, size_bucket="md", gap_above=None, is_top_of_page=False,
            is_centered=False, caps=None, indent_bucket=0,
            relative_font_ratio=1.0,
        )

    def _cand(text: str) -> RankedCandidate:
        return RankedCandidate(
            candidate=Candidate(adapter="prose", request_id="r", text=text),
            score=1.0,
        )

    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"text": "Chapter", "level_hint": 1}),
        # A worked_example region carries a container <h4> label INSIDE a
        # non-heading region body (Phase 7), which would skip h1 -> h4.
        Region(kind="paragraph", feature_block_indices=(1,),
               payload={"text": "ex", "semantic_class": "worked_example"}),
    ]
    top = {0: _cand("<h2>Chapter</h2>"), 1: _cand("<p>EXAMPLE solve</p>")}
    doc = assemble_document(
        top, regions, [_fb("c"), _fb("e")],
        config=AssemblerConfig(skip_gap_fill=True),
    )
    outcome = _check_heading_hierarchy(doc.html)
    assert outcome.passed, (outcome.message, _levels(doc.html))
    # First heading is h1; the container <h4> auto-demoted to h2 (no skip).
    assert _levels(doc.html)[0] == 1
    assert _levels(doc.html) == [1, 2]


# ---------------------------------------------------------------------------
# Phase 3 (reading-order fix) — feed the assembler a READING-ORDER region list
# (headings interleaved with their body blocks, the post-
# SEMANTIK_READING_ORDER_FIX shape) and prove heading handling does not
# regress: the emitted <hN> outline is contiguous (no level skip) and the
# Stage-10 HEADING_TREE gate passes. Before the fix the assembler was fed the
# SEGREGATED pass-order list (every heading clustered at the front, every body
# at the back); reading-order input is strictly MORE correct, so the heading
# heuristics — tuned against contiguous-headings input — cannot regress on it.
# ---------------------------------------------------------------------------


def _no_level_skip(levels: list[int]) -> bool:
    """True iff the level sequence never jumps down by more than one (the
    HEADING_TREE contract: each heading is at most one deeper than the
    deepest open level so far)."""
    depth = 0
    for lv in levels:
        if lv > depth + 1:
            return False
        depth = lv
    return True


def test_heading_tree_under_reading_order(monkeypatch):
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    from semantik_structure.assembler.api import AssemblerConfig, assemble_document
    from semantik_structure.qwen_specialists.types import Candidate
    from semantik_structure.soft_reranker.types import RankedCandidate
    from semantik_structure.structure_graph import Region
    from semantik_structure.types import FeatureBlock, RawBlock

    def _fb(text: str) -> FeatureBlock:
        raw = RawBlock(
            text=text, page=1, bbox=(0.0, 0.0, 10.0, 10.0),
            page_width=100.0, page_height=100.0,
        )
        return FeatureBlock(
            raw=raw, size_bucket="md", gap_above=None, is_top_of_page=False,
            is_centered=False, caps=None, indent_bucket=0,
            relative_font_ratio=1.0,
        )

    def _cand(text: str) -> RankedCandidate:
        return RankedCandidate(
            candidate=Candidate(adapter="prose", request_id="r", text=text),
            score=1.0,
        )

    # READING ORDER: heading, body, heading, body, heading, body — exactly the
    # interleaving the reading-order sort restores (vs the segregated
    # H,H,H,...,body,body,body the assembler saw before the fix). The third
    # heading's specialist emits a deep <h4> (would skip h2 -> h4); the
    # whole-document contiguity pass must demote it, never skip.
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"text": "Chapter 1", "level_hint": 1}),
        Region(kind="paragraph", feature_block_indices=(1,),
               payload={"text": "intro body"}),
        Region(kind="heading", feature_block_indices=(2,),
               payload={"text": "Section 1.1", "level_hint": 2}),
        Region(kind="paragraph", feature_block_indices=(3,),
               payload={"text": "section body"}),
        Region(kind="heading", feature_block_indices=(4,),
               payload={"text": "Section 1.2", "level_hint": 2}),
        Region(kind="paragraph", feature_block_indices=(5,),
               payload={"text": "more body"}),
    ]
    top = {
        0: _cand("<h1>Chapter 1</h1>"),
        1: _cand("<p>intro body</p>"),
        2: _cand("<h2>Section 1.1</h2>"),
        3: _cand("<p>section body</p>"),
        4: _cand("<h4>Section 1.2</h4>"),  # a level skip a specialist emitted
        5: _cand("<p>more body</p>"),
    }
    fbs = [_fb("c"), _fb("i"), _fb("s1"), _fb("sb"), _fb("s2"), _fb("mb")]
    doc = assemble_document(
        top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True),
    )

    levels = _levels(doc.html)
    outcome = _check_heading_hierarchy(doc.html)
    assert outcome.passed, (outcome.message, levels)
    # First heading h1; no level skip anywhere (the deep h4 demoted forward).
    assert levels[0] == 1
    assert _no_level_skip(levels), levels
    assert 4 not in levels  # the skipping <h4> was re-leveled out
    # Every heading's text survives the re-level (text byte-preserved).
    htexts = _texts(doc.html)
    for label in ("Chapter 1", "Section 1.1", "Section 1.2"):
        assert any(label in t for t in htexts), (label, htexts)
    # Body prose renders interleaved between the headings (reading order
    # preserved through assembly — bodies are NOT segregated to the tail).
    assert "intro body" in doc.html
    assert doc.html.index("Chapter 1") < doc.html.index("intro body")
    assert doc.html.index("intro body") < doc.html.index("Section 1.1")
