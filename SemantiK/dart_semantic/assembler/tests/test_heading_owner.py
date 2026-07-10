"""ITEM4 Phase 3 — containment builder owns structural heading levels.

pass_9a Sub-task 2 is apply-only under containment (reads ``tree.levels``); the
whole-doc contiguity pass is demoted to verification (a structural-drift
diagnostic). CPU-only, no model load, no GPU. Determinism-sensitive under
PYTHONHASHSEED=0,1,2 (I5).
"""

from __future__ import annotations

import random
import types

from dart_semantic.assembler.api import (
    AssemblerConfig,
    _count_structural_heading_drift,
    assemble_document,
)
from dart_semantic.assembler.heading_tree import normalize_heading_levels
from dart_semantic.containment import build_containment_tree
from dart_semantic.soft_reranker.types import RankedCandidate
from dart_semantic.qwen_specialists.types import Candidate
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


def _fb(text: str) -> FeatureBlock:
    raw = RawBlock(text=text, page=1, bbox=(0.0, 0.0, 10.0, 10.0),
                   page_width=100.0, page_height=100.0)
    return FeatureBlock(raw=raw, size_bucket="md", gap_above=None,
                        is_top_of_page=False, is_centered=False, caps=None,
                        indent_bucket=0, relative_font_ratio=1.0)


def _stage6(text):
    return RankedCandidate(candidate=Candidate(adapter="prose", request_id="r", text=text), score=1.0)


def _build_case(rng):
    n = rng.randint(1, 12)
    regions, top, fbs = [], {}, []
    for idx in range(n):
        fbs.append(_fb(f"t{idx}"))
        if rng.random() < 0.5:
            lvl = rng.randint(1, 6)
            regions.append(Region(kind="heading", feature_block_indices=(idx,),
                                  payload={"text": f"H{idx}", "level_hint": lvl}))
            top[idx] = _stage6(f"<h2>H{idx}</h2>")
        else:
            regions.append(Region(kind="paragraph", feature_block_indices=(idx,),
                                  payload={"text": f"p{idx}"}))
            top[idx] = _stage6(f"<p>p{idx}</p>")
    return regions, top, fbs


def test_levels_parity_builder_vs_subtask2():
    """The builder's levels equal the legacy normalize output for randomized
    level_hint sequences (the pure ownership move)."""
    rng = random.Random(7)
    for _ in range(50):
        regions, _top, fbs = _build_case(rng)
        html = [f'<h2 id="h{i}">x</h2>' if r.kind == "heading" else f"<p>{i}</p>"
                for i, r in enumerate(regions)]
        tree = build_containment_tree(regions, fbs, region_html=html)
        heading_indices = [i for i, r in enumerate(regions) if r.kind == "heading"]
        raw = [regions[i].payload["level_hint"] for i in heading_indices]
        legacy = normalize_heading_levels(raw)
        builder = [tree.levels[i] for i in heading_indices]
        assert builder == legacy


def test_assemble_byte_identical_containment_on_vs_off(monkeypatch):
    """Sub-task 2 apply-only is byte-parity: gold-shell-off assembly is
    byte-identical whether the builder or the legacy path computed levels."""
    rng = random.Random(11)
    for _ in range(20):
        regions, top, fbs = _build_case(rng)
        monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
        monkeypatch.setenv("SEMANTIK_CONTAINMENT", "1")
        on = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True)).html
        monkeypatch.setenv("SEMANTIK_CONTAINMENT", "0")
        off = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True)).html
        assert on == off


# ---------------------------------------------------------------------------
# Structural-drift diagnostic
# ---------------------------------------------------------------------------


def _doc_with_region_html(region_html):
    return types.SimpleNamespace(sub_task_log={"region_html": list(region_html)})


def test_drift_counter_zero_when_structural_levels_preserved():
    doc = _doc_with_region_html(['<h2 id="a">A</h2>', "<p>x</p>", '<h3 id="b">B</h3>'])
    new_html = '<section><h2 id="a">A</h2><p>x</p><h3 id="b">B</h3></section>'
    assert _count_structural_heading_drift(doc, new_html) == 0


def test_drift_counter_detects_structural_level_change():
    doc = _doc_with_region_html(['<h2 id="a">A</h2>', '<h3 id="b">B</h3>'])
    # The doc pass re-leveled the structural heading id="b" from h3 -> h4.
    new_html = '<h2 id="a">A</h2><h4 id="b">B</h4>'
    assert _count_structural_heading_drift(doc, new_html) == 1


def test_drift_ignores_embedded_headings():
    # An embedded <h6> (no structural region_html id) is silent normal repair.
    doc = _doc_with_region_html(['<h2 id="a">A</h2>'])
    new_html = '<h2 id="a">A</h2><p>body<h6>EXAMPLE</h6></p>'
    assert _count_structural_heading_drift(doc, new_html) == 0


def test_drift_zero_on_clean_assembled_doc(monkeypatch):
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)  # default ON
    fbs = [_fb("H1"), _fb("body"), _fb("H2")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"text": "H1", "level_hint": 1}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "body"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"text": "H2", "level_hint": 2}),
    ]
    top = {0: _stage6("<h2>H1</h2>"), 1: _stage6("<p>body</p>"), 2: _stage6("<h2>H2</h2>")}
    doc = assemble_document(top, regions, fbs, config=AssemblerConfig(skip_gap_fill=True))
    assert doc.sub_task_log.get("heading_contiguity_structural_drift") == 0
