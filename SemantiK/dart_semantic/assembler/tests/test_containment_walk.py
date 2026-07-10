"""ITEM4 Phase 2 — assembler walks the containment tree.

The gold-shell branch WALKS the tree (section grouping, unit boxes, caption
nesting, multi-pair <dl>) instead of the legacy _group_regions_into_sections +
gold-absorb. The absorb resolvers short-circuit under containment; the legacy
absorb is reachable only under SEMANTIK_CONTAINMENT=0 (the byte-identical OFF
lever). CPU-only, no model load, no GPU.
"""

from __future__ import annotations

import re

from dart_semantic.assembler.api import AssemblerConfig, assemble_document
from dart_semantic.assembler.shell import (
    resolve_gold_absorb_mode,
    resolve_narrow_table_absorb_mode,
)
from dart_semantic.council.types import BertOutput, CouncilState, TypedSignal
from dart_semantic.soft_reranker.types import RankedCandidate
from dart_semantic.qwen_specialists.types import Candidate
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


_WE_DIV_RE = re.compile(r'<div class="algorithm worked-example"[^>]*>(.*?)</div>', re.DOTALL)


def _fb(text: str, page: int = 1) -> FeatureBlock:
    raw = RawBlock(text=text, page=page, bbox=(0.0, 0.0, 10.0, 10.0),
                   page_width=100.0, page_height=100.0)
    return FeatureBlock(raw=raw, size_bucket="md", gap_above=None,
                        is_top_of_page=False, is_centered=False, caps=None,
                        indent_bucket=0, relative_font_ratio=1.0)


def _region(idx, semantic_class=None, *, kind="paragraph", css_class=None,
            caption_fb_index=None, source_region_id=None, fbs=None, text=None):
    payload = {"text": text if text is not None else f"body {idx}"}
    if semantic_class is not None:
        payload["semantic_class"] = semantic_class
    if css_class is not None:
        payload["css_class"] = css_class
    if caption_fb_index is not None:
        payload["caption_fb_index"] = caption_fb_index
    return Region(kind=kind, feature_block_indices=tuple(fbs or (idx,)),
                  payload=payload, source_region_id=source_region_id)


def _heading_region(idx, level=1):
    return Region(kind="heading", feature_block_indices=(idx,),
                  payload={"text": f"H{idx}", "level_hint": level})


def _stage6(text):
    return RankedCandidate(candidate=Candidate(adapter="prose", request_id="r", text=text), score=1.0)


def _assemble(top, regions, fbs, *, council_state=None):
    return assemble_document(top, regions, fbs, council_state=council_state,
                             config=AssemblerConfig(skip_gap_fill=True))


# ---------------------------------------------------------------------------
# Unit boxing via the tree
# ---------------------------------------------------------------------------


def test_walk_boxes_unit_once(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)  # default ON
    fbs = [_fb("ex"), _fb("sol"), _fb("body"), _fb("h2")]
    regions = [
        _region(0, "worked_example"),
        _region(1, css_class="pedagogy-solution"),
        _region(2, None),
        _heading_region(3, level=1),
    ]
    top = {
        0: _stage6("<p>EXAMPLE 1 solve</p>"),
        1: _stage6("<p>Solution body</p>"),
        2: _stage6("<p>step value</p>"),
        3: _stage6("<h2>Next</h2>"),
    }
    html = _assemble(top, regions, fbs).html
    m = _WE_DIV_RE.search(html)
    assert m is not None, html
    box = m.group(1)
    assert "Solution body" in box and "step value" in box
    assert "Next" not in box  # heading is a boundary
    # each absorbed fragment appears exactly once (splice-key safety)
    for frag in ("<p>Solution body</p>", "<p>step value</p>"):
        assert html.count(frag) == 1, frag


def test_non_gold_flat_concat_unchanged(monkeypatch):
    # gold OFF + containment ON == gold OFF + containment OFF (both flat concat).
    fbs = [_fb("a"), _fb("b"), _fb("c")]
    regions = [_region(0), _region(1, "worked_example"), _region(2)]
    top = {0: _stage6("<p>a</p>"), 1: _stage6("<p>b</p>"), 2: _stage6("<p>c</p>")}
    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    monkeypatch.setenv("SEMANTIK_CONTAINMENT", "1")
    on = _assemble(top, regions, fbs).html
    monkeypatch.setenv("SEMANTIK_CONTAINMENT", "0")
    off = _assemble(top, regions, fbs).html
    assert on == off


# ---------------------------------------------------------------------------
# Caption nesting + suppression
# ---------------------------------------------------------------------------


def test_caption_nested_once(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)  # default ON
    fbs = [_fb("img"), _fb("Figure 1: a cat")]
    regions = [
        _region(0, kind="figure", caption_fb_index=1, fbs=(0,)),
        _region(1, kind="paragraph", fbs=(1,), text="Figure 1: a cat"),
    ]
    top = {0: None, 1: None}  # fallback path
    html = _assemble(top, regions, fbs).html
    # The caption <p> ships exactly ONCE, nested inside the <figure>.
    assert html.count("<p>Figure 1: a cat</p>") == 1
    assert "<figcaption>" not in html  # duplicate figcaption suppressed
    fig = re.search(r"<figure>.*?</figure>", html, re.DOTALL)
    assert fig is not None and "<p>Figure 1: a cat</p>" in fig.group(0)


def test_caption_guard_falls_back_to_duplicate_legacy(monkeypatch):
    # Multi-FB owner with extra text -> guard fails -> no caption edge -> legacy
    # duplicate emit (figcaption present + the paragraph also renders).
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)
    fbs = [_fb("img"), _fb("Figure 2"), _fb("extra sentence")]
    regions = [
        _region(0, kind="figure", caption_fb_index=1, fbs=(0,)),
        _region(1, kind="paragraph", fbs=(1, 2), text="Figure 2 extra sentence"),
    ]
    top = {0: None, 1: None}
    html = _assemble(top, regions, fbs).html
    assert "<figcaption>" in html  # NOT suppressed (guard failed)


# ---------------------------------------------------------------------------
# dl multi-pair
# ---------------------------------------------------------------------------


def _council(role_by_fb):
    sigs = [TypedSignal(head_name="structural_role", region_id=fb,
                        top_k_labels=[role], top_k_confidences=[0.9])
            for fb, role in role_by_fb.items()]
    return CouncilState(outputs={"structure": BertOutput(bert_name="structure", signals=sigs)})


def test_dl_multi_pair_emit(monkeypatch):
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)  # default ON
    fbs = [_fb("Term A"), _fb("Def A"), _fb("Term B"), _fb("Def B")]
    regions = [_region(0, kind="definition_list", fbs=(0, 1, 2, 3))]
    top = {0: None}  # fallback -> fallback_definition_list
    council = _council({0: "definition_term", 1: "definition_def",
                        2: "definition_term", 3: "definition_def"})
    html = _assemble(top, regions, fbs, council_state=council).html
    assert html.count("<dt>") == 2 and html.count("<dd>") == 2
    assert "<dt>Term A</dt><dd>Def A</dd>" in html
    assert "<dt>Term B</dt><dd>Def B</dd>" in html


def test_dl_legacy_without_pairs(monkeypatch):
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)
    fbs = [_fb("Term"), _fb("Def one"), _fb("Def two")]
    regions = [_region(0, kind="definition_list", fbs=(0, 1, 2))]
    top = {0: None}
    html = _assemble(top, regions, fbs, council_state=None).html  # no stamp
    # legacy single-pair flatten: one <dt> + one <dd> (defs joined).
    assert html.count("<dt>") == 1 and html.count("<dd>") == 1


# ---------------------------------------------------------------------------
# Absorb retirement — resolvers short-circuit; OFF lever restores legacy
# ---------------------------------------------------------------------------


def test_absorb_resolvers_short_circuit_under_containment(monkeypatch):
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)  # default ON
    assert resolve_gold_absorb_mode() is False
    assert resolve_narrow_table_absorb_mode() is False
    # OFF lever: with containment explicitly off the legacy absorb re-engages.
    monkeypatch.setenv("SEMANTIK_CONTAINMENT", "0")
    assert resolve_gold_absorb_mode() is True


def test_narrow_table_absorb_short_circuited_table_nested(monkeypatch):
    # Narrow-table absorb subsumed: a trailing FB-adjacent passthrough table is
    # nested into the unit box via the tree (not the narrow absorb resolver).
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "1")
    monkeypatch.delenv("SEMANTIK_CONTAINMENT", raising=False)
    assert resolve_narrow_table_absorb_mode() is False
    fbs = [_fb("ex"), _fb("tbl")]
    regions = [
        _region(0, "worked_example", fbs=(0,)),
        _region(1, kind="table", source_region_id=7, fbs=(1,)),
    ]
    top = {0: _stage6("<p>EXAMPLE box</p>"), 1: _stage6("<table><tr><td>x</td></tr></table>")}
    html = _assemble(top, regions, fbs).html
    m = _WE_DIV_RE.search(html)
    assert m is not None, html
    assert "<table><tr" in m.group(1)  # table nested INSIDE the unit box
    assert html.count("<table><tr") == 1  # the data table ships exactly once


def test_off_lever_absorb_still_boxes_unit(monkeypatch):
    # SEMANTIK_CONTAINMENT=0 + GOLD_SHELL=1 + GOLD_ABSORB=1 reproduces the legacy
    # absorb: the worked-example box encloses its solution + body.
    monkeypatch.setenv("SEMANTIK_CONTAINMENT", "0")
    monkeypatch.setenv("SEMANTIK_GOLD_SHELL", "1")
    monkeypatch.setenv("SEMANTIK_GOLD_ABSORB", "1")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")
    fbs = [_fb("ex"), _fb("sol"), _fb("h2")]
    regions = [
        _region(0, "worked_example"),
        _region(1, css_class="pedagogy-solution"),
        _heading_region(2, level=1),
    ]
    top = {0: _stage6("<p>EXAMPLE</p>"), 1: _stage6("<p>Solution body</p>"), 2: _stage6("<h2>Next</h2>")}
    html = _assemble(top, regions, fbs).html
    m = _WE_DIV_RE.search(html)
    assert m is not None and "Solution body" in m.group(1)
