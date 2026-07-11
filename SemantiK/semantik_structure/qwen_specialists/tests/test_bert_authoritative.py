"""Phase 5 — SEMANTIK_BERT_AUTHORITATIVE precedence flip + band-aid retirement.

Covers the flag-ON behaviour: the council Structure head's ``structural_role``
argmax becomes the Region kind (authoritative), ``role_confidence`` is stamped,
the Stage-5d reviewer is demoted to a confidence-gated span adjuster (the
unconditional code_block/table blanket + heading-always-in-scope are retired),
clean_structure sub-pass B (pedagogical demotion) is skipped, and a pedagogical
kind routes through the gold semantic-class wrapper instead of the
``<p class="…">`` band-aid. The flag-OFF byte-stability is asserted by the
existing reviewer / cascade / clean-structure suites (unchanged).
"""
from __future__ import annotations

from types import SimpleNamespace

from semantik_structure.assembler.fallbacks import _css_class_attr
from semantik_structure.assembler.gold_shell_markup import _wrap_semantic_class
from semantik_structure.qwen_specialists.deterministic_structure import clean_structure
from semantik_structure.qwen_specialists.reviewer import (
    _authoritative_dispatch_gate,
    _content_block_dispatch_gate,
    run_structure_review,
)
from semantik_structure.structure_graph import (
    Region,
    _apply_bert_authoritative_kinds,
    resolve_bert_authoritative,
)

from semantik_structure.qwen_specialists.tests.test_reviewer import (
    _ScriptedRuntime,
    _fb,
    _verdict_json,
)


def _state(heads: dict[str, dict[int, tuple[list[str], list[float]]]]):
    """Minimal CouncilState exposing the named structure heads per FB index."""
    signals = []
    for head_name, by_fb in heads.items():
        for fb_idx, (labels, confs) in by_fb.items():
            signals.append(
                SimpleNamespace(
                    head_name=head_name,
                    region_id=fb_idx,
                    top_k_labels=list(labels),
                    top_k_confidences=list(confs),
                )
            )
    return SimpleNamespace(outputs={"structure": SimpleNamespace(signals=signals)})


# ---------------------------------------------------------------------------
# Resolver (default OFF, parse-with-fallback).
# ---------------------------------------------------------------------------


def test_resolver_default_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BERT_AUTHORITATIVE", raising=False)
    assert resolve_bert_authoritative() is False


def test_resolver_truthy(monkeypatch):
    for val in ("1", "true", "YES", "On"):
        monkeypatch.setenv("SEMANTIK_BERT_AUTHORITATIVE", val)
        assert resolve_bert_authoritative() is True


def test_resolver_garbage_off(monkeypatch):
    for val in ("0", "false", "", "banana"):
        monkeypatch.setenv("SEMANTIK_BERT_AUTHORITATIVE", val)
        assert resolve_bert_authoritative() is False


# ---------------------------------------------------------------------------
# Precedence flip — role argmax -> Region.kind + role_confidence stamp.
# ---------------------------------------------------------------------------


def test_role_argmax_becomes_region_kind():
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "x"})]
    state = _state({"structural_role": {0: (["code_block", "paragraph"], [0.9, 0.05])}})
    out = _apply_bert_authoritative_kinds(regions, state)
    assert out[0].kind == "code_block"
    assert out[0].payload["role_confidence"] == 0.9


def test_role_confidence_stamped_even_when_kind_unchanged():
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "x"})]
    state = _state({"structural_role": {0: (["paragraph", "list_item"], [0.7, 0.2])}})
    out = _apply_bert_authoritative_kinds(regions, state)
    assert out[0].kind == "paragraph"
    assert out[0].payload["role_confidence"] == 0.7


def test_list_item_role_maps_to_list_kind():
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "first item"})]
    state = _state({"structural_role": {0: (["list_item"], [0.8])}})
    out = _apply_bert_authoritative_kinds(regions, state)
    assert out[0].kind == "list"


def test_definition_roles_map_to_definition_list():
    for role in ("definition_term", "definition_def"):
        regions = [Region(kind="paragraph", feature_block_indices=(0,),
                          payload={"text": "term"})]
        state = _state({"structural_role": {0: ([role], [0.8])}})
        out = _apply_bert_authoritative_kinds(regions, state)
        assert out[0].kind == "definition_list"


def test_heading_role_never_promotes_a_content_region():
    # Heading membership stays deterministic (is_heading) — the role flip never
    # promotes a content region TO heading even if structural_role argmax says so.
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "x"})]
    state = _state({"structural_role": {0: (["heading"], [0.95])}})
    out = _apply_bert_authoritative_kinds(regions, state)
    assert out[0].kind == "paragraph"
    assert out[0].payload["role_confidence"] == 0.95


def test_passthrough_region_skipped():
    # source_region_id set -> Stage-4 table/math passthrough; never re-typed.
    regions = [Region(kind="table", feature_block_indices=(0,),
                      payload={"cell_grid": []}, source_region_id=3)]
    state = _state({"structural_role": {0: (["paragraph"], [0.9])}})
    out = _apply_bert_authoritative_kinds(regions, state)
    assert out[0].kind == "table"
    assert "role_confidence" not in (out[0].payload or {})


def test_missing_signal_leaves_region_untouched():
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "x"})]
    out = _apply_bert_authoritative_kinds(regions, _state({}))
    assert out[0].kind == "paragraph"
    assert "role_confidence" not in (out[0].payload or {})


# ---------------------------------------------------------------------------
# Pedagogical kind -> semantic_class -> gold wrapper (not the <p class> band-aid).
# ---------------------------------------------------------------------------


def test_pedagogical_role_stamps_semantic_class():
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "EXAMPLE 1"})]
    state = _state({
        "structural_role": {0: (["paragraph"], [0.8])},
        "pedagogical_role": {0: (["example_open"], [0.9])},
    })
    out = _apply_bert_authoritative_kinds(regions, state)
    assert out[0].payload["semantic_class"] == "worked_example"


def test_pedagogical_none_stamps_no_semantic_class():
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "ordinary body"})]
    state = _state({
        "structural_role": {0: (["paragraph"], [0.8])},
        "pedagogical_role": {0: (["none"], [0.95])},
    })
    out = _apply_bert_authoritative_kinds(regions, state)
    assert "semantic_class" not in out[0].payload


def test_pedagogical_kind_routes_through_wrap_not_css_class_attr(monkeypatch):
    # A pedagogical region carries semantic_class -> the gold wrapper boxes it,
    # and the <p class="…"> band-aid is retired (returns "" under the flag).
    region = Region(kind="paragraph", feature_block_indices=(0,),
                    payload={"semantic_class": "worked_example",
                             "css_class": "pedagogy-example", "text": "EXAMPLE 1"})
    wrapped = _wrap_semantic_class("<p>EXAMPLE 1</p>", region, doc_ids=set())
    assert "worked-example" in wrapped  # routed through the gold container
    monkeypatch.setenv("SEMANTIK_BERT_AUTHORITATIVE", "1")
    assert _css_class_attr(region) == ""  # band-aid retired


def test_css_class_attr_kept_when_flag_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BERT_AUTHORITATIVE", raising=False)
    region = Region(kind="paragraph", feature_block_indices=(0,),
                    payload={"css_class": "pedagogy-example"})
    assert _css_class_attr(region) == ' class="pedagogy-example"'


# ---------------------------------------------------------------------------
# Reviewer dispatch gate — adjuster predicate (tau / margin / cross-head).
# ---------------------------------------------------------------------------


def _para(idx: int) -> Region:
    return Region(kind="paragraph", feature_block_indices=(idx,),
                  payload={"text": "body"})


def test_gate_dispatches_below_tau():
    region = _para(0)
    state = _state({"structural_role": {0: (["paragraph", "list_item"], [0.4, 0.1])}})
    assert _authoritative_dispatch_gate(region, state, []) is True


def test_gate_dispatches_low_margin():
    # top-1 above tau but the top1-top2 margin is narrow -> dispatch.
    region = _para(0)
    state = _state({"structural_role": {0: (["paragraph", "list_item"], [0.55, 0.45])}})
    assert _authoritative_dispatch_gate(region, state, []) is True


def test_gate_dispatches_cross_head_conflict():
    # is_heading votes heading; structural_role argmax is paragraph -> conflict.
    region = _para(0)
    state = _state({
        "structural_role": {0: (["paragraph", "list_item"], [0.9, 0.02])},
        "is_heading": {0: (["heading", "not_heading"], [0.9, 0.1])},
    })
    assert _authoritative_dispatch_gate(region, state, []) is True


def test_gate_trusts_confident_consistent_span():
    region = _para(0)
    state = _state({
        "structural_role": {0: (["paragraph", "list_item"], [0.9, 0.02])},
        "is_heading": {0: (["not_heading", "heading"], [0.95, 0.05])},
    })
    assert _authoritative_dispatch_gate(region, state, []) is False


def test_gate_missing_state_dispatches():
    assert _authoritative_dispatch_gate(_para(0), None, []) is True


# ---------------------------------------------------------------------------
# Unconditional code_block/table blanket retired under the flag.
# ---------------------------------------------------------------------------


def test_unconditional_blanket_retired(monkeypatch):
    # A HIGH-confidence code_block is dispatched UNCONDITIONALLY in legacy mode
    # but TRUSTED (no prompt) under the authoritative flag.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    region = Region(kind="code_block", feature_block_indices=(0,),
                    payload={"text": "x = compute(y)"})
    state = _state({"structural_role": {0: (["code_block", "paragraph"], [0.95, 0.01])}})

    monkeypatch.delenv("SEMANTIK_BERT_AUTHORITATIVE", raising=False)
    assert _content_block_dispatch_gate(region, state, []) is True  # legacy blanket

    monkeypatch.setenv("SEMANTIK_BERT_AUTHORITATIVE", "1")
    assert _content_block_dispatch_gate(region, state, []) is False  # retired


def test_heading_gated_under_flag(monkeypatch):
    # Flag ON: a confident, conflict-free heading is NOT dispatched (the
    # heading-always-in-scope blanket is narrowed to the confidence gate).
    monkeypatch.setenv("SEMANTIK_BERT_AUTHORITATIVE", "1")
    fbs = [_fb("Chapter 1")]
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 1, "text": "Chapter 1"})]
    state = _state({
        "structural_role": {0: (["heading", "paragraph"], [0.95, 0.01])},
        "is_heading": {0: (["heading", "not_heading"], [0.95, 0.05])},
    })
    rt = _ScriptedRuntime([])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert rt.batch_calls == []  # confident heading not re-judged
    assert out == regions


def test_uncertain_heading_dispatched_under_flag(monkeypatch):
    # Flag ON: a low-confidence heading IS still dispatched (the adjuster fires).
    monkeypatch.setenv("SEMANTIK_BERT_AUTHORITATIVE", "1")
    fbs = [_fb("Maybe a heading")]
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 2, "text": "Maybe a heading"})]
    state = _state({"structural_role": {0: (["heading"], [0.30])}})
    rt = _ScriptedRuntime([_verdict_json(0, verdict="ok")])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 1


# ---------------------------------------------------------------------------
# clean_structure sub-pass B (pedagogical demotion) retired under the flag.
# ---------------------------------------------------------------------------


def _peda_heading_case():
    fbs = [_fb("EXAMPLE 1")]
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 4, "text": "EXAMPLE 1"})]
    return regions, fbs


def test_subpass_b_runs_when_flag_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BERT_AUTHORITATIVE", raising=False)
    regions, fbs = _peda_heading_case()
    out, diag = clean_structure(regions, fbs)
    assert diag["pedagogical_demoted"] == 1
    assert out[0].kind == "paragraph"
    assert out[0].payload.get("css_class") == "pedagogy-example"


def test_subpass_b_retired_when_flag_on(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BERT_AUTHORITATIVE", "1")
    regions, fbs = _peda_heading_case()
    out, diag = clean_structure(regions, fbs)
    assert diag["pedagogical_demoted"] == 0
    assert out[0].kind == "heading"  # head emits the kind directly; no demotion
    assert "css_class" not in (out[0].payload or {})
