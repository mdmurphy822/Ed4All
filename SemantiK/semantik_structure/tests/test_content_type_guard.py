"""Content-type sanity guard — code_block / definition_list false-positive veto.

Pins the deterministic, domain-agnostic ``SEMANTIK_CONTENT_TYPE_GUARD`` guard
(``structure_graph._looks_like_code`` / ``_looks_like_definition_list``, AND-ed
into the Pass-6 ``_is_code`` / ``_is_definition`` predicates) that stops the
council BERT from shipping non-code as ``code_block`` and running-prose
fragments as ``definition_list``. A vetoed region falls through to a
text-preserving 1-FB paragraph (safe demotion — no FB lost).

Predicate unit tests use the confirmed cross-genre eval false-positives; the
integration test drives the REAL ``build_structure_graph`` path on a synthetic,
GPU-free CouncilState (mirroring ``test_structure_reading_order.py``).
"""

from __future__ import annotations

import pytest

from semantik_structure.council.cross_reranker import arbitrate
from semantik_structure.council.types import BertOutput, CouncilState, TypedSignal
from semantik_structure.structure_graph import (
    _looks_like_code,
    _looks_like_definition_list,
    build_structure_graph,
    resolve_content_type_guard,
)
from semantik_structure.types import FeatureBlock, RawBlock

_FLAG = "SEMANTIK_CONTENT_TYPE_GUARD"


# ---------------------------------------------------------------------------
# Phase 0 — _looks_like_code predicate.
# ---------------------------------------------------------------------------

# Shape-preserving SYNTHETIC analogs of confirmed cross-genre false-positives
# (names/digits perturbed; the structural shape the predicate keys on is
# preserved) — NONE contain code.
_CODE_FALSE_POSITIVES = [
    # wiki-style numeric TABLE row
    "Centerville 1,234,567 1,102,845 86.49% | 187,663 24.78% | 84,007",
    # wiki-style TABLE header
    "Age group | 2006 | 2020",
    # GPO-style printer furniture (token grammar generic to every FR page)
    "o r VerDate Sep<11>2014 16:04 Jan 02, 2013 Jkt 000000 PO 00000 "
    "Frm 00001 Fmt 8010 Sfmt 8010",
    # bare label
    "authorize:",
]

# Tricky prose that a naive symbol/paren-density detector would regress.
_CODE_TRICKY_PROSE = [
    "See section 3(a)(2) for details; the rule applies.",
    "Revenue increased 5% (up from 3%).",
    "Age group | 2006 | 2020",
]

# Real code samples — MUST look like code.
_CODE_REAL = [
    "def foo(x):\n    return x+1",
    "for(int i=0;i<n;i++){s+=a[i];}",
    "import numpy as np",
    "<div class='x'><span>hi</span></div>",
    "  config.timeout = 30\n  config.retries = 3",
]


@pytest.mark.parametrize("text", _CODE_FALSE_POSITIVES)
def test_looks_like_code_false_on_false_positives(text):
    assert _looks_like_code(text) is False, f"non-code accepted as code: {text!r}"


@pytest.mark.parametrize("text", _CODE_TRICKY_PROSE)
def test_looks_like_code_false_on_tricky_prose(text):
    assert _looks_like_code(text) is False, f"prose accepted as code: {text!r}"


@pytest.mark.parametrize("text", _CODE_REAL)
def test_looks_like_code_true_on_real_code(text):
    assert _looks_like_code(text) is True, f"real code rejected: {text!r}"


def test_looks_like_code_empty_is_false():
    assert _looks_like_code("") is False
    assert _looks_like_code("   \n  ") is False


# ---------------------------------------------------------------------------
# Phase 1 — _looks_like_definition_list predicate.
# ---------------------------------------------------------------------------

# Synthetic regulatory-register-style running-prose fragments (shape of the
# confirmed false-positives: lowercase mid-sentence start, comma-truncated,
# no term/definition delimiter) — NOT a definition list.
_DEF_FALSE_POSITIVES = [
    "processing of invoices, arranging transport",
    "activities for the maintenance of water, power,",
]

# Genuine term-definition shapes.
_DEF_REAL = [
    "Kinetic energy — the energy an object has due to its motion",
    "Kinetic energy: the energy an object has due to its motion",
    "Photosynthesis: the process by which plants convert light to energy",
]


@pytest.mark.parametrize("text", _DEF_FALSE_POSITIVES)
def test_looks_like_definition_false_on_prose_fragments(text):
    assert _looks_like_definition_list(text) is False, (
        f"running-prose fragment accepted as definition list: {text!r}"
    )


@pytest.mark.parametrize("text", _DEF_REAL)
def test_looks_like_definition_true_on_genuine_terms(text):
    assert _looks_like_definition_list(text) is True, (
        f"genuine term-definition rejected: {text!r}"
    )


def test_looks_like_definition_conservative_on_bare_term_and_body():
    # A split ``definition_term`` / ``definition_def`` FB (bare short term OR a
    # capitalized definition-body sentence) is NOT vetoed — no regression on a
    # genuine two-FB definition list.
    assert _looks_like_definition_list("Kinetic energy") is True
    assert (
        _looks_like_definition_list(
            "The energy an object has due to its motion."
        )
        is True
    )


def test_looks_like_definition_empty_is_false():
    assert _looks_like_definition_list("") is False


# ---------------------------------------------------------------------------
# Phase 2 — flag resolver semantics.
# ---------------------------------------------------------------------------


def test_resolver_default_off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert resolve_content_type_guard() is False


@pytest.mark.parametrize("val", ["1", "true", "YES", "On"])
def test_resolver_truthy(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    assert resolve_content_type_guard() is True


@pytest.mark.parametrize("val", ["0", "false", "off", "garbage", ""])
def test_resolver_falsey_and_garbage(monkeypatch, val):
    monkeypatch.setenv(_FLAG, val)
    assert resolve_content_type_guard() is False


# ---------------------------------------------------------------------------
# Phase 3 — integration over the REAL build_structure_graph path.
# ---------------------------------------------------------------------------


def _fb(text: str) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=1,
        bbox=(0.0, 0.0, 100.0, 20.0),
        page_width=200.0,
        page_height=300.0,
        font_size=11.0,
    )
    try:
        raw.is_bold = False  # type: ignore[attr-defined]
    except Exception:
        pass
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
        is_image=False,
    )


def _role_signals(idx: int, role: str) -> list[TypedSignal]:
    """Confident non-heading FB with ``structural_role`` top-1 == ``role``."""
    return [
        TypedSignal("is_heading", idx, ["body", "heading"], [0.99, 0.01]),
        TypedSignal("structural_role", idx, [role, "paragraph"], [0.92, 0.08]),
    ]


def _build_code_graph():
    """FB0 = a wiki-style numeric table row the council mis-typed code_block;
    FB1 = a genuine C for-loop the council typed code_block. No GPU / model."""
    feature_blocks = [
        _fb("Centerville 1,234,567 1,102,845 86.49% | 187,663 24.78% | 84,007"),
        _fb("for(int i=0;i<n;i++){s+=a[i];}"),
    ]
    struct_signals: list[TypedSignal] = []
    struct_signals += _role_signals(0, "code_block")
    struct_signals += _role_signals(1, "code_block")
    state = CouncilState(
        outputs={"structure": BertOutput("structure", struct_signals)}
    )
    decisions = arbitrate(state, [])
    return state, feature_blocks, [], decisions


def _kind_of_fb(regions, fb_idx: int) -> str:
    for r in regions:
        if fb_idx in r.feature_block_indices:
            return r.kind
    raise AssertionError(f"FB {fb_idx} not owned by any region")


def _assert_full_coverage(regions, n_fb: int):
    owned = [j for r in regions for j in r.feature_block_indices]
    assert sorted(owned) == list(range(n_fb)), f"FB coverage broken: {sorted(owned)}"
    assert len(owned) == len(set(owned)), "an FB is owned by more than one region"


def test_code_guard_off_keeps_council_code_block(monkeypatch):
    """Flag OFF -> byte-identical: the mis-typed table row STAYS code_block."""
    monkeypatch.setenv(_FLAG, "0")
    state, fbs, cands, decs = _build_code_graph()
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    assert _kind_of_fb(regions, 0) == "code_block"
    assert _kind_of_fb(regions, 1) == "code_block"


def test_code_guard_on_demotes_non_code_keeps_real_code(monkeypatch):
    """Flag ON -> the non-code table row demotes to paragraph (text preserved,
    no FB lost) while the genuine for-loop stays code_block."""
    monkeypatch.setenv(_FLAG, "on")
    state, fbs, cands, decs = _build_code_graph()
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    assert _kind_of_fb(regions, 0) == "paragraph", "non-code row was not demoted"
    assert _kind_of_fb(regions, 1) == "code_block", "genuine code was wrongly demoted"


def _build_definition_graph():
    """FB0 = a regulatory-register-style running-prose fragment the council
    mis-typed definition_term; FB1 = a genuine ``Term: definition`` line."""
    feature_blocks = [
        _fb("processing of invoices, arranging transport"),
        _fb("Kinetic energy: the energy an object has due to its motion"),
    ]
    struct_signals: list[TypedSignal] = []
    struct_signals += _role_signals(0, "definition_term")
    struct_signals += _role_signals(1, "definition_term")
    state = CouncilState(
        outputs={"structure": BertOutput("structure", struct_signals)}
    )
    decisions = arbitrate(state, [])
    return state, feature_blocks, [], decisions


def test_definition_guard_off_keeps_council_definition_list(monkeypatch):
    """Flag OFF -> byte-identical: both FBs stay definition_list."""
    monkeypatch.setenv(_FLAG, "0")
    state, fbs, cands, decs = _build_definition_graph()
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    assert _kind_of_fb(regions, 0) == "definition_list"
    assert _kind_of_fb(regions, 1) == "definition_list"


def test_definition_guard_on_demotes_prose_fragment(monkeypatch):
    """Flag ON -> the running-prose fragment demotes to paragraph (text
    preserved) while the genuine term-definition stays definition_list."""
    monkeypatch.setenv(_FLAG, "on")
    state, fbs, cands, decs = _build_definition_graph()
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    assert _kind_of_fb(regions, 0) == "paragraph", "prose fragment was not demoted"
    assert _kind_of_fb(regions, 1) == "definition_list", (
        "genuine definition was wrongly demoted"
    )


# ---------------------------------------------------------------------------
# ITEM6 Phase 2 — guard runner-up fallback (D3a). A vetoed FB consults the
# council's full structural_role distribution and mints the runner-up kind
# when it clears that kind's existing floor + shape predicate, instead of the
# blanket paragraph demotion. heading is never minted; flag-off byte-identical.
# ---------------------------------------------------------------------------


def _region_for_fb(regions, fb_idx: int):
    for r in regions:
        if fb_idx in r.feature_block_indices:
            return r
    raise AssertionError(f"FB {fb_idx} not owned by any region")


def _runner_up_graph(text: str, top_label: str, runner_label: str, runner_conf: float):
    """FB0 = a block the council typed ``top_label`` (a guard-vetoed kind) with
    ``runner_label`` as the distribution runner-up; FB1 = filler prose so the
    graph is non-trivial. No GPU / model."""
    feature_blocks = [_fb(text), _fb("Ordinary filler prose sentence here.")]
    struct_signals: list[TypedSignal] = [
        TypedSignal("is_heading", 0, ["body", "heading"], [0.99, 0.01]),
        TypedSignal(
            "structural_role",
            0,
            [top_label, runner_label, "paragraph"],
            [0.70, runner_conf, max(0.0, 0.30 - runner_conf)],
        ),
        TypedSignal("is_heading", 1, ["body", "heading"], [0.99, 0.01]),
        TypedSignal("structural_role", 1, ["paragraph", "list_item"], [0.9, 0.1]),
    ]
    state = CouncilState(outputs={"structure": BertOutput("structure", struct_signals)})
    decisions = arbitrate(state, [])
    return state, feature_blocks, [], decisions


_WIKI_ROW = "Centerville 1,234,567 1,102,845 86.49% | 187,663 24.78% | 84,007"


def test_runner_up_blockquote_from_code_veto(monkeypatch):
    monkeypatch.setenv(_FLAG, "on")
    # code_block top-1 vetoed (non-code); runner-up blockquote@0.45 >= 0.4 floor.
    state, fbs, cands, decs = _runner_up_graph(_WIKI_ROW, "code_block", "blockquote", 0.45)
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    reg = _region_for_fb(regions, 0)
    assert reg.kind == "blockquote"
    assert reg.provenance.get("pass") == "guard_runner_up"
    assert reg.provenance.get("vetoed") == "code_block"
    assert reg.provenance.get("runner_up") == "blockquote"


def test_runner_up_blockquote_below_floor_stays_paragraph(monkeypatch):
    monkeypatch.setenv(_FLAG, "on")
    # runner-up blockquote@0.30 < 0.4 floor -> paragraph fallback.
    state, fbs, cands, decs = _runner_up_graph(_WIKI_ROW, "code_block", "blockquote", 0.30)
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    reg = _region_for_fb(regions, 0)
    assert reg.kind == "paragraph"
    assert reg.provenance.get("pass") != "guard_runner_up"


def test_runner_up_definition_requires_shape(monkeypatch):
    monkeypatch.setenv(_FLAG, "on")
    # runner-up definition_term but the lowercase prose fragment has no
    # term-definition shape (_looks_like_definition_list False) -> paragraph.
    state, fbs, cands, decs = _runner_up_graph(
        "processing of invoices, arranging transport",
        "code_block",
        "definition_term",
        0.45,
    )
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    assert _kind_of_fb(regions, 0) == "paragraph"


def test_runner_up_list_requires_marker(monkeypatch):
    monkeypatch.setenv(_FLAG, "on")
    # list_item@0.6 but no list marker -> paragraph.
    state, fbs, cands, decs = _runner_up_graph(_WIKI_ROW, "code_block", "list_item", 0.6)
    regions = build_structure_graph(state, fbs, cands, decs)
    assert _kind_of_fb(regions, 0) == "paragraph"
    # With a real ordered-list marker -> 1-item list, Pass-4 payload shape.
    state, fbs, cands, decs = _runner_up_graph(
        "1. Combine like terms before isolating the variable.",
        "code_block",
        "list_item",
        0.6,
    )
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    reg = _region_for_fb(regions, 0)
    assert reg.kind == "list"
    assert reg.provenance.get("pass") == "guard_runner_up"
    assert reg.payload["ordered"] is True
    assert reg.payload["items"] == [
        {"fb_index": 0, "text": "1. Combine like terms before isolating the variable.", "depth": 0}
    ]


def test_runner_up_paragraph_byte_identical_to_legacy(monkeypatch):
    # Runner-up is paragraph -> _guard_runner_up_region returns None, so the
    # normal paragraph fallback mints it: identical to pre-change guard demote.
    monkeypatch.setenv(_FLAG, "on")
    state, fbs, cands, decs = _runner_up_graph(_WIKI_ROW, "code_block", "paragraph", 0.28)
    regions = build_structure_graph(state, fbs, cands, decs)
    _assert_full_coverage(regions, len(fbs))
    reg = _region_for_fb(regions, 0)
    assert reg.kind == "paragraph"
    assert reg.provenance.get("pass") != "guard_runner_up"


def test_runner_up_never_mints_heading(monkeypatch):
    monkeypatch.setenv(_FLAG, "on")
    # Even a high-confidence heading runner-up -> paragraph (heading membership
    # stays Pass-2-owned; never minted in the guard fallback).
    state, fbs, cands, decs = _runner_up_graph(_WIKI_ROW, "code_block", "heading", 0.29)
    regions = build_structure_graph(state, fbs, cands, decs)
    reg = _region_for_fb(regions, 0)
    assert reg.kind != "heading"
    assert reg.kind == "paragraph"


def test_runner_up_empty_without_guard(monkeypatch):
    # Guard OFF -> guard_vetoed stays empty -> the wiki row keeps its council
    # code_block (byte-identical legacy), no runner-up path taken.
    monkeypatch.setenv(_FLAG, "0")
    state, fbs, cands, decs = _runner_up_graph(_WIKI_ROW, "code_block", "blockquote", 0.45)
    regions = build_structure_graph(state, fbs, cands, decs)
    reg = _region_for_fb(regions, 0)
    assert reg.kind == "code_block"
    assert reg.provenance.get("pass") != "guard_runner_up"
