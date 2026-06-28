"""Stage-5e cascade-wiring acceptance — gate, audit section, byte-stability.

Drives the Stage-5e seam helpers in the SAME order ``run_full_cascade`` calls
them (gate -> resegment_blocks -> audit-section build -> result-dict insert)
WITHOUT pulling council/extraction/axe. Proves:

  * the gate (``resolve_block_resegment_mode``) is off by default / on truthy;
  * flag-ON: a deterministic merge re-partitions the region list AND the
    audit section is populated, with stable ids (lowest source id preserved);
  * flag-OFF: the result dict carries NO ``block_resegment`` key (byte-stable).

The cascade module pulls axe/playwright at import; we stub those for IMPORT
ONLY (mirrors ``test_cascade_stage5d.py``). The helpers under test never run
axe.
"""

from __future__ import annotations

import sys
import types


def _install_axe_stubs() -> None:
    if "axe_playwright_python" not in sys.modules:
        pkg = types.ModuleType("axe_playwright_python")
        sub = types.ModuleType("axe_playwright_python.sync_playwright")
        sub.Axe = object
        pkg.sync_playwright = sub
        sys.modules["axe_playwright_python"] = pkg
        sys.modules["axe_playwright_python.sync_playwright"] = sub
    if "playwright" not in sys.modules:
        pw = types.ModuleType("playwright")
        pw_sync = types.ModuleType("playwright.sync_api")
        pw_sync.sync_playwright = lambda *a, **k: None
        pw.sync_api = pw_sync
        sys.modules["playwright"] = pw
        sys.modules["playwright.sync_api"] = pw_sync


_install_axe_stubs()

from dart_semantic import cascade  # noqa: E402,F401  (import-loads the module)
from dart_semantic.qwen_specialists.block_resegment import (  # noqa: E402
    resegment_blocks,
    resolve_block_resegment_mode,
    resolve_unit_regroup_mode,
)
from dart_semantic.structure_graph import Region  # noqa: E402
from dart_semantic.types import FeatureBlock, RawBlock  # noqa: E402


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


def _region(kind, fb_indices, *, text=None, pages=None):
    payload = {}
    if text is not None:
        payload["text"] = text
    if pages is not None:
        payload["pages"] = pages
    return Region(kind=kind, feature_block_indices=tuple(fb_indices), payload=payload)


class _EmptyState:
    outputs = {}


# ---------------------------------------------------------------------------
# Gate.
# ---------------------------------------------------------------------------


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    assert resolve_block_resegment_mode() is False


def test_gate_on_when_truthy(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "1")
    assert resolve_block_resegment_mode() is True


# ---------------------------------------------------------------------------
# Flag-ON seam: resegment -> audit section build (cascade snippet).
# ---------------------------------------------------------------------------


def test_seam_flag_on_populates_audit_and_repartitions(monkeypatch):
    """The exact seam the cascade runs: gate -> resegment_blocks -> audit."""
    # Phase 6 gates the same-kind detectors inside resegment_blocks; enable the
    # flag so the deterministic merge fires.
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")
    fbs = [_fb("a clause that runs", page=1), _fb("over the page edge", page=2)]
    structure_regions = [
        _region("paragraph", [0], text="a clause that runs", pages=[1]),
        _region("paragraph", [1], text="over the page edge", pages=[2]),
    ]

    resegmented_regions, resegment_ops = resegment_blocks(
        structure_regions, fbs, _EmptyState(), runtime=None
    )

    # The cross-page continuation merge fired.
    assert len(resegmented_regions) == 1
    assert resegmented_regions[0].feature_block_indices == (0, 1)
    # Stable id: lowest source id preserved.
    assert min(resegmented_regions[0].feature_block_indices) == 0

    # Build the audit section exactly as the cascade snippet does.
    resegment_ops_audit = [
        {
            "op": op.op,
            "source_ids": list(op.source_ids),
            "origin": op.origin,
            "conservation_verified": True,
        }
        for op in resegment_ops
    ]
    assert resegment_ops_audit
    assert resegment_ops_audit[0]["op"] == "merge"
    assert resegment_ops_audit[0]["origin"] == "deterministic"
    assert resegment_ops_audit[0]["conservation_verified"] is True
    # Source ids are the merged regions' min FB indices.
    assert resegment_ops_audit[0]["source_ids"] == [0, 1]


def test_result_dict_omits_key_when_off():
    """The cascade only sets result['block_resegment'] when the audit is not
    None; an OFF run leaves the audit variable None -> key absent (byte-stable)."""
    resegment_ops_audit = None  # the cascade's flag-OFF state
    result = {"region_provenance": []}
    if resegment_ops_audit is not None:
        result["block_resegment"] = resegment_ops_audit
    assert "block_resegment" not in result


def test_result_dict_carries_key_when_on():
    resegment_ops_audit = [
        {"op": "split", "source_ids": [3], "origin": "deterministic",
         "conservation_verified": True}
    ]
    result = {"region_provenance": []}
    if resegment_ops_audit is not None:
        result["block_resegment"] = resegment_ops_audit
    assert result["block_resegment"] == resegment_ops_audit


# ---------------------------------------------------------------------------
# Phase 6 — widened cascade gate + flag-off byte-identity (the sharp edge).
# ---------------------------------------------------------------------------


def _stage5e_seam(structure_regions, fbs, state):
    """Replicate the cascade's Stage-5e gate + seam EXACTLY (the widened gate +
    audit-section build + result-key population), returning the result dict.

    Mirrors cascade.py: enter Stage-5e on
    ``resolve_block_resegment_mode() or resolve_unit_regroup_mode()``; the
    audit variable stays None when the gate is closed, so result['block_resegment']
    is omitted (byte-stable)."""
    resegment_ops_audit = None
    if resolve_block_resegment_mode() or resolve_unit_regroup_mode():
        regions, ops = resegment_blocks(structure_regions, fbs, state, runtime=None)
        structure_regions = regions
        resegment_ops_audit = [
            {
                "op": op.op,
                "source_ids": list(op.source_ids),
                "origin": op.origin,
                "conservation_verified": True,
            }
            for op in ops
        ]
    result = {"region_provenance": [], "n_regions": len(structure_regions)}
    if resegment_ops_audit is not None:
        result["block_resegment"] = resegment_ops_audit
    return result


def test_cascade_gate_widened_to_unit_regroup(monkeypatch):
    """The widened gate fires Stage-5e on SEMANTIK_UNIT_REGROUP alone (block
    resegment off) — and per-flag gating inside the driver keeps it regroup-only."""
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    assert (resolve_block_resegment_mode() or resolve_unit_regroup_mode()) is True


def test_cascade_gate_byte_identical_flag_off(monkeypatch):
    """LOAD-BEARING: with SEMANTIK_UNIT_REGROUP ABSENT vs explicitly OFF (both
    with block-resegment off), the full Stage-5e seam result dict is BYTE-
    IDENTICAL — the widened gate stays closed exactly as the legacy
    resolve_block_resegment_mode()-only gate, and no block_resegment key leaks."""
    fbs = [_fb("EXAMPLE 1"), _fb("solve x"), _fb("Solution factor")]

    def _build_regions():
        # Build a contiguous label->body unit that WOULD regroup if the gate
        # opened — proving the byte-identity is the GATE staying closed, not an
        # empty input.
        from dart_semantic.structure_graph import Region

        def _r(kind, idxs, text, css=None):
            payload = {"text": text}
            if css is not None:
                payload["css_class"] = css
            return Region(kind=kind, feature_block_indices=tuple(idxs), payload=payload)

        return [
            _r("paragraph", [0], "EXAMPLE 1", css="pedagogy-example"),
            _r("paragraph", [1], "solve x"),
            _r("paragraph", [2], "Solution factor", css="pedagogy-solution"),
        ]

    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)

    # Run A: SEMANTIK_UNIT_REGROUP entirely ABSENT.
    monkeypatch.delenv("SEMANTIK_UNIT_REGROUP", raising=False)
    result_absent = _stage5e_seam(_build_regions(), fbs, _EmptyState())

    # Run B: SEMANTIK_UNIT_REGROUP explicitly OFF.
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")
    result_off = _stage5e_seam(_build_regions(), fbs, _EmptyState())

    # Byte-identical result dicts, and NO block_resegment key in either.
    assert result_absent == result_off
    assert "block_resegment" not in result_absent
    assert "block_resegment" not in result_off
    # And the gate is closed in both (no resegment ran -> region count unchanged).
    assert result_absent["n_regions"] == 3


def test_cascade_gate_closed_when_both_flags_off(monkeypatch):
    """With BOTH resegment + regroup off the gate is byte-identical to a
    no-Stage-5e run (no key, regions untouched)."""
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_UNIT_REGROUP", raising=False)
    fbs = [_fb("EXAMPLE 1"), _fb("solve x")]
    from dart_semantic.structure_graph import Region

    regions = [
        Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "EXAMPLE 1", "css_class": "pedagogy-example"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "solve x"}),
    ]
    result = _stage5e_seam(regions, fbs, _EmptyState())
    assert "block_resegment" not in result
    assert result["n_regions"] == 2


# ---------------------------------------------------------------------------
# table-v2 Phase 4 — Stage-5e survival coverage (the table is NEVER folded).
#
# Option C's central claim at the cascade level: the SEMANTIK_UNIT_REGROUP_TABLE
# sub-flag makes ZERO Stage-5e change — the v1 passthrough STOP stays, so the
# trailing Solution table survives Stage-5e as a distinct
# Region(kind='table', source_region_id=N). The render-time narrow absorb
# (Phase 2) runs LATER, on bytes, so source_region_id / GLM-OCR enrichment /
# table-adapter routing are never at risk.
# ---------------------------------------------------------------------------

from dart_semantic.qwen_specialists.block_resegment import (  # noqa: E402
    assert_partition_conservation,
    resolve_unit_regroup_table_mode,
)


def _unit_with_table():
    """A worked-example unit (label + solution body) followed by a passthrough
    Solution TABLE carrying a Stage-4 source_region_id, all FB-contiguous."""
    fbs = [_fb("EXAMPLE 1"), _fb("Solution factor"), _fb("table cells")]
    regions = [
        Region(
            kind="paragraph",
            feature_block_indices=(0,),
            payload={
                "text": "EXAMPLE 1",
                "css_class": "pedagogy-example",
                "semantic_class": "worked_example",
            },
        ),
        Region(
            kind="paragraph",
            feature_block_indices=(1,),
            payload={"text": "Solution factor", "css_class": "pedagogy-solution"},
        ),
        Region(
            kind="table",
            feature_block_indices=(2,),
            payload={"text": "table cells"},
            source_region_id=7,
        ),
    ]
    return regions, fbs


def test_in_box_table_keeps_source_region_id(monkeypatch):
    """Sub-flag on + a merged-prose unit + a passthrough table -> the
    post-Stage-5e region list still carries the table as
    Region(kind='table', source_region_id=N); the regroup fused only the prose."""
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "on")
    regions, fbs = _unit_with_table()
    out, _ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    # The prose (label + solution) fused into one region; the table stayed distinct.
    tables = [r for r in out if r.kind == "table"]
    assert len(tables) == 1
    assert tables[0].source_region_id == 7
    assert tables[0].feature_block_indices == (2,)


def test_in_box_table_rpart_untouched(monkeypatch):
    """assert_partition_conservation on the Stage-5e output is byte-identical with
    the sub-flag ON vs OFF — no FB folded (the table-v2 sub-flag is Stage-5e-agnostic)."""
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")

    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "on")
    on_regions, _ = resegment_blocks(*_unit_with_table(), _EmptyState(), runtime=None)
    on_fb = [r.feature_block_indices for r in on_regions]

    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "off")
    regions, fbs = _unit_with_table()
    off_regions, _ = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    off_fb = [r.feature_block_indices for r in off_regions]

    # Same partition (the sub-flag changes NOTHING at Stage-5e).
    assert on_fb == off_fb
    # And R-PART holds on the output (re-derives from raw FB indices).
    assert_partition_conservation(regions, off_regions)


def test_in_box_table_routes_to_table_adapter(monkeypatch):
    """The in-box table's kind=='table' is preserved, so adapter_for routes it to
    the TABLE specialist (not prose) — the <table>/<th scope> structure is generated."""
    from dart_semantic.qwen_specialists.routing import adapter_for, AdapterID

    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "on")
    regions, fbs = _unit_with_table()
    out, _ = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    table = next(r for r in out if r.kind == "table")
    assert adapter_for(table.kind) is AdapterID.TABLE


def test_in_box_table_enriched_by_glm_ocr_predicate(monkeypatch):
    """The Stage-5b enrich predicate (region.kind=='table' AND
    source_region_id is not None) still matches the post-Stage-5e table, so
    enrich_table_regions_with_glm_ocr would find + enrich it exactly as in v1."""
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "on")
    regions, fbs = _unit_with_table()
    out, _ = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    # Mirror glm_ocr_enrich.py:421 exactly.
    enrich_eligible = [
        r for r in out if r.kind == "table" and r.source_region_id is not None
    ]
    assert len(enrich_eligible) == 1
    assert enrich_eligible[0].source_region_id == 7


def test_stage5e_partition_byte_identical_with_table_subflag(monkeypatch):
    """resegment_blocks output is byte-identical with the sub-flag on vs off —
    Stage-5e is SEMANTIK_UNIT_REGROUP_TABLE-agnostic (the hard byte-stability gate)."""
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "on")
    monkeypatch.setenv("SEMANTIK_READING_ORDER_FIX", "on")

    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP_TABLE", "on")
    on_regions, on_ops = resegment_blocks(*_unit_with_table(), _EmptyState(), runtime=None)

    monkeypatch.delenv("SEMANTIK_UNIT_REGROUP_TABLE", raising=False)
    off_regions, off_ops = resegment_blocks(*_unit_with_table(), _EmptyState(), runtime=None)

    assert [r.feature_block_indices for r in on_regions] == [
        r.feature_block_indices for r in off_regions
    ]
    assert [(r.kind, r.source_region_id) for r in on_regions] == [
        (r.kind, r.source_region_id) for r in off_regions
    ]
    assert len(on_ops) == len(off_ops)
