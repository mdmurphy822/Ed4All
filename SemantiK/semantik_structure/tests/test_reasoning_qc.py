"""Unit tests for the Stage-9b reasoning-QC orchestrator (``reasoning_qc``).

**2026-07-12 text-only, document-level pivot.** The QC pass reasons over the
ASSEMBLED accessible-HTML block sequence of the WHOLE document (one QC window),
NOT per-page image rasters. The whole document's ordered block list is
partitioned into ``SEMANTIK_REASONING_QC_WINDOW_BLOCKS`` windows + junction
SEAM strips (seams now span page boundaries too), each judged thinking-on by a
reasoning TEXT model; the same JSON verdict schema drives the UNCHANGED
downstream reconcile / stitch / conservation code.

Pinned here:
  * resolver: default OFF, shadow/on opt-in, garbage → off;
  * SHADOW: audit + capture present, ``capped`` byte-identical (same object);
  * ON: re-type / drop / merge / move land via the existing block-ID op layer
    with fail-closed conservation revert + never-ship-worse adopt gate;
  * ToC harvest recovers declared N.M ordinals + flags a declared-missing one;
  * judgment→FlaggedBlock conversion (phantom / apparatus / misorder modes);
  * document-level partition + seam plan + stitch precedence + qc_incomplete;
  * per-UNIT fan-out (verdicts applied in original window order).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import pytest

from semantik_structure import reasoning_qc


# ---------------------------------------------------------------------------
# Isolate the per-unit resume cache (SEMANTIK_REASONING_QC_CHECKPOINT is default
# ON) into a per-test tmp dir so fixture text reused across tests never
# cross-contaminates via the shared on-disk sidecar. Every test starts with an
# EMPTY cache → always a MISS → the judgment mock is called exactly as before the
# cache existed. The dedicated cache tests below opt back into a shared dir where
# they need cross-run persistence.
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _isolate_qc_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path / "qc_cache_root"))
    monkeypatch.delenv("SEMANTIK_REASONING_QC_CHECKPOINT", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    monkeypatch.delenv("SEMANTIK_STOP_SENTINEL", raising=False)


# ---------------------------------------------------------------------------
# Minimal shims.
# ---------------------------------------------------------------------------
@dataclass
class _Raw:
    text: str
    page: int


@dataclass
class _FB:
    raw: _Raw


@dataclass(frozen=True)
class _Region:
    kind: str
    feature_block_indices: tuple = ()
    payload: dict = field(default_factory=dict)


class _Assembled:
    def __init__(self, region_provenance):
        self.region_provenance = region_provenance


def _record(text, *, type="paragraph", page=1, level=None, role=None):
    """A text-only QC block record (mirrors reasoning_qc._block_record output)."""
    return {"type": type, "role": role or type, "level": level, "page": page, "text": text}


def _simple_doc():
    feature_blocks = [
        _FB(_Raw("1.1 Add Whole Numbers", 1)),
        _FB(_Raw("To add whole numbers, line up the columns.", 1)),
        _FB(_Raw("Running header — Chapter 1", 2)),
        _FB(_Raw("More body text on page two.", 2)),
    ]
    capped = [
        _Region(kind="heading", feature_block_indices=(0,)),
        _Region(kind="paragraph", feature_block_indices=(1,)),
        _Region(kind="heading", feature_block_indices=(2,)),
        _Region(kind="paragraph", feature_block_indices=(3,)),
    ]
    order = [0, 1, 2, 3]
    return capped, feature_blocks, _Assembled(order), order


@pytest.fixture
def _stub_seat(monkeypatch):
    monkeypatch.setattr(reasoning_qc, "_resolve_qc_seat", lambda: object())
    monkeypatch.setattr(reasoning_qc, "_unload_seat", lambda seat: None)
    monkeypatch.setattr(reasoning_qc, "_build_reasoning_qc_capture", lambda: None)
    monkeypatch.setattr(reasoning_qc, "resolve_reasoning_qc_model", lambda default_model=None: "qc:test")


# ---------------------------------------------------------------------------
# Resolver.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "val,expected",
    [
        (None, "off"), ("", "off"), ("off", "off"), ("0", "off"), ("false", "off"),
        ("garbage", "off"), ("shadow", "shadow"), ("SHADOW", "shadow"), ("on", "on"), ("ON", "on"),
    ],
)
def test_resolver(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC", val)
    assert reasoning_qc.resolve_reasoning_qc_mode() == expected


# ---------------------------------------------------------------------------
# SHADOW — audit present, capped byte-identical.
# ---------------------------------------------------------------------------
def test_shadow_byte_identical(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    # Document-level: one window over [0,1,2,3]; flag the running-header region 2.
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"phantom_headings": [{"index": 2, "reason": "hdr"}]},
    )
    capped, fbs, assembled, order = _simple_doc()
    new_capped, verdicts, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert new_capped is capped  # SAME object list — byte-identical
    assert audit["mode"] == "shadow" and audit["ran"] is True
    assert audit["retype"]["applied"] == 0 and audit["merge_move"]["applied"] == 0
    flags = audit["flagged"]
    assert any(f["failure_mode"] == "example_as_heading" and f["region_index"] == 2 for f in flags)
    assert all(f["applied"] is False for f in flags)


# ---------------------------------------------------------------------------
# ON with a reviewer that DECLINES → nothing applied, capped reverts to snapshot.
# ---------------------------------------------------------------------------
def test_on_declined_review_applies_nothing(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"phantom_headings": [{"index": 2, "reason": "hdr"}]},
    )
    # A reviewer that returns the regions UNCHANGED (declines) → applied 0.
    monkeypatch.setattr(
        reasoning_qc, "_run_structure_review",
        lambda regions, fb, rt, *, restrict_to, feedback_by_idx: (list(regions), []),
    )
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)
    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    assert audit["mode"] == "on"
    assert audit["retype"]["applied"] == 0 and audit["merge_move"]["applied"] == 0
    assert audit["retype"]["proposed"] >= 1
    assert [r.kind for r in new_capped] == [r.kind for r in capped]


# ---------------------------------------------------------------------------
# ToC harvest.
# ---------------------------------------------------------------------------
def test_toc_harvest_recovers_ordinals_and_flags_missing():
    feature_blocks = [
        _FB(_Raw("Table of Contents 1.1 Add 1.2 Subtract 1.3 Multiply", 1)),
        _FB(_Raw("1.1 Add Whole Numbers", 2)),
        _FB(_Raw("1.2 Subtract Whole Numbers", 3)),
    ]
    capped = [
        _Region(kind="paragraph", feature_block_indices=(0,)),
        _Region(kind="heading", feature_block_indices=(1,)),
        _Region(kind="heading", feature_block_indices=(2,)),
    ]
    toc = reasoning_qc.harvest_declared_toc_spine(capped, feature_blocks)
    assert toc["declared_ordinals"] == ["1.1", "1.2", "1.3"]
    assert toc["heading_ordinals"] == ["1.1", "1.2"]
    missing = [o for o in toc["declared_ordinals"] if o not in set(toc["heading_ordinals"])]
    assert missing == ["1.3"]


# ---------------------------------------------------------------------------
# Judgment → FlaggedBlock conversion.
# ---------------------------------------------------------------------------
def test_judgments_to_flagged_blocks_modes():
    window = reasoning_qc.QCWindow(
        page=1,
        region_indices=[10, 11, 12],  # capped indices for blocks 0,1,2
        block_records=[_record("a"), _record("b"), _record("c")],
        emit_positions=[0, 1, 2],
    )
    verdict = {
        "phantom_headings": [{"index": 0, "reason": "toc entry"}],
        "apparatus_retype": [{"index": 1, "reason": "answer key"}],
        "misordered": [{"run": [1, 2], "reason": "swapped"}],
    }
    flagged = reasoning_qc.judgments_to_flagged_blocks(verdict, window)
    modes = {f.failure_mode: f for f in flagged}
    assert modes["example_as_heading"].region_index == 10
    assert modes["example_as_heading"].fixable is True
    assert modes["mistyped_component"].region_index == 11
    mis = modes["example_misordered_from_body"]
    assert mis.region_index == 11
    assert mis.proposed_regroup_run == (11, 12)
    assert mis.fixable is False


def test_judgments_drop_out_of_range_indices():
    window = reasoning_qc.QCWindow(
        page=1, region_indices=[5], block_records=[_record("a")], emit_positions=[0]
    )
    verdict = {"phantom_headings": [{"index": 99, "reason": "bogus"}], "misordered": [{"run": [0, 7]}]}
    flagged = reasoning_qc.judgments_to_flagged_blocks(verdict, window)
    assert flagged == []


# ---------------------------------------------------------------------------
# Off-contract verdict shapes must NOT kill the cascade.
#
# Live regression (2026-07-13, ch01-reval): a QC unit returned a FLOAT where the
# contract says findings-list, and the unguarded ``for item in v.get(key) or ()``
# raised TypeError("'float' object is not iterable") out of _absorb_window_findings
# — killing a ~2h cascade at the final stitch, after every QC unit was paid for.
# A malformed FIELD is a model-shape error: drop that key, keep the rest.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [0.87, 1, "none", {"index": 0}, True])
def test_qc_findings_drops_non_list_verdict_key(bad):
    assert reasoning_qc._qc_findings({"phantom_headings": bad}, "phantom_headings") == ()


def test_qc_findings_passes_through_real_lists_and_missing_keys():
    v = {"phantom_headings": [{"index": 0}], "apparatus_retype": None}
    assert reasoning_qc._qc_findings(v, "phantom_headings") == ({"index": 0},)
    assert reasoning_qc._qc_findings(v, "apparatus_retype") == ()
    assert reasoning_qc._qc_findings(v, "absent") == ()
    assert reasoning_qc._qc_findings("not-a-dict", "phantom_headings") == ()


def test_absorb_window_findings_survives_float_where_list_expected():
    """The exact crash: a float on a findings key must not raise."""
    merged = {"phantom_headings": [], "apparatus_retype": [], "misordered": []}
    verdict = {
        "phantom_headings": 0.87,  # <-- the live off-contract value
        "apparatus_retype": [{"index": 1, "reason": "answer key"}],
        "misordered": 0.5,
    }
    reasoning_qc._absorb_window_findings(verdict, 10, 13, merged)
    # The malformed keys are dropped; the WELL-FORMED sibling still lands.
    assert merged["phantom_headings"] == []
    assert merged["apparatus_retype"] == [{"index": 11, "reason": "answer key"}]


def test_judgments_to_flagged_blocks_survives_float_verdict_keys():
    window = reasoning_qc.QCWindow(
        page=1,
        region_indices=[10, 11],
        block_records=[_record("a"), _record("b")],
        emit_positions=[0, 1],
    )
    verdict = {
        "phantom_headings": 0.9,
        "apparatus_retype": [{"index": 1, "reason": "answer key"}],
        "misordered": 3.14,
    }
    flagged = reasoning_qc.judgments_to_flagged_blocks(verdict, window)
    assert [f.region_index for f in flagged] == [11]


# ---------------------------------------------------------------------------
# MOVE-op default shadow — a misorder flag is proposed, not applied.
# ---------------------------------------------------------------------------
def test_misorder_audited_not_applied(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.delenv("SEMANTIK_MOVE_OP", raising=False)  # default shadow
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"misordered": [{"run": [0, 1], "reason": "swap"}]},
    )
    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    misorder = [f for f in audit["flagged"] if f["failure_mode"] == "example_misordered_from_body"]
    assert misorder and misorder[0]["proposed_regroup_run"] == [0, 1]
    assert audit["merge_move"]["applied"] == 0
    assert [r.kind for r in new_capped] == [r.kind for r in capped]


# ---------------------------------------------------------------------------
# CHECK (A) — toc_reconcile emits reconcile ops with evidence, degrades on no ToC.
# ---------------------------------------------------------------------------
def test_toc_reconcile_flags_phantom_and_apparatus():
    feature_blocks = [
        _FB(_Raw("Table of Contents 1.1 Add 1.2 Subtract", 1)),
        _FB(_Raw("1.1 Add Whole Numbers", 2)),
        _FB(_Raw("Body prose for section 1.1.", 2)),
        _FB(_Raw("1.9 Phantom Heading", 3)),
        _FB(_Raw("1.9 Phantom Heading", 3)),
        _FB(_Raw("Chapter Review", 4)),
    ]
    capped = [
        _Region(kind="paragraph", feature_block_indices=(0,)),
        _Region(kind="heading", feature_block_indices=(1,)),
        _Region(kind="paragraph", feature_block_indices=(2,)),
        _Region(kind="heading", feature_block_indices=(3,)),
        _Region(kind="heading", feature_block_indices=(4,)),
        _Region(kind="heading", feature_block_indices=(5,)),
    ]
    toc = reasoning_qc.harvest_declared_toc_spine(capped, feature_blocks)
    flags = reasoning_qc.toc_reconcile(capped, feature_blocks, toc)
    by_idx = {f.region_index: f for f in flags}
    assert 1 not in by_idx
    assert by_idx[3].failure_mode == "example_as_heading"
    assert "1.9" in by_idx[3].fix_hint
    assert by_idx[5].failure_mode == "mistyped_component"


def test_toc_reconcile_no_spine_skips_gracefully():
    feature_blocks = [_FB(_Raw("1.1 Add Whole Numbers", 1))]
    capped = [_Region(kind="heading", feature_block_indices=(0,))]
    logs = []
    flags = reasoning_qc.toc_reconcile(
        capped, feature_blocks, {"declared_ordinals": []}, log=logs.append
    )
    assert flags == []
    assert any("no declared ToC spine" in m for m in logs)


# ---------------------------------------------------------------------------
# CHECK (B) — order_verify synthesizes a reorder from a divergent reading_order.
# ---------------------------------------------------------------------------
def test_order_verify_synthesizes_reorder_from_reading_order(monkeypatch):
    win = reasoning_qc.QCWindow(
        page=1, region_indices=[3, 4], block_records=[_record("a"), _record("b")], emit_positions=[0, 1]
    )
    monkeypatch.setattr(
        reasoning_qc, "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"reading_order": [1, 0], "confidence": 0.7},
    )
    verdict, flagged, div = reasoning_qc.order_verify(object(), "/tmp/x.pdf", win)
    assert div == 2
    mis = [f for f in flagged if f.failure_mode == "example_misordered_from_body"]
    assert len(mis) == 1
    assert mis[0].proposed_regroup_run == (4, 3)  # window indices in VLM order [1,0]
    assert mis[0].fixable is False


def test_page_order_verify_alias_exists():
    """The historic per-page name stays as a back-compat alias of order_verify."""
    assert reasoning_qc.page_order_verify is reasoning_qc.order_verify


# ---------------------------------------------------------------------------
# ON apply channels — re-type lands via run_structure_review.
# ---------------------------------------------------------------------------
def test_on_retype_demotes_phantom(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"phantom_headings": [{"index": 2, "reason": "hdr"}]},
    )
    captured = {}

    def _fake_review(regions, feature_blocks, runtime, *, restrict_to, feedback_by_idx):
        captured["restrict_to"] = set(restrict_to)
        captured["feedback"] = dict(feedback_by_idx)
        out = list(regions)
        for i in restrict_to:
            out[i] = _Region(kind="paragraph", feature_block_indices=out[i].feature_block_indices)
        return out, []

    monkeypatch.setattr(reasoning_qc, "_run_structure_review", _fake_review)
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    assert captured["restrict_to"] == {2}
    assert new_capped[2].kind == "paragraph"
    assert new_capped[2].feature_block_indices == capped[2].feature_block_indices
    assert audit["retype"]["applied"] == 1


def test_on_failclosed_conservation_revert(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"phantom_headings": [{"index": 2, "reason": "hdr"}]},
    )

    def _corrupting_review(regions, feature_blocks, runtime, *, restrict_to, feedback_by_idx):
        out = list(regions)
        for i in restrict_to:
            out[i] = _Region(kind="metadata_drop", feature_block_indices=out[i].feature_block_indices)
        return out, []

    monkeypatch.setattr(reasoning_qc, "_run_structure_review", _corrupting_review)

    def _boom(*a, **k):
        raise RuntimeError("token conservation violated")

    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", _boom)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    assert audit["retype"]["applied"] == 0
    assert [r.kind for r in new_capped] == [r.kind for r in capped]


def test_never_ship_worse(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"phantom_headings": [{"index": 2, "reason": "hdr"}]},
    )
    monkeypatch.setattr(
        reasoning_qc, "_run_structure_review",
        lambda regions, fb, rt, *, restrict_to, feedback_by_idx: (list(regions), []),
    )
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    assert audit["retype"]["applied"] == 0
    assert new_capped is capped or [r.kind for r in new_capped] == [r.kind for r in capped]


def test_move_live_reorders(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"misordered": [{"run": [0, 1], "reason": "swap"}]},
    )

    sentinel = [_Region(kind="MERGED", feature_block_indices=(0, 1))]

    class _Op:
        op = "merge"

    def _fake_unit_fix(regions, fb, runs):
        assert runs and list(runs[0]) == [0, 1]
        return sentinel, [_Op()]

    monkeypatch.setattr(reasoning_qc, "_apply_proposed_unit_fix", _fake_unit_fix)
    monkeypatch.setattr(reasoning_qc, "_assert_partition_conservation", lambda *a, **k: None)
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)
    monkeypatch.setattr(reasoning_qc, "_build_resegment_audit_rows", lambda ops: [{"op": "move"}])

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert audit["merge_move"]["applied"] == 1
    assert [r.kind for r in new_capped] == ["MERGED"]
    assert audit["applied_ops"] == [{"op": "move"}]


def test_move_live_conservation_revert(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"misordered": [{"run": [0, 1], "reason": "swap"}]},
    )

    class _Op:
        op = "move"

    monkeypatch.setattr(
        reasoning_qc, "_apply_proposed_unit_fix",
        lambda regions, fb, runs: ([_Region(kind="CORRUPT", feature_block_indices=())], [_Op()]),
    )

    def _boom(*a, **k):
        raise RuntimeError("partition conservation violated")

    monkeypatch.setattr(reasoning_qc, "_assert_partition_conservation", _boom)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert audit["merge_move"]["applied"] == 0
    assert [r.kind for r in new_capped] == [r.kind for r in capped]


# ===========================================================================
# Required-case coverage (mock the model client — no real inference).
# ===========================================================================
def _retype_review(new_kind):
    def _review(regions, feature_blocks, runtime, *, restrict_to, feedback_by_idx):
        out = list(regions)
        for i in restrict_to:
            out[i] = _Region(kind=new_kind, feature_block_indices=out[i].feature_block_indices)
        return out, []

    return _review


def test_phantom_chapter_drop_op(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"phantom_headings": [{"index": 2, "reason": "running header"}]},
    )
    monkeypatch.setattr(reasoning_qc, "_run_structure_review", _retype_review("metadata_drop"))
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    assert new_capped[2].kind == "metadata_drop"
    assert new_capped[2].feature_block_indices == capped[2].feature_block_indices
    assert audit["retype"]["applied"] == 1
    assert any(
        f["failure_mode"] == "example_as_heading" and f["region_index"] == 2
        for f in audit["flagged"]
    )


def test_apparatus_heading_retype_op(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"apparatus_retype": [{"index": 2, "reason": "answer key banner"}]},
    )
    captured = {}

    def _review(regions, feature_blocks, runtime, *, restrict_to, feedback_by_idx):
        captured["restrict_to"] = set(restrict_to)
        captured["feedback"] = dict(feedback_by_idx)
        out = list(regions)
        for i in restrict_to:
            out[i] = _Region(kind="paragraph", feature_block_indices=out[i].feature_block_indices)
        return out, []

    monkeypatch.setattr(reasoning_qc, "_run_structure_review", _review)
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    assert captured["restrict_to"] == {2}
    assert "re-type apparatus" in captured["feedback"][2]
    assert new_capped[2].kind == "paragraph"
    assert audit["retype"]["applied"] == 1
    assert any(f["failure_mode"] == "mistyped_component" for f in audit["flagged"])


def test_scrambled_order_reorder_op(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    # A divergent reading_order over the whole 4-block window (swap first two).
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"reading_order": [1, 0, 2, 3], "confidence": 0.7},
    )
    reordered = [_Region(kind="REORDERED", feature_block_indices=(1, 0))]

    class _Op:
        op = "move"

    monkeypatch.setattr(
        reasoning_qc, "_apply_proposed_unit_fix",
        lambda regions, fb, runs: (reordered, [_Op()]),
    )
    monkeypatch.setattr(reasoning_qc, "_assert_partition_conservation", lambda *a, **k: None)
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)
    monkeypatch.setattr(reasoning_qc, "_build_resegment_audit_rows", lambda ops: [{"op": "move"}])

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert any(f["failure_mode"] == "example_misordered_from_body" for f in audit["flagged"])
    assert audit["merge_move"]["applied"] == 1
    assert [r.kind for r in new_capped] == ["REORDERED"]


def test_healthy_input_zero_ops(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", lambda seat, pdf, page, blocks: {})

    def _no_review(*a, **k):
        raise AssertionError("reviewer must not be driven on a healthy document")

    def _no_unit_fix(*a, **k):
        raise AssertionError("unit-fix must not be driven on a healthy document")

    monkeypatch.setattr(reasoning_qc, "_run_structure_review", _no_review)
    monkeypatch.setattr(reasoning_qc, "_apply_proposed_unit_fix", _no_unit_fix)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    assert audit["ran"] is True
    assert audit["flagged"] == []
    assert audit["retype"] == {"applied": 0, "proposed": 0}
    assert audit["merge_move"] == {"applied": 0, "proposed": 0}
    assert audit["toc_reconcile"] == {"proposed": 0}
    assert [r.kind for r in new_capped] == [r.kind for r in capped]
    assert [r.feature_block_indices for r in new_capped] == [
        r.feature_block_indices for r in capped
    ]


def test_flag_off_byte_identical_noop(monkeypatch):
    monkeypatch.delenv("SEMANTIK_REASONING_QC", raising=False)
    monkeypatch.setattr(
        reasoning_qc,
        "_resolve_qc_seat",
        lambda: (_ for _ in ()).throw(AssertionError("seat resolved with flag OFF")),
    )
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("judgment called with flag OFF")),
    )
    capped, fbs, assembled, order = _simple_doc()
    verdicts_in = object()
    new_capped, verdicts_out, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_verdicts=verdicts_in
    )
    assert new_capped is capped
    assert verdicts_out is verdicts_in
    assert audit["ran"] is False and audit["mode"] == "off"


def _pageless_doc():
    """A doc whose sole region has no resolvable physical page."""
    feature_blocks = [_FB(_Raw("Body text with no page number.", None))]
    capped = [_Region(kind="paragraph", feature_block_indices=(0,))]
    order = [0]
    return capped, feature_blocks, _Assembled(order), order


def test_pageless_doc_still_judged_text_only(monkeypatch, _stub_seat):
    """After the text-only pivot a region with NO page is STILL judged (no page
    image needed) — the historic 'skip pageless window' behavior is retired."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    seen: list = []
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: seen.append(len(blocks)) or {},
    )
    capped, fbs, assembled, order = _pageless_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert new_capped is capped  # shadow → byte-identical
    assert seen == [1]  # the pageless block WAS dispatched (no skip)
    assert audit["ran"] is True
    assert len(audit["windows"]) == 1


# ===========================================================================
# Per-UNIT fan-out (bounded ThreadPoolExecutor).
# ===========================================================================
@pytest.mark.parametrize(
    "val,expected",
    [(None, 8), ("", 8), ("garbage", 8), ("0", 8), ("-3", 8), ("1", 1), ("8", 8), ("16", 16)],
)
def test_qc_concurrency_resolver(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_CONCURRENCY", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_CONCURRENCY", val)
    assert reasoning_qc.resolve_reasoning_qc_concurrency() == expected


def _multi_unit_doc(n=12):
    fbs = [_FB(_Raw(f"blk {i}", 1 + i // 4)) for i in range(n)]
    capped = [_Region(kind="paragraph", feature_block_indices=(i,)) for i in range(n)]
    order = list(range(n))
    return capped, fbs, _Assembled(order), order


def test_fanout_all_units_called_flagged_in_original_order(monkeypatch, _stub_seat):
    """Many UNITS fanned out concurrently → every unit judged, findings STITCHED
    in original (ascending block) order even when completion order is inverted.

    12 blocks / WINDOW=4 → windows (0,4)(4,8)(8,12) + seams (2,6)(6,10) = 5 units.
    Each WINDOW flags its first block phantom; seams return nothing. A Barrier
    makes the units start together, then each sleeps inversely by its first block
    so completion order is reversed. The stitched flags must be ascending."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", "4")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", "2")
    monkeypatch.delenv("SEMANTIK_REASONING_QC_CONCURRENCY", raising=False)  # 8 ≥ 5 units

    barrier = threading.Barrier(5)
    completion: list[int] = []
    lock = threading.Lock()

    def _judge(seat, pdf, page, blocks):
        # Route by the unit's first block text ("blk N").
        first = int(blocks[0]["text"].split()[1])
        barrier.wait()
        time.sleep((12 - first) * 0.01)  # later blocks finish first
        with lock:
            completion.append(first)
        # Only a full window (a block whose local index 0 is a window start)
        # flags a phantom; use the block text to phantom-flag its own position.
        return {"phantom_headings": [{"index": 0, "reason": f"blk {first}"}]}

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _judge)

    capped, fbs, assembled, order = _multi_unit_doc(12)
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    # All five units ran; completion order was genuinely inverted by the sleeps.
    assert sorted(completion) == [0, 2, 4, 6, 8]
    assert completion == [8, 6, 4, 2, 0]
    # Window phantoms land at window starts 0/4/8; seam phantoms (2/6) are
    # DISCARDED (window owns intra findings). Flagged records ascending.
    region_idxs = [f["region_index"] for f in audit["flagged"]]
    assert region_idxs == sorted(region_idxs)
    assert region_idxs == [0, 4, 8]
    assert new_capped is capped  # shadow → byte-identical


def test_fanout_failsoft_on_worker_raise(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")

    def _boom(seat, pdf, page, blocks):
        raise RuntimeError("judgment worker blew up")

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _boom)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert new_capped is capped
    assert audit["ran"] is True
    assert audit["flagged"] == []
    assert len(audit["windows"]) == 1
    assert audit["windows"][0]["n_flagged"] == 0


# ===========================================================================
# PARTITION + SEAM windowing — plan, stitch precedence, seam qc_incomplete,
# single-window no-seam, never-non-thinking on seams (document-level).
# ===========================================================================
@pytest.mark.parametrize(
    "val,expected",
    [(None, 30), ("", 30), ("garbage", 30), ("0", 4), ("2", 4), ("4", 4), ("10", 10), ("50", 50)],
)
def test_resolve_window_blocks(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", val)
    assert reasoning_qc.resolve_reasoning_qc_window_blocks() == expected


@pytest.mark.parametrize(
    "val,expected",
    [(None, 5), ("", 5), ("garbage", 5), ("0", 2), ("1", 2), ("2", 2), ("5", 5), ("9", 9)],
)
def test_resolve_seam_blocks(monkeypatch, val, expected):
    if val is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", val)
    assert reasoning_qc.resolve_reasoning_qc_seam_blocks() == expected


def test_plan_single_window_no_seams(monkeypatch):
    monkeypatch.delenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", raising=False)
    wins, seams = reasoning_qc._plan_page_units(4)
    assert wins == [(0, 4)] and seams == []


def test_plan_partition_exact_and_seams(monkeypatch):
    monkeypatch.delenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", raising=False)  # 30
    monkeypatch.delenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", raising=False)    # 5
    wins, seams = reasoning_qc._plan_page_units(61)
    assert wins == [(0, 30), (30, 60), (60, 61)]
    assert wins[0][0] == 0 and wins[-1][1] == 61
    for a, b in zip(wins, wins[1:]):
        assert a[1] == b[0]
    assert len(seams) == len(wins) - 1
    assert seams == [(25, 35), (55, 61)]
    for k, (s, e) in enumerate(seams):
        junction = wins[k][1]
        assert s < junction < e


def test_plan_env_override(monkeypatch):
    monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", "10")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", "3")
    wins, seams = reasoning_qc._plan_page_units(25)
    assert wins == [(0, 10), (10, 20), (20, 25)]
    assert seams == [(7, 13), (17, 23)]


def _win6():
    """A 6-block window (capped indices 10..15)."""
    return reasoning_qc.QCWindow(
        page=1, region_indices=[10, 11, 12, 13, 14, 15],
        block_records=[_record("b")] * 6, emit_positions=list(range(6)),
    )


def test_stitch_seam_authoritative_order_vs_window_retype():
    window = _win6()
    subwins, seams = [(0, 3), (3, 6)], [(1, 5)]
    raw = {
        (0, "window", 0, 3): {
            "phantom_headings": [{"index": 2, "reason": "win block2 phantom"}],
            "misordered": [{"run": [0, 1], "reason": "intra-window"}],
        },
        (0, "window", 3, 6): {},
        (0, "seam", 1, 5): {
            "reading_order": [3, 2, 1, 0],
            "phantom_headings": [{"index": 2, "reason": "seam phantom — MUST be ignored"}],
        },
    }
    merged, _div = reasoning_qc._stitch_from_raw(raw, 0, window, subwins, seams)
    ph = merged.get("phantom_headings", [])
    assert any(p["index"] == 2 for p in ph)
    assert all(p["index"] != 3 for p in ph)
    runs = [r["run"] for r in merged["misordered"]]
    assert runs == [[0, 1], [4, 3, 2, 1]]


def test_stitch_seam_qc_incomplete_recorded():
    window = _win6()
    subwins, seams = [(0, 3), (3, 6)], [(1, 5)]
    raw = {
        (0, "window", 0, 3): {},
        (0, "window", 3, 6): {},
        (0, "seam", 1, 5): {"_qc_incomplete": [0, 1, 2, 3]},
    }
    merged, _div = reasoning_qc._stitch_from_raw(raw, 0, window, subwins, seams)
    assert merged["_qc_incomplete"] == [1, 2, 3, 4]


def test_single_window_doc_no_seam_calls(monkeypatch, _stub_seat):
    """A document that fits one window dispatches exactly ONE unit — ZERO seams."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.delenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", raising=False)  # 30
    seen: list = []
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: seen.append(len(blocks)) or {},
    )
    capped, fbs, assembled, order = _simple_doc()  # 4 blocks ≤ 30 → one window
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert seen == [4]  # one document unit over all 4 blocks, no seam units


def test_audit_records_qc_incomplete(monkeypatch, _stub_seat):
    """Every unit incomplete → the window audit entry honestly lists the capped
    region indices the split ladder left unverified."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", "4")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", "2")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: {"_qc_incomplete": list(range(len(blocks)))},
    )
    fbs = [_FB(_Raw(f"blk {i}", 1)) for i in range(6)]
    capped = [_Region(kind="paragraph", feature_block_indices=(i,)) for i in range(6)]
    order = list(range(6))
    _nc, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, _Assembled(order), order, pdf_path="/tmp/x.pdf"
    )
    assert len(audit["windows"]) == 1
    assert audit["windows"][0]["qc_incomplete"] == [0, 1, 2, 3, 4, 5]


def test_seam_units_dispatched_thinking_on_never_off(monkeypatch, _stub_seat):
    """OWNER-DIRECTIVE regression guard at the orchestrator: forcing a partition
    + seam plan over the DOCUMENT, EVERY dispatched unit (windows AND seams)
    routes through the real thinking-on TEXT POST — no request body carries a
    thinking-off block OR image bytes — and a genuine junction seam strip is
    among the dispatched units."""
    from semantik_structure import reasoning_qc_vlm
    from semantik_structure.extract_shared import VLMSeat

    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", "4")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", "2")
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_VLM_DISABLE_THINKING", raising=False)

    bodies: list = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": '{"reading_order": []}'}}]}

    class _Req:
        def post(self, url, json=None, headers=None, timeout=None):
            bodies.append(json)
            return _Resp()

    req = _Req()
    real_seat = VLMSeat(provider="local", base_url="http://localhost:11434/v1", api_key=None, model="m")
    monkeypatch.setattr(reasoning_qc, "_resolve_qc_seat", lambda: real_seat)
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, blocks: reasoning_qc_vlm.run_qc_judgment(
            seat, pdf, page, blocks, requests_module=req
        ),
    )

    # 10 blocks → windows (0,4)(4,8)(8,10) + seams (2,6)(6,10).
    fbs = [_FB(_Raw(f"b{i}", 1)) for i in range(10)]
    capped = [_Region(kind="paragraph", feature_block_indices=(i,)) for i in range(10)]
    order = list(range(10))
    reasoning_qc.run_reasoning_qc(capped, fbs, _Assembled(order), order, pdf_path="/tmp/x.pdf")

    assert bodies  # units were dispatched
    for b in bodies:
        # Never a thinking-off body: chat_template_kwargs may carry ONLY the
        # Nemotron-3 reasoning_budget on the thinking-on path.
        ctk = b.get("chat_template_kwargs") or {}
        assert "thinking" not in ctk and "enable_thinking" not in ctk
        user = b["messages"][1]["content"]
        assert all(part.get("type") != "image_url" for part in user)  # never an image

    def _texts(b):
        user = b["messages"][1]["content"]
        t = next(p for p in user if p["type"] == "text")["text"]
        return {line.rsplit(" ", 1)[-1] for line in t.splitlines() if line.startswith("[")}

    seam_sets = [_texts(b) for b in bodies]
    # The junction-4 seam strip [2,6) — b2,b3,b4,b5 — was judged thinking-on.
    assert {"b2", "b3", "b4", "b5"} in seam_sets


# ===========================================================================
# Per-UNIT resume cache/sidecar + in-fan-out stop poll
# (SEMANTIK_REASONING_QC_CHECKPOINT graceful-stop contract).
# ===========================================================================
def _cache_files(reasoning_qc_mod=reasoning_qc):
    root = reasoning_qc_mod._qc_cache_root()
    return list(root.rglob("*.json")) if root.exists() else []


@pytest.mark.parametrize(
    "site,family,expected",
    [
        (None, None, True),        # both unset → default ON
        (None, "0", False),        # family falsey → off
        (None, "garbage", True),   # family garbage → on
        ("0", None, False),        # site falsey wins
        ("off", "1", False),       # site falsey beats family truthy
        ("on", "0", True),         # site truthy beats family falsey
        ("1", None, True),
    ],
)
def test_checkpoint_resolver(monkeypatch, site, family, expected):
    if site is None:
        monkeypatch.delenv("SEMANTIK_REASONING_QC_CHECKPOINT", raising=False)
    else:
        monkeypatch.setenv("SEMANTIK_REASONING_QC_CHECKPOINT", site)
    if family is None:
        monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    else:
        monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", family)
    assert reasoning_qc.resolve_reasoning_qc_checkpoint() is expected


def test_cache_hit_skips_post(monkeypatch, _stub_seat):
    """A second run over the SAME unit is served from the sidecar — no re-POST."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    calls = {"n": 0}

    def _judge(seat, pdf, page, blocks):
        calls["n"] += 1
        return {"phantom_headings": [{"index": 0, "reason": "cached me"}]}

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _judge)

    capped, fbs, assembled, order = _simple_doc()  # one window unit
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert calls["n"] == 1
    assert _cache_files()  # the verdict was persisted

    # Second run: identical input → cache HIT → judgment fn NOT called again.
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert calls["n"] == 1


def test_fingerprint_varies_by_model_prompt_text_kind(monkeypatch):
    blocks = [_record("Add whole numbers.", type="paragraph", page=1)]
    base = reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window")

    # Same input → stable.
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window") == base
    # Model change → different key.
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m2", kind="window") != base
    # Unit kind change → different key.
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="seam") != base
    # Block text change → different key.
    other = [_record("Subtract whole numbers.", type="paragraph", page=1)]
    assert reasoning_qc._qc_unit_fingerprint(other, model="m1", kind="window") != base
    # Prompt-contract version bump → different key.
    from semantik_structure import reasoning_qc_vlm

    monkeypatch.setattr(reasoning_qc_vlm, "QC_PROMPT_VERSION", 999)
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window") != base


def test_fingerprint_varies_by_sampling(monkeypatch):
    """Effective sampling (temperature / top_p / max_tokens) is in the key — a
    cache HIT across different sampling would be a wrong verdict."""
    blocks = [_record("Add whole numbers.", type="paragraph", page=1)]
    monkeypatch.delenv("SEMANTIK_REASONING_QC_DISABLE_THINKING", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_TEMPERATURE", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_TOP_P", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", raising=False)
    base = reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window")

    monkeypatch.setenv("SEMANTIK_REASONING_QC_TEMPERATURE", "0.3")
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window") != base
    monkeypatch.delenv("SEMANTIK_REASONING_QC_TEMPERATURE", raising=False)

    monkeypatch.setenv("SEMANTIK_REASONING_QC_TOP_P", "0.8")
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window") != base
    monkeypatch.delenv("SEMANTIK_REASONING_QC_TOP_P", raising=False)

    monkeypatch.setenv("SEMANTIK_REASONING_QC_MAX_TOKENS", "4096")
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window") != base
    monkeypatch.delenv("SEMANTIK_REASONING_QC_MAX_TOKENS", raising=False)

    # Reasoning-budget change → different key (budget shapes the deliberation
    # the verdict came from; a cross-budget cache HIT would be a wrong verdict).
    monkeypatch.setenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", "2048")
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window") != base
    monkeypatch.setenv("SEMANTIK_REASONING_QC_REASONING_BUDGET", "0")  # disabled
    assert reasoning_qc._qc_unit_fingerprint(blocks, model="m1", kind="window") != base


def test_empty_verdict_not_cached(monkeypatch, _stub_seat):
    """A fail-soft {} verdict is NEVER persisted — a transport blip must re-run."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    empty_calls = {"n": 0}

    def _empty(seat, pdf, page, blocks):
        empty_calls["n"] += 1
        return {}

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _empty)
    capped, fbs, assembled, order = _simple_doc()
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert empty_calls["n"] == 2  # no cache hit
    assert not _cache_files()  # nothing written

    # The cacheability predicate is unchanged — neither an empty nor an incomplete
    # verdict is a POSITIVE finding (an incomplete verdict rides the run-scoped
    # NEGATIVE ledger instead — see test_incomplete_verdict_run_scoped_ledger).
    assert reasoning_qc._qc_verdict_cacheable({}) is False
    assert reasoning_qc._qc_verdict_cacheable({"_qc_incomplete": [0]}) is False
    assert reasoning_qc._qc_verdict_cacheable({"phantom_headings": [{"index": 0}]}) is True


def test_incomplete_verdict_run_scoped_ledger(monkeypatch, _stub_seat):
    """A split-ladder _qc_incomplete TERMINAL failure is persisted as a run-scoped
    NEGATIVE: a same-run resume re-POSTs NOTHING (the 4h re-attempt bug), but a NEW
    run re-attempts (never permanently starved)."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    inc_calls = {"n": 0}

    def _incomplete(seat, pdf, page, blocks):
        inc_calls["n"] += 1
        return {"_qc_incomplete": list(range(len(blocks)))}

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _incomplete)
    capped, fbs, assembled, order = _simple_doc()

    # Same run scope across both passes → the terminal failure is a negative HIT.
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-x")
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    first = inc_calls["n"]
    assert first >= 1
    assert _cache_files()  # the negative record IS written (unlike the empty {}).
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert inc_calls["n"] == first  # resume of the SAME run → zero re-POST.

    # A genuinely NEW run recomputes the same fingerprint, sees a CROSS-scope
    # negative → miss → re-attempts (a later run with a healthy endpoint retries).
    monkeypatch.setenv("ED4ALL_RUN_ID", "run-y")
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert inc_calls["n"] == 2 * first


def test_checkpoint_off_no_cache_dir_all_units_posted(monkeypatch, _stub_seat):
    """Flag OFF → no reads, no writes: the cache dir is never created and EVERY
    unit is POSTed (no hit)."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_CHECKPOINT", "0")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", "4")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", "2")
    calls = {"n": 0}

    def _judge(seat, pdf, page, blocks):
        calls["n"] += 1
        return {"phantom_headings": [{"index": 0, "reason": "x"}]}

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _judge)
    capped, fbs, assembled, order = _multi_unit_doc(12)  # 3 windows + 2 seams = 5 units
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert calls["n"] == 5  # every unit dispatched, no cache
    assert not reasoning_qc._qc_cache_root().exists()  # dir never created

    # A second run STILL re-POSTs every unit (no persistence at all).
    reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")
    assert calls["n"] == 10
    assert not reasoning_qc._qc_cache_root().exists()


def test_stop_mid_fanout_persists_completed_and_propagates(monkeypatch, _stub_seat, tmp_path):
    """A stop sentinel written after the first unit completes → the fan-out stops
    SUBMITTING new units, the completed unit is persisted to its sidecar, and the
    real CascadeStopRequested propagates out (the runner's pause path)."""
    from semantik_structure.stop_seam import CascadeStopRequested

    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_WINDOW_BLOCKS", "4")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_SEAM_BLOCKS", "2")
    monkeypatch.setenv("SEMANTIK_REASONING_QC_CONCURRENCY", "1")  # one unit at a time

    sentinel = tmp_path / "STOP_REQUESTED"
    monkeypatch.setenv("SEMANTIK_STOP_SENTINEL", str(sentinel))
    calls = {"n": 0}

    def _judge(seat, pdf, page, blocks):
        calls["n"] += 1
        # After the FIRST unit completes, an operator issues `ed4all stop`.
        sentinel.write_text("stop")
        return {"phantom_headings": [{"index": 0, "reason": "unit0"}]}

    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", _judge)

    capped, fbs, assembled, order = _multi_unit_doc(12)  # 5 units total
    with pytest.raises(CascadeStopRequested):
        reasoning_qc.run_reasoning_qc(capped, fbs, assembled, order, pdf_path="/tmp/x.pdf")

    # Only the first unit ran (concurrency 1 → sentinel seen before unit 2 submits).
    assert calls["n"] == 1
    # The completed unit was checkpointed → a resume is a cache HIT (no re-POST).
    assert len(_cache_files()) == 1
