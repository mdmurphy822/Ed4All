"""Unit tests for the Stage-9b reasoning-QC orchestrator (``reasoning_qc``).

STEP-1 scope (skeleton + flag gating + shadow/advisory + DecisionCapture + ToC
harvest). The op-apply-with-conservation channels are STEP-2 stubs, so the
ON-mode *apply* assertions (re-type/drop/merge/move landing + fail-closed
conservation revert + never-ship-worse adopt gate) are marked here with a clear
STEP-2 marker rather than pinned against the stub.

Pinned at STEP 1:
  * resolver: default OFF, shadow/on opt-in, garbage → off;
  * SHADOW: audit + capture present, ``capped`` byte-identical (same object);
  * ON with stub channels: applied=0, ``capped`` reverts to the input snapshot;
  * ToC harvest recovers declared N.M ordinals + flags a declared-missing one;
  * judgment→FlaggedBlock conversion (phantom / apparatus / misorder modes),
    out-of-range indices dropped;
  * MOVE-op default-shadow: a misorder flag is audited but proposes-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from semantik_structure import reasoning_qc


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
        (None, "off"),
        ("", "off"),
        ("off", "off"),
        ("0", "off"),
        ("false", "off"),
        ("garbage", "off"),
        ("shadow", "shadow"),
        ("SHADOW", "shadow"),
        ("on", "on"),
        ("ON", "on"),
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
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"phantom_headings": [{"index": 0, "reason": "hdr"}]}
        if page == 2
        else {},
    )
    capped, fbs, assembled, order = _simple_doc()
    new_capped, verdicts, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert new_capped is capped  # SAME object list — byte-identical
    assert audit["mode"] == "shadow" and audit["ran"] is True
    assert audit["retype"]["applied"] == 0 and audit["merge_move"]["applied"] == 0
    # The page-2 phantom flag is recorded (region index 2), applied=False.
    flags = audit["flagged"]
    assert any(f["failure_mode"] == "example_as_heading" and f["region_index"] == 2 for f in flags)
    assert all(f["applied"] is False for f in flags)


# ---------------------------------------------------------------------------
# ON with STEP-1 stub channels — nothing applied, capped reverts to snapshot.
# ---------------------------------------------------------------------------
def test_on_stub_channels_apply_nothing(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"phantom_headings": [{"index": 0, "reason": "hdr"}]}
        if page == 2
        else {},
    )
    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert audit["mode"] == "on"
    assert audit["retype"]["applied"] == 0 and audit["merge_move"]["applied"] == 0
    # Proposed count reflects the phantom flag (a fixable re-type op).
    assert audit["retype"]["proposed"] >= 1
    # Nothing applied → capped equals the input snapshot content.
    assert [r.kind for r in new_capped] == [r.kind for r in capped]


# ---------------------------------------------------------------------------
# ToC harvest.
# ---------------------------------------------------------------------------
def test_toc_harvest_recovers_ordinals_and_flags_missing():
    feature_blocks = [
        _FB(_Raw("Table of Contents 1.1 Add 1.2 Subtract 1.3 Multiply", 1)),
        _FB(_Raw("1.1 Add Whole Numbers", 2)),
        _FB(_Raw("1.2 Subtract Whole Numbers", 3)),
        # 1.3 declared in the ToC but NO heading region for it → declared-missing.
    ]
    capped = [
        _Region(kind="paragraph", feature_block_indices=(0,)),
        _Region(kind="heading", feature_block_indices=(1,)),
        _Region(kind="heading", feature_block_indices=(2,)),
    ]
    toc = reasoning_qc.harvest_declared_toc_spine(capped, feature_blocks)
    assert toc["declared_ordinals"] == ["1.1", "1.2", "1.3"]
    assert toc["heading_ordinals"] == ["1.1", "1.2"]
    # The orchestrator computes declared_missing = declared - heading.
    missing = [o for o in toc["declared_ordinals"] if o not in set(toc["heading_ordinals"])]
    assert missing == ["1.3"]


# ---------------------------------------------------------------------------
# Judgment → FlaggedBlock conversion.
# ---------------------------------------------------------------------------
def test_judgments_to_flagged_blocks_modes():
    window = reasoning_qc.QCWindow(
        page=1,
        region_indices=[10, 11, 12],  # capped indices for blocks 0,1,2
        block_texts=["a", "b", "c"],
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
    # misorder → capped run [11, 12], anchored on 11, not fixable.
    mis = modes["example_misordered_from_body"]
    assert mis.region_index == 11
    assert mis.proposed_regroup_run == (11, 12)
    assert mis.fixable is False


def test_judgments_drop_out_of_range_indices():
    window = reasoning_qc.QCWindow(
        page=1, region_indices=[5], block_texts=["a"], emit_positions=[0]
    )
    verdict = {"phantom_headings": [{"index": 99, "reason": "bogus"}], "misordered": [{"run": [0, 7]}]}
    flagged = reasoning_qc.judgments_to_flagged_blocks(verdict, window)
    # index 99 dropped; misorder run has only one valid capped index → dropped.
    assert flagged == []


# ---------------------------------------------------------------------------
# MOVE-op default shadow — a misorder flag is proposed, not applied (STEP-1
# stub already applies nothing; this pins the audit records the run).
# ---------------------------------------------------------------------------
def test_misorder_audited_not_applied(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.delenv("SEMANTIK_MOVE_OP", raising=False)  # default shadow
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"misordered": [{"run": [0, 1], "reason": "swap"}]}
        if page == 1
        else {},
    )
    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    # Misorder flag recorded with its capped run; merge/move applied=0.
    misorder = [f for f in audit["flagged"] if f["failure_mode"] == "example_misordered_from_body"]
    assert misorder and misorder[0]["proposed_regroup_run"] == [0, 1]
    assert audit["merge_move"]["applied"] == 0
    assert [r.kind for r in new_capped] == [r.kind for r in capped]  # order unchanged


# ---------------------------------------------------------------------------
# CHECK (A) — toc_reconcile emits reconcile ops with evidence, degrades on no ToC.
# ---------------------------------------------------------------------------
def test_toc_reconcile_flags_phantom_and_apparatus():
    feature_blocks = [
        _FB(_Raw("Table of Contents 1.1 Add 1.2 Subtract", 1)),
        _FB(_Raw("1.1 Add Whole Numbers", 2)),
        _FB(_Raw("Body prose for section 1.1.", 2)),
        _FB(_Raw("1.9 Phantom Heading", 3)),   # declared-absent ordinal, no body → phantom
        _FB(_Raw("1.9 Phantom Heading", 3)),   # (next region is a heading → no body)
        _FB(_Raw("Chapter Review", 4)),         # apparatus banner mistyped as heading
    ]
    capped = [
        _Region(kind="paragraph", feature_block_indices=(0,)),   # the ToC listing
        _Region(kind="heading", feature_block_indices=(1,)),     # 1.1 real (declared, has body)
        _Region(kind="paragraph", feature_block_indices=(2,)),   # body of 1.1
        _Region(kind="heading", feature_block_indices=(3,)),     # 1.9 phantom (no body)
        _Region(kind="heading", feature_block_indices=(4,)),     # another heading right after
        _Region(kind="heading", feature_block_indices=(5,)),     # Chapter Review apparatus
    ]
    toc = reasoning_qc.harvest_declared_toc_spine(capped, feature_blocks)
    flags = reasoning_qc.toc_reconcile(capped, feature_blocks, toc)
    by_idx = {f.region_index: f for f in flags}
    # 1.1 (region 1) is real → NOT flagged.
    assert 1 not in by_idx
    # 1.9 (region 3) is a declared-absent, body-less heading → phantom demote.
    assert by_idx[3].failure_mode == "example_as_heading"
    assert "1.9" in by_idx[3].fix_hint
    # Chapter Review (region 5) is an apparatus banner → re-type furniture.
    assert by_idx[5].failure_mode == "mistyped_component"


def test_toc_reconcile_no_spine_skips_gracefully():
    feature_blocks = [_FB(_Raw("1.1 Add Whole Numbers", 1))]
    capped = [_Region(kind="heading", feature_block_indices=(0,))]
    # Empty declared spine → skip with logged reason, return [] (never crash).
    logs = []
    flags = reasoning_qc.toc_reconcile(
        capped, feature_blocks, {"declared_ordinals": []}, log=logs.append
    )
    assert flags == []
    assert any("no declared ToC spine" in m for m in logs)


# ---------------------------------------------------------------------------
# CHECK (B) — page_order_verify degrades gracefully + synthesizes reorder.
# ---------------------------------------------------------------------------
def test_page_order_verify_skips_pageless_window():
    win = reasoning_qc.QCWindow(page=0, region_indices=[0], block_texts=["x"], emit_positions=[0])
    logs = []
    verdict, flagged, div = reasoning_qc.page_order_verify(object(), "/tmp/x.pdf", win, log=logs.append)
    assert verdict == {} and flagged == [] and div == 0
    assert any("no renderable page" in m for m in logs)


def test_page_order_verify_synthesizes_reorder_from_reading_order(monkeypatch):
    win = reasoning_qc.QCWindow(
        page=1, region_indices=[3, 4], block_texts=["a", "b"], emit_positions=[0, 1]
    )
    monkeypatch.setattr(
        reasoning_qc, "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"reading_order": [1, 0], "confidence": 0.7},
    )
    verdict, flagged, div = reasoning_qc.page_order_verify(object(), "/tmp/x.pdf", win)
    assert div == 2
    mis = [f for f in flagged if f.failure_mode == "example_misordered_from_body"]
    assert len(mis) == 1
    # capped run is the window indices in the VLM's proposed order [1,0] → [4,3].
    assert mis[0].proposed_regroup_run == (4, 3)
    assert mis[0].fixable is False


# ---------------------------------------------------------------------------
# STEP-2 apply channels — re-type lands via run_structure_review.
# ---------------------------------------------------------------------------
def test_step2_on_retype_demotes_phantom(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"phantom_headings": [{"index": 0, "reason": "hdr"}]}
        if page == 2
        else {},
    )

    captured = {}

    def _fake_review(regions, feature_blocks, runtime, *, restrict_to, feedback_by_idx):
        # The QC pass drives the reviewer with restrict_to + feedback.
        captured["restrict_to"] = set(restrict_to)
        captured["feedback"] = dict(feedback_by_idx)
        out = list(regions)
        for i in restrict_to:
            out[i] = _Region(kind="paragraph", feature_block_indices=out[i].feature_block_indices)
        return out, []

    monkeypatch.setattr(reasoning_qc, "_run_structure_review", _fake_review)
    # Token conservation is a no-op here (kinds change, FB partition unchanged).
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    # region 2 (page-2 heading) demoted to paragraph; FB indices preserved.
    assert captured["restrict_to"] == {2}
    assert new_capped[2].kind == "paragraph"
    assert new_capped[2].feature_block_indices == capped[2].feature_block_indices
    assert audit["retype"]["applied"] == 1


def test_step2_on_failclosed_conservation_revert(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"phantom_headings": [{"index": 0, "reason": "hdr"}]}
        if page == 2
        else {},
    )

    def _corrupting_review(regions, feature_blocks, runtime, *, restrict_to, feedback_by_idx):
        # Return a demoted region list, but token conservation will REJECT it.
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
    # Conservation raised → whole-revert to the input snapshot.
    assert audit["retype"]["applied"] == 0
    assert [r.kind for r in new_capped] == [r.kind for r in capped]


def test_step2_never_ship_worse(monkeypatch, _stub_seat):
    # A round where the reviewer DECLINES the correction (returns unchanged) must
    # ship the snapshot — never a spurious change.
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"phantom_headings": [{"index": 0, "reason": "hdr"}]}
        if page == 2
        else {},
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


def test_step2_move_live_reorders(monkeypatch, _stub_seat):
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"misordered": [{"run": [0, 1], "reason": "swap"}]}
        if page == 1
        else {},
    )

    sentinel = [_Region(kind="MERGED", feature_block_indices=(0, 1))]

    class _Op:
        op = "merge"

    def _fake_unit_fix(regions, fb, runs):
        # Under MOVE=live the reorder graduates through apply_proposed_unit_fix.
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
    assert [r.kind for r in new_capped] == ["MERGED"]  # the reordered/merged list is adopted
    assert audit["applied_ops"] == [{"op": "move"}]


def test_step2_move_live_conservation_revert(monkeypatch, _stub_seat):
    # Under MOVE=live, a conservation raise on the applied move → whole-revert.
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"misordered": [{"run": [0, 1], "reason": "swap"}]}
        if page == 1
        else {},
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
# STEP-3 required-case coverage (mock the model client — no real inference).
#
# Each of the seven STEP-3 acceptance cases pinned as its own test:
#   (a) phantom-chapter  → a DROP op (re-type to metadata_drop) lands;
#   (b) apparatus heading → a re-type op lands via the reviewer channel;
#   (c) scrambled order   → a reorder op (already covered above, cross-ref);
#   (d) HEALTHY input     → verdict pass, ZERO ops, capped byte-identical;
#   (e) flag OFF          → byte-identical no-op (ran=False, capped IS the input);
#   (f) missing page image → orchestrator skips the pageless window (no VLM call);
#   (g) DecisionCapture    → test_reasoning_qc_capture.py (cross-ref).
# ===========================================================================
def _retype_review(new_kind):
    """A fake reviewer that re-types every restricted region to ``new_kind``."""

    def _review(regions, feature_blocks, runtime, *, restrict_to, feedback_by_idx):
        out = list(regions)
        for i in restrict_to:
            out[i] = _Region(kind=new_kind, feature_block_indices=out[i].feature_block_indices)
        return out, []

    return _review


# --- (a) phantom-chapter → DROP op (re-type to metadata_drop) ---------------
def test_step3_phantom_chapter_drop_op(monkeypatch, _stub_seat):
    """A VLM-flagged phantom chapter heading is DROPPED (re-typed metadata_drop)."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    # Page-2 region (index 2, the running-header "heading") flagged phantom.
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"phantom_headings": [{"index": 0, "reason": "running header"}]}
        if page == 2
        else {},
    )
    monkeypatch.setattr(reasoning_qc, "_run_structure_review", _retype_review("metadata_drop"))
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    # The phantom heading region was DROPPED (re-typed to metadata_drop).
    assert new_capped[2].kind == "metadata_drop"
    # Partition-immutable: FB indices survive the drop (verbatim source rides assembly).
    assert new_capped[2].feature_block_indices == capped[2].feature_block_indices
    assert audit["retype"]["applied"] == 1
    # Audit records a phantom flag from the VLM source.
    assert any(
        f["failure_mode"] == "example_as_heading" and f["region_index"] == 2
        for f in audit["flagged"]
    )


# --- (b) apparatus heading → re-type op -------------------------------------
def test_step3_apparatus_heading_retype_op(monkeypatch, _stub_seat):
    """A VLM-flagged apparatus heading is re-typed via the reviewer channel."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    # Page-2 region (index 2) flagged apparatus (exercise/answer-key furniture).
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"apparatus_retype": [{"index": 0, "reason": "answer key banner"}]}
        if page == 2
        else {},
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
    # The apparatus flag routed through the re-type channel with a feedback hint.
    assert captured["restrict_to"] == {2}
    assert "re-type apparatus" in captured["feedback"][2]
    assert new_capped[2].kind == "paragraph"
    assert audit["retype"]["applied"] == 1
    assert any(f["failure_mode"] == "mistyped_component" for f in audit["flagged"])


# --- (c) scrambled page order → reorder op (cross-ref) ----------------------
def test_step3_scrambled_order_reorder_op_crossref(monkeypatch, _stub_seat):
    """Scrambled reading order emits a reorder op under MOVE=live (STEP-3 case c)."""
    # A focused restatement of test_step2_move_live_reorders, pinned to case (c).
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda seat, pdf, page, texts: {"reading_order": [1, 0], "confidence": 0.7}
        if page == 1
        else {},
    )
    reordered = [_Region(kind="REORDERED", feature_block_indices=(1, 0))]

    class _Op:
        op = "move"

    monkeypatch.setattr(
        reasoning_qc,
        "_apply_proposed_unit_fix",
        lambda regions, fb, runs: (reordered, [_Op()]),
    )
    monkeypatch.setattr(reasoning_qc, "_assert_partition_conservation", lambda *a, **k: None)
    monkeypatch.setattr(reasoning_qc, "_assert_token_conservation", lambda *a, **k: None)
    monkeypatch.setattr(reasoning_qc, "_build_resegment_audit_rows", lambda ops: [{"op": "move"}])

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    # A reorder op was proposed from the reading_order permutation AND applied.
    assert any(
        f["failure_mode"] == "example_misordered_from_body" for f in audit["flagged"]
    )
    assert audit["merge_move"]["applied"] == 1
    assert [r.kind for r in new_capped] == ["REORDERED"]


# --- (d) HEALTHY input → verdict pass, zero ops -----------------------------
def test_step3_healthy_input_zero_ops(monkeypatch, _stub_seat):
    """A clean document (every verdict empty, no ToC phantom) → zero ops, byte-identical."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "on")
    # The VLM finds nothing wrong on any page.
    monkeypatch.setattr(reasoning_qc, "_run_qc_judgment", lambda seat, pdf, page, texts: {})
    # A reviewer that must NEVER be driven when there is nothing to fix.
    called = {"review": False, "unit_fix": False}

    def _no_review(*a, **k):
        called["review"] = True
        raise AssertionError("reviewer must not be driven on a healthy document")

    def _no_unit_fix(*a, **k):
        called["unit_fix"] = True
        raise AssertionError("unit-fix must not be driven on a healthy document")

    monkeypatch.setattr(reasoning_qc, "_run_structure_review", _no_review)
    monkeypatch.setattr(reasoning_qc, "_apply_proposed_unit_fix", _no_unit_fix)

    capped, fbs, assembled, order = _simple_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_runtime=object()
    )
    assert audit["ran"] is True
    assert audit["flagged"] == []  # verdict pass — nothing flagged
    assert audit["retype"] == {"applied": 0, "proposed": 0}
    assert audit["merge_move"] == {"applied": 0, "proposed": 0}
    assert audit["toc_reconcile"] == {"proposed": 0}
    assert called == {"review": False, "unit_fix": False}
    # Byte-identical: never-ship-worse returns the input snapshot content.
    assert [r.kind for r in new_capped] == [r.kind for r in capped]
    assert [r.feature_block_indices for r in new_capped] == [
        r.feature_block_indices for r in capped
    ]


# --- (e) flag OFF → byte-identical no-op ------------------------------------
def test_step3_flag_off_byte_identical_noop(monkeypatch):
    """With the flag OFF the (defensively direct-called) pass is a byte-identical no-op."""
    monkeypatch.delenv("SEMANTIK_REASONING_QC", raising=False)
    # A seat resolver that MUST NOT be reached when the flag is off.
    monkeypatch.setattr(
        reasoning_qc,
        "_resolve_qc_seat",
        lambda: (_ for _ in ()).throw(AssertionError("seat resolved with flag OFF")),
    )
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("VLM called with flag OFF")),
    )
    capped, fbs, assembled, order = _simple_doc()
    verdicts_in = object()
    new_capped, verdicts_out, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf", review_verdicts=verdicts_in
    )
    assert new_capped is capped  # SAME object — byte-identical
    assert verdicts_out is verdicts_in  # untouched
    assert audit["ran"] is False and audit["mode"] == "off"


# --- (f) missing page image → orchestrator skips the pageless window --------
def _pageless_doc():
    """A doc whose sole region has no resolvable physical page → a pageless window."""
    feature_blocks = [_FB(_Raw("Body text with no page number.", None))]
    capped = [_Region(kind="paragraph", feature_block_indices=(0,))]
    order = [0]
    return capped, feature_blocks, _Assembled(order), order


def test_step3_missing_page_image_window_skipped(monkeypatch, _stub_seat):
    """A window whose regions have no renderable page is recorded skipped, no VLM call."""
    monkeypatch.setenv("SEMANTIK_REASONING_QC", "shadow")
    monkeypatch.setattr(
        reasoning_qc,
        "_run_qc_judgment",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("VLM called for a pageless window")),
    )
    capped, fbs, assembled, order = _pageless_doc()
    new_capped, _v, audit = reasoning_qc.run_reasoning_qc(
        capped, fbs, assembled, order, pdf_path="/tmp/x.pdf"
    )
    assert new_capped is capped  # shadow → byte-identical
    assert audit["ran"] is True
    # The single window (page 0) is recorded as skipped; nothing flagged.
    assert audit["windows"] == [{"page": 0, "regions": [0], "skipped": True}]
    assert audit["flagged"] == []
    assert audit["capture_fired"] == 0
