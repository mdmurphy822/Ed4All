"""Phase-2 acceptance — Stage-5d cascade wiring + cap-safety + provenance + audit.

Covers the §9 Phase-2 acceptance list verbatim (mocked endpoint, NO GPU):

  * flag-OFF byte-identical: with SEMANTIK_STRUCTURE_REVIEW unset, the
    regions / provenance / audit are byte-identical to a no-Stage-5d
    baseline (direct equality against a no-verdicts call).
  * flag-ON propagation: a corrected kind flows through
    ``_build_region_provenance`` into the region_provenance entry; a
    paragraph->heading promotion yields non-null ``heading_text``.
  * cap-safety (C2): a re-role (phantom heading->paragraph or
    ->metadata_drop) that WOULD evict a real paragraph at the cap fails
    closed / the real FB survives.
  * audit: ``build_conformance_audit`` carries ``structure_review`` when
    provided + ``None`` when off; ``AUDIT_REQUIRED_KEYS`` unchanged.
  * smallest gated-insertion integration check: the Stage-5d helpers,
    driven in the SAME order ``run_full_cascade`` calls them (review ->
    metadata_drop-id capture -> cap with exemption -> FB-survival assert
    -> provenance + audit), and the ``resolve_structure_review_mode`` gate.

``semantik_structure.cascade`` imports ``semantik_structure.validate`` which imports
``axe_playwright_python`` + ``playwright`` — both ABSENT from the slim test
venv (a PRE-EXISTING, unrelated blocker). The helpers under test never run
axe; we stub the two modules for IMPORT ONLY so the cascade module loads.
The stub is import-scaffold, not behaviour.
"""

from __future__ import annotations

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Import scaffold — stub axe_playwright_python + playwright so semantik_structure.
# cascade can be imported in the slim venv (PRE-EXISTING blocker, unrelated to
# Stage 5d). The helpers under test never invoke axe.
# ---------------------------------------------------------------------------


def _install_axe_stubs() -> None:
    if "axe_playwright_python" not in sys.modules:
        pkg = types.ModuleType("axe_playwright_python")
        sub = types.ModuleType("axe_playwright_python.sync_playwright")
        sub.Axe = object  # only referenced at import time
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

from semantik_structure import cascade  # noqa: E402
from semantik_structure.cascade import (  # noqa: E402
    CapSafetyError,
    _assert_fb_survival_through_cap,
    _build_region_provenance,
    _cap_regions_per_kind,
    _review_by_region_index,
    _surviving_fb_indices,
)
from semantik_structure.conformance_audit import (  # noqa: E402
    AUDIT_REQUIRED_KEYS,
    build_conformance_audit,
)
from semantik_structure.qwen_specialists.reviewer import (  # noqa: E402
    ReviewVerdict,
    resolve_structure_review_mode,
    run_structure_review,
)
from semantik_structure.structure_graph import Region  # noqa: E402
from semantik_structure.types import FeatureBlock, RawBlock  # noqa: E402


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _fb(text: str) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=1,
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


def _region(kind, fb_idx, *, text=None, level=None):
    payload = {}
    if text is not None:
        payload["text"] = text
    if level is not None:
        payload["level_hint"] = level
    return Region(kind=kind, feature_block_indices=(fb_idx,), payload=payload)


class _ScriptedRuntime:
    """Injectable runtime; generate_batch returns crafted JSON strings."""

    def __init__(self, completions):
        self._completions = completions
        self.batch_calls = []

    def generate_batch(self, prompts, *, max_tokens, temperature=0.6,
                        top_p=0.95, seed=None, repeat_penalty=1.0):
        self.batch_calls.append({"prompts": list(prompts)})
        return list(self._completions)


def _vjson(block_id, *, kind=None, level=None, verdict="corrected", note="n"):
    import json
    return json.dumps({
        "block_id": block_id,
        "verdict": verdict,
        "corrected_kind": kind,
        "corrected_level": level,
        "corrected_doc_role": None,
        "review_note": note,
    })


# ---------------------------------------------------------------------------
# Gate resolver (insertion-point gate).
# ---------------------------------------------------------------------------


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW", raising=False)
    assert resolve_structure_review_mode() is False


def test_gate_on_when_truthy(monkeypatch):
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW", "on")
    assert resolve_structure_review_mode() is True


# ---------------------------------------------------------------------------
# flag-OFF byte-identical.
# ---------------------------------------------------------------------------


def test_provenance_flag_off_byte_identical():
    """With review_verdicts=None (flag OFF), provenance == no-verdicts call."""
    regions = [_region("heading", 0, text="Chapter 1", level=1), _region("paragraph", 1)]
    fbs = [_fb("Chapter 1"), _fb("body text here")]
    order = [0, 1]

    base = _build_region_provenance(order, regions, fbs, {})
    off = _build_region_provenance(order, regions, fbs, {}, review_verdicts=None)

    assert off == base
    # No 'review' key ever appears when off.
    assert all("review" not in e for e in off)


def test_cap_flag_off_byte_identical():
    """metadata_drop ids None -> cap is byte-identical to the historical cap."""
    regions = [_region("paragraph", i) for i in range(5)]
    base = _cap_regions_per_kind(regions, cap=3)
    off = _cap_regions_per_kind(regions, cap=3, stage5d_metadata_drop_ids=None)
    assert [r.kind for r in off] == [r.kind for r in base]
    assert len(off) == 3


def test_audit_structure_review_none_when_off():
    """build_conformance_audit defaults structure_review to None (reviewer off)."""
    audit = _minimal_audit(structure_review=None)
    assert audit["structure_review"] is None
    # Required-keys schema unchanged (not in AUDIT_REQUIRED_KEYS).
    assert "structure_review" not in AUDIT_REQUIRED_KEYS
    for k in AUDIT_REQUIRED_KEYS:
        assert k in audit


# ---------------------------------------------------------------------------
# flag-ON propagation.
# ---------------------------------------------------------------------------


def test_corrected_kind_flows_into_provenance():
    """A heading->paragraph re-role (corrected kind) appears in region_provenance."""
    regions = [_region("paragraph", 0), _region("heading", 1, text="Answer Key", level=2)]
    fbs = [_fb("intro"), _fb("Answer Key")]
    # Verdict: region 1 heading -> metadata_drop (phantom).
    verdicts = [
        ReviewVerdict(0, "ok", "paragraph", "paragraph", None, None, ""),
        ReviewVerdict(1, "drop_injected_header", "heading", "metadata_drop", 2, None, "phantom"),
    ]
    # The corrected region (kind already mutated upstream by run_structure_review).
    corrected = [regions[0], _region("metadata_drop", 1, text="Answer Key")]
    prov = _build_region_provenance([0, 1], corrected, fbs, {}, review_verdicts=verdicts)

    entry = next(e for e in prov if e["region_index"] == 1)
    assert entry["region_kind"] == "metadata_drop"
    assert "review" in entry
    assert entry["review"]["corrected_from"] == "heading"
    assert entry["review"]["corrected_to"] == "metadata_drop"
    # heading->metadata_drop nulls heading_text for free.
    assert entry["heading_text"] is None


def test_promotion_yields_non_null_heading_text():
    """A paragraph->heading promotion (C1) yields non-null heading_text.

    NOTE (heading-scoping): the driver reviews ONLY heading regions, so a
    paragraph->heading PROMOTION is no longer driver-reachable (a paragraph is
    never sent to the 70B). We exercise the C1 promotion-text logic DIRECTLY
    via ``_apply_verdict`` (mirroring ``test_reviewer.test_promotion_sets_
    source_text_c1``) and then assert the correction flows into
    ``_build_region_provenance`` with non-null heading_text.
    """
    from semantik_structure.qwen_specialists import reviewer as rv

    regions = [_region("paragraph", 0), _region("paragraph", 1)]
    fbs = [_fb("intro para"), _fb("Section Two")]

    # Apply a paragraph->heading promotion to region 1 directly (the C1 path).
    obj = {
        "block_id": 1, "verdict": "corrected", "corrected_kind": "heading",
        "corrected_level": 2, "corrected_doc_role": None,
        "review_note": "missed heading",
    }
    promoted, verdict1 = rv._apply_verdict(regions[1], 1, obj, fbs)

    # The promoted region carries payload['text'] == source (C1).
    assert promoted.kind == "heading"
    assert (promoted.payload or {}).get("text") == "Section Two"

    verdicts = [
        ReviewVerdict(0, "ok", "paragraph", "paragraph", None, None, ""),
        verdict1,
    ]
    corrected = [regions[0], promoted]

    prov = _build_region_provenance([0, 1], corrected, fbs, {}, review_verdicts=verdicts)
    entry = next(e for e in prov if e["region_index"] == 1)
    assert entry["region_kind"] == "heading"
    assert entry["heading_text"] == "Section Two"  # non-null, == source
    assert entry["review"]["corrected_to"] == "heading"


# ---------------------------------------------------------------------------
# cap-safety (C2 / R12).
# ---------------------------------------------------------------------------


def test_metadata_drop_is_cap_exempt():
    """A Stage-5d metadata_drop re-tag does NOT consume a content-bucket slot."""
    # 3 paragraphs + 1 phantom re-roled to metadata_drop at index 1. cap=2.
    regions = [
        _region("paragraph", 0),
        _region("metadata_drop", 1),  # re-roled this stage (exempt)
        _region("paragraph", 2),
        _region("paragraph", 3),
    ]
    # Without exemption, the metadata_drop region would not count against
    # the paragraph bucket anyway (different kind) — so to make the point,
    # build the realistic case: the phantom was a heading re-roled to
    # PARAGRAPH (counts against paragraph bucket) vs metadata_drop (exempt).
    regions_para = [
        _region("paragraph", 0),
        _region("paragraph", 1),  # phantom re-roled to paragraph
        _region("paragraph", 2),
        _region("paragraph", 3),
    ]
    # cap=3, no exemption: only 3 survive; FB 3 (real) evicted.
    capped_para = _cap_regions_per_kind(regions_para, cap=3)
    survived = {i for r in capped_para for i in r.feature_block_indices}
    assert 3 not in survived  # real paragraph FB evicted under naive cap

    # With the phantom (index 1) re-roled to metadata_drop AND cap-exempt,
    # the 3 real paragraphs (0,2,3) all fit the cap=3 budget; the drop rides
    # along without consuming a slot.
    regions_drop = [
        _region("paragraph", 0),
        _region("metadata_drop", 1),
        _region("paragraph", 2),
        _region("paragraph", 3),
    ]
    capped_drop = _cap_regions_per_kind(
        regions_drop, cap=3, stage5d_metadata_drop_ids=frozenset({1})
    )
    survived_drop = {i for r in capped_drop for i in r.feature_block_indices}
    assert {0, 2, 3} <= survived_drop  # every real paragraph survives


def test_fb_survival_assertion_fails_closed_on_eviction():
    """A re-role that evicts a real paragraph at the cap fails closed (C2)."""
    # Baseline (flag-OFF): 1 heading + 2 paragraphs, cap=2 -> all survive
    # (heading bucket=1, paragraph bucket=2).
    baseline = [
        _region("heading", 0, text="H", level=1),
        _region("paragraph", 1),
        _region("paragraph", 2),
    ]
    baseline_capped = _cap_regions_per_kind(baseline, cap=2)
    base_survivors = _surviving_fb_indices(baseline_capped)
    assert set(base_survivors) == {0, 1, 2}

    # Reviewed: the heading is re-roled to paragraph WITHOUT cap-exemption
    # (heading->paragraph is NOT metadata_drop, so it counts). Now the
    # paragraph bucket has 3 members at cap=2 -> the LAST real paragraph
    # (FB 2) is evicted.
    reviewed = [
        _region("paragraph", 0),  # was heading, re-roled to body
        _region("paragraph", 1),
        _region("paragraph", 2),
    ]
    reviewed_capped = _cap_regions_per_kind(reviewed, cap=2)  # no exemption
    rev_survivors = _surviving_fb_indices(reviewed_capped)
    assert 2 not in rev_survivors  # real FB 2 lost to the re-role

    with pytest.raises(CapSafetyError):
        _assert_fb_survival_through_cap(baseline_capped, reviewed_capped)


def test_fb_survival_assertion_passes_when_no_eviction():
    """No eviction -> the FB-survival assertion is a clean pass."""
    baseline = [_region("heading", 0, text="H", level=1), _region("paragraph", 1)]
    capped = _cap_regions_per_kind(baseline, cap=5)
    # Same FBs survive -> no error.
    _assert_fb_survival_through_cap(capped, capped)


# ---------------------------------------------------------------------------
# audit: structure_review carried when provided.
# ---------------------------------------------------------------------------


def test_audit_carries_structure_review_when_provided():
    sr = [{"block_id": 0, "verdict": "corrected", "kind_before": "heading",
           "kind_after": "metadata_drop", "review_note": "phantom"}]
    audit = _minimal_audit(structure_review=sr)
    assert audit["structure_review"] == sr
    # And the required-keys schema is still complete.
    for k in AUDIT_REQUIRED_KEYS:
        assert k in audit


# ---------------------------------------------------------------------------
# Smallest gated-insertion integration: drive the helpers in cascade order.
# ---------------------------------------------------------------------------


def test_seam_helpers_in_cascade_order_flag_on():
    """Exercise review -> drop-id capture -> cap(exempt) -> survival -> prov.

    This mirrors the exact helper sequence the Stage-5d seam runs inside
    ``run_full_cascade`` (without pulling council/extraction), proving the
    wired helpers compose: a phantom heading re-roled to metadata_drop is
    cap-exempt, the real content survives the cap, and the correction lands
    in region_provenance.
    """
    structure_regions = [
        _region("paragraph", 0),
        _region("heading", 1, text="Answer Key 3.2", level=2),  # phantom
        _region("paragraph", 2),
    ]
    fbs = [_fb("intro"), _fb("Answer Key 3.2"), _fb("real body")]
    pre_review_regions = list(structure_regions)

    # Heading-scoping: ONLY the heading region (index 1) is sent to the 70B,
    # so the runtime is scripted with exactly one completion whose block_id is
    # the heading's region index (1). The two paragraphs pass through verbatim
    # with verdict='ok' and never reach the runtime.
    runtime = _ScriptedRuntime([
        _vjson(1, kind="metadata_drop", verdict="drop_injected_header"),
    ])
    reviewed, verdicts = run_structure_review(structure_regions, fbs, runtime)

    # metadata_drop-id capture (cascade snippet).
    drop_ids = frozenset(
        i for i, (b, a) in enumerate(zip(pre_review_regions, reviewed))
        if a.kind == "metadata_drop" and b.kind != "metadata_drop"
    )
    assert drop_ids == frozenset({1})

    # cap with exemption.
    capped = _cap_regions_per_kind(reviewed, cap=2, stage5d_metadata_drop_ids=drop_ids)
    baseline_capped = _cap_regions_per_kind(pre_review_regions, cap=2)
    # FB-survival holds (no real content evicted).
    _assert_fb_survival_through_cap(baseline_capped, capped)

    # provenance carries the review of the phantom drop.
    order = list(range(len(capped)))
    # region_order indexes into `capped`; build a positional order map.
    prov = _build_region_provenance(order, capped, fbs, {}, review_verdicts=verdicts)
    drop_entries = [e for e in prov if e.get("region_kind") == "metadata_drop"]
    assert drop_entries, "phantom drop should appear in provenance"
    assert drop_entries[0]["review"]["corrected_to"] == "metadata_drop"


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _minimal_audit(*, structure_review):
    """Build a minimal conformance audit exercising the structure_review param.

    Hand-builds the smallest gate/theta/assembly stand-ins build_conformance_audit
    reads — no model runs, no axe. Mirrors what test_conformance_audit would feed.
    """
    class _Check:
        check = "axe"
        passed = True
        skipped = False
        message = ""
        details = {}

    class _GateResult:
        stage = "document"
        passed = True
        checks = [_Check()]

    class _Theta:
        # dataclasses.is_dataclass -> False so build uses dict() branch.
        def keys(self):  # make dict(theta_report) work
            return self._d.keys()

        def __getitem__(self, k):
            return self._d[k]

        _d = {
            "wcag_status": "passed",
            "flags": [],
            "retry_history": [],
            "dimensions": {},
            "theta_version": "test",
        }

    class _Assembled:
        gaps_found = []
        gaps_resolved = []
        gaps_fallback = []
        heading_tree = []
        landmarks = {}
        html = "<html></html>"

    return build_conformance_audit(
        pdf_path="x.pdf",
        runtime_mode="mock",
        k=2,
        lane_used="fast",
        offline_retry_fired=False,
        exit_action="ship",
        region_gate_results={},
        regions=[],
        stage7_skip_counts={},
        stage7_violation_counts={},
        n_candidates_in=0,
        n_candidates_out=0,
        n_regions_no_survivor=0,
        doc_gate_result=_GateResult(),
        theta_report=_Theta(),
        assembled=_Assembled(),
        wcag_coverage=None,
        structure_review=structure_review,
    )


# ---------------------------------------------------------------------------
# Phase 5 — index-stability: re-key the cap exemption + verdict->provenance
# join by stable min-FB so they survive a Stage-5e N->1 region merge.
# ---------------------------------------------------------------------------


def _mregion(kind, fb_indices, *, text=None):
    """A region spanning MULTIPLE FBs (the merged-unit shape)."""
    payload = {}
    if text is not None:
        payload["text"] = text
    return Region(kind=kind, feature_block_indices=tuple(fb_indices), payload=payload)


def test_metadata_drop_exempt_survives_regroup():
    """Stage-5d metadata_drop + a Stage-5e regroup that shifts positions: the
    cap exemption is keyed by the metadata_drop's STABLE min-FB, so it still
    lands on it after the N->1 merge shifts its position — the drop survives
    the (full) metadata bucket and no real-content FB is evicted (no spurious
    CapSafetyError / revert)."""
    # pre-5e (post-5d) reviewed list: a 2-region content unit + a pre-existing
    # metadata_drop + a NEWLY-dropped metadata region.
    before = [
        _region("paragraph", 0, text="EXAMPLE 1"),
        _region("paragraph", 1, text="body a"),
        _region("metadata_drop", 2, text="pre-existing junk"),
        _region("heading", 3, text="Answer Key"),  # becomes metadata_drop
    ]
    after = [
        _region("paragraph", 0, text="EXAMPLE 1"),
        _region("paragraph", 1, text="body a"),
        _region("metadata_drop", 2, text="pre-existing junk"),
        _region("metadata_drop", 3, text="Answer Key"),  # NEWLY dropped
    ]
    # Cascade Phase-5 exempt build: keyed by each NEWLY-dropped region's min-FB.
    drop_ids = frozenset(
        mf
        for b, a in zip(before, after)
        if a.kind == "metadata_drop"
        and b.kind != "metadata_drop"
        and (mf := cascade._region_min_fb(a)) is not None
    )
    assert drop_ids == frozenset({3})  # the newly-dropped region's min-FB

    # A Stage-5e regroup fuses the 2 content regions into ONE (fb 0,1); both
    # metadata_drops shift LEFT by one position. The newly-dropped region moves
    # 3 -> 2, but its min-FB (3) is unchanged.
    post_5e = [
        _mregion("paragraph", (0, 1), text="EXAMPLE 1 body a"),  # merged unit
        after[2],  # pre-existing metadata_drop (fb2), NOT exempt
        after[3],  # newly-dropped metadata_drop (fb3), exempt by min-FB
    ]
    assert cascade._region_min_fb(post_5e[2]) == 3  # stale position 3 -> now 2

    cap = 1  # tiny: the metadata bucket holds ONE unless exempt
    capped = _cap_regions_per_kind(
        post_5e, cap=cap, stage5d_metadata_drop_ids=drop_ids
    )
    survivors = {i for r in capped for i in r.feature_block_indices}
    # The newly-dropped region (fb3) survives ONLY via the min-FB exemption
    # (the pre-existing drop already consumed the cap=1 metadata bucket slot).
    assert 3 in survivors

    # FB-survival vs the pre-5e baseline holds (no real content evicted) -> the
    # C2 cap-safety assertion is a clean pass (no spurious revert).
    baseline = _cap_regions_per_kind(before, cap=cap)
    _assert_fb_survival_through_cap(baseline, capped)  # must NOT raise


def test_cap_exempt_byte_identical_no_merge():
    """No merge fires -> the min-FB-keyed exemption is byte-identical to the
    historical positional exemption (each region's min-FB == its index)."""
    regions = [
        _region("paragraph", 0),
        _region("metadata_drop", 1),
        _region("paragraph", 2),
        _region("paragraph", 3),
    ]
    # Position == min-FB for every region (no merge), so the min-FB exempt set
    # {1} targets the same region a positional {1} would.
    for i, r in enumerate(regions):
        assert cascade._region_min_fb(r) == i
    capped = _cap_regions_per_kind(
        regions, cap=3, stage5d_metadata_drop_ids=frozenset({1})
    )
    survived = {i for r in capped for i in r.feature_block_indices}
    # Every real paragraph survives; the exempt metadata_drop rides along
    # without consuming a paragraph slot — identical to the positional baseline.
    assert {0, 2, 3} <= survived
    assert 1 in survived


def test_review_provenance_attaches_by_min_fb():
    """A Stage-5d verdict + a Stage-5e regroup: the review sub-block lands on
    the post-merge region whose stable min-FB matches the verdict's original
    block (not its stale positional index)."""
    fbs = [_fb("EXAMPLE 1"), _fb("body a"), _fb("body b"), _fb("tail")]
    # pre-5e regions the verdict block_ids index into. Region 0 (the anchor)
    # was corrected (heading -> paragraph).
    pre_review = [
        _region("paragraph", 0, text="EXAMPLE 1"),
        _region("paragraph", 1, text="body a"),
        _region("paragraph", 2, text="body b"),
        _region("paragraph", 3, text="tail"),
    ]
    verdicts = [
        ReviewVerdict(0, "corrected", "heading", "paragraph", 2, None, "demoted"),
    ]
    # A regroup fuses pre[0..2] into ONE region (the merged unit inherits the
    # anchor's min-FB = 0); the tail stays separate. Post positions shift.
    post = [
        _mregion("paragraph", (0, 1, 2), text="EXAMPLE 1 body a body b"),
        pre_review[3],
    ]
    order = list(range(len(post)))
    prov = _build_region_provenance(
        order, post, fbs, {},
        review_verdicts=verdicts,
        review_regions=pre_review,
    )
    merged_entry = next(e for e in prov if e["first_raw_block_index"] == 0)
    assert "review" in merged_entry
    assert merged_entry["review"]["corrected_from"] == "heading"
    assert merged_entry["review"]["corrected_to"] == "paragraph"
    # The tail region (min-FB 3) carries NO review (the verdict was for fb0).
    tail_entry = next(e for e in prov if e["first_raw_block_index"] == 3)
    assert "review" not in tail_entry


def test_review_join_byte_identical_no_merge():
    """No merge: the min-FB-rekeyed verdict->region join is byte-identical to
    the legacy positional join (region_index == block_id == min-FB)."""
    fbs = [_fb("intro"), _fb("Answer Key")]
    regions = [
        _region("paragraph", 0),
        _region("metadata_drop", 1, text="Answer Key"),
    ]
    verdicts = [
        ReviewVerdict(0, "ok", "paragraph", "paragraph", None, None, ""),
        ReviewVerdict(1, "drop_injected_header", "heading", "metadata_drop", 2, None, "phantom"),
    ]
    order = [0, 1]
    legacy = _build_region_provenance(
        order, regions, fbs, {}, review_verdicts=verdicts
    )  # review_regions=None -> positional join
    rekeyed = _build_region_provenance(
        order, regions, fbs, {}, review_verdicts=verdicts, review_regions=regions
    )  # min-FB join
    assert rekeyed == legacy
