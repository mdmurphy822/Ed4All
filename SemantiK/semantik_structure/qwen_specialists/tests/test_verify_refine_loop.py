"""Phase-6 acceptance — bounded, fail-safe verify-refine loop (SEMANTIK_SECOND_PASS).

Exercises ``cascade._verify_refine_loop`` directly with a scripted endpoint
(monkeypatched ``run_second_pass_verify`` / ``run_structure_review``) + a
faithful injected ``run_inner`` that mirrors the cascade's re-assemble contract
(assembles the PASSED ``regions`` into a fake assembled doc, writes
``lane_outputs['fast']``). GPU-free — no council / extraction / axe.

The two ``_run_inner``-refactor tests assert the closure-rebind structurally
(every ``capped`` read became ``regions``) + the re-assemble-reflects-retype
contract through the ``regions=``-parameterized re-assemble primitive.

``semantik_structure.cascade`` imports ``semantik_structure.validate`` which pulls
``axe_playwright_python`` + ``playwright`` (absent from the slim venv); we stub
them for IMPORT ONLY (mirrors ``test_cascade_stage5d.py``).
"""

from __future__ import annotations

import inspect
import re
import sys
import types
from dataclasses import replace

import pytest


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

from semantik_structure import cascade  # noqa: E402
from semantik_structure.cascade import _build_region_provenance, _verify_refine_loop  # noqa: E402
from semantik_structure.qwen_specialists import reviewer as rv  # noqa: E402
from semantik_structure.qwen_specialists.reviewer import (  # noqa: E402
    FlaggedBlock,
    ReviewVerdict,
    VerifierVerdict,
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


def _region(kind: str, fb_idx: int) -> Region:
    return Region(kind=kind, feature_block_indices=(fb_idx,), payload={})


_TAG_FOR_KIND = {
    "heading": "h2",
    "paragraph": "p",
    "math": "div",
    "table": "table",
    "metadata_drop": "p",
}


def _tag(kind: str) -> str:
    return _TAG_FOR_KIND.get(kind, "p")


class _FakeAssembled:
    """A minimal AssembledDoc stand-in slice_region_html can read.

    The per-region fragment's TAG reflects the region kind (so a re-type is
    visible) while its VISIBLE TEXT ("body N") is kind-INDEPENDENT (so a
    legitimate re-tag conserves document text — the fail-safe never trips on a
    real re-type).
    """

    def __init__(self, kinds: list[str]):
        self.region_provenance = list(range(len(kinds)))
        self._frags = [f"<{_tag(k)}>body {i}</{_tag(k)}>" for i, k in enumerate(kinds)]
        self.sub_task_log = {"region_html": list(self._frags)}
        self.html = "<html>" + "".join(self._frags) + "</html>"
        self.heading_tree = []
        self.landmarks = {}


class _Gate:
    def __init__(self, passed: bool = True):
        self.passed = passed


class _Report:
    def __init__(self, theta_score: float | None = 0.9, lane: str = "fast"):
        self.theta_score = theta_score
        self.lane = lane


def _lane_dict(assembled: _FakeAssembled, *, gate_passed: bool = True) -> dict:
    return {
        "assembled": assembled,
        "gate_result": _Gate(gate_passed),
        "wcag_status": "passed",
        "stage7_results": {},
        "n_cand_in": 0,
        "n_cand_out": 0,
    }


def _make_run_inner(lane_outputs: dict, *, gate_passed: bool = True, theta: float = 0.9):
    """A faithful injected ``run_inner`` mirroring the cascade contract: it
    assembles the PASSED ``regions`` (the closure-rebind seam) and overwrites
    ``lane_outputs[lane]``."""
    calls: list[dict] = []

    def run_inner(lane: str, regions=None):
        calls.append({"lane": lane, "regions": None if regions is None else list(regions)})
        kinds = [r.kind for r in (regions or [])]
        lane_outputs[lane] = _lane_dict(_FakeAssembled(kinds), gate_passed=gate_passed)
        return _Report(theta_score=theta, lane=lane)

    run_inner.calls = calls  # type: ignore[attr-defined]
    return run_inner


class _ScriptedReviewer:
    """Scripted Pass-2 verifier + Pass-1 re-driver.

    ``defects`` maps a capped region index -> (bad_kind, failure_mode, fixable,
    fix_hint): the verifier FLAGS the region while it still carries ``bad_kind``.
    ``fix_to`` maps index -> the kind ``run_structure_review`` re-types it to
    (when in ``restrict_to``). ``drift`` optionally re-types an UNFLAGGED region
    too (to exercise the not-better revert). ``no_fix`` makes the re-driver a
    no-op (the verifier never clears -> the bounded-rounds path).
    """

    def __init__(self, defects, fix_to=None, *, drift=None, no_fix=False, raise_on_verify=False):
        self.defects = defects
        self.fix_to = fix_to or {}
        self.drift = drift or {}
        self.no_fix = no_fix
        self.raise_on_verify = raise_on_verify
        self.verify_calls: list[dict] = []
        self.review_calls: list[dict] = []

    def verify(self, assembled, regions, feature_blocks, runtime, *, council_state=None):
        self.verify_calls.append({"regions": [r.kind for r in regions]})
        if self.raise_on_verify:
            raise RuntimeError("scripted verifier boom")
        flagged = []
        for idx, (bad_kind, mode, fixable, hint) in self.defects.items():
            if idx < len(regions) and regions[idx].kind == bad_kind:
                flagged.append(
                    FlaggedBlock(region_index=idx, failure_mode=mode, fix_hint=hint, fixable=fixable)
                )
        if flagged:
            return VerifierVerdict(passed=False, flagged=tuple(flagged))
        return VerifierVerdict(passed=True, flagged=())

    def review(
        self,
        regions,
        feature_blocks,
        runtime,
        *,
        max_tokens=512,
        council_state=None,
        restrict_to=None,
        feedback_by_idx=None,
    ):
        self.review_calls.append(
            {"restrict_to": restrict_to, "feedback": dict(feedback_by_idx or {})}
        )
        new = list(regions)
        if not self.no_fix:
            for idx, good in self.fix_to.items():
                if restrict_to is None or idx in restrict_to:
                    new[idx] = replace(new[idx], kind=good)
            for idx, drifted in self.drift.items():
                new[idx] = replace(new[idx], kind=drifted)
        # One capped-space ReviewVerdict per input region (block_id = capped idx).
        verdicts = []
        for i, (before, after) in enumerate(zip(regions, new)):
            changed = before.kind != after.kind
            verdicts.append(
                ReviewVerdict(
                    i,
                    "corrected" if changed else "ok",
                    before.kind,
                    after.kind,
                    None,
                    None,
                    "scripted",
                )
            )
        return new, verdicts


def _patch(monkeypatch, scripted: _ScriptedReviewer, *, rounds: int = 2):
    monkeypatch.setattr(rv, "run_second_pass_verify", scripted.verify)
    monkeypatch.setattr(rv, "run_structure_review", scripted.review)
    monkeypatch.setattr(rv, "resolve_second_pass_rounds", lambda: rounds)


def _drive(capped, lane_outputs, scripted, *, run_inner=None, pass1_verdicts=None):
    run_inner = run_inner or _make_run_inner(lane_outputs)
    fbs = [_fb(f"body {i}") for i in range(len(capped))]
    pass1_verdicts = pass1_verdicts if pass1_verdicts is not None else [
        ReviewVerdict(i, "ok", r.kind, r.kind, None, None, "") for i, r in enumerate(capped)
    ]
    return (
        _verify_refine_loop(
            capped,
            pass1_verdicts,
            _Report(0.9),
            lane_outputs,
            object(),  # runtime sentinel — never used (verify/review monkeypatched)
            fbs,
            object(),  # council_state sentinel
            run_inner=run_inner,
            log=lambda _m: None,
        ),
        run_inner,
        fbs,
    )


# ---------------------------------------------------------------------------
# 1. _run_inner refactor — closure rebind + re-assemble-reflects-retype.
# ---------------------------------------------------------------------------


def _run_inner_body() -> str:
    src = inspect.getsource(cascade.run_full_cascade)
    start = src.index("def _run_inner(")
    end = src.index('fast_report = _run_inner("fast")')
    return src[start:end]


def test_run_inner_regions_none_byte_identical():
    """The refactor defaults ``regions`` to the enclosing ``capped`` and rebinds
    EVERY ``capped`` read to ``regions`` — so ``regions=None`` is byte-identical
    to the pre-refactor call (every legacy site passes no regions)."""
    body = _run_inner_body()
    assert "def _run_inner(lane: str, regions: list[Region] | None = None)" in body
    assert "regions = capped if regions is None else regions" in body
    # No bare positional ``capped,`` read survives (the 7 call sites use regions).
    assert "capped," not in body
    assert "len(capped)" not in body
    assert "list(capped)" not in body
    # The 7 documented reads all bind to ``regions``.
    assert "run_qwen_specialists(\n            regions," in body
    assert "_apply_h43_to_table_candidates(cands, regions)" in body
    assert "rerank_per_region(survs, regions, feature_blocks)" in body
    assert "_reading_order_at_risk_indices(top, list(regions))" in body
    assert "regions=regions," in body  # evaluate(regions=regions)


def test_reassemble_reflects_retyped_region():
    """A re-reviewed ``regions`` passed to the ``regions=``-parameterized
    re-assemble primitive yields an assembled doc reflecting the RE-TYPE (not
    the pass-1 regions) — guards the closure-rebind."""
    lane_outputs: dict = {}
    run_inner = _make_run_inner(lane_outputs)
    pass1 = [_region("heading", 0), _region("heading", 1), _region("paragraph", 2)]
    run_inner("fast", regions=pass1)
    assert "<h2>body 1</h2>" in lane_outputs["fast"]["assembled"].html

    retyped = [pass1[0], replace(pass1[1], kind="paragraph"), pass1[2]]
    run_inner("fast", regions=retyped)
    asm = lane_outputs["fast"]["assembled"]
    assert "<h2>body 1</h2>" not in asm.html  # the re-type DID flow through
    assert "<p>body 1</p>" in asm.html


# ---------------------------------------------------------------------------
# 2. Loop behaviour.
# ---------------------------------------------------------------------------


def test_clean_verify_no_bounce(monkeypatch):
    """Verifier passes round-1 -> zero re-reviews; capped/verdicts/report kept."""
    capped = [_region("heading", 0), _region("paragraph", 1)]
    lane_outputs = {"fast": _lane_dict(_FakeAssembled([r.kind for r in capped]))}
    scripted = _ScriptedReviewer(defects={})  # nothing flagged
    _patch(monkeypatch, scripted)
    (new_capped, new_verdicts, report, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted
    )
    assert scripted.review_calls == []
    assert [r.kind for r in new_capped] == ["heading", "paragraph"]
    assert signals["adopted"] is False
    assert len(signals["rounds"]) == 1 and signals["rounds"][0]["passed"] is True


def test_flagged_bounces_only_fixable_indices(monkeypatch):
    """A FAIL flagging fixable {1} + deferred {2}: re-review restrict_to={1}; the
    deferred idx is recorded, never re-driven."""
    capped = [_region("paragraph", 0), _region("heading", 1), _region("paragraph", 2)]
    lane_outputs = {"fast": _lane_dict(_FakeAssembled([r.kind for r in capped]))}
    scripted = _ScriptedReviewer(
        defects={
            1: ("heading", "example_as_heading", True, "demote me"),
            2: ("paragraph", "section_no_body", False, "merge me"),
        },
        fix_to={1: "paragraph"},
    )
    _patch(monkeypatch, scripted)
    (new_capped, _v, _r, signals), _ri, _fbs = _drive(capped, lane_outputs, scripted)
    # restrict_to on the first (only) review was exactly {1}.
    assert scripted.review_calls[0]["restrict_to"] == frozenset({1})
    assert 2 not in scripted.review_calls[0]["restrict_to"]
    # The deferred mode is recorded in the signals.
    assert "section_no_body" in signals["rounds"][0]["deferred_modes"]
    assert signals["rounds"][0]["fixable_indices"] == [1]


def test_restrict_to_uses_provenance_value(monkeypatch):
    """The flagged ``region_index`` VALUE drives restrict_to + the re-type lands
    on the correct capped region."""
    capped = [_region("paragraph", 0), _region("paragraph", 1), _region("heading", 2)]
    lane_outputs = {"fast": _lane_dict(_FakeAssembled([r.kind for r in capped]))}
    scripted = _ScriptedReviewer(
        defects={2: ("heading", "example_as_heading", True, "fix")},
        fix_to={2: "paragraph"},
    )
    _patch(monkeypatch, scripted)
    (new_capped, _v, _r, signals), _ri, _fbs = _drive(capped, lane_outputs, scripted)
    assert scripted.review_calls[0]["restrict_to"] == frozenset({2})
    assert new_capped[2].kind == "paragraph"  # the re-type targeted capped idx 2
    assert signals["adopted"] is True


def test_bounded_to_n_rounds(monkeypatch):
    """A verifier that never passes (re-driver is a no-op) -> exactly N rounds
    of re-review, then keep-best (= Pass-1)."""
    capped = [_region("heading", 0), _region("paragraph", 1)]
    lane_outputs = {"fast": _lane_dict(_FakeAssembled([r.kind for r in capped]))}
    scripted = _ScriptedReviewer(
        defects={0: ("heading", "example_as_heading", True, "fix")},
        fix_to={0: "paragraph"},
        no_fix=True,  # re-driver does nothing -> verifier never clears
    )
    _patch(monkeypatch, scripted, rounds=3)
    (new_capped, _v, _r, signals), _ri, _fbs = _drive(capped, lane_outputs, scripted)
    assert len(scripted.review_calls) == 3  # exactly the bound
    assert len(signals["rounds"]) == 3
    assert signals["adopted"] is False
    assert [r.kind for r in new_capped] == ["heading", "paragraph"]  # Pass-1 kept


def test_failsafe_keeps_pass1_on_verifier_error(monkeypatch):
    """``run_second_pass_verify`` raises -> Pass-1 capped/verdicts/lane restored
    byte-identical."""
    capped = [_region("heading", 0), _region("paragraph", 1)]
    pass1_assembled = _FakeAssembled([r.kind for r in capped])
    lane_outputs = {"fast": _lane_dict(pass1_assembled)}
    pass1_lane_snapshot = dict(lane_outputs["fast"])
    scripted = _ScriptedReviewer(defects={}, raise_on_verify=True)
    _patch(monkeypatch, scripted)
    pass1_verdicts = [ReviewVerdict(i, "ok", r.kind, r.kind, None, None, "") for i, r in enumerate(capped)]
    (new_capped, new_verdicts, _r, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted, pass1_verdicts=pass1_verdicts
    )
    assert [r.kind for r in new_capped] == ["heading", "paragraph"]
    assert new_verdicts == pass1_verdicts
    assert signals["adopted"] is False
    # lane_outputs['fast'] restored to the Pass-1 snapshot (same assembled obj).
    assert lane_outputs["fast"]["assembled"] is pass1_assembled
    assert lane_outputs["fast"] == pass1_lane_snapshot


def test_not_better_reverts_to_pass1(monkeypatch):
    """A re-review whose re-assemble drifts an UNFLAGGED region (clears the flag
    but changes a non-flagged region) -> not-better -> revert + break."""
    capped = [_region("heading", 0), _region("heading", 1), _region("paragraph", 2)]
    pass1_assembled = _FakeAssembled([r.kind for r in capped])
    lane_outputs = {"fast": _lane_dict(pass1_assembled)}
    scripted = _ScriptedReviewer(
        defects={1: ("heading", "example_as_heading", True, "fix")},
        fix_to={1: "paragraph"},
        drift={0: "paragraph"},  # re-types the UNFLAGGED region 0 too
    )
    _patch(monkeypatch, scripted)
    (new_capped, _v, _r, signals), _ri, _fbs = _drive(capped, lane_outputs, scripted)
    assert [r.kind for r in new_capped] == ["heading", "heading", "paragraph"]  # Pass-1
    assert signals["adopted"] is False
    assert len(scripted.review_calls) == 1  # broke after the first not-better round
    assert "unflagged_region_drift" in signals["rounds"][0]["reasons"]
    assert lane_outputs["fast"]["assembled"] is pass1_assembled  # reverted


def test_unflagged_regions_byte_identical_across_rounds(monkeypatch):
    """An ADOPTED round keeps every UNFLAGGED region's assembled bytes identical
    to the prior round."""
    from semantik_structure.assembler.verify_digest import slice_region_html

    capped = [_region("heading", 0), _region("heading", 1), _region("paragraph", 2)]
    pass1_assembled = _FakeAssembled([r.kind for r in capped])
    lane_outputs = {"fast": _lane_dict(pass1_assembled)}
    scripted = _ScriptedReviewer(
        defects={1: ("heading", "example_as_heading", True, "fix")},
        fix_to={1: "paragraph"},
    )
    _patch(monkeypatch, scripted)
    (new_capped, _v, _r, signals), _ri, _fbs = _drive(capped, lane_outputs, scripted)
    assert signals["adopted"] is True
    converged = lane_outputs["fast"]["assembled"]
    # Unflagged region 0 + 2 byte-identical pre/post; flagged region 1 changed.
    assert slice_region_html(converged, 0) == slice_region_html(pass1_assembled, 0)
    assert slice_region_html(converged, 2) == slice_region_html(pass1_assembled, 2)
    assert slice_region_html(converged, 1) != slice_region_html(pass1_assembled, 1)


def test_verdicts_threaded_back_on_adopt(monkeypatch):
    """An adopted round returns the converged (capped-space) verdicts reflecting
    the re-typed kind."""
    capped = [_region("paragraph", 0), _region("heading", 1)]
    lane_outputs = {"fast": _lane_dict(_FakeAssembled([r.kind for r in capped]))}
    scripted = _ScriptedReviewer(
        defects={1: ("heading", "example_as_heading", True, "fix")},
        fix_to={1: "paragraph"},
    )
    _patch(monkeypatch, scripted)
    (new_capped, new_verdicts, _r, signals), _ri, _fbs = _drive(capped, lane_outputs, scripted)
    assert signals["adopted"] is True
    v1 = next(v for v in new_verdicts if v.block_id == 1)
    assert v1.kind_before == "heading" and v1.kind_after == "paragraph"


def test_verdict_joins_correct_capped_region(monkeypatch):
    """DELTA drift-fix: an adopted re-type's CAPPED-space verdict joins to the
    correct capped region via ``_build_region_provenance(review_regions=None)``
    (NEVER pre_review_regions, which would corrupt the min-FB join)."""
    capped = [_region("paragraph", 0), _region("heading", 1), _region("paragraph", 2)]
    lane_outputs = {"fast": _lane_dict(_FakeAssembled([r.kind for r in capped]))}
    scripted = _ScriptedReviewer(
        defects={1: ("heading", "example_as_heading", True, "fix")},
        fix_to={1: "paragraph"},
    )
    _patch(monkeypatch, scripted)
    (new_capped, new_verdicts, _r, signals), _ri, fbs = _drive(capped, lane_outputs, scripted)
    assert signals["adopted"] is True
    # The cascade, on adopt, joins with review_regions=None (capped-consistent).
    order = list(range(len(new_capped)))
    prov = _build_region_provenance(
        order, list(new_capped), fbs, {}, review_verdicts=new_verdicts, review_regions=None
    )
    entry = next(e for e in prov if e["region_index"] == 1)
    assert "review" in entry
    assert entry["review"]["corrected_from"] == "heading"
    assert entry["review"]["corrected_to"] == "paragraph"
    # The unflagged capped region 0 carries NO review.
    assert "review" not in next(e for e in prov if e["region_index"] == 0)


def test_doc_text_conservation_across_rounds(monkeypatch):
    """Converged HTML visible text == Pass-1 visible text (no dropped/dup source)."""
    capped = [_region("heading", 0), _region("heading", 1), _region("paragraph", 2)]
    pass1_assembled = _FakeAssembled([r.kind for r in capped])
    lane_outputs = {"fast": _lane_dict(pass1_assembled)}
    scripted = _ScriptedReviewer(
        defects={1: ("heading", "example_as_heading", True, "fix")},
        fix_to={1: "paragraph"},
    )
    _patch(monkeypatch, scripted)
    (new_capped, _v, _r, signals), _ri, _fbs = _drive(capped, lane_outputs, scripted)
    assert signals["adopted"] is True
    converged = lane_outputs["fast"]["assembled"]
    pass1_tokens = sorted(cascade._visible_text_tokens(pass1_assembled.html))
    conv_tokens = sorted(cascade._visible_text_tokens(converged.html))
    assert conv_tokens == pass1_tokens


def test_offline_inherits_converged_capped(monkeypatch):
    """The loop returns the CONVERGED capped (the re-typed list) — the cascade
    rebinds ``capped`` to it, so the offline retry (which reads ``capped`` after
    the loop) INHERITS the converged regions."""
    capped = [_region("paragraph", 0), _region("heading", 1)]
    lane_outputs = {"fast": _lane_dict(_FakeAssembled([r.kind for r in capped]))}
    scripted = _ScriptedReviewer(
        defects={1: ("heading", "example_as_heading", True, "fix")},
        fix_to={1: "paragraph"},
    )
    _patch(monkeypatch, scripted)
    (converged_capped, _v, _r, signals), run_inner, _fbs = _drive(capped, lane_outputs, scripted)
    assert signals["adopted"] is True
    assert converged_capped is not capped  # NOT the Pass-1 list
    assert [r.kind for r in converged_capped] == ["paragraph", "paragraph"]
    # Demonstrate offline would re-assemble THIS converged list, not Pass-1.
    run_inner("offline", regions=converged_capped)
    assert "<h2>" not in lane_outputs["offline"]["assembled"].html


# ---------------------------------------------------------------------------
# 3. Flag-off / guard no-ops (the cascade host guard).
# ---------------------------------------------------------------------------


def test_flag_off_loop_no_op_byte_identical(monkeypatch):
    """Flag off -> the cascade host guard (resolve_second_pass_mode) is False, so
    _verify_refine_loop is never called and no second_pass_verify key is set."""
    monkeypatch.delenv("SEMANTIK_SECOND_PASS", raising=False)
    assert rv.resolve_second_pass_mode() is False
    # The cascade guard is `resolve_second_pass_mode() and review_verdicts is not
    # None`; with the mode off the loop body is unreachable.
    host_src = inspect.getsource(cascade.run_full_cascade)
    assert "if resolve_second_pass_mode() and review_verdicts is not None:" in host_src
    # The audit arm is conditional on signals being produced.
    assert 'result["second_pass_verify"] = second_pass_signals' in host_src
    assert "second_pass_signals is not None and second_pass_signals.get(\"rounds\")" in host_src


# ---------------------------------------------------------------------------
# 4. Phase 6b — MERGE channel (SEMANTIK_UNIT_REGROUP).
# ---------------------------------------------------------------------------


from semantik_structure.qwen_specialists import block_resegment as brs  # noqa: E402


class _FakeAssembledFB:
    """Merge-aware assembled stand-in: per-region visible text = the region's
    OWNED feature-block tokens (``fb{j}``). A count-SHRINKING merge therefore
    CONSERVES the document visible-text multiset (the merged region owns the
    union of its members' FBs), so ``_merge_round_is_better``'s doc-text check
    passes on a legitimate merge (and trips on a dropped/dup'd token)."""

    def __init__(self, regions: list[Region]):
        self.region_provenance = list(range(len(regions)))
        frags = [
            "<div>" + " ".join(f"fb{j}" for j in r.feature_block_indices) + "</div>"
            for r in regions
        ]
        self._frags = frags
        self.sub_task_log = {"region_html": list(frags)}
        self.html = "<html>" + "".join(frags) + "</html>"
        self.heading_tree = []
        self.landmarks = {}


def _make_run_inner_fb(lane_outputs: dict, *, gate_passed: bool = True, theta: float = 0.9):
    """Merge-aware injected ``run_inner`` — assembles the PASSED ``regions`` into
    an FB-token-derived fake doc (so a merge conserves document text)."""
    calls: list[dict] = []

    def run_inner(lane: str, regions=None):
        rs = list(regions or [])
        calls.append({"lane": lane, "n": None if regions is None else len(rs)})
        lane_outputs[lane] = _lane_dict(_FakeAssembledFB(rs), gate_passed=gate_passed)
        return _Report(theta_score=theta, lane=lane)

    run_inner.calls = calls  # type: ignore[attr-defined]
    return run_inner


class _ScriptedMergeReviewer:
    """Scripted Pass-2 verifier for the MERGE channel.

    Flags ONE ``section_no_body`` merge mode (``fixable=False`` +
    ``proposed_regroup_run``) while the region list is the ORIGINAL length;
    PASSES once the merge has shrunk it. ``always_fail`` keeps flagging even
    post-merge (the bounded-rounds path)."""

    def __init__(self, *, orig_len: int, run, always_fail: bool = False):
        self.orig_len = orig_len
        self.run = tuple(run)
        self.always_fail = always_fail
        self.verify_calls: list[dict] = []

    def verify(self, assembled, regions, feature_blocks, runtime, *, council_state=None):
        self.verify_calls.append({"n": len(regions)})
        if not self.always_fail and len(regions) < self.orig_len:
            return VerifierVerdict(passed=True, flagged=())
        fb = FlaggedBlock(
            region_index=self.run[0],
            failure_mode="section_no_body",
            fix_hint="merge the heading with its orphaned body",
            fixable=False,
            proposed_regroup_run=self.run,
        )
        return VerifierVerdict(passed=False, flagged=(fb,))


def _patch_merge(monkeypatch, scripted, *, rounds: int = 2, regroup_on: bool = True):
    monkeypatch.setattr(rv, "run_second_pass_verify", scripted.verify)
    monkeypatch.setattr(rv, "resolve_second_pass_rounds", lambda: rounds)
    monkeypatch.setattr(brs, "resolve_unit_regroup_mode", lambda: regroup_on)


def _merge_capped() -> list[Region]:
    # A heading + three following body paragraphs, each owning ONE FB in
    # document order (so run (0,1,2) is FB-contiguous and R-PART-valid).
    return [
        _region("heading", 0),
        _region("paragraph", 1),
        _region("paragraph", 2),
        _region("paragraph", 3),
    ]


def test_merge_channel_fixes_section_no_body(monkeypatch):
    """A FAIL flagging ``section_no_body`` with a contiguous
    ``proposed_regroup_run=(0,1,2)`` + SEMANTIK_UNIT_REGROUP on -> the merge
    channel fires, ``len(new_capped) < len(capped)``, re-verify passes, adopted;
    the threaded-back verdicts are None (count-shrink provenance safety)."""
    capped = _merge_capped()
    lane_outputs = {"fast": _lane_dict(_FakeAssembledFB(capped))}
    scripted = _ScriptedMergeReviewer(orig_len=4, run=(0, 1, 2))
    _patch_merge(monkeypatch, scripted)
    (new_capped, new_verdicts, _r, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted, run_inner=_make_run_inner_fb(lane_outputs)
    )
    assert len(new_capped) == 2 < len(capped)
    assert signals["adopted"] is True
    assert signals.get("merge_adopted") is True
    assert signals["rounds"][0]["merge_adopted"] is True
    assert signals["rounds"][0]["merge_ops"] >= 1
    # SAFEST count-shrink thread-back: verdicts threaded back as None.
    assert new_verdicts is None


def test_merge_channel_gated_off_when_regroup_off(monkeypatch):
    """Same flags but SEMANTIK_UNIT_REGROUP off -> merge channel DEAD; the round
    is a deferred-only no-op (byte-identical to committed P6): no re-assemble,
    Pass-1 kept, verdicts unchanged."""
    capped = _merge_capped()
    lane_outputs = {"fast": _lane_dict(_FakeAssembledFB(capped))}
    scripted = _ScriptedMergeReviewer(orig_len=4, run=(0, 1, 2))
    _patch_merge(monkeypatch, scripted, regroup_on=False)
    run_inner = _make_run_inner_fb(lane_outputs)
    pass1_verdicts = [
        ReviewVerdict(i, "ok", r.kind, r.kind, None, None, "") for i, r in enumerate(capped)
    ]
    (new_capped, new_verdicts, _r, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted, run_inner=run_inner, pass1_verdicts=pass1_verdicts
    )
    assert [r.kind for r in new_capped] == ["heading", "paragraph", "paragraph", "paragraph"]
    assert signals["adopted"] is False
    assert signals["rounds"][0]["deferred_only"] is True
    assert new_verdicts == pass1_verdicts  # NOT None — no merge adopted
    assert run_inner.calls == []  # no re-assemble fired


def test_merge_noop_keeps_prior(monkeypatch):
    """A non-contiguous proposed run (0,2) is dropped by the R-PART gate ->
    ``apply_proposed_regroups`` returns no ops -> merge no-op -> Pass-1 kept,
    never regress."""
    capped = _merge_capped()
    pass1_assembled = _FakeAssembledFB(capped)
    lane_outputs = {"fast": _lane_dict(pass1_assembled)}
    scripted = _ScriptedMergeReviewer(orig_len=4, run=(0, 2))  # skips region 1 -> R-PART drop
    _patch_merge(monkeypatch, scripted)
    pass1_verdicts = [
        ReviewVerdict(i, "ok", r.kind, r.kind, None, None, "") for i, r in enumerate(capped)
    ]
    (new_capped, new_verdicts, _r, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted, run_inner=_make_run_inner_fb(lane_outputs),
        pass1_verdicts=pass1_verdicts,
    )
    assert [r.kind for r in new_capped] == [r.kind for r in capped]  # Pass-1 kept
    assert signals["adopted"] is False
    assert signals["rounds"][0].get("merge_noop") is True
    assert new_verdicts == pass1_verdicts  # no adopt -> verdicts unchanged
    assert lane_outputs["fast"]["assembled"] is pass1_assembled  # reverted


def test_merge_adopt_provenance_not_corrupted(monkeypatch):
    """An adopted merge -> ``_build_region_provenance`` over the MERGED list
    emits one entry per merged region, NO IndexError, NO stale-verdict
    misattribution (verdicts threaded back as None -> no ``review`` sub-block)."""
    capped = _merge_capped()
    lane_outputs = {"fast": _lane_dict(_FakeAssembledFB(capped))}
    scripted = _ScriptedMergeReviewer(orig_len=4, run=(0, 1, 2))
    _patch_merge(monkeypatch, scripted)
    (merged_capped, new_verdicts, _r, signals), _ri, fbs = _drive(
        capped, lane_outputs, scripted, run_inner=_make_run_inner_fb(lane_outputs)
    )
    assert signals["adopted"] is True
    assert new_verdicts is None
    order = list(range(len(merged_capped)))
    prov = _build_region_provenance(
        order, list(merged_capped), fbs, {}, review_verdicts=new_verdicts, review_regions=None
    )
    assert len(prov) == len(merged_capped)  # one entry per merged region
    assert sorted(e["region_index"] for e in prov) == order
    assert all("review" not in e for e in prov)  # no stale verdict attached


def test_merge_doc_text_conserved(monkeypatch):
    """An adopted merge -> converged HTML visible text == Pass-1 visible text
    (no dropped / duplicated source token)."""
    capped = _merge_capped()
    pass1_assembled = _FakeAssembledFB(capped)
    lane_outputs = {"fast": _lane_dict(pass1_assembled)}
    scripted = _ScriptedMergeReviewer(orig_len=4, run=(0, 1, 2))
    _patch_merge(monkeypatch, scripted)
    (merged_capped, _v, _r, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted, run_inner=_make_run_inner_fb(lane_outputs)
    )
    assert signals["adopted"] is True
    converged = lane_outputs["fast"]["assembled"]
    pass1_tokens = sorted(cascade._visible_text_tokens(pass1_assembled.html))
    conv_tokens = sorted(cascade._visible_text_tokens(converged.html))
    assert conv_tokens == pass1_tokens


def test_merge_channel_bounded(monkeypatch):
    """A verifier that NEVER passes (always_fail) -> the merge lands each round
    but re-verify keeps failing -> bounded to N rounds, then keep-Pass-1."""
    capped = _merge_capped()
    pass1_assembled = _FakeAssembledFB(capped)
    lane_outputs = {"fast": _lane_dict(pass1_assembled)}
    scripted = _ScriptedMergeReviewer(orig_len=4, run=(0, 1, 2), always_fail=True)
    _patch_merge(monkeypatch, scripted, rounds=3)
    (new_capped, new_verdicts, _r, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted, run_inner=_make_run_inner_fb(lane_outputs)
    )
    assert len(signals["rounds"]) == 3  # exactly the bound
    assert all(row.get("retry") for row in signals["rounds"])
    assert signals["adopted"] is False
    assert [r.kind for r in new_capped] == [r.kind for r in capped]  # Pass-1 kept
    assert lane_outputs["fast"]["assembled"] is pass1_assembled  # reverted to Pass-1


def test_review_verdicts_none_guard(monkeypatch):
    """Pass-1 reviewer off (review_verdicts is None) -> the host guard short-
    circuits before referencing review_runtime, so the loop no-ops."""
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "on")
    assert rv.resolve_second_pass_mode() is True
    host_src = inspect.getsource(cascade.run_full_cascade)
    # The guard AND-composes the mode with `review_verdicts is not None` so a
    # no-reviewer run (review_verdicts stays None) never enters the loop.
    guard = "if resolve_second_pass_mode() and review_verdicts is not None:"
    assert guard in host_src
    # second_pass_adopted defaults False -> provenance join keeps pre_review_regions.
    assert "second_pass_adopted = False" in host_src
    assert "review_regions=(None if second_pass_adopted else pre_review_regions)" in host_src


class _ScriptedMoveMergeReviewer:
    """ITEM3 Phase-3 verifier: flags ``example_misordered_from_body`` with a
    NON-contiguous ``proposed_regroup_run`` until the move+merge decomposition
    shrinks the region list."""

    def __init__(self, *, orig_len: int, run):
        self.orig_len = orig_len
        self.run = tuple(run)
        self.verify_calls: list[dict] = []

    def verify(self, assembled, regions, feature_blocks, runtime, *, council_state=None):
        self.verify_calls.append({"n": len(regions)})
        if len(regions) < self.orig_len:
            return VerifierVerdict(passed=True, flagged=())
        fb = FlaggedBlock(
            region_index=self.run[0],
            failure_mode="example_misordered_from_body",
            fix_hint="move the misordered example body up to its label",
            fixable=False,
            proposed_regroup_run=self.run,
        )
        return VerifierVerdict(passed=False, flagged=(fb,))


def test_verify_refine_merge_channel_move(monkeypatch):
    """A deferred ``example_misordered_from_body`` flag whose
    ``proposed_regroup_run=(0,2)`` is NON-contiguous is now FIXED under
    ``SEMANTIK_MOVE_OP=live``: the Phase-6b channel (``apply_proposed_unit_fix``)
    decomposes it into a MOVE + a contiguous merge, the region list shrinks, the
    re-verify passes, and the round is adopted (previously this run was silently
    dropped by the R-PART order gate)."""
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "live")
    capped = _merge_capped()
    lane_outputs = {"fast": _lane_dict(_FakeAssembledFB(capped))}
    scripted = _ScriptedMoveMergeReviewer(orig_len=4, run=(0, 2))  # NON-contiguous
    _patch_merge(monkeypatch, scripted)
    (new_capped, _v, _r, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted, run_inner=_make_run_inner_fb(lane_outputs)
    )
    assert len(new_capped) < len(capped)  # the move+merge decomposition landed
    assert signals["adopted"] is True
    assert signals["rounds"][0]["merge_ops"] >= 1


def test_verify_refine_noncontiguous_dropped_when_not_live(monkeypatch):
    """The SAME non-contiguous run under the default (shadow) mode delegates to
    ``apply_proposed_regroups`` and is DROPPED -> Pass-1 kept (byte-identical to
    the pre-ITEM3 behaviour)."""
    monkeypatch.setenv("SEMANTIK_MOVE_OP", "shadow")
    capped = _merge_capped()
    pass1_assembled = _FakeAssembledFB(capped)
    lane_outputs = {"fast": _lane_dict(pass1_assembled)}
    scripted = _ScriptedMoveMergeReviewer(orig_len=4, run=(0, 2))
    pass1_verdicts = [
        ReviewVerdict(i, "ok", r.kind, r.kind, None, None, "") for i, r in enumerate(capped)
    ]
    _patch_merge(monkeypatch, scripted)
    (new_capped, _v, _r, signals), _ri, _fbs = _drive(
        capped, lane_outputs, scripted, run_inner=_make_run_inner_fb(lane_outputs),
        pass1_verdicts=pass1_verdicts,
    )
    assert [r.kind for r in new_capped] == [r.kind for r in capped]  # Pass-1 kept
    assert signals["adopted"] is False
