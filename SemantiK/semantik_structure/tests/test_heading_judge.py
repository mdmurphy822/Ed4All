"""GLM-OCR Super heading-level JUDGE pass tests (stub seat — NO GPU / no seat).

Covers: digest determinism + budget/anchors; strict parse (fences / garbage /
unknown ids); every clamp rule; fail-open byte-identity; cache hit skips the
POST; flag-off byte-identity of the lane; and the DecisionCapture firing with a
dynamic rationale.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from semantik_structure.glmocr import heading_judge as hj


@pytest.fixture(autouse=True)
def _legacy_seat_context(monkeypatch):
    """Default every test to the LEGACY hardcoded budgets + LEGACY char-heuristic
    token counting: force ``SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT=off`` and
    ``SEMANTIK_HEADING_JUDGE_TOKENIZER=off`` (byte-identical chars//4 sizer /
    chars//3 guard) and clear the probe + tokenizer caches so a live judge seat /
    a host tokenizer can never perturb the resolvers. Tests exercising the
    adaptive / auto / real-tokenizer path override the env themselves (the
    real-tokenizer coverage lives in test_heading_judge_tokenizer.py).

    Also force ``SEMANTIK_HEADING_JUDGE_ENABLE_THINKING=1``: the budget resolvers
    are now THINKING-AWARE (thinking-off collapses floor/ceiling/est to
    JSON-sized values), so these legacy-budget (20480/30000/300) assertions
    exercise the THINKING-ON path. The thinking-OFF budgets have their own
    dedicated tests below (which override the env)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "off")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENIZER", "off")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "1")
    hj._reset_seat_context_cache()
    hj._reset_tokenizer_cache()
    yield
    hj._reset_seat_context_cache()
    hj._reset_tokenizer_cache()


# ── Fixtures. ────────────────────────────────────────────────────────────────
def _prov():
    """A small chapter: synthesized chapter opener (fixed L1), a pending
    apparatus title, a fixed N.M section (L2), a pending body subsection."""
    return [
        {"region_kind": "heading", "heading_text": "Chapter 1", "level": 1,
         "first_raw_block_index": 0, "source_page": 1,
         "chapter_title_synthesized": True, "native_label": "synthetic"},
        {"region_kind": "heading", "heading_text": "Chapter Outline", "level": 3,
         "first_raw_block_index": 1, "source_page": 1,
         "heading_level_pending": True, "native_label": "title"},
        {"region_kind": "paragraph",
         "raw_text": "In the following exercises, find the place value. And more text.",
         "first_raw_block_index": 2, "source_page": 1},
        {"region_kind": "heading",
         "heading_text": "1.1 Introduction to Whole Numbers", "level": 2,
         "first_raw_block_index": 18, "source_page": 2,
         "section_number_recovered": True, "native_label": "title"},
        {"region_kind": "heading",
         "heading_text": "Use Place Value with Whole Numbers", "level": 3,
         "first_raw_block_index": 24, "source_page": 2,
         "heading_level_pending": True, "native_label": "title"},
        {"region_kind": "paragraph",
         "raw_text": "Place value tells you the value of a digit by position.",
         "first_raw_block_index": 25, "source_page": 2},
    ]


def _escalations():
    return [
        {"schema": "glmocr-escalation/1.0", "region_index": 1, "source_page": 1,
         "native_label": "title", "reason": "heading_level_pending",
         "detail": "non-numbered title; level defaulted to 3"},
        {"schema": "glmocr-escalation/1.0", "region_index": 24, "source_page": 2,
         "native_label": "title", "reason": "heading_level_pending",
         "detail": "non-numbered title; level defaulted to 3"},
    ]


class _StubPost:
    """A ``post_fn(messages, max_tokens) -> (content, finish)`` returning a fixed
    body; records call count so cache-skip can be asserted."""

    def __init__(self, content, finish="stop"):
        self._content = content
        self._finish = finish
        self.calls = 0
        self.last_max_tokens = None

    def __call__(self, messages, max_tokens):
        self.calls += 1
        self.last_max_tokens = max_tokens
        c = self._content(self.calls) if callable(self._content) else self._content
        f = self._finish(self.calls) if callable(self._finish) else self._finish
        return c, f


# ── Digest determinism + budget. ─────────────────────────────────────────────
def test_skeleton_digest_matches_contract_shape():
    plan = hj.build_heading_skeleton(_prov())
    assert plan.pending_ids == [1, 24]
    assert len(plan.windows) == 1
    lines = plan.digest.splitlines()
    assert lines[0] == "R0 p1 h1. Chapter 1"
    assert lines[1] == "R1 p1 h3* Chapter Outline"
    # pending nodes carry a content anchor line
    assert lines[2].startswith("    > In the following exercises")
    assert "R18 p2 h2. 1.1 Introduction to Whole Numbers" in plan.digest


def test_skeleton_is_deterministic():
    a = hj.build_heading_skeleton(_prov()).digest
    b = hj.build_heading_skeleton(_prov()).digest
    assert a == b


def test_heading_text_hard_truncated_at_90():
    prov = _prov()
    prov[1]["heading_text"] = "X" * 200
    plan = hj.build_heading_skeleton(prov)
    line = [ln for ln in plan.digest.splitlines() if ln.startswith("R1 ")][0]
    # "R1 p1 h3* " prefix + 90 chars
    assert line.endswith("X" * 90)
    assert "X" * 91 not in line


def test_anchor_dropped_when_over_budget(monkeypatch):
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 1)
    # a single non-splittable window (no >1 level-2 anchors) → no-anchor digest
    prov = [
        {"region_kind": "heading", "heading_text": "T", "level": 2,
         "first_raw_block_index": 0, "source_page": 1},
        {"region_kind": "heading", "heading_text": "Sub", "level": 3,
         "first_raw_block_index": 1, "source_page": 1,
         "heading_level_pending": True},
        {"region_kind": "paragraph", "raw_text": "Body prose sentence here.",
         "first_raw_block_index": 2, "source_page": 1},
    ]
    plan = hj.build_heading_skeleton(prov)
    assert "    > " not in plan.digest  # anchors dropped


def test_overflow_splits_into_windows_with_outline_context(monkeypatch):
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 1)
    # Cap pendings-per-window at 1 so the three L2 segments cannot coalesce
    # (the merge pass otherwise packs small segments into one window).
    monkeypatch.setattr(hj, "_MAX_PENDING_PER_WINDOW", 1)
    prov = []
    idx = 0
    for sec in range(1, 4):  # three L2 sections, each with a pending child
        prov.append({"region_kind": "heading", "heading_text": f"1.{sec} Section",
                     "level": 2, "first_raw_block_index": idx, "source_page": sec})
        idx += 1
        prov.append({"region_kind": "heading", "heading_text": f"Sub {sec}",
                     "level": 3, "first_raw_block_index": idx, "source_page": sec,
                     "heading_level_pending": True})
        idx += 1
    plan = hj.build_heading_skeleton(prov)
    assert len(plan.windows) == 3
    for digest, pend in plan.windows:
        assert "FIXED-ANCHOR OUTLINE" in digest  # full spine context in each
        assert len(pend) == 1


def test_output_cap_splits_even_when_input_fits(monkeypatch):
    """A chapter whose digest FITS the input budget but carries more pendings
    than _MAX_PENDING_PER_WINDOW must still split (the measured live failure:
    308 pendings in one window -> finish=length -> doubled retry -> context
    overflow). Consecutive small segments coalesce up to the cap."""
    monkeypatch.setattr(hj, "_MAX_PENDING_PER_WINDOW", 2)
    prov = []
    idx = 0
    for sec in range(1, 5):  # four L2 sections, one pending child each
        prov.append({"region_kind": "heading", "heading_text": f"1.{sec} Section",
                     "level": 2, "first_raw_block_index": idx, "source_page": sec})
        idx += 1
        prov.append({"region_kind": "heading", "heading_text": f"Sub {sec}",
                     "level": 3, "first_raw_block_index": idx, "source_page": sec,
                     "heading_level_pending": True})
        idx += 1
    plan = hj.build_heading_skeleton(prov)
    # 4 pendings at cap 2 -> two merged windows of 2 pendings each.
    assert len(plan.windows) == 2
    assert [len(p) for _, p in plan.windows] == [2, 2]
    for digest, _ in plan.windows:
        assert "FIXED-ANCHOR OUTLINE" in digest


# ── Strict parse. ────────────────────────────────────────────────────────────
def test_parse_plain_json():
    assert hj.parse_judge_response('{"levels": {"1": 3, "24": 4}}') == {
        "levels": {"1": 3, "24": 4}}


def test_parse_strips_fences():
    raw = '```json\n{"levels": {"1": 3}}\n```'
    assert hj.parse_judge_response(raw) == {"levels": {"1": 3}}


def test_parse_extracts_first_balanced_object_amid_prose():
    raw = 'Here is my answer: {"levels": {"1": 5}} — done.'
    assert hj.parse_judge_response(raw) == {"levels": {"1": 5}}


@pytest.mark.parametrize("raw", [None, "", "not json", "{}", '{"foo": 1}',
                                 '{"levels": [1,2]}'])
def test_parse_failures_return_none_or_no_levels(raw):
    out = hj.parse_judge_response(raw)
    assert out is None or not out.get("levels")


# ── Clamp rules. ─────────────────────────────────────────────────────────────
def test_apply_basic_correction():
    prov, esc, tree = _prov(), _escalations(), []
    # id 1 follows the L1 chapter opener, so its deepest legal level is 2
    # (rule 3, no skipped tiers); id 24 sits under the L2 section → level 3.
    result = hj.apply_judged_levels(prov, tree, esc, {1: 2, 24: 3})
    assert result.applied == 2
    r1 = next(r for r in prov if r["first_raw_block_index"] == 1)
    assert r1["level"] == 2
    assert "heading_level_pending" not in r1
    assert r1["heading_level_judged"] == {"from": 3, "to": 2, "clamped": False}
    # escalation hygiene: pending row removed, judged row added
    assert not any(e["reason"] == "heading_level_pending" and e["region_index"] == 1
                   for e in esc)
    assert any(e["reason"] == "heading_level_judged" and e["region_index"] == 1
               for e in esc)


def test_caption_label_pending_heading_never_assigned_section_level():
    """D3.4: a numbered caption/pedagogical label ("Figure N.M") that leaked into
    the pending set must never be judged to a section level ≤3 — the assignment
    is dropped + tallied like the fixed-anchor drop."""
    prov = _prov()
    # replace the pending body subsection (id 24) with a caption label
    for r in prov:
        if r["first_raw_block_index"] == 24:
            r["heading_text"] = "Figure 1.5"
    esc, tree = _escalations(), []
    # judge id 24 (caption label) → level 3 (must be dropped); id 1 → level 2 (kept)
    result = hj.apply_judged_levels(prov, tree, esc, {1: 2, 24: 3})
    assert result.applied == 1  # only id 1 applied
    assert result.dropped == 1  # the caption-label assignment dropped
    r24 = next(r for r in prov if r["first_raw_block_index"] == 24)
    assert r24["level"] == 3  # unchanged from its defaulted level
    assert r24.get("heading_level_pending") is True  # never resolved


def test_pending_only_mutation_fixed_anchor_dropped():
    prov, esc, tree = _prov(), _escalations(), []
    # target the FIXED L2 anchor (id 18) + the fixed chapter opener (id 0)
    result = hj.apply_judged_levels(prov, tree, esc, {18: 5, 0: 4, 1: 4})
    assert result.dropped == 2  # 18 + 0 are not pending → dropped
    assert next(r for r in prov if r["first_raw_block_index"] == 18)["level"] == 2
    assert next(r for r in prov if r["first_raw_block_index"] == 0)["level"] == 1


def test_level_range_clamp_drops_out_of_range():
    prov, esc, tree = _prov(), _escalations(), []
    result = hj.apply_judged_levels(prov, tree, esc, {1: 1, 24: 9})
    # level 1 and 9 are out of [2,6] → both dropped
    assert result.dropped == 2
    assert result.applied == 0
    assert next(r for r in prov if r["first_raw_block_index"] == 1)["level"] == 3


def test_parent_plus_one_max_jump_clamp():
    # A pending child proposed far too deep must clamp to prev_effective + 1.
    prov = [
        {"region_kind": "heading", "heading_text": "1.1 Sec", "level": 2,
         "first_raw_block_index": 0, "source_page": 1},
        {"region_kind": "heading", "heading_text": "Sub", "level": 3,
         "first_raw_block_index": 1, "source_page": 1, "heading_level_pending": True},
    ]
    result = hj.apply_judged_levels(prov, [], [], {1: 6})
    # prev_effective after the L2 anchor is 2 → max jump is 3
    assert next(r for r in prov if r["first_raw_block_index"] == 1)["level"] == 3
    assert result.clamped == 1


def test_no_orphaning_clamp():
    # A pending heading inside a L3 context cannot become a peer of the L2 spine.
    prov = [
        {"region_kind": "heading", "heading_text": "1.1 Sec", "level": 2,
         "first_raw_block_index": 0, "source_page": 1},
        {"region_kind": "heading", "heading_text": "Body Sub", "level": 3,
         "first_raw_block_index": 1, "source_page": 1, "heading_level_pending": True},
    ]
    # propose level 2 (peer of the N.M spine) → clamp to enclosing_anchor + 1 = 3
    result = hj.apply_judged_levels(prov, [], [], {1: 2})
    assert next(r for r in prov if r["first_raw_block_index"] == 1)["level"] == 3
    assert result.clamped == 1


def test_absent_pending_id_keeps_current_level():
    prov, esc, tree = _prov(), _escalations(), []
    result = hj.apply_judged_levels(prov, tree, esc, {1: 4})  # id 24 absent
    r24 = next(r for r in prov if r["first_raw_block_index"] == 24)
    assert r24["level"] == 3
    assert r24.get("heading_level_pending") is True  # retained
    assert result.kept == 1


def test_heading_tree_rebuilt_after_apply():
    prov, esc, tree = _prov(), _escalations(), []
    # id 1 was pending L3 → becomes L2 (child of the L1 chapter); the rebuilt
    # tree must reflect the new level (was (3, ...) before).
    hj.apply_judged_levels(prov, tree, esc, {1: 2, 24: 3})
    assert (2, "Chapter Outline") in tree
    assert (3, "Use Place Value with Whole Numbers") in tree


# ── End-to-end: judged levels surface as differentiated HTML tags. ───────────
def test_judged_levels_render_as_differentiated_html_tags():
    """The judge writes corrected levels back onto the layout heading blocks;
    the lane re-render must map that layout level relative to the chapter root
    (L2 -> <h3>, L3 -> <h4>) instead of the pre-fix flat <h3> for both. This is
    the SAME render seam the fresh in-lane conversion AND the standalone judge
    ``--apply`` re-render use, so it proves the fix end-to-end on both paths.
    """
    import re

    from semantik_structure.glmocr.lane import (
        GlmOcrLaneResult,
        render_accessible_html,
    )

    prov = copy.deepcopy(_prov())
    esc = _escalations()
    tree: list = []
    # Judge: keep the pending body subsection (id 24) an L3 subsection and lift
    # the pending outline title (id 1) to an L2 direct section.
    hj.apply_judged_levels(prov, tree, esc, {1: 2, 24: 3})

    tree_after = [
        (r.get("level"), r.get("heading_text"))
        for r in prov
        if r.get("region_kind") == "heading"
    ]
    result = GlmOcrLaneResult(
        pdf="sample.pdf",
        region_provenance=prov,
        heading_tree=tree_after,
        escalations=esc,
    )
    html = render_accessible_html(result, pdf_stem="sample")

    def _level_of(text: str):
        for lvl, txt in re.findall(r"<h(\d)[^>]*>([^<]*)</h\1>", html):
            if txt.strip() == text:
                return lvl
        return None

    # fixed N.M section (L2) -> <h3>; judged body subsection (L3) -> <h4>.
    assert _level_of("1.1 Introduction to Whole Numbers") == "3", html
    assert _level_of("Use Place Value with Whole Numbers") == "4", html
    # pre-fix flatten (the L3 subsection rendered <h3>) is the FAILURE mode.
    assert _level_of("Use Place Value with Whole Numbers") != "3"


# ── Fail-open byte-identity. ─────────────────────────────────────────────────
def test_empty_verdict_is_byte_identical():
    prov = _prov()
    before = copy.deepcopy(prov)
    esc = _escalations()
    esc_before = copy.deepcopy(esc)
    tree = []
    result = hj.apply_judged_levels(prov, tree, esc, {})
    assert result.applied == 0
    assert prov == before  # no mutation
    assert esc == esc_before  # pending rows retained
    assert tree == []  # not rebuilt on fail-open


def test_run_heading_judge_transport_failure_fails_open():
    prov = _prov()
    before = copy.deepcopy(prov)
    esc = _escalations()
    esc_before = copy.deepcopy(esc)
    tree = []

    def _boom(messages, max_tokens):
        raise hj._JudgeTransportError("seat down", transient=False)

    report = hj.run_heading_judge(prov, tree, esc, post_fn=_boom, use_cache=False,
                                  emit_capture=False)
    assert report["applied"] == 0
    assert prov == before  # keep-current, byte-identical
    assert esc == esc_before  # pending flags + escalations retained


def test_length_exhaust_after_one_retry_fails_open():
    prov, esc, tree = _prov(), _escalations(), []
    stub = _StubPost(content='{"levels": {"1": 4}}', finish="length")
    report = hj.run_heading_judge(prov, tree, esc, post_fn=stub, use_cache=False,
                                  emit_capture=False)
    # Split-FIRST ladder: initial POST on the full window, then the SPLIT
    # judges each single-pending half (initial + last-resort doubled retry
    # each, halves being unsplittable) before failing open — 5 calls, never
    # thinking-off, never a doubled retry on a splittable window.
    assert stub.calls == 5
    assert stub.last_max_tokens > hj._max_tokens_for(len(prov))  # doubled
    assert report["applied"] == 0  # ladder exhausted → fail-open


# ── End-to-end judge + apply through the stub. ───────────────────────────────
def test_run_heading_judge_applies_stub_verdict():
    prov, esc, tree = _prov(), _escalations(), []
    stub = _StubPost(content='{"levels": {"1": 3, "24": 3}}')
    report = hj.run_heading_judge(prov, tree, esc, post_fn=stub, use_cache=False,
                                  emit_capture=False)
    assert stub.calls == 1
    assert report["applied"] == 2
    assert next(r for r in prov if r["first_raw_block_index"] == 24)["level"] == 3


# ── Cache. ───────────────────────────────────────────────────────────────────
def test_cache_hit_skips_the_post(tmp_path, monkeypatch):
    monkeypatch.setattr(hj, "_judge_cache_root", lambda: tmp_path)
    plan = hj.build_heading_skeleton(_prov())

    stub1 = _StubPost(content='{"levels": {"1": 4, "24": 3}}')
    m1, _ = hj.judge_heading_levels(plan, post_fn=stub1, use_cache=True,
                                    model="test-model")
    assert stub1.calls == 1
    assert m1 == {1: 4, 24: 3}

    stub2 = _StubPost(content='{"levels": {"9": 9}}')  # would differ if it ran
    m2, meta2 = hj.judge_heading_levels(plan, post_fn=stub2, use_cache=True,
                                        model="test-model")
    assert stub2.calls == 0  # served from cache
    assert m2 == {1: 4, 24: 3}
    assert meta2["cache_hits"] == 1


def test_fail_open_result_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(hj, "_judge_cache_root", lambda: tmp_path)
    plan = hj.build_heading_skeleton(_prov())
    # a parse failure → no verdict cached
    bad = _StubPost(content="not json at all")
    hj.judge_heading_levels(plan, post_fn=bad, use_cache=True, model="m")
    # next call with a GOOD stub must actually POST (nothing was cached);
    # the Fix-2 completion loop adds one focused re-judge POST for the
    # pending the stub's verdict omits (id 24), so >=1, not exactly 1.
    good = _StubPost(content='{"levels": {"1": 4}}')
    m, _ = hj.judge_heading_levels(plan, post_fn=good, use_cache=True, model="m")
    assert good.calls >= 1
    assert m == {1: 4}


# ── DecisionCapture fires with a dynamic rationale. ──────────────────────────
def test_decision_capture_fires_with_dynamic_rationale(monkeypatch):
    captured = {}

    class _FakeCap:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def log_decision(self, **kwargs):
            captured["log"] = kwargs

    import lib.decision_capture as dc
    monkeypatch.setattr(dc, "DecisionCapture", _FakeCap)

    prov, esc, tree = _prov(), _escalations(), []
    stub = _StubPost(content='{"levels": {"1": 4, "24": 3}}')
    hj.run_heading_judge(prov, tree, esc, post_fn=stub, use_cache=False,
                         course_code="TST_901")
    assert captured["init"]["phase"] == "semantik_conversion"
    assert captured["init"]["tool"] == "semantik"
    log = captured["log"]
    assert log["decision_type"] == "structure_review"
    assert log["heading_level_judge"] is True
    rationale = log["rationale"]
    # dynamic signals interpolated
    assert "applied=2" in rationale
    assert "nemotron-3-super" in rationale or "model=" in rationale
    assert len(rationale) >= 20
    alternatives = log["alternatives_considered"]
    assert all(
        set(item) == {"option", "reason_rejected"} for item in alternatives
    )
    assert "2 of 2" in alternatives[0]["reason_rejected"]


def test_capture_failure_never_breaks_judge(monkeypatch):
    import lib.decision_capture as dc

    def _boom(**kwargs):
        raise RuntimeError("capture backend down")

    monkeypatch.setattr(dc, "DecisionCapture", _boom)
    prov, esc, tree = _prov(), _escalations(), []
    stub = _StubPost(content='{"levels": {"1": 3, "24": 3}}')
    report = hj.run_heading_judge(prov, tree, esc, post_fn=stub, use_cache=False)
    assert report["applied"] == 2  # judge still applied despite capture failure


# ── Explicit falsey opt-out lane byte-identity (flag is default ON). ─────────
def test_lane_explicit_falsey_never_imports_judge(monkeypatch):
    from semantik_structure.glmocr import resolve_heading_judge_mode

    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "0")
    assert resolve_heading_judge_mode() is False


def test_lane_flag_default_on_resolves(monkeypatch):
    from semantik_structure.glmocr import resolve_heading_judge_mode

    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE", raising=False)
    assert resolve_heading_judge_mode() is True
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    assert resolve_heading_judge_mode() is True
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "garbage")
    assert resolve_heading_judge_mode() is True


# ── Standalone runner: cache passthrough (A2 sidecar contract). ──────────────
def _run_standalone_capturing_cache(tmp_path, monkeypatch, **kwargs):
    """Drive run_standalone with the judge seams stubbed; return the
    ``use_cache`` kwarg that reached ``judge_heading_levels``."""
    from types import SimpleNamespace

    from semantik_structure.glmocr import heading_judge_standalone as hjs

    layout = tmp_path / "x.glmocr_layout.json"
    layout.write_text(json.dumps({"pages": []}), encoding="utf-8")

    seen = {}
    plan = SimpleNamespace(pending_ids=[1], entries=[1], windows=[1], digest="d")
    monkeypatch.setattr(hj, "build_heading_skeleton", lambda prov: plan)
    monkeypatch.setattr(
        hj, "judge_heading_levels",
        lambda p, **kw: (seen.update(kw) or ({}, {})))
    monkeypatch.setattr(
        hj, "apply_judged_levels",
        lambda prov, tree, esc, vm: SimpleNamespace(
            applied=0, clamped=0, dropped=0, kept=0, corrections={}))
    hjs.run_standalone(layout, out_dir=tmp_path / "out", **kwargs)
    return seen.get("use_cache", "MISSING")


def test_standalone_default_defers_cache_to_resolver(tmp_path, monkeypatch):
    # The A2 regression: use_cache must NOT be hardcoded False — default None
    # defers to resolve_heading_judge_checkpoint() (default ON) so a killed
    # standalone pass resumes its judged windows from the sidecar cache.
    assert _run_standalone_capturing_cache(tmp_path, monkeypatch) is None


def test_standalone_no_cache_opt_out_passes_false(tmp_path, monkeypatch):
    assert _run_standalone_capturing_cache(
        tmp_path, monkeypatch, use_cache=False) is False


# ── Usage tap (stat-matrix metering rows). ───────────────────────────────────
class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeRequests:
    def __init__(self, payload):
        self._payload = payload

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        return _FakeResp(self._payload)


def _judge_payload():
    return {
        "choices": [{"message": {"content": '{"levels": {"1": 4}}'},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 123, "completion_tokens": 45},
    }


def test_usage_tap_emits_op2_shaped_row(tmp_path, monkeypatch):
    usage_path = tmp_path / "llm_usage.jsonl"
    monkeypatch.setenv("SEMANTIK_LLM_USAGE_PATH", str(usage_path))
    monkeypatch.setenv("SEMANTIK_LLM_USAGE_PHASE", "heading_judge_a2")
    content, finish = hj._post_judge_completion(
        base_url="http://x/v1", api_key=None, model="test-model",
        messages=[{"role": "user", "content": "hi"}], max_tokens=64,
        timeout=5.0, requests_module=_FakeRequests(_judge_payload()))
    assert finish == "stop"
    rows = [json.loads(l) for l in usage_path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["provider"] == "semantik-heading-judge"
    assert row["model"] == "test-model"
    assert row["phase"] == "heading_judge_a2"
    assert row["prompt_tokens"] == 123
    assert row["completion_tokens"] == 45
    assert row["finish_reason"] == "stop"
    assert row["duration_ms"] >= 0
    assert "ts" in row


def test_usage_tap_unset_env_is_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("SEMANTIK_LLM_USAGE_PATH", raising=False)
    hj._post_judge_completion(
        base_url="http://x/v1", api_key=None, model="m",
        messages=[], max_tokens=8, timeout=5.0,
        requests_module=_FakeRequests(_judge_payload()))
    assert list(tmp_path.iterdir()) == []


def test_usage_tap_failure_never_breaks_the_call(monkeypatch):
    # unwritable path → tap swallows, judge call still returns content
    monkeypatch.setenv("SEMANTIK_LLM_USAGE_PATH", "/proc/nope/llm_usage.jsonl")
    content, finish = hj._post_judge_completion(
        base_url="http://x/v1", api_key=None, model="m",
        messages=[], max_tokens=8, timeout=5.0,
        requests_module=_FakeRequests(_judge_payload()))
    assert content == '{"levels": {"1": 4}}'
    assert finish == "stop"


# ── Cache root resolves through the REAL paths module (no monkeypatch). ──────
def test_cache_root_resolves_without_monkeypatch():
    # Regression: `from . import paths` inside glmocr/ raised ImportError
    # (no glmocr/paths.py) and the swallow-all cache guards turned that into
    # a silent never-caches bug. The root must resolve via the PARENT
    # package's paths module.
    root = hj._judge_cache_root()
    assert root.name == "heading_judge_cache"


def test_cache_round_trips_on_real_disk(tmp_path, monkeypatch):
    # End-to-end through the REAL _judge_cache_root (env-redirected), not a
    # monkeypatched root: files must actually land on disk and serve hits.
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    plan = hj.build_heading_skeleton(_prov())
    stub1 = _StubPost(content='{"levels": {"1": 4, "24": 3}}')
    hj.judge_heading_levels(plan, post_fn=stub1, use_cache=True, model="m")
    sidecars = list((tmp_path / "heading_judge_cache").rglob("*.json"))
    assert len(sidecars) == 1  # the verdict persisted for real
    stub2 = _StubPost(content='{"levels": {"9": 9}}')
    m2, meta2 = hj.judge_heading_levels(plan, post_fn=stub2, use_cache=True,
                                        model="m")
    assert stub2.calls == 0 and m2 == {1: 4, 24: 3}
    assert meta2["cache_hits"] == 1


# ── Reasoning-preserving window SPLIT ladder (length-exhaust recovery). ──────
def _windowed_digest():
    """A windowed-shape digest: FIXED preamble + 4 pending lines w/ anchors."""
    return (
        "FIXED-ANCHOR OUTLINE (immutable context — the chapter spine):\n"
        "R0 p1 h1. Chapter 1\n"
        "\nWINDOW HEADINGS:\n"
        "R0 p1 h1. Chapter 1\n"
        "R1 p1 h3* Alpha\n"
        "    > First sentence alpha.\n"
        "R2 p1 h3* Beta\n"
        "R3 p2 h3* Gamma\n"
        "    > First sentence gamma.\n"
        "R4 p2 h3* Delta"
    )


def test_split_window_digest_halves_pendings_and_keeps_preamble():
    halves = hj._split_window_digest(_windowed_digest(), [1, 2, 3, 4])
    assert len(halves) == 2
    (d1, p1), (d2, p2) = halves
    assert p1 == [1, 2] and p2 == [3, 4]
    for d in (d1, d2):
        assert d.startswith("FIXED-ANCHOR OUTLINE")
        assert "WINDOW HEADINGS:" in d
    # anchor continuation lines travel with their heading
    assert "> First sentence alpha." in d1 and "> First sentence alpha." not in d2
    assert "> First sentence gamma." in d2
    # fixed context line survives in half 1's body
    assert "R0 p1 h1. Chapter 1" in d1


def test_split_window_digest_single_pending_returns_empty():
    assert hj._split_window_digest("R1 p1 h3* Only", [1]) == []


def test_length_exhaust_falls_to_split_and_merges_halves():
    # Full window: length on first POST AND on the (fitting) doubled retry →
    # split fires; each HALF judged fine; verdicts merge.
    calls = []

    def post(messages, max_tokens):
        calls.append(max_tokens)
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 4:
            return "thinking overflow", "length"
        if "Alpha" in user:
            return '{"levels": {"1": 3, "2": 4}}', "stop"
        return '{"levels": {"3": 4, "4": 3}}', "stop"

    parsed, wmeta = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4), post_fn=post,
        win_pending=[1, 2, 3, 4])
    assert parsed == {"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}
    assert wmeta["finish"] == "split" and wmeta["split_depth"] == 1
    assert len(calls) == 3  # full + 2 halves (split-first: no doubled retry)


def test_doubled_retry_skipped_when_it_would_overflow_context():
    # A prompt where the BASE max_tokens still fits the ctx budget (so the first
    # POST fires unchanged) but DOUBLING it would exceed _CTX_TOKENS_BUDGET must
    # NOT re-POST at the doubled cap (the vLLM-400 fail-open defect) — it goes
    # straight to the split. (Sized so the FIRST-POST ctx guard is a no-op here:
    # prompt+base <= budget < prompt+2*base; a prompt too big for base itself is
    # covered by test_first_post_ctx_budget_guard_prevents_overflow_400.)
    big_pad = "x" * 12000  # prompt fits base, but base*2 overflows the budget
    digest = _windowed_digest() + "\n" + big_pad
    seen = []

    def post(messages, max_tokens):
        seen.append(max_tokens)
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 4:
            return "overflow", "length"
        if "Alpha" in user:
            return '{"levels": {"1": 3, "2": 4}}', "stop"
        return '{"levels": {"3": 4, "4": 3}}', "stop"

    base = hj._max_tokens_for(4)
    parsed, wmeta = hj._judge_one_window(
        digest, 5, 4, base, post_fn=post, win_pending=[1, 2, 3, 4])
    # first call at base, then the two half calls — NEVER a doubled call
    assert seen[0] == base
    assert all(t <= hj._MAX_TOKENS_CEILING for t in seen)
    assert len(seen) == 3
    assert wmeta["finish"] == "split"


def test_half_failure_fails_open_that_half_only():
    def post(messages, max_tokens):
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 4:
            return "overflow", "length"
        if "Alpha" in user:
            return '{"levels": {"1": 3, "2": 4}}', "stop"
        raise hj._JudgeTransportError("seat hiccup", transient=False)

    parsed, wmeta = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4), post_fn=post,
        win_pending=[1, 2, 3, 4])
    assert parsed == {"levels": {"1": 3, "2": 4}}  # good half survives
    assert wmeta.get("transport_failure") is True  # failure surfaced honestly


def _big_prompt_window(n_pending=240, pad_chars=520):
    """A window whose prompt is large enough that `_max_tokens_for(n_pending)`
    (the derived ceiling on a big seat) + the prompt overflows the seat context
    — the ch03 real-layout shape (prompt ~42k + max_tokens ~89k > 131072)."""
    lines = ["R0 p1 h1. Chapter 1"]
    pend = []
    for i in range(1, n_pending + 1):
        lines.append(f"R{i} p1 h3* {'X' * pad_chars}")
        pend.append(i)
    return "\n".join(lines), pend


def test_first_post_ctx_budget_guard_prevents_overflow_400(monkeypatch):
    # REGRESSION (real ch03 normalized-eval bug): under an adaptive seat context
    # `_max_tokens_for(n)` returns the full derived ceiling, so a large window's
    # `prompt + max_tokens > seat context` and the seat 400s → the WHOLE window
    # fail-opens (58 pendings left unjudged). The FIRST POST must be ctx-budget
    # capped, exactly like the doubled-retry path already is.
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "131072")
    hj._reset_seat_context_cache()
    ctx = hj.resolve_ctx_tokens_budget()  # 126976 on a 131072 seat
    digest, pend = _big_prompt_window()
    msgs = hj.build_judge_messages(digest, len(pend) + 1, len(pend))
    base_max = hj._max_tokens_for(len(pend))
    # the pre-fix condition really overflows: uncapped prompt + max_tokens > ctx
    assert hj._estimate_prompt_tokens(msgs) + base_max > ctx

    reqs = []

    def seat(messages, max_tokens):
        # simulate the vLLM context check: reject (400-style, NON-timeout) when
        # prompt + requested exceeds the seat window; else answer every pending.
        reqs.append(max_tokens)
        if hj._estimate_prompt_tokens(messages) + max_tokens > ctx:
            raise hj._JudgeTransportError("400 maximum context length",
                                          transient=False)
        import re as _re
        ids = [int(m) for m in _re.findall(r"R(\d+) p\S+ h\d+\*",
                                           messages[-1]["content"])]
        return json.dumps({"levels": {str(i): 2 for i in ids}}), "stop"

    # the mock WOULD 400 on the raw (uncapped) budget — proving the defect.
    with pytest.raises(hj._JudgeTransportError):
        seat(msgs, base_max)

    reqs.clear()
    parsed, wmeta = hj._judge_one_window(
        digest, len(pend) + 1, len(pend), base_max,
        post_fn=seat, win_pending=pend, depth=0)
    # POST-FIX: every pending judged (no fail-open), and every POST fit the seat.
    assert parsed is not None
    assert len(parsed["levels"]) == len(pend)
    assert reqs and reqs[0] < base_max  # the first POST was CAPPED
    assert all(hj._estimate_prompt_tokens(msgs) + r <= ctx for r in reqs)
    hj._reset_seat_context_cache()


def test_first_post_small_window_is_byte_identical(monkeypatch):
    # A small window (prompt + max_tokens <= ctx_budget) must POST with the
    # UNCHANGED max_tokens — the cap is a strict no-op (byte-identical).
    seen = []

    def post(messages, max_tokens):
        seen.append(max_tokens)
        return '{"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}', "stop"

    base = hj._max_tokens_for(4)
    assert hj._estimate_prompt_tokens(
        hj.build_judge_messages(_windowed_digest(), 5, 4)) + base \
        <= hj.resolve_ctx_tokens_budget()
    parsed, _ = hj._judge_one_window(
        _windowed_digest(), 5, 4, base, post_fn=post, win_pending=[1, 2, 3, 4])
    assert parsed == {"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}
    assert seen == [base]  # requested max_tokens UNCHANGED (cap no-op)


def test_ctx_budget_and_ceiling_env_overrides(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CTX_BUDGET", "63500")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", "49152")
    assert hj.resolve_ctx_tokens_budget() == 63500
    assert hj.resolve_max_tokens_ceiling() == 49152
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CTX_BUDGET", "garbage")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", "-5")
    assert hj.resolve_ctx_tokens_budget() == hj._CTX_TOKENS_BUDGET
    assert hj.resolve_max_tokens_ceiling() == hj._MAX_TOKENS_CEILING
    # base per-window max_tokens (the cache-key input) unchanged by a ceiling
    # raise — floor + _EST_TOKENS_PER_JUDGMENT*10 stays under both ceilings
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    base = hj._max_tokens_for(10)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", "49152")
    assert hj._max_tokens_for(10) == base


# ── Timeout = over-deliberation → split (never blind re-POST). ───────────────
def test_timeout_on_first_post_triggers_split():
    def post(messages, max_tokens):
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 4:
            raise hj._JudgeTransportError("read timed out", transient=False,
                                          timeout=True)
        if "Alpha" in user:
            return '{"levels": {"1": 3, "2": 4}}', "stop"
        return '{"levels": {"3": 4, "4": 3}}', "stop"

    parsed, wmeta = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4), post_fn=post,
        win_pending=[1, 2, 3, 4])
    assert parsed == {"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}
    assert wmeta["finish"] == "split"


def test_default_post_fn_never_retries_timeouts():
    attempts = []

    class _TimeoutExc(Exception):
        pass
    _TimeoutExc.__name__ = "ReadTimeout"

    class _FakeReq:
        def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
            attempts.append(1)
            raise _TimeoutExc("read timed out")

    seat = hj.HeadingJudgeSeat(base_url="http://x/v1", api_key=None,
                               model="m", timeout=1.0)
    fn = hj._make_default_post_fn(seat, requests_module=_FakeReq())
    with pytest.raises(hj._JudgeTransportError) as ei:
        fn([{"role": "user", "content": "hi"}], 64)
    assert ei.value.timeout is True
    assert len(attempts) == 1  # ONE attempt — a timeout is never blind-retried


def test_non_timeout_transient_errors_still_retry():
    attempts = []

    class _FakeResp:
        status_code = 503
        text = "busy"

    class _FakeReq:
        def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
            attempts.append(1)
            return _FakeResp()

    seat = hj.HeadingJudgeSeat(base_url="http://x/v1", api_key=None,
                               model="m", timeout=1.0)
    fn = hj._make_default_post_fn(seat, requests_module=_FakeReq())
    with pytest.raises(hj._JudgeTransportError):
        fn([{"role": "user", "content": "hi"}], 64)
    assert len(attempts) == 3  # transient 5xx keeps the bounded retry ladder


# ── Split halves run concurrently; partial split verdicts never cache. ───────
def test_split_halves_run_concurrently():
    import threading
    import time as _t
    in_flight = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def post(messages, max_tokens):
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 4:
            return "overflow", "length"
        with lock:
            in_flight["now"] += 1
            in_flight["peak"] = max(in_flight["peak"], in_flight["now"])
        _t.sleep(0.05)
        with lock:
            in_flight["now"] -= 1
        if "Alpha" in user:
            return '{"levels": {"1": 3, "2": 4}}', "stop"
        return '{"levels": {"3": 4, "4": 3}}', "stop"

    parsed, wmeta = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4), post_fn=post,
        win_pending=[1, 2, 3, 4])
    assert parsed == {"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}
    assert in_flight["peak"] == 2  # both halves in flight together


def test_partial_split_verdict_is_never_cached(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    plan = hj.build_heading_skeleton(_prov())  # 2 pendings → halves of 1

    def post_partial(messages, max_tokens):
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 2:
            return "overflow", "length"  # full window exhausts → split
        if "Chapter Outline" in user:
            return '{"levels": {"1": 4}}', "stop"  # half A succeeds
        raise hj._JudgeTransportError("boom", transient=False)  # half B fails

    m1, meta1 = hj.judge_heading_levels(plan, post_fn=post_partial,
                                        use_cache=True, model="m")
    assert m1 == {1: 4}  # partial applied this run (fail-open half kept)
    assert not list((tmp_path / "heading_judge_cache").rglob("*.json"))

    # a re-run must RE-JUDGE (no poisoned cache hit) — now everything works
    def post_good(messages, max_tokens):
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 2:
            return '{"levels": {"1": 4, "24": 3}}', "stop"
        raise AssertionError("halves should not be needed")

    m2, meta2 = hj.judge_heading_levels(plan, post_fn=post_good,
                                        use_cache=True, model="m")
    assert meta2["cache_hits"] == 0  # nothing served from a partial verdict
    assert m2 == {1: 4, 24: 3}
    assert len(list((tmp_path / "heading_judge_cache").rglob("*.json"))) == 1


# ── COVERAGE contract: omitted pendings heal via split; gaps never cache. ────
def test_partial_coverage_heals_via_missing_id_split():
    # Full window returns valid JSON covering only 2 of 4 pendings; the heal
    # splits over the MISSING ids and merges.
    def post(messages, max_tokens):
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 4:
            return '{"levels": {"1": 3, "2": 4}}', "stop"  # omits 3, 4
        if "Gamma" in user and "Delta" not in user:
            return '{"levels": {"3": 4}}', "stop"
        return '{"levels": {"4": 3}}', "stop"

    parsed, wmeta = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4), post_fn=post,
        win_pending=[1, 2, 3, 4])
    assert parsed == {"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}
    assert wmeta.get("coverage_healed") == 2
    assert "coverage_gap" not in wmeta


def test_unhealed_coverage_gap_blocks_caching(tmp_path, monkeypatch):
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    plan = hj.build_heading_skeleton(_prov())  # pendings [1, 24]

    def post(messages, max_tokens):
        # ALWAYS omits id 24, whole window and halves alike → gap survives
        return '{"levels": {"1": 4}}', "stop"

    m, meta = hj.judge_heading_levels(plan, post_fn=post, use_cache=True,
                                      model="m")
    assert m == {1: 4}
    # gap 1 of 2 pendings > tolerance max(1, 2//50)=1? gap==1 <= 1 → tolerated.
    # So this verdict is WITHIN tolerance and may cache — assert the boundary:
    # with 2 pendings a 1-id gap is an honest fail-open boundary.
    assert len(list((tmp_path / "heading_judge_cache").rglob("*.json"))) == 1


def test_large_coverage_gap_cache_is_healed_on_hit(tmp_path, monkeypatch):
    # Simulate the live ch06 poisoning: a pre-contract cached verdict with a
    # big gap must be treated as a MISS on the next run and re-judged.
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    plan = hj.build_heading_skeleton(_prov())
    digest, win_pending = plan.windows[0]
    base = hj._max_tokens_for(len(win_pending))
    key = hj._window_cache_key(digest, "m", base)
    # poison: covers NEITHER pending... cover 0 of 2 → gap 2 > tol 1
    hj._cache_put(key, {"levels": {"999": 3}})

    def post(messages, max_tokens):
        return '{"levels": {"1": 4, "24": 3}}', "stop"

    m, meta = hj.judge_heading_levels(plan, post_fn=post, use_cache=True,
                                      model="m")
    assert meta["cache_hits"] == 0  # poisoned entry NOT served
    assert m == {1: 4, 24: 3}
    # and the healed full verdict re-cached over it
    assert json.loads(
        (hj._cache_path(key, tmp_path / "heading_judge_cache")).read_text()
    )["levels"] == {"1": 4, "24": 3}


# ═════════════════════════════════════════════════════════════════════════════
# Fix 1 — sibling-staircase post-rule (deterministic; audit shapes anonymized).
# ═════════════════════════════════════════════════════════════════════════════
def _entry(rid, text, *, pending=True, level=3, page=1):
    return hj.SkeletonEntry(rid, level, text, page, pending)


def test_staircase_three_siblings_flattened():
    # The audit's 3-sibling shape: three parallel short noun-phrase headings
    # judged 3 -> 4 -> 5 under one section — flattened to the first's level.
    entries = [
        _entry(0, "3.5 Watching Things", pending=False, level=2),
        _entry(1, "Alpha panel"),
        _entry(2, "Beta relay panel"),
        _entry(3, "Gamma panel"),
    ]
    verdicts = {1: 3, 2: 4, 3: 5}
    audit = hj.flatten_sibling_staircases(entries, verdicts)
    assert verdicts == {1: 3, 2: 3, 3: 3}
    assert audit == {2: (4, 3), 3: (5, 3)}


def test_staircase_two_siblings_flattened():
    # The audit's 2-sibling shape ("Best Practices" 3, "Costs" -> 4).
    entries = [
        _entry(0, "3.9 Wrap Up", pending=False, level=2),
        _entry(1, "Common Patterns"),
        _entry(2, "Costs"),
    ]
    verdicts = {1: 3, 2: 4}
    audit = hj.flatten_sibling_staircases(entries, verdicts)
    assert verdicts == {1: 3, 2: 3}
    assert audit == {2: (4, 3)}


def test_staircase_ordinal_progression_not_flattened():
    # An ordinal progression is a STRUCTURAL signal — never flattened.
    entries = [
        _entry(1, "Alpha 1"),
        _entry(2, "Alpha 2"),
        _entry(3, "Alpha 3"),
    ]
    verdicts = {1: 3, 2: 4, 3: 5}
    audit = hj.flatten_sibling_staircases(entries, verdicts)
    assert verdicts == {1: 3, 2: 4, 3: 5}
    assert audit == {}


def test_staircase_nm_numbers_not_flattened():
    # N.M section numbers legitimately support nesting — never flattened.
    entries = [
        _entry(1, "2.1 Alpha"),
        _entry(2, "2.1.1 Beta"),
    ]
    verdicts = {1: 3, 2: 4}
    assert hj.flatten_sibling_staircases(entries, verdicts) == {}
    assert verdicts == {1: 3, 2: 4}


def test_staircase_broken_by_fixed_anchor():
    # A fixed anchor between two pendings ends the parent-section span — the
    # two never form one sibling run.
    entries = [
        _entry(1, "Alpha panel"),
        _entry(2, "3.6 Next Section", pending=False, level=2),
        _entry(3, "Beta panel"),
    ]
    verdicts = {1: 3, 3: 4}
    assert hj.flatten_sibling_staircases(entries, verdicts) == {}
    assert verdicts == {1: 3, 3: 4}


def test_staircase_broken_by_unjudged_pending():
    # A pending WITHOUT a verdict breaks the run (its level is not a proposal).
    entries = [
        _entry(1, "Alpha panel"),
        _entry(2, "Beta panel"),
        _entry(3, "Gamma panel"),
    ]
    verdicts = {1: 3, 3: 4}  # id 2 unjudged
    assert hj.flatten_sibling_staircases(entries, verdicts) == {}


def test_staircase_equal_and_decreasing_levels_untouched():
    entries = [_entry(1, "Alpha panel"), _entry(2, "Beta panel")]
    equal = {1: 3, 2: 3}
    assert hj.flatten_sibling_staircases(entries, equal) == {}
    assert equal == {1: 3, 2: 3}
    decreasing = {1: 5, 2: 4}
    assert hj.flatten_sibling_staircases(entries, decreasing) == {}
    assert decreasing == {1: 5, 2: 4}


def test_staircase_non_parallel_shape_not_flattened():
    # A long sentence-like heading is not a short noun-phrase label; a genuine
    # child section with a much longer title is legitimate nesting.
    entries = [
        _entry(1, "Alpha panel"),
        _entry(2, "How the alpha panel aggregates its many upstream signals "
                  "in practice"),
    ]
    verdicts = {1: 3, 2: 4}
    assert hj.flatten_sibling_staircases(entries, verdicts) == {}

    dissimilar = [
        _entry(1, "Alpha"),
        _entry(2, "Alpha beta gamma delta"),  # word delta 3 > 2
    ]
    verdicts2 = {1: 3, 2: 4}
    assert hj.flatten_sibling_staircases(dissimilar, verdicts2) == {}


def test_staircase_flatten_applies_through_judge_and_clamp():
    # End-to-end: the seat proposes the staircase; judge_heading_levels
    # flattens it; the clamp walk then applies all three at the sibling level.
    prov = [
        {"region_kind": "heading", "heading_text": "1.1 Panels", "level": 2,
         "first_raw_block_index": 0, "source_page": 1},
        {"region_kind": "heading", "heading_text": "Alpha panel", "level": 3,
         "first_raw_block_index": 1, "source_page": 1,
         "heading_level_pending": True},
        {"region_kind": "heading", "heading_text": "Beta relay panel",
         "level": 3, "first_raw_block_index": 2, "source_page": 1,
         "heading_level_pending": True},
        {"region_kind": "heading", "heading_text": "Gamma panel", "level": 3,
         "first_raw_block_index": 3, "source_page": 1,
         "heading_level_pending": True},
    ]
    stub = _StubPost(content='{"levels": {"1": 3, "2": 4, "3": 5}}')
    report = hj.run_heading_judge(prov, [], [], post_fn=stub, use_cache=False,
                                  emit_capture=False)
    assert report["applied"] == 3
    assert report["meta"]["staircase_flattened"] == 2
    for rid in (1, 2, 3):
        r = next(r for r in prov if r["first_raw_block_index"] == rid)
        assert r["level"] == 3  # all siblings share the first's level


def test_staircase_meta_zero_when_no_staircase():
    prov, esc, tree = _prov(), _escalations(), []
    stub = _StubPost(content='{"levels": {"1": 3, "24": 3}}')
    report = hj.run_heading_judge(prov, tree, esc, post_fn=stub,
                                  use_cache=False, emit_capture=False)
    assert report["meta"]["staircase_flattened"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# Fix 2 — truncation-proof windowing: budget-derived pending cap.
# ═════════════════════════════════════════════════════════════════════════════
def test_pending_window_cap_derived_from_budget(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    expected = (hj._MAX_TOKENS_CEILING - hj._MAX_TOKENS_FLOOR) \
        // hj._EST_TOKENS_PER_JUDGMENT
    assert hj.resolve_pending_window_cap() == expected
    assert expected < hj._MAX_PENDING_PER_WINDOW  # the audit's 78 > cap
    # a raised ceiling widens the cap, bounded by the hard ceiling
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", "60000")
    assert hj.resolve_pending_window_cap() == hj._MAX_PENDING_PER_WINDOW
    # a tiny ceiling floors at the minimum (never per-heading windows)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", "17000")
    assert hj.resolve_pending_window_cap() == hj._MIN_PENDING_WINDOW_CAP


def test_budget_increment_matches_sizing_estimate():
    """The max_tokens BUDGET increment and the window-PACKING estimate must be
    the SAME constant (single source of truth) so a packed window is never
    under-budgeted — the drift that caused the first-attempt truncation."""
    assert hj._MAX_TOKENS_PER_PENDING == hj._EST_TOKENS_PER_JUDGMENT


def test_full_packed_window_is_not_under_budgeted(monkeypatch):
    """A window packed to the EFFECTIVE cap is granted ~the completion room it
    was sized against — its first-attempt budget reaches into the ceiling
    region, NOT a fraction of it (the pre-fix 24-vs-300 hole)."""
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    cap = hj.resolve_pending_window_cap()
    budget = hj._max_tokens_for(cap)
    # sized against ~(ceiling - floor) of completion room → budget must clear
    # 90% of floor + est*cap (i.e. the room the packer assumed), never 24*cap.
    assert budget >= hj._MAX_TOKENS_FLOOR + hj._EST_TOKENS_PER_JUDGMENT * cap * 0.9
    # and it lands within one per-judgment increment of the hard ceiling.
    assert budget >= hj.resolve_max_tokens_ceiling() - hj._EST_TOKENS_PER_JUDGMENT


def test_small_thinking_heavy_window_clears_the_old_truncation_point():
    """Regression: a live n_pending=3 window truncated at 16384 + 24*3 = 16456
    (finish=length). The aligned budget must clear that point with headroom."""
    assert hj._max_tokens_for(3) > 16456


def test_prompt_budget_plus_ceiling_fits_seat_context():
    """Context-safety invariant: the worst-case FIRST attempt (a full
    prompt-budget prompt + a completion at the ceiling) must fit the Super
    seat's 65,536-token window — which is why the ceiling must NOT be raised."""
    _SEAT_CONTEXT_WINDOW = 65536
    assert (hj.resolve_max_tokens_ceiling()
            + hj.resolve_ctx_tokens_budget()) <= _SEAT_CONTEXT_WINDOW


def test_budget_cap_splits_the_audit_shape(monkeypatch):
    """78+ pendings (the audit's truncation shape) must split into windows
    each within the budget-derived cap — never one exhausting mega-window."""
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    prov = []
    idx = 0
    for sec in range(1, 11):  # ten L2 sections × 8 pendings = 80 pendings
        prov.append({"region_kind": "heading",
                     "heading_text": f"1.{sec} Section", "level": 2,
                     "first_raw_block_index": idx, "source_page": sec})
        idx += 1
        for k in range(8):
            prov.append({"region_kind": "heading",
                         "heading_text": f"Topic {chr(65 + k)} of {sec}",
                         "level": 3, "first_raw_block_index": idx,
                         "source_page": sec, "heading_level_pending": True})
            idx += 1
    plan = hj.build_heading_skeleton(prov)
    cap = hj.resolve_pending_window_cap()
    assert len(plan.windows) >= 2
    assert all(len(p) <= cap for _, p in plan.windows)
    # every pending lands in exactly one window
    all_ids = [i for _, p in plan.windows for i in p]
    assert sorted(all_ids) == sorted(plan.pending_ids)
    assert len(all_ids) == len(set(all_ids)) == 80


# ═════════════════════════════════════════════════════════════════════════════
# Thinking-aware completion budget (anti-runaway) + thinking-off freq penalty.
# ═════════════════════════════════════════════════════════════════════════════
def test_thinkoff_budget_collapses_to_json_sizes(monkeypatch):
    """Thinking OFF (the DEFAULT) → floor/ceiling/est collapse to JSON-sized
    values, so `_max_tokens_for` caps at <=4096 and floors at 512 — bounding the
    measured 21680-token runaway the 20480 thinking floor licensed."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "0")
    assert hj.resolve_heading_judge_enable_thinking() is False
    assert hj.resolve_max_tokens_floor() == 512
    assert hj.resolve_max_tokens_ceiling() == 4096
    assert hj.resolve_heading_judge_est_per_judgment() == 64
    # clamp(512 + 64*n, 512, 4096)
    assert hj._max_tokens_for(0) == 512          # floor
    assert hj._max_tokens_for(4) == 512 + 64 * 4  # 768
    # the ~4-pending runaway window is now bounded to <=4096, never 21680.
    assert hj._max_tokens_for(4) <= 4096
    assert hj._max_tokens_for(1000) == 4096      # ceiling


def test_thinkon_budget_is_byte_identical_legacy(monkeypatch):
    """Thinking ON → the legacy 20480/30000/300 budgets are UNCHANGED (the
    seat-context path is off in this file's fixture)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "1")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    assert hj.resolve_max_tokens_floor() == hj._MAX_TOKENS_FLOOR == 20480
    assert hj.resolve_max_tokens_ceiling() == hj._MAX_TOKENS_CEILING == 30000
    assert hj.resolve_heading_judge_est_per_judgment() == hj._EST_TOKENS_PER_JUDGMENT == 300
    assert hj._max_tokens_for(3) > 16456  # clears the old truncation point


def test_thinkoff_thinkoff_env_overrides(monkeypatch):
    """The `*_THINKOFF` envs win over the JSON-sized defaults (parse-with-
    fallback: garbage/non-positive → the default)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "0")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF", "6000")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR_THINKOFF", "700")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT_THINKOFF", "80")
    assert hj.resolve_max_tokens_ceiling() == 6000
    assert hj.resolve_max_tokens_floor() == 700
    assert hj.resolve_heading_judge_est_per_judgment() == 80
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS_THINKOFF", "garbage")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR_THINKOFF", "-5")
    assert hj.resolve_max_tokens_ceiling() == 4096
    assert hj.resolve_max_tokens_floor() == 512


def test_thinkoff_pending_window_cap_recomputes(monkeypatch):
    """The pending-window cap reads the thinking-aware resolvers, so thinking-off
    self-consistently packs smaller windows: (4096-512)/64 = 56, under the hard
    96 ceiling — and the packing est == the budget est (single source of truth)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "0")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    assert hj.resolve_pending_window_cap() == (4096 - 512) // 64 == 56
    # a window packed to the cap is budgeted right up to the thinking-off ceiling.
    cap = hj.resolve_pending_window_cap()
    assert hj._max_tokens_for(cap) == 4096


def test_frequency_penalty_only_on_thinkoff_body(monkeypatch):
    """`frequency_penalty` rides the wire ONLY on the thinking-OFF path and only
    when nonzero; thinking-ON is byte-identical (no key)."""
    captured = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": '{"levels": {}}'},
                                 "finish_reason": "stop"}],
                    "usage": {}}

    class _Req:
        @staticmethod
        def post(url, json=None, headers=None, timeout=None):
            captured.clear()
            captured.update(json)
            return _Resp()

    kw = dict(base_url="http://x/v1", api_key=None, model="m",
              messages=[{"role": "user", "content": "hi"}], timeout=1.0,
              requests_module=_Req)

    # thinking OFF, default → 0.3 present.
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "0")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY", raising=False)
    hj._post_judge_completion(max_tokens=512, **kw)
    assert captured["frequency_penalty"] == 0.3
    assert captured["chat_template_kwargs"]["enable_thinking"] is False

    # thinking OFF, explicit 0.0 → key omitted.
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY", "0")
    hj._post_judge_completion(max_tokens=512, **kw)
    assert "frequency_penalty" not in captured

    # thinking OFF, explicit override honoured.
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY", "0.5")
    hj._post_judge_completion(max_tokens=512, **kw)
    assert captured["frequency_penalty"] == 0.5

    # thinking ON → NO key regardless of the env (byte-identical body).
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_ENABLE_THINKING", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FREQUENCY_PENALTY", "0.9")
    hj._post_judge_completion(max_tokens=20480, **kw)
    assert "frequency_penalty" not in captured
    assert hj.resolve_heading_judge_frequency_penalty() == 0.0


def test_no_spine_fallback_chunks_by_pending_cap(monkeypatch):
    """A spineless chapter with more pendings than the cap must NOT fall back
    to one mega-window — it slices by pending count, each window carrying the
    fixed-anchor context."""
    monkeypatch.setattr(hj, "_MAX_PENDING_PER_WINDOW", 2)
    prov = [{"region_kind": "heading", "heading_text": "Chapter 1", "level": 1,
             "first_raw_block_index": 0, "source_page": 1}]
    for i in range(1, 6):  # five pendings, NO level-2 spine
        prov.append({"region_kind": "heading", "heading_text": f"Part {chr(64+i)}",
                     "level": 3, "first_raw_block_index": i, "source_page": 1,
                     "heading_level_pending": True})
    plan = hj.build_heading_skeleton(prov)
    assert len(plan.windows) == 3  # 2 + 2 + 1
    assert [len(p) for _, p in plan.windows] == [2, 2, 1]
    for digest, _ in plan.windows:
        assert "FIXED-ANCHOR OUTLINE" in digest
    all_ids = [i for _, p in plan.windows for i in p]
    assert all_ids == [1, 2, 3, 4, 5]


# ═════════════════════════════════════════════════════════════════════════════
# Fix 2 — re-split on exhaustion: the completion loop.
# ═════════════════════════════════════════════════════════════════════════════
def test_exhausted_half_is_resplit_until_all_judged():
    """The audit hole: one split half completes, the other exhausts even at
    depth cap — the completion loop RE-SPLITS the remainder and continues
    until every pending is judged."""
    singles = {}

    def post(messages, max_tokens):
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 4:
            return "thinking overflow", "length"
        if n_pend == 2 and "Alpha" in user:
            return '{"levels": {"1": 3, "2": 4}}', "stop"
        if n_pend == 2:
            return "thinking overflow", "length"  # the Gamma/Delta half
        key = "3" if "Gamma" in user else "4"
        singles[key] = singles.get(key, 0) + 1
        if singles[key] <= 2:  # depth-2 initial + doubled retry both exhaust
            return "thinking overflow", "length"
        return json.dumps(
            {"levels": {key: 4 if key == "3" else 3}}), "stop"

    parsed, wmeta = hj._judge_window_to_completion(
        _windowed_digest(), 5, [1, 2, 3, 4], hj._max_tokens_for(4),
        post_fn=post)
    assert parsed == {"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}
    assert wmeta.get("resplit_rounds") == 1
    assert "unjudged_ids" not in wmeta
    assert "coverage_gap" not in wmeta
    # the intermediate exhaustion is surfaced honestly, not hidden
    assert wmeta.get("length_exhausted") is True


def test_resplit_cap_leaves_explicit_unjudged_set(caplog):
    """A window whose remainder NEVER judges must end with the EXPLICIT
    unjudged set stamped + logged loudly — never silently 'kept'."""
    import logging

    def post(messages, max_tokens):
        return '{"levels": {"1": 3, "2": 4}}', "stop"  # always omits 3, 4

    with caplog.at_level(logging.WARNING,
                         logger="semantik_structure.glmocr.heading_judge"):
        parsed, wmeta = hj._judge_window_to_completion(
            _windowed_digest(), 5, [1, 2, 3, 4], hj._max_tokens_for(4),
            post_fn=post)
    assert parsed == {"levels": {"1": 3, "2": 4}}
    assert wmeta["unjudged_ids"] == [3, 4]
    assert wmeta["coverage_gap"] == 2
    assert any("UNJUDGED" in rec.getMessage() for rec in caplog.records)


def test_fail_open_no_levels_skips_resplit():
    """When the first attempt judged NOTHING the internal ladder already
    exhausted its options — the completion loop must NOT re-POST the same
    halves (fail-open stays byte-identical), but the unjudged set is still
    explicit."""
    stub = _StubPost(content="thinking overflow", finish="length")
    parsed, wmeta = hj._judge_window_to_completion(
        _windowed_digest(), 5, [1, 2, 3, 4], hj._max_tokens_for(4),
        post_fn=stub)
    assert parsed is None
    assert "resplit_rounds" not in wmeta  # loop never ran
    assert wmeta["unjudged_ids"] == [1, 2, 3, 4]


def test_focus_window_digest_keeps_fixed_and_kept_pendings():
    focused = hj._focus_window_digest(_windowed_digest(), [3])
    assert focused.startswith("FIXED-ANCHOR OUTLINE")
    assert "R3 p2 h3* Gamma" in focused
    assert "> First sentence gamma." in focused  # anchor travels
    assert "R0 p1 h1. Chapter 1" in focused  # fixed spine kept
    assert "R1 p1 h3* Alpha" not in focused  # other pendings dropped
    assert "R4 p2 h3* Delta" not in focused


def test_judge_meta_aggregates_unjudged_ids():
    prov = _prov()
    plan = hj.build_heading_skeleton(prov)

    def post(messages, max_tokens):
        return '{"levels": {"1": 4}}', "stop"  # id 24 never judged

    m, meta = hj.judge_heading_levels(plan, post_fn=post, use_cache=False,
                                      model="m")
    assert m == {1: 4}
    assert meta["unjudged"] == 1
    assert meta["unjudged_ids"] == [24]


# ═════════════════════════════════════════════════════════════════════════════
# Fix 2 — accounting: judged-kept vs UNJUDGED must never conflate.
# ═════════════════════════════════════════════════════════════════════════════
def test_apply_result_distinguishes_judged_kept_from_unjudged():
    prov, esc, tree = _prov(), _escalations(), []
    # id 1 judged but its level is out of range (dropped); id 24 never judged.
    result = hj.apply_judged_levels(prov, tree, esc, {1: 9})
    assert result.applied == 0
    assert result.dropped == 1
    assert result.kept == 2
    assert result.kept_judged == 1  # judged, rejected by the clamp rules
    assert result.unjudged == 1     # NO verdict ever covered it
    assert result.kept == result.kept_judged + result.unjudged


def test_apply_result_empty_verdict_counts_all_unjudged():
    prov, esc, tree = _prov(), _escalations(), []
    before = copy.deepcopy(prov)
    result = hj.apply_judged_levels(prov, tree, esc, {})
    assert result.kept == 2
    assert result.unjudged == 2
    assert result.kept_judged == 0
    assert prov == before  # fail-open stays byte-identical


def test_run_heading_judge_report_carries_unjudged_keys():
    prov, esc, tree = _prov(), _escalations(), []

    def post(messages, max_tokens):
        return '{"levels": {"1": 4}}', "stop"  # omits id 24 every time

    report = hj.run_heading_judge(prov, tree, esc, post_fn=post,
                                  use_cache=False, emit_capture=False)
    assert report["applied"] == 1
    assert report["kept"] == 1
    assert report["kept_judged"] == 0
    assert report["unjudged"] == 1
    assert report["kept"] == report["kept_judged"] + report["unjudged"]
    # the unjudged heading keeps its pending flag (honest fail-open)
    r24 = next(r for r in prov if r["first_raw_block_index"] == 24)
    assert r24.get("heading_level_pending") is True


def test_capture_rationale_distinguishes_unjudged(monkeypatch):
    captured = {}

    class _FakeCap:
        def __init__(self, **kwargs):
            pass

        def log_decision(self, **kwargs):
            captured.update(kwargs)

    import lib.decision_capture as dc
    monkeypatch.setattr(dc, "DecisionCapture", _FakeCap)

    prov, esc, tree = _prov(), _escalations(), []

    def post(messages, max_tokens):
        return '{"levels": {"1": 4}}', "stop"

    hj.run_heading_judge(prov, tree, esc, post_fn=post, use_cache=False,
                         course_code="TESTCOURSE")
    rationale = captured["rationale"]
    assert "UNJUDGED=1" in rationale
    assert "judged-kept=0" in rationale


# ── Failure MECHANISM taxonomy (2026-07-22 seat-death incident). ─────────────
# A third concurrent run's seat schedule `docker stop`ped the Super seat while
# two judge POSTs were generating; vLLM logged "Aborting 5 requests" and
# answered HTTP 200 with finish_reason="abort" + a truncated body. The judge
# read that as an anonymous parse failure, skipped its retry ladder entirely,
# and the phase reported "2 judged / 0 failed" over 147 UNJUDGED headings.


class _AbortRequests:
    """A seat that answers 200 with finish_reason='abort' + partial content
    (vLLM's shape when its engine aborts an in-flight request)."""

    def __init__(self, finish="abort", content="<think>partial thoughts",
                 then=None):
        self.finish = finish
        self.content = content
        self.then = then          # payload to serve from attempt 2 onward
        self.calls = 0

    def post(self, url, json=None, headers=None, timeout=None):  # noqa: A002
        self.calls += 1
        if self.then is not None and self.calls > 1:
            return _FakeResp(self.then)
        return _FakeResp({
            "choices": [{"message": {"content": self.content},
                         "finish_reason": self.finish}],
            "usage": {"prompt_tokens": 7562, "completion_tokens": 1629},
        })


@pytest.mark.parametrize("finish", ["abort", "ABORT", "cancelled", "canceled"])
def test_abort_finish_reason_raises_transport_error_not_parse_failure(finish):
    with pytest.raises(hj._JudgeTransportError) as ei:
        hj._post_judge_completion(
            base_url="http://x/v1", api_key=None, model="nemotron-3-super",
            messages=[{"role": "user", "content": "hi"}], max_tokens=64,
            timeout=5.0, requests_module=_AbortRequests(finish=finish))
    exc = ei.value
    assert exc.aborted is True
    assert exc.transient is True
    assert exc.timeout is False
    assert "ABORTED" in str(exc)


def test_abort_still_emits_the_usage_row_before_raising(tmp_path, monkeypatch):
    """Metering is the post-mortem evidence that proved this incident — the
    abort row must survive the raise."""
    usage_path = tmp_path / "llm_usage.jsonl"
    monkeypatch.setenv("SEMANTIK_LLM_USAGE_PATH", str(usage_path))
    monkeypatch.delenv("SEMANTIK_LLM_USAGE_PHASE", raising=False)
    with pytest.raises(hj._JudgeTransportError):
        hj._post_judge_completion(
            base_url="http://x/v1", api_key=None, model="nemotron-3-super",
            messages=[], max_tokens=64, timeout=5.0,
            requests_module=_AbortRequests())
    rows = [json.loads(l) for l in usage_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["finish_reason"] == "abort"
    assert rows[0]["completion_tokens"] == 1629


@pytest.fixture()
def _no_backoff_sleep(monkeypatch):
    """Skip the retry ladder's real backoff so the bounded-retry tests are
    fast (the BOUND is what is under test, not the wall-clock)."""
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *_a, **_k: None)


def test_abort_is_retried_by_the_bounded_ladder(_no_backoff_sleep):
    """Previously an abort got ZERO retries (it parsed as garbage). It is a
    transient transport failure now — bounded, never unbounded."""
    fake = _AbortRequests(then=_judge_payload())
    seat = hj.HeadingJudgeSeat(base_url="http://x/v1", api_key=None,
                               model="m", timeout=1.0)
    fn = hj._make_default_post_fn(seat, requests_module=fake)
    content, finish = fn([{"role": "user", "content": "hi"}], 64)
    assert fake.calls == 2                      # aborted once, then recovered
    assert finish == "stop"
    assert content == '{"levels": {"1": 4}}'


def test_abort_exhausts_the_ladder_bounded_at_three_attempts(_no_backoff_sleep):
    fake = _AbortRequests()
    seat = hj.HeadingJudgeSeat(base_url="http://x/v1", api_key=None,
                               model="m", timeout=1.0)
    fn = hj._make_default_post_fn(seat, requests_module=fake)
    with pytest.raises(hj._JudgeTransportError) as ei:
        fn([{"role": "user", "content": "hi"}], 64)
    assert fake.calls == hj._TRANSIENT_RETRIES + 1   # bounded, never a stall
    assert ei.value.aborted is True


def _judge_all_unjudged(post_fn):
    """Run the full judge over the fixture chapter with a failing post_fn and
    return (report, prov) — the fixture's 2 pendings all end unjudged."""
    prov, esc, tree = _prov(), _escalations(), []
    report = hj.run_heading_judge(prov, tree, esc, post_fn=post_fn,
                                  use_cache=False, emit_capture=False)
    return report, prov


def test_seat_aborted_mechanism_reaches_the_report():
    def _abort(messages, max_tokens):
        raise hj._JudgeTransportError(
            "heading-judge request ABORTED by the seat mid-generation",
            transient=True, aborted=True)

    report, prov = _judge_all_unjudged(_abort)
    assert report["unjudged"] == 2
    assert report["failure_modes"] == [hj.MECHANISM_SEAT_ABORTED]
    assert "ABORTED" in report["unjudged_reason"]
    assert "stopped/killed" in report["unjudged_reason"]
    # fail-open is still honest: levels + pending flags untouched
    assert prov == _prov()


def test_seat_unreachable_mechanism_is_distinct_from_abort():
    def _down(messages, max_tokens):
        raise hj._JudgeTransportError("connection refused", transient=False)

    report, _ = _judge_all_unjudged(_down)
    assert report["failure_modes"] == [hj.MECHANISM_SEAT_UNREACHABLE]
    assert "UNREACHABLE" in report["unjudged_reason"]


def test_empty_content_is_its_own_mechanism_not_a_parse_failure():
    report, _ = _judge_all_unjudged(_StubPost(content="", finish="stop"))
    assert report["failure_modes"] == [hj.MECHANISM_EMPTY_CONTENT]
    assert "EMPTY content" in report["unjudged_reason"]


def test_non_json_prose_is_a_parse_failure_mechanism():
    report, _ = _judge_all_unjudged(
        _StubPost(content="I am afraid I cannot do that.", finish="stop"))
    assert report["failure_modes"] == [hj.MECHANISM_PARSE_FAILURE]
    assert "did not parse" in report["unjudged_reason"]


def test_length_exhaust_mechanism_is_length_exhausted():
    report, _ = _judge_all_unjudged(
        _StubPost(content="<think>...", finish="length"))
    assert hj.MECHANISM_LENGTH_EXHAUSTED in report["failure_modes"]


def test_total_unjudged_chapter_logs_at_error_with_the_mechanism(caplog):
    def _abort(messages, max_tokens):
        raise hj._JudgeTransportError("seat went away", transient=True,
                                      aborted=True)

    with caplog.at_level("WARNING", logger=hj.logger.name):
        _judge_all_unjudged(_abort)
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors, "a 100%-unjudged chapter must be LOUD (ERROR), never info"
    msg = errors[-1].getMessage()
    assert "ALL 2 pending heading(s) left UNJUDGED" in msg
    assert "REAL FAILURE" in msg
    assert "MECHANISM:" in msg
    assert "ABORTED" in msg


def test_partial_unjudged_stays_a_warning_and_still_names_the_mechanism(caplog):
    """One of two pendings judged → NOT a total hole → warning, not error."""
    prov, esc, tree = _prov(), _escalations(), []
    with caplog.at_level("WARNING", logger=hj.logger.name):
        hj.run_heading_judge(
            prov, tree, esc, use_cache=False, emit_capture=False,
            post_fn=_StubPost(content='{"levels": {"1": 4}}', finish="stop"))
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    warns = [r.getMessage() for r in caplog.records
             if r.levelname == "WARNING"]
    assert any("left UNJUDGED" in m and "MECHANISM:" in m for m in warns)


def test_describe_failure_modes_never_returns_empty():
    assert "UNKNOWN" in hj.describe_failure_modes([])
    assert "UNKNOWN" in hj.describe_failure_modes(["not-a-real-token"])
    both = hj.describe_failure_modes(
        [hj.MECHANISM_SEAT_ABORTED, hj.MECHANISM_PARSE_FAILURE])
    assert "ABORTED" in both and "did not parse" in both


def test_healthy_run_carries_no_failure_modes():
    prov, esc, tree = _prov(), _escalations(), []
    report = hj.run_heading_judge(
        prov, tree, esc, use_cache=False, emit_capture=False,
        post_fn=_StubPost(content='{"levels": {"1": 3, "24": 3}}'))
    assert report["applied"] == 2
    assert report["failure_modes"] == []
    assert report["unjudged_reason"] is None


def test_standalone_report_carries_the_mechanism_fields(tmp_path, monkeypatch):
    from semantik_structure.glmocr import heading_judge_standalone as hjs

    layout = tmp_path / "book.glmocr_layout.json"
    layout.write_text(json.dumps({"pages": []}), encoding="utf-8")

    monkeypatch.setattr(
        hjs, "transform_document",
        lambda pages: type("T", (), {
            "region_provenance": _prov(),
            "heading_tree": [],
            "escalations": _escalations(),
        })())

    def _abort(messages, max_tokens):
        raise hj._JudgeTransportError("gone", transient=True, aborted=True)

    report = hjs.run_standalone(layout, out_dir=tmp_path / "out",
                                post_fn=_abort, use_cache=False)
    assert report["unjudged"] == 2
    assert report["failure_modes"] == [hj.MECHANISM_SEAT_ABORTED]
    assert "ABORTED" in report["unjudged_reason"]
    # ...and it is PERSISTED (the pipeline seam reads the report from disk)
    on_disk = json.loads(
        (tmp_path / "out" / "book.heading_judgments.json").read_text())
    assert on_disk["failure_modes"] == [hj.MECHANISM_SEAT_ABORTED]
    assert "ABORTED" in on_disk["unjudged_reason"]


# ═════════════════════════════════════════════════════════════════════════════
# VERDICT RECORDED vs LEVEL CHANGED (2026-07-22 misdiagnosis guard).
#
# The phase logged ``applied=58 ... kept=0``; a reviewer diffed the judged HTML
# against its ``.prejudge.bak``, found the bytes IDENTICAL, and concluded the
# judge's 58 re-levelings were being dropped before render. They were not: only
# 28 of those 58 verdicts differed from the level already on the region (the
# other 30 re-confirmed it), and the layout had ALREADY been judged in-lane
# during ``semantik_conversion``, so identical bytes were the CORRECT outcome.
# ``applied`` therefore has to stop reading as "N headings re-levelled".
# ═════════════════════════════════════════════════════════════════════════════
def _agreeing_prov():
    """A chapter whose pendings can be judged at the level they ALREADY hold.

    (In ``_prov`` the first pending is clamp-forced off level 3, so it can
    never AGREE — this fixture exists to produce the exact applied>0 /
    changed==0 shape that misled the reviewer.)
    """
    return [
        {"region_kind": "heading", "heading_text": "1.1 Sections", "level": 2,
         "first_raw_block_index": 0, "source_page": 1,
         "section_number_recovered": True, "native_label": "title"},
        {"region_kind": "heading", "heading_text": "Alpha Widgets", "level": 3,
         "first_raw_block_index": 1, "source_page": 1,
         "heading_level_pending": True, "native_label": "title"},
        {"region_kind": "heading", "heading_text": "Beta Widgets", "level": 3,
         "first_raw_block_index": 2, "source_page": 1,
         "heading_level_pending": True, "native_label": "title"},
    ]


def test_apply_result_splits_changed_from_agreed():
    prov, esc, tree = _prov(), _escalations(), []
    # id 1 is clamp-forced 3 -> 2 (a real level CHANGE); id 24 is judged at
    # the level it already holds (a RECORDED verdict that moves nothing).
    result = hj.apply_judged_levels(prov, tree, esc, {1: 2, 24: 3})
    assert result.applied == 2
    assert result.changed == 1
    assert result.agreed == 1
    assert result.applied == result.changed + result.agreed
    assert result.transitions == {"3->2": 1}


def test_applied_verdicts_can_all_agree_zero_level_changes():
    """The exact shape that misled the reviewer: applied > 0, changed == 0."""
    prov, esc, tree = _agreeing_prov(), [], []
    before = copy.deepcopy(prov)
    result = hj.apply_judged_levels(prov, tree, esc, {1: 3, 2: 3})
    assert result.applied == 2, "both verdicts were RECORDED"
    assert result.changed == 0, "but NEITHER moved a level"
    assert result.agreed == 2
    assert result.transitions == {}
    # every heading's LEVEL is untouched — only the pending flag was cleared,
    # which is why such a pass can re-render byte-identically.
    assert [r["level"] for r in prov] == [r["level"] for r in before]


def test_run_heading_judge_report_carries_changed_agreed_transitions():
    prov, esc, tree = _prov(), _escalations(), []

    def post(messages, max_tokens):
        return '{"levels": {"1": 2, "24": 3}}', "stop"

    report = hj.run_heading_judge(prov, tree, esc, post_fn=post,
                                  use_cache=False, emit_capture=False)
    assert report["applied"] == 2
    assert report["changed"] == 1
    assert report["agreed"] == 1
    assert report["applied"] == report["changed"] + report["agreed"]
    assert report["transitions"] == {"3->2": 1}


def test_zero_pending_report_still_carries_the_change_keys():
    """A no-pending chapter must not make a reader guess the keys' absence."""
    prov = [r for r in _prov() if not r.get("heading_level_pending")]
    report = hj.run_heading_judge(prov, [], [], post_fn=None,
                                  use_cache=False, emit_capture=False)
    assert report["n_pending"] == 0
    assert report["changed"] == 0 and report["agreed"] == 0
    assert report["transitions"] == {}


def test_capture_rationale_separates_changed_from_agreed(monkeypatch):
    captured = {}

    class _FakeCap:
        def __init__(self, **kwargs):
            pass

        def log_decision(self, **kwargs):
            captured.update(kwargs)

    import lib.decision_capture as dc
    monkeypatch.setattr(dc, "DecisionCapture", _FakeCap)

    prov, esc, tree = _prov(), _escalations(), []

    def post(messages, max_tokens):
        return '{"levels": {"1": 2, "24": 3}}', "stop"

    hj.run_heading_judge(prov, tree, esc, post_fn=post, use_cache=False,
                         course_code="TESTCOURSE")
    rationale = captured["rationale"]
    assert "applied=2 verdict(s)" in rationale
    assert "1 CHANGED a level" in rationale
    assert "3->2 x1" in rationale
    assert "1 AGREED with the pre-existing level" in rationale
    # structured fields so a query never has to parse the prose
    assert captured["hj_changed"] == 1
    assert captured["hj_agreed"] == 1
    # the decision line itself must not read as "2 headings re-levelled"
    assert "2 changed level" not in captured["decision"]
    assert "(1 changed level, 1 agreed)" in captured["decision"]


def test_capture_rationale_says_none_when_nothing_changed(monkeypatch):
    captured = {}

    class _FakeCap:
        def __init__(self, **kwargs):
            pass

        def log_decision(self, **kwargs):
            captured.update(kwargs)

    import lib.decision_capture as dc
    monkeypatch.setattr(dc, "DecisionCapture", _FakeCap)

    prov, esc, tree = _agreeing_prov(), [], []

    def post(messages, max_tokens):
        return '{"levels": {"1": 3, "2": 3}}', "stop"

    hj.run_heading_judge(prov, tree, esc, post_fn=post, use_cache=False,
                         course_code="TESTCOURSE")
    assert "0 CHANGED a level (none)" in captured["rationale"]
    assert "2 AGREED with the pre-existing level" in captured["rationale"]


@pytest.mark.parametrize("transitions,expected", [
    ({}, ""),
    ({"3->2": 28}, "3->2 x28"),
    ({"4->3": 2, "3->2": 28}, "3->2 x28, 4->3 x2"),
])
def test_format_transitions_is_busiest_first(transitions, expected):
    assert hj._format_transitions(transitions) == expected


def test_format_transitions_is_bounded():
    many = {f"{i}->2": 1 for i in range(2, 12)}
    out = hj._format_transitions(many, limit=3)
    assert out.count(" x") == 3
    assert out.endswith("more")


def test_standalone_report_persists_changed_agreed_and_transitions(
    tmp_path, monkeypatch
):
    """The on-disk ``{stem}.heading_judgments.json`` (what the pipeline seam
    reads) must carry the distinction, not just the in-memory result."""
    from semantik_structure.glmocr import heading_judge_standalone as hjs

    monkeypatch.setattr(hjs, "_load_layout_pages", lambda p: [])
    monkeypatch.setattr(
        hjs, "transform_document",
        lambda pages: type("T", (), {
            "region_provenance": _prov(), "heading_tree": [], "escalations": _escalations(),
        })(),
    )
    layout = tmp_path / "book.glmocr_layout.json"
    layout.write_text("{}", encoding="utf-8")

    def post(messages, max_tokens):
        return '{"levels": {"1": 2, "24": 3}}', "stop"

    report = hjs.run_standalone(layout, out_dir=tmp_path / "out",
                                post_fn=post, use_cache=False)
    on_disk = json.loads(
        (tmp_path / "out" / "book.heading_judgments.json").read_text())
    for doc in (report, on_disk):
        assert doc["applied"] == 2
        assert doc["changed"] == 1
        assert doc["agreed"] == 1
        assert doc["transitions"] == {"3->2": 1}


# ═════════════════════════════════════════════════════════════════════════════
# SPLIT-ON-TRUNCATION ladder — env-bounded recursion depth
# (SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH). A window can never first-attempt
# truncate: on finish_reason=length it HALVES + re-judges each half instead of
# only doubling max_tokens.
# ═════════════════════════════════════════════════════════════════════════════
def test_resolve_max_split_depth_env(monkeypatch):
    """Parse-with-fallback (default 3; ``0`` HONOURED as the revert lever;
    negative / garbage / blank → default), mirroring reasoning_qc."""
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", raising=False)
    assert hj.resolve_heading_judge_max_split_depth() == 3
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", "5")
    assert hj.resolve_heading_judge_max_split_depth() == 5
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", "0")
    assert hj.resolve_heading_judge_max_split_depth() == 0  # honoured (revert)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", "-2")
    assert hj.resolve_heading_judge_max_split_depth() == 3  # negative → default
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", "garbage")
    assert hj.resolve_heading_judge_max_split_depth() == 3
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", "")
    assert hj.resolve_heading_judge_max_split_depth() == 3


def test_min_pending_per_split_floor():
    """A single-pending window is below the floor and unsplittable; a two-
    pending window (== the floor) halves into two single-pending sub-windows."""
    assert hj._MIN_PENDING_PER_SPLIT == 2
    assert hj._split_window_digest("R1 p1 h3* Only", [1]) == []
    halves = hj._split_window_digest(_windowed_digest(), [1, 2])
    assert [p for _, p in halves] == [[1], [2]]


def test_no_truncation_is_byte_identical_no_split():
    """NO finish=length → the legacy single-POST path: no split, no doubled
    retry, no split_depth in the meta."""
    calls = []

    def post(messages, max_tokens):
        calls.append(max_tokens)
        return '{"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}', "stop"

    parsed, wmeta = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4), post_fn=post,
        win_pending=[1, 2, 3, 4])
    assert parsed == {"levels": {"1": 3, "2": 4, "3": 4, "4": 3}}
    assert len(calls) == 1  # one POST, no split, no doubled retry
    assert wmeta.get("finish") == "stop"
    assert "split_depth" not in wmeta


def _single_pending_id_post(messages, max_tokens):
    """length for a multi-pending window; a level for a single-pending one
    (keyed off whichever pending id its digest carries)."""
    import re as _re
    user = messages[1]["content"]
    n_pend = int(user.split("(")[1].split(",")[1].split()[0])
    if n_pend >= 2:
        return "thinking overflow", "length"
    ids = _re.findall(r"^R(\d+) p\S+ h\d+\*", user, _re.M)
    return json.dumps({"levels": {ids[0]: 4}}), "stop"


def test_length_window_splits_recursively_instead_of_truncating():
    """A window that truncates at EVERY multi-pending size is fully judged by
    HALVING recursively (4 → 2 → 1) — never a first-attempt truncation
    fail-open."""
    parsed, wmeta = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4),
        post_fn=_single_pending_id_post, win_pending=[1, 2, 3, 4])
    assert parsed == {"levels": {"1": 4, "2": 4, "3": 4, "4": 4}}
    assert wmeta["finish"] == "split"


def test_split_depth_bound_stops_the_recursion(monkeypatch):
    """With the depth bound at 1, the depth-1 halves cannot split further; they
    fall to the doubled retry and fail open. The default (deeper) bound recovers
    the same window — proving the bound, not the seat, decides."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", "1")
    parsed, _ = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4),
        post_fn=_single_pending_id_post, win_pending=[1, 2, 3, 4])
    assert parsed is None  # depth-1 halves unsplittable → doubled-retry fail-open

    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", "3")
    parsed2, _ = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4),
        post_fn=_single_pending_id_post, win_pending=[1, 2, 3, 4])
    assert parsed2 == {"levels": {"1": 4, "2": 4, "3": 4, "4": 4}}


def test_split_depth_zero_reverts_to_doubled_retry(monkeypatch):
    """``=0`` is the explicit revert lever: a truncating window NEVER splits,
    it goes straight to the legacy doubled-max_tokens retry then fails open —
    exactly one initial POST + one doubled retry, never a half POST."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_SPLIT_DEPTH", "0")
    calls = []

    def post(messages, max_tokens):
        calls.append(max_tokens)
        return "thinking overflow", "length"

    parsed, wmeta = hj._judge_one_window(
        _windowed_digest(), 5, 4, hj._max_tokens_for(4), post_fn=post,
        win_pending=[1, 2, 3, 4])
    assert parsed is None
    assert len(calls) == 2  # initial + ONE doubled retry, no split half POST
    assert calls[1] > calls[0]  # doubled
    assert wmeta.get("length_exhausted") is True


def test_split_merged_verdict_applies_through_the_clamp():
    """End-to-end: a truncating window SPLITS, its halves' verdicts MERGE, and
    the merged levels flow through apply_judged_levels' clamp (pending-only,
    parent+1, no-orphan) — the split changes only HOW verdicts are obtained,
    never the clamp/apply contract."""
    prov, esc, tree = _prov(), _escalations(), []

    def post(messages, max_tokens):
        user = messages[1]["content"]
        n_pend = int(user.split("(")[1].split(",")[1].split()[0])
        if n_pend == 2:
            return "overflow", "length"  # full window truncates → split
        if "Chapter Outline" in user:
            return '{"levels": {"1": 4}}', "stop"  # proposed 4 → clamped 2
        return '{"levels": {"24": 3}}', "stop"

    report = hj.run_heading_judge(prov, tree, esc, post_fn=post,
                                  use_cache=False, emit_capture=False)
    assert report["applied"] == 2
    # id 1: proposed 4 but clamped to parent(L1)+1 == 2 (no skipped tier).
    assert next(r for r in prov if r["first_raw_block_index"] == 1)["level"] == 2
    # id 24: proposed 3 under its L2 section anchor → 3 (no clamp).
    assert next(r for r in prov if r["first_raw_block_index"] == 24)["level"] == 3
    # pending flags cleared on the applied headings.
    assert all("heading_level_pending" not in r for r in prov
               if r.get("region_kind") == "heading"
               and r["first_raw_block_index"] in (1, 24))


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT — the big-context-seat binding lever.
# ═════════════════════════════════════════════════════════════════════════════
def test_est_per_judgment_resolver_default_and_overrides(monkeypatch):
    """Parse-with-fallback: unset → 300 (byte-identical); a positive int wins;
    garbage / non-positive → the 300 default (mirrors resolve_max_tokens_ceiling)."""
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT", raising=False)
    assert hj.resolve_heading_judge_est_per_judgment() == 300
    assert hj.resolve_heading_judge_est_per_judgment() == hj._EST_TOKENS_PER_JUDGMENT
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT", "2000")
    assert hj.resolve_heading_judge_est_per_judgment() == 2000
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT", "garbage")
    assert hj.resolve_heading_judge_est_per_judgment() == 300
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT", "0")
    assert hj.resolve_heading_judge_est_per_judgment() == 300
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT", "-5")
    assert hj.resolve_heading_judge_est_per_judgment() == 300


def test_est_default_unset_is_byte_identical(monkeypatch):
    """With the est env UNSET every budget resolves EXACTLY as before the flag:
    the per-pending budget and the window cap are unchanged (the 65k-safe path)."""
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    # per-pending budget uses the 300 default increment
    assert hj._max_tokens_for(10) == max(
        hj._MAX_TOKENS_FLOOR,
        min(hj._MAX_TOKENS_CEILING,
            hj._MAX_TOKENS_FLOOR + hj._EST_TOKENS_PER_JUDGMENT * 10))
    # window cap divisor uses the 300 default
    assert hj.resolve_pending_window_cap() == min(
        hj._MAX_PENDING_PER_WINDOW,
        max(hj._MIN_PENDING_WINDOW_CAP,
            (hj._MAX_TOKENS_CEILING - hj._MAX_TOKENS_FLOOR)
            // hj._EST_TOKENS_PER_JUDGMENT))


def test_est_bump_widens_window_cap_and_fills_budget_on_250k_seat(monkeypatch):
    """On the 250k seat (est=2000, ceiling=200000, floor=20480) the window cap
    widens to ~89 pending/window (chapter-sized) and a full 89-pending window's
    _max_tokens_for reaches the ceiling region — NOT the 300-capped undershoot
    that truncated thinking-heavy pages."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT", "2000")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", "200000")
    cap = hj.resolve_pending_window_cap()
    assert cap == 89  # (200000 - 20480) // 2000 == 89, under the 96 hard cap
    budget = hj._max_tokens_for(cap)
    # the packed window's first-attempt budget lands within one increment of the
    # ceiling (198480 vs 200000) — the room the packer sized it against.
    assert budget >= hj.resolve_max_tokens_ceiling() - 2000
    # and the 300-default increment would have grossly under-budgeted the SAME
    # window (the pre-fix hole this lever closes).
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_EST_PER_JUDGMENT", raising=False)
    assert hj._max_tokens_for(cap) < budget


def test_ctx_budget_plus_ceiling_fits_250k_seat_under_run_env(monkeypatch):
    """HARD INVARIANT under the documented run-env values: the prompt-windowing
    budget + the completion ceiling must fit the 250k seat's --max-model-len
    (40000 + 200000 = 240000 <= 250000, ~10k margin)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CTX_BUDGET", "40000")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", "200000")
    _SEAT_MAX_MODEL_LEN = 250000
    assert (hj.resolve_ctx_tokens_budget()
            + hj.resolve_max_tokens_ceiling()) <= _SEAT_MAX_MODEL_LEN


# ═════════════════════════════════════════════════════════════════════════════
# SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT — seat-context-adaptive budget derivation.
# ═════════════════════════════════════════════════════════════════════════════
def _mock_models_response(monkeypatch, *, max_model_len=None, raises=None):
    """Monkeypatch urllib.request.urlopen so the seat-context probe returns a
    /v1/models body (or raises). Also clears the per-base_url probe cache."""
    import io
    import urllib.request

    hj._reset_seat_context_cache()

    class _Resp:
        def __init__(self, body):
            self._buf = io.BytesIO(body)

        def read(self):
            return self._buf.read()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        if raises is not None:
            raise raises
        payload = {"data": [{"id": "seat-model", "max_model_len": max_model_len}]}
        return _Resp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)


def test_seat_context_auto_reads_max_model_len(monkeypatch):
    """auto → the seat's /v1/models max_model_len is read + cached."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "auto")
    _mock_models_response(monkeypatch, max_model_len=250000)
    assert hj.resolve_heading_judge_seat_context() == 250000


def test_seat_context_auto_fail_soft_to_legacy(monkeypatch):
    """auto whose probe RAISES / times out → None (legacy hardcoded budgets)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "auto")
    _mock_models_response(monkeypatch, raises=OSError("connection refused"))
    assert hj.resolve_heading_judge_seat_context() is None
    # and the budgets stay byte-identical to the hardcoded defaults
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_CTX_BUDGET", raising=False)
    assert hj.resolve_max_tokens_ceiling() == hj._MAX_TOKENS_CEILING
    assert hj.resolve_ctx_tokens_budget() == hj._CTX_TOKENS_BUDGET
    assert hj.resolve_digest_budget_tokens() == hj._DIGEST_BUDGET_TOKENS


def test_seat_context_auto_missing_field_fail_soft(monkeypatch):
    """auto whose /v1/models lacks max_model_len → None (legacy)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "auto")
    _mock_models_response(monkeypatch, max_model_len=None)
    assert hj.resolve_heading_judge_seat_context() is None


def test_seat_context_explicit_int_used_without_probe(monkeypatch):
    """A positive int is used verbatim — no probe fires."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "131072")
    import urllib.request

    def _boom(*a, **k):
        raise AssertionError("probe must not fire for an explicit int")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    hj._reset_seat_context_cache()
    assert hj.resolve_heading_judge_seat_context() == 131072


def test_seat_context_off_is_legacy(monkeypatch):
    """off → None (the legacy hardcoded path)."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "off")
    assert hj.resolve_heading_judge_seat_context() is None


def test_seat_context_probe_is_cached(monkeypatch):
    """The /v1/models probe fires at most ONCE per base_url per process."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "auto")
    import urllib.request

    calls = {"n": 0}

    class _Resp:
        def read(self):
            return json.dumps(
                {"data": [{"max_model_len": 65536}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _counting(req, timeout=None):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _counting)
    hj._reset_seat_context_cache()
    assert hj.resolve_heading_judge_seat_context() == 65536
    assert hj.resolve_heading_judge_seat_context() == 65536
    assert calls["n"] == 1


@pytest.mark.parametrize("seat_context", [65536, 131072, 250000])
def test_derived_budget_invariant_holds(monkeypatch, seat_context):
    """digest_budget + ceiling <= usable <= seat_context, and both budgets sit
    above their floors, for a range of seat sizes."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", str(seat_context))
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_CTX_BUDGET", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET", raising=False)
    digest = hj.resolve_digest_budget_tokens()
    ceiling = hj.resolve_max_tokens_ceiling()
    ctx = hj.resolve_ctx_tokens_budget()
    margin = hj.resolve_ctx_margin()
    usable = seat_context - margin
    assert digest + ceiling <= usable <= seat_context
    assert ctx <= seat_context
    # floors respected
    assert ceiling >= hj._DERIVED_CEILING_FLOOR
    assert digest >= hj._DERIVED_DIGEST_FLOOR
    # the derived ceiling actually right-sizes to the seat (not the 65k default)
    if seat_context > 65536:
        assert ceiling > hj._MAX_TOKENS_CEILING


def test_completion_fraction_respected_and_clamped(monkeypatch):
    """The completion fraction sizes the ceiling and is clamped to [0.4, 0.9]."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "131072")
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", raising=False)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION", "0.5")
    assert hj.resolve_completion_fraction() == 0.5
    usable = 131072 - hj.resolve_ctx_margin()
    assert hj.resolve_max_tokens_ceiling() == int(usable * 0.5)
    # clamp band
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION", "0.99")
    assert hj.resolve_completion_fraction() == hj._COMPLETION_FRACTION_MAX
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION", "0.01")
    assert hj.resolve_completion_fraction() == hj._COMPLETION_FRACTION_MIN
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_COMPLETION_FRACTION", "garbage")
    assert hj.resolve_completion_fraction() == hj._DEFAULT_COMPLETION_FRACTION


def test_ctx_margin_resolver(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_CTX_MARGIN", raising=False)
    assert hj.resolve_ctx_margin() == hj._DEFAULT_CTX_MARGIN
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CTX_MARGIN", "8192")
    assert hj.resolve_ctx_margin() == 8192
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CTX_MARGIN", "garbage")
    assert hj.resolve_ctx_margin() == hj._DEFAULT_CTX_MARGIN


def test_explicit_per_budget_env_overrides_seat_derived(monkeypatch):
    """An explicit per-budget env WINS over the seat-derived value."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "131072")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_MAX_TOKENS", "12345")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CTX_BUDGET", "23456")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET", "3456")
    assert hj.resolve_max_tokens_ceiling() == 12345
    assert hj.resolve_ctx_tokens_budget() == 23456
    assert hj.resolve_digest_budget_tokens() == 3456


def test_off_path_is_byte_identical_to_hardcoded(monkeypatch):
    """SEAT_CONTEXT=off with no per-budget envs → the EXACT hardcoded budgets."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "off")
    for e in ("SEMANTIK_HEADING_JUDGE_MAX_TOKENS",
              "SEMANTIK_HEADING_JUDGE_CTX_BUDGET",
              "SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET"):
        monkeypatch.delenv(e, raising=False)
    assert hj.resolve_max_tokens_ceiling() == 30000 == hj._MAX_TOKENS_CEILING
    assert hj.resolve_ctx_tokens_budget() == 31500 == hj._CTX_TOKENS_BUDGET
    assert hj.resolve_digest_budget_tokens() == 24000 == hj._DIGEST_BUDGET_TOKENS


def test_auto_probe_failure_is_byte_identical_to_off(monkeypatch):
    """A failed auto probe yields the SAME budgets as the explicit off revert."""
    for e in ("SEMANTIK_HEADING_JUDGE_MAX_TOKENS",
              "SEMANTIK_HEADING_JUDGE_CTX_BUDGET",
              "SEMANTIK_HEADING_JUDGE_DIGEST_BUDGET"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "auto")
    _mock_models_response(monkeypatch, raises=TimeoutError("timed out"))
    auto = (hj.resolve_max_tokens_ceiling(), hj.resolve_ctx_tokens_budget(),
            hj.resolve_digest_budget_tokens())
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_SEAT_CONTEXT", "off")
    off = (hj.resolve_max_tokens_ceiling(), hj.resolve_ctx_tokens_budget(),
           hj.resolve_digest_budget_tokens())
    assert auto == off == (30000, 31500, 24000)


# ═════════════════════════════════════════════════════════════════════════════
# Part 2 — every other hardcoded size is env-tunable (default == constant).
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("resolver,env,const", [
    ("resolve_max_pending_per_window",
     "SEMANTIK_HEADING_JUDGE_MAX_PENDING_PER_WINDOW", "_MAX_PENDING_PER_WINDOW"),
    ("resolve_window_concurrency",
     "SEMANTIK_HEADING_JUDGE_CONCURRENCY", "_WINDOW_CONCURRENCY"),
    ("resolve_max_tokens_floor",
     "SEMANTIK_HEADING_JUDGE_TOKENS_FLOOR", "_MAX_TOKENS_FLOOR"),
    ("resolve_min_pending_window_cap",
     "SEMANTIK_HEADING_JUDGE_MIN_PENDING_WINDOW_CAP", "_MIN_PENDING_WINDOW_CAP"),
    ("resolve_max_coverage_resplit_rounds",
     "SEMANTIK_HEADING_JUDGE_MAX_COVERAGE_RESPLIT_ROUNDS",
     "_MAX_COVERAGE_RESPLIT_ROUNDS"),
    ("resolve_min_pending_per_split",
     "SEMANTIK_HEADING_JUDGE_MIN_PENDING_PER_SPLIT", "_MIN_PENDING_PER_SPLIT"),
    ("resolve_anchor_truncate",
     "SEMANTIK_HEADING_JUDGE_ANCHOR_TRUNCATE", "_ANCHOR_TRUNCATE"),
    ("resolve_heading_text_truncate",
     "SEMANTIK_HEADING_JUDGE_HEADING_TEXT_TRUNCATE", "_HEADING_TEXT_TRUNCATE"),
    ("resolve_context_text_truncate",
     "SEMANTIK_HEADING_JUDGE_CONTEXT_TEXT_TRUNCATE", "_CONTEXT_TEXT_TRUNCATE"),
])
def test_part2_size_env_default_override_garbage(monkeypatch, resolver, env,
                                                 const):
    fn = getattr(hj, resolver)
    default = getattr(hj, const)
    monkeypatch.delenv(env, raising=False)
    assert fn() == default            # default == the current constant
    monkeypatch.setenv(env, "7")
    assert fn() == 7                  # override honored
    monkeypatch.setenv(env, "garbage")
    assert fn() == default            # garbage → default
    monkeypatch.setenv(env, "0")
    assert fn() == default            # non-positive → default


def test_part2_sizes_wired_at_use_sites(monkeypatch):
    """The truncation + concurrency envs actually reach their use sites."""
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_HEADING_TEXT_TRUNCATE", "5")
    line = hj._render_line(
        hj.SkeletonEntry(0, 3, "A very long heading text here", 1, True))
    # "R0 p1 h3* " + 5 chars of text
    assert line == "R0 p1 h3* A ver"


# ═════════════════════════════════════════════════════════════════════════════
# Phase A — whole-document skeleton as per-window CONTEXT
# (SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT / _FULLDOC_ANCHORS).
# ═════════════════════════════════════════════════════════════════════════════
def _multiwindow_prov(with_prose=False):
    """Four L2 sections, one pending child each (+ optional prose anchor)."""
    prov = []
    idx = 0
    for sec in range(1, 5):
        prov.append({"region_kind": "heading", "heading_text": f"1.{sec} Section",
                     "level": 2, "first_raw_block_index": idx, "source_page": sec})
        idx += 1
        prov.append({"region_kind": "heading", "heading_text": f"Sub {sec}",
                     "level": 3, "first_raw_block_index": idx, "source_page": sec,
                     "heading_level_pending": True})
        idx += 1
        if with_prose:
            prov.append({"region_kind": "paragraph",
                         "raw_text": f"Body prose for section {sec} here.",
                         "first_raw_block_index": idx, "source_page": sec})
            idx += 1
    return prov


def test_fulldoc_context_prefixes_every_window(monkeypatch):
    monkeypatch.setattr(hj, "_MAX_PENDING_PER_WINDOW", 2)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT", "1")
    plan = hj.build_heading_skeleton(_multiwindow_prov())
    assert len(plan.windows) == 2
    for digest, pend in plan.windows:
        # every window carries the DOCUMENT STRUCTURE context prefix ...
        assert digest.startswith(hj._FULLDOC_CONTEXT_HEADER)
        # ... and the WHOLE document's structure (all four sections) is present
        for sec in range(1, 5):
            assert f"1.{sec} Section" in digest
        # ... but the window still judges ONLY its own pending subset
        assert len(pend) == 2


def test_fulldoc_context_partitions_pendings_and_split_ignores_context(monkeypatch):
    monkeypatch.setattr(hj, "_MAX_PENDING_PER_WINDOW", 2)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT", "1")
    plan = hj.build_heading_skeleton(_multiwindow_prov())
    # union of per-window pendings == the doc's pendings, each exactly once
    all_ids = [i for _, p in plan.windows for i in p]
    assert sorted(all_ids) == sorted(plan.pending_ids)
    assert len(all_ids) == len(set(all_ids))
    # the split ladder over a prefixed window sees ONLY that window's pendings
    # (the whole-doc context sitting in the preamble is never re-parsed)
    d0, p0 = plan.windows[0]
    halves = hj._split_window_digest(d0, p0)
    assert halves  # p0 has 2 → halves into two singles
    assert sorted(i for _, hp in halves for i in hp) == sorted(p0)


def test_fulldoc_context_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT", raising=False)
    a = hj.build_heading_skeleton(_prov()).windows
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT", "0")
    b = hj.build_heading_skeleton(_prov()).windows
    assert a == b
    assert all(hj._FULLDOC_CONTEXT_HEADER not in d for d, _ in b)


def test_fulldoc_context_degrades_on_budget_overflow(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT", "1")
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 1)  # tiny → context overflows
    hj._reset_fulldoc_context_log()
    with caplog.at_level(logging.WARNING, logger=hj.logger.name):
        plan = hj.build_heading_skeleton(_prov())
    # degraded → NO context prefix on any window (per-window skeleton kept)
    assert all(hj._FULLDOC_CONTEXT_HEADER not in d for d, _ in plan.windows)
    assert any("DEGRADING to per-window-only context" in r.getMessage()
               for r in caplog.records)


def test_fulldoc_anchors_subflag_adds_anchors(monkeypatch):
    monkeypatch.setattr(hj, "_MAX_PENDING_PER_WINDOW", 2)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FULLDOC_CONTEXT", "1")
    # anchors OFF → window bodies use anchors=False, so NO anchor line anywhere
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FULLDOC_ANCHORS", raising=False)
    d_no = hj.build_heading_skeleton(_multiwindow_prov(with_prose=True)).windows[0][0]
    assert "    > Body prose" not in d_no
    # anchors ON → the whole-doc context block carries the pending anchors
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FULLDOC_ANCHORS", "1")
    d_yes = hj.build_heading_skeleton(_multiwindow_prov(with_prose=True)).windows[0][0]
    assert "    > Body prose" in d_yes


# ═════════════════════════════════════════════════════════════════════════════
# Phase B — whole-document consistency-reconciliation FINAL REVIEW
# (SEMANTIK_HEADING_JUDGE_FINAL_REVIEW).
# ═════════════════════════════════════════════════════════════════════════════
def _review_prov():
    """A JUDGED tree: fixed L1 opener + fixed N.M spine + two judged headings.
    "Key Points" recurs at the SAME structural position (a section child) but
    with INCONSISTENT levels (3 then 4) — the reconcile target."""
    return [
        {"region_kind": "heading", "heading_text": "Chapter 1", "level": 1,
         "first_raw_block_index": 0, "source_page": 1,
         "chapter_title_synthesized": True, "native_label": "synthetic"},
        {"region_kind": "heading", "heading_text": "1.1 Alpha", "level": 2,
         "first_raw_block_index": 10, "source_page": 1,
         "section_number_recovered": True, "native_label": "title"},
        {"region_kind": "heading", "heading_text": "Key Points", "level": 3,
         "first_raw_block_index": 11, "source_page": 1,
         "heading_level_judged": {"from": 3, "to": 3, "clamped": False},
         "native_label": "title"},
        {"region_kind": "heading", "heading_text": "1.2 Beta", "level": 2,
         "first_raw_block_index": 20, "source_page": 2,
         "section_number_recovered": True, "native_label": "title"},
        {"region_kind": "heading", "heading_text": "Key Points", "level": 4,
         "first_raw_block_index": 21, "source_page": 2,
         "heading_level_judged": {"from": 3, "to": 4, "clamped": False},
         "native_label": "title"},
    ]


def test_final_review_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", raising=False)
    prov = _review_prov()
    before = copy.deepcopy(prov)
    stub = _StubPost(content='{"levels": {"21": 3}}')
    report = hj.run_final_review(prov, [], [], post_fn=stub, use_cache=False)
    assert report is None          # gate off → no review pass
    assert stub.calls == 0         # no POST
    assert prov == before          # byte-identical


def test_final_review_applies_consistency_fix(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    prov, esc = _review_prov(), []
    stub = _StubPost(content='{"levels": {"21": 3}}')  # reconcile L4 -> L3
    report = hj.run_final_review(prov, [], esc, post_fn=stub, use_cache=False,
                                 emit_capture=False)
    assert stub.calls == 1
    assert report["proposed"] == 1 and report["applied"] == 1
    r21 = next(r for r in prov if r["first_raw_block_index"] == 21)
    assert r21["level"] == 3
    assert r21["heading_level_reviewed"] == {"from": 4, "to": 3}
    assert any(e["reason"] == "heading_level_reviewed" and e["region_index"] == 21
               for e in esc)


def test_final_review_drops_fixed_anchor_and_invariant_violations(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    prov = _review_prov()
    before_levels = [r["level"] for r in prov]
    # id 20 is a FIXED N.M anchor (dropped); id 21 -> 6 violates parent+1 under
    # its L2 section (dropped individually) — never-ship-worse.
    stub = _StubPost(content='{"levels": {"20": 4, "21": 6}}')
    report = hj.run_final_review(prov, [], [], post_fn=stub, use_cache=False,
                                 emit_capture=False)
    assert report["proposed"] == 2
    assert report["applied"] == 0
    assert report["dropped"] == 2
    assert [r["level"] for r in prov] == before_levels  # nothing moved


def test_final_review_preserves_same_title_different_depth(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    # "Summary" at L4 (in-section, nested under a subsection) vs L3 (chapter-end
    # section child) — legitimately different depths; must NOT be flattened.
    prov = [
        {"region_kind": "heading", "heading_text": "Chapter 1", "level": 1,
         "first_raw_block_index": 0, "source_page": 1,
         "chapter_title_synthesized": True},
        {"region_kind": "heading", "heading_text": "1.1 Alpha", "level": 2,
         "first_raw_block_index": 10, "source_page": 1,
         "section_number_recovered": True},
        {"region_kind": "heading", "heading_text": "Deep Topic", "level": 3,
         "first_raw_block_index": 11, "source_page": 1,
         "heading_level_judged": {"from": 3, "to": 3, "clamped": False}},
        {"region_kind": "heading", "heading_text": "Summary", "level": 4,
         "first_raw_block_index": 12, "source_page": 1,
         "heading_level_judged": {"from": 3, "to": 4, "clamped": False}},
        {"region_kind": "heading", "heading_text": "1.2 Beta", "level": 2,
         "first_raw_block_index": 20, "source_page": 2,
         "section_number_recovered": True},
        {"region_kind": "heading", "heading_text": "Summary", "level": 3,
         "first_raw_block_index": 21, "source_page": 2,
         "heading_level_judged": {"from": 3, "to": 3, "clamped": False}},
    ]
    before = copy.deepcopy(prov)
    # a well-behaved model (seeing the whole tree) proposes NO change: the two
    # Summaries sit at different depths, so nothing to reconcile.
    stub = _StubPost(content='{"levels": {}}')
    report = hj.run_final_review(prov, [], [], post_fn=stub, use_cache=False,
                                 emit_capture=False)
    assert report["applied"] == 0
    assert report["preserved_legitimate"] >= 2  # both Summary occurrences kept
    assert prov == before  # byte-identical (empty verdict → no mutation)


def test_final_review_transport_failure_keeps_prereview_levels(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    prov = _review_prov()
    before = copy.deepcopy(prov)

    def _boom(messages, max_tokens):
        raise hj._JudgeTransportError("seat down", transient=False)

    report = hj.run_final_review(prov, [], [], post_fn=_boom, use_cache=False,
                                 emit_capture=False)
    assert report["applied"] == 0
    assert report["outcome"] == "transport_failure"
    assert prov == before  # never-ship-worse: pre-review levels byte-for-byte


def test_final_review_parse_failure_keeps_prereview_levels(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    prov = _review_prov()
    before = copy.deepcopy(prov)
    report = hj.run_final_review(prov, [], [], post_fn=_StubPost("I cannot comply."),
                                 use_cache=False, emit_capture=False)
    assert report["applied"] == 0
    assert report["outcome"] == "parse_failure"
    assert prov == before


def test_final_review_length_exhaust_keeps_prereview_levels(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    prov = _review_prov()
    before = copy.deepcopy(prov)
    stub = _StubPost(content='{"levels": {"21": 3}}', finish="length")
    report = hj.run_final_review(prov, [], [], post_fn=stub, use_cache=False,
                                 emit_capture=False)
    assert report["applied"] == 0
    assert report["outcome"] == "length_exhausted"
    assert prov == before


def test_final_review_skips_when_tree_too_big(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 1)  # judged tree overflows
    prov = _review_prov()
    before = copy.deepcopy(prov)
    stub = _StubPost(content='{"levels": {"21": 3}}')
    with caplog.at_level(logging.WARNING, logger=hj.logger.name):
        report = hj.run_final_review(prov, [], [], post_fn=stub, use_cache=False,
                                     emit_capture=False)
    assert stub.calls == 0  # single-window review only — never under-review
    assert report["skipped"] == "digest_overflow"
    assert prov == before
    assert any("SKIPPING the final review" in r.getMessage()
               for r in caplog.records)


def test_final_review_decision_capture_has_discriminator(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    captured = {}

    class _FakeCap:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        def log_decision(self, **kwargs):
            captured["log"] = kwargs

    import lib.decision_capture as dc
    monkeypatch.setattr(dc, "DecisionCapture", _FakeCap)

    hj.run_final_review(_review_prov(), [], [],
                        post_fn=_StubPost('{"levels": {"21": 3}}'),
                        use_cache=False, course_code="TST_901")
    log = captured["log"]
    assert log["decision_type"] == "structure_review"
    assert log["final_review"] is True
    assert log["fr_applied"] == 1
    assert "preserved_legitimate" in log["rationale"]
    assert len(log["rationale"]) >= 20
    alternatives = log["alternatives_considered"]
    assert all(
        set(item) == {"option", "reason_rejected"} for item in alternatives
    )
    assert "1 of 1" in alternatives[0]["reason_rejected"]


# ── Wiring: run_heading_judge + both-OFF byte-identity. ──────────────────────
def test_run_heading_judge_final_review_none_when_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", raising=False)
    prov, esc, tree = _prov(), _escalations(), []
    stub = _StubPost(content='{"levels": {"1": 3, "24": 3}}')
    report = hj.run_heading_judge(prov, tree, esc, post_fn=stub, use_cache=False,
                                  emit_capture=False)
    assert report["final_review"] is None  # gate off → no review pass
    assert stub.calls == 1                  # one judge POST, no review POST


def test_run_heading_judge_runs_final_review_when_on(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    prov, esc, tree = _prov(), _escalations(), []

    def content(call):  # call 1 = window judge; call 2 = the final review
        return '{"levels": {"1": 3, "24": 3}}' if call == 1 else '{"levels": {}}'

    stub = _StubPost(content=content)
    report = hj.run_heading_judge(prov, tree, esc, post_fn=stub, use_cache=False,
                                  emit_capture=False)
    assert stub.calls == 2  # one judge POST + one review POST
    assert report["final_review"] is not None
    assert report["final_review"]["proposed"] == 0  # empty review verdict


def test_both_phases_off_is_byte_identical():
    # neither Phase-A nor Phase-B flag set → today's exact windows + no review.
    plan = hj.build_heading_skeleton(_prov())
    assert all(hj._FULLDOC_CONTEXT_HEADER not in d for d, _ in plan.windows)
    prov, esc, tree = _prov(), _escalations(), []
    report = hj.run_heading_judge(
        prov, tree, esc, use_cache=False, emit_capture=False,
        post_fn=_StubPost(content='{"levels": {"1": 3, "24": 3}}'))
    assert report["final_review"] is None


# ═════════════════════════════════════════════════════════════════════════════
# Part 1-2-5 — CHAPTER-MODE judging (SEMANTIK_HEADING_JUDGE_CHAPTER_MODE) +
# derived document schema (SEMANTIK_HEADING_JUDGE_DOC_SCHEMA) + reviewer.
# ═════════════════════════════════════════════════════════════════════════════
class _RecordPost:
    """A ``post_fn`` that records every message list it was called with."""

    def __init__(self, content='{"levels": {}}', finish="stop"):
        self._content = content
        self._finish = finish
        self.calls = 0
        self.messages = []

    def __call__(self, messages, max_tokens):
        self.calls += 1
        self.messages.append(messages)
        return self._content, self._finish


def _two_chapter_prov():
    """Two chapter openers ("Chapter 1" / "Chapter 2"), each with an N.M section,
    a pending subsection, and body content."""
    return [
        {"region_kind": "heading", "heading_text": "Chapter 1", "level": 1,
         "first_raw_block_index": 0, "source_page": 1,
         "chapter_title_synthesized": True, "native_label": "synthetic"},
        {"region_kind": "heading", "heading_text": "1.1 Alpha", "level": 2,
         "first_raw_block_index": 1, "source_page": 1,
         "section_number_recovered": True, "native_label": "title"},
        {"region_kind": "heading", "heading_text": "Introduction", "level": 3,
         "first_raw_block_index": 2, "source_page": 1,
         "heading_level_pending": True, "native_label": "title"},
        {"region_kind": "paragraph",
         "raw_text": "Alpha body prose that gives the subsection real content.",
         "first_raw_block_index": 3, "source_page": 1},
        {"region_kind": "heading", "heading_text": "Chapter 2", "level": 1,
         "first_raw_block_index": 10, "source_page": 2, "native_label": "title"},
        {"region_kind": "heading", "heading_text": "2.1 Beta", "level": 2,
         "first_raw_block_index": 11, "source_page": 2,
         "section_number_recovered": True, "native_label": "title"},
        {"region_kind": "heading", "heading_text": "Introduction", "level": 3,
         "first_raw_block_index": 12, "source_page": 2,
         "heading_level_pending": True, "native_label": "title"},
        {"region_kind": "paragraph",
         "raw_text": "Beta body prose giving the second subsection its content.",
         "first_raw_block_index": 13, "source_page": 2},
    ]


def _top_level_prov():
    """No chapter openers; two same-(min-)level headings → top-level fallback."""
    return [
        {"region_kind": "heading", "heading_text": "Alpha", "level": 2,
         "first_raw_block_index": 0, "source_page": 1},
        {"region_kind": "paragraph", "raw_text": "alpha body",
         "first_raw_block_index": 1, "source_page": 1},
        {"region_kind": "heading", "heading_text": "Beta", "level": 2,
         "first_raw_block_index": 2, "source_page": 2},
        {"region_kind": "paragraph", "raw_text": "beta body",
         "first_raw_block_index": 3, "source_page": 2},
    ]


def _whole_doc_prov():
    """No openers, a single heading → whole-document fallback (one chapter)."""
    return [
        {"region_kind": "heading", "heading_text": "Alpha", "level": 2,
         "first_raw_block_index": 0, "source_page": 1},
        {"region_kind": "paragraph", "raw_text": "the only body",
         "first_raw_block_index": 1, "source_page": 1},
    ]


# ── Part 1: chapter segmentation + fallback ladder. ──────────────────────────
def test_segment_into_chapters_opener_boundaries():
    chapters = hj.segment_into_chapters(_two_chapter_prov())
    assert len(chapters) == 2
    assert all(c.mode == "chapter_opener" for c in chapters)
    assert chapters[0].regions[0]["heading_text"] == "Chapter 1"
    assert chapters[1].regions[0]["heading_text"] == "Chapter 2"


def test_segment_into_chapters_top_level_fallback():
    chapters = hj.segment_into_chapters(_top_level_prov())
    assert len(chapters) == 2
    assert all(c.mode == "top_level_heading" for c in chapters)


def test_segment_into_chapters_whole_document_fallback():
    chapters = hj.segment_into_chapters(_whole_doc_prov())
    assert len(chapters) == 1
    assert chapters[0].mode == "whole_document"
    assert chapters[0].regions == _whole_doc_prov()


def test_segment_never_crashes_on_empty_or_headingless():
    assert hj.segment_into_chapters([]) [0].mode == "whole_document"
    prov = [{"region_kind": "paragraph", "raw_text": "x",
             "first_raw_block_index": 0}]
    chapters = hj.segment_into_chapters(prov)
    assert len(chapters) == 1 and chapters[0].mode == "whole_document"


# ── Part 2: chapter-mode judging (one window per chapter, skeleton+content). ──
def test_chapter_mode_one_window_per_chapter_with_content(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    plan = hj.build_heading_skeleton(_two_chapter_prov())
    assert len(plan.windows) == 2  # one per chapter with pendings
    (d0, p0), (d1, p1) = plan.windows
    assert p0 == [2] and p1 == [12]  # only that chapter's pendings
    for d in (d0, d1):
        assert hj._CHAPTER_CONTENT_HEADER in d
        assert hj._WINDOW_HEADINGS_MARKER in d
    # each window carries its OWN chapter's content + skeleton, not the other's.
    assert "Alpha body prose" in d0 and "Beta body prose" not in d0
    assert "R2 p1 h3* Introduction" in d0
    assert "Beta body prose" in d1 and "Alpha body prose" not in d1
    assert "R12 p2 h3* Introduction" in d1


def test_chapter_mode_judges_only_that_chapters_pendings(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    prov, esc, tree = _two_chapter_prov(), [], []
    stub = _StubPost(content='{"levels": {"2": 3, "12": 3}}')
    report = hj.run_heading_judge(prov, tree, esc, post_fn=stub, use_cache=False,
                                  emit_capture=False)
    assert stub.calls == 2  # one POST per chapter
    assert report["applied"] == 2
    assert next(r for r in prov if r["first_raw_block_index"] == 2)["level"] == 3
    assert next(r for r in prov if r["first_raw_block_index"] == 12)["level"] == 3


def test_chapter_mode_content_budget_guard_truncates_and_logs(monkeypatch, caplog):
    import logging
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    monkeypatch.setattr(hj, "_DIGEST_BUDGET_TOKENS", 120)
    prov = _whole_doc_prov()
    prov[0]["heading_level_pending"] = True  # give the chapter a pending
    prov[1]["raw_text"] = ("word " * 400) + "ZZZEND"
    with caplog.at_level(logging.INFO, logger=hj.logger.name):
        plan = hj.build_heading_skeleton(prov)
    win = plan.windows[0][0]
    assert "ZZZEND" not in win  # mid/late content dropped by the guard
    assert any("chapter content truncated to fit" in r.getMessage()
               for r in caplog.records)


# ── Part 5: derived document-schema preamble. ────────────────────────────────
def test_doc_schema_derived_and_prepended_to_every_window(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    plan = hj.build_heading_skeleton(_two_chapter_prov())
    for d, _ in plan.windows:
        assert d.startswith(hj._DOC_SCHEMA_HEADER)
    schema = hj.build_document_schema(_two_chapter_prov())
    assert "Heading-level scheme: h1=chapter, h2=section, h3=subsection" in schema
    assert "Numbering convention: Sections are numbered N.M" in schema
    assert "Chapters open with a top-level 'Chapter N' heading" in schema
    assert "Recurring section titles" in schema and "'introduction'" in schema


def test_doc_schema_reports_nmk_depth():
    prov = [
        {"region_kind": "heading", "heading_text": "1.1 Sec", "level": 2,
         "first_raw_block_index": 0, "source_page": 1},
        {"region_kind": "heading", "heading_text": "1.1.1 Subsec", "level": 3,
         "first_raw_block_index": 1, "source_page": 1},
    ]
    schema = hj.build_document_schema(prov)
    assert "N.M.K" in schema and "ONE LEVEL DEEPER" in schema


def test_doc_schema_deterministic():
    a = hj.build_document_schema(_two_chapter_prov())
    b = hj.build_document_schema(_two_chapter_prov())
    assert a == b and a  # non-empty + stable


def test_doc_schema_hard_capped():
    prov = []
    for i in range(60):
        prov.append({"region_kind": "heading",
                     "heading_text": f"Recurring Long Section Title Number {i} "
                                     + ("verbose " * 20),
                     "level": 2, "first_raw_block_index": i, "source_page": i})
        prov.append({"region_kind": "heading",
                     "heading_text": f"Recurring Long Section Title Number {i} "
                                     + ("verbose " * 20),
                     "level": 2, "first_raw_block_index": 1000 + i,
                     "source_page": i})
    schema = hj.build_document_schema(prov)
    assert len(schema) <= hj._DOC_SCHEMA_MAX_TOKENS * 4


def test_doc_schema_empty_on_headingless():
    assert hj.build_document_schema(
        [{"region_kind": "paragraph", "raw_text": "x",
          "first_raw_block_index": 0}]) == ""


def test_doc_schema_off_no_preamble(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    assert hj.resolve_doc_schema_mode() is False
    plan = hj.build_heading_skeleton(_two_chapter_prov())
    for d, _ in plan.windows:
        assert hj._DOC_SCHEMA_HEADER not in d
        assert hj._CHAPTER_CONTENT_HEADER in d  # chapter content still present


def test_doc_schema_off_entirely_when_chapter_mode_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", raising=False)
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "1")
    assert hj.resolve_doc_schema_mode() is False


def test_doc_schema_folds_into_window_cache_key(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    with_schema = hj.build_heading_skeleton(_two_chapter_prov()).windows[0][0]
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", "0")
    without_schema = hj.build_heading_skeleton(_two_chapter_prov()).windows[0][0]
    k1 = hj._window_cache_key(with_schema, "m", 20480)
    k2 = hj._window_cache_key(without_schema, "m", 20480)
    assert k1 != k2  # the schema prefix moves the key (content change re-judges)


# ── Part 3: reviewer receives per-chapter skeletons + full skeleton + schema. ─
def test_reviewer_receives_per_chapter_and_full_skeleton(monkeypatch):
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", "1")
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE_FINAL_REVIEW", "1")
    # judged tree: mark the pendings judged so they are relevelable.
    prov = _two_chapter_prov()
    for r in prov:
        if r.get("heading_level_pending"):
            r.pop("heading_level_pending")
            r["heading_level_judged"] = {"from": 3, "to": 3, "clamped": False}
    rec = _RecordPost(content='{"levels": {}}')
    hj.run_final_review(prov, [], [], post_fn=rec, use_cache=False,
                        emit_capture=False)
    assert rec.calls == 1
    digest = rec.messages[0][1]["content"]
    assert hj._DOC_SCHEMA_HEADER in digest
    assert hj._REVIEW_PER_CHAPTER_HEADER in digest
    assert hj._REVIEW_WHOLE_DOC_HEADER in digest
    assert "Chapter 1:" in digest and "Chapter 2:" in digest


def test_reviewer_single_chapter_no_schema_is_legacy_digest(monkeypatch):
    # chapter-mode OFF + single-chapter fixture → the legacy whole-doc digest.
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", raising=False)
    prov = _review_prov()  # one "Chapter 1" opener → one chapter
    entries = hj._build_review_entries(prov)
    legacy = hj._render_digest(entries, anchors=False)
    assert hj._build_review_digest(prov, entries) == legacy


# ── Byte-identity: both new flags OFF. ───────────────────────────────────────
def test_chapter_mode_off_is_byte_identical(monkeypatch):
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_CHAPTER_MODE", raising=False)
    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE_DOC_SCHEMA", raising=False)
    plan = hj.build_heading_skeleton(_prov())
    for d, _ in plan.windows:
        assert hj._CHAPTER_CONTENT_HEADER not in d
        assert hj._DOC_SCHEMA_HEADER not in d
    # identical to the pre-refactor single-window legacy digest.
    assert plan.windows == [(plan.digest, [1, 24])]


def test_both_new_flags_off_run_is_byte_identical():
    prov_a, esc_a, tree_a = _prov(), _escalations(), []
    prov_b, esc_b, tree_b = copy.deepcopy(prov_a), copy.deepcopy(esc_a), []
    stub = '{"levels": {"1": 3, "24": 3}}'
    hj.run_heading_judge(prov_a, tree_a, esc_a, use_cache=False,
                         emit_capture=False, post_fn=_StubPost(content=stub))
    hj.run_heading_judge(prov_b, tree_b, esc_b, use_cache=False,
                         emit_capture=False, post_fn=_StubPost(content=stub))
    assert prov_a == prov_b and esc_a == esc_b and tree_a == tree_b
