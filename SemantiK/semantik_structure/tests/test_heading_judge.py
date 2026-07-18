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
    assert stub.calls == 2  # one initial + one length retry
    assert stub.last_max_tokens > hj._max_tokens_for(len(prov))  # doubled
    assert report["applied"] == 0  # second length → fail-open


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
    # next call with a GOOD stub must actually POST (nothing was cached)
    good = _StubPost(content='{"levels": {"1": 4}}')
    m, _ = hj.judge_heading_levels(plan, post_fn=good, use_cache=True, model="m")
    assert good.calls == 1
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
                         course_code="PHYS_101")
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


def test_capture_failure_never_breaks_judge(monkeypatch):
    import lib.decision_capture as dc

    def _boom(**kwargs):
        raise RuntimeError("capture backend down")

    monkeypatch.setattr(dc, "DecisionCapture", _boom)
    prov, esc, tree = _prov(), _escalations(), []
    stub = _StubPost(content='{"levels": {"1": 3, "24": 3}}')
    report = hj.run_heading_judge(prov, tree, esc, post_fn=stub, use_cache=False)
    assert report["applied"] == 2  # judge still applied despite capture failure


# ── Flag-off lane byte-identity. ─────────────────────────────────────────────
def test_lane_flag_off_never_imports_judge(monkeypatch):
    from semantik_structure.glmocr import resolve_heading_judge_mode

    monkeypatch.delenv("SEMANTIK_HEADING_JUDGE", raising=False)
    assert resolve_heading_judge_mode() is False


def test_lane_flag_on_resolves(monkeypatch):
    from semantik_structure.glmocr import resolve_heading_judge_mode

    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "1")
    assert resolve_heading_judge_mode() is True
    monkeypatch.setenv("SEMANTIK_HEADING_JUDGE", "garbage")
    assert resolve_heading_judge_mode() is False
