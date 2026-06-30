"""recall_self_check — resolver + planner pass + decision-capture tests."""

from __future__ import annotations

import pytest

from lib.generation.recall_self_check import (
    apply_recall_self_check,
    resolve_recall_format,
    resolve_recall_self_check,
)


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", " On "])
def test_resolver_truthy(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_RECALL_SELF_CHECK", val)
    assert resolve_recall_self_check() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "garbage", ""])
def test_resolver_falsey_and_garbage(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_RECALL_SELF_CHECK", val)
    assert resolve_recall_self_check() is False


def test_resolver_unset(monkeypatch):
    monkeypatch.delenv("ED4ALL_RECALL_SELF_CHECK", raising=False)
    assert resolve_recall_self_check() is False


def test_resolve_recall_format_parse_with_fallback():
    assert resolve_recall_format("free_recall") == "free_recall"
    assert resolve_recall_format("cloze") == "cloze"
    assert resolve_recall_format("bogus") is None
    assert resolve_recall_format(None) is None
    assert resolve_recall_format(123) is None


def _week_plan():
    # An exposition block teaches CO-01, then a later self-check probes CO-01
    # (spaced) at a low Bloom -> cloze; a self-check on an UNtaught CO-99 stays
    # unmarked; a concept block is untouched.
    return [
        {"block_type": "concept", "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "self_check_question", "target_co_ids": ["CO-01"], "target_bloom": "remember"},
        {"block_type": "self_check_question", "target_co_ids": ["CO-99"], "target_bloom": "apply"},
    ]


def test_apply_marks_only_taught_co_checks():
    selected, marked, fmts = apply_recall_self_check(_week_plan(), enabled=True)
    # The CO-01 check is taught earlier -> marked cloze; CO-99 untaught -> None.
    assert selected[1]["recall_format"] == "cloze"
    assert "recall_format" not in selected[2]
    # Other block types untouched.
    assert "recall_format" not in selected[0]
    assert marked == 1
    assert fmts == {"cloze": 1}


def test_apply_partition_invariant_no_new_block_no_co_invented():
    plan = _week_plan()
    n_before = len(plan)
    co_before = [list(b.get("target_co_ids", [])) for b in plan]
    selected, _marked, _ = apply_recall_self_check(plan, enabled=True)
    assert len(selected) == n_before
    assert [list(b.get("target_co_ids", [])) for b in selected] == co_before


def test_apply_free_recall_for_higher_bloom():
    plan = [
        {"block_type": "explanation", "target_co_ids": ["CO-02"], "target_bloom": "apply"},
        {"block_type": "self_check_question", "target_co_ids": ["CO-02"], "target_bloom": "analyze"},
    ]
    selected, marked, fmts = apply_recall_self_check(plan, enabled=True)
    assert selected[1]["recall_format"] == "free_recall"
    assert marked == 1
    assert fmts == {"free_recall": 1}


def test_apply_off_is_identity(monkeypatch):
    monkeypatch.delenv("ED4ALL_RECALL_SELF_CHECK", raising=False)
    plan = _week_plan()
    selected, marked, fmts = apply_recall_self_check(plan)  # enabled resolves False
    assert marked == 0
    assert fmts == {}
    assert all("recall_format" not in b for b in selected)


class _FakeCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


def test_planner_pass_emits_content_selection_capture(monkeypatch):
    # Drive the pass through the planner helper so the DecisionCapture fires.
    monkeypatch.setenv("ED4ALL_RECALL_SELF_CHECK", "1")
    from lib.generation.block_planner import _apply_post_floor_field_stamps

    cap = _FakeCapture()
    plan = _week_plan()
    _apply_post_floor_field_stamps(
        selected=plan,
        chapter_objectives=[{"id": "CO-01"}, {"id": "CO-99"}],
        capture=cap,
        course_code="TEST_101",
        to_id="TO-01",
    )
    recall_events = [
        e for e in cap.events
        if e.get("decision_type") == "content_selection"
        and "recall_self_check" in str(e.get("decision", ""))
    ]
    assert len(recall_events) == 1
    assert len(recall_events[0]["rationale"]) >= 20
