"""misconception_rich — resolver + grounder + planner-pass tests."""

from __future__ import annotations

import pytest

from lib.generation.misconception_rich import (
    apply_misconception_grounding,
    compose_predict_prompt,
    compose_reconcile,
    ground_named_misconception,
    resolve_misconception_rich,
)


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "On"])
def test_resolver_truthy(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_MISCONCEPTION_RICH", val)
    assert resolve_misconception_rich() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "garbage", ""])
def test_resolver_falsey_and_garbage(monkeypatch, val):
    monkeypatch.setenv("ED4ALL_MISCONCEPTION_RICH", val)
    assert resolve_misconception_rich() is False


def test_resolver_unset(monkeypatch):
    monkeypatch.delenv("ED4ALL_MISCONCEPTION_RICH", raising=False)
    assert resolve_misconception_rich() is False


def test_grounder_returns_only_present_tag():
    descriptor = {"misconception": "Students think like terms cannot be combined."}
    tags = ["like_terms", "coefficient"]
    named = ground_named_misconception(descriptor, tags)
    # "like terms" is named in the descriptor -> the salient tag.
    assert named == "like_terms"


def test_grounder_anti_fabrication_none_when_unmatched():
    # Empty tag set -> None (never invents).
    assert ground_named_misconception({"misconception": "x"}, []) is None
    # A tag set with no descriptor mention still returns a member (first), never
    # a value absent from the input.
    named = ground_named_misconception({"misconception": "unrelated"}, ["foo", "bar"])
    assert named in {"foo", "bar"}


def test_grounder_never_returns_value_absent_from_input():
    tags = ["alpha", "beta"]
    named = ground_named_misconception({"misconception": "alpha beta gamma"}, tags)
    assert named in tags


def test_grounder_domain_vocab_gate_rejects_offdomain_tag():
    # tag "zeta" is not in the domain vocab -> rejected; "alpha" is kept.
    descriptor = {"misconception": "alpha and zeta"}
    named = ground_named_misconception(
        descriptor, ["zeta", "alpha"], domain_vocab_surface_forms=["alpha", "beta"]
    )
    assert named == "alpha"


def test_compose_seams_none_without_concept():
    assert compose_predict_prompt(None) is None
    assert compose_reconcile(None) is None
    assert compose_predict_prompt("like_terms")
    assert compose_reconcile("like_terms")
    assert "like terms" in compose_predict_prompt("like_terms")


def test_apply_grounding_stamps_named_and_leaves_generic():
    selected = [
        {"block_type": "misconception", "target_co_ids": ["CO-01"]},
        {"block_type": "misconception", "target_co_ids": ["CO-02"]},
        {"block_type": "concept", "target_co_ids": ["CO-01"]},
    ]
    tags_by_co = {"CO-01": ["like_terms"], "CO-02": []}
    counts = apply_misconception_grounding(
        selected, concept_tags_by_co=tags_by_co, enabled=True
    )
    assert selected[0]["mc_named_concept"] == "like_terms"
    assert selected[0]["mc_predict_prompt"]
    assert selected[0]["mc_reconcile"]
    assert "mc_named_concept" not in selected[1]  # no tags -> generic
    assert "mc_named_concept" not in selected[2]  # not a misconception block
    assert counts == {"seen": 2, "named": 1, "generic": 1}


def test_apply_grounding_off_is_identity():
    selected = [{"block_type": "misconception", "target_co_ids": ["CO-01"]}]
    counts = apply_misconception_grounding(
        selected, concept_tags_by_co={"CO-01": ["x"]}, enabled=False
    )
    assert counts == {"seen": 0, "named": 0, "generic": 0}
    assert "mc_named_concept" not in selected[0]


class _FakeCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


def test_planner_pass_emits_content_selection_capture(monkeypatch):
    monkeypatch.setenv("ED4ALL_MISCONCEPTION_RICH", "1")
    from lib.generation.block_planner import _apply_post_floor_field_stamps

    cap = _FakeCapture()
    selected = [{"block_type": "misconception", "target_co_ids": ["CO-01"]}]
    _apply_post_floor_field_stamps(
        selected=selected,
        chapter_objectives=[{"id": "CO-01", "concept_tags": ["like_terms"]}],
        capture=cap,
        course_code="TEST_101",
        to_id="TO-01",
    )
    mc_events = [
        e for e in cap.events
        if e.get("decision_type") == "content_selection"
        and "misconception" in str(e.get("decision", ""))
    ]
    assert len(mc_events) == 1
    assert len(mc_events[0]["rationale"]) >= 20
    assert selected[0]["mc_named_concept"] == "like_terms"
