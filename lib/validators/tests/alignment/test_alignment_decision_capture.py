"""IB3.8 — DecisionCapture wiring for the IB3 validators.

Asserts (flag ON) each new/extended validator fires its decision event with a
>=20-char dynamic rationale; (flag OFF) the NEW validators (anchored_rubric /
triangle_completeness) emit NOTHING.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPTS_DIR = _PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.alignment.anchored_rubric import AnchoredRubricValidator  # noqa: E402
from lib.validators.alignment.triangle_completeness import (  # noqa: E402
    TriangleCompletenessValidator,
)
from lib.validators.assessment_objective_alignment import (  # noqa: E402
    AssessmentObjectiveAlignmentValidator,
)
from lib.validators.block_objective_delivery import (  # noqa: E402
    BlockObjectiveDeliveryValidator,
)


class _StubCapture:
    def __init__(self):
        self.calls = []

    def log_decision(self, decision_type, decision, rationale, **kw):
        self.calls.append({"decision_type": decision_type, "rationale": rationale})


def _types(cap):
    return [c["decision_type"] for c in cap.calls]


def test_anchored_rubric_fires(monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    cap = _StubCapture()
    block = Block(block_id="p#scenario_a_0", block_type="scenario", page_id="p",
                  sequence=0, content="Design a system.", objective_ids=("CO-01",))
    AnchoredRubricValidator().validate({
        "blocks": [block],
        "objectives": {"CO-01": {"id": "CO-01", "bloom_level": "create",
                                 "abcd": {"behavior": {"verb": "design"}}}},
        "decision_capture": cap,
    })
    assert "anchored_rubric_check" in _types(cap)
    assert all(len(c["rationale"]) >= 20 for c in cap.calls)


def test_triangle_fires(monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    cap = _StubCapture()
    block = Block(block_id="p#concept_a_0", block_type="concept", page_id="p",
                  sequence=0, content="A concept.", objective_ids=("CO-01",))
    TriangleCompletenessValidator().validate({
        "blocks": [block],
        "objectives": {"CO-01": {"id": "CO-01", "bloom_level": "apply",
                                 "abcd": {"behavior": {"verb": "solve"}}}},
        "decision_capture": cap,
    })
    assert "alignment_triangle_check" in _types(cap)
    assert all(len(c["rationale"]) >= 20 for c in cap.calls)


def test_new_validators_silent_when_flag_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", raising=False)
    block = Block(block_id="p#scenario_o_0", block_type="scenario", page_id="p",
                  sequence=0, content="Design.", objective_ids=("CO-01",))
    objs = {"CO-01": {"id": "CO-01", "bloom_level": "create",
                      "abcd": {"behavior": {"verb": "design"}}}}
    cap_a = _StubCapture()
    AnchoredRubricValidator().validate({"blocks": [block], "objectives": objs,
                                        "decision_capture": cap_a})
    cap_t = _StubCapture()
    TriangleCompletenessValidator().validate({"blocks": [block], "objectives": objs,
                                              "decision_capture": cap_t})
    assert cap_a.calls == []
    assert cap_t.calls == []


def test_block_objective_delivery_verb_triple_in_rationale(monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    cap = _StubCapture()
    objs = {"CO-01": {"id": "CO-01", "bloom_level": "evaluate", "bloom_verb": "critique",
                      "statement": "Critique it", "abcd": {"behavior": {"verb": "critique"}}}}
    block = Block(block_id="p#activity_a_0", block_type="activity", page_id="p",
                  sequence=0, content="List the steps.", objective_ids=("CO-01",))
    BlockObjectiveDeliveryValidator(nli=None).validate({
        "blocks": [block], "objectives": objs,
        "objective_statements": {"CO-01": "Critique it"},
        "decision_capture": cap,
    })
    triple_events = [c for c in cap.calls
                     if c["decision_type"] == "block_objective_delivery_check"
                     and "verb_triple_status" in c["rationale"]]
    assert triple_events
    assert all(len(c["rationale"]) >= 20 for c in triple_events)


def test_evidence_form_decision_fires(tmp_path, monkeypatch):
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    cap = _StubCapture()
    assessments = tmp_path / "a.json"
    chunks = tmp_path / "c.jsonl"
    objectives = tmp_path / "o.json"
    assessments.write_text(json.dumps({"questions": [
        {"question_id": "Q1", "objective_id": "CO-01", "question_type": "multiple_choice"},
    ]}), encoding="utf-8")
    with open(chunks, "w", encoding="utf-8") as f:
        f.write(json.dumps({"id": "c1", "learning_outcome_refs": ["co-01"]}) + "\n")
    objectives.write_text(json.dumps({"learning_outcomes": [
        {"id": "CO-01", "bloom_level": "apply", "statement": "Apply it"},
    ]}), encoding="utf-8")
    AssessmentObjectiveAlignmentValidator().validate({
        "assessments_path": str(assessments),
        "chunks_path": str(chunks),
        "synthesized_objectives_path": str(objectives),
        "decision_capture": cap,
    })
    evidence_events = [c for c in cap.calls
                       if c["decision_type"] == "assessment_objective_alignment_check"
                       and "evidence-form" in c["rationale"]]
    assert evidence_events
    assert all(len(c["rationale"]) >= 20 for c in evidence_events)
