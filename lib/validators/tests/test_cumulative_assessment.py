"""FR-COURSE-03 — tests for CumulativeAssessmentValidator.

Confirms the cumulative-retrieval breadth floor for the B14 final:

  * >= 4 TOs + final spans only 1 TO → ``CUMULATIVE_RETRIEVAL_TOO_NARROW``;
  * >= 4 TOs + final spans 2+ TOs → PASS;
  * < 4 TOs → strict no-op (PASS, CUMULATIVE_BREADTH_NA);
  * CO->TO resolution via ``terminal_id``;
  * missing objectives → graceful skip;
  * decision-capture fires.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib.validators.cumulative_assessment import CumulativeAssessmentValidator


class _SpyCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(
        self, *, decision_type: str, decision: str, rationale: str, **kw: Any
    ) -> None:
        self.events.append(
            {
                "decision_type": decision_type,
                "decision": decision,
                "rationale": rationale,
                **kw,
            }
        )


def _write_objectives(
    tmp_path: Path,
    *,
    n_tos: int,
    co_map: Optional[Dict[str, str]] = None,
) -> str:
    tos = [
        {"id": f"TO-{i:02d}", "statement": f"terminal {i}"}
        for i in range(1, n_tos + 1)
    ]
    cos = []
    for cid, tid in (co_map or {}).items():
        cos.append({"id": cid, "statement": f"chapter {cid}", "terminal_id": tid})
    doc = {"terminal_objectives": tos, "chapter_objectives": cos}
    p = tmp_path / "synthesized_objectives.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return str(p)


def _write_assessments(tmp_path: Path, questions: List[Dict[str, Any]]) -> str:
    p = tmp_path / "assessments.json"
    p.write_text(json.dumps({"questions": questions}), encoding="utf-8")
    return str(p)


def _validate(
    objectives_path: str,
    assessments_path: str,
    capture: Optional[_SpyCapture] = None,
):
    v = CumulativeAssessmentValidator()
    inputs: Dict[str, Any] = {
        "synthesized_objectives_path": objectives_path,
        "assessments_path": assessments_path,
    }
    if capture is not None:
        inputs["decision_capture"] = capture
    return v.validate(inputs)


# ---------------------------------------------------------------------------
# Too narrow → FLAG
# ---------------------------------------------------------------------------


def test_too_narrow_flags(tmp_path):
    obj = _write_objectives(tmp_path, n_tos=5)
    # All graded questions target the same single TO.
    asmt = _write_assessments(
        tmp_path,
        [
            {"id": "q1", "objective_id": "TO-01", "type": "multiple_choice"},
            {"id": "q2", "objective_id": "TO-01", "type": "true_false"},
        ],
    )
    result = _validate(obj, asmt)
    assert result.passed is False
    assert result.action == "regenerate"
    codes = [i.code for i in result.issues]
    assert "CUMULATIVE_RETRIEVAL_TOO_NARROW" in codes
    issue = result.issues[0]
    assert issue.severity == "warning"


# ---------------------------------------------------------------------------
# Broad enough → PASS
# ---------------------------------------------------------------------------


def test_spans_two_tos_passes(tmp_path):
    obj = _write_objectives(tmp_path, n_tos=5)
    asmt = _write_assessments(
        tmp_path,
        [
            {"id": "q1", "objective_id": "TO-01", "type": "multiple_choice"},
            {"id": "q2", "objective_id": "TO-03", "type": "multiple_choice"},
        ],
    )
    result = _validate(obj, asmt)
    assert result.passed is True
    assert result.issues == []


def test_co_resolves_to_parent_to(tmp_path):
    # Two COs mapping to two different TOs → spans 2 TOs → PASS.
    obj = _write_objectives(
        tmp_path, n_tos=4, co_map={"CO-01": "TO-01", "CO-02": "TO-02"}
    )
    asmt = _write_assessments(
        tmp_path,
        [
            {"id": "q1", "objective_id": "CO-01", "type": "multiple_choice"},
            {"id": "q2", "objective_id": "CO-02", "type": "multiple_choice"},
        ],
    )
    result = _validate(obj, asmt)
    assert result.passed is True


# ---------------------------------------------------------------------------
# < 4 TOs → strict no-op
# ---------------------------------------------------------------------------


def test_fewer_than_four_tos_is_noop(tmp_path):
    obj = _write_objectives(tmp_path, n_tos=3)
    # Even a single-TO final must pass when course has < 4 TOs.
    asmt = _write_assessments(
        tmp_path,
        [{"id": "q1", "objective_id": "TO-01", "type": "multiple_choice"}],
    )
    result = _validate(obj, asmt)
    assert result.passed is True
    codes = [i.code for i in result.issues]
    assert codes == ["CUMULATIVE_BREADTH_NA"]
    assert result.issues[0].severity == "info"


# ---------------------------------------------------------------------------
# Missing objectives → graceful skip
# ---------------------------------------------------------------------------


def test_missing_objectives_graceful_skip(tmp_path):
    asmt = _write_assessments(
        tmp_path,
        [{"id": "q1", "objective_id": "TO-01", "type": "multiple_choice"}],
    )
    v = CumulativeAssessmentValidator()
    result = v.validate({"assessments_path": asmt})
    assert result.passed is True
    assert [i.code for i in result.issues] == ["OBJECTIVES_UNAVAILABLE"]


# ---------------------------------------------------------------------------
# Decision capture fires
# ---------------------------------------------------------------------------


def test_decision_capture_fires(tmp_path):
    capture = _SpyCapture()
    obj = _write_objectives(tmp_path, n_tos=5)
    asmt = _write_assessments(
        tmp_path,
        [{"id": "q1", "objective_id": "TO-01", "type": "multiple_choice"}],
    )
    _validate(obj, asmt, capture)
    assert len(capture.events) == 1
    ev = capture.events[0]
    assert ev["decision_type"] == "assessment_quality"
    assert ev["decision"] == "cumulative_breadth_too_narrow"
    assert len(ev["rationale"]) >= 20
    feats = ev["ml_features"]
    assert feats["n_tos"] == 5
    assert feats["n_tos_spanned"] == 1
    assert feats["passed"] is False
