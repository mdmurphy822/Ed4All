"""W2 Defect B — tests for ObjectiveSpecificityValidator.

Deterministic, embedding-FREE checks over CO statements. Covers: the default-OFF
skip-with-pass, a specific CO passing clean, each of V1 (vacuous) / V2 (generic
object) / V3 (unanchored) firing, the missing-chunkset NO_CHUNK_UNIVERSE degrade,
the vacuous-rate headline, decision-capture, the router registration, and an
integration pass through the shared coverage builder on a temp fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lib.validators.objective_specificity import ObjectiveSpecificityValidator


class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def _codes(result) -> List[str]:
    return [i.code for i in result.issues]


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_OBJECTIVE_SPECIFICITY", "1")


def _co(co_id, statement, chunk_ids=("c1",)):
    return {
        "id": co_id,
        "statement": statement,
        "source_chunk_ids": list(chunk_ids),
    }


def _doc(cos):
    return {"chapter_objectives": cos}


_CHUNKS = {
    "c1": {
        "id": "c1",
        "text": (
            "To factor a quadratic trinomial into two binomials, find two "
            "numbers whose product is the constant term and whose sum is the "
            "middle coefficient."
        ),
    },
    "c2": {
        "id": "c2",
        "text": "Photosynthesis converts sunlight into chemical energy in plant cells.",
    },
}


# --------------------------------------------------------------------- #


def test_disabled_flag_is_noop(monkeypatch):
    monkeypatch.delenv("ED4ALL_OBJECTIVE_SPECIFICITY", raising=False)
    result = ObjectiveSpecificityValidator().validate(
        {"synthesized_objectives": _doc([_co("CO-01", "anything at all")])}
    )
    assert result.passed is True
    assert result.score == 1.0
    assert _codes(result) == ["OBJECTIVE_SPECIFICITY_DISABLED"]


def test_specific_objective_passes_clean():
    doc = _doc([_co("CO-01", "Factor a quadratic trinomial into two binomials.", ["c1"])])
    result = ObjectiveSpecificityValidator().validate(
        {"synthesized_objectives": doc, "chunks_by_id": _CHUNKS}
    )
    assert result.passed is True
    assert result.score == 1.0
    assert _codes(result) == []


def test_v1_vacuous_objective_flagged():
    doc = _doc([_co("CO-01", "Apply various techniques to solve problems.", ["c1"])])
    result = ObjectiveSpecificityValidator().validate(
        {"synthesized_objectives": doc, "chunks_by_id": _CHUNKS}
    )
    codes = _codes(result)
    assert "OBJECTIVE_VACUOUS" in codes
    # Single CO vacuous → 100% > 5% floor → headline fires.
    assert "OBJECTIVE_VACUOUS_RATE_HIGH" in codes
    assert result.passed is True  # warning day-1


def test_v2_generic_object_flagged():
    # residual {cellular, respiration} = 2 (>= min 2 → not vacuous) but < 4 AND
    # hits the "various concepts" vague-object phrase.
    doc = _doc([_co("CO-01", "Analyze various concepts of cellular respiration.", ["c2"])])
    result = ObjectiveSpecificityValidator().validate(
        {"synthesized_objectives": doc, "chunks_by_id": _CHUNKS}
    )
    codes = _codes(result)
    assert "OBJECTIVE_GENERIC_OBJECT" in codes
    assert "OBJECTIVE_VACUOUS" not in codes


def test_v3_unanchored_statement_flagged():
    # A concrete statement whose content words are absent from its cited chunk
    # (cites c2 = photosynthesis, but the statement is about factoring).
    doc = _doc([_co("CO-01", "Factor a quadratic trinomial into two binomials.", ["c2"])])
    result = ObjectiveSpecificityValidator().validate(
        {"synthesized_objectives": doc, "chunks_by_id": _CHUNKS}
    )
    codes = _codes(result)
    assert "OBJECTIVE_UNANCHORED_STATEMENT" in codes


def test_no_chunk_universe_degrades_with_pass():
    doc = _doc([_co("CO-01", "Factor a quadratic trinomial into two binomials.", ["c1"])])
    result = ObjectiveSpecificityValidator().validate({"synthesized_objectives": doc})
    codes = _codes(result)
    assert "NO_CHUNK_UNIVERSE" in codes
    # V3 skipped (no chunks); V1/V2 still ran → this concrete CO is not flagged.
    assert "OBJECTIVE_UNANCHORED_STATEMENT" not in codes
    assert result.passed is True


def test_decision_capture_emitted():
    cap = _RecordingCapture()
    doc = _doc([_co("CO-01", "Apply various techniques to concepts.", ["c1"])])
    ObjectiveSpecificityValidator().validate(
        {"synthesized_objectives": doc, "chunks_by_id": _CHUNKS, "decision_capture": cap}
    )
    assert cap.events, "expected a validation_result decision"
    ev = cap.events[0]
    assert ev["decision_type"] == "validation_result"
    assert len(ev["rationale"]) >= 20


def test_config_thresholds_from_inputs():
    # Raise the vacuous-rate floor above 1.0 → headline never fires even when
    # every CO is vacuous (proves the config key is honored).
    doc = _doc([_co("CO-01", "Apply various techniques generally.", ["c1"])])
    result = ObjectiveSpecificityValidator().validate(
        {"synthesized_objectives": doc, "chunks_by_id": _CHUNKS, "max_vacuous_rate": 1.0}
    )
    assert "OBJECTIVE_VACUOUS_RATE_HIGH" not in _codes(result)


# --------------------------------------------------------------------- #
# Router registration + builder integration
# --------------------------------------------------------------------- #


def test_router_resolves_objective_specificity_to_coverage_builder():
    from MCP.hardening.gate_input_routing import (
        _build_chapter_objective_coverage_inputs,
        default_router,
    )

    r = default_router()
    dotted = "lib.validators.objective_specificity.ObjectiveSpecificityValidator"
    assert dotted in r.builders, (
        f"W2 Defect B regression: no builder registered for {dotted}; gate would "
        f"silently skip via __no_builder_registered__."
    )
    assert r.builders[dotted] is _build_chapter_objective_coverage_inputs


def test_integration_through_builder_on_temp_fixture(tmp_path):
    from MCP.hardening.gate_input_routing import (
        _build_chapter_objective_coverage_inputs,
    )

    objectives_path = tmp_path / "synthesized_objectives.json"
    objectives_path.write_text(
        json.dumps(
            _doc([
                _co("CO-01", "Apply various techniques to concepts.", ["k1"]),
                _co("CO-02", "Factor a quadratic trinomial into two binomials.", ["k1"]),
            ])
        ),
        encoding="utf-8",
    )
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        json.dumps({"id": "k1", "text": "Factor a quadratic trinomial into two binomials."})
        + "\n",
        encoding="utf-8",
    )

    phase_outputs = {
        "course_planning": {"synthesized_objectives_path": str(objectives_path)},
        "chunking": {"dart_chunks_path": str(chunks_path)},
    }
    inputs, missing = _build_chapter_objective_coverage_inputs(phase_outputs, {})
    assert missing == []
    assert inputs["synthesized_objectives_path"] == str(objectives_path)
    assert inputs["dart_chunks_path"] == str(chunks_path)

    result = ObjectiveSpecificityValidator().validate(inputs)
    codes = _codes(result)
    # CO-01 is vacuous; CO-02 is specific + anchored.
    assert "OBJECTIVE_VACUOUS" in codes
    assert result.passed is True
