"""Tests for the prerequisite-sequencing validator."""
from __future__ import annotations

import pytest

from lib.validators.prereq_sequencing import PrereqSequencingValidator


class _FakeCapture:
    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def _to(tid, concepts):
    return {"id": tid, "key_concepts": list(concepts), "statement": tid}


def _graph(*edges):
    return {
        "kind": "concept_semantic",
        "edges": [
            {"source": s, "target": t, "type": "prerequisite", "confidence": c}
            for (s, t, c) in edges
        ],
    }


def _codes(result):
    return {i.code for i in result.issues}


# --------------------------------------------------------------------------- #
# Off → byte-stable no-op
# --------------------------------------------------------------------------- #


def test_validator_off_passes(monkeypatch):
    monkeypatch.delenv("ED4ALL_PREREQ_SEQUENCING", raising=False)
    cap = _FakeCapture()
    result = PrereqSequencingValidator(decision_capture=cap).validate(
        {
            "terminal_objectives": [_to("TO-B", ["b"]), _to("TO-A", ["a"])],
            "concept_graph": _graph(("b", "a", 0.6)),
        }
    )
    assert result.passed is True
    assert _codes(result) == {"PREREQ_SEQUENCING_DISABLED"}
    assert result.metadata["prerequisite_sequencing"] == {"enabled": False}
    assert cap.calls == []


# --------------------------------------------------------------------------- #
# On + violation
# --------------------------------------------------------------------------- #


def test_validator_flags_violation(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_SEQUENCING", "1")
    # Emitted order [B, A] but B depends on A (A should precede B) → violation.
    result = PrereqSequencingValidator().validate(
        {
            "terminal_objectives": [_to("TO-B", ["b"]), _to("TO-A", ["a"])],
            "concept_graph": _graph(("b", "a", 0.6)),
        }
    )
    assert result.passed is True  # warning-day-1
    assert "PREREQ_ORDER_VIOLATION" in _codes(result)
    assert result.metadata["prerequisite_sequencing"]["n_violations"] == 1


def test_validator_no_violation_when_ordered(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_SEQUENCING", "1")
    # Emitted order [A, B]; B depends on A → satisfied, no violation.
    result = PrereqSequencingValidator().validate(
        {
            "terminal_objectives": [_to("TO-A", ["a"]), _to("TO-B", ["b"])],
            "concept_graph": _graph(("b", "a", 0.6)),
        }
    )
    assert "PREREQ_ORDER_VIOLATION" not in _codes(result)
    assert result.metadata["prerequisite_sequencing"]["n_violations"] == 0


# --------------------------------------------------------------------------- #
# Graceful skips
# --------------------------------------------------------------------------- #


def test_validator_no_graph_skip(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_SEQUENCING", "1")
    result = PrereqSequencingValidator().validate(
        {"terminal_objectives": [_to("TO-A", ["a"])]}
    )
    assert result.passed is True
    assert "PREREQ_SEQUENCING_NO_GRAPH" in _codes(result)


def test_validator_no_objectives_skip(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_SEQUENCING", "1")
    result = PrereqSequencingValidator().validate(
        {"concept_graph": _graph(("b", "a", 0.6))}
    )
    assert result.passed is True
    assert "PREREQ_SEQUENCING_NO_OBJECTIVES" in _codes(result)


def test_validator_no_edges_skip(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_SEQUENCING", "1")
    result = PrereqSequencingValidator().validate(
        {
            "terminal_objectives": [_to("TO-A", ["a"]), _to("TO-B", ["b"])],
            "concept_graph": _graph(),  # no edges
        }
    )
    assert result.passed is True
    assert "PREREQ_SEQUENCING_NO_EDGES" in _codes(result)


# --------------------------------------------------------------------------- #
# Anti-fabrication: unmapped edge → no edges → skip
# --------------------------------------------------------------------------- #


def test_validator_anti_fabrication_unmapped(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_SEQUENCING", "1")
    result = PrereqSequencingValidator().validate(
        {
            "terminal_objectives": [_to("TO-A", ["a"]), _to("TO-B", ["b"])],
            "concept_graph": _graph(("ghost", "a", 0.6)),
        }
    )
    assert "PREREQ_SEQUENCING_NO_EDGES" in _codes(result)


# --------------------------------------------------------------------------- #
# Decision capture
# --------------------------------------------------------------------------- #


def test_validator_decision_capture_fires(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_SEQUENCING", "1")
    cap = _FakeCapture()
    PrereqSequencingValidator(decision_capture=cap).validate(
        {
            "terminal_objectives": [_to("TO-B", ["b"]), _to("TO-A", ["a"])],
            "concept_graph": _graph(("b", "a", 0.6)),
        }
    )
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["decision_type"] == "validation_result"
    assert len(call["rationale"]) >= 20
    assert call["ml_features"]["n_violations"] == 1


# --------------------------------------------------------------------------- #
# Input builder threads the POST-sequencing TO order (emitted order) so the
# gate audits residual violations, not the pre-reorder planning artifact.
# --------------------------------------------------------------------------- #


def test_builder_prefers_sequenced_sidecar(tmp_path):
    import json as _json

    from MCP.hardening.gate_input_routing import _build_prereq_sequencing

    # The planning artifact on disk carries the PRE-reorder (Ward) order [B, A]
    # (which would FLAG a violation against edge b->a), while the post-sequencing
    # sidecar carries the EMITTED, fixed order [A, B].
    synthesized = tmp_path / "synthesized_objectives.json"
    synthesized.write_text(
        _json.dumps(
            {"terminal_objectives": [_to("TO-B", ["b"]), _to("TO-A", ["a"])]}
        ),
        encoding="utf-8",
    )
    (tmp_path / "synthesized_objectives.sequenced.json").write_text(
        _json.dumps(
            {"terminal_objectives": [_to("TO-A", ["a"]), _to("TO-B", ["b"])]}
        ),
        encoding="utf-8",
    )

    inputs, _ = _build_prereq_sequencing(
        {}, {"objectives_path": str(synthesized)}
    )
    # Sidecar (emitted) order preferred → validator sees the fixed order.
    assert [t["id"] for t in inputs["terminal_objectives"]] == ["TO-A", "TO-B"]
    assert inputs["synthesized_objectives_path"] == str(synthesized)


def test_builder_without_sidecar_uses_path_only(tmp_path):
    import json as _json

    from MCP.hardening.gate_input_routing import _build_prereq_sequencing

    synthesized = tmp_path / "synthesized_objectives.json"
    synthesized.write_text(
        _json.dumps(
            {"terminal_objectives": [_to("TO-B", ["b"]), _to("TO-A", ["a"])]}
        ),
        encoding="utf-8",
    )
    inputs, _ = _build_prereq_sequencing(
        {}, {"objectives_path": str(synthesized)}
    )
    # No sidecar → builder threads only the path (validator reads it).
    assert "terminal_objectives" not in inputs
    assert inputs["synthesized_objectives_path"] == str(synthesized)
