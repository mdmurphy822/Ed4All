"""Tests for the deterministic prerequisite-DAG TO sequencer."""
from __future__ import annotations

import random

import pytest

from lib.generation.prereq_sequencer import (
    resolve_prereq_sequencing,
    sequence_terminal_objectives,
)


class _FakeCapture:
    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def _to(tid, concepts):
    return {"id": tid, "key_concepts": list(concepts), "statement": tid}


def _graph(*edges):
    """edges: (dependent_concept, prerequisite_concept, confidence)."""
    return {
        "kind": "concept_semantic",
        "edges": [
            {"source": s, "target": t, "type": "prerequisite", "confidence": c}
            for (s, t, c) in edges
        ],
    }


def _ids(tos):
    return [t["id"] for t in tos]


# --------------------------------------------------------------------------- #
# resolve flag parse-with-fallback
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "val,expected",
    [("1", True), ("true", True), ("On", True), ("yes", True),
     ("0", False), ("nope", False), ("", False)],
)
def test_resolve_flag(val, expected):
    assert resolve_prereq_sequencing(val) is expected


# --------------------------------------------------------------------------- #
# Flag-off call still works (the sequencer itself is flag-agnostic; the caller
# gates it). With no edges → identity.
# --------------------------------------------------------------------------- #


def test_no_prereq_edges_noop():
    tos = [_to("TO-01", ["a"]), _to("TO-02", ["b"])]
    out, sig = sequence_terminal_objectives(tos, _graph())
    assert _ids(out) == ["TO-01", "TO-02"]
    assert sig["reordered"] is False
    assert sig["to_edges_used"] == 0


def test_missing_graph_noop():
    tos = [_to("TO-01", ["a"]), _to("TO-02", ["b"])]
    out, sig = sequence_terminal_objectives(tos, None)
    assert _ids(out) == ["TO-01", "TO-02"]
    assert sig["reordered"] is False


def test_empty_key_concepts_noop():
    tos = [_to("TO-01", []), _to("TO-02", [])]
    out, sig = sequence_terminal_objectives(tos, _graph(("b", "a", 0.6)))
    assert _ids(out) == ["TO-01", "TO-02"]
    assert sig["to_edges_used"] == 0


# --------------------------------------------------------------------------- #
# Simple topological reorder
# --------------------------------------------------------------------------- #


def test_topo_reorder_simple():
    # TO-B teaches concept_b whose prerequisite concept_a is owned by TO-A.
    # Input order [B, A] -> output [A, B].
    tos = [_to("TO-B", ["concept_b"]), _to("TO-A", ["concept_a"])]
    g = _graph(("concept_b", "concept_a", 0.6))  # B depends on A
    out, sig = sequence_terminal_objectives(tos, g)
    assert _ids(out) == ["TO-A", "TO-B"]
    assert set(_ids(out)) == {"TO-A", "TO-B"}  # partition invariant
    assert sig["reordered"] is True
    assert sig["n_reordered"] == 2
    assert sig["to_edges_used"] == 1


def test_namespaced_concept_ids_match():
    # Graph endpoints carry a {course}: prefix; key_concepts are bare slugs.
    tos = [_to("TO-B", ["concept_b"]), _to("TO-A", ["concept_a"])]
    g = _graph(("crs:concept_b", "crs:concept_a", 0.6))
    out, _ = sequence_terminal_objectives(tos, g)
    assert _ids(out) == ["TO-A", "TO-B"]


# --------------------------------------------------------------------------- #
# Stable tie-break: independent TOs keep original order
# --------------------------------------------------------------------------- #


def test_stable_tiebreak_independent():
    tos = [_to(f"TO-{i:02d}", [f"c{i}"]) for i in range(1, 6)]
    out, sig = sequence_terminal_objectives(tos, _graph())
    assert _ids(out) == [f"TO-{i:02d}" for i in range(1, 6)]
    assert sig["reordered"] is False


def test_partial_order_preserves_independent_tail():
    # C depends on A; B and D independent → A before C, others keep order.
    tos = [_to("TO-A", ["a"]), _to("TO-B", ["b"]),
           _to("TO-C", ["c"]), _to("TO-D", ["d"])]
    g = _graph(("c", "a", 0.6))  # C depends on A (already satisfied)
    out, _ = sequence_terminal_objectives(tos, g)
    # A already precedes C; order unchanged.
    assert _ids(out) == ["TO-A", "TO-B", "TO-C", "TO-D"]
    # Now reverse: A depends on C.
    g2 = _graph(("a", "c", 0.6))
    out2, sig2 = sequence_terminal_objectives(tos, g2)
    assert out2[_ids(out2).index("TO-C")]["id"] == "TO-C"
    assert _ids(out2).index("TO-C") < _ids(out2).index("TO-A")
    assert set(_ids(out2)) == {"TO-A", "TO-B", "TO-C", "TO-D"}


# --------------------------------------------------------------------------- #
# Cycle break: deterministic, lowest-confidence-first, no TO lost
# --------------------------------------------------------------------------- #


def test_cycle_break_deterministic():
    # A<->B reciprocal prereqs (a 2-cycle). The lower-confidence edge is dropped.
    tos = [_to("TO-A", ["a"]), _to("TO-B", ["b"])]
    g = _graph(
        ("a", "b", 0.6),  # A depends on B (B prereq of A)
        ("b", "a", 0.3),  # B depends on A (A prereq of B) — weaker
    )
    out, sig = sequence_terminal_objectives(tos, g)
    assert set(_ids(out)) == {"TO-A", "TO-B"}  # no TO lost
    assert sig["cycles_broken"] == 1
    # Deterministic across runs.
    out2, sig2 = sequence_terminal_objectives(tos, g)
    assert _ids(out) == _ids(out2)
    assert sig2["cycles_broken"] == 1


# --------------------------------------------------------------------------- #
# Anti-fabrication: edge endpoints mapping to no TO are ignored
# --------------------------------------------------------------------------- #


def test_anti_fabrication_unmapped_edge_ignored():
    tos = [_to("TO-01", ["a"]), _to("TO-02", ["b"])]
    # Edge references a concept owned by no TO.
    g = _graph(("ghost", "a", 0.6))
    out, sig = sequence_terminal_objectives(tos, g)
    assert _ids(out) == ["TO-01", "TO-02"]
    assert sig["to_edges_used"] == 0
    assert sig["ignored_unmapped_edges"] == 1


# --------------------------------------------------------------------------- #
# Property: partition invariant on random graphs
# --------------------------------------------------------------------------- #


def test_partition_invariant_property():
    rng = random.Random(1234)
    for _ in range(50):
        n = rng.randint(2, 8)
        tos = [_to(f"TO-{i:02d}", [f"c{i}"]) for i in range(n)]
        edges = []
        for _ in range(rng.randint(0, n * 2)):
            i, j = rng.randint(0, n - 1), rng.randint(0, n - 1)
            if i != j:
                edges.append((f"c{i}", f"c{j}", round(rng.random(), 3)))
        out, sig = sequence_terminal_objectives(tos, _graph(*edges))
        assert sorted(_ids(out)) == sorted(_ids(tos))  # multiset equal
        assert len(out) == n


# --------------------------------------------------------------------------- #
# Decision capture fires once with dynamic rationale
# --------------------------------------------------------------------------- #


def test_decision_capture_fires_once():
    cap = _FakeCapture()
    tos = [_to("TO-B", ["concept_b"]), _to("TO-A", ["concept_a"])]
    g = _graph(("concept_b", "concept_a", 0.6))
    sequence_terminal_objectives(tos, g, capture=cap, course_code="C1")
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["decision_type"] == "content_selection"
    assert len(call["rationale"]) >= 20
    assert "C1" in call["rationale"]
    assert call["ml_features"]["to_edges_used"] == 1
