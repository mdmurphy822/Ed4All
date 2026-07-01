"""Tests for the W3.1 content-dependency prerequisite rule.

Deterministic, no-LLM, synthetic in-memory graphs/chunks only.
"""
from __future__ import annotations

import pytest

from lib.generation.prerequisite_from_definition_mention import (
    RULE_NAME,
    infer,
    resolve_prereq_definition_mention,
)


class _FakeCapture:
    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def _chunk(cid, tags, los):
    return {"id": cid, "concept_tags": list(tags), "learning_outcome_refs": list(los)}


def _graph(*nodes):
    """nodes: (node_id, [occurrence_chunk_ids...])."""
    return {
        "kind": "concept_semantic",
        "nodes": [{"id": nid, "occurrences": list(occ)} for nid, occ in nodes],
        "edges": [],
    }


# --------------------------------------------------------------------------- #
# Flag resolution (parse-with-fallback)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "val,expected",
    [("1", True), ("true", True), ("On", True), ("yes", True),
     ("0", False), ("nope", False), ("", False), (None, False)],
)
def test_resolve_flag(val, expected):
    assert resolve_prereq_definition_mention(val) is expected


# --------------------------------------------------------------------------- #
# Default OFF → empty (byte-identical guarantee)
# --------------------------------------------------------------------------- #


def test_flag_off_returns_empty():
    chunks = [
        _chunk("c1", ["mitosis"], ["TO-01"]),
        _chunk("c2", ["mitosis"], ["TO-02"]),
    ]
    g = _graph(("mitosis", ["c1", "c2"]))
    # enabled defaults to env (unset in test env) → off.
    assert infer(chunks, None, g) == []


# --------------------------------------------------------------------------- #
# Core signal: concept defined in TO-01, assumed in TO-02 → TO-02 -> TO-01
# --------------------------------------------------------------------------- #


def test_definition_mention_emits_federation_edge():
    chunks = [
        _chunk("c1", ["mitosis"], ["TO-01"]),  # definition chunk (occurrences[0])
        _chunk("c2", ["mitosis"], ["TO-02"]),  # assumed (used, not defined)
    ]
    g = _graph(("mitosis", ["c1", "c2"]))
    edges = infer(chunks, None, g, enabled="1")
    assert len(edges) == 1
    e = edges[0]
    assert e["source"] == "TO-02"  # dependent
    assert e["target"] == "TO-01"  # prerequisite
    assert e["type"] == "prerequisite"
    assert e["provenance"]["rule"] == RULE_NAME
    ev = e["provenance"]["evidence"]
    assert ev["bridge_concept"] == "mitosis"
    assert ev["definition_chunk"] == "c1"
    assert ev["defining_lo"] == "TO-01"
    assert ev["assuming_lo"] == "TO-02"


def test_lowercase_refs_normalized_to_canonical_to():
    # Chunks store refs lowercased; edges emit canonical uppercase TO ids.
    chunks = [
        _chunk("c1", ["mitosis"], ["to-01"]),
        _chunk("c2", ["mitosis"], ["to-02"]),
    ]
    g = _graph(("mitosis", ["c1", "c2"]))
    edges = infer(chunks, None, g, enabled="1")
    assert [(e["source"], e["target"]) for e in edges] == [("TO-02", "TO-01")]


def test_terminal_objectives_pins_canonical_id_form():
    tos = [{"id": "TO-01"}, {"id": "TO-02"}]
    chunks = [
        _chunk("c1", ["mitosis"], ["to-01"]),
        _chunk("c2", ["mitosis"], ["to-02"]),
    ]
    g = _graph(("mitosis", ["c1", "c2"]))
    edges = infer(chunks, None, g, terminal_objectives=tos, enabled="1")
    assert edges[0]["target"] == "TO-01"
    assert edges[0]["source"] == "TO-02"


# --------------------------------------------------------------------------- #
# Anti-fabrication
# --------------------------------------------------------------------------- #


def test_no_edge_when_definition_chunk_has_no_terminal_owner():
    # Definition chunk carries a chapter-objective (CO) ref only → no TO owns
    # the definition → skip (never guessed).
    chunks = [
        _chunk("c1", ["mitosis"], ["CO-05"]),
        _chunk("c2", ["mitosis"], ["TO-02"]),
    ]
    g = _graph(("mitosis", ["c1", "c2"]))
    assert infer(chunks, None, g, enabled="1") == []


def test_same_to_defines_and_uses_no_edge():
    # Concept defined AND used within TO-01 only → no cross-TO dependency.
    chunks = [
        _chunk("c1", ["mitosis"], ["TO-01"]),
        _chunk("c2", ["mitosis"], ["TO-01"]),
    ]
    g = _graph(("mitosis", ["c1", "c2"]))
    assert infer(chunks, None, g, enabled="1") == []


def test_concept_without_occurrences_ignored():
    chunks = [_chunk("c1", ["ghost"], ["TO-01"]), _chunk("c2", ["ghost"], ["TO-02"])]
    g = {"kind": "concept_semantic", "nodes": [{"id": "ghost"}], "edges": []}
    assert infer(chunks, None, g, enabled="1") == []


def test_definition_only_when_used_in_defining_to_no_edge():
    # mitosis defined in TO-01 (c1) and only ever mentioned in TO-01.
    chunks = [
        _chunk("c1", ["mitosis"], ["TO-01"]),
        _chunk("c3", ["meiosis"], ["TO-02"]),
    ]
    g = _graph(("mitosis", ["c1"]), ("meiosis", ["c3"]))
    assert infer(chunks, None, g, enabled="1") == []


# --------------------------------------------------------------------------- #
# Determinism + dedup
# --------------------------------------------------------------------------- #


def test_multiple_concepts_same_pair_deduped():
    chunks = [
        _chunk("c1", ["mitosis", "chromosome"], ["TO-01"]),
        _chunk("c2", ["mitosis", "chromosome"], ["TO-02"]),
    ]
    g = _graph(("mitosis", ["c1", "c2"]), ("chromosome", ["c1", "c2"]))
    edges = infer(chunks, None, g, enabled="1")
    assert len(edges) == 1  # one TO->TO pair, deduped
    assert edges[0]["provenance"]["evidence"]["bridge_concept_count"] == 2


def test_deterministic_across_runs():
    chunks = [
        _chunk("c1", ["a"], ["TO-01"]),
        _chunk("c2", ["a"], ["TO-03"]),
        _chunk("c3", ["b"], ["TO-02"]),
        _chunk("c4", ["b"], ["TO-01"]),
    ]
    g = _graph(("a", ["c1", "c2"]), ("b", ["c3", "c4"]))
    out1 = infer(chunks, None, g, enabled="1")
    out2 = infer(chunks, None, g, enabled="1")
    assert [(e["source"], e["target"]) for e in out1] == [
        (e["source"], e["target"]) for e in out2
    ]
    # Sorted by (source, target).
    pairs = [(e["source"], e["target"]) for e in out1]
    assert pairs == sorted(pairs)


# --------------------------------------------------------------------------- #
# Decision capture
# --------------------------------------------------------------------------- #


def test_decision_capture_fires_when_edges_emitted():
    cap = _FakeCapture()
    chunks = [_chunk("c1", ["mitosis"], ["TO-01"]), _chunk("c2", ["mitosis"], ["TO-02"])]
    g = _graph(("mitosis", ["c1", "c2"]))
    infer(chunks, None, g, capture=cap, course_code="BIO", enabled="1")
    assert len(cap.calls) == 1
    call = cap.calls[0]
    assert call["decision_type"] == "typed_edge_inference"
    assert len(call["rationale"]) >= 20
    assert "BIO" in call["rationale"]
    assert call["ml_features"]["edges_emitted"] == 1


def test_no_capture_when_no_edges():
    cap = _FakeCapture()
    chunks = [_chunk("c1", ["mitosis"], ["TO-01"]), _chunk("c2", ["mitosis"], ["TO-01"])]
    g = _graph(("mitosis", ["c1", "c2"]))
    infer(chunks, None, g, capture=cap, course_code="BIO", enabled="1")
    assert cap.calls == []


def test_never_raises_on_garbage_input():
    assert infer(None, None, None, enabled="1") == []
    assert infer([{"bad": 1}], None, {"nodes": "x"}, enabled="1") == []
