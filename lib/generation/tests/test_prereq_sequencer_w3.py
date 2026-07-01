"""Tests for the W3 prereq_sequencer additions (federation union, transitive
reduction, concept-centrality tie-break). Synthetic in-memory graphs only."""
from __future__ import annotations

import pytest

from lib.generation.prereq_sequencer import (
    resolve_centrality_tiebreak,
    resolve_transitive_reduction,
    sequence_terminal_objectives,
)


def _to(tid, concepts):
    return {"id": tid, "key_concepts": list(concepts), "statement": tid}


def _ids(tos):
    return [t["id"] for t in tos]


def _concept_graph(*edges):
    """edges: (source, target, type, confidence)."""
    return {
        "kind": "concept_semantic",
        "nodes": [],
        "edges": [
            {"source": s, "target": t, "type": ty, "confidence": c}
            for (s, t, ty, c) in edges
        ],
    }


# --------------------------------------------------------------------------- #
# Flag resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("val,expected", [("1", True), ("", False), ("no", False)])
def test_flag_resolvers(val, expected):
    assert resolve_transitive_reduction(val) is expected
    assert resolve_centrality_tiebreak(val) is expected


# --------------------------------------------------------------------------- #
# W3.1 — federation TO->TO prerequisite edges consumed directly
# --------------------------------------------------------------------------- #


def test_federation_to_edge_reorders():
    # A federation edge whose endpoints are already TO ids: TO-B depends on
    # TO-A (edge source=TO-B dependent, target=TO-A prereq).
    tos = [_to("TO-B", ["b"]), _to("TO-A", ["a"])]
    g = _concept_graph(("TO-B", "TO-A", "prerequisite", 0.7))
    out, sig = sequence_terminal_objectives(tos, g)
    assert _ids(out) == ["TO-A", "TO-B"]
    assert sig["federation_edges"] == 1
    assert sig["concept_projected_edges"] == 0
    assert sig["to_edges_used"] == 1
    assert sig["reordered"] is True


def test_federation_case_insensitive_match():
    tos = [_to("TO-B", ["b"]), _to("TO-A", ["a"])]
    g = _concept_graph(("to-b", "to-a", "prerequisite", 0.7))
    out, sig = sequence_terminal_objectives(tos, g)
    assert _ids(out) == ["TO-A", "TO-B"]
    assert sig["federation_edges"] == 1


def test_union_concept_and_federation():
    # Concept edge (c depends on a) + federation edge (TO-D depends on TO-C).
    tos = [_to("TO-A", ["a"]), _to("TO-B", ["b"]),
           _to("TO-C", ["c"]), _to("TO-D", ["d"])]
    g = _concept_graph(
        ("d_concept", "a", "prerequisite", 0.6),          # unmapped source -> ignored
        ("b", "a", "prerequisite", 0.6),                  # TO-B depends on TO-A
        ("TO-D", "TO-C", "prerequisite", 0.7),            # federation
    )
    out, sig = sequence_terminal_objectives(tos, g)
    assert sig["federation_edges"] == 1
    assert sig["concept_projected_edges"] == 1
    assert sig["ignored_unmapped_edges"] == 1
    # A precedes B; C precedes D.
    ids = _ids(out)
    assert ids.index("TO-A") < ids.index("TO-B")
    assert ids.index("TO-C") < ids.index("TO-D")
    assert set(ids) == {"TO-A", "TO-B", "TO-C", "TO-D"}


def test_federation_off_signal_default_zero():
    # No federation edges present → count 0, concept path used.
    tos = [_to("TO-B", ["b"]), _to("TO-A", ["a"])]
    g = _concept_graph(("b", "a", "prerequisite", 0.6))
    out, sig = sequence_terminal_objectives(tos, g)
    assert _ids(out) == ["TO-A", "TO-B"]
    assert sig["federation_edges"] == 0
    assert sig["concept_projected_edges"] == 1


# --------------------------------------------------------------------------- #
# W3.3 — transitive reduction (gated, default OFF → byte-identical)
# --------------------------------------------------------------------------- #


def test_transitive_reduction_off_keeps_all_edges(monkeypatch):
    monkeypatch.delenv("ED4ALL_PREREQ_TRANSITIVE_REDUCTION", raising=False)
    # A->B->C chain plus the redundant skip A->C, expressed as TO federation.
    tos = [_to("TO-A", ["a"]), _to("TO-B", ["b"]), _to("TO-C", ["c"])]
    g = _concept_graph(
        ("TO-B", "TO-A", "prerequisite", 0.7),  # B dep on A
        ("TO-C", "TO-B", "prerequisite", 0.7),  # C dep on B
        ("TO-C", "TO-A", "prerequisite", 0.7),  # C dep on A (redundant skip)
    )
    out, sig = sequence_terminal_objectives(tos, g)
    assert sig["to_edges_used"] == 3
    assert sig["transitively_reduced"] == 0
    assert _ids(out) == ["TO-A", "TO-B", "TO-C"]


def test_transitive_reduction_on_drops_skip_edge(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_TRANSITIVE_REDUCTION", "1")
    tos = [_to("TO-A", ["a"]), _to("TO-B", ["b"]), _to("TO-C", ["c"])]
    g = _concept_graph(
        ("TO-B", "TO-A", "prerequisite", 0.7),
        ("TO-C", "TO-B", "prerequisite", 0.7),
        ("TO-C", "TO-A", "prerequisite", 0.7),  # redundant
    )
    out, sig = sequence_terminal_objectives(tos, g)
    # One redundant edge removed (A->C skip); reachability preserved.
    assert sig["transitively_reduced"] == 1
    assert sig["to_edges_used"] == 2
    assert _ids(out) == ["TO-A", "TO-B", "TO-C"]


def test_transitive_reduction_preserves_two_cycle(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_TRANSITIVE_REDUCTION", "1")
    tos = [_to("TO-A", ["a"]), _to("TO-B", ["b"])]
    g = _concept_graph(
        ("TO-A", "TO-B", "prerequisite", 0.6),
        ("TO-B", "TO-A", "prerequisite", 0.3),
    )
    out, sig = sequence_terminal_objectives(tos, g)
    # Neither edge is reachable without itself → both kept → cycle broken once.
    assert sig["transitively_reduced"] == 0
    assert set(_ids(out)) == {"TO-A", "TO-B"}
    assert sig["cycles_broken"] == 1


# --------------------------------------------------------------------------- #
# W3.4 — concept-centrality tie-break (gated, default OFF → byte-identical)
# --------------------------------------------------------------------------- #


def test_centrality_off_keeps_original_tiebreak(monkeypatch):
    monkeypatch.delenv("ED4ALL_PREREQ_CENTRALITY_TIEBREAK", raising=False)
    tos = [_to("TO-A", ["a"]), _to("TO-B", ["b"]), _to("TO-C", ["c"])]
    # Only C depends on A; B independent. Original tie-break keeps B before...
    g = _concept_graph(("c", "a", "prerequisite", 0.6))
    out, sig = sequence_terminal_objectives(tos, g)
    assert sig["centrality_tiebreak_enabled"] is False
    assert sig["concept_centrality"] == {}
    # A already precedes C; original order preserved.
    assert _ids(out) == ["TO-A", "TO-B", "TO-C"]


def test_centrality_on_front_loads_high_centrality(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_CENTRALITY_TIEBREAK", "1")
    monkeypatch.setenv("ED4ALL_PREREQ_CENTRALITY_METHOD", "in_degree")
    # Two independent-from-each-other TOs that BOTH become "ready" together,
    # plus a dependency to force a topo sort. TO-HUB owns a high-in-degree
    # concept; TO-LOW owns a peripheral one. Both depend on nothing, TO-DEP
    # depends on TO-HUB. Centrality should surface TO-HUB before TO-LOW.
    tos = [_to("TO-LOW", ["low"]), _to("TO-HUB", ["hub"]), _to("TO-DEP", ["dep"])]
    g = {
        "kind": "concept_semantic",
        "nodes": [{"id": "hub"}, {"id": "low"}, {"id": "dep"},
                  {"id": "x"}, {"id": "y"}, {"id": "z"}],
        "edges": [
            # hub has in-degree 3 (x,y,z point at it); low has 0.
            {"source": "x", "target": "hub", "type": "related-to", "confidence": 0.5},
            {"source": "y", "target": "hub", "type": "related-to", "confidence": 0.5},
            {"source": "z", "target": "hub", "type": "related-to", "confidence": 0.5},
            # TO-DEP depends on TO-HUB (federation) so a real order forms.
            {"source": "TO-DEP", "target": "TO-HUB", "type": "prerequisite",
             "confidence": 0.7},
        ],
    }
    out, sig = sequence_terminal_objectives(tos, g)
    assert sig["centrality_tiebreak_enabled"] is True
    assert sig["centrality_method"] == "in_degree"
    assert sig["concept_centrality"].get("hub", 0) == 3.0
    ids = _ids(out)
    # HUB (high centrality) is front-loaded ahead of LOW despite LOW's lower
    # original index, and DEP follows its prerequisite HUB.
    assert ids.index("TO-HUB") < ids.index("TO-LOW")
    assert ids.index("TO-HUB") < ids.index("TO-DEP")
    assert set(ids) == {"TO-HUB", "TO-LOW", "TO-DEP"}


def test_centrality_pagerank_method_exposed(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_CENTRALITY_TIEBREAK", "1")
    monkeypatch.setenv("ED4ALL_PREREQ_CENTRALITY_METHOD", "pagerank")
    tos = [_to("TO-A", ["a"]), _to("TO-B", ["b"])]
    g = {
        "kind": "concept_semantic",
        "nodes": [{"id": "a"}, {"id": "b"}],
        "edges": [
            {"source": "a", "target": "b", "type": "related-to", "confidence": 0.5},
            {"source": "TO-B", "target": "TO-A", "type": "prerequisite",
             "confidence": 0.7},
        ],
    }
    out, sig = sequence_terminal_objectives(tos, g)
    assert sig["centrality_method"] == "pagerank"
    # PageRank scores are floats summing ~1.0 across nodes.
    assert set(sig["concept_centrality"].keys()) == {"a", "b"}
    assert abs(sum(sig["concept_centrality"].values()) - 1.0) < 1e-6
    assert _ids(out) == ["TO-A", "TO-B"]


def test_partition_invariant_with_all_flags_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_PREREQ_TRANSITIVE_REDUCTION", "1")
    monkeypatch.setenv("ED4ALL_PREREQ_CENTRALITY_TIEBREAK", "1")
    tos = [_to(f"TO-{i:02d}", [f"c{i}"]) for i in range(5)]
    g = _concept_graph(
        ("TO-01", "TO-00", "prerequisite", 0.7),
        ("TO-02", "TO-01", "prerequisite", 0.7),
        ("TO-02", "TO-00", "prerequisite", 0.7),
    )
    out, _ = sequence_terminal_objectives(tos, g)
    assert sorted(_ids(out)) == sorted(_ids(tos))
    assert len(out) == 5
