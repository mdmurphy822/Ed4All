"""Tests for the per-node top-K related-to fan-out cap."""

from __future__ import annotations

import copy
import json

from lib.ontology.related_edge_cap import cap_related_fanout


def _edge(source, target, etype="related-to", confidence=0.5, weight=1):
    return {
        "source": source,
        "target": target,
        "type": etype,
        "confidence": confidence,
        "weight": weight,
    }


def _edge_keys(graph):
    return {
        (e["source"], e["target"], e["type"])
        for e in graph["edges"]
        if e["type"] == "related-to"
    }


def test_star_graph_keeps_top_k_by_confidence_weight():
    # Hub with 10 spokes, strictly varied confidence so ranking is unambiguous.
    edges = []
    for i in range(10):
        edges.append(
            _edge("hub", f"spoke{i}", confidence=0.10 * (i + 1), weight=i + 1)
        )
    graph = {"nodes": [], "edges": edges}

    out = cap_related_fanout(graph, k=3)

    survivors = _edge_keys(out)
    # Top-3 by confidence are spoke9, spoke8, spoke7.
    assert ("hub", "spoke9", "related-to") in survivors
    assert ("hub", "spoke8", "related-to") in survivors
    assert ("hub", "spoke7", "related-to") in survivors
    # Lower-ranked spokes dropped (union: each spoke is also a leaf node whose
    # only edge is to the hub, so spokes are top-K for themselves... they would
    # all survive. Use a hub-only union check via degrees below). Here every
    # spoke has degree 1 so it survives as its own top-K — verify that.
    assert len(survivors) == 10
    # But the hub must only rank 3 as its own top-K; confirm the dropped-from-hub
    # ones survived solely on the spoke side by re-running with spokes that have
    # competing edges -> covered in the dedicated union test.
    assert out["related_fanout_cap_k"] == 3


def test_hub_only_top_k_drops_lower_edges():
    # Pure star where spokes have NO other edges would let every spoke survive on
    # its own. To isolate the hub cap, give each spoke a second, stronger edge so
    # the hub edge is NOT the spoke's top-K either.
    edges = []
    for i in range(10):
        # Weak hub edge.
        edges.append(_edge("hub", f"spoke{i}", confidence=0.10 * (i + 1), weight=1))
        # Strong competing edge on the spoke so hub edge isn't spoke's top-1.
        edges.append(_edge(f"spoke{i}", f"anchor{i}", confidence=0.99, weight=99))
    graph = {"nodes": [], "edges": edges}

    out = cap_related_fanout(graph, k=1)
    survivors = _edge_keys(out)

    # Hub's single top-1 is spoke9 (confidence 1.0). That hub edge survives.
    assert ("hub", "spoke9", "related-to") in survivors
    # Lower hub edges are NOT top-1 for hub and NOT top-1 for their spoke
    # (the anchor edge wins on the spoke) -> dropped.
    assert ("hub", "spoke0", "related-to") not in survivors
    assert ("hub", "spoke5", "related-to") not in survivors
    # Each spoke's strong anchor edge survives.
    assert ("spoke0", "anchor0", "related-to") in survivors


def test_union_semantics_edge_survives_via_target():
    # 'a' has many strong outgoing edges; the a->b edge is weak so it is NOT in
    # a's top-K. But b is peripheral: a->b is b's strongest (only) edge, so it is
    # b's top-K. Union semantics must keep it.
    edges = [
        _edge("a", "b", confidence=0.1, weight=1),  # weak for a, top for b
        _edge("a", "c", confidence=0.9, weight=9),
        _edge("a", "d", confidence=0.8, weight=8),
        _edge("a", "e", confidence=0.7, weight=7),
    ]
    graph = {"nodes": [], "edges": edges}

    out = cap_related_fanout(graph, k=2)
    survivors = _edge_keys(out)

    # a's own top-2: a->c, a->d. a->b is NOT top-2 for a.
    assert ("a", "c", "related-to") in survivors
    assert ("a", "d", "related-to") in survivors
    # a->b survives because it is b's top-K (b's only incident edge).
    assert ("a", "b", "related-to") in survivors
    # a->e is top-K for neither (a's rank 3, e's... e only has this edge so it's
    # e's top-K) -> survives via e. Confirm union keeps it.
    assert ("a", "e", "related-to") in survivors


def test_union_drops_edge_top_k_for_neither():
    # Construct an edge that is top-K for NEITHER endpoint.
    edges = [
        # Node 'x' strong edges.
        _edge("x", "x1", confidence=0.9, weight=9),
        _edge("x", "x2", confidence=0.8, weight=8),
        # Node 'y' strong edges.
        _edge("y", "y1", confidence=0.95, weight=10),
        _edge("y", "y2", confidence=0.85, weight=9),
        # The weak bridge: weak for x (rank 3) and weak for y (rank 3).
        _edge("x", "y", confidence=0.1, weight=1),
    ]
    graph = {"nodes": [], "edges": edges}

    out = cap_related_fanout(graph, k=2)
    survivors = _edge_keys(out)

    # The bridge is top-2 for neither x nor y -> dropped.
    assert ("x", "y", "related-to") not in survivors
    # Strong edges retained.
    assert ("x", "x1", "related-to") in survivors
    assert ("y", "y1", "related-to") in survivors


def test_non_related_edges_never_dropped():
    edges = [
        _edge("c", "def", etype="defined-by", confidence=0.01, weight=1),
        _edge("c", "ex", etype="exemplifies", confidence=0.01, weight=1),
        _edge("c", "p", etype="is-a", confidence=0.01, weight=1),
        # Lots of strong related-to edges on c so a tiny k would drop them.
        _edge("c", "r1", confidence=0.9, weight=9),
        _edge("c", "r2", confidence=0.8, weight=8),
        _edge("c", "r3", confidence=0.7, weight=7),
    ]
    graph = {"nodes": [], "edges": edges}

    out = cap_related_fanout(graph, k=1)
    types = {(e["source"], e["target"], e["type"]) for e in out["edges"]}

    # Non-related-to edges always present regardless of k.
    assert ("c", "def", "defined-by") in types
    assert ("c", "ex", "exemplifies") in types
    assert ("c", "p", "is-a") in types


def test_determinism_two_runs_identical():
    edges = [
        _edge("hub", f"s{i}", confidence=0.1 * (i + 1), weight=i + 1)
        for i in range(8)
    ]
    graph = {"nodes": [{"id": "hub"}], "edges": edges}

    out1 = cap_related_fanout(copy.deepcopy(graph), k=3)
    out2 = cap_related_fanout(copy.deepcopy(graph), k=3)

    assert json.dumps(out1, sort_keys=True) == json.dumps(out2, sort_keys=True)


def test_determinism_stable_under_input_reordering():
    edges = [
        _edge("hub", f"s{i}", confidence=0.1 * (i + 1), weight=i + 1)
        for i in range(8)
    ]
    graph_a = {"nodes": [], "edges": list(edges)}
    graph_b = {"nodes": [], "edges": list(reversed(edges))}

    out_a = cap_related_fanout(graph_a, k=3)
    out_b = cap_related_fanout(graph_b, k=3)

    # Same surviving edge SET regardless of input order.
    assert _edge_keys(out_a) == _edge_keys(out_b)


def test_k_larger_than_degree_is_identity():
    edges = [
        _edge("hub", f"s{i}", confidence=0.1 * (i + 1), weight=i + 1)
        for i in range(5)
    ]
    edges.append(_edge("hub", "d", etype="defined-by"))
    graph = {"nodes": [], "edges": edges}

    out = cap_related_fanout(graph, k=100)

    assert len(out["edges"]) == len(edges)
    assert _edge_keys(out) == _edge_keys({"edges": edges})


def test_missing_confidence_weight_fall_back_to_zero():
    # Edges with absent fields must still rank (as 0) without raising.
    edges = [
        {"source": "a", "target": "b", "type": "related-to"},  # no conf/weight
        _edge("a", "c", confidence=0.9, weight=9),
    ]
    graph = {"nodes": [], "edges": edges}

    out = cap_related_fanout(graph, k=1)
    survivors = _edge_keys(out)
    # a->c (strong) is a's top-1; a->b survives only via b (b's only edge).
    assert ("a", "c", "related-to") in survivors
    assert ("a", "b", "related-to") in survivors


def test_input_not_mutated():
    edges = [_edge("hub", f"s{i}", confidence=0.1 * (i + 1)) for i in range(5)]
    graph = {"nodes": [], "edges": edges}
    before = copy.deepcopy(graph)

    cap_related_fanout(graph, k=1)

    assert graph == before
    assert "related_fanout_cap_k" not in graph
