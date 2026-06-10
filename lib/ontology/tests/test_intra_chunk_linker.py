"""Tests for the intra-chunk concept-linking pass.

Covers the core fragmentation fix (co-located concepts pairwise linked), the
non-linking of concepts in different chunks, no-duplication of an existing
``related-to`` edge, the per-chunk pair cap on a dense chunk, and determinism
(byte-identical across two runs).
"""

from __future__ import annotations

import copy

import pytest

from lib.ontology.intra_chunk_linker import (
    RULE_NAME,
    RULE_VERSION,
    link_intra_chunk_concepts,
)
from lib.validators.evidence import get_schema


def _concept(node_id, *, frequency=1, label=None):
    return {
        "id": node_id,
        "label": label if label is not None else node_id.replace("-", " ").title(),
        "frequency": frequency,
        "class": "DomainConcept",
    }


def _chunk(chunk_id):
    return {"id": chunk_id, "class": "Chunk", "label": chunk_id}


def _edge(source, target, *, type="defined-by", confidence=0.7):
    return {"source": source, "target": target, "type": type, "confidence": confidence}


def _related_edges(graph):
    return [e for e in graph["edges"] if e.get("type") == "related-to"]


def _intra_edges(graph):
    return [
        e
        for e in graph["edges"]
        if e.get("type") == "related-to"
        and isinstance(e.get("provenance"), dict)
        and e["provenance"].get("origin") == "intra_chunk"
    ]


def _undirected(edge):
    return frozenset((edge["source"], edge["target"]))


# ---------------------------------------------------------------------------
# Core: co-located concepts get pairwise-linked
# ---------------------------------------------------------------------------


def test_three_colocated_concepts_pairwise_linked():
    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _concept("state-graph"),
            _concept("conditional-routing"),
            _concept("terminal-states"),
        ],
        "edges": [
            _edge("state-graph", "chunk_00001"),
            _edge("conditional-routing", "chunk_00001"),
            _edge("terminal-states", "chunk_00001"),
        ],
    }

    out = link_intra_chunk_concepts(copy.deepcopy(graph))

    intra = _intra_edges(out)
    assert len(intra) == 3  # 3 choose 2
    assert out["intra_chunk_links_added"] == 3

    # Every added edge is a low-confidence intra_chunk related-to.
    for e in intra:
        assert e["type"] == "related-to"
        assert e["confidence"] == 0.35
        assert e["provenance"]["origin"] == "intra_chunk"

    # The three concepts are now pairwise connected (undirected).
    pairs = {_undirected(e) for e in intra}
    assert pairs == {
        frozenset({"state-graph", "conditional-routing"}),
        frozenset({"state-graph", "terminal-states"}),
        frozenset({"conditional-routing", "terminal-states"}),
    }

    # Anchor edges and concept nodes are untouched; nodes unchanged.
    assert out["nodes"] == graph["nodes"]
    assert len([e for e in out["edges"] if e.get("type") == "defined-by"]) == 3


def test_exemplifies_also_anchors_colocation():
    """``exemplifies`` edges anchor co-location just like ``defined-by``."""
    graph = {
        "nodes": [
            _chunk("chunk_00009"),
            _concept("tool-execution"),
            _concept("llm-call"),
        ],
        "edges": [
            _edge("tool-execution", "chunk_00009", type="exemplifies"),
            _edge("llm-call", "chunk_00009", type="exemplifies"),
        ],
    }

    out = link_intra_chunk_concepts(graph)
    assert out["intra_chunk_links_added"] == 1
    assert _undirected(_intra_edges(out)[0]) == frozenset({"tool-execution", "llm-call"})


# ---------------------------------------------------------------------------
# Concepts in different chunks are NOT linked
# ---------------------------------------------------------------------------


def test_concepts_in_different_chunks_not_linked():
    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _chunk("chunk_00002"),
            _concept("alpha"),
            _concept("beta"),
        ],
        "edges": [
            _edge("alpha", "chunk_00001"),
            _edge("beta", "chunk_00002"),
        ],
    }

    out = link_intra_chunk_concepts(graph)
    assert out["intra_chunk_links_added"] == 0
    assert _intra_edges(out) == []


# ---------------------------------------------------------------------------
# Existing related-to between co-located concepts is NOT duplicated
# ---------------------------------------------------------------------------


def test_existing_related_to_not_duplicated():
    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _concept("alpha"),
            _concept("beta"),
            _concept("gamma"),
        ],
        "edges": [
            _edge("alpha", "chunk_00001"),
            _edge("beta", "chunk_00001"),
            _edge("gamma", "chunk_00001"),
            # Pre-existing real co-occurrence related-to (reverse direction of
            # the sorted pair) — must NOT be duplicated.
            {"source": "beta", "target": "alpha", "type": "related-to",
             "confidence": 0.8, "weight": 4},
        ],
    }

    out = link_intra_chunk_concepts(graph)

    # alpha-beta already exists -> only alpha-gamma and beta-gamma added.
    assert out["intra_chunk_links_added"] == 2
    intra_pairs = {_undirected(e) for e in _intra_edges(out)}
    assert frozenset({"alpha", "beta"}) not in intra_pairs
    assert intra_pairs == {
        frozenset({"alpha", "gamma"}),
        frozenset({"beta", "gamma"}),
    }

    # The original high-confidence edge survives unchanged exactly once.
    ab = [
        e for e in _related_edges(out)
        if _undirected(e) == frozenset({"alpha", "beta"})
    ]
    assert len(ab) == 1
    assert ab[0]["confidence"] == 0.8


# ---------------------------------------------------------------------------
# Per-chunk cap on a dense chunk
# ---------------------------------------------------------------------------


def test_dense_chunk_falls_back_to_star_no_concept_dropped():
    """A chunk whose clique overflows the cap links via a hub star.

    Regression for chunk_00034: the old top-k-clique cap dropped the
    lowest-ranked concepts past the cap, re-stranding them as singletons. The
    star fallback links EVERY co-located concept to the highest-frequency hub,
    so no concept is left with zero ``related-to`` edges.
    """
    # 6 concepts in one chunk -> 15 clique pairs; cap at 5 forces the star.
    concepts = [f"c{i}" for i in range(6)]
    graph = {
        "nodes": [_chunk("chunk_00001")]
        + [_concept(c, frequency=10 - i) for i, c in enumerate(concepts)],
        "edges": [_edge(c, "chunk_00001") for c in concepts],
    }

    out = link_intra_chunk_concepts(graph, max_pairs_per_chunk=5)

    # Star = N-1 edges (5 spokes off the hub), well under the 15-pair clique.
    assert out["intra_chunk_links_added"] == 5
    assert len(_intra_edges(out)) == out["intra_chunk_links_added"]

    # EVERY concept is linked at least once — nothing is dropped.
    linked_nodes = set()
    for e in _intra_edges(out):
        linked_nodes.add(e["source"])
        linked_nodes.add(e["target"])
    assert linked_nodes == set(concepts)
    assert "c5" in linked_nodes  # lowest-frequency concept is NOT stranded

    # The hub is the highest-frequency concept (c0, frequency 10): it touches
    # every other concept; the spokes touch only the hub.
    from collections import Counter

    deg = Counter()
    for e in _intra_edges(out):
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    assert deg["c0"] == 5  # hub
    for spoke in concepts[1:]:
        assert deg[spoke] == 1


def test_real_course_chunk_id_form_star_links_all_concepts():
    """Regression: real ``<course>_chunk_NNNNN`` id form, both anchor orientations.

    Mirrors ``demo_course_chunk_00034``: a single chunk anchors 16 concepts by
    ``defined-by``, with the chunk on BOTH sides of the anchor edge (concept ->
    chunk AND chunk -> concept). The chunk is far past the default cap so the
    star fallback engages — every concept must end up with >= 1 ``related-to``
    edge (zero singletons), reproducing the exact bug from the audit.
    """
    chunk_id = "demo_course_chunk_00034"
    # Names mirror the 16 real concepts on chunk_00034.
    concepts = [
        "agents", "checkpointresume", "conditional-edge", "declarative-graph",
        "deprecated", "dynamic", "edge", "human-in-the-loop", "llm-call",
        "memory", "multiple-agents", "node", "orchestration-model",
        "state-graph", "tool-execution", "workflow",
    ]
    # ``agents`` is the highest-frequency hub (mirrors the real freq=5).
    freqs = {c: (5 if c == "agents" else 2) for c in concepts}

    nodes = [_chunk(chunk_id)] + [
        _concept(c, frequency=freqs[c]) for c in concepts
    ]
    # Anchor edges in BOTH orientations: even-index concepts are the source
    # (concept -> chunk), odd-index concepts are the target (chunk -> concept).
    edges = []
    for i, c in enumerate(concepts):
        if i % 2 == 0:
            edges.append(_edge(c, chunk_id))            # concept -> chunk
        else:
            edges.append(_edge(chunk_id, c))            # chunk -> concept
    graph = {"nodes": nodes, "edges": edges}

    out = link_intra_chunk_concepts(copy.deepcopy(graph))

    # 16 concepts -> 120 clique pairs >> default cap (20) -> star: 15 edges.
    assert out["intra_chunk_links_added"] == len(concepts) - 1

    from collections import Counter

    deg = Counter()
    for e in _intra_edges(out):
        deg[e["source"]] += 1
        deg[e["target"]] += 1

    # Zero singletons: every co-located concept has at least one related edge.
    for c in concepts:
        assert deg[c] >= 1, f"{c} left a singleton"

    # The 10 originally-stranded concepts from the audit are now connected.
    stranded = [
        "deprecated", "dynamic", "human-in-the-loop", "llm-call", "memory",
        "multiple-agents", "orchestration-model", "state-graph",
        "tool-execution", "workflow",
    ]
    assert all(deg[c] >= 1 for c in stranded)

    # ``agents`` is the hub (touches all 15 others).
    assert deg["agents"] == len(concepts) - 1


def test_cap_does_not_restrict_small_chunks():
    """A chunk under the cap links every pair."""
    concepts = ["a", "b", "c", "d"]  # 6 pairs
    graph = {
        "nodes": [_chunk("chunk_00001")] + [_concept(c) for c in concepts],
        "edges": [_edge(c, "chunk_00001") for c in concepts],
    }
    out = link_intra_chunk_concepts(graph, max_pairs_per_chunk=20)
    assert out["intra_chunk_links_added"] == 6


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_across_runs():
    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _chunk("chunk_00002"),
            _concept("state-graph", frequency=5),
            _concept("conditional-routing", frequency=3),
            _concept("terminal-states", frequency=2),
            _concept("node-implementation", frequency=4),
            _concept("human-in-the-loop", frequency=1),
        ],
        "edges": [
            _edge("state-graph", "chunk_00001"),
            _edge("conditional-routing", "chunk_00001"),
            _edge("terminal-states", "chunk_00001"),
            _edge("node-implementation", "chunk_00002"),
            _edge("human-in-the-loop", "chunk_00002"),
        ],
    }

    out1 = link_intra_chunk_concepts(copy.deepcopy(graph))
    out2 = link_intra_chunk_concepts(copy.deepcopy(graph))
    assert out1["edges"] == out2["edges"]
    assert out1["intra_chunk_links_added"] == out2["intra_chunk_links_added"]


def test_input_not_mutated():
    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _concept("alpha"),
            _concept("beta"),
        ],
        "edges": [
            _edge("alpha", "chunk_00001"),
            _edge("beta", "chunk_00001"),
        ],
    }
    before = copy.deepcopy(graph)
    link_intra_chunk_concepts(graph)
    assert graph == before


# ---------------------------------------------------------------------------
# Provenance: rule_version stamping + schema validation
# ---------------------------------------------------------------------------


def test_emitted_edges_carry_integer_rule_version():
    """Every intra-chunk edge stamps integer ``rule_version >= 1`` provenance.

    Regression for the schema-invalid emit: the linker previously stamped
    ``provenance = {"rule": "intra_chunk_link", "origin": "intra_chunk"}`` with
    NO ``rule_version``, so the edge matched no ``oneOf`` arm of
    ``concept_graph_semantic.schema.json`` (FallbackProvenance requires an
    integer ``rule_version``) and any flagged graph failed schema validation.
    """
    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _concept("alpha"),
            _concept("beta"),
            _concept("gamma"),
        ],
        "edges": [
            _edge("alpha", "chunk_00001"),
            _edge("beta", "chunk_00001"),
            _edge("gamma", "chunk_00001"),
        ],
    }

    out = link_intra_chunk_concepts(graph)

    intra = _intra_edges(out)
    assert intra, "expected at least one intra-chunk edge"
    for e in intra:
        prov = e["provenance"]
        assert prov["rule"] == RULE_NAME == "intra_chunk_link"
        rv = prov["rule_version"]
        assert isinstance(rv, int) and not isinstance(rv, bool)
        assert rv >= 1
        assert rv == RULE_VERSION


def test_rule_versions_map_carries_intra_chunk_link_entry():
    """The graph's top-level ``rule_versions`` map gains an integer entry.

    ``kg_quality_report._summarize_rule_outputs`` keys
    ``rule_outputs[].rule_version`` off this map, so a missing entry surfaced
    as ``rule_version: null``. The pass must reflect its version into the map.
    """
    graph = {
        "rule_versions": {"related_from_cooccurrence": 1},
        "nodes": [
            _chunk("chunk_00001"),
            _concept("alpha"),
            _concept("beta"),
        ],
        "edges": [
            _edge("alpha", "chunk_00001"),
            _edge("beta", "chunk_00001"),
        ],
    }

    out = link_intra_chunk_concepts(graph)

    versions = out["rule_versions"]
    assert versions[RULE_NAME] == RULE_VERSION
    assert isinstance(versions[RULE_NAME], int) and not isinstance(versions[RULE_NAME], bool)
    # Pre-existing entries are preserved; input is not mutated.
    assert versions["related_from_cooccurrence"] == 1
    assert "rule_versions" not in graph or RULE_NAME not in graph["rule_versions"]


def test_no_edges_added_leaves_rule_versions_untouched():
    """A no-op pass must not stamp the ``rule_versions`` map (byte stability)."""
    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _chunk("chunk_00002"),
            _concept("alpha"),
            _concept("beta"),
        ],
        "edges": [
            _edge("alpha", "chunk_00001"),
            _edge("beta", "chunk_00002"),  # different chunks → nothing to link
        ],
    }

    out = link_intra_chunk_concepts(graph)
    assert out["intra_chunk_links_added"] == 0
    assert "rule_versions" not in out


def test_emitted_graph_fragment_validates_against_fallback_arm():
    """A graph fragment carrying intra-chunk edges validates against the schema.

    The intra_chunk_link rule is NOT one of the modeled rules, so its edges
    flow to the lenient ``FallbackProvenance`` arm — which requires
    ``[rule, rule_version]`` with integer ``rule_version >= 1``. Validated
    against the full concept_graph_semantic schema (lenient mode) to confirm
    the fix makes flagged graphs schema-VALID.
    """
    jsonschema = pytest.importorskip("jsonschema")

    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _concept("state-graph"),
            _concept("conditional-routing"),
            _concept("terminal-states"),
        ],
        "edges": [
            _edge("state-graph", "chunk_00001"),
            _edge("conditional-routing", "chunk_00001"),
            _edge("terminal-states", "chunk_00001"),
        ],
    }

    out = link_intra_chunk_concepts(graph)

    # The intra-chunk edges (every one of them) validate against the full
    # concept_semantic schema via the lenient FallbackProvenance arm. (The
    # bare ``_edge`` anchor-helper edges in this fixture carry no provenance,
    # which the schema requires on EVERY edge; that's a test-fixture shortcut,
    # not the production anchor-edge shape — the real co-occurrence anchor
    # edges always carry provenance. We validate the edges this pass emits.)
    schema = get_schema(strict=False)
    intra = _intra_edges(out)
    assert len(intra) == 3
    artifact = {
        "kind": "concept_semantic",
        "generated_at": "2026-06-09T00:00:00+00:00",
        "nodes": out["nodes"],
        "edges": intra,
    }
    jsonschema.validate(instance=artifact, schema=schema)


def test_emitted_intra_edge_rejected_without_rule_version():
    """Sentinel: stripping rule_version makes the edge schema-INVALID.

    Confirms the schema arm genuinely enforces the field the fix adds (guards
    against the schema silently accepting the pre-fix shape).
    """
    jsonschema = pytest.importorskip("jsonschema")

    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _concept("alpha"),
            _concept("beta"),
        ],
        "edges": [
            _edge("alpha", "chunk_00001"),
            _edge("beta", "chunk_00001"),
        ],
    }
    out = link_intra_chunk_concepts(graph)
    bad_edge = copy.deepcopy(_intra_edges(out)[0])
    bad_edge["provenance"].pop("rule_version")

    artifact = {
        "kind": "concept_semantic",
        "generated_at": "2026-06-09T00:00:00+00:00",
        "nodes": [],
        "edges": [bad_edge],
    }
    schema = get_schema(strict=False)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=artifact, schema=schema)


def test_non_concept_nodes_untouched():
    """Chunk nodes are never linked to each other or treated as concepts."""
    graph = {
        "nodes": [
            _chunk("chunk_00001"),
            _chunk("chunk_00002"),
            _concept("alpha"),
        ],
        "edges": [
            _edge("alpha", "chunk_00001"),
            # Two chunks, but only one concept co-located -> nothing to link.
            _edge("alpha", "chunk_00002"),
        ],
    }
    out = link_intra_chunk_concepts(graph)
    assert out["intra_chunk_links_added"] == 0
