"""W3 — cooccurrence page-aggregation tests (T14-T18).

Synthetic chunk fixtures only. Uses TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE
so single-occurrence tags survive the node-mint floor and the edge structure is
observable on tiny fixtures.
"""

from __future__ import annotations

import pytest

from lib.ontology.cooccurrence_graph import build_cooccurrence_graph


@pytest.fixture(autouse=True)
def _single_occurrence(monkeypatch):
    # Drop the 2-chunk node-mint floor so the tiny fixtures mint every tag as a
    # node; the cooccurrence pair structure (the thing under test) is then
    # observable without padding the fixtures to freq>=2.
    monkeypatch.setenv("TRAINFORGE_CONCEPT_GRAPH_INCLUDE_SINGLE_OCCURRENCE", "true")
    yield


def _edge_pairs(graph) -> dict:
    return {
        tuple(sorted([e["source"], e["target"]])): e["weight"]
        for e in graph["edges"]
    }


def _node_signature(graph) -> list:
    """Node id + frequency + occurrences[] — the chunk-level node provenance."""
    return [
        (n["id"], n["frequency"], tuple(n.get("occurrences", [])))
        for n in graph["nodes"]
    ]


# Two chunks on the SAME page. Chunk A teaches {rdf-graph, triple};
# chunk B teaches {triple, sparql}. rdf-graph and sparql NEVER share a chunk.
_SAME_PAGE_CHUNKS = [
    {
        "id": "c1",
        "concept_tags": ["rdf-graph", "triple"],
        "source": {"lesson_id": "week_04", "section_heading": "Graphs"},
    },
    {
        "id": "c2",
        "concept_tags": ["triple", "sparql"],
        "source": {"lesson_id": "week_04", "section_heading": "Queries"},
    },
]


# --------------------------------------------------------------------------
# T14 — chunk-local fragments, page-level connects
# --------------------------------------------------------------------------
def test_t14_chunk_fragments_page_connects():
    chunk_edges = _edge_pairs(
        build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="chunk")
    )
    page_edges = _edge_pairs(
        build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="page")
    )

    # Chunk-local: rdf-graph and sparql never share a chunk → NO edge.
    assert ("rdf-graph", "sparql") not in chunk_edges
    # Page-level: the page's union tag set connects them.
    assert ("rdf-graph", "sparql") in page_edges
    # The within-chunk pairs survive in both.
    assert ("rdf-graph", "triple") in chunk_edges
    assert ("rdf-graph", "triple") in page_edges


# --------------------------------------------------------------------------
# T15 — node set + occurrences unchanged (only edges/weights change)
# --------------------------------------------------------------------------
def test_t15_node_set_byte_identical():
    chunk_graph = build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="chunk")
    page_graph = build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="page")
    section_graph = build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="section")

    sig = _node_signature(chunk_graph)
    assert _node_signature(page_graph) == sig
    assert _node_signature(section_graph) == sig


# --------------------------------------------------------------------------
# T16 — section grouping keys on section_heading
# --------------------------------------------------------------------------
def test_t16_section_grouping():
    # Same as T14 but the two chunks live in DIFFERENT sections, so section
    # grouping must NOT connect rdf-graph<->sparql (they're in different
    # sections), while page grouping (same lesson_id) does.
    section_edges = _edge_pairs(
        build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="section")
    )
    assert ("rdf-graph", "sparql") not in section_edges

    # When the two chunks SHARE a section, section grouping connects them.
    shared_section = [
        {
            "id": "c1",
            "concept_tags": ["rdf-graph", "triple"],
            "source": {"lesson_id": "week_04", "section_heading": "Same"},
        },
        {
            "id": "c2",
            "concept_tags": ["triple", "sparql"],
            "source": {"lesson_id": "week_04", "section_heading": "Same"},
        },
    ]
    shared_edges = _edge_pairs(
        build_cooccurrence_graph(shared_section, group_by="section")
    )
    assert ("rdf-graph", "sparql") in shared_edges


# --------------------------------------------------------------------------
# T17 — env override drives the level through the _run_concept_extraction caller
# --------------------------------------------------------------------------
def test_t17_env_override_unset_is_chunk_byte_stable(monkeypatch):
    # The caller resolves os.environ["TRAINFORGE_COOCCURRENCE_GROUP_BY"] or
    # "chunk". Replicate the caller's resolution + assert the result matches
    # the explicit group_by="chunk" graph (byte-stable legacy when unset).
    monkeypatch.delenv("TRAINFORGE_COOCCURRENCE_GROUP_BY", raising=False)
    import os
    resolved = os.environ.get("TRAINFORGE_COOCCURRENCE_GROUP_BY", "").strip() or "chunk"
    assert resolved == "chunk"
    unset_graph = build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by=resolved)
    legacy_graph = build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="chunk")
    assert _edge_pairs(unset_graph) == _edge_pairs(legacy_graph)


def test_t17_env_override_page(monkeypatch):
    monkeypatch.setenv("TRAINFORGE_COOCCURRENCE_GROUP_BY", "page")
    import os
    resolved = os.environ.get("TRAINFORGE_COOCCURRENCE_GROUP_BY", "").strip() or "chunk"
    assert resolved == "page"
    page_edges = _edge_pairs(
        build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by=resolved)
    )
    assert ("rdf-graph", "sparql") in page_edges


def test_unrecognized_group_by_falls_back_to_chunk():
    edges = _edge_pairs(
        build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="bogus")
    )
    legacy = _edge_pairs(
        build_cooccurrence_graph(_SAME_PAGE_CHUNKS, group_by="chunk")
    )
    assert edges == legacy


def test_missing_group_key_falls_back_to_per_chunk():
    # Chunks with no lesson/module/section: page grouping must degrade to
    # per-chunk (never collapse into one all-chunks group), so rdf-graph and
    # sparql (different chunks, no page key) stay disconnected.
    no_key = [
        {"id": "c1", "concept_tags": ["rdf-graph", "triple"], "source": {}},
        {"id": "c2", "concept_tags": ["triple", "sparql"], "source": {}},
    ]
    page_edges = _edge_pairs(build_cooccurrence_graph(no_key, group_by="page"))
    assert ("rdf-graph", "sparql") not in page_edges
    assert ("rdf-graph", "triple") in page_edges


# --------------------------------------------------------------------------
# T18 — related-to edge from page-aggregated weight into the semantic graph
# --------------------------------------------------------------------------
def test_t18_edges_into_semantic_graph(monkeypatch):
    from Trainforge.rag.typed_edge_inference import build_semantic_graph

    # A pair that co-occurs in >=3 PAGES but in 0 shared chunks. Build a
    # corpus of 3 pages; on each page rdf-graph and sparql appear in DIFFERENT
    # chunks (never sharing a chunk), so only PAGE aggregation reaches the
    # related-to threshold of 3.
    chunks = []
    for page in range(3):
        chunks.append({
            "id": f"p{page}_a",
            "concept_tags": ["rdf-graph"],
            "source": {"lesson_id": f"week_{page:02d}"},
        })
        chunks.append({
            "id": f"p{page}_b",
            "concept_tags": ["sparql"],
            "source": {"lesson_id": f"week_{page:02d}"},
        })

    chunk_cog = build_cooccurrence_graph(chunks, group_by="chunk")
    page_cog = build_cooccurrence_graph(chunks, group_by="page")

    chunk_edges = _edge_pairs(chunk_cog)
    page_edges = _edge_pairs(page_cog)
    # Chunk-local: rdf-graph and sparql never share a chunk → no co-occurs edge.
    assert ("rdf-graph", "sparql") not in chunk_edges
    # Page-aggregated: weight == 3 (one per page).
    assert page_edges.get(("rdf-graph", "sparql")) == 3

    # Feed the page-aggregated cooccurrence graph into the semantic builder and
    # assert a related-to edge lands with the documented confidence formula.
    semantic = build_semantic_graph(chunks, None, page_cog)
    related = [
        e
        for e in semantic["edges"]
        if e.get("type") == "related-to"
        and set([e["source"], e["target"]]) == {"rdf-graph", "sparql"}
    ]
    assert related, "page-aggregated weight>=3 should yield a related-to edge"
    edge = related[0]
    assert edge["provenance"]["evidence"]["cooccurrence_weight"] == 3
    # confidence = min(1.0, 0.4 + 0.05 * weight) = 0.4 + 0.15 = 0.55
    assert abs(edge["confidence"] - 0.55) < 1e-9
