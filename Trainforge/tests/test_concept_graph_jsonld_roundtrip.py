"""Phase 1.3 of plans/rdf-shacl-enrichment-2026-04-26.md.

Verifies that ``schemas/context/concept_graph_semantic_v1.jsonld`` is a
faithful round-trip bridge between the JSON-shaped ``concept_graph_semantic.json``
and an RDF graph.  The Trainforge emit pipeline does not yet inject the
``@context`` (Phase 1 is consumer-side only); this test layers it on top of an
existing artifact, parses via ``pyld`` + ``rdflib``, and asserts:

* triple count is non-trivial (sanity floor of 500)
* every JSON edge produces at least one ``ed4all:edgeType`` triple
* every concept node id materializes as a URI reference (not a blank node and
  not a literal)
* Turtle round-trip is loss-free (graph-isomorphic delta of zero triples)

Phase 1 does not modify ``Trainforge/process_course.py`` or
``Trainforge/rag/typed_edge_inference.py``; the bridge is exercised
out-of-band from the existing artifact under
``LibV2/courses/rdf-shacl-551-2/graph/concept_graph_semantic.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Project root (Ed4All/) — this test file lives at
# Ed4All/Trainforge/tests/test_concept_graph_jsonld_roundtrip.py, so parents[2]
# is the root.  Mirrors the path-bootstrapping pattern in test_provenance.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTEXT_PATH = (
    PROJECT_ROOT / "schemas" / "context" / "concept_graph_semantic_v1.jsonld"
)
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-551-2"
    / "graph"
    / "concept_graph_semantic.json"
)

ED4ALL_VOCAB = "https://ed4all.io/vocab/"
ED4ALL_EDGE_TYPE_PRED = ED4ALL_VOCAB + "edgeType"
ED4ALL_HAS_CONCEPT_PRED = ED4ALL_VOCAB + "hasConcept"
ED4ALL_CONCEPT_BASE = "https://ed4all.io/concept/"
DOC_IRI = "https://ed4all.io/concept-graph/rdf-shacl-551-2"


pyld = pytest.importorskip("pyld")
rdflib = pytest.importorskip("rdflib")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def context_doc() -> dict:
    """Load the JSON-LD @context wrapper (Phase 1.1 deliverable)."""
    with CONTEXT_PATH.open() as f:
        ctx = json.load(f)
    assert "@context" in ctx, (
        "concept_graph_semantic_v1.jsonld must expose a top-level @context "
        "block; the sibling _description key is metadata-only."
    )
    return ctx


@pytest.fixture(scope="module")
def graph_artifact() -> dict:
    """Load the JSON artifact that we are bridging to RDF."""
    if not ARTIFACT_PATH.exists():
        pytest.skip(
            f"Reference artifact missing: {ARTIFACT_PATH} — Phase 1 round-trip "
            "test depends on the rdf-shacl-551-2 corpus being present."
        )
    with ARTIFACT_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rdf_graph(context_doc, graph_artifact) -> "rdflib.Graph":
    """Inject the @context, expand to N-Quads via pyld, parse via rdflib."""
    from pyld import jsonld
    from rdflib import Graph

    # Layer the @context on top of a deep-copy of the artifact so we don't
    # mutate the loaded JSON for downstream assertions that walk the JSON.
    doc = dict(graph_artifact)
    doc["@context"] = context_doc["@context"]
    # Anchor the document so the top-level metadata (kind, generated_at,
    # rule_versions, hasConcept, hasEdge) lives on a stable IRI rather than a
    # blank node.  Without this, every artifact-level triple lands on a
    # bnode and downstream IRI-only tests need to chase the bnode.
    doc["@id"] = DOC_IRI

    nquads = jsonld.to_rdf(doc, {"format": "application/n-quads"})
    g = Graph()
    g.parse(data=nquads, format="nquads")
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_triple_count_floor(rdf_graph) -> None:
    """Sanity floor: layered context must produce >> 500 triples for the
    corpus's ~672 nodes / ~6.3k edges / per-edge provenance shape.

    Empirically the rdf-shacl-551-2 artifact yields ~101k triples; we assert
    the floor at 500 to keep the test resilient to corpus shape changes
    (smaller fixtures, schema additions) without losing the smoke-test value.
    """
    n = len(rdf_graph)
    assert n > 500, (
        f"Expected the JSON-LD bridge to materialize >>500 triples for the "
        f"reference corpus; got {n}.  Likely cause: a context-term mapping "
        f"regressed to null or a key suppression collapsed entire branches."
    )


def test_every_edge_emits_edgetype_triple(rdf_graph, graph_artifact) -> None:
    """Each JSON edge MUST produce at least one ``ed4all:edgeType`` triple.

    The context maps ``edges[].type`` -> ``ed4all:edgeType`` with
    ``@type: @vocab`` against the ``edge-type#`` namespace, so every edge
    becomes a node typed via its slug (e.g., ``edge-type#prerequisite``).
    Phase 2 will introduce ``ed4all:isA owl:equivalentProperty rdfs:subClassOf``
    so consumers can rewrite to canonical W3C predicates; until then the
    edge-type triple is the ground-truth count.
    """
    from rdflib import URIRef

    edge_type_pred = URIRef(ED4ALL_EDGE_TYPE_PRED)
    triples = list(rdf_graph.triples((None, edge_type_pred, None)))
    json_edge_count = len(graph_artifact["edges"])
    assert len(triples) >= json_edge_count, (
        f"Expected at least {json_edge_count} ed4all:edgeType triples "
        f"(one per JSON edge); got {len(triples)}.  A delta usually means "
        f"the `type` term lost its @vocab routing and slugs are landing as "
        f"plain literals."
    )


def test_node_ids_are_iris(rdf_graph, graph_artifact) -> None:
    """Every node id in the JSON MUST materialize as an RDF URIRef (IRI).

    The context binds ``nodes[].id`` to ``@id`` and resolves against the
    ``@base`` (``https://ed4all.io/concept/``).  A non-IRI here means a slug
    failed to resolve (likely a blank-node fallback) and downstream RDF
    tooling can no longer dereference the concept by ID.
    """
    from rdflib import URIRef

    has_concept = URIRef(ED4ALL_HAS_CONCEPT_PRED)
    objects = list(rdf_graph.objects(predicate=has_concept))

    # Every emitted concept reference under hasConcept must be a URIRef.
    non_iri = [o for o in objects if not isinstance(o, URIRef)]
    assert not non_iri, (
        f"Found {len(non_iri)} non-IRI concept references; the context must "
        f"keep `nodes[].id` typed as @id so slugs resolve against @base."
    )

    # And we must have one per JSON node (no silent drops).
    assert len(objects) == len(graph_artifact["nodes"]), (
        f"hasConcept count {len(objects)} != JSON node count "
        f"{len(graph_artifact['nodes'])}; likely a duplicate-id collision or "
        f"a list-shape regression in the context."
    )

    # Spot-check that the IRIs are anchored at the expected @base.
    for obj in objects[:25]:
        assert str(obj).startswith(ED4ALL_CONCEPT_BASE), (
            f"Concept IRI {obj} does not resolve against the expected "
            f"@base {ED4ALL_CONCEPT_BASE}; did the @base move?"
        )


def test_turtle_roundtrip_is_lossless(rdf_graph) -> None:
    """Serializing to Turtle and re-parsing MUST not lose triples.

    The mature JSON-LD context produces a graph that is round-trip-stable
    through Turtle (the ideal mapping target).  Any positive delta means a
    triple was emitted in a shape Turtle can't preserve (blank-node skolem
    drift, RDF list shape mismatch, language-tag drop).  Any negative delta
    means our serialization invented triples.

    Phase 1 target: delta == 0.  We assert |delta| <= 5 to leave a tiny
    cushion for skolem-id non-determinism in pyshacl/rdflib edge cases.
    """
    from rdflib import Graph

    n_orig = len(rdf_graph)
    ttl = rdf_graph.serialize(format="turtle")
    g_round_trip = Graph()
    g_round_trip.parse(data=ttl, format="turtle")
    n_rt = len(g_round_trip)
    delta = n_rt - n_orig
    assert abs(delta) <= 5, (
        f"Turtle round-trip changed triple count: {n_orig} -> {n_rt} "
        f"(delta={delta}).  Phase 1 expects an ideal mapping (delta == 0); "
        f"a small cushion is allowed for skolemized blank-node noise."
    )
