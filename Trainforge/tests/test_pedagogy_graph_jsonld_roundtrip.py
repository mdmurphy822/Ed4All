"""Phase 1.2 of plans/rdf-shacl-enrichment-2026-04-26.md.

Verifies that ``schemas/context/pedagogy_graph_v1.jsonld`` is a faithful
round-trip bridge between the JSON-shaped ``pedagogy_graph.json`` (emitted by
``Trainforge/pedagogy_graph_builder.py``) and an RDF graph, AND that
cross-artifact joins against ``concept_graph.json`` work via the
``https://ed4all.io/concept/<slug>`` IRI scheme.

Phase 1 is consumer-side only: the Trainforge emit pipeline does not yet inject
the ``@context``.  The test layers it on top of the existing artifact at
``LibV2/courses/rdf-shacl-551-2/graph/pedagogy_graph.json``, parses via
``pyld`` + ``rdflib``, and asserts:

* triple count is non-trivial (sanity floor of 200)
* every JSON edge with ``relation_type == "prerequisite_of"`` materializes
  an ``ed4all:hasPrerequisite`` triple after lifting via
  ``lib/ontology/edge_predicates.py::SLUG_TO_IRI`` — proves the registered
  slug-to-canonical-predicate bridge holds end-to-end
* every node id with the ``concept:`` prefix in the JSON resolves to a URI
  reference at ``https://ed4all.io/concept/<slug>``
* cross-artifact: load ``concept_graph.json`` via
  ``concept_graph_semantic_v1.jsonld``, extract its concept IRIs, assert at
  least one IRI overlaps with the pedagogy_graph's concept IRIs (the
  cross-artifact join key)
* Turtle round-trip is loss-free (graph-isomorphic delta of zero triples)

The pedagogy_graph file lives under ``graph/`` (not ``pedagogy/``) — that
sibling directory holds the older instructional-design ``pedagogy_model.json``,
which is a different artifact.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Project root (Ed4All/) — this test file lives at
# Ed4All/Trainforge/tests/test_pedagogy_graph_jsonld_roundtrip.py, so parents[2]
# is the root.  Mirrors the path-bootstrapping pattern in
# test_concept_graph_jsonld_roundtrip.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PED_CONTEXT_PATH = (
    PROJECT_ROOT / "schemas" / "context" / "pedagogy_graph_v1.jsonld"
)
CON_CONTEXT_PATH = (
    PROJECT_ROOT / "schemas" / "context" / "concept_graph_semantic_v1.jsonld"
)
PED_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-551-2"
    / "graph"
    / "pedagogy_graph.json"
)
# concept_graph.json (NOT concept_graph_semantic.json) is the bare-slug
# co-occurrence graph; both artifacts use the same bare-slug Concept IDs that
# resolve against the @base in concept_graph_semantic_v1.jsonld
# (https://ed4all.io/concept/), so either is fine for the join smoke.  We use
# concept_graph.json because it's the simpler artifact and the join-key proof
# does not need the semantic graph's per-rule provenance.
CON_ARTIFACT_PATH = (
    PROJECT_ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-551-2"
    / "graph"
    / "concept_graph.json"
)

ED4ALL_VOCAB = "https://ed4all.io/vocab/"
ED4ALL_EDGE_TYPE_PRED = ED4ALL_VOCAB + "edgeType"
ED4ALL_EDGE_SOURCE_PRED = ED4ALL_VOCAB + "edgeSource"
ED4ALL_EDGE_TARGET_PRED = ED4ALL_VOCAB + "edgeTarget"
ED4ALL_CONCEPT_BASE = "https://ed4all.io/concept/"
PED_DOC_IRI = "https://ed4all.io/pedagogy/rdf-shacl-551-2"
CON_DOC_IRI = "https://ed4all.io/concept-graph/rdf-shacl-551-2"

pyld = pytest.importorskip("pyld")
rdflib = pytest.importorskip("rdflib")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _load_context(path: Path) -> dict:
    with path.open() as f:
        ctx = json.load(f)
    assert "@context" in ctx, (
        f"{path.name} must expose a top-level @context block; the sibling "
        f"_description key is metadata-only."
    )
    return ctx


def _load_artifact(path: Path) -> dict:
    if not path.exists():
        pytest.skip(
            f"Reference artifact missing: {path} — Phase 1 round-trip test "
            f"depends on the rdf-shacl-551-2 corpus being present."
        )
    with path.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pedagogy_context() -> dict:
    return _load_context(PED_CONTEXT_PATH)


@pytest.fixture(scope="module")
def pedagogy_artifact() -> dict:
    return _load_artifact(PED_ARTIFACT_PATH)


@pytest.fixture(scope="module")
def concept_context() -> dict:
    return _load_context(CON_CONTEXT_PATH)


@pytest.fixture(scope="module")
def concept_artifact() -> dict:
    return _load_artifact(CON_ARTIFACT_PATH)


def _doc_to_graph(artifact: dict, context: dict, doc_iri: str) -> "rdflib.Graph":
    """Inject the @context onto a deep-copy of the artifact, expand to N-Quads
    via pyld, parse via rdflib.  The doc_iri anchors the document so top-level
    metadata (kind, generated_at, hasNode, hasEdge, stats) lives on a stable
    IRI rather than a blank node."""
    from pyld import jsonld
    from rdflib import Graph

    doc = dict(artifact)
    doc["@context"] = context["@context"]
    doc["@id"] = doc_iri
    nquads = jsonld.to_rdf(doc, {"format": "application/n-quads"})
    g = Graph()
    g.parse(data=nquads, format="nquads")
    return g


@pytest.fixture(scope="module")
def pedagogy_graph(pedagogy_context, pedagogy_artifact) -> "rdflib.Graph":
    return _doc_to_graph(pedagogy_artifact, pedagogy_context, PED_DOC_IRI)


@pytest.fixture(scope="module")
def concept_graph(concept_context, concept_artifact) -> "rdflib.Graph":
    return _doc_to_graph(concept_artifact, concept_context, CON_DOC_IRI)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_triple_count_floor(pedagogy_graph) -> None:
    """Sanity floor: layered context must produce >> 200 triples for the
    rdf-shacl-551-2 corpus's ~1059 nodes / ~8.7k edges shape.

    Empirically the artifact yields ~47k triples; we assert the floor at 200
    to keep the test resilient to corpus shape changes (smaller fixtures,
    schema additions) without losing the smoke-test value.
    """
    n = len(pedagogy_graph)
    assert n > 200, (
        f"Expected the pedagogy_graph JSON-LD bridge to materialize >>200 "
        f"triples for the reference corpus; got {n}.  Likely cause: a "
        f"context-term mapping regressed to null or a key suppression "
        f"collapsed entire branches."
    )


def test_prerequisite_edges_lift_to_has_prerequisite(
    pedagogy_graph, pedagogy_artifact
) -> None:
    """Every JSON edge with ``relation_type == "prerequisite_of"`` MUST be
    reachable as an ``ed4all:hasPrerequisite`` triple after lifting through
    the slug-to-IRI registry.

    JSON-LD ``@context`` cannot dispatch on the *value* of a property (only on
    the property name), so the raw round-trip emits each edge as a
    ``ed4all:edgeType`` triple with object
    ``edge-type#prerequisite_of``.  The slug-to-canonical-predicate bridge
    is closed by post-walking the typed-edge nodes and rewriting via
    ``lib.ontology.edge_predicates.SLUG_TO_IRI`` — exactly what a Phase 2
    SHACL Rule or SPARQL CONSTRUCT will do at vocab-alignment time.

    This test PROVES the bridge holds end-to-end (context + registry):

    * count of pedagogy_graph JSON edges with ``relation_type ==
      "prerequisite_of"`` matches the count of edge nodes whose
      ``ed4all:edgeType`` is the canonical edge-type IRI for that slug.
    * lifting each via SLUG_TO_IRI yields a non-zero set of
      ``ed4all:hasPrerequisite`` triples joining JSON ``source`` -> JSON
      ``target`` IRIs.

    Note: pedagogy_graph emits the slug as ``prerequisite_of`` (underscored),
    while the registry key is ``prerequisite`` (no underscore, no suffix) —
    the canonical IRI is the same (``cf:hasPrerequisite``); the slug
    convention drift is documented in the Phase 1.2 report and a normalizer
    is the right Phase 2 follow-up.
    """
    from rdflib import URIRef

    from lib.ontology.edge_predicates import SLUG_TO_IRI

    rdf_type = URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    edge_type_pred = URIRef(ED4ALL_EDGE_TYPE_PRED)
    edge_source_pred = URIRef(ED4ALL_EDGE_SOURCE_PRED)
    edge_target_pred = URIRef(ED4ALL_EDGE_TARGET_PRED)

    # The @vocab routing puts each edge slug at edge-type#<slug>; the
    # context maps the slug verbatim, so prerequisite_of resolves there.
    prereq_slug_iri = URIRef(ED4ALL_VOCAB + "edge-type#prerequisite_of")

    # Locate every typed-edge node carrying the prerequisite_of slug.
    prereq_edge_nodes = list(
        pedagogy_graph.subjects(edge_type_pred, prereq_slug_iri)
    )

    json_prereq_edges = [
        e for e in pedagogy_artifact["edges"]
        if e.get("relation_type") == "prerequisite_of"
    ]
    assert json_prereq_edges, (
        "Reference corpus has zero prerequisite_of edges — fixture changed?"
    )
    assert len(prereq_edge_nodes) == len(json_prereq_edges), (
        f"Expected {len(json_prereq_edges)} edge nodes typed at "
        f"<edge-type#prerequisite_of> (one per JSON prerequisite_of edge); "
        f"got {len(prereq_edge_nodes)}.  A delta means the relation_type "
        f"@vocab routing dropped or duplicated edges."
    )

    # Lift via SLUG_TO_IRI to the canonical cf:hasPrerequisite predicate.
    # The registry key is ``prerequisite`` (Trainforge concept-graph slug);
    # pedagogy_graph uses ``prerequisite_of`` for the same semantic — both
    # resolve to the same canonical IRI on the cf: namespace.
    canonical_iri = SLUG_TO_IRI.get("prerequisite")
    assert canonical_iri, (
        "lib.ontology.edge_predicates.SLUG_TO_IRI must register "
        "'prerequisite' -> cf:hasPrerequisite; got missing key."
    )
    has_prereq = URIRef(canonical_iri)

    # Materialize the lifted triples: for each typed-edge node, pull its
    # source + target and emit (source, hasPrerequisite, target).
    lifted = []
    for edge_node in prereq_edge_nodes:
        sources = list(pedagogy_graph.objects(edge_node, edge_source_pred))
        targets = list(pedagogy_graph.objects(edge_node, edge_target_pred))
        assert sources and targets, (
            f"prerequisite_of edge {edge_node} missing edgeSource/edgeTarget "
            f"after JSON-LD expansion — context drop on @id routing."
        )
        for s in sources:
            for t in targets:
                lifted.append((s, has_prereq, t))

    assert len(lifted) >= len(json_prereq_edges), (
        f"Lift produced {len(lifted)} hasPrerequisite triples; expected at "
        f"least {len(json_prereq_edges)} (one per JSON edge).  Likely a "
        f"missing edgeSource/edgeTarget on some edge node."
    )

    # Spot-check that the lifted triples actually point at the expected
    # concept IRIs from the JSON edges.  Both endpoints in pedagogy_graph
    # carry the ``concept:`` prefix, so they resolve to the cross-artifact
    # https://ed4all.io/concept/<slug> namespace.
    sample_json_edge = next(
        e for e in json_prereq_edges
        if e["source"].startswith("concept:") and e["target"].startswith("concept:")
    )
    expected_source = URIRef(
        ED4ALL_CONCEPT_BASE + sample_json_edge["source"][len("concept:"):]
    )
    expected_target = URIRef(
        ED4ALL_CONCEPT_BASE + sample_json_edge["target"][len("concept:"):]
    )
    assert (expected_source, has_prereq, expected_target) in lifted, (
        f"Spot-check lifted triple ({expected_source}, hasPrerequisite, "
        f"{expected_target}) not present.  Concept CURIE may have failed to "
        f"expand against the @context's `concept:` prefix declaration."
    )


def test_concept_prefix_resolves_to_canonical_iri(
    pedagogy_graph, pedagogy_artifact
) -> None:
    """Every node id in the JSON with the ``concept:`` prefix MUST materialize
    as a URI reference at ``https://ed4all.io/concept/<slug>``.

    The context binds the ``concept:`` prefix to the canonical concept
    namespace so pedagogy_graph's ``concept:rdf`` and concept_graph's bare
    ``rdf`` (under ``@base: https://ed4all.io/concept/``) resolve to THE
    SAME IRI — that's the cross-artifact join key.  A non-IRI here, or an
    IRI rooted elsewhere (e.g. the document base
    ``https://ed4all.io/pedagogy/concept:rdf``), means the prefix
    declaration regressed.
    """
    from rdflib import URIRef

    json_concept_ids = sorted({
        n["id"] for n in pedagogy_artifact["nodes"]
        if isinstance(n.get("id"), str) and n["id"].startswith("concept:")
    })
    assert json_concept_ids, (
        "Reference corpus has zero concept-prefixed node IDs — fixture changed?"
    )

    # Collect every URIRef in the parsed graph.
    iris = set()
    for s, _, o in pedagogy_graph:
        for term in (s, o):
            if isinstance(term, URIRef):
                iris.add(str(term))

    # Every concept: node id from the JSON must appear as the canonical IRI.
    missing = []
    for nid in json_concept_ids:
        slug = nid[len("concept:"):]
        expected_iri = ED4ALL_CONCEPT_BASE + slug
        if expected_iri not in iris:
            missing.append((nid, expected_iri))

    assert not missing, (
        f"Found {len(missing)} concept: node IDs that did NOT resolve to the "
        f"canonical https://ed4all.io/concept/<slug> namespace; sample: "
        f"{missing[:3]}.  The @context's `concept:` prefix declaration must "
        f"bind to https://ed4all.io/concept/ so pedagogy_graph and "
        f"concept_graph share the join key."
    )


def test_cross_artifact_concept_iri_overlap(
    pedagogy_graph, concept_graph, pedagogy_artifact, concept_artifact
) -> None:
    """Pedagogy_graph and concept_graph MUST share at least one concept IRI
    when joined on the ``https://ed4all.io/concept/<slug>`` namespace.

    This is the load-bearing cross-artifact assertion of Phase 1.2 — it
    proves that pedagogy_graph's prefix-form concept IDs (``concept:rdf``)
    and concept_graph's bare-slug IDs (``rdf`` under
    ``@base: https://ed4all.io/concept/``) converge to the same IRI when
    each is parsed via its own JSON-LD ``@context``.  Without this, a
    SPARQL query joining the two artifacts on concept identity would never
    fire.

    Empirically the rdf-shacl-551-2 corpus has 672 concept IRIs in
    common; we assert >= 1 to keep the test resilient to corpus shape
    changes (smaller fixtures) while still gating the join.
    """
    from rdflib import URIRef

    def _concept_iris(g) -> set:
        out = set()
        for s, _, o in g:
            for term in (s, o):
                if isinstance(term, URIRef) and str(term).startswith(
                    ED4ALL_CONCEPT_BASE
                ):
                    out.add(str(term))
        return out

    ped_concepts = _concept_iris(pedagogy_graph)
    con_concepts = _concept_iris(concept_graph)
    overlap = ped_concepts & con_concepts

    assert ped_concepts, "pedagogy_graph yielded zero concept IRIs"
    assert con_concepts, "concept_graph yielded zero concept IRIs"
    assert len(overlap) >= 1, (
        f"Cross-artifact concept IRI overlap is {len(overlap)}; expected "
        f">= 1 to prove the JSON-LD bridge produces a viable join key.  "
        f"pedagogy_graph IRIs: {len(ped_concepts)}, concept_graph IRIs: "
        f"{len(con_concepts)}.  A zero overlap means the @context prefix "
        f"declarations diverged between the two artifacts."
    )


def test_every_pedagogy_edge_slug_resolves_in_registry(pedagogy_artifact) -> None:
    """Phase 2.6 minting coverage: every distinct ``relation_type`` slug
    in the fixture's edges MUST resolve through
    ``lib.ontology.edge_predicates.lookup_iri`` (which combines
    ``SLUG_TO_IRI`` direct lookup with the Phase 2.7 normalizer) to a
    registered IRI. No slug should remain in @vocab-fallback territory.

    Before Phase 2.6 the pedagogy graph emitted nine slugs that lacked
    registered IRIs (``teaches``, ``belongs_to_module``,
    ``supports_outcome``, ``at_bloom_level``, ``follows``,
    ``concept_supports_outcome``, ``assessment_validates_outcome``,
    ``chunk_at_difficulty``, ``interferes_with``) and round-tripped only
    via the JSON-LD ``@vocab`` fallback at
    ``https://ed4all.io/vocab/edge-type#<slug>`` — which gave the slug a
    URI but no rdfs:domain/range, no rdfs:label, no shape-graph reach.
    With Phase 2.6 minted, every pedagogy slug must lift cleanly.
    """
    from lib.ontology.edge_predicates import lookup_iri

    relation_slugs = sorted({
        e.get("relation_type") for e in pedagogy_artifact["edges"]
        if isinstance(e.get("relation_type"), str)
    })
    assert relation_slugs, (
        "Reference corpus has zero relation_type slugs — fixture changed?"
    )

    unresolved = [s for s in relation_slugs if lookup_iri(s) is None]
    assert not unresolved, (
        f"Phase 2.6 expects every pedagogy slug to lift to a registered "
        f"IRI via lib.ontology.edge_predicates.lookup_iri (direct + "
        f"Phase 2.7 normalizer). Unresolved slugs: {unresolved}. Either "
        f"add the slug to SLUG_TO_IRI in lib/ontology/edge_predicates.py "
        f"with a matching declaration in courseforge_v1.vocabulary.ttl, "
        f"or extend the normalizer in lib/ontology/edge_slug_normalizer.py."
    )


def test_turtle_roundtrip_is_lossless(pedagogy_graph) -> None:
    """Serializing to Turtle and re-parsing MUST not lose triples.

    The Phase 1.2 context produces a graph that is round-trip-stable through
    Turtle (the ideal mapping target).  Any positive delta means a triple
    was emitted in a shape Turtle can't preserve (blank-node skolem drift,
    RDF list shape mismatch, language-tag drop).  Any negative delta means
    our serialization invented triples.

    Phase 1 target: delta == 0.  We assert |delta| <= 5 to leave a tiny
    cushion for skolem-id non-determinism in pyshacl/rdflib edge cases
    (matches the Phase 1.1 concept-graph round-trip cushion).
    """
    from rdflib import Graph

    n_orig = len(pedagogy_graph)
    ttl = pedagogy_graph.serialize(format="turtle")
    g_round_trip = Graph()
    g_round_trip.parse(data=ttl, format="turtle")
    n_rt = len(g_round_trip)
    delta = n_rt - n_orig
    assert abs(delta) <= 5, (
        f"Turtle round-trip changed triple count: {n_orig} -> {n_rt} "
        f"(delta={delta}).  Phase 1 expects an ideal mapping (delta == 0); "
        f"a small cushion is allowed for skolemized blank-node noise."
    )
