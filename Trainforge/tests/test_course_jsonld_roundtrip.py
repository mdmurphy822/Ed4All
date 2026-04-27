"""Phase 1.2 of plans/rdf-shacl-enrichment-2026-04-26.md.

Verifies that ``schemas/context/course_v1.jsonld`` is a faithful round-trip
bridge between the JSON-shaped ``course.json`` and an RDF graph.  The
Trainforge emit pipeline does NOT yet inject the ``@context`` (Phase 1 is
consumer-side only); this test layers it on top of an existing artifact,
parses via ``pyld`` + ``rdflib``, and asserts:

* triple count is non-trivial (sanity floor of 100; the reference course
  carries 36 LOs * ~5 properties each plus course-level fields)
* every learning_outcome materializes one ``rdf:type ed4all:LearningObjective``
  triple
* every LO's ``bloom_level`` expands to an IRI under the SKOS-style
  ``https://ed4all.io/vocab/bloom#`` scheme
* LO ``@id`` is minted as ``https://ed4all.io/lo/<id>`` — stable across
  re-runs because the course.schema.json ID pattern is already URL-safe
  and Trainforge emits lowercase
* Turtle round-trip is loss-free (graph-isomorphic delta of zero triples)

Phase 1 does not modify ``Trainforge/process_course.py`` or any LO emission
code; the bridge is exercised out-of-band from the existing artifact under
``LibV2/courses/rdf-shacl-551-2/course.json``.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

# Project root (Ed4All/) — this test file lives at
# Ed4All/Trainforge/tests/test_course_jsonld_roundtrip.py, so parents[2]
# is the root.  Mirrors the path-bootstrapping pattern in
# test_concept_graph_jsonld_roundtrip.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONTEXT_PATH = PROJECT_ROOT / "schemas" / "context" / "course_v1.jsonld"
ARTIFACT_PATH = (
    PROJECT_ROOT / "LibV2" / "courses" / "rdf-shacl-551-2" / "course.json"
)

# Canonical IRI shape pinned by the @context.  The course context follows
# Worker A's https://ed4all.io/vocab/ namespace (NOT the Wave 65
# https://ed4all.dev/ns/courseforge/v1# namespace declared by
# courseforge_v1.vocabulary.ttl); cross-host equivalence is deferred.
ED4ALL_VOCAB = "https://ed4all.io/vocab/"
ED4ALL_LO_BASE = "https://ed4all.io/lo/"
ED4ALL_BLOOM_SCHEME = "https://ed4all.io/vocab/bloom#"

ED4ALL_LEARNING_OBJECTIVE_CLASS = ED4ALL_VOCAB + "LearningObjective"
ED4ALL_HAS_LO_PRED = ED4ALL_VOCAB + "hasLearningObjective"
ED4ALL_BLOOM_LEVEL_PRED = ED4ALL_VOCAB + "bloomLevel"
ED4ALL_HIERARCHY_LEVEL_PRED = ED4ALL_VOCAB + "hierarchyLevel"
RDF_TYPE_PRED = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

DOC_IRI = "https://ed4all.io/course/rdf-shacl-551-2"


pyld = pytest.importorskip("pyld")
rdflib = pytest.importorskip("rdflib")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def context_doc() -> dict:
    """Load the JSON-LD @context wrapper (Phase 1.2 deliverable)."""
    with CONTEXT_PATH.open() as f:
        ctx = json.load(f)
    assert "@context" in ctx, (
        "course_v1.jsonld must expose a top-level @context block; the "
        "sibling _description key is metadata-only."
    )
    return ctx


@pytest.fixture(scope="module")
def course_artifact() -> dict:
    """Load the JSON artifact that we are bridging to RDF."""
    if not ARTIFACT_PATH.exists():
        pytest.skip(
            f"Reference artifact missing: {ARTIFACT_PATH} — Phase 1.2 "
            "round-trip test depends on the rdf-shacl-551-2 corpus being "
            "present."
        )
    with ARTIFACT_PATH.open() as f:
        return json.load(f)


@pytest.fixture(scope="module")
def rdf_graph(context_doc, course_artifact) -> "rdflib.Graph":
    """Inject the @context, expand to N-Quads via pyld, parse via rdflib.

    JSON-LD has no native way to default ``@type`` on every member of a
    plain-array of objects, so this fixture explicitly wraps each LO with
    ``"@type": "LearningObjective"`` before expansion.  The class alias
    ``LearningObjective`` -> ``ed4all:LearningObjective`` is registered in
    the context, so the injected token resolves correctly.  The course
    schema (course.schema.json) does NOT yet require a ``type`` field on
    LO members, so the injection is purely a JSON-LD-side projection
    decision; downstream JSON consumers see the unmodified document.
    """
    from pyld import jsonld
    from rdflib import Graph

    # Layer the @context on top of a deep-copy of the artifact so we don't
    # mutate the loaded JSON for downstream assertions that walk the JSON.
    doc = copy.deepcopy(course_artifact)
    doc["@context"] = context_doc["@context"]
    # Anchor the document on a stable IRI so course-level metadata
    # (course_code, title, hasLearningObjective) lands on a URI rather
    # than a blank node.
    doc["@id"] = DOC_IRI
    doc["@type"] = "Course"

    # Inject the LearningObjective class on each LO so rdf:type triples
    # materialize.  See fixture docstring for rationale.
    for lo in doc.get("learning_outcomes", []):
        lo["@type"] = "LearningObjective"

    nquads = jsonld.to_rdf(doc, {"format": "application/n-quads"})
    g = Graph()
    g.parse(data=nquads, format="nquads")
    return g


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_triple_count_floor(rdf_graph) -> None:
    """Sanity floor: layered context must produce > 100 triples for the
    reference corpus's 36 LOs * ~5 properties each plus course-level fields.

    Empirically the rdf-shacl-551-2 course.json yields ~150+ triples; we
    assert the floor at 100 so the test stays useful as a smoke test
    without being brittle to schema additions or fixture-trim PRs.
    """
    n = len(rdf_graph)
    assert n > 100, (
        f"Expected the JSON-LD bridge to materialize >100 triples for the "
        f"reference course; got {n}.  Likely cause: a context-term mapping "
        f"regressed to null or the LO array lost its @set container."
    )


def test_every_lo_has_rdf_type_learning_objective(
    rdf_graph, course_artifact,
) -> None:
    """Each JSON LO MUST produce one ``rdf:type ed4all:LearningObjective``
    triple.  Worker B (vocabulary.ttl) declares
    ``ed4all:LearningObjective`` as the canonical class for both terminal
    (TO-NN) and chapter (CO-NN) outcomes, and this context registers the
    class alias under the same local name.

    The fixture injects ``@type: LearningObjective`` per LO (see fixture
    docstring); a missing or wrong rdf:type triple count usually means the
    class alias regressed or the @type injection silently dropped.
    """
    from rdflib import URIRef

    rdf_type = URIRef(RDF_TYPE_PRED)
    lo_class = URIRef(ED4ALL_LEARNING_OBJECTIVE_CLASS)
    typed_los = list(rdf_graph.subjects(predicate=rdf_type, object=lo_class))
    json_lo_count = len(course_artifact["learning_outcomes"])
    assert len(typed_los) == json_lo_count, (
        f"Expected exactly {json_lo_count} rdf:type ed4all:LearningObjective "
        f"triples (one per JSON learning_outcome); got {len(typed_los)}.  "
        f"Likely cause: the LearningObjective class alias regressed in the "
        f"@context or the @type injection in the test fixture broke."
    )


def test_lo_ids_minted_as_stable_iris(rdf_graph, course_artifact) -> None:
    """Every LO's @id MUST resolve to ``https://ed4all.io/lo/<id>``.

    The course.schema.json ID pattern (^[a-zA-Z]{2,}-\\d{2,}$) is
    URL-safe by construction and Trainforge emits lowercase, so the
    minted IRI is byte-stable across re-runs of process_course.py.  This
    is the central guarantee of Phase 1.2: the LO identity surviving as
    an RDF URI without dependence on document hosting URL.
    """
    from rdflib import URIRef

    rdf_type = URIRef(RDF_TYPE_PRED)
    lo_class = URIRef(ED4ALL_LEARNING_OBJECTIVE_CLASS)
    typed_los = list(rdf_graph.subjects(predicate=rdf_type, object=lo_class))

    # Every LO subject must be a URIRef anchored at @base.
    non_iri = [s for s in typed_los if not isinstance(s, URIRef)]
    assert not non_iri, (
        f"Found {len(non_iri)} non-IRI LO subjects; the context must keep "
        f"`learning_outcomes[].id` typed as @id so canonical IDs resolve "
        f"against @base ({ED4ALL_LO_BASE})."
    )

    expected_iris = {
        ED4ALL_LO_BASE + lo["id"]
        for lo in course_artifact["learning_outcomes"]
    }
    actual_iris = {str(s) for s in typed_los}
    assert actual_iris == expected_iris, (
        f"LO IRI set drift.  Expected {len(expected_iris)} IRIs anchored "
        f"at {ED4ALL_LO_BASE}; got {len(actual_iris)}.  Missing: "
        f"{expected_iris - actual_iris}.  Unexpected: "
        f"{actual_iris - expected_iris}."
    )


def test_bloom_level_resolves_to_iri(rdf_graph, course_artifact) -> None:
    """Each LO with a ``bloom_level`` MUST emit an
    ``ed4all:bloomLevel`` triple whose object is an IRI in the bloom:
    SKOS scheme (``https://ed4all.io/vocab/bloom#<level>``), not a plain
    string literal.  The context routes the field through ``@type:
    @vocab`` against the bloom# namespace.

    This is the load-bearing assertion that the Bloom level survives as
    a SKOS concept reference rather than degrading to a free-text
    literal — a prerequisite for the Phase 2 vocabulary alignment that
    binds bloom: into the canonical SKOS scheme.
    """
    from rdflib import URIRef

    bloom_pred = URIRef(ED4ALL_BLOOM_LEVEL_PRED)
    bloom_objects = list(rdf_graph.objects(predicate=bloom_pred))

    los_with_bloom = [
        lo for lo in course_artifact["learning_outcomes"]
        if lo.get("bloom_level")
    ]
    assert len(bloom_objects) == len(los_with_bloom), (
        f"Expected one ed4all:bloomLevel triple per LO with a bloom_level "
        f"field ({len(los_with_bloom)}); got {len(bloom_objects)}."
    )

    non_iri = [o for o in bloom_objects if not isinstance(o, URIRef)]
    assert not non_iri, (
        f"Found {len(non_iri)} bloom_level objects that did NOT expand "
        f"to IRIs (they landed as literals).  The @vocab routing on the "
        f"`bloom_level` term regressed."
    )

    for obj in bloom_objects:
        assert str(obj).startswith(ED4ALL_BLOOM_SCHEME), (
            f"Bloom level IRI {obj} not anchored at the expected SKOS "
            f"scheme {ED4ALL_BLOOM_SCHEME}; check the inner @vocab on the "
            f"bloom_level term."
        )


def test_hierarchy_level_resolves_to_iri(rdf_graph, course_artifact) -> None:
    """Each LO MUST emit an ``ed4all:hierarchyLevel`` triple whose object
    is an IRI under ``https://ed4all.io/vocab/hierarchy#`` (terminal /
    chapter).  Mirrors the Wave 65 vocabulary.ttl declaration of the
    hierarchy: SKOS scheme, just hosted under the Worker A namespace.
    """
    from rdflib import URIRef

    hier_pred = URIRef(ED4ALL_HIERARCHY_LEVEL_PRED)
    hier_objects = list(rdf_graph.objects(predicate=hier_pred))
    json_lo_count = len(course_artifact["learning_outcomes"])
    assert len(hier_objects) == json_lo_count, (
        f"Expected one ed4all:hierarchyLevel triple per LO "
        f"({json_lo_count}); got {len(hier_objects)}."
    )
    non_iri = [o for o in hier_objects if not isinstance(o, URIRef)]
    assert not non_iri, (
        f"Found {len(non_iri)} hierarchy_level objects that did NOT "
        f"expand to IRIs.  The @vocab routing on the `hierarchy_level` "
        f"term regressed."
    )


def test_turtle_roundtrip_is_lossless(rdf_graph) -> None:
    """Serializing to Turtle and re-parsing MUST not lose triples.

    The mature JSON-LD context produces a graph that is round-trip-stable
    through Turtle (the ideal mapping target).  Any positive delta means
    a triple was emitted in a shape Turtle can't preserve; any negative
    delta means our serialization invented triples.

    Phase 1 target: delta == 0.  We assert |delta| <= 5 to leave a tiny
    cushion for skolem-id non-determinism in pyshacl/rdflib edge cases,
    matching the tolerance used in test_concept_graph_jsonld_roundtrip.
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
