"""Wave 87 — chunk_v4_v1.jsonld `_phase2_followup.predicates[]` minted out.

Phase 1.2 (schemas/context/chunk_v4_v1.jsonld) introduced a chunk-level
JSON-LD @context that referenced 40 ed4all: IRIs not yet formally declared
in the Phase 2.1 vocabulary (schemas/context/courseforge_v1.vocabulary.ttl).
Those IRIs were tracked in the JSON file's `_phase2_followup.predicates[]`
block until a wave could mint them with rdfs:domain / rdfs:range axioms.

Wave 87 closed the gap: 33 chunk-structural predicates plus 2 new anchor
classes (KeyTerm, SourceReference) were declared in the Turtle vocabulary
so RDFS reasoning over chunks materialised through the Phase 1.2 bridge
finally gets endpoint type entailments. Four IRIs in the original list
were already declared at Phase 2.1 / Wave 60 / Wave 65 (Misconception,
TargetedConcept, hasMisconception, correction) and are also covered here
to prevent regression. One IRI (metadataTrace) remains deferred — its
value is an open-shape diagnostic dict and is now tracked under
`_phase3_followup` for a later wave that can pair the declaration with
an emit-side schema lockdown.

This test parametrises the 35 minted IRIs against the Turtle vocabulary
graph and asserts each one carries the rdfs:domain + rdfs:range triples
that downstream RDFS / OWL-RL reasoners need.

Style template: schemas/tests/test_courseforge_vocabulary.py
::test_minted_edge_predicates_declared_with_domain_and_range.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

rdflib = pytest.importorskip(
    "rdflib", reason="rdflib is required for vocabulary parsing tests."
)
from rdflib import Graph, Namespace  # noqa: E402
from rdflib.namespace import OWL, RDF, RDFS  # noqa: E402

_VOCAB_PATH = (
    _PROJECT_ROOT / "schemas" / "context" / "courseforge_v1.vocabulary.ttl"
)

# The vocabulary file declares ed4all: under https://ed4all.dev/ns/courseforge/v1#
# (the formal RDFS/OWL surface). The chunk_v4_v1.jsonld @context routes the
# same local names through https://ed4all.io/vocab/ at JSON-LD round-trip
# time; the cross-namespace bridge lives in schemas/context/aliases.ttl
# (Phase 2.5). The vocabulary tests target the .dev/ns surface directly,
# matching the existing Phase 2.1 / 2.6 test conventions.
ED4ALL = Namespace("https://ed4all.dev/ns/courseforge/v1#")
SCHEMA = Namespace("http://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")
SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
XSD = Namespace("http://www.w3.org/2001/XMLSchema#")


@pytest.fixture(scope="module")
def vocab_graph() -> Graph:
    g = Graph()
    g.parse(_VOCAB_PATH, format="turtle")
    return g


# ---------------------------------------------------------------------- #
# Wave 87 anchor classes
# ---------------------------------------------------------------------- #

_WAVE_87_ANCHOR_CLASSES = [
    # (class IRI, expected superclass — None if no rdfs:subClassOf check)
    (ED4ALL.KeyTerm, SCHEMA.DefinedTerm),
    (ED4ALL.SourceReference, PROV.Entity),
    # Already declared at Phase 2.1 / Wave 60 — included here because the
    # original chunk_v4_v1.jsonld _phase2_followup list flagged them and
    # Wave 87 removed them from that list. Regression-guard their continued
    # presence in the vocabulary.
    (ED4ALL.Misconception, None),
    (ED4ALL.TargetedConcept, None),
]


@pytest.mark.parametrize("class_iri,expected_superclass", _WAVE_87_ANCHOR_CLASSES)
def test_wave_87_anchor_class_declared(vocab_graph, class_iri, expected_superclass):
    """Wave 87 introduces two new anchor classes (KeyTerm, SourceReference)
    plus regression-guards two pre-existing ones (Misconception,
    TargetedConcept) that the chunk_v4_v1.jsonld _phase2_followup list
    flagged. Each must be declared as rdfs:Class / owl:Class in the
    vocabulary."""
    assert (class_iri, RDF.type, RDFS.Class) in vocab_graph or (
        class_iri,
        RDF.type,
        OWL.Class,
    ) in vocab_graph, (
        f"Wave 87 anchor class {class_iri} must be declared as "
        f"rdfs:Class / owl:Class in courseforge_v1.vocabulary.ttl"
    )
    if expected_superclass is not None:
        assert (class_iri, RDFS.subClassOf, expected_superclass) in vocab_graph, (
            f"Wave 87 class {class_iri} must declare rdfs:subClassOf "
            f"{expected_superclass} so non-ed4all: tooling can recognise "
            f"the type without learning our namespace."
        )


# ---------------------------------------------------------------------- #
# Wave 87 minted predicates
#
# Every predicate listed here was either (a) flagged in
# chunk_v4_v1.jsonld's `_phase2_followup.predicates[]` block and minted
# in this wave, or (b) already declared at an earlier phase but flagged
# in that list — included here for regression coverage.
#
# Tuple shape: (predicate IRI, expected rdfs:domain, expected rdfs:range).
# ---------------------------------------------------------------------- #

_WAVE_87_MINTED_PREDICATES = [
    # ---- Identity / version / body / surface ----
    (ED4ALL.schemaVersion, ED4ALL.Chunk, XSD.string),
    (ED4ALL.chunkType, ED4ALL.Chunk, SKOS.Concept),
    (ED4ALL.html, ED4ALL.Chunk, XSD.string),
    (ED4ALL.summary, ED4ALL.Chunk, XSD.string),
    (ED4ALL.retrievalText, ED4ALL.Chunk, XSD.string),
    # ---- Sequence / position ----
    (ED4ALL.followsChunk, ED4ALL.Chunk, ED4ALL.Chunk),
    (ED4ALL.positionInModule, ED4ALL.Chunk, XSD.nonNegativeInteger),
    # ---- Module / lesson scope ----
    (ED4ALL.moduleId, ED4ALL.Chunk, XSD.string),
    (ED4ALL.moduleTitle, ED4ALL.Chunk, XSD.string),
    (ED4ALL.lessonId, ED4ALL.Chunk, XSD.string),
    (ED4ALL.lessonTitle, ED4ALL.Chunk, XSD.string),
    (ED4ALL.resourceType, ED4ALL.Chunk, XSD.string),
    (ED4ALL.sectionHeading, ED4ALL.Chunk, XSD.string),
    # ---- Audit-trail (HTML element resolution) ----
    (ED4ALL.htmlXPath, ED4ALL.Chunk, XSD.string),
    (ED4ALL.charSpan, ED4ALL.Chunk, XSD.integer),
    (ED4ALL.itemPath, ED4ALL.Chunk, XSD.string),
    # ---- Concept / LO ties ----
    (ED4ALL.hasConceptTag, ED4ALL.Chunk, ED4ALL.Concept),
    (ED4ALL.learningOutcomeRef, ED4ALL.Chunk, ED4ALL.LearningObjective),
    # ---- Difficulty + token / word counts ----
    (ED4ALL.difficulty, ED4ALL.Chunk, ED4ALL.DifficultyLevel),
    (ED4ALL.tokensEstimate, ED4ALL.Chunk, XSD.nonNegativeInteger),
    (ED4ALL.wordCount, ED4ALL.Chunk, XSD.nonNegativeInteger),
    # ---- Bloom-level provenance / secondary ----
    (ED4ALL.bloomLevelSource, ED4ALL.Chunk, XSD.string),
    (ED4ALL.bloomLevelSecondary, ED4ALL.Chunk, SKOS.Concept),
    # ---- Chunk -> structured sub-entities ----
    (ED4ALL.hasKeyTerm, ED4ALL.Chunk, ED4ALL.KeyTerm),
    (ED4ALL.hasTargetedConcept, ED4ALL.Chunk, ED4ALL.TargetedConcept),
    (ED4ALL.hasSource, ED4ALL.Chunk, ED4ALL.SourceReference),
    # ---- SourceReference attributes ----
    (ED4ALL.sourceId, ED4ALL.SourceReference, XSD.string),
    (ED4ALL.sourceRole, ED4ALL.SourceReference, XSD.string),
    (ED4ALL.sourceWeight, ED4ALL.SourceReference, XSD.decimal),
    (ED4ALL.sourceConfidence, ED4ALL.SourceReference, XSD.decimal),
    (ED4ALL.sourcePages, ED4ALL.SourceReference, XSD.integer),
    (ED4ALL.sourceExtractor, ED4ALL.SourceReference, XSD.string),
    # ---- Misconception (alternate statement predicate) ----
    (ED4ALL.misconceptionStatement, ED4ALL.Misconception, XSD.string),
    # ---- Pre-existing predicates flagged in _phase2_followup
    # (regression-guard their continued presence) ----
    (ED4ALL.hasMisconception, ED4ALL.CourseModule, ED4ALL.Misconception),
    (ED4ALL.correction, ED4ALL.Misconception, XSD.string),
]


@pytest.mark.parametrize(
    "predicate_iri,expected_domain,expected_range",
    _WAVE_87_MINTED_PREDICATES,
)
def test_wave_87_predicate_declared_with_domain_and_range(
    vocab_graph, predicate_iri, expected_domain, expected_range
):
    """Every Wave 87 minted predicate (and the four pre-existing predicates
    flagged in chunk_v4_v1.jsonld's original _phase2_followup list) must be
    declared as rdf:Property / owl:DatatypeProperty / owl:ObjectProperty
    with rdfs:domain + rdfs:range pointing at the right ed4all: classes
    or XSD datatypes. Without these declarations RDFS reasoners can't
    infer endpoint types over chunk_v4 data routed through the Phase 1.2
    JSON-LD bridge."""
    is_rdf_prop = (predicate_iri, RDF.type, RDF.Property) in vocab_graph
    is_owl_obj_prop = (
        predicate_iri,
        RDF.type,
        OWL.ObjectProperty,
    ) in vocab_graph
    is_owl_data_prop = (
        predicate_iri,
        RDF.type,
        OWL.DatatypeProperty,
    ) in vocab_graph
    assert is_rdf_prop or is_owl_obj_prop or is_owl_data_prop, (
        f"Predicate {predicate_iri} must be declared as rdf:Property, "
        f"owl:ObjectProperty, or owl:DatatypeProperty in "
        f"courseforge_v1.vocabulary.ttl"
    )
    assert (predicate_iri, RDFS.domain, expected_domain) in vocab_graph, (
        f"Predicate {predicate_iri} must declare rdfs:domain "
        f"{expected_domain}"
    )
    assert (predicate_iri, RDFS.range, expected_range) in vocab_graph, (
        f"Predicate {predicate_iri} must declare rdfs:range "
        f"{expected_range}"
    )


@pytest.mark.parametrize(
    "predicate_iri,_d,_r",
    _WAVE_87_MINTED_PREDICATES,
)
def test_wave_87_predicate_has_english_label(
    vocab_graph, predicate_iri, _d, _r
):
    """Every Wave 87 predicate carries a non-empty rdfs:label so SPARQL
    endpoints / SHACL result authoring can surface a human name without
    learning the cf: namespace. Mirrors the Phase 2.6 label test."""
    labels = list(vocab_graph.objects(predicate_iri, RDFS.label))
    assert labels, f"Predicate {predicate_iri} missing rdfs:label"
    assert any(str(lbl).strip() for lbl in labels), (
        f"Predicate {predicate_iri} has only empty rdfs:label values"
    )


# ---------------------------------------------------------------------- #
# Drift guard — chunk_v4_v1.jsonld no longer carries _phase2_followup,
# only _phase3_followup with the single deferred metadataTrace entry.
# ---------------------------------------------------------------------- #

_CHUNK_CONTEXT_PATH = (
    _PROJECT_ROOT / "schemas" / "context" / "chunk_v4_v1.jsonld"
)


def test_chunk_v4_followup_marker_renamed():
    """chunk_v4_v1.jsonld must no longer carry the legacy `_phase2_followup`
    block. Wave 87 minted out the 35 entries that block tracked and
    renamed the residual marker to `_phase3_followup`, holding only
    the single `metadataTrace` entry deferred for an emit-side schema
    lockdown wave."""
    import json

    with _CHUNK_CONTEXT_PATH.open() as fh:
        ctx = json.load(fh)

    assert "_phase2_followup" not in ctx, (
        "chunk_v4_v1.jsonld still carries `_phase2_followup`. Wave 87 "
        "minted those entries; the residual marker must be renamed to "
        "`_phase3_followup` and contain only `metadataTrace`."
    )
    assert "_phase3_followup" in ctx, (
        "chunk_v4_v1.jsonld must carry `_phase3_followup` after Wave 87 "
        "to track the single deferred metadataTrace IRI."
    )
    deferred = ctx["_phase3_followup"].get("predicates", [])
    assert deferred == ["ed4all:metadataTrace"], (
        f"Expected `_phase3_followup.predicates` to hold exactly "
        f"['ed4all:metadataTrace']; got {deferred!r}"
    )
