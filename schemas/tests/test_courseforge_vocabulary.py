"""Wave 65 — Courseforge RDFS/OWL vocabulary file parses + declares the
expected class/predicate landscape.

Companion to the JSON-LD @context (Wave 62), SHACL shapes (Wave 63), and
pyld document loader (Wave 64). This wave adds a Turtle vocabulary file
(courseforge_v1.vocabulary.ttl) that formally declares every ed4all:
class and predicate we emit, with rdfs:subClassOf / rdfs:subPropertyOf /
owl:equivalentProperty pointers to Schema.org, Dublin Core, SKOS, and
PROV-O. RDFS/OWL reasoners can then infer that ed4all:CourseModule is a
schema:LearningResource without having to learn our namespace.

Covers:

* The vocabulary file parses as Turtle.
* Every expected ed4all: class is declared as rdfs:Class / owl:Class.
* Key subClassOf relationships land (CourseModule → LearningResource,
  LearningObjective → DefinedTerm, Section → CreativeWork).
* Predicates declared with correct domain / range / subPropertyOf.
* SKOS concept schemes for Bloom, cognitive domain, hierarchy each
  carry the expected six / four / two concepts.
* Every Bloom-level IRI our Wave 62 @context produces on expansion is
  declared here as a skos:Concept in skos:inScheme bloom: — no drift
  between the emit and the vocabulary declaration.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

rdflib = pytest.importorskip(
    "rdflib", reason="rdflib is required for vocabulary parsing tests."
)
from rdflib import Graph, Namespace, URIRef  # noqa: E402
from rdflib.namespace import OWL, RDF, RDFS, SKOS  # noqa: E402

_VOCAB_PATH = (
    _PROJECT_ROOT / "schemas" / "context" / "courseforge_v1.vocabulary.ttl"
)

ED4ALL = Namespace("https://ed4all.dev/ns/courseforge/v1#")
BLOOM = Namespace("https://ed4all.dev/vocab/bloom#")
COGDOMAIN = Namespace("https://ed4all.dev/vocab/cognitive-domain#")
HIERARCHY = Namespace("https://ed4all.dev/vocab/hierarchy#")
SCHEMA = Namespace("http://schema.org/")
DCTERMS = Namespace("http://purl.org/dc/terms/")


@pytest.fixture(scope="module")
def vocab_graph() -> Graph:
    g = Graph()
    g.parse(_VOCAB_PATH, format="turtle")
    return g


# ---------------------------------------------------------------------- #
# 1. File parses + ontology header
# ---------------------------------------------------------------------- #


def test_vocabulary_file_parses_as_turtle(vocab_graph):
    assert len(vocab_graph) > 0


def test_ontology_header_declared(vocab_graph):
    ont = URIRef("https://ed4all.dev/ns/courseforge/v1")
    assert (ont, RDF.type, OWL.Ontology) in vocab_graph, (
        "Expected the canonical Courseforge URI to be declared as owl:Ontology"
    )


# ---------------------------------------------------------------------- #
# 2. Classes
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "class_iri",
    [
        ED4ALL.CourseModule,
        ED4ALL.LearningObjective,
        ED4ALL.Section,
        ED4ALL.Misconception,
        ED4ALL.TargetedConcept,
        ED4ALL.BloomDistribution,
    ],
)
def test_every_expected_class_is_declared(vocab_graph, class_iri):
    assert (class_iri, RDF.type, RDFS.Class) in vocab_graph or (
        class_iri,
        RDF.type,
        OWL.Class,
    ) in vocab_graph, f"Expected {class_iri} declared as rdfs:Class / owl:Class"


@pytest.mark.parametrize(
    "subclass,superclass",
    [
        (ED4ALL.CourseModule, SCHEMA.LearningResource),
        (ED4ALL.LearningObjective, SCHEMA.DefinedTerm),
        (ED4ALL.Section, SCHEMA.CreativeWork),
    ],
)
def test_subclass_of_schema_org_declared(vocab_graph, subclass, superclass):
    assert (subclass, RDFS.subClassOf, superclass) in vocab_graph, (
        f"Expected {subclass} rdfs:subClassOf {superclass} (alignment to "
        f"Schema.org). Without this axiom RDFS reasoners can't answer "
        f"'is this a LearningResource?' without learning our namespace."
    )


# ---------------------------------------------------------------------- #
# 3. Predicates
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "prop,prop_type",
    [
        (ED4ALL.parentObjective, OWL.ObjectProperty),
        (ED4ALL.targetsConcept, OWL.ObjectProperty),
        (ED4ALL.hasMisconception, OWL.ObjectProperty),
        (ED4ALL.bloomLevel, OWL.ObjectProperty),
        (ED4ALL.cognitiveDomain, OWL.ObjectProperty),
        (ED4ALL.hierarchyLevel, OWL.ObjectProperty),
        (ED4ALL.bloomVerb, OWL.DatatypeProperty),
        (ED4ALL.correction, OWL.DatatypeProperty),
        (ED4ALL.total, OWL.DatatypeProperty),
    ],
)
def test_every_expected_property_declared(vocab_graph, prop, prop_type):
    assert (prop, RDF.type, prop_type) in vocab_graph, (
        f"Expected {prop} declared as {prop_type}"
    )


def test_parent_objective_subproperty_of_dcterms_is_part_of(vocab_graph):
    """Wave 65 declares that parentObjective behaves like dcterms:isPartOf —
    Dublin Core tooling can traverse LO hierarchy without knowing ed4all:."""
    assert (
        ED4ALL.parentObjective,
        RDFS.subPropertyOf,
        DCTERMS.isPartOf,
    ) in vocab_graph


def test_parent_objective_domain_and_range(vocab_graph):
    assert (ED4ALL.parentObjective, RDFS.domain, ED4ALL.LearningObjective) in vocab_graph
    assert (ED4ALL.parentObjective, RDFS.range, ED4ALL.LearningObjective) in vocab_graph


def test_targets_concept_domain_and_range(vocab_graph):
    """Wave 57 edge: LO → TargetedConcept → concept."""
    assert (ED4ALL.targetsConcept, RDFS.domain, ED4ALL.LearningObjective) in vocab_graph
    assert (ED4ALL.targetsConcept, RDFS.range, ED4ALL.TargetedConcept) in vocab_graph


# ---------------------------------------------------------------------- #
# 4. SKOS concept schemes
# ---------------------------------------------------------------------- #


def test_bloom_concept_scheme_has_six_top_concepts(vocab_graph):
    """The six Bloom levels are declared and linked to the scheme."""
    expected_levels = {
        BLOOM.remember,
        BLOOM.understand,
        BLOOM.apply,
        BLOOM.analyze,
        BLOOM.evaluate,
        BLOOM.create,
    }
    # Each declared as skos:Concept in skos:inScheme bloom:
    in_scheme = set(vocab_graph.subjects(SKOS.inScheme, BLOOM[""]))
    missing = expected_levels - in_scheme
    assert not missing, f"Bloom concepts missing from scheme: {missing}"
    # Scheme declared
    assert (BLOOM[""], RDF.type, SKOS.ConceptScheme) in vocab_graph


def test_bloom_concepts_have_pref_labels(vocab_graph):
    for lvl in ("remember", "understand", "apply", "analyze", "evaluate", "create"):
        concept = BLOOM[lvl]
        labels = list(vocab_graph.objects(concept, SKOS.prefLabel))
        assert labels, f"{concept} missing skos:prefLabel"


def test_bloom_concepts_have_ordering_via_broader(vocab_graph):
    """Lower levels are skos:broader of higher ones (so reasoners can
    traverse remember → create)."""
    # remember's broader is understand (one step up)
    assert (BLOOM.remember, SKOS.broader, BLOOM.understand) in vocab_graph
    # evaluate's broader is create (one step up)
    assert (BLOOM.evaluate, SKOS.broader, BLOOM.create) in vocab_graph


def test_cognitive_domain_scheme_has_four_concepts(vocab_graph):
    expected = {
        COGDOMAIN.factual,
        COGDOMAIN.conceptual,
        COGDOMAIN.procedural,
        COGDOMAIN.metacognitive,
    }
    in_scheme = set(vocab_graph.subjects(SKOS.inScheme, COGDOMAIN[""]))
    missing = expected - in_scheme
    assert not missing, f"Cognitive-domain concepts missing: {missing}"
    assert (COGDOMAIN[""], RDF.type, SKOS.ConceptScheme) in vocab_graph


def test_hierarchy_scheme_has_terminal_and_chapter(vocab_graph):
    expected = {HIERARCHY.terminal, HIERARCHY.chapter}
    in_scheme = set(vocab_graph.subjects(SKOS.inScheme, HIERARCHY[""]))
    missing = expected - in_scheme
    assert not missing


# ---------------------------------------------------------------------- #
# 5. No drift between Wave 62 @context expansion and Wave 65 concept IRIs
# ---------------------------------------------------------------------- #


def test_every_canonical_bloom_level_has_skos_concept(vocab_graph):
    """Every Bloom level our lib.ontology.bloom exports must also be
    declared as a skos:Concept in the vocabulary. Prevents emit/vocab
    drift where the code produces a level IRI that has no formal
    declaration."""
    from lib.ontology.bloom import BLOOM_LEVELS

    declared_concepts = set(vocab_graph.subjects(RDF.type, SKOS.Concept))
    for level in BLOOM_LEVELS:
        iri = BLOOM[level]
        assert iri in declared_concepts, (
            f"lib.ontology.bloom exports {level!r} but the vocabulary file "
            f"has no skos:Concept for {iri}. Emit would produce a dangling "
            f"IRI with no formal semantics."
        )


def test_every_cognitive_domain_has_skos_concept(vocab_graph):
    from lib.ontology.bloom import COGNITIVE_DOMAINS

    declared_concepts = set(vocab_graph.subjects(RDF.type, SKOS.Concept))
    for domain in COGNITIVE_DOMAINS:
        iri = COGDOMAIN[domain]
        assert iri in declared_concepts, (
            f"Cognitive domain {domain!r} missing skos:Concept in vocabulary"
        )
