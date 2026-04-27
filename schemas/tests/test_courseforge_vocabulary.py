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


# ---------------------------------------------------------------------- #
# 6. Phase 2.1 — Concept-graph edge-predicate alignment
#
# The 9-slug edge enum in concept_graph_semantic.schema.json is mapped to
# real RDF predicates via lib/ontology/edge_predicates.py. These tests
# enforce three invariants:
#
#   (a) Every slug in the JSON enum has a registered IRI.
#   (b) Every minted ed4all: predicate parses out of the Turtle vocab
#       with the declared rdfs:domain + rdfs:range we promised.
#   (c) The slug -> IRI -> slug round-trip is exact (registry is bijective).
# ---------------------------------------------------------------------- #


import json  # noqa: E402


_CONCEPT_GRAPH_SEMANTIC_SCHEMA = (
    _PROJECT_ROOT
    / "schemas"
    / "knowledge"
    / "concept_graph_semantic.schema.json"
)


def _load_edge_type_enum():
    """Read the source-of-truth slug enum from the JSON schema."""
    with _CONCEPT_GRAPH_SEMANTIC_SCHEMA.open() as fh:
        schema = json.load(fh)
    return schema["properties"]["edges"]["items"]["properties"]["type"]["enum"]


def test_every_json_edge_slug_has_registered_iri():
    """The JSON enum is the source of truth for which slugs exist; every
    one of those slugs must have a registered IRI in the edge-predicate
    registry. Drift here means the JSON-LD bridge breaks for that slug."""
    from lib.ontology.edge_predicates import SLUG_TO_IRI

    enum_slugs = set(_load_edge_type_enum())
    registered_slugs = set(SLUG_TO_IRI.keys())
    missing = enum_slugs - registered_slugs
    assert not missing, (
        f"Edge-type slugs declared in concept_graph_semantic.schema.json "
        f"but absent from lib.ontology.edge_predicates.SLUG_TO_IRI: "
        f"{sorted(missing)}. Add a slug -> IRI binding (and the matching "
        f"predicate declaration in courseforge_v1.vocabulary.ttl)."
    )


def test_slug_iri_roundtrip_is_exact():
    """Round-trip: for every slug s, IRI_TO_SLUG[SLUG_TO_IRI[s]] == s.
    Failure means the registry is non-bijective (two slugs share an IRI),
    which would make consumer round-trip parsing ambiguous."""
    from lib.ontology.edge_predicates import IRI_TO_SLUG, SLUG_TO_IRI

    for slug, iri in SLUG_TO_IRI.items():
        assert IRI_TO_SLUG[iri] == slug, (
            f"Round-trip mismatch: SLUG_TO_IRI[{slug!r}] -> {iri} -> "
            f"IRI_TO_SLUG[{iri!r}] = {IRI_TO_SLUG.get(iri)!r}, expected {slug!r}. "
            f"Two slugs may share an IRI — break the collision."
        )


# Predicates minted in this phase. (is-a / related-to reuse W3C predicates
# and are NOT declared in our Turtle file — they're imported by reference.)
# Each tuple: (slug, predicate IRI, expected rdfs:domain, expected rdfs:range).
_PHASE_2_1_MINTED_EDGE_PREDICATES = [
    ("prerequisite", ED4ALL.hasPrerequisite, ED4ALL.Concept, ED4ALL.Concept),
    ("defined-by", ED4ALL.isDefinedBy, ED4ALL.Concept, ED4ALL.Chunk),
    (
        "derived-from-objective",
        ED4ALL.isDerivedFromObjective,
        ED4ALL.Chunk,
        ED4ALL.LearningObjective,
    ),
    ("exemplifies", ED4ALL.exemplifiedBy, ED4ALL.Chunk, ED4ALL.Concept),
    (
        "misconception-of",
        ED4ALL.isMisconceptionOf,
        ED4ALL.Misconception,
        ED4ALL.Concept,
    ),
    (
        "assesses",
        ED4ALL.assessesObjective,
        ED4ALL.AssessmentQuestion,
        ED4ALL.LearningObjective,
    ),
    # 'targets-concept' is NOT in this list — its predicate
    # (ed4all:targetsConcept) was minted in Wave 57 with the reified-edge
    # range ed4all:TargetedConcept (carrying the Bloom qualifier on the
    # qualifier node). The existing
    # test_targets_concept_domain_and_range covers that declaration.
]


@pytest.mark.parametrize(
    "slug,predicate_iri,expected_domain,expected_range",
    _PHASE_2_1_MINTED_EDGE_PREDICATES,
)
def test_minted_edge_predicates_declared_with_domain_and_range(
    vocab_graph, slug, predicate_iri, expected_domain, expected_range
):
    """Every Phase 2.1 minted ed4all: predicate must be declared in the
    Turtle vocabulary as rdf:Property with rdfs:domain + rdfs:range
    pointing at the right ed4all: classes. Without these declarations
    RDFS reasoners can't infer the type of edge endpoints."""
    # Declared as an rdf:Property (we also typed it as owl:ObjectProperty,
    # but rdf:Property is the minimal claim we make here).
    is_rdf_prop = (predicate_iri, RDF.type, RDF.Property) in vocab_graph
    is_owl_obj_prop = (predicate_iri, RDF.type, OWL.ObjectProperty) in vocab_graph
    assert is_rdf_prop or is_owl_obj_prop, (
        f"Slug {slug!r} -> {predicate_iri} must be declared as "
        f"rdf:Property or owl:ObjectProperty in courseforge_v1.vocabulary.ttl"
    )
    assert (predicate_iri, RDFS.domain, expected_domain) in vocab_graph, (
        f"Slug {slug!r} -> {predicate_iri} must declare "
        f"rdfs:domain {expected_domain}"
    )
    assert (predicate_iri, RDFS.range, expected_range) in vocab_graph, (
        f"Slug {slug!r} -> {predicate_iri} must declare "
        f"rdfs:range {expected_range}"
    )


def test_phase_2_1_classes_declared(vocab_graph):
    """Phase 2.1 added three new ed4all: classes that anchor the new
    edge-predicate domains/ranges: Concept, Chunk, AssessmentQuestion."""
    for cls in (ED4ALL.Concept, ED4ALL.Chunk, ED4ALL.AssessmentQuestion):
        assert (cls, RDF.type, RDFS.Class) in vocab_graph or (
            cls,
            RDF.type,
            OWL.Class,
        ) in vocab_graph, f"Phase 2.1 class {cls} must be declared"


def test_concept_class_subclass_of_skos_concept(vocab_graph):
    """ed4all:Concept rdfs:subClassOf skos:Concept lets SKOS tooling
    treat our concept graph as a SKOS concept scheme without learning
    ed4all:."""
    assert (ED4ALL.Concept, RDFS.subClassOf, SKOS.Concept) in vocab_graph


# ---------------------------------------------------------------------- #
# 7. Phase 2.6 — Pedagogy-graph edge predicates (mint the remaining 9)
#
# pedagogy_graph_builder.py emits 13 distinct relation_type slugs; four
# overlap with Phase 2.1 (or normalize back to it via Phase 2.7). The nine
# below were unminted before this phase and rounded-tripped through the
# JSON-LD @vocab fallback only. These tests enforce four invariants:
#
#   (a) Each predicate is declared as rdf:Property / owl:ObjectProperty.
#   (b) Each carries the rdfs:domain / rdfs:range we promised.
#   (c) Each carries an English rdfs:label.
#   (d) The slug -> IRI -> slug round-trip via SLUG_TO_IRI / IRI_TO_SLUG
#       is exact (registry stays bijective after the 9-entry extension).
#
# Domain/range targets were verified against the rdf-shacl-551-2 fixture
# (LibV2/courses/rdf-shacl-551-2/graph/pedagogy_graph.json) — see Phase
# 2.6 implementation notes.
# ---------------------------------------------------------------------- #


_PHASE_2_6_MINTED_EDGE_PREDICATES = [
    # (slug, predicate IRI, expected rdfs:domain, expected rdfs:range)
    ("teaches", ED4ALL.teaches, ED4ALL.Chunk, ED4ALL.LearningObjective),
    (
        "belongs_to_module",
        ED4ALL.belongsToModule,
        ED4ALL.Chunk,
        ED4ALL.Module,
    ),
    (
        "supports_outcome",
        ED4ALL.supportsOutcome,
        ED4ALL.LearningObjective,
        ED4ALL.LearningObjective,
    ),
    (
        "at_bloom_level",
        ED4ALL.atBloomLevel,
        ED4ALL.LearningObjective,
        ED4ALL.BloomLevel,
    ),
    ("follows", ED4ALL.follows, ED4ALL.Module, ED4ALL.Module),
    (
        "concept_supports_outcome",
        ED4ALL.conceptSupportsOutcome,
        ED4ALL.Concept,
        ED4ALL.LearningObjective,
    ),
    (
        "assessment_validates_outcome",
        ED4ALL.assessmentValidatesOutcome,
        ED4ALL.Chunk,
        ED4ALL.LearningObjective,
    ),
    (
        "chunk_at_difficulty",
        ED4ALL.chunkAtDifficulty,
        ED4ALL.Chunk,
        ED4ALL.DifficultyLevel,
    ),
    (
        "interferes_with",
        ED4ALL.interferesWith,
        ED4ALL.Misconception,
        ED4ALL.Concept,
    ),
]


def test_phase_2_6_anchor_classes_declared(vocab_graph):
    """Phase 2.6 introduces three new anchor classes (Module, BloomLevel,
    DifficultyLevel) used as rdfs:domain / rdfs:range targets for the
    nine new pedagogy edge predicates. Without these declarations the
    new predicates would point at undeclared resources and RDFS
    reasoners couldn't infer endpoint types."""
    for cls in (ED4ALL.Module, ED4ALL.BloomLevel, ED4ALL.DifficultyLevel):
        assert (cls, RDF.type, RDFS.Class) in vocab_graph or (
            cls,
            RDF.type,
            OWL.Class,
        ) in vocab_graph, f"Phase 2.6 anchor class {cls} must be declared"


@pytest.mark.parametrize(
    "slug,predicate_iri,expected_domain,expected_range",
    _PHASE_2_6_MINTED_EDGE_PREDICATES,
)
def test_phase_2_6_predicate_declared_with_domain_and_range(
    vocab_graph, slug, predicate_iri, expected_domain, expected_range
):
    """Every Phase 2.6 minted predicate must be declared as
    rdf:Property / owl:ObjectProperty with rdfs:domain + rdfs:range
    pointing at the right ed4all: classes. Domain/range targets were
    confirmed empirically against the rdf-shacl-551-2 pedagogy_graph.json
    fixture (see Phase 2.6 implementation notes)."""
    is_rdf_prop = (predicate_iri, RDF.type, RDF.Property) in vocab_graph
    is_owl_obj_prop = (predicate_iri, RDF.type, OWL.ObjectProperty) in vocab_graph
    assert is_rdf_prop or is_owl_obj_prop, (
        f"Slug {slug!r} -> {predicate_iri} must be declared as "
        f"rdf:Property or owl:ObjectProperty in courseforge_v1.vocabulary.ttl"
    )
    assert (predicate_iri, RDFS.domain, expected_domain) in vocab_graph, (
        f"Slug {slug!r} -> {predicate_iri} must declare "
        f"rdfs:domain {expected_domain}"
    )
    assert (predicate_iri, RDFS.range, expected_range) in vocab_graph, (
        f"Slug {slug!r} -> {predicate_iri} must declare "
        f"rdfs:range {expected_range}"
    )


@pytest.mark.parametrize(
    "slug,predicate_iri,_d,_r",
    _PHASE_2_6_MINTED_EDGE_PREDICATES,
)
def test_phase_2_6_predicate_has_english_label(
    vocab_graph, slug, predicate_iri, _d, _r
):
    """Every Phase 2.6 predicate carries a non-empty rdfs:label so
    SPARQL endpoints / SHACL result authoring can surface a human name
    without having to learn the cf: namespace."""
    labels = list(vocab_graph.objects(predicate_iri, RDFS.label))
    assert labels, f"Slug {slug!r} -> {predicate_iri} missing rdfs:label"
    # The vocabulary file declares plain string labels (no @en tag in
    # Turtle); rdflib parses these as Literal values whose ``.language``
    # is None. Enforce that the label string is non-empty either way —
    # the project convention is English labels regardless of language tag.
    assert any(str(lbl).strip() for lbl in labels), (
        f"Slug {slug!r} -> {predicate_iri} has only empty rdfs:label values"
    )


@pytest.mark.parametrize(
    "slug,predicate_iri,_d,_r",
    _PHASE_2_6_MINTED_EDGE_PREDICATES,
)
def test_phase_2_6_slug_iri_roundtrip(slug, predicate_iri, _d, _r):
    """Phase 2.6 extension keeps SLUG_TO_IRI / IRI_TO_SLUG bijective:
    every new slug round-trips slug -> IRI -> slug exactly."""
    from lib.ontology.edge_predicates import IRI_TO_SLUG, SLUG_TO_IRI

    assert SLUG_TO_IRI.get(slug) == str(predicate_iri), (
        f"SLUG_TO_IRI[{slug!r}] = {SLUG_TO_IRI.get(slug)!r}; "
        f"expected {str(predicate_iri)!r}"
    )
    assert IRI_TO_SLUG.get(str(predicate_iri)) == slug, (
        f"IRI_TO_SLUG[{str(predicate_iri)!r}] = "
        f"{IRI_TO_SLUG.get(str(predicate_iri))!r}; expected {slug!r}"
    )
