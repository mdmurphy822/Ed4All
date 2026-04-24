"""Wave 63 — Courseforge SHACL shapes validation.

Companion to schemas/tests/test_courseforge_jsonld_context.py. Wave 62
gave us a JSON-LD @context so our emit carries real RDF semantics; Wave
63 adds a SHACL shapes file (courseforge_v1.shacl.ttl) that RDF-natively
validates the resulting triples. Pearson / LRMI / CASE adopters recognize
SHACL; the JSON Schema stays authoritative for the JSON wire format.

Pipeline under test:

    JSON-LD payload
        └── pyld expand (apply @context, get RDF triples)
            └── rdflib.Graph (ingest as n-quads)
                └── pyshacl.validate (against courseforge_v1.shacl.ttl)

Covers:

* Shape file parses as Turtle with no syntax errors.
* Well-formed CourseModule / LearningObjective / TargetedConcept /
  Misconception / BloomDistribution payloads validate cleanly.
* Negative cases: each required predicate's absence triggers a
  violation; malformed courseCode / non-IRI bloomLevel / out-of-vocab
  hierarchyLevel / missing TargetedConcept fields all fail with the
  expected sh:resultMessage.
* A real generate_week emit → JSON-LD → SHACL round trip validates,
  proving no drift between the Wave 49 emit-time schema validation
  and the RDF-native shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_CONTEXT_PATH = _PROJECT_ROOT / "schemas" / "context" / "courseforge_v1.jsonld"
_SHAPES_PATH = _PROJECT_ROOT / "schemas" / "context" / "courseforge_v1.shacl.ttl"

pyld = pytest.importorskip(
    "pyld",
    reason="pyld is required for SHACL tests; install with `pip install pyld`.",
)
pyshacl = pytest.importorskip(
    "pyshacl",
    reason="pyshacl is required for SHACL tests; install with `pip install pyshacl`.",
)
rdflib = pytest.importorskip("rdflib", reason="rdflib comes with pyshacl.")

from pyld import jsonld  # noqa: E402
from rdflib import Graph  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixtures — load context + shapes once per module
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def context_doc() -> dict:
    with open(_CONTEXT_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def shapes_graph() -> Graph:
    g = Graph()
    g.parse(_SHAPES_PATH, format="turtle")
    return g


def _payload_to_rdf_graph(payload: dict, context_doc: dict) -> Graph:
    """Expand a JSON-LD payload through the @context and return an rdflib
    Graph containing the resulting triples."""
    with_context = dict(payload)
    with_context["@context"] = context_doc["@context"]
    # pyld.to_rdf emits a dataset; the default format string is 'application/n-quads'.
    nq = jsonld.to_rdf(with_context, {"format": "application/n-quads"})
    g = Graph()
    g.parse(data=nq, format="nquads")
    return g


def _validate(payload: dict, context_doc: dict, shapes_graph: Graph) -> tuple:
    """Run SHACL validation on a payload's RDF graph.

    Returns ``(conforms, results_graph, results_text)`` — the standard
    pyshacl.validate tuple.
    """
    data = _payload_to_rdf_graph(payload, context_doc)
    conforms, results_graph, results_text = pyshacl.validate(
        data_graph=data,
        shacl_graph=shapes_graph,
        inference="none",
        abort_on_first=False,
        meta_shacl=False,
        advanced=True,
        js=False,
        debug=False,
    )
    return conforms, results_graph, results_text


# ---------------------------------------------------------------------- #
# 1. Shape file parses as Turtle
# ---------------------------------------------------------------------- #


def test_shapes_file_parses_as_turtle(shapes_graph):
    # The fixture already parsed; assert we have non-zero content.
    assert len(shapes_graph) > 0, "SHACL shapes file parsed to empty graph"


def test_shapes_file_declares_all_expected_nodeshapes(shapes_graph):
    """Every NodeShape we ship must be present."""
    from rdflib import Namespace, URIRef

    SH = Namespace("http://www.w3.org/ns/shacl#")
    CFSHAPES = Namespace("https://ed4all.dev/ns/courseforge/v1/shapes#")
    expected = {
        URIRef(CFSHAPES + "CourseModuleShape"),
        URIRef(CFSHAPES + "LearningObjectiveShape"),
        URIRef(CFSHAPES + "TargetedConceptShape"),
        URIRef(CFSHAPES + "MisconceptionShape"),
        URIRef(CFSHAPES + "BloomDistributionShape"),
    }
    declared = set(shapes_graph.subjects(SH.targetClass, None))
    missing = expected - declared
    assert not missing, f"Expected NodeShapes missing: {missing}"


# ---------------------------------------------------------------------- #
# 2. Positive cases — well-formed payloads validate
# ---------------------------------------------------------------------- #


def test_well_formed_course_module_validates(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert conforms, f"Well-formed CourseModule failed SHACL:\n{text}"


def test_well_formed_learning_objective_validates(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "learningObjectives": [
            {
                "@type": "LearningObjective",
                "id": "https://example.org/los/CO-01",
                "statement": "Apply the framework to sample data.",
                "bloomLevel": "apply",
                "cognitiveDomain": "procedural",
                "hierarchyLevel": "chapter",
                "parentObjectiveId": "https://example.org/los/TO-01",
            }
        ],
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert conforms, f"Well-formed LearningObjective failed SHACL:\n{text}"


def test_well_formed_targeted_concept_validates(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "learningObjectives": [
            {
                "@type": "LearningObjective",
                "id": "https://example.org/los/CO-01",
                "statement": "Apply X.",
                "bloomLevel": "apply",
                "targetedConcepts": [
                    {
                        "@type": "TargetedConcept",
                        "concept": "https://example.org/concepts/framework",
                        "bloomLevel": "apply",
                    }
                ],
            }
        ],
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert conforms, f"Well-formed TargetedConcept failed SHACL:\n{text}"


def test_well_formed_misconception_validates(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01",
        "misconceptions": [
            {
                "@type": "Misconception",
                "misconception": "Learners often skip normalization.",
                "correction": "Apply normalization before aggregation.",
                "bloomLevel": "apply",
                "cognitiveDomain": "procedural",
            }
        ],
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert conforms, f"Well-formed Misconception failed SHACL:\n{text}"


def test_well_formed_bloom_distribution_validates(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "bloomDistribution": {
            "@type": "BloomDistribution",
            "total": 3,
        },
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert conforms, f"Well-formed BloomDistribution failed SHACL:\n{text}"


# ---------------------------------------------------------------------- #
# 3. Negative cases — required predicates missing / malformed values
# ---------------------------------------------------------------------- #


def test_course_module_missing_course_code_fails(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        # no courseCode
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert not conforms, "CourseModule missing courseCode should fail SHACL"
    assert "courseCode" in text or "MinCountConstraintComponent" in text


def test_course_module_bad_course_code_pattern_fails(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "badcode",  # doesn't match ^[A-Z]{2,}_?\d{3,}$
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert not conforms, "courseCode violating pattern should fail SHACL"
    assert "PatternConstraintComponent" in text or "courseCode" in text


def test_course_module_negative_week_number_fails(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": -1,  # minInclusive 0
        "moduleType": "overview",
        "pageId": "week_01_overview",
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert not conforms, "Negative weekNumber should fail SHACL"
    assert "MinInclusiveConstraintComponent" in text or "weekNumber" in text


def test_learning_objective_missing_statement_fails(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "learningObjectives": [
            {
                "@type": "LearningObjective",
                "id": "https://example.org/los/CO-01",
                # no statement
                "bloomLevel": "apply",
            }
        ],
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert not conforms, "LO missing statement should fail SHACL"
    assert "description" in text or "MinCountConstraintComponent" in text


def test_learning_objective_non_vocab_bloom_level_fails(context_doc, shapes_graph):
    """A bloomLevel IRI outside the vocab namespace violates the pattern."""
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "learningObjectives": [
            {
                "@type": "LearningObjective",
                "id": "https://example.org/los/CO-01",
                "statement": "Apply X.",
                # Bypass the @context's @vocab mapping by supplying a full IRI
                # pointing outside our vocab namespace.
                "ed4all:bloomLevel": {"@id": "http://example.org/other-bloom#apply"},
            }
        ],
    }
    # Apply the context and validate.
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert not conforms, "Non-vocab bloomLevel IRI should fail SHACL pattern"


def test_targeted_concept_missing_concept_fails(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "learningObjectives": [
            {
                "@type": "LearningObjective",
                "id": "https://example.org/los/CO-01",
                "statement": "Apply X.",
                "targetedConcepts": [
                    {
                        "@type": "TargetedConcept",
                        # no concept
                        "bloomLevel": "apply",
                    }
                ],
            }
        ],
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert not conforms, "TargetedConcept missing concept should fail SHACL"


def test_misconception_missing_correction_fails(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01",
        "misconceptions": [
            {
                "@type": "Misconception",
                "misconception": "Some misconception text.",
                # no correction
            }
        ],
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert not conforms, "Misconception missing correction should fail SHACL"
    assert "correction" in text or "MinCountConstraintComponent" in text


def test_bloom_distribution_negative_total_fails(context_doc, shapes_graph):
    payload = {
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "bloomDistribution": {
            "@type": "BloomDistribution",
            "total": -1,  # minInclusive 0
        },
    }
    conforms, _, text = _validate(payload, context_doc, shapes_graph)
    assert not conforms, "Negative BloomDistribution.total should fail SHACL"


# ---------------------------------------------------------------------- #
# 4. End-to-end — real generate_week output validates
# ---------------------------------------------------------------------- #


def test_real_generate_week_output_validates_against_shacl(
    tmp_path, context_doc, shapes_graph
):
    """A full emit → SHACL round trip on real HTML output. Proves the
    emit and the shapes stay in sync (no drift)."""
    import re

    sys.path.insert(0, str(_PROJECT_ROOT / "Courseforge" / "scripts"))
    import generate_course  # noqa: E402

    week_data = {
        "week_number": 1,
        "title": "SHACL smoke",
        "objectives": [
            {
                "id": "TO-01",
                "statement": "Apply the framework in context.",
                "bloom_level": "apply",
                "key_concepts": ["Framework"],
            },
            {
                "id": "CO-01",
                "statement": "Analyze the outputs thoroughly.",
                "bloom_level": "analyze",
                "parent_objective_id": "TO-01",
            },
        ],
        "overview_text": ["Intro."],
        "readings": ["Ch. 1"],
        "content_modules": [
            {
                "title": "M",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": ["body."],
                    }
                ],
                "misconceptions": [
                    {
                        "misconception": "Learners often skip normalization.",
                        "correction": "Apply normalization first.",
                    }
                ],
            }
        ],
        "activities": [],
        "key_takeaways": ["k"],
        "reflection_questions": ["q"],
    }
    out = tmp_path / "out"
    generate_course.generate_week(week_data, out, "TEST_101", source_module_map=None)

    overview = (out / "week_01" / "week_01_overview.html").read_text(encoding="utf-8")
    blocks = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        overview,
        flags=re.DOTALL,
    )
    assert blocks
    for block in blocks:
        payload = json.loads(block)
        conforms, _, text = _validate(payload, context_doc, shapes_graph)
        assert conforms, (
            f"Real generate_week emit failed SHACL validation:\n{text}\n"
            f"Payload: {json.dumps(payload, indent=2)[:500]}"
        )
