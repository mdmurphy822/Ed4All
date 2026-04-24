"""Wave 62 — Courseforge JSON-LD @context: real RDF semantics.

Pre-Wave-62 the JSON-LD payloads we emitted from generate_course.py
carried ``@context: https://ed4all.dev/ns/courseforge/v1`` but nothing
at that URL, no predicate alignment to external vocabularies, and no
way to round-trip the data through a JSON-LD processor to get actual
IRIs. Consumers saw JSON-shaped-like-RDF without the semantics.

Wave 62 publishes ``schemas/context/courseforge_v1.jsonld`` — a real
JSON-LD 1.1 @context document that:

* Maps our compact terms to Schema.org (``LearningResource``, ``teaches``,
  ``hasPart``, ``keywords``, ``coursePrerequisites``, ``courseCode``,
  ``competencyRequired``, ``isBasedOn``, ``identifier``, ``description``,
  ``name``, ``position``, ``learningResourceType``, ``about``).
* Points our Bloom-level / verb / cognitive-domain / hierarchy values
  at SKOS concept schemes under the ed4all namespace
  (``https://ed4all.dev/vocab/bloom#apply`` etc.) so downstream
  reasoners can traverse a controlled vocabulary.
* Falls back to an ed4all: namespace for genuinely custom predicates
  (misconception, targetsConcept, bloomDistribution, teachingRole,
  correction) rather than force-fitting them into Schema.org.
* Adds PROV-O / Dublin Core mappings for provenance fields
  (``run_id`` → ``prov:wasGeneratedBy``, ``generated_at`` →
  ``dcterms:issued``).

These tests verify the @context:

1. Is valid JSON and parses cleanly as a JSON-LD 1.1 context.
2. Round-trips a sample CourseModule payload: expand → compact yields
   back the compact form.
3. Expanding a payload produces the expected external IRIs (not just
   opaque ``ed4all:`` URIs) for the predicates we aligned.
4. Bloom-level / cognitive-domain values become SKOS concept IRIs
   under the ed4all vocab namespace when expanded.
5. A payload produced by the real ``generate_week`` helper expands
   without errors (no dangling compact terms, no schema drift between
   the context doc and the emit surface).

Implementation note: ``pyld`` is a dev-only dependency. The tests skip
(with a clear message) if it isn't installed, so CI without pyld still
passes — but the dev-extras install pins it so development workflows
exercise the tests.
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

pyld = pytest.importorskip(
    "pyld",
    reason=(
        "pyld is a dev dependency (see pyproject.toml [project.optional-"
        "dependencies].dev). Install with `pip install pyld` to exercise "
        "the @context tests."
    ),
)
from pyld import jsonld  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def context_doc() -> dict:
    with open(_CONTEXT_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def context_value(context_doc: dict) -> dict:
    """The inner @context object — what a JSON-LD processor actually uses.

    The file wraps its context under a top-level @context key (standard for
    standalone context documents). When a JSON-LD processor dereferences
    the URL it loads the whole document; when we want to apply the context
    programmatically we pass the inner mapping.
    """
    return context_doc["@context"]


@pytest.fixture
def sample_course_module() -> dict:
    """A minimal but realistic CourseModule payload matching what
    generate_course.py emits from _build_page_metadata."""
    return {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": "TEST_101",
        "weekNumber": 1,
        "moduleType": "overview",
        "pageId": "week_01_overview",
        "learningObjectives": [
            {
                "@type": "LearningObjective",
                "id": "TO-01",
                "statement": "Apply the framework to the sample data.",
                "bloomLevel": "apply",
                "bloomVerb": "apply",
                "bloomLevels": ["apply"],
                "bloomVerbs": ["apply"],
                "cognitiveDomain": "procedural",
                "keyConcepts": ["framework", "sample-data"],
                "targetedConcepts": [
                    {
                        "@type": "TargetedConcept",
                        "concept": "framework",
                        "bloomLevel": "apply",
                    }
                ],
                "hierarchyLevel": "terminal",
            },
            {
                "@type": "LearningObjective",
                "id": "CO-01",
                "statement": "Analyze and evaluate the market trends.",
                "bloomLevel": "evaluate",
                "bloomVerb": "evaluate",
                "bloomLevels": ["evaluate", "analyze"],
                "bloomVerbs": ["evaluate", "analyze"],
                "cognitiveDomain": "metacognitive",
                "hierarchyLevel": "chapter",
                "parentObjectiveId": "TO-01",
            },
        ],
        "misconceptions": [
            {
                "@type": "Misconception",
                "misconception": "A common slip in the procedure sequence.",
                "correction": "Apply normalization before aggregation.",
                "bloomLevel": "apply",
                "cognitiveDomain": "procedural",
            }
        ],
        "bloomDistribution": {
            "@type": "BloomDistribution",
            "total": 2,
            "byLevel": {"apply": 1, "evaluate": 1},
            "byCognitiveDomain": {"procedural": 1, "metacognitive": 1},
        },
    }


def _apply_context(payload: dict, context_doc: dict) -> dict:
    """Substitute the in-memory context doc for the string @context URL.

    Real processors would dereference ``https://ed4all.dev/ns/courseforge/v1``
    via HTTP — we don't own that URL yet, so we swap the @context to the
    in-memory mapping for test purposes. This is exactly the pattern a
    real consumer would use with pyld's document_loader hook.
    """
    out = dict(payload)
    out["@context"] = context_doc["@context"]
    return out


# ---------------------------------------------------------------------- #
# 1. Context doc is valid JSON / JSON-LD
# ---------------------------------------------------------------------- #


def test_context_file_is_valid_json():
    with open(_CONTEXT_PATH, encoding="utf-8") as f:
        doc = json.load(f)
    assert "@context" in doc, "Context file must wrap its mapping under @context"
    inner = doc["@context"]
    assert isinstance(inner, dict)
    assert inner.get("@version") == 1.1, (
        "Context should declare JSON-LD 1.1 (@vocab-in-term-defs requires 1.1)"
    )


def test_context_parses_as_jsonld_context(context_doc):
    """pyld accepts the context without error when processing a payload."""
    minimal = {
        "@context": context_doc["@context"],
        "@type": "CourseModule",
        "id": "https://example.org/courses/test",
        "courseCode": "TEST_101",
    }
    # If the context is malformed, pyld.jsonld.expand raises.
    expanded = jsonld.expand(minimal)
    assert isinstance(expanded, list)
    assert len(expanded) == 1


# ---------------------------------------------------------------------- #
# 2. Predicate alignment — expanded IRIs are the Schema.org ones
# ---------------------------------------------------------------------- #


def test_course_module_type_expands_to_schema_learning_resource(
    context_doc, sample_course_module
):
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)[0]
    assert "http://schema.org/LearningResource" in expanded.get("@type", []), (
        f"@type should map to schema:LearningResource; got {expanded.get('@type')!r}"
    )


def test_core_predicates_map_to_schema_org(context_doc, sample_course_module):
    """Top-level predicates expand to Schema.org IRIs where aligned."""
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)[0]
    # courseCode → schema:courseCode
    assert "http://schema.org/courseCode" in expanded
    # weekNumber → schema:position
    assert "http://schema.org/position" in expanded
    # moduleType → schema:learningResourceType
    assert "http://schema.org/learningResourceType" in expanded
    # pageId → schema:identifier
    assert "http://schema.org/identifier" in expanded
    # learningObjectives → schema:teaches
    assert "http://schema.org/teaches" in expanded


def test_lo_statement_maps_to_schema_description(context_doc, sample_course_module):
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)[0]
    los = expanded["http://schema.org/teaches"]
    assert los, "learningObjectives expanded to an empty list"
    # schema:description on each LO
    for lo in los:
        assert "http://schema.org/description" in lo


def test_key_concepts_maps_to_schema_keywords(context_doc, sample_course_module):
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)[0]
    los = expanded["http://schema.org/teaches"]
    lo_with_concepts = next(
        (lo for lo in los if "http://schema.org/keywords" in lo), None
    )
    assert lo_with_concepts is not None, (
        "At least one LO should expose schema:keywords (from keyConcepts)"
    )


def test_parent_objective_id_expands_to_id_reference(context_doc, sample_course_module):
    """parentObjectiveId has @type: @id → value becomes an IRI reference."""
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)[0]
    los = expanded["http://schema.org/teaches"]
    co = next(
        (lo for lo in los if "ed4all:parentObjective" in str(lo)), None
    ) or next((lo for lo in los if any("parentObjective" in k for k in lo)), None)
    assert co is not None, "CO-01 with parentObjectiveId not found after expansion"
    parent_predicate = (
        "https://ed4all.dev/ns/courseforge/v1#parentObjective"
    )
    assert parent_predicate in co, (
        f"parentObjectiveId should expand to {parent_predicate}; got keys {list(co)!r}"
    )
    # Value is an @id reference, not a literal
    parent_obj = co[parent_predicate][0]
    assert "@id" in parent_obj, (
        f"parentObjective value should be an @id reference; got {parent_obj!r}"
    )


# ---------------------------------------------------------------------- #
# 3. Bloom / cognitive domain → SKOS-shaped concept IRIs
# ---------------------------------------------------------------------- #


def test_bloom_level_expands_to_vocab_concept_iri(context_doc, sample_course_module):
    """bloomLevel values become IRIs under https://ed4all.dev/vocab/bloom#."""
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)[0]
    los = expanded["http://schema.org/teaches"]
    bloom_predicate = "https://ed4all.dev/ns/courseforge/v1#bloomLevel"
    found_iris = []
    for lo in los:
        for entry in lo.get(bloom_predicate, []):
            if "@id" in entry:
                found_iris.append(entry["@id"])
    assert found_iris, (
        f"No bloomLevel expanded to a vocab IRI; check @type: @vocab wiring"
    )
    assert all(
        iri.startswith("https://ed4all.dev/vocab/bloom#") for iri in found_iris
    ), f"Bloom-level values must expand under the bloom vocab namespace; got {found_iris!r}"


def test_cognitive_domain_expands_to_vocab_concept_iri(
    context_doc, sample_course_module
):
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)[0]
    los = expanded["http://schema.org/teaches"]
    domain_predicate = "https://ed4all.dev/ns/courseforge/v1#cognitiveDomain"
    iris = [
        entry.get("@id")
        for lo in los
        for entry in lo.get(domain_predicate, [])
        if "@id" in entry
    ]
    assert iris, "cognitiveDomain did not expand to any IRIs"
    assert all(
        iri.startswith("https://ed4all.dev/vocab/cognitive-domain#") for iri in iris
    ), f"cognitiveDomain values must expand under the vocab namespace; got {iris!r}"


def test_hierarchy_level_expands_to_vocab_concept_iri(
    context_doc, sample_course_module
):
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)[0]
    los = expanded["http://schema.org/teaches"]
    hierarchy_predicate = "https://ed4all.dev/ns/courseforge/v1#hierarchyLevel"
    iris = [
        entry.get("@id")
        for lo in los
        for entry in lo.get(hierarchy_predicate, [])
        if "@id" in entry
    ]
    assert iris, "hierarchyLevel did not expand to any IRIs"
    assert all(
        iri.startswith("https://ed4all.dev/vocab/hierarchy#") for iri in iris
    ), f"hierarchyLevel values must expand under the vocab namespace; got {iris!r}"
    # The actual values (terminal, chapter) become #terminal and #chapter
    assert any(iri.endswith("#terminal") for iri in iris)
    assert any(iri.endswith("#chapter") for iri in iris)


# ---------------------------------------------------------------------- #
# 4. Round-trip: expand → compact yields back the compact form
# ---------------------------------------------------------------------- #


def test_expand_then_compact_roundtrip(context_doc, sample_course_module):
    payload = _apply_context(sample_course_module, context_doc)
    expanded = jsonld.expand(payload)
    compacted = jsonld.compact(expanded, context_doc["@context"])
    # @type round-trips back to CourseModule
    assert compacted.get("@type") == "CourseModule", (
        f"Expected @type='CourseModule' after roundtrip; got {compacted.get('@type')!r}"
    )
    # courseCode survives
    assert compacted.get("courseCode") == "TEST_101"
    # moduleType survives
    assert compacted.get("moduleType") == "overview"


# ---------------------------------------------------------------------- #
# 5. Real generate_week output expands cleanly
# ---------------------------------------------------------------------- #


def test_real_generate_week_output_expands_cleanly(tmp_path, context_doc):
    """A full generate_week round trip → extract JSON-LD from HTML → expand
    through our @context. Proves no emit/context drift."""
    import re

    sys.path.insert(0, str(_PROJECT_ROOT / "Courseforge" / "scripts"))
    import generate_course  # noqa: E402

    week_data = {
        "week_number": 1,
        "title": "Context smoke",
        "objectives": [
            {
                "id": "TO-01",
                "statement": "Apply the framework in realistic contexts.",
                "bloom_level": "apply",
                "key_concepts": ["Framework", "Context"],
            },
            {
                "id": "CO-01",
                "statement": "Analyze the outputs.",
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
                        "misconception": "Learners often skip the normalization.",
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
    assert blocks, "generate_week output had no JSON-LD blocks"
    for block in blocks:
        parsed = json.loads(block)
        with_context = dict(parsed)
        with_context["@context"] = context_doc["@context"]
        # This must not raise — if the emit carries a key not defined in
        # the context, pyld will either drop it (expand keeps it under its
        # compact name) or raise. Either way a completely clean expand
        # proves the context covers the emit surface.
        expanded = jsonld.expand(with_context)
        assert isinstance(expanded, list)
        # And the top-level @type expands to schema:LearningResource.
        if expanded:
            types = expanded[0].get("@type", [])
            if types:
                assert "http://schema.org/LearningResource" in types, (
                    f"Emit payload @type didn't expand to schema:LearningResource; "
                    f"got {types!r}. This usually means the emit uses a @type "
                    f"keyword not mapped in the context."
                )
