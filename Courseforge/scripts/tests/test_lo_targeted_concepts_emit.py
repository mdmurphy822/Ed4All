"""Wave 57 — Bloom-qualified LO→concept edges emitted on every LO.

The ``LearningObjective`` JSON-LD shape gains an optional ``targetedConcepts``
array. Each entry is a ``{concept, bloomLevel}`` edge — the concept is a
canonical slug from ``lib.ontology.slugs.canonical_slug`` and the bloomLevel
is inherited from the parent LO's ``bloomLevel``. This lets downstream KG
consumers materialize ``LO --[bloomLevel]--> concept`` edges directly
instead of inferring them via chunk co-occurrence.

Covers:

* Schema round-trip: the ``targetedConcepts`` field validates against the
  updated ``courseforge_jsonld_v1.schema.json`` for a minted LO.
* Emit round-trip: ``_build_objectives_metadata`` populates
  ``targetedConcepts`` when ``key_concepts`` AND ``bloom_level`` are
  present, with every entry pairing a ``keyConcepts`` slug to the parent
  LO's ``bloomLevel``.
* Elision: ``targetedConcepts`` is absent when ``bloom_level`` is None or
  when ``key_concepts`` is empty.
* Invariant: every slug in ``targetedConcepts`` appears verbatim in
  ``keyConcepts`` (drift guard for future refactors).
* Backward compat: legacy LOs without ``targetedConcepts`` still validate.
* End-to-end: a generated page emits ``targetedConcepts`` inside the JSON-LD
  ``<script type="application/ld+json">`` block, visible to downstream
  consumers reading the HTML.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, RefResolver

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_course  # noqa: E402
from generate_course import _build_objectives_metadata  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_JSONLD_SCHEMA_PATH = (
    _PROJECT_ROOT / "schemas" / "knowledge" / "courseforge_jsonld_v1.schema.json"
)
_BLOOM_VERBS_SCHEMA_PATH = (
    _PROJECT_ROOT / "schemas" / "taxonomies" / "bloom_verbs.json"
)
_COGNITIVE_DOMAIN_SCHEMA_PATH = (
    _PROJECT_ROOT / "schemas" / "taxonomies" / "cognitive_domain.json"
)


def _lo_validator() -> Draft202012Validator:
    """Validator bound to the LearningObjective $defs subschema.

    The JSON-LD root schema uses `$ref` to external taxonomy schemas for
    bloomLevel / cognitiveDomain / question_type. We register a RefResolver
    with all referenced schemas so a LO subschema can be validated in
    isolation without a live network fetch.
    """
    with open(_JSONLD_SCHEMA_PATH, encoding="utf-8") as f:
        root = json.load(f)
    with open(_BLOOM_VERBS_SCHEMA_PATH, encoding="utf-8") as f:
        bloom = json.load(f)
    with open(_COGNITIVE_DOMAIN_SCHEMA_PATH, encoding="utf-8") as f:
        cog = json.load(f)
    question_type_path = (
        _PROJECT_ROOT / "schemas" / "taxonomies" / "question_type.json"
    )
    with open(question_type_path, encoding="utf-8") as f:
        qtype = json.load(f)

    store = {
        root["$id"]: root,
        bloom["$id"]: bloom,
        cog["$id"]: cog,
        qtype["$id"]: qtype,
    }
    resolver = RefResolver.from_schema(root, store=store)

    lo_subschema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": f"{root['$id']}#/$defs/LearningObjective",
    }
    return Draft202012Validator(lo_subschema, resolver=resolver)


# ---------------------------------------------------------------------- #
# 1. Schema round-trip
# ---------------------------------------------------------------------- #


def test_schema_accepts_targeted_concepts_field():
    """A LO with targetedConcepts validates against the updated schema."""
    validator = _lo_validator()
    lo = {
        "id": "CO-01",
        "statement": "Apply the photosynthesis cycle to a novel ecosystem.",
        "bloomLevel": "apply",
        "bloomVerb": "apply",
        "cognitiveDomain": "procedural",
        "keyConcepts": ["photosynthesis-cycle", "ecosystem"],
        "targetedConcepts": [
            {"concept": "photosynthesis-cycle", "bloomLevel": "apply"},
            {"concept": "ecosystem", "bloomLevel": "apply"},
        ],
    }
    errors = sorted(validator.iter_errors(lo), key=lambda e: list(e.absolute_path))
    assert not errors, f"Unexpected schema errors: {[e.message for e in errors]}"


def test_schema_rejects_targeted_concept_missing_bloom_level():
    """Edge entries must carry bloomLevel — no bare concept slugs allowed."""
    validator = _lo_validator()
    lo = {
        "id": "CO-02",
        "statement": "Explain the water cycle.",
        "bloomLevel": "understand",
        "bloomVerb": "explain",
        "cognitiveDomain": "conceptual",
        "targetedConcepts": [{"concept": "water-cycle"}],
    }
    errors = list(validator.iter_errors(lo))
    assert errors, "Edge missing bloomLevel should fail schema validation"


def test_schema_rejects_targeted_concept_invalid_bloom_level():
    """bloomLevel on each edge must be a canonical enum member."""
    validator = _lo_validator()
    lo = {
        "id": "CO-03",
        "statement": "Analyze the market trends.",
        "bloomLevel": "analyze",
        "bloomVerb": "analyze",
        "cognitiveDomain": "conceptual",
        "targetedConcepts": [{"concept": "market", "bloomLevel": "bogus"}],
    }
    errors = list(validator.iter_errors(lo))
    assert errors, "Invalid bloomLevel enum value should fail schema validation"


def test_schema_backward_compatible_without_field():
    """Legacy LO payloads lacking targetedConcepts must still validate."""
    validator = _lo_validator()
    lo = {
        "id": "CO-04",
        "statement": "Recall the parts of a cell.",
        "bloomLevel": "remember",
        "bloomVerb": "recall",
        "cognitiveDomain": "factual",
        "keyConcepts": ["cell-parts"],
    }
    errors = list(validator.iter_errors(lo))
    assert not errors, (
        f"Legacy LO without targetedConcepts should validate; got "
        f"{[e.message for e in errors]}"
    )


# ---------------------------------------------------------------------- #
# 2. Emit round-trip — _build_objectives_metadata populates edges
# ---------------------------------------------------------------------- #


def test_build_objectives_emits_targeted_concepts_from_key_concepts_x_bloom():
    """Every LO with key_concepts AND bloom_level emits targetedConcepts."""
    objectives = [
        {
            "id": "CO-01",
            "statement": "Apply the framework to novel scenarios.",
            "bloom_level": "apply",
            "bloom_verb": "apply",
            "key_concepts": ["Framework", "Novel Scenarios"],
        }
    ]
    result = _build_objectives_metadata(objectives)
    assert len(result) == 1
    lo = result[0]
    assert "targetedConcepts" in lo, (
        f"Expected targetedConcepts on LO with key_concepts + bloom_level; "
        f"got {lo!r}"
    )
    # Edges carry the inherited bloomLevel
    assert lo["targetedConcepts"] == [
        {"concept": "framework", "bloomLevel": "apply"},
        {"concept": "novel-scenarios", "bloomLevel": "apply"},
    ]


def test_build_objectives_targeted_concepts_subset_of_key_concepts():
    """Invariant: every slug in targetedConcepts appears in keyConcepts."""
    objectives = [
        {
            "id": "CO-05",
            "statement": "Evaluate the merits of two approaches.",
            "bloom_level": "evaluate",
            "bloom_verb": "evaluate",
            "key_concepts": ["Approach A", "Approach B", "Merit Criteria"],
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    targeted_slugs = {edge["concept"] for edge in lo["targetedConcepts"]}
    key_slugs = set(lo["keyConcepts"])
    assert targeted_slugs <= key_slugs, (
        f"targetedConcepts slugs {targeted_slugs} must be a subset of "
        f"keyConcepts slugs {key_slugs}"
    )
    assert targeted_slugs == key_slugs, (
        "Wave 57 emits one edge per keyConcept — sets should be equal "
        "until a future wave introduces edge-specific filtering."
    )


def test_build_objectives_bloom_level_inherited_on_every_edge():
    """Every edge's bloomLevel equals the parent LO's bloomLevel."""
    objectives = [
        {
            "id": "CO-06",
            "statement": "Design a data pipeline.",
            "bloom_level": "create",
            "bloom_verb": "design",
            "key_concepts": ["Pipeline", "Data Source"],
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    parent_level = lo["bloomLevel"]
    for edge in lo["targetedConcepts"]:
        assert edge["bloomLevel"] == parent_level, (
            f"Edge bloomLevel {edge['bloomLevel']!r} must equal parent "
            f"LO bloomLevel {parent_level!r}"
        )


# ---------------------------------------------------------------------- #
# 3. Elision cases
# ---------------------------------------------------------------------- #


def test_build_objectives_elides_targeted_concepts_without_bloom_level():
    """No Bloom level → no targetedConcepts (can't tag the edge)."""
    # Statement with no detectable Bloom verb forces bloom_level to None.
    objectives = [
        {
            "id": "CO-07",
            "statement": "Mystery statement with no taxonomy verb here.",
            "key_concepts": ["Some Concept"],
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert lo.get("bloomLevel") is None
    assert "targetedConcepts" not in lo, (
        f"targetedConcepts must be elided when bloom_level is None; got {lo!r}"
    )


def test_build_objectives_elides_targeted_concepts_without_key_concepts():
    """No key concepts → no targetedConcepts (nothing to link)."""
    objectives = [
        {
            "id": "CO-08",
            "statement": "Analyze the outcomes of the experiment.",
            "bloom_level": "analyze",
            "bloom_verb": "analyze",
            # no key_concepts
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert "targetedConcepts" not in lo, (
        f"targetedConcepts must be elided when key_concepts is missing; "
        f"got {lo!r}"
    )


def test_build_objectives_elides_targeted_concepts_with_empty_key_concepts():
    """Empty key_concepts list → no targetedConcepts."""
    objectives = [
        {
            "id": "CO-09",
            "statement": "Apply the framework.",
            "bloom_level": "apply",
            "bloom_verb": "apply",
            "key_concepts": [],
        }
    ]
    result = _build_objectives_metadata(objectives)
    lo = result[0]
    assert "targetedConcepts" not in lo
    assert "keyConcepts" not in lo, (
        "Empty key_concepts should also elide keyConcepts (existing behavior)"
    )


# ---------------------------------------------------------------------- #
# 4. End-to-end — JSON-LD block in generated HTML carries the edges
# ---------------------------------------------------------------------- #


def test_generated_page_jsonld_carries_targeted_concepts(tmp_path):
    """A full generate_week run emits targetedConcepts inside the JSON-LD.

    Proves the field survives the round trip from objectives dict → JSON-LD
    block → serialized HTML file, not just the helper function.
    """
    week_data = {
        "week_number": 1,
        "title": "Foundations",
        "objectives": [
            {
                "id": "CO-01",
                "statement": "Apply the framework to sample data.",
                "bloom_level": "apply",
                "bloom_verb": "apply",
                "key_concepts": ["Framework", "Sample Data"],
            },
            {
                "id": "CO-02",
                "statement": "Analyze the results of the sample run.",
                "bloom_level": "analyze",
                "bloom_verb": "analyze",
                "key_concepts": ["Results"],
            },
        ],
        "overview_text": ["Intro paragraph."],
        "readings": ["Ch. 1 pp. 1-20"],
        "content_modules": [
            {
                "title": "Basics",
                "sections": [
                    {
                        "heading": "Definition",
                        "content_type": "definition",
                        "paragraphs": ["Some definition text."],
                    },
                ],
            }
        ],
        "activities": [],
        "key_takeaways": ["Framework is foundational."],
        "reflection_questions": ["How would you apply this?"],
    }
    out = tmp_path / "out"
    generate_course.generate_week(week_data, out, "TEST_101", source_module_map=None)

    # The overview page carries the page-level learningObjectives.
    overview_html = (out / "week_01" / "week_01_overview.html").read_text(
        encoding="utf-8"
    )
    jsonld_blocks = re.findall(
        r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
        overview_html,
        flags=re.DOTALL,
    )
    assert jsonld_blocks, "Generated overview.html emitted no JSON-LD blocks"
    parsed = [json.loads(b) for b in jsonld_blocks]
    with_los = [p for p in parsed if p.get("learningObjectives")]
    assert with_los, "No JSON-LD block on overview.html carried learningObjectives"
    los = with_los[0]["learningObjectives"]
    assert len(los) == 2
    for lo in los:
        assert "targetedConcepts" in lo, (
            f"Generated JSON-LD LO missing targetedConcepts: {lo!r}"
        )
        parent_level = lo["bloomLevel"]
        parent_slugs = set(lo["keyConcepts"])
        for edge in lo["targetedConcepts"]:
            assert edge["bloomLevel"] == parent_level
            assert edge["concept"] in parent_slugs
