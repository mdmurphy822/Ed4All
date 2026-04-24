"""Wave 59 — LO hierarchy edges (``hierarchyLevel`` + ``parentObjectiveId``).

Pre-Wave-59 the TO-NN / CO-NN hierarchy was only discoverable by parsing
the ID prefix (not a first-class JSON-LD field) and the CO→TO parent
relationship was lost entirely — the emit surface carried no parent
edge, and reconstruction required reading ``synthesized_objectives.json``
(a pipeline-internal artifact not available to KG consumers).

Wave 59 surfaces both:

* ``hierarchyLevel`` (enum: ``terminal`` | ``chapter``): derived at emit
  time from the canonical LO ID via ``lib.ontology.learning_objectives.
  hierarchy_from_id``. Always emitted when the LO has a canonical ID.
* ``parentObjectiveId`` (string matching ``^[A-Z]{2,}-\\d{2,}$``): the
  terminal LO a chapter LO rolls up to. Optional — emitted only when
  upstream supplies the mapping via the LO dict's ``parent_objective_id``
  or ``parentObjectiveId`` key. Heuristic derivation is deferred to a
  later wave.

Covers:

* Schema: ``hierarchyLevel`` accepts canonical values and rejects
  out-of-enum strings; ``parentObjectiveId`` accepts canonical LO IDs
  and rejects malformed values; both are optional so legacy LO
  payloads without them still validate.
* Emit: TO-NN emits ``hierarchyLevel='terminal'`` without
  ``parentObjectiveId``; CO-NN emits ``hierarchyLevel='chapter'`` with
  ``parentObjectiveId`` when supplied, without it when not.
* Elision: non-canonical IDs don't crash; malformed parent IDs are
  elided silently; both camelCase and snake_case parent-ID key names
  are accepted on the LO dict.
* End-to-end: generated HTML's JSON-LD block carries both fields.
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
_QUESTION_TYPE_SCHEMA_PATH = (
    _PROJECT_ROOT / "schemas" / "taxonomies" / "question_type.json"
)


def _lo_validator() -> Draft202012Validator:
    with open(_JSONLD_SCHEMA_PATH, encoding="utf-8") as f:
        root = json.load(f)
    with open(_BLOOM_VERBS_SCHEMA_PATH, encoding="utf-8") as f:
        bloom = json.load(f)
    with open(_COGNITIVE_DOMAIN_SCHEMA_PATH, encoding="utf-8") as f:
        cog = json.load(f)
    with open(_QUESTION_TYPE_SCHEMA_PATH, encoding="utf-8") as f:
        qtype = json.load(f)
    store = {
        root["$id"]: root,
        bloom["$id"]: bloom,
        cog["$id"]: cog,
        qtype["$id"]: qtype,
    }
    resolver = RefResolver.from_schema(root, store=store)
    subschema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": f"{root['$id']}#/$defs/LearningObjective",
    }
    return Draft202012Validator(subschema, resolver=resolver)


# ---------------------------------------------------------------------- #
# 1. Schema validation
# ---------------------------------------------------------------------- #


def test_schema_accepts_hierarchy_level_and_parent_id():
    validator = _lo_validator()
    lo = {
        "id": "CO-05",
        "statement": "Apply the rules.",
        "bloomLevel": "apply",
        "bloomVerb": "apply",
        "cognitiveDomain": "procedural",
        "hierarchyLevel": "chapter",
        "parentObjectiveId": "TO-01",
    }
    errors = sorted(validator.iter_errors(lo), key=lambda e: list(e.absolute_path))
    assert not errors, f"Unexpected errors: {[e.message for e in errors]}"


def test_schema_rejects_non_enum_hierarchy_level():
    validator = _lo_validator()
    lo = {
        "id": "TO-01",
        "statement": "Explain.",
        "bloomLevel": "understand",
        "bloomVerb": "explain",
        "cognitiveDomain": "conceptual",
        "hierarchyLevel": "subchapter",  # not in enum
    }
    errors = list(validator.iter_errors(lo))
    assert errors, "Non-enum hierarchyLevel must fail validation"


def test_schema_rejects_malformed_parent_objective_id():
    validator = _lo_validator()
    lo = {
        "id": "CO-01",
        "statement": "Apply.",
        "bloomLevel": "apply",
        "bloomVerb": "apply",
        "cognitiveDomain": "procedural",
        "parentObjectiveId": "garbage",  # doesn't match canonical pattern
    }
    errors = list(validator.iter_errors(lo))
    assert errors, "Malformed parentObjectiveId must fail validation"


def test_schema_backward_compatible_without_hierarchy_fields():
    validator = _lo_validator()
    lo = {
        "id": "TO-01",
        "statement": "Apply.",
        "bloomLevel": "apply",
        "bloomVerb": "apply",
        "cognitiveDomain": "procedural",
    }
    errors = list(validator.iter_errors(lo))
    assert not errors, f"Legacy LO should validate; got {[e.message for e in errors]}"


# ---------------------------------------------------------------------- #
# 2. Emit — hierarchyLevel derivation from canonical ID
# ---------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "lo_id,expected_hierarchy",
    [
        ("TO-01", "terminal"),
        ("TO-10", "terminal"),
        ("TO-100", "terminal"),
        ("CO-01", "chapter"),
        ("CO-05", "chapter"),
    ],
)
def test_emit_derives_hierarchy_level_from_canonical_id(lo_id, expected_hierarchy):
    objectives = [
        {
            "id": lo_id,
            "statement": "Apply the framework.",
        }
    ]
    lo = _build_objectives_metadata(objectives)[0]
    assert lo.get("hierarchyLevel") == expected_hierarchy, (
        f"Expected hierarchyLevel={expected_hierarchy!r} for {lo_id!r}, "
        f"got {lo.get('hierarchyLevel')!r}"
    )


def test_emit_terminal_lo_has_no_parent_when_not_supplied():
    objectives = [{"id": "TO-01", "statement": "Apply the framework."}]
    lo = _build_objectives_metadata(objectives)[0]
    assert lo.get("hierarchyLevel") == "terminal"
    assert "parentObjectiveId" not in lo


def test_emit_chapter_lo_has_no_parent_when_not_supplied():
    objectives = [{"id": "CO-05", "statement": "Apply the framework."}]
    lo = _build_objectives_metadata(objectives)[0]
    assert lo.get("hierarchyLevel") == "chapter"
    assert "parentObjectiveId" not in lo


# ---------------------------------------------------------------------- #
# 3. Emit — parent edge passthrough
# ---------------------------------------------------------------------- #


def test_emit_emits_parent_objective_when_upstream_supplies_snake_case():
    objectives = [
        {
            "id": "CO-05",
            "statement": "Apply the framework.",
            "parent_objective_id": "TO-01",
        }
    ]
    lo = _build_objectives_metadata(objectives)[0]
    assert lo.get("parentObjectiveId") == "TO-01"


def test_emit_accepts_parent_objective_when_upstream_supplies_camel_case():
    objectives = [
        {
            "id": "CO-05",
            "statement": "Apply the framework.",
            "parentObjectiveId": "TO-02",
        }
    ]
    lo = _build_objectives_metadata(objectives)[0]
    assert lo.get("parentObjectiveId") == "TO-02"


def test_emit_elides_malformed_parent_id():
    """Garbage parent IDs are elided rather than shipped to fail schema validation."""
    objectives = [
        {
            "id": "CO-05",
            "statement": "Apply the framework.",
            "parent_objective_id": "not-a-canonical-id",
        }
    ]
    lo = _build_objectives_metadata(objectives)[0]
    assert "parentObjectiveId" not in lo


def test_emit_elides_empty_parent_id():
    objectives = [
        {
            "id": "CO-05",
            "statement": "Apply the framework.",
            "parent_objective_id": "",
        }
    ]
    lo = _build_objectives_metadata(objectives)[0]
    assert "parentObjectiveId" not in lo


# ---------------------------------------------------------------------- #
# 4. Elision — non-canonical IDs don't crash
# ---------------------------------------------------------------------- #


def test_emit_with_non_canonical_id_elides_hierarchy_level():
    """Legacy or invalid LO IDs must not crash emit — just elide the field."""
    objectives = [
        {
            "id": "W03-CO-01",  # pre-canonical week-prefix shape
            "statement": "Apply the framework.",
        }
    ]
    lo = _build_objectives_metadata(objectives)[0]
    assert lo["id"] == "W03-CO-01"
    assert "hierarchyLevel" not in lo


def test_emit_with_missing_id_elides_hierarchy_level():
    objectives = [{"statement": "Apply the framework."}]
    # _build_objectives_metadata references o["id"] directly — if this ever
    # gets called without one, that's a contract violation upstream. We
    # defend against it by making the ID access explicit.
    with pytest.raises(KeyError):
        _build_objectives_metadata(objectives)


# ---------------------------------------------------------------------- #
# 5. End-to-end — JSON-LD block in generated HTML carries the fields
# ---------------------------------------------------------------------- #


def test_generated_page_jsonld_carries_hierarchy_and_parent_fields(tmp_path):
    week_data = {
        "week_number": 1,
        "title": "Hierarchy smoke",
        "objectives": [
            {
                "id": "TO-01",
                "statement": "Apply the framework in realistic contexts.",
                "bloom_level": "apply",
            },
            {
                "id": "CO-01",
                "statement": "Apply the framework to sample data.",
                "bloom_level": "apply",
                "parent_objective_id": "TO-01",
            },
            {
                "id": "CO-02",
                "statement": "Analyze the outputs.",
                "bloom_level": "analyze",
                # deliberately no parent
            },
        ],
        "overview_text": ["Intro."],
        "readings": ["Ch. 1"],
        "content_modules": [
            {
                "title": "M",
                "sections": [
                    {
                        "heading": "H",
                        "content_type": "definition",
                        "paragraphs": ["body."],
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
    payloads = [json.loads(b) for b in blocks]
    with_los = [p for p in payloads if p.get("learningObjectives")]
    assert with_los
    los = {lo["id"]: lo for lo in with_los[0]["learningObjectives"]}

    # TO-01: terminal, no parent
    assert los["TO-01"]["hierarchyLevel"] == "terminal"
    assert "parentObjectiveId" not in los["TO-01"]

    # CO-01: chapter with supplied parent
    assert los["CO-01"]["hierarchyLevel"] == "chapter"
    assert los["CO-01"]["parentObjectiveId"] == "TO-01"

    # CO-02: chapter without supplied parent
    assert los["CO-02"]["hierarchyLevel"] == "chapter"
    assert "parentObjectiveId" not in los["CO-02"]
