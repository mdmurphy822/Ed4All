"""Wave 9 — Courseforge JSON-LD ``sourceReferences`` schema extension.

Confirms ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` accepts
the new ``sourceReferences[]`` slot at both page level and inside
``Section``, and rejects malformed entries. The schema ``$ref``s the
shared ``source_reference.schema.json`` shape so these tests are also a
contract check that the ref-resolution works.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PAGE_SCHEMA = PROJECT_ROOT / "schemas" / "knowledge" / "courseforge_jsonld_v1.schema.json"
SRCREF_SCHEMA = PROJECT_ROOT / "schemas" / "knowledge" / "source_reference.schema.json"


def _require_jsonschema():
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema


_TAXONOMY_FILES = [
    "bloom_verbs.json",
    "module_type.json",
    "content_type.json",
    "cognitive_domain.json",
    "question_type.json",
]


def _validator():
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    page = json.loads(PAGE_SCHEMA.read_text())
    srcref = json.loads(SRCREF_SCHEMA.read_text())
    resources = [
        (srcref["$id"], Resource.from_contents(srcref)),
        (page["$id"], Resource.from_contents(page)),
    ]
    tax_dir = PROJECT_ROOT / "schemas" / "taxonomies"
    for name in _TAXONOMY_FILES:
        tax = json.loads((tax_dir / name).read_text())
        resources.append((tax["$id"], Resource.from_contents(tax)))
    registry = Registry().with_resources(resources)
    return Draft202012Validator(page, registry=registry)


def _base_page() -> dict:
    return {
        "@context": "https://ed4all.dev/ns/courseforge/v1",
        "@type": "CourseModule",
        "courseCode": "SAMPLE_101",
        "weekNumber": 1,
        "moduleType": "content",
        "pageId": "week_01_content_01_intro",
    }


# ---------------------------------------------------------------------- #
# Meta: schema stays a valid draft-2020-12 schema after Wave 9
# ---------------------------------------------------------------------- #


def test_page_schema_remains_valid_draft_2020_12():
    _require_jsonschema()
    from jsonschema import Draft202012Validator

    schema = json.loads(PAGE_SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)


def test_page_schema_declares_page_level_source_references():
    schema = json.loads(PAGE_SCHEMA.read_text())
    assert "sourceReferences" in schema["properties"]
    props = schema["properties"]["sourceReferences"]
    assert props["type"] == "array"
    assert props["items"]["$ref"] == (
        "https://ed4all.dev/schemas/knowledge/source_reference.schema.json"
    )


def test_section_schema_declares_source_references():
    schema = json.loads(PAGE_SCHEMA.read_text())
    section_props = schema["$defs"]["Section"]["properties"]
    assert "sourceReferences" in section_props
    assert section_props["sourceReferences"]["type"] == "array"
    assert section_props["sourceReferences"]["items"]["$ref"] == (
        "https://ed4all.dev/schemas/knowledge/source_reference.schema.json"
    )


def test_section_additional_properties_still_strict():
    """Strict-mode block on unknown keys must survive the additive change."""
    schema = json.loads(PAGE_SCHEMA.read_text())
    assert schema["$defs"]["Section"]["additionalProperties"] is False
    assert schema["additionalProperties"] is False


# ---------------------------------------------------------------------- #
# Positive: pages with + without sourceReferences both validate
# ---------------------------------------------------------------------- #


def test_legacy_page_without_source_refs_validates():
    """Backward compat: pre-Wave-9 pages stay schema-clean."""
    page = _base_page()
    errors = list(_validator().iter_errors(page))
    assert errors == [], [e.message for e in errors]


def test_page_with_empty_source_refs_validates():
    """Empty array is allowed (emitter elides when empty — but accept both)."""
    page = {**_base_page(), "sourceReferences": []}
    errors = list(_validator().iter_errors(page))
    assert errors == [], [e.message for e in errors]


def test_page_with_valid_source_refs_validates():
    page = {
        **_base_page(),
        "sourceReferences": [
            {
                "sourceId": "dart:science_of_learning#s3_c0",
                "role": "primary",
                "confidence": 0.9,
            },
            {
                "sourceId": "dart:science_of_learning#s4_p0",
                "role": "contributing",
            },
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors == [], [e.message for e in errors]


def test_section_with_valid_source_refs_validates():
    page = {
        **_base_page(),
        "sections": [
            {
                "heading": "Definition",
                "contentType": "definition",
                "sourceReferences": [
                    {
                        "sourceId": "dart:science_of_learning#s5_c1",
                        "role": "primary",
                    }
                ],
            }
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors == [], [e.message for e in errors]


def test_fully_populated_source_ref_validates():
    """Every optional SourceReference field supplied."""
    page = {
        **_base_page(),
        "sourceReferences": [
            {
                "sourceId": "dart:foo#a3f9d812ac04bbc1",
                "role": "primary",
                "weight": 0.7,
                "confidence": 0.85,
                "pages": [3, 4],
                "extractor": "pdfplumber",
            }
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors == [], [e.message for e in errors]


# ---------------------------------------------------------------------- #
# Negative: malformed sourceReferences fail
# ---------------------------------------------------------------------- #


def test_source_ref_missing_source_id_fails():
    page = {
        **_base_page(),
        "sourceReferences": [{"role": "primary"}],
    }
    errors = list(_validator().iter_errors(page))
    assert errors, "Missing sourceId must fail"


def test_source_ref_missing_role_fails():
    page = {
        **_base_page(),
        "sourceReferences": [{"sourceId": "dart:slug#s0"}],
    }
    errors = list(_validator().iter_errors(page))
    assert errors, "Missing role must fail"


def test_source_ref_invalid_role_enum_fails():
    page = {
        **_base_page(),
        "sourceReferences": [
            {"sourceId": "dart:slug#s0", "role": "supporting"}
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors, "Role outside enum must fail"


def test_source_ref_invalid_source_id_pattern_fails():
    page = {
        **_base_page(),
        "sourceReferences": [
            {"sourceId": "foobar", "role": "primary"}
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors, "Malformed sourceId pattern must fail"


def test_source_ref_additional_property_fails():
    page = {
        **_base_page(),
        "sourceReferences": [
            {"sourceId": "dart:slug#s0", "role": "primary", "bogus": True}
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors, "Extra property on SourceReference must fail"


def test_section_source_ref_bad_weight_fails():
    page = {
        **_base_page(),
        "sections": [
            {
                "heading": "T",
                "contentType": "definition",
                "sourceReferences": [
                    {"sourceId": "dart:s#b", "role": "primary", "weight": 1.5}
                ],
            }
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors, "weight > 1 must fail"


def test_source_references_wrong_type_fails():
    """sourceReferences must be an array, not a scalar."""
    page = {**_base_page(), "sourceReferences": "dart:s#b"}
    errors = list(_validator().iter_errors(page))
    assert errors


def test_section_with_empty_source_refs_validates():
    """Empty section sourceReferences -> valid (emitter elides; accept either)."""
    page = {
        **_base_page(),
        "sections": [
            {
                "heading": "T",
                "contentType": "definition",
                "sourceReferences": [],
            }
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors == [], [e.message for e in errors]


def test_mixed_roles_in_source_refs_validates():
    """Multiple roles on a single page are legal."""
    page = {
        **_base_page(),
        "sourceReferences": [
            {"sourceId": "dart:slug#s0", "role": "primary"},
            {"sourceId": "dart:slug#s1", "role": "contributing"},
            {"sourceId": "dart:slug#s2", "role": "corroborating"},
        ],
    }
    errors = list(_validator().iter_errors(page))
    assert errors == [], [e.message for e in errors]
