"""GPT Feedback v2 Wave 1 / W1.A — Courseforge JSON-LD ``observedBloomLevel`` /
``bloomAlignment`` schema extensions.

Confirms ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` accepts the
three new optional Bloom-related fields on ``$defs.AssessmentItem``
(``bloomLevel``, ``observedBloomLevel``, ``bloomAlignment``) and the two new
fields on ``$defs.Block`` (``observedBloomLevel``, ``bloomAlignment``), while
preserving each shape's ``additionalProperties: false`` contract.

Backward compatibility: every new field is optional. Pre-Wave-1 dicts (no
new fields) MUST validate untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
PAGE_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "courseforge_jsonld_v1.schema.json"


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _build_validator_for_def(def_name: str):
    """Return a Draft202012Validator that validates against ``$defs/<def_name>``.

    Uses a $ref-only top-level schema so we can validate a sub-shape
    directly while still resolving cross-schema $refs (BloomLevel taxonomy)
    through the registry.
    """
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    page = json.loads(PAGE_SCHEMA_PATH.read_text())

    # Build a registry covering every JSON schema we ship so taxonomy /
    # source_reference $refs resolve. Mirrors the loader pattern in
    # test_chunk_v4_bloom_secondary.py.
    id_to_schema: Dict[str, Any] = {}
    for p in SCHEMAS_DIR.rglob("*.json"):
        try:
            s = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        sid = s.get("$id")
        if sid:
            id_to_schema[sid] = s

    resources = [
        (sid, Resource.from_contents(s, default_specification=DRAFT202012))
        for sid, s in id_to_schema.items()
    ]
    registry = Registry().with_resources(resources)
    sub_schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$ref": page["$id"] + "#/$defs/" + def_name,
    }
    return Draft202012Validator(sub_schema, registry=registry)


def _legacy_assessment_item() -> Dict[str, Any]:
    return {
        "stem": "What does sh:datatype constrain?",
        "answer_key": "The literal datatype of a property's object.",
        "distractors": [
            {"text": "The literal datatype of a property's object."},
            {"text": "The cardinality of a property."},
            {"text": "The class of a focus node."},
        ],
        "correct_answer_index": 0,
    }


def _legacy_block() -> Dict[str, Any]:
    return {
        "blockId": "week_01_overview#objective_lo_one_0",
        "blockType": "objective",
        "sequence": 0,
    }


# ---------------------------------------------------------------------- #
# AssessmentItem
# ---------------------------------------------------------------------- #


def test_legacy_assessment_item_validates():
    """Pre-Wave-1 AssessmentItem (no Bloom fields) must validate untouched."""
    validator = _build_validator_for_def("AssessmentItem")
    item = _legacy_assessment_item()
    errors = list(validator.iter_errors(item))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


def test_assessment_item_with_all_three_bloom_fields_validates():
    validator = _build_validator_for_def("AssessmentItem")
    item = _legacy_assessment_item()
    item["bloomLevel"] = "apply"
    item["observedBloomLevel"] = "apply"
    item["bloomAlignment"] = True
    errors = list(validator.iter_errors(item))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("bloomLevel", "remember"),
        ("bloomLevel", None),
        ("observedBloomLevel", "evaluate"),
        ("observedBloomLevel", None),
        ("bloomAlignment", True),
        ("bloomAlignment", False),
        ("bloomAlignment", None),
    ],
)
def test_assessment_item_with_one_optional_bloom_field_validates(field_name, value):
    """Each new field is independently optional."""
    validator = _build_validator_for_def("AssessmentItem")
    item = _legacy_assessment_item()
    item[field_name] = value
    errors = list(validator.iter_errors(item))
    assert errors == [], (
        f"field={field_name} value={value!r} unexpectedly errored: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_assessment_item_rejects_non_canonical_bloom_level():
    validator = _build_validator_for_def("AssessmentItem")
    item = _legacy_assessment_item()
    item["bloomLevel"] = "synthesize"  # legacy Bloom verb, not canonical
    errors = list(validator.iter_errors(item))
    assert errors, "expected schema rejection of non-canonical bloomLevel"


def test_assessment_item_rejects_non_canonical_observed_bloom_level():
    validator = _build_validator_for_def("AssessmentItem")
    item = _legacy_assessment_item()
    item["observedBloomLevel"] = "memorize"
    errors = list(validator.iter_errors(item))
    assert errors, "expected schema rejection of non-canonical observedBloomLevel"


def test_assessment_item_additional_properties_still_rejected():
    """Regression test for the $defs.AssessmentItem.additionalProperties: false
    contract — confirms we didn't accidentally widen it when adding the new
    Bloom fields."""
    validator = _build_validator_for_def("AssessmentItem")
    item = _legacy_assessment_item()
    item["unknownField"] = "should-fail"
    errors = list(validator.iter_errors(item))
    assert errors, "expected additionalProperties: false to reject unknown key"


# ---------------------------------------------------------------------- #
# Block
# ---------------------------------------------------------------------- #


def test_legacy_block_validates():
    validator = _build_validator_for_def("Block")
    block = _legacy_block()
    errors = list(validator.iter_errors(block))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


def test_block_with_observed_bloom_fields_validates():
    validator = _build_validator_for_def("Block")
    block = _legacy_block()
    block["bloomLevel"] = "apply"
    block["observedBloomLevel"] = "understand"
    block["bloomAlignment"] = False
    errors = list(validator.iter_errors(block))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


@pytest.mark.parametrize(
    "field_name,value",
    [
        ("observedBloomLevel", "create"),
        ("observedBloomLevel", None),
        ("bloomAlignment", True),
        ("bloomAlignment", None),
    ],
)
def test_block_with_one_optional_bloom_field_validates(field_name, value):
    validator = _build_validator_for_def("Block")
    block = _legacy_block()
    block[field_name] = value
    errors = list(validator.iter_errors(block))
    assert errors == [], (
        f"field={field_name} value={value!r} unexpectedly errored: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_block_rejects_non_canonical_observed_bloom_level():
    validator = _build_validator_for_def("Block")
    block = _legacy_block()
    block["observedBloomLevel"] = "synthesize"
    errors = list(validator.iter_errors(block))
    assert errors, "expected schema rejection of non-canonical observedBloomLevel"


def test_block_additional_properties_still_rejected():
    validator = _build_validator_for_def("Block")
    block = _legacy_block()
    block["unknownField"] = "should-fail"
    errors = list(validator.iter_errors(block))
    assert errors, "expected additionalProperties: false to reject unknown key on Block"
