"""IB1 — Courseforge JSON-LD six-slot anatomy (``$defs.BlockAnatomy``) +
optional additive ``anatomy`` property on ``$defs.Block``.

Confirms ``schemas/knowledge/courseforge_jsonld_v1.schema.json`` accepts the
new optional ``anatomy`` sub-object on ``$defs.Block`` and the
``$defs.BlockAnatomy`` shape (all-optional string slots + the five-stage
``lifecycle`` map), while preserving each shape's ``additionalProperties:
false`` contract.

Backward compatibility: the ``anatomy`` field is optional. Pre-IB1 blocks
(no ``anatomy`` key) MUST validate untouched. The BODY slot is the block's
content (NOT repeated inside ``anatomy``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
PAGE_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "courseforge_jsonld_v1.schema.json"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ib1"


def _require_jsonschema():
    return pytest.importorskip("jsonschema")


def _build_validator_for_def(def_name: str):
    """Return a Draft202012Validator validating against ``$defs/<def_name>``.

    Mirrors the loader pattern in test_courseforge_jsonld_observed_bloom.py.
    """
    _require_jsonschema()
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource
    from referencing.jsonschema import DRAFT202012

    page = json.loads(PAGE_SCHEMA_PATH.read_text())

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


def _legacy_block() -> Dict[str, Any]:
    return {
        "blockId": "week_01_overview#objective_lo_one_0",
        "blockType": "objective",
        "sequence": 0,
    }


# ---------------------------------------------------------------------- #
# Block — optional additive anatomy property
# ---------------------------------------------------------------------- #


def test_legacy_block_without_anatomy_validates():
    """Pre-IB1 Block (no anatomy key) must validate untouched."""
    validator = _build_validator_for_def("Block")
    errors = list(validator.iter_errors(_legacy_block()))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


def test_block_with_anatomy_validates_from_fixture():
    validator = _build_validator_for_def("Block")
    block = json.loads((FIXTURES_DIR / "block_with_anatomy_accept.json").read_text())
    errors = list(validator.iter_errors(block))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


def test_block_with_anatomy_unknown_slot_rejected_from_fixture():
    validator = _build_validator_for_def("Block")
    block = json.loads(
        (FIXTURES_DIR / "block_with_anatomy_reject_unknown_slot.json").read_text()
    )
    errors = list(validator.iter_errors(block))
    assert errors, "expected $defs.BlockAnatomy additionalProperties:false to reject bogusSlot"


# ---------------------------------------------------------------------- #
# BlockAnatomy — sub-shape
# ---------------------------------------------------------------------- #


def test_empty_anatomy_validates():
    """All slots optional → an empty anatomy object validates."""
    validator = _build_validator_for_def("BlockAnatomy")
    errors = list(validator.iter_errors({}))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


@pytest.mark.parametrize(
    "slot",
    ["heading", "purposeTag", "interaction", "feedback", "transition"],
)
def test_each_anatomy_slot_independently_optional(slot):
    validator = _build_validator_for_def("BlockAnatomy")
    errors = list(validator.iter_errors({slot: "value"}))
    assert errors == [], (
        f"slot={slot} unexpectedly errored: "
        f"{[(e.absolute_path, e.message) for e in errors]}"
    )


def test_anatomy_lifecycle_map_validates():
    validator = _build_validator_for_def("BlockAnatomy")
    anatomy = {
        "heading": "H",
        "lifecycle": {
            "activate": ["heading", "purpose_tag"],
            "present": ["content"],
            "apply": ["interaction"],
            "check": [],
            "consolidate": [],
        },
    }
    errors = list(validator.iter_errors(anatomy))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


def test_anatomy_rejects_unknown_slot_key():
    validator = _build_validator_for_def("BlockAnatomy")
    errors = list(validator.iter_errors({"heading": "H", "bogus": "x"}))
    assert errors, "expected additionalProperties: false to reject unknown slot key"


def test_anatomy_lifecycle_rejects_unknown_stage():
    validator = _build_validator_for_def("BlockAnatomy")
    errors = list(validator.iter_errors({"lifecycle": {"bogus_stage": []}}))
    assert errors, "expected additionalProperties: false to reject unknown lifecycle stage"


def test_block_additional_properties_still_rejected():
    """Confirm $defs.Block.additionalProperties:false unchanged by IB1."""
    validator = _build_validator_for_def("Block")
    block = _legacy_block()
    block["unknownField"] = "should-fail"
    errors = list(validator.iter_errors(block))
    assert errors, "expected additionalProperties: false to reject unknown key on Block"
