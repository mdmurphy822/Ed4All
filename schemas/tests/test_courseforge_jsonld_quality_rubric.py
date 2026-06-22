"""IB2.4 — Courseforge JSON-LD quality-rubric (``$defs.QualityScore``) +
optional additive ``qualityRubric`` / ``frameworkBlock`` properties on
``$defs.Block``.

Confirms ``schemas/knowledge/courseforge_jsonld_v1.schema.json``:
  * ACCEPTS a Block carrying a populated ``qualityRubric`` + ``frameworkBlock``;
  * REJECTS a QualityScore with score out of 0–3 or a bogus dimension;
  * still ACCEPTS a Block carrying NEITHER key (the default-empty case this
    wave always emits) — proving both keys are optional + additive.

Mirrors the loader pattern in test_courseforge_jsonld_anatomy.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
PAGE_SCHEMA_PATH = SCHEMAS_DIR / "knowledge" / "courseforge_jsonld_v1.schema.json"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ib2"


def _build_validator_for_def(def_name: str):
    pytest.importorskip("jsonschema")
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
# Block — populated rubric + frameworkBlock
# ---------------------------------------------------------------------- #


def test_legacy_block_without_rubric_validates():
    """The default-empty case (no qualityRubric / frameworkBlock) validates."""
    validator = _build_validator_for_def("Block")
    errors = list(validator.iter_errors(_legacy_block()))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


def test_block_with_quality_rubric_validates_from_fixture():
    validator = _build_validator_for_def("Block")
    block = json.loads((FIXTURES_DIR / "block_with_quality_rubric_accept.json").read_text())
    errors = list(validator.iter_errors(block))
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


def test_block_quality_rubric_score_out_of_range_rejected():
    validator = _build_validator_for_def("Block")
    block = json.loads(
        (FIXTURES_DIR / "block_quality_rubric_reject_score_out_of_range.json").read_text()
    )
    errors = list(validator.iter_errors(block))
    assert errors, "expected score:4 (out of 0–3) to reject"


def test_block_quality_rubric_bogus_dimension_rejected():
    validator = _build_validator_for_def("Block")
    block = json.loads(
        (FIXTURES_DIR / "block_quality_rubric_reject_bogus_dimension.json").read_text()
    )
    errors = list(validator.iter_errors(block))
    assert errors, "expected dimension:'bogus' to reject"


@pytest.mark.parametrize("code", ["B01", "B07", "B15", None])
def test_block_framework_block_accepts_valid_codes(code):
    validator = _build_validator_for_def("Block")
    block = _legacy_block()
    block["frameworkBlock"] = code
    errors = list(validator.iter_errors(block))
    assert errors == [], f"frameworkBlock={code!r} unexpectedly errored"


@pytest.mark.parametrize("code", ["B00", "B16", "b01", "B1"])
def test_block_framework_block_rejects_invalid_codes(code):
    validator = _build_validator_for_def("Block")
    block = _legacy_block()
    block["frameworkBlock"] = code
    errors = list(validator.iter_errors(block))
    assert errors, f"expected frameworkBlock={code!r} to reject"


# ---------------------------------------------------------------------- #
# QualityScore — sub-shape
# ---------------------------------------------------------------------- #


def test_quality_score_requires_core_keys():
    validator = _build_validator_for_def("QualityScore")
    errors = list(validator.iter_errors({"dimension": "alignment"}))
    assert errors, "expected missing score + applicable to reject"


def test_quality_score_null_score_validates():
    validator = _build_validator_for_def("QualityScore")
    errors = list(
        validator.iter_errors({"dimension": "multimedia", "score": None, "applicable": False})
    )
    assert errors == [], f"unexpected errors: {[(e.absolute_path, e.message) for e in errors]}"


def test_quality_score_rejects_unknown_key():
    validator = _build_validator_for_def("QualityScore")
    errors = list(
        validator.iter_errors(
            {"dimension": "alignment", "score": 1, "applicable": True, "bogus": 1}
        )
    )
    assert errors, "expected additionalProperties:false to reject unknown key"
