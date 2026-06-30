"""Validate the checked-in generic Bloom target curve + its sibling schema."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.ontology.bloom import BLOOM_LEVELS

_TAX_DIR = Path(__file__).resolve().parents[1]
_CURVE_PATH = _TAX_DIR / "bloom_target_distribution.json"
_SCHEMA_PATH = _TAX_DIR / "bloom_target_distribution.schema.json"

_EPS = 1e-6


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_curve_keys_subset_of_bloom_levels():
    doc = _load(_CURVE_PATH)
    shares = doc["target_shares"]
    assert set(shares.keys()) <= set(BLOOM_LEVELS)
    # The canonical default declares every level.
    assert set(shares.keys()) == set(BLOOM_LEVELS)


def test_curve_shares_in_unit_interval_and_sum_to_one():
    shares = _load(_CURVE_PATH)["target_shares"]
    for level, share in shares.items():
        assert 0.0 <= share <= 1.0, level
    total = sum(shares.values())
    assert abs(total - 1.0) < 1e-3


def test_curve_band_knobs_present_and_in_range():
    doc = _load(_CURVE_PATH)
    for key in ("recall_ceiling", "higher_order_floor", "top_heavy_ceiling"):
        assert key in doc, key
        assert 0.0 <= doc[key] <= 1.0


def test_curve_no_corpus_content():
    """Anti-hardcoding: the file is six floats + three knobs, no slugs/paths."""
    doc = _load(_CURVE_PATH)
    allowed = {
        "$schema", "title", "description", "target_shares",
        "recall_ceiling", "higher_order_floor", "top_heavy_ceiling",
    }
    assert set(doc.keys()) <= allowed


def test_curve_validates_against_sibling_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load(_SCHEMA_PATH)
    doc = _load(_CURVE_PATH)
    jsonschema.validate(instance=doc, schema=schema)


def test_schema_rejects_out_of_range_share():
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load(_SCHEMA_PATH)
    bad = {"target_shares": {"apply": 5.0}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_schema_rejects_non_bloom_level_key():
    jsonschema = pytest.importorskip("jsonschema")
    schema = _load(_SCHEMA_PATH)
    bad = {"target_shares": {"bogus_level": 0.5}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)
