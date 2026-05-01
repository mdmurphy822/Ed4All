"""Wave 137b - tests for ``lib.ontology.family_map.load_family_map``.

Seven tests pin the loader contract:

1. ``test_load_family_map_returns_none_when_missing`` - Map file absent
   returns ``None`` so the validator no-ops cleanly.
2. ``test_load_valid_family_map`` - Synthetic valid map round-trips
   through partition validation and exposes ``family_of`` reverse index.
3. ``test_rejects_double_listed_curie_across_families`` - Same CURIE in
   two families raises ``ValueError``.
4. ``test_rejects_family_singleton_collision`` - CURIE in a family AND
   the singletons list raises ``ValueError``.
5. ``test_rejects_family_with_under_two_curies`` - Single-CURIE family
   raises ``ValueError`` (must move to singletons).
6. ``test_rejects_curie_not_in_manifest_when_manifest_present`` - When
   a property manifest exists for the family, a family-map CURIE that
   isn't in the manifest raises ``ValueError``.
7. ``test_rdf_shacl_fixture_validates_against_schema`` - The shipped
   ``family_map.rdf_shacl.yaml`` fixture passes JSON Schema validation
   AND every CURIE it references is in the rdf_shacl property manifest.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml  # noqa: E402

from lib.ontology import family_map as fm_mod  # noqa: E402
from lib.ontology.family_map import (  # noqa: E402
    FamilyMap,
    compute_family_coverage,
    load_family_map,
)


def _patch_schema_dir(monkeypatch, target: Path) -> None:
    """Redirect the loader's _SCHEMA_DIR to a tmp dir + clear cache."""
    monkeypatch.setattr(fm_mod, "_SCHEMA_DIR", target)
    load_family_map.cache_clear()


def _write(target: Path, payload: dict) -> None:
    target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


# ----------------------------------------------------------------------
# Test 1: missing file -> None
# ----------------------------------------------------------------------


def test_load_family_map_returns_none_when_missing(monkeypatch, tmp_path):
    """No family_map.<family>.yaml present -> loader returns None."""
    _patch_schema_dir(monkeypatch, tmp_path)
    assert load_family_map("does_not_exist") is None


# ----------------------------------------------------------------------
# Test 2: valid map round-trip
# ----------------------------------------------------------------------


def test_load_valid_family_map(monkeypatch, tmp_path):
    """A valid YAML loads + exposes families, singletons, and reverse index."""
    _patch_schema_dir(monkeypatch, tmp_path)
    # Copy the real schema into the tmp dir so JSON Schema validation
    # runs against the canonical schema.
    real_schema = (
        PROJECT_ROOT / "schemas" / "training" / "family_map.schema.json"
    )
    (tmp_path / "family_map.schema.json").write_text(
        real_schema.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write(
        tmp_path / "family_map.test_family.yaml",
        {
            "family": "test_family",
            "families": {
                "alpha": ["test:A1", "test:A2"],
                "beta": ["test:B1", "test:B2", "test:B3"],
            },
            "singletons": ["test:S1", "test:S2"],
        },
    )
    fm = load_family_map("test_family")
    assert isinstance(fm, FamilyMap)
    assert fm.family == "test_family"
    assert fm.families == {
        "alpha": ["test:A1", "test:A2"],
        "beta": ["test:B1", "test:B2", "test:B3"],
    }
    assert fm.singletons == ["test:S1", "test:S2"]
    assert fm.family_of["test:A1"] == "alpha"
    assert fm.family_of["test:A2"] == "alpha"
    assert fm.family_of["test:B3"] == "beta"
    assert fm.family_of["test:S1"] == "<singleton>"


# ----------------------------------------------------------------------
# Test 3: double-listed CURIE across families
# ----------------------------------------------------------------------


def test_rejects_double_listed_curie_across_families(monkeypatch, tmp_path):
    """Same CURIE in two families raises ValueError."""
    _patch_schema_dir(monkeypatch, tmp_path)
    real_schema = (
        PROJECT_ROOT / "schemas" / "training" / "family_map.schema.json"
    )
    (tmp_path / "family_map.schema.json").write_text(
        real_schema.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write(
        tmp_path / "family_map.test_family.yaml",
        {
            "family": "test_family",
            "families": {
                "alpha": ["test:Shared", "test:A1"],
                "beta": ["test:Shared", "test:B1"],
            },
            "singletons": [],
        },
    )
    with pytest.raises(ValueError, match="appears in two families"):
        load_family_map("test_family")


# ----------------------------------------------------------------------
# Test 4: family vs singleton collision
# ----------------------------------------------------------------------


def test_rejects_family_singleton_collision(monkeypatch, tmp_path):
    """CURIE listed in a family AND singletons raises ValueError."""
    _patch_schema_dir(monkeypatch, tmp_path)
    real_schema = (
        PROJECT_ROOT / "schemas" / "training" / "family_map.schema.json"
    )
    (tmp_path / "family_map.schema.json").write_text(
        real_schema.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write(
        tmp_path / "family_map.test_family.yaml",
        {
            "family": "test_family",
            "families": {
                "alpha": ["test:X", "test:A1"],
            },
            "singletons": ["test:X", "test:S1"],
        },
    )
    with pytest.raises(ValueError, match="AND singletons"):
        load_family_map("test_family")


# ----------------------------------------------------------------------
# Test 5: family with <2 CURIEs
# ----------------------------------------------------------------------


def test_rejects_family_with_under_two_curies(monkeypatch, tmp_path):
    """Single-CURIE family is rejected by JSON Schema (minItems: 2).

    Both the schema-level constraint and the partition-level constraint
    in :func:`_validate_partition` enforce ``len(curies) >= 2`` — a
    family with 0 or 1 entries is not a family.
    """
    _patch_schema_dir(monkeypatch, tmp_path)
    real_schema = (
        PROJECT_ROOT / "schemas" / "training" / "family_map.schema.json"
    )
    (tmp_path / "family_map.schema.json").write_text(
        real_schema.read_text(encoding="utf-8"), encoding="utf-8"
    )
    _write(
        tmp_path / "family_map.test_family.yaml",
        {
            "family": "test_family",
            "families": {
                "alpha": ["test:Lone"],
            },
            "singletons": [],
        },
    )
    # JSON Schema fails before partition validation, so we accept either
    # error class — both encode the same constraint.
    with pytest.raises(Exception):
        load_family_map("test_family")


# ----------------------------------------------------------------------
# Test 6: CURIE not in manifest when manifest present
# ----------------------------------------------------------------------


def test_rejects_curie_not_in_manifest_when_manifest_present(monkeypatch, tmp_path):
    """Family-map CURIE missing from the property manifest raises ValueError."""
    _patch_schema_dir(monkeypatch, tmp_path)
    real_schema = (
        PROJECT_ROOT / "schemas" / "training" / "family_map.schema.json"
    )
    (tmp_path / "family_map.schema.json").write_text(
        real_schema.read_text(encoding="utf-8"), encoding="utf-8"
    )
    # Manifest lists test:A1 / test:A2 only; the family map references
    # test:Bogus which is not in the manifest.
    (tmp_path / "property_manifest.test_family.yaml").write_text(
        yaml.safe_dump(
            {
                "family": "test_family",
                "properties": [
                    {
                        "id": "a1", "uri": "http://x/a1", "curie": "test:A1",
                        "label": "A1", "surface_forms": ["test:A1"],
                        "min_pairs": 2,
                    },
                    {
                        "id": "a2", "uri": "http://x/a2", "curie": "test:A2",
                        "label": "A2", "surface_forms": ["test:A2"],
                        "min_pairs": 2,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write(
        tmp_path / "family_map.test_family.yaml",
        {
            "family": "test_family",
            "families": {
                "alpha": ["test:A1", "test:Bogus"],
            },
            "singletons": ["test:A2"],
        },
    )
    with pytest.raises(ValueError, match="not declared in the property manifest"):
        load_family_map("test_family")


# ----------------------------------------------------------------------
# Test 7: shipped rdf_shacl fixture validates against schema
# ----------------------------------------------------------------------


def test_rdf_shacl_fixture_validates_against_schema():
    """The shipped family_map.rdf_shacl.yaml validates + matches the manifest.

    Cleared cache + load against the canonical (un-monkey-patched)
    _SCHEMA_DIR. Asserts:
      - Loader returns a FamilyMap (no exception).
      - 10 families + 11 singletons (Plan B contract).
      - Every CURIE referenced exists in the rdf_shacl property manifest.
    """
    load_family_map.cache_clear()
    fm = load_family_map("rdf_shacl")
    assert isinstance(fm, FamilyMap), "shipped fixture failed to load"
    assert fm.family == "rdf_shacl"
    assert len(fm.families) == 10, (
        f"expected 10 families per Plan B; got {len(fm.families)}"
    )
    assert len(fm.singletons) == 11, (
        f"expected 11 singletons per Plan B; got {len(fm.singletons)}"
    )
    # Cross-check: every CURIE in the family map should resolve via family_of.
    total_curies = sum(len(v) for v in fm.families.values()) + len(fm.singletons)
    assert total_curies == 40, (
        f"expected 40 CURIEs (matching property manifest); got {total_curies}"
    )
    # Manifest cross-check ran during load_family_map (the loader raises
    # if a CURIE is missing) — if we got here, the fixture is consistent
    # with the rdf_shacl property manifest.


# ----------------------------------------------------------------------
# compute_family_coverage helper
# ----------------------------------------------------------------------


def test_compute_family_coverage_classifies_complete_partial_untouched():
    """compute_family_coverage returns the right per-family status."""
    fm = FamilyMap(
        family="test",
        families={
            "all_complete": ["test:A1", "test:A2"],
            "all_untouched": ["test:U1", "test:U2"],
            "mixed": ["test:M1", "test:M2"],
        },
        singletons=["test:S1"],
        family_of={
            "test:A1": "all_complete",
            "test:A2": "all_complete",
            "test:U1": "all_untouched",
            "test:U2": "all_untouched",
            "test:M1": "mixed",
            "test:M2": "mixed",
            "test:S1": "<singleton>",
        },
    )
    form_data = {
        "test:A1": SimpleNamespace(anchored_status="complete"),
        "test:A2": SimpleNamespace(anchored_status="complete"),
        # all_untouched: both entries missing entirely
        "test:M1": SimpleNamespace(anchored_status="complete"),
        "test:M2": SimpleNamespace(anchored_status="degraded_placeholder"),
    }
    out = compute_family_coverage(fm, form_data)
    assert out["all_complete"]["status"] == "complete"
    assert out["all_complete"]["complete"] == 2
    assert out["all_complete"]["total"] == 2
    assert out["all_untouched"]["status"] == "untouched"
    assert out["all_untouched"]["complete"] == 0
    assert out["mixed"]["status"] == "partial"
    assert out["mixed"]["complete"] == 1
