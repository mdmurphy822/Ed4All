"""Wave 137b - tests for ``lib.validators.family_completeness``.

Eight scenarios pin the FamilyCompletenessValidator contract:

1. Passes uniformly complete (all CURIEs in every family complete).
2. Passes uniformly degraded (all CURIEs degraded — no family is
   partially complete).
3. Fails on partial cardinality (sh:minCount complete + sh:maxCount
   degraded).
4. Fails on partial validation_results (sh:result complete + others
   degraded).
5. Skips singletons (a degraded singleton beside a complete singleton
   doesn't fire FAMILY_PARTIALLY_COMPLETE — singletons belong to no
   family).
6. No family map -> validator passes cleanly (FAMILY_MAP_NOT_FOUND
   info issue only).
7. decision_capture fires with metadata-shaped rationale (counts only,
   never CURIE content).
8. Missing CURIE -> fails FAMILY_CURIE_MISSING_FROM_FORM_DATA.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.ontology import family_map as fm_mod  # noqa: E402
from lib.ontology.family_map import FamilyMap  # noqa: E402
from lib.validators.family_completeness import (  # noqa: E402
    FamilyCompletenessValidator,
)


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


def _make_family_map() -> FamilyMap:
    """Synthetic 3-family + 2-singleton map for the test scenarios."""
    families = {
        "cardinality": ["sh:minCount", "sh:maxCount"],
        "validation_results": [
            "sh:ValidationReport",
            "sh:ValidationResult",
            "sh:result",
        ],
        "domain_range": ["rdfs:domain", "rdfs:range"],
    }
    singletons = ["sh:datatype", "sh:nodeKind"]
    family_of: Dict[str, str] = {}
    for fam_name, curies in families.items():
        for c in curies:
            family_of[c] = fam_name
    for c in singletons:
        family_of[c] = "<singleton>"
    return FamilyMap(
        family="rdf_shacl",
        families=families,
        singletons=singletons,
        family_of=family_of,
    )


def _entry(curie: str, status: str) -> Any:
    """Duck-typed FORM_DATA entry stand-in."""
    return SimpleNamespace(curie=curie, anchored_status=status)


def _all_complete_form_data(fm: FamilyMap) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for curies in fm.families.values():
        for c in curies:
            out[c] = _entry(c, "complete")
    for c in fm.singletons:
        out[c] = _entry(c, "complete")
    return out


def _all_degraded_form_data(fm: FamilyMap) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for curies in fm.families.values():
        for c in curies:
            out[c] = _entry(c, "degraded_placeholder")
    for c in fm.singletons:
        out[c] = _entry(c, "degraded_placeholder")
    return out


@pytest.fixture(autouse=True)
def _patch_loader(monkeypatch):
    """Pin ``load_family_map`` to the synthetic map for every test."""
    fm = _make_family_map()

    def fake_load(family: str):
        if family == "rdf_shacl":
            return fm
        return None

    # Patch on the validator's import target (lazy import inside .validate).
    monkeypatch.setattr(fm_mod, "load_family_map", fake_load)
    yield


# ----------------------------------------------------------------------
# DecisionCapture stub for capture wiring tests.
# ----------------------------------------------------------------------


class _StubCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_passes_when_all_curies_complete():
    """Every family entirely complete -> gate passes."""
    fm = _make_family_map()
    form_data = _all_complete_form_data(fm)
    validator = FamilyCompletenessValidator()
    result = validator.validate({
        "family": "rdf_shacl",
        "form_data": form_data,
    })
    assert result.passed, (
        f"expected pass; issues={[i.code for i in result.issues]}"
    )
    assert result.critical_count == 0


def test_passes_when_all_curies_degraded():
    """All entries degraded -> no family is partial -> gate passes."""
    fm = _make_family_map()
    form_data = _all_degraded_form_data(fm)
    validator = FamilyCompletenessValidator()
    result = validator.validate({
        "family": "rdf_shacl",
        "form_data": form_data,
    })
    assert result.passed, (
        f"expected pass; issues={[i.code for i in result.issues]}"
    )


def test_fails_on_partial_cardinality_family():
    """sh:minCount complete + sh:maxCount degraded -> FAMILY_PARTIALLY_COMPLETE."""
    fm = _make_family_map()
    form_data = _all_complete_form_data(fm)
    form_data["sh:maxCount"] = _entry("sh:maxCount", "degraded_placeholder")
    validator = FamilyCompletenessValidator()
    result = validator.validate({
        "family": "rdf_shacl",
        "form_data": form_data,
    })
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "FAMILY_PARTIALLY_COMPLETE" in codes
    # Make sure the family name lands in the message so operators can act.
    msgs = [i.message for i in result.issues if i.code == "FAMILY_PARTIALLY_COMPLETE"]
    assert any("cardinality" in m for m in msgs)


def test_fails_on_partial_validation_results_family():
    """One CURIE complete in a 3-CURIE family -> FAMILY_PARTIALLY_COMPLETE."""
    fm = _make_family_map()
    form_data = _all_degraded_form_data(fm)
    form_data["sh:result"] = _entry("sh:result", "complete")
    validator = FamilyCompletenessValidator()
    result = validator.validate({
        "family": "rdf_shacl",
        "form_data": form_data,
    })
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "FAMILY_PARTIALLY_COMPLETE" in codes
    msgs = [i.message for i in result.issues if i.code == "FAMILY_PARTIALLY_COMPLETE"]
    assert any("validation_results" in m for m in msgs)


def test_singletons_evaluated_independently():
    """Mixed-status singletons do NOT trigger family rules."""
    fm = _make_family_map()
    form_data = _all_complete_form_data(fm)
    # Flip one singleton degraded; no family rule should fire.
    form_data["sh:datatype"] = _entry("sh:datatype", "degraded_placeholder")
    validator = FamilyCompletenessValidator()
    result = validator.validate({
        "family": "rdf_shacl",
        "form_data": form_data,
    })
    # Gate passes because no family is partially complete.
    assert result.passed, (
        f"singletons should not trigger family rules; issues="
        f"{[i.code for i in result.issues]}"
    )


def test_no_family_map_passes_cleanly(monkeypatch):
    """No family_map for this family -> validator no-ops + passes."""
    monkeypatch.setattr(fm_mod, "load_family_map", lambda family: None)
    validator = FamilyCompletenessValidator()
    result = validator.validate({
        "family": "unknown_family",
        "form_data": {},
    })
    assert result.passed
    assert result.critical_count == 0
    codes = [i.code for i in result.issues]
    assert "FAMILY_MAP_NOT_FOUND" in codes


def test_decision_capture_fires_with_metadata_rationale():
    """capture.log_decision fires with metadata-shaped rationale."""
    fm = _make_family_map()
    form_data = _all_complete_form_data(fm)
    capture = _StubCapture()
    validator = FamilyCompletenessValidator()
    result = validator.validate({
        "family": "rdf_shacl",
        "form_data": form_data,
        "capture": capture,
    })
    assert result.passed
    assert len(capture.events) == 1
    event = capture.events[0]
    assert event["decision_type"] == "family_completeness_decision"
    assert event["decision"] == "family_completeness::passed"
    rationale = event["rationale"]
    # Rationale must be metadata-shaped: contains family slug + counts.
    assert "family=rdf_shacl" in rationale
    assert "passed=True" in rationale
    assert "families_total=3" in rationale
    assert "families_complete=3" in rationale
    # Rationale must NOT contain CURIE-content keys (definitions /
    # usage_examples / pyshacl payload). The form_data entries don't
    # carry those fields in this stub, but we still assert the
    # rationale doesn't reach into them by guarding common content
    # keywords from the real entries.
    assert "definitions" not in rationale
    assert "usage_examples" not in rationale


def test_fails_when_curie_missing_from_form_data():
    """A CURIE declared in the family map but absent from form_data fails."""
    fm = _make_family_map()
    form_data = _all_complete_form_data(fm)
    # Drop one entry — the family map still references it.
    form_data.pop("sh:result")
    validator = FamilyCompletenessValidator()
    result = validator.validate({
        "family": "rdf_shacl",
        "form_data": form_data,
    })
    assert not result.passed
    codes = [i.code for i in result.issues]
    assert "FAMILY_CURIE_MISSING_FROM_FORM_DATA" in codes
