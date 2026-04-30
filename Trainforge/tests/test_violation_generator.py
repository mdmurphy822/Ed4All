"""Tests for the SHACL violation-detection generator (Audit 2026-04-30,
Wave 125a expansion).

Covers the contract spelled out in the audit fix:

* Built-in shape catalog: programmatically expanded to >= 800
  pyshacl-validated pairs (Wave 125a) covering all 6 surface forms.
  Six pinned canonical fixture names preserved for back-compat:
  ``datatype_int_age``, ``class_constraint_owns``, ``nodeshape_min_count``,
  ``propertyshape_max_count``, ``subclass_of_class_constraint``,
  ``sameas_iri_kind``.
* Pyshacl oracle agrees with every generator-claimed validity (zero
  disagreements). Wrong-labeled pairs are dropped, never emitted.
* Each emitted pair validates against `instruction_pair.schema.json`
  (prompt 40-400 chars, completion 50-600 chars).
* `chunk_id` anchoring: when the property manifest has a surface form,
  the pair anchors to a chunk teaching that form.
* Decision capture fires once per fixture.
* `max_pairs` cap (Wave 125a) trims emit with family-balanced
  round-robin so every surface form keeps representation.
* Pyshacl missing -> `pytest.skip` rather than hard fail.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Skip the entire module if pyshacl isn't installed.
pyshacl = pytest.importorskip("pyshacl")
rdflib = pytest.importorskip("rdflib")

from Trainforge.generators.violation_generator import (  # noqa: E402
    ShapeFixture,
    ViolationStats,
    built_in_shape_catalog,
    generate_violation_pairs,
)


PAIR_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "instruction_pair.schema.json"
)


class _FakeCapture:
    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []
        self._counter = 0

    def log_decision(self, **kwargs: Any) -> None:
        self._counter += 1
        record = dict(kwargs)
        record["event_id"] = f"EVT_{self._counter:06d}"
        self.decisions.append(record)


def _validate_pair(pair: Dict[str, Any]) -> None:
    import jsonschema

    schema = json.loads(PAIR_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(pair, schema)


def test_built_in_catalog_preserves_pinned_fixture_names() -> None:
    """Wave 125a: catalog programmatically expanded but the 6 pinned
    canonical fixture names must remain so existing wiring + downstream
    tests keep working."""
    catalog = built_in_shape_catalog()
    names = {f.name for f in catalog}
    pinned = {
        "datatype_int_age",
        "class_constraint_owns",
        "nodeshape_min_count",
        "propertyshape_max_count",
        "subclass_of_class_constraint",
        "sameas_iri_kind",
    }
    missing = pinned - names
    assert not missing, f"missing pinned fixture names: {missing}"
    # Each fixture has at least one valid + one invalid graph.
    for f in catalog:
        assert isinstance(f, ShapeFixture)
        assert len(f.graphs) >= 2
        assert any(valid for _, valid in f.graphs)
        assert any(not valid for _, valid in f.graphs)


def test_catalog_has_at_least_800_pairs() -> None:
    """Wave 125a target: pyshacl-validated catalog >= 800 pairs."""
    capture = _FakeCapture()
    pairs, stats = generate_violation_pairs(capture=capture)
    assert len(pairs) >= 800, (
        f"expected >= 800 pairs, got {len(pairs)} (catalog underfilled)"
    )
    assert stats.pairs_emitted == len(pairs)
    # Sanity: at least the 6 canonical fixtures contributed.
    assert stats.fixtures_used >= 6


def test_catalog_no_oracle_disagreements() -> None:
    """Wave 125a contract: zero pyshacl/fixture disagreements at the
    full catalog scale."""
    capture = _FakeCapture()
    _, stats = generate_violation_pairs(capture=capture)
    assert stats.oracle_disagreements == 0, (
        f"pyshacl disagreed with {stats.oracle_disagreements} fixture "
        f"graph(s); fix the catalog before emitting."
    )


def test_catalog_covers_all_six_surface_forms_with_volume() -> None:
    """Each of the 6 RDF/SHACL surface forms must hold >= 50 pairs."""
    from collections import Counter

    capture = _FakeCapture()
    pairs, _ = generate_violation_pairs(capture=capture)
    counts = Counter(p["shape_curie"] for p in pairs)
    expected_forms = {
        "sh:datatype",
        "sh:class",
        "sh:NodeShape",
        "sh:PropertyShape",
        "rdfs:subClassOf",
        "owl:sameAs",
    }
    for form in expected_forms:
        assert counts.get(form, 0) >= 50, (
            f"{form!r} has only {counts.get(form, 0)} pairs; need >= 50"
        )


def test_no_duplicate_pair_prompts() -> None:
    """Every prompt must be unique — duplicate prompts collapse the
    de-duplication step downstream and starve the corpus of diversity."""
    capture = _FakeCapture()
    pairs, _ = generate_violation_pairs(capture=capture)
    prompts = [p["prompt"] for p in pairs]
    assert len(prompts) == len(set(prompts)), (
        f"duplicate prompts: {len(prompts)} total, {len(set(prompts))} unique"
    )


def test_pair_lengths_within_schema_bounds() -> None:
    """instruction_pair.schema.json: prompt 40-400, completion 50-600.
    Critical — schema validation fails otherwise."""
    capture = _FakeCapture()
    pairs, _ = generate_violation_pairs(capture=capture)
    for pair in pairs:
        plen = len(pair["prompt"])
        clen = len(pair["completion"])
        assert 40 <= plen <= 400, (
            f"prompt length {plen} outside [40,400] for "
            f"{pair.get('shape_curie')}/{pair.get('expected_validity')}"
        )
        assert 50 <= clen <= 600, (
            f"completion length {clen} outside [50,600] for "
            f"{pair.get('shape_curie')}/{pair.get('expected_validity')}"
        )


def test_compound_fixtures_present() -> None:
    """Compound fixtures (2+ distinct SHACL constraint predicates in
    one shape body) are the highest-value teaching surface; the catalog
    must carry at least 50 of them."""
    catalog = built_in_shape_catalog()
    shacl_predicates = (
        "sh:datatype", "sh:class", "sh:minCount", "sh:maxCount",
        "sh:nodeKind", "sh:minLength", "sh:maxLength", "sh:pattern",
        "sh:hasValue", "sh:in", "sh:minInclusive", "sh:maxInclusive",
    )
    compound_count = 0
    for f in catalog:
        ttl = f.shape_ttl
        distinct = {p for p in shacl_predicates if p in ttl}
        if len(distinct) >= 2:
            compound_count += 1
    assert compound_count >= 50, (
        f"only {compound_count} compound fixtures (need >= 50); "
        f"compound shapes are the richest teaching surface."
    )


def test_max_pairs_caps_emit_with_balanced_families() -> None:
    """Wave 125a: max_pairs=200 truncates to <= 200 with all 6 surface
    forms still represented."""
    from collections import Counter

    capture = _FakeCapture()
    capped, stats = generate_violation_pairs(
        capture=capture, max_pairs=200,
    )
    assert len(capped) <= 200, (
        f"max_pairs=200 returned {len(capped)} pairs"
    )
    # Need a meaningful cap effect.
    assert len(capped) >= 100
    counts = Counter(p["shape_curie"] for p in capped)
    expected_forms = {
        "sh:datatype",
        "sh:class",
        "sh:NodeShape",
        "sh:PropertyShape",
        "rdfs:subClassOf",
        "owl:sameAs",
    }
    for form in expected_forms:
        assert counts.get(form, 0) > 0, (
            f"capped output dropped surface form {form!r} entirely; "
            f"family-balanced round-robin must preserve every form."
        )
    # Stats must mirror the trimmed list, not the pre-cap catalog.
    assert stats.pairs_emitted == len(capped)
    assert stats.valid_pairs + stats.invalid_pairs == len(capped)
    assert stats.oracle_disagreements == 0


def test_max_pairs_zero_emits_nothing() -> None:
    """Edge case: ``max_pairs=0`` is a meaningful operator request to
    skip all violation pairs without disabling the generator wiring."""
    capture = _FakeCapture()
    pairs, stats = generate_violation_pairs(capture=capture, max_pairs=0)
    assert pairs == []
    assert stats.pairs_emitted == 0


def test_max_pairs_above_catalog_returns_full_catalog() -> None:
    """Cap above the catalog size is a no-op."""
    capture = _FakeCapture()
    full, _ = generate_violation_pairs(capture=capture)
    capture2 = _FakeCapture()
    capped, _ = generate_violation_pairs(capture=capture2, max_pairs=10_000)
    assert len(capped) == len(full)


def test_pyshacl_oracle_agrees_with_every_fixture() -> None:
    """Contract: zero oracle disagreements. Every generator-claimed
    validity matches the pyshacl verdict; wrong-labeled pairs are
    dropped, never emitted."""
    capture = _FakeCapture()
    _, stats = generate_violation_pairs(capture=capture)
    assert stats.oracle_disagreements == 0, (
        f"pyshacl disagreed with {stats.oracle_disagreements} fixture "
        f"graph(s); fix the catalog before emitting."
    )


def test_each_pair_is_schema_valid() -> None:
    capture = _FakeCapture()
    pairs, _ = generate_violation_pairs(capture=capture)
    assert pairs, "expected at least one pair"
    for pair in pairs:
        _validate_pair(pair)


def test_chunk_id_anchors_to_property_manifest_chunk_when_available() -> None:
    """When `chunks_by_surface_form` provides a chunk that teaches the
    fixture's surface form, the pair anchors to that chunk."""
    capture = _FakeCapture()
    chunks_by_form = {
        "sh:datatype": ["rdf_shacl_551_chunk_00100"],
        "sh:NodeShape": ["rdf_shacl_551_chunk_00200"],
    }
    pairs, _ = generate_violation_pairs(
        capture=capture,
        chunks_by_surface_form=chunks_by_form,
    )
    for pair in pairs:
        if pair["shape_curie"] == "sh:datatype":
            assert pair["chunk_id"] == "rdf_shacl_551_chunk_00100"
        elif pair["shape_curie"] == "sh:NodeShape":
            assert pair["chunk_id"] == "rdf_shacl_551_chunk_00200"


def test_chunk_id_falls_back_to_synthetic_when_no_manifest_match() -> None:
    """Without a `chunks_by_surface_form` mapping, the pair carries a
    synthetic `violation_fixture:<name>` id and the CURIE in
    `concept_tags` keeps the property linkage explicit."""
    capture = _FakeCapture()
    pairs, _ = generate_violation_pairs(capture=capture)
    for pair in pairs:
        assert pair["chunk_id"].startswith("violation_fixture:")
        assert pair["concept_tags"] == [pair["shape_curie"]]


def test_decision_capture_fires_once_per_fixture() -> None:
    capture = _FakeCapture()
    _, stats = generate_violation_pairs(capture=capture)
    types = [d["decision_type"] for d in capture.decisions]
    # One violation_generation event per fixture used (Wave 125a:
    # catalog programmatically expanded — count must match
    # stats.fixtures_used, not a fixed 6).
    assert types.count("violation_generation") == stats.fixtures_used
    assert stats.fixtures_used >= 6
    for event in capture.decisions:
        rationale = event["rationale"]
        assert len(rationale) >= 20
        # Per CLAUDE.md, rationale must interpolate dynamic signals.
        assert "kind=" in rationale
        assert "pyshacl_version=" in rationale
        assert "seed=" in rationale
        # alternatives_considered shape: dicts with option / reason_rejected.
        alts = event.get("alternatives_considered", []) or []
        for alt in alts:
            assert isinstance(alt, dict)
            assert "option" in alt
            assert "reason_rejected" in alt


def test_capture_required() -> None:
    with pytest.raises(ValueError, match="capture"):
        generate_violation_pairs(capture=None)


def test_pair_carries_violation_marker_fields() -> None:
    capture = _FakeCapture()
    pairs, _ = generate_violation_pairs(capture=capture)
    for pair in pairs:
        assert pair["template_id"].startswith("violation_detection.")
        assert pair["template_id"].endswith(("valid", "invalid"))
        assert pair["expected_validity"] in ("valid", "invalid")
        assert pair["bloom_level"] in ("evaluate", "apply")
        if pair["expected_validity"] == "invalid":
            assert pair["bloom_level"] == "evaluate"
            assert pair["completion"].lower().startswith("no.")
        else:
            assert pair["bloom_level"] == "apply"
            assert pair["completion"].lower().startswith("yes.")


def test_invalid_completion_carries_pyshacl_reason() -> None:
    """Invalid pairs must include the oracle's actual violation
    message — the corpus teaches the model to give a real reason, not
    a generic "the graph is wrong"."""
    capture = _FakeCapture()
    pairs, _ = generate_violation_pairs(capture=capture)
    invalid_pairs = [
        p for p in pairs if p["expected_validity"] == "invalid"
    ]
    for pair in invalid_pairs:
        completion = pair["completion"]
        assert "Reason:" in completion
        # Real violation messages mention either "Constraint Violation"
        # or a SHACL component name.
        assert (
            "Constraint Violation" in completion
            or "ConstraintComponent" in completion
        ), f"completion looks faked: {completion!r}"


def test_pyshacl_disagreement_is_detected_and_skipped() -> None:
    """Manually craft a fixture whose pyshacl verdict disagrees with
    the labeled validity; the generator must skip rather than emit a
    wrong-labeled pair."""
    bad_fixture = ShapeFixture(
        name="intentionally_wrong",
        kind="datatype",
        curie="sh:datatype",
        surface_form="sh:datatype",
        shape_ttl=(
            "ex:S a sh:NodeShape ; sh:targetClass ex:P ;\n"
            "  sh:property [ sh:path ex:age ; sh:datatype xsd:integer ] .\n"
        ),
        graphs=[
            # Label says valid, but the data has a string where an
            # integer is required -> pyshacl says invalid -> skip.
            (
                "ex:a a ex:P ; ex:age \"thirty\" .\n",
                True,  # WRONG label intentionally
            ),
        ],
    )
    capture = _FakeCapture()
    pairs, stats = generate_violation_pairs(
        capture=capture, fixtures=[bad_fixture],
    )
    assert pairs == []
    assert stats.oracle_disagreements == 1
    assert stats.pairs_emitted == 0
