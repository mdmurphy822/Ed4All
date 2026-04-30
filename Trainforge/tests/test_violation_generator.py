"""Tests for the SHACL violation-detection generator (Audit 2026-04-30).

Covers the contract spelled out in the audit fix:

* Built-in shape catalog: 6 shapes × 2 graphs each -> 12 pairs minimum.
* Pyshacl oracle agrees with every generator-claimed validity (zero
  disagreements). Wrong-labeled pairs are dropped, never emitted.
* Each emitted pair validates against `instruction_pair.schema.json`.
* `chunk_id` anchoring: when the property manifest has a surface form,
  the pair anchors to a chunk teaching that form.
* Decision capture fires once per fixture.
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


def test_built_in_catalog_has_six_shapes() -> None:
    catalog = built_in_shape_catalog()
    assert len(catalog) == 6
    # Each fixture has at least one valid + one invalid graph.
    for f in catalog:
        assert isinstance(f, ShapeFixture)
        assert len(f.graphs) >= 2
        assert any(valid for _, valid in f.graphs)
        assert any(not valid for _, valid in f.graphs)


def test_emits_at_least_twelve_pairs() -> None:
    """6 fixtures × 2 graphs = 12 pairs minimum on a clean run."""
    capture = _FakeCapture()
    pairs, stats = generate_violation_pairs(capture=capture)
    assert len(pairs) >= 12
    assert stats.pairs_emitted == len(pairs)
    assert stats.fixtures_used == 6


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
    generate_violation_pairs(capture=capture)
    types = [d["decision_type"] for d in capture.decisions]
    # 6 fixtures, 6 events.
    assert types.count("violation_generation") == 6
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
