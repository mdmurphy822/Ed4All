"""Wave W-D1 T1.5 — ``violation_generator`` capture-wiring regression test.

Pins one ``violation_generation`` event per FIXTURE (not per pair) at
``Trainforge/generators/violation_generator.py:1538-1577``. Capture is
**required** — ``capture=None`` raises ``ValueError`` per
``violation_generator.py:1476-1481``.

Skips the test when pyshacl isn't available — a clean dev install
without the ``[training]`` extras shouldn't fail this test.

Three assertions per file (per plan §2.5):

1. Required-capture: ``capture=None`` raises ``ValueError`` with the
   advertised ``violation_generator requires`` prefix.
2. Wired capture: one ``violation_generation`` event fires per fixture;
   rationale interpolates dynamic signals (pyshacl version, seed) and
   is at least 20 chars; ``decision`` field carries the fixture name.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Skip when pyshacl is missing — the violation_generator's pyshacl
# oracle is the gating dependency and a clean dev install without the
# [training] extras has no pyshacl wheel.
pytest.importorskip("pyshacl")

from Trainforge.generators.violation_generator import (  # noqa: E402
    ShapeFixture,
    generate_violation_pairs,
)


class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        kwargs = {**kwargs, "event_id": f"evt_{len(self.decisions):04d}"}
        self.decisions.append(kwargs)
        self.events.append(kwargs)


def _trivial_fixture() -> ShapeFixture:
    """Minimal sh:datatype-style shape with one valid + one invalid
    graph. Pyshacl is the oracle — it confirms ``ex:age "42"^^xsd:integer``
    is a valid sh:datatype assertion and ``ex:age "not_a_number"`` is
    not."""
    shape_ttl = """
    @prefix sh: <http://www.w3.org/ns/shacl#> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix ex: <http://example.org/> .

    ex:PersonShape a sh:NodeShape ;
        sh:targetClass ex:Person ;
        sh:property [
            sh:path ex:age ;
            sh:datatype xsd:integer ;
        ] .
    """
    valid_graph = """
    @prefix ex: <http://example.org/> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    ex:alice a ex:Person ; ex:age "42"^^xsd:integer .
    """
    invalid_graph = """
    @prefix ex: <http://example.org/> .
    ex:bob a ex:Person ; ex:age "not_a_number" .
    """
    return ShapeFixture(
        name="datatype_int_age_test",
        kind="sh:NodeShape",
        curie="sh:datatype",
        surface_form="sh:datatype",
        shape_ttl=shape_ttl,
        graphs=[(valid_graph, True), (invalid_graph, False)],
    )


def test_required_capture_raises_when_none():
    with pytest.raises(
        ValueError,
        match="violation_generator requires",
    ):
        generate_violation_pairs(
            capture=None,
            fixtures=[_trivial_fixture()],
        )


def test_capture_fires_per_fixture():
    capture = _RecordingCapture()
    # Pre-emit decision so ``_last_event_id`` resolves (Wave 112
    # invariant). violation_generator's _last_event_id helper at
    # ``violation_generator.py:1343`` returns "" on empty decisions
    # rather than raising, but resolved-id propagation through
    # ``_build_pair`` is still the audit anchor we want exercised.
    capture.log_decision(
        decision_type="stage_start",
        decision="violation gen stage",
        rationale="x" * 30,
    )
    pairs, _stats = generate_violation_pairs(
        capture=capture,
        fixtures=[_trivial_fixture()],
        seed=17,
    )
    fixture_events = [
        e for e in capture.events
        if e["decision_type"] == "violation_generation"
    ]
    assert len(fixture_events) == 1
    ev = fixture_events[0]
    rationale = ev["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    # Dynamic signals per ``violation_generator.py:1547-1556``.
    assert "datatype_int_age_test" in ev["decision"]
    assert "pyshacl_version=" in rationale
    assert "seed=17" in rationale
    # Sanity — pyshacl produced both pairs (one valid, one invalid).
    assert len(pairs) == 2
