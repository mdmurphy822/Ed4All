"""Wave W-D1 T1.5 — ``schema_translation_generator`` capture-wiring
regression test.

Pins one ``schema_translation_generation`` event per emitted pair
(``Trainforge/generators/deterministic/schema_translation_generator.py:2908-2924``).
Capture is **required** — ``capture=None`` raises ``ValueError`` per
``schema_translation_generator.py:2836-2841``.

Three assertions per file (per plan §2.5):

1. Required-capture: ``capture=None`` raises ``ValueError`` with the
   advertised ``schema_translation_generator requires`` prefix.
2. Wired capture: one ``schema_translation_generation`` event fires
   per emitted pair (``len(events) == len(pairs)``).
3. Rationale interpolates dynamic signals (seed, manifest family) and
   ``decision`` field carries the CURIE under translation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.deterministic.schema_translation_generator import (  # noqa: E402
    generate_schema_translation_pairs,
)
from lib.ontology.property_manifest import (  # noqa: E402
    PropertyEntry,
    PropertyManifest,
)


class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        kwargs = {**kwargs, "event_id": f"evt_{len(self.decisions):04d}"}
        self.decisions.append(kwargs)
        self.events.append(kwargs)


def _rdf_shacl_manifest() -> PropertyManifest:
    """Minimal manifest pointing at one of the 6 hand-curated CURIEs in
    ``_RDF_SHACL_FALLBACK_FORM_DATA``. ``sh:datatype`` is a known
    ``"complete"`` entry per the FORM_DATA contract — confirmed at
    ``schema_translation_generator.py:231-248``."""
    return PropertyManifest(
        family="rdf_shacl",
        properties=[
            PropertyEntry(
                id="datatype",
                uri="http://www.w3.org/ns/shacl#datatype",
                curie="sh:datatype",
                label="sh:datatype",
                surface_forms=["sh:datatype"],
                min_pairs=1,
            ),
        ],
    )


def test_required_capture_raises_when_none():
    with pytest.raises(
        ValueError,
        match="schema_translation_generator requires",
    ):
        generate_schema_translation_pairs(
            _rdf_shacl_manifest(),
            capture=None,
        )


def test_capture_fires_per_pair():
    capture = _RecordingCapture()
    # Pre-emit decision so ``_last_event_id`` resolves on the first
    # pair (Wave 112 invariant — schema_translation_generator.py
    # tracks ``decisions`` for decision_capture_id resolution).
    capture.log_decision(
        decision_type="stage_start",
        decision="schema translation stage",
        rationale="x" * 30,
    )
    pairs, _stats = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=10,
        seed=17,
    )
    st_events = [
        e for e in capture.events
        if e["decision_type"] == "schema_translation_generation"
    ]
    # Per-pair emit contract.
    assert len(st_events) == len(pairs)
    assert len(pairs) >= 1


def test_capture_rationale_carries_dynamic_signals():
    capture = _RecordingCapture()
    capture.log_decision(
        decision_type="stage_start",
        decision="x",
        rationale="x" * 30,
    )
    _pairs, _ = generate_schema_translation_pairs(
        _rdf_shacl_manifest(),
        capture=capture,
        max_pairs=5,
        seed=17,
    )
    st_events = [
        e for e in capture.events
        if e["decision_type"] == "schema_translation_generation"
    ]
    assert st_events
    ev = st_events[0]
    rationale = ev["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    # Dynamic signals per ``schema_translation_generator.py:2914-2924``.
    assert "sh:datatype" in ev["decision"]
    assert "seed=17" in rationale
    assert "manifest_family='rdf_shacl'" in rationale
