"""Wave W-D1 T1.5 — ``kg_metadata_generator`` capture-wiring regression test.

Pins one ``kg_metadata_generation`` event per RELATION BATCH (not per
pair) at ``Trainforge/generators/deterministic/kg_metadata_generator.py:370-408``.
Capture is **required** — ``capture=None`` raises ``ValueError`` per
``kg_metadata_generator.py:319-324``.

Three assertions per file (per plan §2.5):

1. Required-capture: ``capture=None`` raises ``ValueError`` with the
   advertised ``kg_metadata_generator requires`` prefix.
2. Wired capture: one ``kg_metadata_generation`` event fires per
   relation batch — a graph with two relation types yields two events.
3. Rationale interpolates dynamic signals (negatives count, seed) and
   is at least 20 chars.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.deterministic.kg_metadata_generator import (  # noqa: E402
    generate_kg_metadata_pairs,
)


class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        kwargs = {**kwargs, "event_id": f"evt_{len(self.decisions):04d}"}
        self.decisions.append(kwargs)
        self.events.append(kwargs)


def _stub_pedagogy_graph() -> Dict[str, Any]:
    """Minimal graph with 2 relation types — gives 2 batches per
    ``kg_metadata_generator.py:339-408``."""
    return {
        "nodes": [
            {"id": "concept:a", "class": "Concept", "label": "A"},
            {"id": "concept:b", "class": "Concept", "label": "B"},
            {"id": "concept:c", "class": "Concept", "label": "C"},
        ],
        "edges": [
            {"source": "concept:b", "target": "concept:a",
             "relation_type": "is_a"},
            {"source": "concept:c", "target": "concept:a",
             "relation_type": "prerequisite"},
        ],
    }


def test_required_capture_raises_when_none():
    with pytest.raises(
        ValueError,
        match="kg_metadata_generator requires",
    ):
        generate_kg_metadata_pairs(
            _stub_pedagogy_graph(),
            capture=None,
        )


def test_capture_fires_per_relation_batch():
    capture = _RecordingCapture()
    # Pre-emit decision so ``_last_event_id`` resolves on first batch
    # (Wave 112 invariant — kg_metadata_generator.py:265-280 raises
    # RuntimeError on empty decisions).
    capture.log_decision(
        decision_type="stage_start",
        decision="kg metadata stage start",
        rationale="x" * 30,
    )
    _pairs, _stats = generate_kg_metadata_pairs(
        _stub_pedagogy_graph(),
        capture=capture,
        max_pairs=20,
        negatives_per_positive=1,
        seed=17,
    )
    kg_events = [
        e for e in capture.events
        if e["decision_type"] == "kg_metadata_generation"
    ]
    # 2 relations -> 2 batches.
    assert len(kg_events) == 2


def test_capture_rationale_carries_dynamic_signals():
    capture = _RecordingCapture()
    capture.log_decision(
        decision_type="stage_start",
        decision="x",
        rationale="x" * 30,
    )
    _pairs, _ = generate_kg_metadata_pairs(
        _stub_pedagogy_graph(),
        capture=capture,
        max_pairs=20,
        seed=17,
    )
    kg_events = [
        e for e in capture.events
        if e["decision_type"] == "kg_metadata_generation"
    ]
    assert kg_events
    rationale = kg_events[0]["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    # Dynamic signals per ``kg_metadata_generator.py:379-388``.
    assert "Negatives" in rationale or "negatives" in rationale
    assert "seed=17" in rationale
