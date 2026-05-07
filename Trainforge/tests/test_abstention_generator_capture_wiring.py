"""Wave W-D1 T1.5 — ``abstention_generator`` capture-wiring regression test.

Pins one ``abstention_generation`` event per emitted pair
(``Trainforge/generators/abstention_generator.py:408-444``). Capture is
**required** — ``capture=None`` raises ``ValueError`` per the
``abstention_generator requires a DecisionCapture`` contract at
``abstention_generator.py:329-334``.

Three assertions per file (per plan §2.5):

1. Required-capture: ``capture=None`` raises ``ValueError`` with the
   advertised message.
2. Wired capture: one ``abstention_generation`` event fires per emitted
   pair; rationale interpolates dynamic signals (addressed-concept
   count, seed) and is at least 20 chars.
3. ``decision`` field carries chunk + silent_concept identifiers so
   audit replay can distinguish chunks.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.generators.abstention_generator import (  # noqa: E402
    generate_abstention_pairs,
)


# T1.5 stub — see plan §2 "Shared test-stub helpers". The
# abstention_generator resolves ``decision_capture_id`` via
# ``_last_event_id`` so we stamp event_id on every emit.
class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.decisions: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        kwargs = {**kwargs, "event_id": f"evt_{len(self.decisions):04d}"}
        self.decisions.append(kwargs)
        self.events.append(kwargs)


def _stub_pedagogy_graph() -> Dict[str, Any]:
    """Minimal graph: 3 concepts (so silent set is non-empty), 1 chunk
    addressing the first concept via an ``assesses`` edge.

    Node detection uses ``class`` (lowercased) per
    ``abstention_generator.py:107``; chunk addresses iterate edges where
    ``relation_type`` is one of ``_ADDRESS_RELATIONS`` per
    ``abstention_generator.py:130-134``.
    """
    return {
        "nodes": [
            {"id": "concept:topic_x", "class": "Concept",
             "label": "Topic X"},
            {"id": "concept:topic_y", "class": "Concept",
             "label": "Topic Y"},
            {"id": "concept:topic_z", "class": "Concept",
             "label": "Topic Z"},
        ],
        "edges": [
            {
                "source": "chunk:c1",
                "target": "concept:topic_x",
                "relation_type": "assesses",
            },
        ],
    }


def test_required_capture_raises_when_none():
    with pytest.raises(
        ValueError,
        match="abstention_generator requires a DecisionCapture",
    ):
        generate_abstention_pairs(
            _stub_pedagogy_graph(),
            capture=None,
        )


def test_capture_fires_per_pair():
    capture = _RecordingCapture()
    # Pre-emit decision so ``_last_event_id`` resolves to a non-empty
    # event id (Wave 112 invariant — see
    # ``abstention_generator.py:270-280``).
    capture.log_decision(
        decision_type="stage_start",
        decision="abstention generator stage",
        rationale=(
            "Pre-emit decision so decision_capture_id resolves "
            "against a non-empty event id."
        ),
    )
    pairs, _stats = generate_abstention_pairs(
        _stub_pedagogy_graph(),
        capture=capture,
        max_pairs=5,
        seed=17,
    )
    abstention_events = [
        e for e in capture.events
        if e["decision_type"] == "abstention_generation"
    ]
    assert len(abstention_events) == len(pairs)
    assert len(pairs) >= 1
    ev = abstention_events[0]
    rationale = ev["rationale"]
    assert isinstance(rationale, str)
    assert len(rationale) >= 20
    # Dynamic signals: addressed-concept count + seed per
    # ``abstention_generator.py:415-421``.
    assert "addressed-concept" in rationale
    assert "seed=17" in rationale


def test_decision_carries_chunk_and_silent_concept_ids():
    capture = _RecordingCapture()
    capture.log_decision(
        decision_type="stage_start",
        decision="x",
        rationale="x" * 30,
    )
    pairs, _ = generate_abstention_pairs(
        _stub_pedagogy_graph(),
        capture=capture,
        max_pairs=5,
        seed=17,
    )
    abstention_events = [
        e for e in capture.events
        if e["decision_type"] == "abstention_generation"
    ]
    assert pairs
    assert abstention_events
    decision = abstention_events[0]["decision"]
    assert "chunk=" in decision
    assert "silent_concept=" in decision
