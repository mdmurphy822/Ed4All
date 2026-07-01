"""W7.5 (M-TO) — tests for TerminalObjectiveSourceGroundingValidator.

A stub embedder returns deterministic 2D angle vectors so the suite pins
grounded / ungrounded outcomes WITHOUT the sentence-transformers extras. Covers:
the default-OFF no-op pass, the grounded pass, the ungrounded
``TO_SOURCE_UNGROUNDED`` warning (still passed=True day-1), the no-supporting-
chunk case, the cluster_signals authoritative-assignment path, the missing-
chunkset graceful skip, EMBEDDING_DEPS_MISSING degrade, and decision-capture.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

import lib.validators.terminal_objective_source_grounding as mod
from lib.validators.terminal_objective_source_grounding import (
    TerminalObjectiveSourceGroundingValidator,
)


class _StubEmbedder:
    """Deterministic unit vectors on a circle keyed by longest text prefix."""

    def __init__(self, angle_map: Dict[str, float]) -> None:
        self.angle_map = angle_map

    def encode(self, text: str, normalize: bool = True) -> List[float]:
        match_len = -1
        angle = 0.0
        for key, ang in self.angle_map.items():
            if text.startswith(key) and len(key) > match_len:
                match_len = len(key)
                angle = ang
        return [math.cos(angle), math.sin(angle)]


class _RecordingCapture:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(kwargs)


def _codes(result) -> List[str]:
    return [i.code for i in result.issues]


@pytest.fixture(autouse=True)
def _flag_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_TO_SOURCE_GROUNDING", "1")


def _objectives(to_chunk_ids=("c1",)) -> Dict[str, Any]:
    return {
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "photosynthesis converts light to chemical energy",
                "source_refs": [{"chunk_ids": list(to_chunk_ids)}],
            }
        ],
        "chapter_objectives": [
            {
                "id": "CO-01",
                "statement": "explain photosynthesis",
                "terminal_id": "TO-01",
                "source_chunk_ids": list(to_chunk_ids),
            }
        ],
    }


# --------------------------------------------------------------------- #

def test_disabled_flag_is_noop(monkeypatch):
    monkeypatch.delenv("ED4ALL_TO_SOURCE_GROUNDING", raising=False)
    result = TerminalObjectiveSourceGroundingValidator().validate(
        {"synthesized_objectives": _objectives()}
    )
    assert result.passed is True
    assert _codes(result) == ["TO_SOURCE_GROUNDING_DISABLED"]


def test_grounded_terminal_passes():
    embed = _StubEmbedder({"photosynthesis": 0.0})
    inputs = {
        "synthesized_objectives": _objectives(),
        "chunks_by_id": {"c1": "photosynthesis is the process by which plants"},
    }
    result = TerminalObjectiveSourceGroundingValidator(
        embedder=embed
    ).validate(inputs)
    assert result.passed is True
    assert "TO_SOURCE_UNGROUNDED" not in _codes(result)
    assert result.score == 1.0


def test_hallucinated_terminal_flagged():
    # TO statement (photosynthesis, angle 0) vs its cited chunk (insurance,
    # angle π/2) → cosine 0 < floor → ungrounded (the TO-15 class).
    embed = _StubEmbedder({"photosynthesis": 0.0, "insurance": math.pi / 2})
    inputs = {
        "synthesized_objectives": _objectives(),
        "chunks_by_id": {"c1": "insurance premiums are calculated using risk"},
    }
    result = TerminalObjectiveSourceGroundingValidator(
        embedder=embed
    ).validate(inputs)
    assert result.passed is True  # warning-day-1
    assert "TO_SOURCE_UNGROUNDED" in _codes(result)


def test_no_supporting_chunk_flagged():
    # TO cites c9 which is absent from the chunkset → no supporting chunk.
    embed = _StubEmbedder({"photosynthesis": 0.0})
    inputs = {
        "synthesized_objectives": _objectives(to_chunk_ids=("c9",)),
        "chunks_by_id": {"c1": "photosynthesis is the process"},
    }
    result = TerminalObjectiveSourceGroundingValidator(
        embedder=embed
    ).validate(inputs)
    assert "TO_SOURCE_UNGROUNDED" in _codes(result)


def test_cluster_signals_assignment_is_authoritative():
    # source_refs point at c1 (misaligned insurance text); cluster_signals
    # reassigns TO-01 to c2 (aligned photosynthesis text) → grounded via the
    # authoritative planning-side map.
    embed = _StubEmbedder({"photosynthesis": 0.0, "insurance": math.pi / 2})
    inputs = {
        "synthesized_objectives": _objectives(to_chunk_ids=("c1",)),
        "chunks_by_id": {
            "c1": "insurance premiums are calculated",
            "c2": "photosynthesis converts light energy",
        },
        "cluster_signals": {
            "to_source_chunk_assignments": {"TO-01": ["c2"]}
        },
    }
    result = TerminalObjectiveSourceGroundingValidator(
        embedder=embed
    ).validate(inputs)
    assert "TO_SOURCE_UNGROUNDED" not in _codes(result)


def test_missing_chunkset_graceful_skip():
    embed = _StubEmbedder({"photosynthesis": 0.0})
    result = TerminalObjectiveSourceGroundingValidator(embedder=embed).validate(
        {"synthesized_objectives": _objectives()}
    )
    assert result.passed is True
    assert "TO_SOURCE_NO_CHUNKSET" in _codes(result)


def test_embedding_deps_missing_degrade(monkeypatch):
    monkeypatch.setattr(mod, "try_load_embedder", lambda: None)
    inputs = {
        "synthesized_objectives": _objectives(),
        "chunks_by_id": {"c1": "photosynthesis"},
    }
    result = TerminalObjectiveSourceGroundingValidator().validate(inputs)
    assert result.passed is True
    assert "EMBEDDING_DEPS_MISSING" in _codes(result)


def test_decision_capture_fires():
    embed = _StubEmbedder({"photosynthesis": 0.0})
    capture = _RecordingCapture()
    inputs = {
        "synthesized_objectives": _objectives(),
        "chunks_by_id": {"c1": "photosynthesis is the process"},
        "decision_capture": capture,
    }
    TerminalObjectiveSourceGroundingValidator(embedder=embed).validate(inputs)
    assert any(
        e.get("decision_type") == "validation_result" for e in capture.events
    )
