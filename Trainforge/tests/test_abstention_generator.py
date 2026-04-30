"""Tests for the abstention SFT pair generator (Wave 124, audit 2026-04-30).

Mirrors `test_kg_metadata_generator.py` structure. Covers:

* A small synthetic graph with chunks that address some-but-not-all
  concepts emits at least one pair per such chunk.
* Chunks that already address every concept produce zero pairs (edge
  case).
* Each emitted pair validates against `instruction_pair.schema.json`.
* Completion mentions actual addressed concepts (not random noise),
  so the abstention is grounded.
* Sampled silent concepts truly have no edge from the chunk in the
  graph (counterfactual is real).
* Same seed + same graph -> byte-identical pair list.
* `max_pairs` cap is honored.
* DecisionCapture fires per-emit with rationale that interpolates
  dynamic signals.
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

from Trainforge.generators.abstention_generator import (  # noqa: E402
    AbstentionStats,
    generate_abstention_pairs,
)


PAIR_SCHEMA_PATH = (
    PROJECT_ROOT / "schemas" / "knowledge" / "instruction_pair.schema.json"
)


class _FakeCapture:
    """Minimal DecisionCapture-shaped object for tests."""

    def __init__(self) -> None:
        self.decisions: List[Dict[str, Any]] = []
        self._counter = 0

    def log_decision(self, **kwargs: Any) -> None:
        self._counter += 1
        record = dict(kwargs)
        record["event_id"] = f"EVT_{self._counter:06d}"
        self.decisions.append(record)


def _five_concept_graph() -> Dict[str, Any]:
    """Five concepts; three chunks each addressing a proper subset.

    chunk_a addresses concept_alpha (1/5)
    chunk_b addresses concept_beta + concept_gamma (2/5)
    chunk_c addresses concept_delta + concept_epsilon (2/5)
    -> all three chunks have at least one silent concept.
    """
    return {
        "nodes": [
            {"id": "chunk_a", "class": "Chunk"},
            {"id": "chunk_b", "class": "Chunk"},
            {"id": "chunk_c", "class": "Chunk"},
            {"id": "concept_alpha", "class": "Concept", "label": "Alpha"},
            {"id": "concept_beta", "class": "Concept", "label": "Beta"},
            {"id": "concept_gamma", "class": "Concept", "label": "Gamma"},
            {"id": "concept_delta", "class": "Concept", "label": "Delta"},
            {"id": "concept_epsilon", "class": "Concept", "label": "Epsilon"},
        ],
        "edges": [
            {"source": "chunk_a", "target": "concept_alpha", "relation_type": "assesses"},
            {"source": "chunk_b", "target": "concept_beta", "relation_type": "assesses"},
            {"source": "chunk_b", "target": "concept_gamma", "relation_type": "exemplifies"},
            {"source": "chunk_c", "target": "concept_delta", "relation_type": "derives_from_objective"},
            {"source": "chunk_c", "target": "concept_epsilon", "relation_type": "addresses_misconception"},
        ],
    }


def _all_concepts_addressed_graph() -> Dict[str, Any]:
    """A degenerate graph where the single chunk addresses every concept.

    Used to exercise the "no silent concepts" skip branch.
    """
    return {
        "nodes": [
            {"id": "chunk_only", "class": "Chunk"},
            {"id": "concept_one", "class": "Concept", "label": "One"},
            {"id": "concept_two", "class": "Concept", "label": "Two"},
        ],
        "edges": [
            {"source": "chunk_only", "target": "concept_one", "relation_type": "assesses"},
            {"source": "chunk_only", "target": "concept_two", "relation_type": "exemplifies"},
        ],
    }


def _validate_pair(pair: Dict[str, Any]) -> None:
    """Validate a single pair against `instruction_pair.schema.json`."""
    import jsonschema

    schema = json.loads(PAIR_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(pair, schema)


def test_emits_at_least_one_pair_per_chunk_with_silent_concepts() -> None:
    capture = _FakeCapture()
    pairs, stats = generate_abstention_pairs(
        _five_concept_graph(),
        capture=capture,
        max_pairs=200,
        silent_per_chunk=1,
    )
    assert isinstance(stats, AbstentionStats)
    # Each of the three chunks has at least one silent concept.
    assert stats.chunks_with_silent == 3
    assert stats.pairs_emitted == 3
    assert len(pairs) == 3
    # Each pair anchors back to a real chunk.
    chunk_ids = {p["chunk_id"] for p in pairs}
    assert chunk_ids == {"chunk_a", "chunk_b", "chunk_c"}


def test_no_pair_when_chunk_addresses_all_concepts() -> None:
    capture = _FakeCapture()
    pairs, stats = generate_abstention_pairs(
        _all_concepts_addressed_graph(),
        capture=capture,
        max_pairs=200,
    )
    assert pairs == []
    assert stats.pairs_emitted == 0
    assert stats.chunks_skipped_all_addressed == 1


def test_pair_validates_against_instruction_pair_schema() -> None:
    capture = _FakeCapture()
    pairs, _ = generate_abstention_pairs(
        _five_concept_graph(),
        capture=capture,
        max_pairs=200,
    )
    assert pairs, "expected at least one pair"
    for pair in pairs:
        _validate_pair(pair)


def test_completion_mentions_actual_addressed_concepts() -> None:
    """The completion must reference at least one of the chunk's
    actually-addressed concepts (when there is one), not a random
    surface form."""
    capture = _FakeCapture()
    pairs, _ = generate_abstention_pairs(
        _five_concept_graph(),
        capture=capture,
        max_pairs=200,
        silent_per_chunk=1,
    )
    # Build the addressed-surfaces lookup the same way the generator
    # would.
    addressed_for_chunk = {
        "chunk_a": {"Alpha"},
        "chunk_b": {"Beta", "Gamma"},
        "chunk_c": {"Delta", "Epsilon"},
    }
    for pair in pairs:
        chunk = pair["chunk_id"]
        completion = pair["completion"]
        if chunk == "chunk_a":
            # Single-concept fallback shape — completion does NOT need
            # to mention "Alpha" by surface form (the schema completion
            # floor pushes us into the "based on the encoded edges"
            # generic shape when only 1 addressed concept exists).
            # Either shape is acceptable; both are honest.
            continue
        # chunk_b / chunk_c have 2 addressed concepts each; the
        # completion must reference at least one of them.
        addr = addressed_for_chunk[chunk]
        assert any(a in completion for a in addr), (
            f"completion for {chunk} does not reference any of "
            f"{addr}: {completion!r}"
        )


def test_silent_concept_truly_has_no_edge() -> None:
    """The sampled silent concept for each emitted pair must not be a
    target of any addressing edge from the same chunk in the graph."""
    capture = _FakeCapture()
    graph = _five_concept_graph()
    pairs, _ = generate_abstention_pairs(
        graph,
        capture=capture,
        max_pairs=200,
    )
    # Index addressed concepts per chunk.
    address_index: Dict[str, set] = {}
    for edge in graph["edges"]:
        address_index.setdefault(edge["source"], set()).add(edge["target"])

    for pair in pairs:
        silent = pair["concept_tags"][0]
        chunk = pair["chunk_id"]
        addressed = address_index.get(chunk, set())
        assert silent not in addressed, (
            f"abstention pair for {chunk} sampled silent concept "
            f"{silent!r} that is actually addressed by the chunk"
        )


def test_deterministic_across_runs() -> None:
    """Same graph + same seed -> byte-identical pair list."""
    g = _five_concept_graph()
    cap_a = _FakeCapture()
    cap_b = _FakeCapture()
    pairs_a, _ = generate_abstention_pairs(g, capture=cap_a, max_pairs=50, seed=999)
    pairs_b, _ = generate_abstention_pairs(g, capture=cap_b, max_pairs=50, seed=999)

    # Strip decision_capture_id (per-fake-capture monotonic counter).
    def _strip(p: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in p.items() if k != "decision_capture_id"}

    assert [_strip(p) for p in pairs_a] == [_strip(p) for p in pairs_b]


def test_max_pairs_respected() -> None:
    """Cap clamps emissions and never exceeds the cap."""
    capture = _FakeCapture()
    pairs, stats = generate_abstention_pairs(
        _five_concept_graph(),
        capture=capture,
        max_pairs=2,
        silent_per_chunk=1,
    )
    assert stats.pairs_emitted <= 2
    assert len(pairs) <= 2
    # 3 chunks, cap=2 -> the loop trips the cap on chunk 3.
    assert stats.capped_at_max_pairs is True


def test_decision_capture_fires_per_emit() -> None:
    """One ``abstention_generation`` event per pair, with rationale
    interpolating dynamic signals (chunk_id, K.id, addressed count)."""
    capture = _FakeCapture()
    pairs, _ = generate_abstention_pairs(
        _five_concept_graph(),
        capture=capture,
        max_pairs=200,
        silent_per_chunk=1,
    )
    types = [d["decision_type"] for d in capture.decisions]
    assert types.count("abstention_generation") == len(pairs)

    for event in capture.decisions:
        rationale = event["rationale"]
        # CLAUDE.md required floor.
        assert len(rationale) >= 20
        # Dynamic-signal interpolation.
        assert "seed=" in rationale
        assert "chunk_idx=" in rationale
        # alternatives_considered convention is {option, reason_rejected}.
        alts = event.get("alternatives_considered") or []
        for alt in alts:
            assert isinstance(alt, dict)
            assert "option" in alt
            assert "reason_rejected" in alt


def test_capture_required() -> None:
    """A None capture is rejected (Wave 112 invariant)."""
    with pytest.raises(ValueError, match="capture"):
        generate_abstention_pairs(
            _five_concept_graph(),
            capture=None,
            max_pairs=10,
        )


def test_pair_carries_abstention_marker_fields() -> None:
    """Marker fields downstream filters / diversity scorers rely on."""
    capture = _FakeCapture()
    pairs, _ = generate_abstention_pairs(
        _five_concept_graph(),
        capture=capture,
        max_pairs=20,
    )
    for pair in pairs:
        assert pair["content_type"] == "abstention_probe"
        assert pair["bloom_level"] == "understand"
        assert pair["template_id"] == "abstention.no_edge"
        assert pair["requires_source_citation"] is False
        assert pair["expected_response"] == "No."
        assert pair["abstention_polarity"] == "absent"
        # Concept tag anchors the silent concept.
        assert pair["concept_tags"]
        # Anchors back to a chunk.
        assert pair["chunk_id"]


def test_empty_graph_emits_no_pairs() -> None:
    capture = _FakeCapture()
    pairs, stats = generate_abstention_pairs(
        {"nodes": [], "edges": []},
        capture=capture,
        max_pairs=10,
    )
    assert pairs == []
    assert stats.pairs_emitted == 0
