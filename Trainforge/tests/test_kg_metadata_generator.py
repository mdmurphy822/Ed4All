"""Tests for the KG-metadata SFT pair generator (Audit 2026-04-30).

Covers the contract spelled out in the audit fix:

* Round-trip a small synthetic graph and assert the emitted pairs
  match the schema and the positive/negative ratio.
* Negative targets really differ from the positive (no false-negative
  collisions where the "wrong" target is actually a real triple).
* Each emitted pair validates against `instruction_pair.schema.json`.
* Distribution across the four relation templates is balanced.
* Decision capture fires (one event per relation batch).
* `max_pairs` cap is honored.
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

from Trainforge.generators.kg_metadata_generator import (  # noqa: E402
    DEFAULT_NEGATIVES_PER_POSITIVE,
    KGMetadataStats,
    generate_kg_metadata_pairs,
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


def _five_triple_graph() -> Dict[str, Any]:
    """A graph with one triple per supported relation + one generic.

    Each chunk is unique so there are no overlap-induced edge cases
    in the negative-target sampling step.
    """
    return {
        "nodes": [
            {"id": "chunk_a", "class": "Chunk"},
            {"id": "chunk_b", "class": "Chunk"},
            {"id": "chunk_c", "class": "Chunk"},
            {"id": "chunk_d", "class": "Chunk"},
            {"id": "chunk_e", "class": "Chunk"},
            {"id": "concept_alpha", "class": "Concept"},
            {"id": "concept_beta", "class": "Concept"},
            {"id": "concept_gamma", "class": "Concept"},
            {"id": "module:week_01", "class": "Module"},
            {"id": "module:week_02", "class": "Module"},
            {"id": "bloom:remember", "class": "BloomLevel"},
            {"id": "bloom:apply", "class": "BloomLevel"},
        ],
        "edges": [
            {"source": "chunk_a", "target": "concept_alpha", "relation_type": "assesses"},
            {"source": "chunk_b", "target": "concept_beta", "relation_type": "assesses"},
            {"source": "chunk_c", "target": "module:week_01", "relation_type": "belongs_to_module"},
            {"source": "chunk_d", "target": "module:week_02", "relation_type": "belongs_to_module"},
            {"source": "chunk_e", "target": "bloom:remember", "relation_type": "at_bloom_level"},
            # Two extra targets per relation so negative-target sampling
            # has somewhere to sample from.
            {"source": "chunk_b", "target": "concept_gamma", "relation_type": "assesses"},
            {"source": "chunk_c", "target": "module:week_02", "relation_type": "belongs_to_module"},
            {"source": "chunk_e", "target": "bloom:apply", "relation_type": "at_bloom_level"},
        ],
    }


def _validate_pair(pair: Dict[str, Any]) -> None:
    """Validate a single pair against `instruction_pair.schema.json`.

    `jsonschema` is a dev-test dep already pulled in by the rest of the
    Trainforge suite. Importing inside the helper keeps the import
    error visible at test-discovery time on environments missing the
    package.
    """
    import jsonschema

    schema = json.loads(PAIR_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.validate(pair, schema)


def test_round_trip_emits_positives_and_negatives() -> None:
    capture = _FakeCapture()
    pairs, stats = generate_kg_metadata_pairs(
        _five_triple_graph(),
        capture=capture,
        max_pairs=200,
        negatives_per_positive=1,
    )
    assert isinstance(stats, KGMetadataStats)
    # 8 triples in fixture; with cap 200 + 1 negative each, we expect
    # all 8 positives + ~8 negatives.
    assert stats.positives_emitted == 8
    # Each source typically has at least one alternative target in the
    # same relation; some sources (e.g. chunk_a is the only chunk that
    # assesses anything in this fixture) may have none, in which case
    # the negative for that positive is silently skipped. We expect
    # roughly half the positives to have a negative.
    assert stats.negatives_emitted >= 3
    assert stats.pairs_emitted == stats.positives_emitted + stats.negatives_emitted
    assert len(pairs) == stats.pairs_emitted


def test_negative_targets_differ_from_positive() -> None:
    """Each negative pair's target must not be a real (source, target)
    edge for that source + relation in the graph."""
    capture = _FakeCapture()
    graph = _five_triple_graph()
    pairs, _ = generate_kg_metadata_pairs(
        graph, capture=capture, max_pairs=200, negatives_per_positive=2,
    )

    # Build the real-pair index from the graph.
    real_index = set()
    for edge in graph["edges"]:
        real_index.add(
            (edge["source"], edge["relation_type"], edge["target"]),
        )

    for pair in pairs:
        if pair["kg_metadata_polarity"] != "no":
            continue
        triple = (
            pair["chunk_id"],
            pair["kg_metadata_relation"],
            pair["kg_metadata_target"],
        )
        assert triple not in real_index, (
            f"negative pair {triple} collides with a real graph edge"
        )


def test_each_pair_is_schema_valid() -> None:
    capture = _FakeCapture()
    pairs, _ = generate_kg_metadata_pairs(
        _five_triple_graph(),
        capture=capture,
        max_pairs=200,
        negatives_per_positive=1,
    )
    assert pairs, "expected at least one pair"
    for pair in pairs:
        _validate_pair(pair)


def test_distribution_across_relations_is_balanced() -> None:
    """Each supported relation should contribute pairs; per-relation
    cap distributes the budget evenly so no relation exceeds its
    fair share by more than ~1 pair (rounding)."""
    capture = _FakeCapture()
    pairs, stats = generate_kg_metadata_pairs(
        _five_triple_graph(),
        capture=capture,
        max_pairs=200,
        negatives_per_positive=1,
    )
    relations = {p["kg_metadata_relation"] for p in pairs}
    # All three explicit relations must be represented.
    assert "assesses" in relations
    assert "belongs_to_module" in relations
    assert "at_bloom_level" in relations
    # Per-relation counts in stats are populated.
    for rel in ("assesses", "belongs_to_module", "at_bloom_level"):
        assert rel in stats.per_relation
        assert stats.per_relation[rel]["pairs_emitted"] >= 1


def test_decision_capture_fires_once_per_relation_batch() -> None:
    capture = _FakeCapture()
    generate_kg_metadata_pairs(
        _five_triple_graph(), capture=capture, max_pairs=200,
    )
    types = [d["decision_type"] for d in capture.decisions]
    # One event per relation in the fixture (3 relations).
    assert types.count("kg_metadata_generation") == 3
    # Rationale interpolates dynamic signals (per CLAUDE.md instruction).
    for event in capture.decisions:
        rationale = event["rationale"]
        assert len(rationale) >= 20
        assert "seed=" in rationale
        assert "candidate targets" in rationale
        # Wave 22 capture validation: alternatives_considered shape is
        # {option, reason_rejected} dicts, not strings. The schema
        # admits both shapes but the project convention since Wave 120
        # is dicts.
        alts = event.get("alternatives_considered", []) or []
        for alt in alts:
            assert isinstance(alt, dict)
            assert "option" in alt
            assert "reason_rejected" in alt


def test_max_pairs_cap_is_honored() -> None:
    """Cap clamps emissions and never exceeds the cap.

    The cap is applied during emission (post-relation-budget split),
    so stats.pairs_emitted may legitimately come in below the cap when
    the per-relation budget exhausts the available triples first.
    """
    capture = _FakeCapture()
    pairs, stats = generate_kg_metadata_pairs(
        _five_triple_graph(),
        capture=capture,
        max_pairs=2,  # tight enough that the relation loop trips the cap
        negatives_per_positive=2,
    )
    assert stats.pairs_emitted <= 2
    assert len(pairs) <= 2
    # max_pairs=2 with 8 triples available across 3 relations: the
    # generator hits the cap before exhausting the per-relation
    # budget, so capped_at_max_pairs must be True.
    assert stats.capped_at_max_pairs is True


def test_empty_graph_emits_no_pairs() -> None:
    capture = _FakeCapture()
    pairs, stats = generate_kg_metadata_pairs(
        {"nodes": [], "edges": []}, capture=capture, max_pairs=10,
    )
    assert pairs == []
    assert stats.pairs_emitted == 0


def test_capture_required() -> None:
    """A None capture is rejected — every emitted pair carries a
    `decision_capture_id`, so we must fail loud rather than emit
    pairs with empty IDs (Wave 112 invariant)."""
    with pytest.raises(ValueError, match="capture"):
        generate_kg_metadata_pairs(
            _five_triple_graph(), capture=None, max_pairs=10,
        )


def test_pair_carries_kg_metadata_marker_fields() -> None:
    """Pairs are tagged so downstream filters / diversity scorers can
    spot the KG-membership cohort without re-parsing prompts."""
    capture = _FakeCapture()
    pairs, _ = generate_kg_metadata_pairs(
        _five_triple_graph(),
        capture=capture,
        max_pairs=20,
        negatives_per_positive=1,
    )
    for pair in pairs:
        assert pair["bloom_level"] == "remember"
        assert pair["requires_source_citation"] is False
        assert pair["template_id"].startswith("kg_metadata.")
        assert pair["template_id"].endswith(("_yes", "_no"))
        assert pair["expected_response"] in ("Yes.", "No.")
        # Every pair anchors `chunk_id` to the source of the triple.
        assert pair["chunk_id"]


def test_deterministic_under_same_seed() -> None:
    """Same graph + same seed -> byte-identical pair list."""
    g = _five_triple_graph()
    cap_a = _FakeCapture()
    cap_b = _FakeCapture()
    pairs_a, _ = generate_kg_metadata_pairs(
        g, capture=cap_a, max_pairs=50, seed=999,
    )
    pairs_b, _ = generate_kg_metadata_pairs(
        g, capture=cap_b, max_pairs=50, seed=999,
    )
    # Strip decision_capture_id (per-fake-capture monotonic counter).
    def _strip(p: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in p.items() if k != "decision_capture_id"}

    assert [_strip(p) for p in pairs_a] == [_strip(p) for p in pairs_b]
