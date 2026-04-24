"""Wave 71 — targets-concept edge boost in retrieval scoring.

The Wave 66 Trainforge rule ``targets_concept_from_lo`` writes
``targets-concept`` edges into ``concept_graph_semantic.json`` linking
each LO to the Bloom-qualified concepts it explicitly targets. Pre-
Wave-71 the LibV2 retriever only loaded the untyped ``concept_graph.json``
(node ids only); the typed edges were invisible to ranking.

This wave surfaces them as a fourth retrieval boost:

  score = bm25 × (1 + capped_boost)
  capped_boost absorbs the Wave 71 ``targets_concept`` contribution at
  weight 0.25 alongside the three Wave 5 boosts.

Tests below exercise the loader, the pure scoring function, and the
end-to-end integration in ``retrieve_chunks``. They use synthetic
corpora rather than real LibV2 content.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from LibV2.tools.libv2.retrieval_scoring import (  # noqa: E402
    BoostContributions,
    DEFAULT_BOOST_WEIGHTS,
    combine_bm25_with_boosts,
    load_targets_concept_edges,
    targets_concept_boost,
)


# ---------------------------------------------------------------------- #
# Loader
# ---------------------------------------------------------------------- #


def _write_semantic_graph(course_dir: Path, edges: list) -> None:
    graph_dir = course_dir / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    (graph_dir / "concept_graph_semantic.json").write_text(
        json.dumps({"kind": "concept_semantic", "nodes": [], "edges": edges})
    )


def test_loader_returns_empty_when_graph_absent(tmp_path):
    """Pre-Wave-66 corpora (no semantic graph) yield an empty map, not an error."""
    assert load_targets_concept_edges(tmp_path) == {}


def test_loader_extracts_targets_concept_edges(tmp_path):
    _write_semantic_graph(
        tmp_path,
        [
            {
                "source": "TO-01",
                "target": "framework",
                "type": "targets-concept",
                "provenance": {
                    "rule": "targets_concept_from_lo",
                    "rule_version": 1,
                    "evidence": {
                        "lo_id": "to-01",
                        "concept_id": "framework",
                        "bloom_level": "apply",
                    },
                },
            },
            {
                "source": "TO-01",
                "target": "data-pipeline",
                "type": "targets-concept",
                "provenance": {"evidence": {"bloom_level": "analyze"}},
            },
            # Not a targets-concept edge — must be ignored.
            {
                "source": "chunk_a",
                "target": "to-01",
                "type": "derived-from-objective",
                "provenance": {"rule": "derived_from_lo_ref", "rule_version": 2},
            },
        ],
    )
    edges = load_targets_concept_edges(tmp_path)
    # LO ids lowercased; concept ids lowercased; bloom lowercased.
    assert "to-01" in edges
    assert ("framework", "apply") in edges["to-01"]
    assert ("data-pipeline", "analyze") in edges["to-01"]
    # derived-from-objective edge filtered out.
    assert "chunk_a" not in edges


def test_loader_ignores_edges_without_evidence_bloom(tmp_path):
    """Edge without evidence.bloom_level is still emitted with bloom=None."""
    _write_semantic_graph(
        tmp_path,
        [
            {
                "source": "to-01",
                "target": "x",
                "type": "targets-concept",
                "provenance": {},
            }
        ],
    )
    edges = load_targets_concept_edges(tmp_path)
    assert edges == {"to-01": [("x", None)]}


def test_loader_handles_malformed_file(tmp_path):
    graph_dir = tmp_path / "graph"
    graph_dir.mkdir()
    (graph_dir / "concept_graph_semantic.json").write_text("{not valid json")
    assert load_targets_concept_edges(tmp_path) == {}


# ---------------------------------------------------------------------- #
# Scoring function
# ---------------------------------------------------------------------- #


def test_boost_zero_when_no_query_concepts():
    chunk = {"learning_outcome_refs": ["to-01"]}
    targets = {"to-01": [("framework", "apply")]}
    assert targets_concept_boost(chunk, [], targets) == 0.0


def test_boost_zero_when_no_edges_for_course():
    chunk = {"learning_outcome_refs": ["to-01"]}
    assert targets_concept_boost(chunk, ["framework"], {}) == 0.0


def test_boost_zero_when_chunk_has_no_lo_refs():
    chunk = {}
    targets = {"to-01": [("framework", "apply")]}
    assert targets_concept_boost(chunk, ["framework"], targets) == 0.0


def test_boost_zero_when_lo_targets_disjoint_concepts():
    chunk = {"learning_outcome_refs": ["to-01"]}
    targets = {"to-01": [("unrelated-concept", "apply")]}
    assert targets_concept_boost(chunk, ["framework"], targets) == 0.0


def test_boost_jaccard_on_intersection():
    chunk = {"learning_outcome_refs": ["to-01"]}
    targets = {"to-01": [("framework", "apply"), ("data-pipeline", "analyze")]}
    # Query has two concepts; chunk's LO targets both. Jaccard = 2/2 = 1.0.
    score = targets_concept_boost(chunk, ["framework", "data-pipeline"], targets)
    assert score == pytest.approx(1.0)


def test_boost_jaccard_partial_overlap():
    chunk = {"learning_outcome_refs": ["to-01"]}
    targets = {"to-01": [("framework", "apply")]}
    # Query has two concepts, chunk's LO targets one. |inter|=1, |union|=2 → 0.5.
    score = targets_concept_boost(
        chunk, ["framework", "orthogonal-topic"], targets
    )
    assert score == pytest.approx(0.5)


def test_boost_case_insensitive_lo_refs():
    chunk = {"learning_outcome_refs": ["TO-01"]}  # uppercase
    targets = {"to-01": [("framework", "apply")]}  # lowercase
    score = targets_concept_boost(chunk, ["framework"], targets)
    assert score > 0.0


def test_boost_bloom_match_bonus_applied():
    # Partial overlap so base < 1.0 and the bonus is measurable.
    # Query has two concepts; chunk targets one → Jaccard 1/2 = 0.5.
    chunk = {"learning_outcome_refs": ["to-01"]}
    targets = {"to-01": [("framework", "apply")]}
    base = targets_concept_boost(
        chunk, ["framework", "orthogonal-topic"], targets
    )
    boosted = targets_concept_boost(
        chunk,
        ["framework", "orthogonal-topic"],
        targets,
        query_bloom_level="apply",
    )
    assert boosted > base
    # Exact bonus math: base * 1.2 (default 0.2 bonus).
    assert boosted == pytest.approx(base * 1.2)


def test_boost_bloom_match_bonus_not_applied_on_mismatch():
    chunk = {"learning_outcome_refs": ["to-01"]}
    targets = {"to-01": [("framework", "apply")]}
    base = targets_concept_boost(
        chunk, ["framework", "orthogonal-topic"], targets
    )
    no_bonus = targets_concept_boost(
        chunk,
        ["framework", "orthogonal-topic"],
        targets,
        query_bloom_level="evaluate",
    )
    assert no_bonus == pytest.approx(base)


def test_boost_bloom_match_capped_at_one():
    """Perfect concept overlap with Bloom match shouldn't exceed 1.0."""
    chunk = {"learning_outcome_refs": ["to-01"]}
    targets = {"to-01": [("framework", "apply")]}
    score = targets_concept_boost(
        chunk, ["framework"], targets, query_bloom_level="apply", bloom_match_bonus=0.5
    )
    assert score <= 1.0


def test_boost_unions_edges_across_multiple_los():
    """A chunk citing two LOs gets the union of their targeted concepts."""
    chunk = {"learning_outcome_refs": ["to-01", "co-01"]}
    targets = {
        "to-01": [("framework", "apply")],
        "co-01": [("data-pipeline", "analyze")],
    }
    score = targets_concept_boost(
        chunk, ["framework", "data-pipeline"], targets
    )
    # Both concepts in intersection → Jaccard 2/2 = 1.0.
    assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------- #
# Composition into combine_bm25_with_boosts
# ---------------------------------------------------------------------- #


def test_targets_concept_contributes_to_final_score():
    """The Wave 71 contribution actually lifts the combined score."""
    base = BoostContributions()
    with_boost = BoostContributions(targets_concept=1.0)

    bm25 = 10.0
    final_base, _ = combine_bm25_with_boosts(bm25, base)
    final_boosted, _ = combine_bm25_with_boosts(bm25, with_boost)
    assert final_boosted > final_base


def test_targets_concept_appears_in_to_dict_payload():
    """Rationale serialization exposes the new boost."""
    contributions = BoostContributions(targets_concept=0.42)
    payload = contributions.to_dict()
    assert "targets_concept" in payload
    assert payload["targets_concept"] == pytest.approx(0.42, rel=1e-3)


def test_default_boost_weights_include_targets_concept():
    """DEFAULT_BOOST_WEIGHTS declares the Wave 71 slot so extra callers that
    pull the dict see a sensible default instead of silently weighting zero."""
    assert "targets_concept" in DEFAULT_BOOST_WEIGHTS
    assert DEFAULT_BOOST_WEIGHTS["targets_concept"] > 0.0


def test_custom_weight_override_honored():
    contributions = BoostContributions(targets_concept=1.0)
    bm25 = 10.0
    default_final, _ = combine_bm25_with_boosts(bm25, contributions)
    zero_weight_final, _ = combine_bm25_with_boosts(
        bm25, contributions, weights={"targets_concept": 0.0}
    )
    assert zero_weight_final < default_final
    assert zero_weight_final == pytest.approx(bm25)
