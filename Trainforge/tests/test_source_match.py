"""Wave 102 - SourceMatchEvaluator tests.

Synthesizes a tiny holdout split with three chunk-anchored edges and
asserts:

* Match rate equals the fraction of probes where the response cites
  the ground-truth chunk_id.
* Non-chunk-anchored edges (concept->concept) are filtered out so
  source-match only scores edges where a single chunk actually grounds
  the fact.
* Custom citation patterns are honoured.
* Errors in model_callable are recorded but don't crash the eval.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _write_holdout_split(path: Path, edges):
    payload = {
        "course_slug": "test",
        "seed": 42,
        "holdout_pct": 0.1,
        "edges_total": len(edges),
        "edges_held_out": len(edges),
        "per_relation": {},
        "bloom_strata": {},
        "withheld_edges": edges,
        "holdout_graph_hash": "0" * 64,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_source_match_perfect_score(tmp_path):
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "holdout_split.json"
    edges = [
        {"source": "chunk_a1", "target": "concept_x", "relation_type": "teaches"},
        {"source": "chunk_b2", "target": "concept_y", "relation_type": "exemplifies"},
    ]
    _write_holdout_split(holdout, edges)

    def model(_probe: str) -> str:
        # Always cite both possible chunk IDs
        return "yes [chunk_a1] [chunk_b2] explains it"

    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=model,
    ).evaluate()
    assert result["source_match_rate"] == pytest.approx(1.0)
    assert result["matches"] == 2
    assert result["scored_total"] == 2


def test_source_match_filters_non_chunk_edges(tmp_path):
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "holdout_split.json"
    edges = [
        # concept->concept, NOT chunk-anchored: should be filtered out
        {"source": "concept_a", "target": "concept_b",
         "relation_type": "prerequisite_of"},
        # chunk-anchored
        {"source": "chunk_xyz", "target": "concept_z", "relation_type": "teaches"},
    ]
    _write_holdout_split(holdout, edges)

    def model(_probe: str) -> str:
        return "yes [chunk_xyz]"

    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=model,
    ).evaluate()
    # Only one edge scored (the chunk-anchored one); rate is 1.0
    assert result["scored_total"] == 1
    assert result["source_match_rate"] == pytest.approx(1.0)


def test_source_match_partial(tmp_path):
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "holdout.json"
    edges = [
        {"source": "chunk_one", "target": "concept_x", "relation_type": "teaches"},
        {"source": "chunk_two", "target": "concept_y", "relation_type": "teaches"},
        {"source": "chunk_three", "target": "concept_z", "relation_type": "teaches"},
    ]
    _write_holdout_split(holdout, edges)

    # Cites the right chunk only one time out of three
    responses = iter([
        "[chunk_one] yes",       # correct
        "[chunk_wrong] uh-oh",   # wrong cite
        "no citation here",      # missing
    ])

    def model(_probe: str) -> str:
        return next(responses)

    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=model,
    ).evaluate()
    assert result["matches"] == 1
    assert result["scored_total"] == 3
    assert result["source_match_rate"] == pytest.approx(1.0 / 3.0)


def test_source_match_accepts_bracket_form(tmp_path):
    """Wave 105: form 1 — canonical [chunk_NNNN] still works."""
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "h.json"
    edges = [
        {"source": "chunk_00270", "target": "concept", "relation_type": "teaches"},
    ]
    _write_holdout_split(holdout, edges)
    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=lambda _p: "see [chunk_00270] for details",
    ).evaluate()
    assert result["matches"] == 1
    assert result["source_match_rate"] == pytest.approx(1.0)


def test_source_match_accepts_single_quoted_short_form(tmp_path):
    """Wave 105: form 2 — single-quoted bare suffix 'chunk_NNNN'.

    The trained model emitted this exact form in the Wave 104 eval
    (e.g. 'rdf_shacl_551_chunk_00270') so source-match must not
    discount it.
    """
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "h.json"
    edges = [
        {"source": "chunk_00270", "target": "concept", "relation_type": "teaches"},
    ]
    _write_holdout_split(holdout, edges)
    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=lambda _p: "the answer is in 'chunk_00270'",
    ).evaluate()
    assert result["matches"] == 1


def test_source_match_accepts_single_quoted_full_corpus_id(tmp_path):
    """Wave 105: form 3 — 'rdf_shacl_551_chunk_NNNN' single-quoted
    full corpus ID, normalized to ``chunk_NNNN``."""
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "h.json"
    edges = [
        {"source": "chunk_00270", "target": "concept", "relation_type": "teaches"},
    ]
    _write_holdout_split(holdout, edges)
    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=lambda _p: "from 'rdf_shacl_551_chunk_00270' we know",
    ).evaluate()
    assert result["matches"] == 1


def test_source_match_scores_full_corpus_source_ids(tmp_path):
    """Full RDF/SHACL chunk IDs in holdout source must be scored."""
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "h.json"
    edges = [
        {
            "source": "rdf_shacl_551_chunk_00270",
            "target": "concept",
            "relation_type": "teaches",
        },
    ]
    _write_holdout_split(holdout, edges)
    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=lambda _p: "see [rdf_shacl_551_chunk_00270]",
    ).evaluate()
    assert result["scored_total"] == 1
    assert result["matches"] == 1


def test_source_match_accepts_bare_token_form(tmp_path):
    """Wave 105: form 4 — bare ``chunk_NNNN`` without delimiters."""
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "h.json"
    edges = [
        {"source": "chunk_00270", "target": "concept", "relation_type": "teaches"},
    ]
    _write_holdout_split(holdout, edges)
    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=lambda _p: "as chunk_00270 explains, ...",
    ).evaluate()
    assert result["matches"] == 1


def test_source_match_score_non_none_when_holdout_has_chunk_ids(tmp_path):
    """Wave 105: a ground-truth chunk_id present in the holdout means
    source-match returns a numeric score (not None) regardless of
    whether the model cites it."""
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "h.json"
    edges = [
        {"source": "chunk_a", "target": "concept", "relation_type": "teaches"},
        {"source": "chunk_b", "target": "concept", "relation_type": "exemplifies"},
    ]
    _write_holdout_split(holdout, edges)
    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=lambda _p: "no citation",
    ).evaluate()
    assert result["source_match_rate"] is not None
    assert isinstance(result["source_match_rate"], float)
    assert result["scored_total"] == 2


def test_source_match_handles_callable_error(tmp_path):
    from Trainforge.eval.source_match import SourceMatchEvaluator

    holdout = tmp_path / "holdout.json"
    edges = [
        {"source": "chunk_x", "target": "concept", "relation_type": "teaches"},
    ]
    _write_holdout_split(holdout, edges)

    def boom(_probe: str) -> str:
        raise RuntimeError("model down")

    result = SourceMatchEvaluator(
        holdout_split=holdout,
        model_callable=boom,
    ).evaluate()
    assert result["scored_total"] == 0
    assert result["matches"] == 0
    assert result["source_match_rate"] == 0.0
    assert result["errors"]
    assert "model down" in result["errors"][0]
