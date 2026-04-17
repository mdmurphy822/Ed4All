"""Worker J tests: evaluate_retrieval against a synthetic 3-chunk / 2-query
fixture course.  Asserts per-query ranks, recall@k, MRR."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from LibV2.tools.libv2.eval_harness import evaluate_retrieval


def _write_fixture(repo_root: Path, slug: str) -> None:
    """Build a tiny repo with one course carrying 3 chunks + 2 gold queries."""
    cdir = repo_root / "courses" / slug
    (cdir / "corpus").mkdir(parents=True)
    (cdir / "retrieval").mkdir()
    (cdir / "graph").mkdir()

    chunks = [
        {
            "id": "c1", "schema_version": "v4",
            "text": "WCAG SC 1.4.3 Contrast Minimum requires 4.5:1 for body text",
            "chunk_type": "explanation", "difficulty": "intermediate",
            "concept_tags": ["color-contrast", "wcag"],
            "learning_outcome_refs": ["co-05"],
            "source": {"module_id": "week_03_content", "module_title": "Contrast"},
            "tokens_estimate": 30, "bloom_level": "apply",
        },
        {
            "id": "c2", "schema_version": "v4",
            "text": "ARIA live regions announce dynamic content to screen readers",
            "chunk_type": "explanation", "difficulty": "intermediate",
            "concept_tags": ["aria-live", "screen-reader"],
            "learning_outcome_refs": ["co-16"],
            "source": {"module_id": "week_08_content", "module_title": "ARIA"},
            "tokens_estimate": 25, "bloom_level": "apply",
        },
        {
            "id": "c3", "schema_version": "v4",
            "text": "Skip links help keyboard users bypass repetitive navigation",
            "chunk_type": "example", "difficulty": "foundational",
            "concept_tags": ["skip-link", "keyboard-navigation"],
            "learning_outcome_refs": ["co-06"],
            "source": {"module_id": "week_04_content", "module_title": "Keyboard"},
            "tokens_estimate": 20, "bloom_level": "understand",
        },
    ]
    with open(cdir / "corpus" / "chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    (cdir / "manifest.json").write_text(json.dumps({
        "classification": {"primary_domain": "accessibility"},
    }))
    # Minimal concept graph for the boost path
    (cdir / "graph" / "concept_graph.json").write_text(json.dumps({
        "kind": "concept",
        "nodes": [{"id": t, "label": t, "frequency": 1}
                  for t in ["color-contrast", "aria-live", "skip-link",
                            "wcag", "screen-reader", "keyboard-navigation"]],
        "edges": [],
    }))

    # Two gold queries, one obvious, one moderately ambiguous
    gold = [
        {"id": "q1", "query": "color contrast body text WCAG",
         "relevant_chunk_ids": ["c1"], "kind": "hand-curated",
         "notes": "c1 is the canonical contrast chunk"},
        {"id": "q2", "query": "skip link keyboard bypass",
         "relevant_chunk_ids": ["c3"], "kind": "hand-curated",
         "notes": "c3 is the only skip-link chunk"},
    ]
    with open(cdir / "retrieval" / "gold_queries.jsonl", "w") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")


class TestEvaluateRetrieval:
    def test_runs_and_reports_aggregates(self, tmp_path):
        _write_fixture(tmp_path, "fx-course")
        report = evaluate_retrieval(
            course_slug="fx-course", repo_root=tmp_path,
        )
        agg = report["aggregate"]
        # Both queries should land their relevant chunk at rank 1
        assert agg["total_queries"] == 2
        assert agg["mrr"] == 1.0
        assert agg["recall_at_1"] == 1.0
        assert agg["recall_at_5"] == 1.0
        assert agg["recall_at_10"] == 1.0

    def test_per_query_shape(self, tmp_path):
        _write_fixture(tmp_path, "fx-course")
        report = evaluate_retrieval(
            course_slug="fx-course", repo_root=tmp_path,
            include_rationale=True,
        )
        assert len(report["per_query"]) == 2
        for entry in report["per_query"]:
            for key in (
                "id", "query", "relevant_chunk_ids", "retrieved_chunk_ids",
                "matched_chunk_ids", "rank_of_first_relevant", "reciprocal_rank",
                "recall_at_1", "recall_at_5", "recall_at_10",
            ):
                assert key in entry
        # Rationale attached on top result
        for entry in report["per_query"]:
            assert entry.get("top_result_rationale") is not None

    def test_report_written_to_default_path(self, tmp_path):
        _write_fixture(tmp_path, "fx-course")
        evaluate_retrieval(course_slug="fx-course", repo_root=tmp_path)
        out = tmp_path / "courses" / "fx-course" / "retrieval" / "evaluation_results.json"
        assert out.exists()

    def test_missing_gold_queries_raises(self, tmp_path):
        (tmp_path / "courses" / "empty" / "retrieval").mkdir(parents=True)
        # No gold_queries.jsonl written
        with pytest.raises(FileNotFoundError):
            evaluate_retrieval(course_slug="empty", repo_root=tmp_path)
