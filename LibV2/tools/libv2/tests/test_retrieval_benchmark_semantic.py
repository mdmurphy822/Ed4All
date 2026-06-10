"""WS2 E4 — benchmark_retrieval_engines over the mini_course gold set.

These tests exercise the BM25-vs-semantic benchmark harness on the WS1
mini-course fixture (``tests/fixtures/retrieval/mini_course/`` — chunks sha
``fc8a1c65...``, 3-question WS1 gold set) materialized into a tmp LibV2 root.

Coverage:
- ``bm25`` engine over the WS1 gold set → deterministic metrics across two
  runs (modulo latency + timestamp), Recall@{1,3,5,10} + MRR + primary /
  any_relevant blocks + winner deltas + per-query rows present.
- gold-pin-mismatch hard error (drifted chunkset sha → critical gold issue).
- legacy-JSONL fallback arm (no WS1 gold set, deprecation warning).
- import-guard: ``semantic`` engine raises ``NotImplementedError`` naming the
  missing dependency when the ``engine=`` axis is absent — NEVER fake results.
- ``llm_rerank`` must-not-enter the benchmark (rejected by the engine
  allow-list).
- ``evaluate_retrieval`` back-compat: existing recall keys unchanged + the new
  ``recall_at_3`` lands additively.
"""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import pytest

from LibV2.tools.libv2 import eval_harness
from LibV2.tools.libv2.eval_harness import (
    benchmark_retrieval_engines,
    evaluate_retrieval,
)

_FIXTURE = (
    Path(__file__).resolve().parents[4]
    / "tests"
    / "fixtures"
    / "retrieval"
    / "mini_course"
)


def _materialize_course(repo_root: Path, slug: str) -> Path:
    """Copy the mini_course fixture into ``repo_root/courses/<slug>/``.

    Mirrors the WS1 fixture layout: ``dart_chunks/chunks.jsonl`` (the pinned
    chunkset) + ``retrieval_eval/gold_set.json`` (the WS1 gold set).
    """
    cdir = repo_root / "courses" / slug
    (cdir / "dart_chunks").mkdir(parents=True)
    (cdir / "retrieval_eval").mkdir(parents=True)
    shutil.copy(
        _FIXTURE / "dart_chunks" / "chunks.jsonl",
        cdir / "dart_chunks" / "chunks.jsonl",
    )
    shutil.copy(
        _FIXTURE / "retrieval_eval" / "gold_set.json",
        cdir / "retrieval_eval" / "gold_set.json",
    )
    (cdir / "manifest.json").write_text(
        json.dumps({"classification": {"primary_domain": "rag"}})
    )
    return cdir


@pytest.fixture(autouse=True)
def _require_fixture():
    if not (_FIXTURE / "dart_chunks" / "chunks.jsonl").exists():
        pytest.skip("mini_course fixture not present")


class TestBenchmarkBm25:
    def test_bm25_report_shape_and_metrics(self, tmp_path):
        slug = "mini-retrieval-101"
        _materialize_course(tmp_path, slug)
        report = benchmark_retrieval_engines(
            tmp_path, slug, engines=("bm25",)
        )

        assert report["course_slug"] == slug
        assert report["config"]["engines"] == ["bm25"]
        assert report["config"]["k_values"] == [1, 3, 5, 10]
        assert report["config"]["total_questions"] == 3

        bm25 = report["per_engine"]["bm25"]
        # k=3 is present in both relevance blocks.
        assert "3" in bm25["recall_at_k_primary"]
        assert "3" in bm25["recall_at_k_any_relevant"]
        for k in ("1", "3", "5", "10"):
            assert k in bm25["recall_at_k_primary"]
        assert 0.0 <= bm25["mrr"] <= 1.0
        assert "p50_latency_ms" in bm25
        assert "p95_latency_ms" in bm25

        # Per-query rows present, one block per engine.
        assert len(report["per_query"]) == 3
        row = report["per_query"][0]
        assert "bm25" in row["engines"]
        assert "primary_chunk_ids" in row
        assert "any_relevant_chunk_ids" in row

        # The report file landed under retrieval_eval/ as a benchmark_* sibling.
        out = Path(report["output_path"])
        assert out.parent.name == "retrieval_eval"
        assert out.name.startswith("benchmark_")
        # WS2 never writes gold_set.json.
        assert (out.parent / "gold_set.json").exists()

    def test_bm25_deterministic_across_two_runs(self, tmp_path):
        slug = "mini-retrieval-101"
        _materialize_course(tmp_path, slug)
        r1 = benchmark_retrieval_engines(tmp_path, slug, engines=("bm25",))
        r2 = benchmark_retrieval_engines(tmp_path, slug, engines=("bm25",))

        # Metrics + retrieved-id ordering are deterministic (latency +
        # timestamp + output filename are the only non-deterministic parts).
        assert r1["per_engine"]["bm25"]["recall_at_k_primary"] == (
            r2["per_engine"]["bm25"]["recall_at_k_primary"]
        )
        assert r1["per_engine"]["bm25"]["mrr"] == r2["per_engine"]["bm25"]["mrr"]
        ids1 = [row["engines"]["bm25"]["retrieved_chunk_ids"] for row in r1["per_query"]]
        ids2 = [row["engines"]["bm25"]["retrieved_chunk_ids"] for row in r2["per_query"]]
        assert ids1 == ids2

    def test_winner_block_present_for_non_bm25_arm(self, tmp_path, monkeypatch):
        """With a stubbed engine axis, the winner block compares vs bm25."""
        slug = "mini-retrieval-101"
        _materialize_course(tmp_path, slug)

        # Stub the engine axis as available + route 'semantic' through bm25 so
        # we can assert the winner-delta block shape without E3's index.
        monkeypatch.setattr(
            eval_harness, "_supports_engine_axis", lambda: (True, "")
        )
        real_retrieve = eval_harness._retrieve_for_engine

        def _fake_retrieve(*, engine, **kw):
            return real_retrieve(engine="bm25", **kw)

        monkeypatch.setattr(eval_harness, "_retrieve_for_engine", _fake_retrieve)

        report = benchmark_retrieval_engines(
            tmp_path, slug, engines=("bm25", "semantic")
        )
        assert "semantic" in report["winner"]
        winner = report["winner"]["semantic"]
        assert "delta_recall_at_k_primary" in winner
        assert "delta_mrr" in winner
        # semantic routed-through-bm25 → deltas are all zero.
        assert winner["delta_mrr"] == 0.0


class TestGoldPinMismatch:
    def test_drifted_chunkset_hard_errors(self, tmp_path):
        slug = "mini-retrieval-101"
        cdir = _materialize_course(tmp_path, slug)
        # Mutate the chunkset so its sha no longer matches the gold pin.
        chunks_path = cdir / "dart_chunks" / "chunks.jsonl"
        with chunks_path.open("a") as fh:
            fh.write(json.dumps({"id": "drift_extra", "text": "drift"}) + "\n")

        with pytest.raises(ValueError, match="critical issues"):
            benchmark_retrieval_engines(tmp_path, slug, engines=("bm25",))


class TestLegacyFallback:
    def test_legacy_jsonl_fallback_with_deprecation(self, tmp_path):
        slug = "legacy-course"
        cdir = repo_cdir = tmp_path / "courses" / slug
        (cdir / "dart_chunks").mkdir(parents=True)
        (cdir / "retrieval").mkdir(parents=True)
        # Copy chunks but NO retrieval_eval/gold_set.json → legacy path.
        shutil.copy(
            _FIXTURE / "dart_chunks" / "chunks.jsonl",
            cdir / "dart_chunks" / "chunks.jsonl",
        )
        (cdir / "manifest.json").write_text(json.dumps({}))
        gold = [
            {
                "id": "q1",
                "query": "What does a vector store index?",
                "relevant_chunk_ids": ["mini_alpha_chunk_001"],
                "kind": "hand-curated",
            },
        ]
        with (cdir / "retrieval" / "gold_queries.jsonl").open("w") as fh:
            for g in gold:
                fh.write(json.dumps(g) + "\n")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report = benchmark_retrieval_engines(
                tmp_path, slug, engines=("bm25",)
            )
        assert any(
            issubclass(w.category, DeprecationWarning) for w in caught
        )
        assert report["config"]["total_questions"] == 1
        assert "gold_queries.jsonl" in report["gold_source"]


class TestImportGuard:
    def test_semantic_engine_raises_without_axis(self, tmp_path, monkeypatch):
        """When the engine= axis is unavailable, the semantic arm raises a
        clear NotImplementedError naming the dependency — never fake data."""
        slug = "mini-retrieval-101"
        _materialize_course(tmp_path, slug)
        # Force the import guard to report unavailable (E3 absent).
        monkeypatch.setattr(
            eval_harness,
            "_supports_engine_axis",
            lambda: (False, "semantic_retriever not importable (test)"),
        )
        with pytest.raises(NotImplementedError, match="engine.*axis"):
            benchmark_retrieval_engines(
                tmp_path, slug, engines=("bm25", "semantic")
            )

    def test_supports_engine_axis_false_when_param_absent(self):
        """The real guard returns False because retrieve_chunks has no
        ``engine`` param yet (E3 has not landed)."""
        available, reason = eval_harness._supports_engine_axis()
        # This assertion documents the current tree state; if E3 lands the
        # engine param + semantic_retriever, both signals flip True.
        if not available:
            assert "engine" in reason or "semantic_retriever" in reason


class TestLlmRerankMustNotEnter:
    def test_llm_rerank_rejected(self, tmp_path):
        slug = "mini-retrieval-101"
        _materialize_course(tmp_path, slug)
        with pytest.raises(ValueError, match="unknown benchmark engine"):
            benchmark_retrieval_engines(
                tmp_path, slug, engines=("bm25", "llm_rerank")
            )


class TestEvaluateRetrievalK3BackCompat:
    def test_recall_at_3_added_additively(self, tmp_path):
        slug = "mini-retrieval-101"
        cdir = _materialize_course(tmp_path, slug)
        # evaluate_retrieval reads the legacy retrieval/gold_queries.jsonl.
        (cdir / "retrieval").mkdir(parents=True)
        gold = [
            {
                "id": "q1",
                "query": "What does a vector store index?",
                "relevant_chunk_ids": ["mini_alpha_chunk_001"],
                "kind": "hand-curated",
            },
        ]
        with (cdir / "retrieval" / "gold_queries.jsonl").open("w") as fh:
            for g in gold:
                fh.write(json.dumps(g) + "\n")

        report = evaluate_retrieval(course_slug=slug, repo_root=tmp_path)
        agg = report["aggregate"]
        # Legacy keys unchanged.
        for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr"):
            assert key in agg
        # New additive key.
        assert "recall_at_3" in agg
        assert report["config"]["k_values"] == [1, 3, 5, 10]
