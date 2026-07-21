"""WS2 Wave C — CliRunner tests for ``libv2 retrieval-benchmark``.

The command wraps ``eval_harness.benchmark_retrieval_engines`` over the WS1
mini-course fixture (``tests/fixtures/retrieval/mini_course/``, chunks sha
``fc8a1c65...``, 3-question gold set) materialized into a tmp LibV2 root.

Coverage:
- happy path over the fake embedding provider (``--build-index`` then
  benchmark bm25 + semantic + hybrid-rrf) — exit 0, table to stdout, JSON
  artifact under ``retrieval_eval/`` (a ``benchmark_*`` sibling, never the
  gold set), deterministic JSON across two runs (modulo timestamp/filename).
- fail-closed exits: a semantic engine with no index + no ``--build-index``
  exits non-zero with the ``SemanticIndexMissing`` build guidance; a missing
  gold set exits non-zero with ``FileNotFoundError``; an unknown engine and
  the LLM-backed ``llm_rerank`` mode are rejected.
- table sanity: every requested engine + every k cutoff appears.
- ``--gold-set`` override + ``--out`` honored.

All runs use the deterministic ``fake`` provider via
``ED4ALL_EMBEDDING_ALLOW_FAKE`` so no model weights are loaded (CI default).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from LibV2.tools.libv2.cli import main as libv2_main  # noqa: E402

_FIXTURE = (
    PROJECT_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"
)
_SLUG = "mini-retrieval-101"


@pytest.fixture(autouse=True)
def _require_fixture():
    if not (_FIXTURE / "semantik_chunks" / "chunks.jsonl").exists():
        pytest.skip("mini_course fixture not present")


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch):
    """Route every embedding call through the deterministic fake provider."""
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")


def _materialize(repo_root: Path, *, with_gold: bool = True) -> Path:
    """Copy the mini_course chunkset (+ gold set) into ``repo_root``."""
    cdir = repo_root / "courses" / _SLUG
    (cdir / "semantik_chunks").mkdir(parents=True)
    shutil.copy(
        _FIXTURE / "semantik_chunks" / "chunks.jsonl",
        cdir / "semantik_chunks" / "chunks.jsonl",
    )
    if with_gold:
        (cdir / "retrieval_eval").mkdir(parents=True)
        shutil.copy(
            _FIXTURE / "retrieval_eval" / "gold_set.json",
            cdir / "retrieval_eval" / "gold_set.json",
        )
    (cdir / "manifest.json").write_text(
        json.dumps({"classification": {"primary_domain": "rag"}})
    )
    return cdir


def _invoke(repo_root: Path, *args: str):
    return CliRunner().invoke(
        libv2_main, ["--repo", str(repo_root), "retrieval-benchmark", *args]
    )


class TestHappyPath:
    def test_build_index_then_benchmark_all_engines(self, tmp_path):
        pytest.importorskip("numpy")  # --build-index builds a vector index (needs [embedding])
        cdir = _materialize(tmp_path)
        res = _invoke(
            tmp_path,
            "--course",
            _SLUG,
            "--engines",
            "bm25,semantic,hybrid-rrf",
            "--build-index",
        )
        assert res.exit_code == 0, res.output

        # Table mentions every engine + every k cutoff header.
        out = res.output
        for engine in ("bm25", "semantic", "hybrid-rrf"):
            assert engine in out
        for k in ("R@1", "R@3", "R@5", "R@10"):
            assert k in out
        assert "MRR" in out
        assert "vs BM25 baseline" in out

        # JSON artifact landed under retrieval_eval/ as a benchmark_* sibling.
        eval_dir = cdir / "retrieval_eval"
        reports = list(eval_dir.glob("benchmark_*.json"))
        assert len(reports) == 1
        assert (eval_dir / "gold_set.json").exists()  # never overwritten

        report = json.loads(reports[0].read_text())
        assert report["course_slug"] == _SLUG
        assert report["config"]["engines"] == ["bm25", "semantic", "hybrid-rrf"]
        assert report["config"]["total_questions"] == 3
        assert "semantic" in report["per_engine"]
        assert "3" in report["per_engine"]["bm25"]["recall_at_k_primary"]
        # The printed report path resolves to the written file.
        assert str(reports[0]) in out

    def test_bm25_only_runs_without_index(self, tmp_path):
        """The lexical arm needs no vector index — bm25-only succeeds."""
        _materialize(tmp_path)
        res = _invoke(tmp_path, "--course", _SLUG, "--engines", "bm25")
        assert res.exit_code == 0, res.output
        assert "bm25" in res.output

    def test_out_override_and_custom_k(self, tmp_path):
        _materialize(tmp_path)
        out_path = tmp_path / "custom_bench.json"
        res = _invoke(
            tmp_path,
            "--course",
            _SLUG,
            "--engines",
            "bm25",
            "--k",
            "1,5",
            "--out",
            str(out_path),
        )
        assert res.exit_code == 0, res.output
        assert out_path.exists()
        report = json.loads(out_path.read_text())
        assert report["config"]["k_values"] == [1, 5]
        assert set(report["per_engine"]["bm25"]["recall_at_k_primary"]) == {"1", "5"}

    def test_gold_set_override_legacy_jsonl(self, tmp_path):
        """An explicit --gold-set .jsonl path is loaded directly (legacy arm),
        overriding the in-course gold resolution."""
        _materialize(tmp_path, with_gold=False)
        override = tmp_path / "my_gold.jsonl"
        override.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "query": "What does a vector store index?",
                    "relevant_chunk_ids": ["mini_alpha_chunk_001"],
                    "kind": "hand-curated",
                }
            )
            + "\n"
        )
        res = _invoke(
            tmp_path,
            "--course",
            _SLUG,
            "--engines",
            "bm25",
            "--gold-set",
            str(override),
        )
        assert res.exit_code == 0, res.output
        assert str(override) in res.output
        assert "1 questions" in res.output


class TestDeterminism:
    def test_json_deterministic_across_runs(self, tmp_path):
        pytest.importorskip("numpy")  # --build-index builds a vector index (needs [embedding])
        cdir = _materialize(tmp_path)
        out1 = tmp_path / "r1.json"
        out2 = tmp_path / "r2.json"
        for out in (out1, out2):
            res = _invoke(
                tmp_path,
                "--course",
                _SLUG,
                "--engines",
                "bm25,semantic,hybrid-rrf",
                "--build-index",
                "--out",
                str(out),
            )
            assert res.exit_code == 0, res.output

        r1 = json.loads(out1.read_text())
        r2 = json.loads(out2.read_text())
        # Metrics + retrieved id ordering are deterministic (timestamp and
        # latency are the only non-deterministic fields).
        for engine in ("bm25", "semantic", "hybrid-rrf"):
            assert (
                r1["per_engine"][engine]["recall_at_k_primary"]
                == r2["per_engine"][engine]["recall_at_k_primary"]
            )
            assert r1["per_engine"][engine]["mrr"] == r2["per_engine"][engine]["mrr"]
        ids1 = [
            row["engines"]["semantic"]["retrieved_chunk_ids"]
            for row in r1["per_query"]
        ]
        ids2 = [
            row["engines"]["semantic"]["retrieved_chunk_ids"]
            for row in r2["per_query"]
        ]
        assert ids1 == ids2


class TestFailClosed:
    def test_semantic_without_index_fails_closed(self, tmp_path):
        """semantic engine + no index + no --build-index => non-zero exit with
        the build guidance, never a BM25-only result."""
        # Without numpy the semantic path can't even reach the typed
        # SemanticIndexMissing guidance (the [embedding] extra is the
        # precondition for a meaningful fail-closed assertion); the CLI's
        # ImportError handler still exits 1, but this test asserts the
        # *typed* build-guidance, which requires the extra.
        pytest.importorskip("numpy")  # needs [embedding] to exercise the typed guidance
        _materialize(tmp_path)
        res = _invoke(tmp_path, "--course", _SLUG, "--engines", "semantic")
        assert res.exit_code == 1
        assert "vector-index build" in res.output

    def test_missing_gold_set_fails_closed(self, tmp_path):
        _materialize(tmp_path, with_gold=False)
        res = _invoke(tmp_path, "--course", _SLUG, "--engines", "bm25")
        assert res.exit_code == 1
        assert "no gold set" in res.output

    def test_unknown_engine_rejected(self, tmp_path):
        _materialize(tmp_path)
        res = _invoke(tmp_path, "--course", _SLUG, "--engines", "bm25,banana")
        assert res.exit_code == 1
        assert "unknown benchmark engine" in res.output

    def test_llm_rerank_must_not_enter(self, tmp_path):
        _materialize(tmp_path)
        res = _invoke(tmp_path, "--course", _SLUG, "--engines", "bm25,llm_rerank")
        assert res.exit_code == 1
        assert "unknown benchmark engine" in res.output

    def test_unknown_course_fails(self, tmp_path):
        (tmp_path / "courses").mkdir(parents=True)
        res = _invoke(tmp_path, "--course", "no-such", "--engines", "bm25")
        assert res.exit_code == 1
        assert "course not found" in res.output

    def test_bad_k_rejected(self, tmp_path):
        _materialize(tmp_path)
        res = _invoke(tmp_path, "--course", _SLUG, "--engines", "bm25", "--k", "a,b")
        assert res.exit_code == 1
        assert "--k" in res.output
