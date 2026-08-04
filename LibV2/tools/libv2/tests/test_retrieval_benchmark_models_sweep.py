"""CliRunner tests for ``libv2 retrieval-benchmark --models``.

Covers the per-model sweep orchestration the eval harness docstring punts to
the CLI: for each model id the command builds a temp index alongside the
canonical ``vector_index/`` (``vector_index.bench-<tag>/``), runs the
semantic/hybrid benchmark against it, writes one tagged report per model with
the requested model list recorded in ``config.models``, and preserves the
canonical index across the sweep (cleaning temp dirs unless ``--keep``).

All runs use the deterministic ``fake`` provider (``ED4ALL_EMBEDDING_PROVIDER``
+ ``ED4ALL_EMBEDDING_ALLOW_FAKE``) so no model weights are loaded; the two
fake "models" produce identical vectors but exercise the orchestration loop,
temp-dir lifecycle, per-model report naming, and canonical-index preservation.
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

_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "retrieval" / "mini_course"
_SLUG = "mini-retrieval-101"


@pytest.fixture(autouse=True)
def _require_fixture():
    if not (_FIXTURE / "semantik_chunks" / "chunks.jsonl").exists():
        pytest.skip("mini_course fixture not present")


@pytest.fixture(autouse=True)
def _fake_provider(monkeypatch):
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("ED4ALL_EMBEDDING_ALLOW_FAKE", "true")


def _materialize(repo_root: Path) -> Path:
    cdir = repo_root / "courses" / _SLUG
    (cdir / "semantik_chunks").mkdir(parents=True)
    shutil.copy(
        _FIXTURE / "semantik_chunks" / "chunks.jsonl",
        cdir / "semantik_chunks" / "chunks.jsonl",
    )
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


# Two distinct fake "model ids"; both route to the fake provider but exercise
# the per-model loop, naming, and temp-dir lifecycle.
_MODELS = "fakeorg/alpha-embed,fakeorg/beta-embed"


class TestModelSweep:
    def test_sweep_writes_one_report_per_model(self, tmp_path):
        pytest.importorskip("numpy")  # sweep builds vector indexes (needs [embedding])
        cdir = _materialize(tmp_path)
        res = _invoke(
            tmp_path,
            "--course",
            _SLUG,
            "--engines",
            "bm25,semantic,hybrid-rrf",
            "--models",
            _MODELS,
        )
        assert res.exit_code == 0, res.output

        eval_dir = cdir / "retrieval_eval"
        reports = sorted(eval_dir.glob("benchmark_*.json"))
        assert len(reports) == 2, [p.name for p in reports]
        names = {p.name for p in reports}
        assert "benchmark_alpha-embed.json" in names
        assert "benchmark_beta-embed.json" in names

        for rp in reports:
            doc = json.loads(rp.read_text())
            # The requested model list is recorded in every report.
            assert doc["config"]["models"] == [
                "fakeorg/alpha-embed",
                "fakeorg/beta-embed",
            ]
            assert doc["config"]["engines"] == [
                "bm25",
                "semantic",
                "hybrid-rrf",
            ]
            assert "semantic" in doc["per_engine"]

        # gold set never overwritten.
        assert (eval_dir / "gold_set.json").exists()

    def test_temp_dirs_cleaned_by_default(self, tmp_path):
        pytest.importorskip("numpy")  # sweep builds vector indexes (needs [embedding])
        cdir = _materialize(tmp_path)
        res = _invoke(
            tmp_path, "--course", _SLUG, "--engines", "bm25,semantic", "--models", _MODELS
        )
        assert res.exit_code == 0, res.output
        # No bench-* temp dirs left behind; canonical absent (we never built it).
        assert not list(cdir.glob("vector_index.bench-*"))
        assert not (cdir / "vector_index").exists()
        assert not (cdir / "vector_index.canonical-stash").exists()

    def test_keep_leaves_temp_dirs(self, tmp_path):
        pytest.importorskip("numpy")  # sweep builds vector indexes (needs [embedding])
        cdir = _materialize(tmp_path)
        res = _invoke(
            tmp_path,
            "--course",
            _SLUG,
            "--engines",
            "bm25,semantic",
            "--models",
            _MODELS,
            "--keep",
        )
        assert res.exit_code == 0, res.output
        kept = sorted(p.name for p in cdir.glob("vector_index.bench-*"))
        assert kept == [
            "vector_index.bench-alpha-embed",
            "vector_index.bench-beta-embed",
        ]
        for d in kept:
            assert (cdir / d / "manifest.json").exists()

    def test_canonical_index_preserved_across_sweep(self, tmp_path):
        """A pre-existing canonical vector_index/ is restored byte-identically
        after the sweep (the sweep never clobbers the production index)."""
        pytest.importorskip("numpy")  # sweep builds vector indexes (needs [embedding])
        cdir = _materialize(tmp_path)
        # Build a canonical index first.
        res0 = _invoke(
            tmp_path, "--course", _SLUG, "--engines", "bm25,semantic", "--build-index"
        )
        assert res0.exit_code == 0, res0.output
        canonical = cdir / "vector_index"
        assert canonical.exists()
        before = (canonical / "manifest.json").read_bytes()
        emb_before = (canonical / "embeddings.npy").read_bytes()

        res = _invoke(
            tmp_path, "--course", _SLUG, "--engines", "bm25,semantic", "--models", _MODELS
        )
        assert res.exit_code == 0, res.output

        assert canonical.exists()
        assert (canonical / "manifest.json").read_bytes() == before
        assert (canonical / "embeddings.npy").read_bytes() == emb_before
        # No leftover stash / temp dirs.
        assert not (cdir / "vector_index.canonical-stash").exists()
        assert not list(cdir.glob("vector_index.bench-*"))

    def test_models_and_model_mutually_exclusive(self, tmp_path):
        _materialize(tmp_path)
        res = _invoke(
            tmp_path,
            "--course",
            _SLUG,
            "--engines",
            "bm25",
            "--models",
            _MODELS,
            "--model",
            "fakeorg/alpha-embed",
        )
        assert res.exit_code == 1
        assert "mutually exclusive" in res.output
