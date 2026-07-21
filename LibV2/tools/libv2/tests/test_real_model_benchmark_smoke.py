"""WS2 Wave C — opt-in real-model end-to-end ``retrieval-benchmark`` smoke.

Builds a real sentence-transformers vector index over the WS1 mini-course
fixture with the cached ``all-MiniLM-L6-v2`` (Apache-2.0, dim 384) and runs
``libv2 retrieval-benchmark`` end-to-end across bm25 + semantic + hybrid-rrf.

Marked ``@pytest.mark.real_models`` (registered in this package's conftest);
``pytest.skip`` when the model is absent from the HF cache so CI without a
warmed cache stays green. No network: the build runs offline
(``local_files_only`` via the provider's offline path).
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
_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _model_cached() -> bool:
    try:
        from huggingface_hub import scan_cache_dir

        return any(_MODEL in r.repo_id for r in scan_cache_dir().repos)
    except Exception:  # noqa: BLE001 — no cache / no hub => treat as uncached
        return False


def _materialize(repo_root: Path) -> Path:
    cdir = repo_root / "courses" / _SLUG
    (cdir / "semantik_chunks").mkdir(parents=True)
    (cdir / "retrieval_eval").mkdir(parents=True)
    shutil.copy(
        _FIXTURE / "semantik_chunks" / "chunks.jsonl",
        cdir / "semantik_chunks" / "chunks.jsonl",
    )
    shutil.copy(
        _FIXTURE / "retrieval_eval" / "gold_set.json",
        cdir / "retrieval_eval" / "gold_set.json",
    )
    (cdir / "manifest.json").write_text(
        json.dumps({"classification": {"primary_domain": "rag"}})
    )
    return cdir


@pytest.mark.real_models
def test_real_minilm_retrieval_benchmark_end_to_end(tmp_path, monkeypatch):
    if not (_FIXTURE / "semantik_chunks" / "chunks.jsonl").exists():
        pytest.skip("mini_course fixture not present")
    if not _model_cached():
        pytest.skip(f"{_MODEL} not in HF cache; skipping real-model smoke")

    # Real ST provider, offline (cached weights only). The build command in
    # retrieval-benchmark uses the resolved provider; we pin it explicitly so
    # the smoke is hermetic regardless of the ambient env.
    monkeypatch.setenv("ED4ALL_EMBEDDING_PROVIDER", "st")
    monkeypatch.setenv("ED4ALL_EMBEDDING_MODEL", _MODEL)
    monkeypatch.setenv("ED4ALL_EMBEDDING_DEVICE", "cpu")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.delenv("ED4ALL_EMBEDDING_ALLOW_FAKE", raising=False)

    cdir = _materialize(tmp_path)
    out_path = tmp_path / "real_bench.json"

    res = CliRunner().invoke(
        libv2_main,
        [
            "--repo",
            str(tmp_path),
            "retrieval-benchmark",
            "--course",
            _SLUG,
            "--engines",
            "bm25,semantic,hybrid-rrf",
            "--build-index",
            "--provider",
            "st",
            "--model",
            _MODEL,
            "--out",
            str(out_path),
        ],
    )
    assert res.exit_code == 0, res.output
    assert out_path.exists()

    # Real index manifest: MiniLM is 384-dim.
    manifest = json.loads(
        (cdir / "vector_index" / "manifest.json").read_text()
    )
    assert manifest["embedding_dim"] == 384
    assert manifest["embedding_model_id"] == _MODEL

    report = json.loads(out_path.read_text())
    # All three arms ran; semantic returned real nearest-neighbour results.
    for engine in ("bm25", "semantic", "hybrid-rrf"):
        assert engine in report["per_engine"]
    sem_rows = [r["engines"]["semantic"]["retrieved_chunk_ids"]
                for r in report["per_query"]]
    assert all(len(ids) > 0 for ids in sem_rows)
    # Real embeddings should land the primary chunk for at least one question
    # in the top-k (sanity that the index isn't degenerate).
    assert report["per_engine"]["semantic"]["recall_at_k_primary"]["10"] > 0.0
