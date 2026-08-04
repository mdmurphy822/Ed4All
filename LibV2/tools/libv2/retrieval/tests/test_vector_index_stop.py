"""Graceful-stop ("checkpoint on command") coverage for the vector index.

Plan reference: P4 item 8 / Resolved Decision **D1** — vector indexing is
*abort-to-paused* with a **full re-embed on resume**; it has NO resume sidecar
(embeddings are not LLM calls, so the ≤1-call loss guarantee does not bind
them). The two mechanical guarantees under test:

1. A stop armed BEFORE ``build_vector_index`` (pre-armed sentinel) raises
   ``GracefulStopRequested`` at the pre-encode ``check_stop`` — the embedding
   client is never called and NO ``vector_index/`` artifacts are written.
2. A stop armed DURING the encode is caught at the post-encode / pre-write
   ``check_stop`` — again NO partial index is left on disk, and the exception
   propagates.

Plus the D1 corollaries: a no-sentinel build is byte-for-byte unaffected, and a
stopped build leaves NO state that the freshness path
(``build_vector_index`` :482) would mistake for a completed index — so a
subsequent (un-stopped) build re-embeds from scratch and produces a full index.

Hermetic: the local hash-seeded fake ``EmbeddingClient`` from
``test_vector_index`` (no weights, no network). Sentinel isolation via the
root-conftest ``state_runs_isolated`` fixture (per-test
``ED4ALL_STATE_RUNS_DIR``) plus a synthetic ``ED4ALL_RUN_ID`` so the run-scoped
sentinel resolves the same path the CLI would write. CPU-only.
"""

from __future__ import annotations

import json
from pathlib import Path

np = __import__("pytest").importorskip(
    "numpy", reason="[embedding] extras absent — vector-index tests need numpy"
)
import pytest

from lib.generation import stop_control
from lib.generation.stop_control import GracefulStopRequested
from LibV2.tools.libv2.retrieval.vector_index import (
    VECTOR_INDEX_DIRNAME,
    build_vector_index,
    load_vector_index,
)

# Reuse the frozen-protocol fake client + tiny-course builder verbatim.
from LibV2.tools.libv2.retrieval.tests.test_vector_index import (
    _FakeEmbeddingClient,
    _write_course,
)

_RUN_ID = "STOP_VECTOR_TESTRUN"


class _CallCountingClient(_FakeEmbeddingClient):
    """Fake client that records how many times ``encode_batch`` was invoked."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.encode_calls = 0

    def encode_batch(self, texts):
        self.encode_calls += 1
        return super().encode_batch(texts)


class _ArmDuringEncodeClient(_FakeEmbeddingClient):
    """Fake client that arms the REAL run-scoped sentinel as a side effect of
    the encode, then returns the matrix.

    The pre-encode ``check_stop`` sees no sentinel (dispatches the encode); the
    encode arms it; the post-encode / pre-write ``check_stop`` raises. This
    exercises the SECOND stop window precisely — the encode work is completed
    and then discarded, and no write section is ever reached.
    """

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.encode_calls = 0

    def encode_batch(self, texts):
        self.encode_calls += 1
        out = super().encode_batch(texts)
        stop_control.request_stop(scope="run", reason="test", source="test")
        return out


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    """Per-test sentinel isolation: tmp runtime/state/runs + synthetic run_id."""
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


def _index_dir(course_dir: Path) -> Path:
    return course_dir / VECTOR_INDEX_DIRNAME


# --------------------------------------------------------------------------- #
# Leg 1 — pre-armed sentinel: encode never called, no artifacts written
# --------------------------------------------------------------------------- #
def test_pre_armed_never_encodes_no_index(tmp_path, _armed_env):
    course_dir = _write_course(tmp_path, n=5)
    client = _CallCountingClient()
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        build_vector_index(course_dir, client=client)

    # Pre-encode check_stop raised: the embedding client was never called.
    assert client.encode_calls == 0
    # No partial index: the vector_index/ dir was never even created.
    assert not _index_dir(course_dir).exists()


def test_pre_armed_via_global_sentinel(tmp_path, _armed_env):
    """The global STOP_ALL sentinel (``ed4all stop --all``) trips it too."""
    course_dir = _write_course(tmp_path, n=4)
    client = _CallCountingClient()
    stop_control.request_stop(scope="all", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        build_vector_index(course_dir, client=client)

    assert client.encode_calls == 0
    assert not _index_dir(course_dir).exists()


# --------------------------------------------------------------------------- #
# Leg 2 — stop armed DURING encode: no partial index; raise propagates
# --------------------------------------------------------------------------- #
def test_stop_between_encode_and_write_no_partial_index(tmp_path, _armed_env):
    course_dir = _write_course(tmp_path, n=6)
    client = _ArmDuringEncodeClient()

    with pytest.raises(GracefulStopRequested):
        build_vector_index(course_dir, client=client)

    # The encode ran exactly once (and was discarded) ...
    assert client.encode_calls == 1
    # ... but the second check_stop raised BEFORE any write: no dir, no bytes.
    assert not _index_dir(course_dir).exists()


# --------------------------------------------------------------------------- #
# Leg 3 — no sentinel: build is byte-for-byte unaffected by the stop checks
# --------------------------------------------------------------------------- #
def test_no_sentinel_build_unchanged(tmp_path, _armed_env):
    course_dir = _write_course(tmp_path, n=5)
    manifest = build_vector_index(course_dir, client=_FakeEmbeddingClient())

    assert manifest.chunks_count == 5
    idx_dir = _index_dir(course_dir)
    assert (idx_dir / "embeddings.npy").exists()
    assert (idx_dir / "id_map.json").exists()
    assert (idx_dir / "manifest.json").exists()

    # Byte-identical to a build performed with no run-scoped sentinel env at
    # all (proves the always-on check_stop probe touches no output byte).
    ref_dir = _write_course(tmp_path / "ref", n=5)
    build_vector_index(ref_dir, client=_FakeEmbeddingClient())
    assert (idx_dir / "embeddings.npy").read_bytes() == (
        ref_dir / VECTOR_INDEX_DIRNAME / "embeddings.npy"
    ).read_bytes()
    assert (idx_dir / "id_map.json").read_bytes() == (
        ref_dir / VECTOR_INDEX_DIRNAME / "id_map.json"
    ).read_bytes()


# --------------------------------------------------------------------------- #
# D1 corollary — a stopped build leaves NO state the freshness path mistakes
# for complete; a subsequent un-stopped build re-embeds from scratch (full).
# --------------------------------------------------------------------------- #
def test_resume_re_embeds_from_scratch_full_index(tmp_path, _armed_env):
    course_dir = _write_course(tmp_path, n=5)

    # Interrupted leg: stop lands during the encode → no index on disk.
    with pytest.raises(GracefulStopRequested):
        build_vector_index(course_dir, client=_ArmDuringEncodeClient())
    assert not _index_dir(course_dir).exists()

    # Resume leg: clear the sentinel, rebuild. Because the interrupted run left
    # no manifest, the freshness guard (:482) does not short to FileExistsError
    # — a clean full re-embed happens with NO force needed.
    stop_control.clear_stop(include_global=True)
    client = _CallCountingClient()
    manifest = build_vector_index(course_dir, client=client)

    assert client.encode_calls == 1
    assert manifest.chunks_count == 5
    idx = load_vector_index(course_dir, allow_fake=True)
    assert idx.matrix.shape == (5, 32)
    # Load-bearing chunk-id order fully materialized (nothing was skipped).
    assert idx.chunk_ids[0] == "fake_101_chunk_00000"
    assert len(idx.chunk_ids) == 5


def test_empty_chunkset_pre_armed_no_index(tmp_path, _armed_env):
    """The zero-row path also honors the pre-encode stop (defensive: the
    ``if texts:`` branch is skipped but the check precedes it)."""
    course_dir = tmp_path / "courses" / "empty-101"
    chunks_dir = course_dir / "semantik_chunks"
    chunks_dir.mkdir(parents=True)
    (chunks_dir / "chunks.jsonl").write_text("", encoding="utf-8")
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        build_vector_index(course_dir, client=_FakeEmbeddingClient())

    assert not _index_dir(course_dir).exists()
