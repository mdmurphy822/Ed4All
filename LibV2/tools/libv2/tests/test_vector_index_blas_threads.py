"""BLAS thread-team cap around the search GEMV (ED4ALL_RETRIEVAL_BLAS_THREADS).

At the production query cadence — roughly one GEMV per 50 ms — the BLAS thread
team is COLD on every query, and its wakeup cost dominates the arithmetic on a
many-core host whose cores are not equal. Capping the team to a handful of
threads for the duration of one ``matrix @ query`` is a large, free latency
win: the limiter cannot change a single result bit.

That "cannot change a result" property is what these tests pin, alongside the
resolver's parse-with-fallback contract and the fact that the cap is scoped to
the GEMV rather than leaking process-wide.

Hermetic + CPU-only: the limiter itself is exercised through a recording
double substituted for ``threadpoolctl.threadpool_limits``, so the assertions
hold identically on a host where threadpoolctl is not installed.
"""

from __future__ import annotations

import contextlib

np = __import__("pytest").importorskip(
    "numpy", reason="[embedding] extras absent — vector-index tests need numpy"
)
import pytest

from LibV2.tools.libv2 import vector_index as vi
from LibV2.tools.libv2.vector_index import (
    VectorIndex,
    VectorIndexManifest,
    resolve_retrieval_blas_threads,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _index(n: int = 32, dim: int = 8) -> VectorIndex:
    rng = np.random.default_rng(4242)
    raw = rng.standard_normal((n, dim)).astype(np.float32)
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    manifest = VectorIndexManifest(
        schema_version="1.0",
        embedding_provider="fake",
        embedding_kind="fake",
        embedding_model_id="fake-deterministic-v1",
        embedding_model_revision=None,
        embedding_dim=dim,
        normalized=True,
        index_type="exact-numpy",
        chunker_version="v4",
        chunkset_kind="semantik",
        source_chunks_sha256="0" * 64,
        embeddings_sha256="0" * 64,
        id_map_sha256="0" * 64,
        chunks_count=n,
        text_field_policy="text+heading",
        batch_size=16,
        device="cpu",
    )
    ids = [f"chunk_{i:04d}" for i in range(n)]
    return VectorIndex(manifest=manifest, chunk_ids=ids, matrix=raw)


def _query(dim: int = 8) -> np.ndarray:
    q = np.random.default_rng(99).standard_normal(dim).astype(np.float32)
    return q / np.linalg.norm(q)


class _RecordingLimiter:
    """Stand-in for ``threadpoolctl.threadpool_limits``.

    Records every ``(limits, user_api)`` it is entered with, and asserts the
    context is properly exited — a limiter that leaks past the GEMV would
    throttle the rest of the process.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []
        self.open_depth = 0
        self.max_depth = 0

    def __call__(self, *, limits, user_api):
        self.calls.append((limits, user_api))
        return self._scope()

    @contextlib.contextmanager
    def _scope(self):
        self.open_depth += 1
        self.max_depth = max(self.max_depth, self.open_depth)
        try:
            yield
        finally:
            self.open_depth -= 1


@pytest.fixture()
def limiter(monkeypatch) -> _RecordingLimiter:
    rec = _RecordingLimiter()
    monkeypatch.setattr(vi, "_threadpool_limits", rec)
    return rec


# --------------------------------------------------------------------------
# Resolver — parse-with-fallback
# --------------------------------------------------------------------------


def test_default_is_eight(monkeypatch):
    monkeypatch.delenv("ED4ALL_RETRIEVAL_BLAS_THREADS", raising=False)
    assert resolve_retrieval_blas_threads() == 8


@pytest.mark.parametrize("raw,expected", [("1", 1), ("4", 4), (" 16 ", 16)])
def test_explicit_values(monkeypatch, raw, expected):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", raw)
    assert resolve_retrieval_blas_threads() == expected


@pytest.mark.parametrize("raw", ["", "   ", "eight", "4.5", "None"])
def test_garbage_falls_back_to_default_never_raises(monkeypatch, raw):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", raw)
    assert resolve_retrieval_blas_threads() == 8


@pytest.mark.parametrize("raw", ["0", "-1"])
def test_non_positive_is_the_documented_disable(monkeypatch, raw):
    """Non-positive is a deliberate 'let BLAS pick', not a parse failure —
    so it must survive the resolver rather than snapping back to 8."""
    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", raw)
    assert resolve_retrieval_blas_threads() <= 0


def test_resolver_accepts_an_injected_env_mapping():
    assert resolve_retrieval_blas_threads({"ED4ALL_RETRIEVAL_BLAS_THREADS": "3"}) == 3
    assert resolve_retrieval_blas_threads({}) == 8


# --------------------------------------------------------------------------
# Wiring — the limiter is entered, with the resolved value, around the GEMV
# --------------------------------------------------------------------------


def test_search_enters_limiter_with_resolved_value(monkeypatch, limiter):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", "5")
    _index().search(_query(), top_k=3)
    assert limiter.calls == [(5, "blas")]


def test_search_enters_limiter_with_the_default(monkeypatch, limiter):
    monkeypatch.delenv("ED4ALL_RETRIEVAL_BLAS_THREADS", raising=False)
    _index().search(_query(), top_k=3)
    assert limiter.calls == [(8, "blas")]


def test_limiter_scope_is_exited_and_never_nested(monkeypatch, limiter):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", "8")
    index = _index()
    for _ in range(3):
        index.search(_query(), top_k=2)
    assert limiter.open_depth == 0
    assert limiter.max_depth == 1
    assert len(limiter.calls) == 3


def test_zero_disables_the_limiter_entirely(monkeypatch, limiter):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", "0")
    _index().search(_query(), top_k=3)
    assert limiter.calls == []


def test_empty_result_short_circuit_never_enters_the_limiter(monkeypatch, limiter):
    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", "8")
    assert _index().search(_query(), top_k=0) == []
    assert limiter.calls == []


# --------------------------------------------------------------------------
# The load-bearing property: results are byte-identical either way
# --------------------------------------------------------------------------


@pytest.mark.parametrize("threads", ["0", "1", "4", "8", "64"])
def test_results_are_identical_at_every_thread_cap(monkeypatch, threads):
    """A tuning knob may never change an answer. Run against the REAL
    limiter (or its real absence) rather than the recording double, so this
    covers the shipped code path end to end."""
    index = _index(n=257, dim=16)
    q = _query(16)

    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", "8")
    baseline = index.search(q, top_k=25)

    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", threads)
    assert index.search(q, top_k=25) == baseline


def test_absent_threadpoolctl_degrades_to_no_limiter_only(monkeypatch):
    """A missing optional dependency costs latency, never correctness — and
    it must not raise. Warned at DEBUG once per process, not per query."""
    index = _index(n=48, dim=8)
    q = _query()

    monkeypatch.setattr(vi, "_threadpool_limits", None)
    monkeypatch.setattr(vi, "_THREADPOOLCTL_WARNED", False)
    monkeypatch.setenv("ED4ALL_RETRIEVAL_BLAS_THREADS", "8")

    first = index.search(q, top_k=5)
    assert vi._THREADPOOLCTL_WARNED is True
    assert index.search(q, top_k=5) == first
