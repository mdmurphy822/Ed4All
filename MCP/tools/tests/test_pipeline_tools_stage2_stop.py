"""Stage-2 window-synthesis graceful-stop ("checkpoint on command") invariants.

An operator ``ed4all stop`` (or a batch-timeout escalation) arms a filesystem
sentinel. The ``course_planning`` Stage-2 per-window loop must, on the next
unit boundary, checkpoint every COMPLETED window and raise
``GracefulStopRequested`` — never losing a completed 7B call, never leaving a
half-written sidecar. Worst-case loss is ONE in-flight window.

Two stop mechanisms are exercised, both against the SAME per-window sidecar the
resume path replays (``test_pipeline_tools_stage2_checkpoint.py``):

- **Between batches** (real sentinel): ``check_stop`` at the batch-loop top
  raises after the prior batch's appends — the common, fully-checkpointed case.
- **In flight** (P3.0 marker pattern): a stop armed WHILE a batch is dispatched
  makes ``_one_window`` RETURN ``STOP_MARKER`` (never raise, which would
  propagate through ``asyncio.gather`` before the post-gather append loop and
  lose the batch's completed windows). The completed windows ARE appended, the
  markers are skipped, and only THEN is ``GracefulStopRequested`` raised. This
  is the blocker-#1 regression: ``sidecar records == provider calls``.

Hermetic: a mock provider (no GPU / ollama / network), ``build_embedding_client``
forced unavailable, ``ground_candidates`` monkeypatched to a passthrough (no NLI
model). Sentinel isolation via the ``state_runs_isolated`` fixture (per-test
``ED4ALL_STATE_RUNS_DIR``) + a synthetic ``ED4ALL_RUN_ID``.
"""
from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402

_RUN_ID = "STOP_STAGE2_TESTRUN"


class _ProviderError(RuntimeError):
    pass


def _mint(kind: str, idx: int) -> str:
    prefix = {"terminal": "TO", "chapter": "CO"}.get(kind, "XX")
    return f"{prefix}-{idx:02d}"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _WindowProvider:
    """Records every window call; returns one candidate objective per window."""

    def __init__(self, *, model: str = "test-model-v1") -> None:
        self._model = model
        self._max_tokens = 0  # shrink the fixed reserve → 1 chunk per window
        self.window_calls: List[str] = []

    def batch_chapters(self, items, batch_size: int = 10):
        return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

    def synthesize_window_objectives(
        self, window, *, course_name, draft_terminal_objectives
    ):
        cid = str(window.get("chapter_id"))
        widx = window.get("window_index")
        self.window_calls.append(f"{cid}#w{widx}")
        return {
            "candidate_objectives": [
                {
                    "statement": f"Objective for {cid} window {widx}.",
                    "source_chunk_ids": list(window.get("chunk_ids") or []),
                }
            ]
        }

    def reconcile_terminal_objectives(self, draft_tos, chapter_cos, *, course_name):
        return {
            "terminal_objectives": [{"id": "TO-01", "statement": "Reconciled TO."}]
        }


class _ArmAfterProvider(_WindowProvider):
    """Arms the REAL run-scoped stop sentinel AFTER its ``arm_after``-th call.

    With ``batch_size=1`` this drives the BETWEEN-batches path: call N
    completes + is appended, then the next batch-loop-top ``check_stop`` raises.
    """

    def __init__(self, *, arm_after: int, batch_size: int = 1) -> None:
        super().__init__()
        self._arm_after = arm_after
        self._batch_size = batch_size

    def batch_chapters(self, items, batch_size: int = 10):
        return super().batch_chapters(items, batch_size=self._batch_size)

    def synthesize_window_objectives(self, window, **kw):
        out = super().synthesize_window_objectives(window, **kw)
        if len(self.window_calls) == self._arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return out


class _GatedStopProbe:
    """Deterministic in-flight ``stop_requested`` gate (monkeypatched onto pt).

    The first ``false_for`` probes return ``False`` (those windows dispatch);
    every later probe returns ``True`` (those windows get ``STOP_MARKER``).
    Thread-safe counter → the completed/marker SPLIT is exact regardless of
    executor thread interleaving, so ``sidecar records == provider calls`` is a
    deterministic assertion.
    """

    def __init__(self, false_for: int) -> None:
        self.false_for = false_for
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, run_id: Optional[str] = None) -> bool:
        with self._lock:
            self.calls += 1
            return self.calls > self.false_for


class _PassthroughGrounding:
    def __init__(self, grounded: List[Dict[str, Any]]) -> None:
        self.grounded = grounded
        self.ungrounded: List[Dict[str, Any]] = []
        self.available = False
        self.dropped_count = 0
        self.reground_count = 0


def _chapters() -> List[Dict[str, Any]]:
    return [
        {"id": "ch1", "chapter_text": "xxxxxxxxxx"},
        {"id": "ch2", "chapter_text": "yyyyyyyyyy"},
    ]


def _chunks() -> List[Dict[str, Any]]:
    return [
        {"id": "c1", "text": "Alpha body."},
        {"id": "c2", "text": "Bravo body."},
        {"id": "c3", "text": "Charlie body."},
        {"id": "c4", "text": "Delta body."},
    ]


def _run(provider, *, checkpoint_path: Path, monkeypatch):
    """Drive _run_stage2_window_synthesis in chunk_window mode, hermetically."""
    import lib.embedding.providers as _providers
    import lib.objectives.objective_grounding as _og

    def _fake_build(*_a, **_k):
        raise _providers.EmbeddingBackendUnavailable("no extras in test")

    monkeypatch.setattr(_providers, "build_embedding_client", _fake_build)
    monkeypatch.setattr(
        _og,
        "ground_candidates",
        lambda cands, chunks_by_id, require=False: _PassthroughGrounding(
            list(cands)
        ),
    )
    monkeypatch.setenv("TEXTBOOK_SYNTHESIS_NUM_CTX", "80")

    all_chunks = _chunks()
    chunks_by_id = {c["id"]: c for c in all_chunks}
    return asyncio.run(
        pt._run_stage2_window_synthesis(
            provider=provider,
            provider_error=_ProviderError,
            chapters=_chapters(),
            draft_tos=[{"statement": "Draft course objective."}],
            chunks_by_id=chunks_by_id,
            all_chunks=all_chunks,
            grounding_mode="chunk_window",
            course_name="MATH_101",
            provider_env="local",
            chapter_synthesis_failures=[],
            mint_lo_id=_mint,
            kwargs={},
            capture=None,
            checkpoint_path=checkpoint_path,
        )
    )


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    """Per-test sentinel isolation: tmp state/runs + a synthetic run_id."""
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


# ---------------------------------------------------------------------------
# (1) between-batches: stop after call N → exactly N calls + N sidecar records
# ---------------------------------------------------------------------------
def test_stop_between_batches_exact_n(tmp_path, monkeypatch, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    provider = _ArmAfterProvider(arm_after=2, batch_size=1)

    with pytest.raises(GracefulStopRequested):
        _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    # Exactly 2 windows dispatched (batch 3's loop-top check_stop raised) and
    # both are on disk — the mechanical guarantee.
    assert len(provider.window_calls) == 2
    records = pt._load_stage2_windows_checkpoint(cp)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# (2) resume after a stop: only the un-attempted windows re-run; byte-equivalent
# ---------------------------------------------------------------------------
def test_resume_after_stop_byte_equivalent(tmp_path, monkeypatch, _armed_env):
    # Uninterrupted baseline (its OWN sidecar) for the byte-equivalence oracle.
    base_cp = tmp_path / "base" / ".stage2_windows_checkpoint.jsonl"
    res_full = _run(_WindowProvider(), checkpoint_path=base_cp,
                    monkeypatch=monkeypatch)

    # Interrupted leg: stop after 2 of 4 windows.
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    interrupted = _ArmAfterProvider(arm_after=2, batch_size=1)
    with pytest.raises(GracefulStopRequested):
        _run(interrupted, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(interrupted.window_calls) == 2
    assert len(pt._load_stage2_windows_checkpoint(cp)) == 2

    # Resume leg: clear the sentinel, rerun against the SAME sidecar → only the
    # 2 un-attempted windows dispatch; total across both legs == 4.
    stop_control.clear_stop(include_global=True)
    resume = _WindowProvider()
    res_resume = _run(resume, checkpoint_path=cp, monkeypatch=monkeypatch)
    assert len(resume.window_calls) == 2
    assert len(interrupted.window_calls) + len(resume.window_calls) == 4

    # Byte-equivalent to the uninterrupted run modulo the reuse counters.
    assert res_resume[:3] == res_full[:3]
    fresh_sig, resume_sig = dict(res_full[3]), dict(res_resume[3])
    for k in ("windows_reused_from_checkpoint",):
        fresh_sig.pop(k, None)
        resume_sig.pop(k, None)
    assert fresh_sig == resume_sig
    assert res_full[3]["windows_reused_from_checkpoint"] == 0
    assert res_resume[3]["windows_reused_from_checkpoint"] == 2
    assert res_resume[3]["windows_total"] == 4


# ---------------------------------------------------------------------------
# (3) pre-armed sentinel → zero provider calls
# ---------------------------------------------------------------------------
def test_pre_armed_sentinel_zero_calls(tmp_path, monkeypatch, _armed_env):
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    stop_control.request_stop(scope="run", reason="test", source="test")

    provider = _WindowProvider()
    with pytest.raises(GracefulStopRequested):
        _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    assert provider.window_calls == []  # nothing dispatched
    assert not cp.exists()  # no sidecar written


# ---------------------------------------------------------------------------
# (4) P3.0 marker pattern (blocker-#1): completed in-flight windows ARE persisted
# ---------------------------------------------------------------------------
def test_marker_pattern_in_flight_completed_persisted(
    tmp_path, monkeypatch, _armed_env
):
    """A stop armed mid-batch must not lose the batch's COMPLETED windows.

    All 4 windows are dispatched in ONE batch; the deterministic gate lets the
    first 2 through (they complete + append) and refuses the rest with
    STOP_MARKER. If _one_window RAISED instead of returning the marker, gather
    would propagate before the append loop and lose the 2 completed windows.
    The invariant ``sidecar records == provider calls`` catches that regression.
    """
    cp = tmp_path / "01_learning_objectives" / ".stage2_windows_checkpoint.jsonl"
    gate = _GatedStopProbe(false_for=2)
    # _one_window probes pt.stop_requested (the in-flight refusal); the batch
    # loop-top check_stop uses the REAL (un-armed) sentinel, so the raise comes
    # from the post-gather marker sweep.
    monkeypatch.setattr(pt, "stop_requested", gate)

    provider = _WindowProvider()
    with pytest.raises(GracefulStopRequested):
        _run(provider, checkpoint_path=cp, monkeypatch=monkeypatch)

    # Exactly the 2 gate-passed windows completed AND are on disk.
    assert len(provider.window_calls) == 2
    records = pt._load_stage2_windows_checkpoint(cp)
    assert len(records) == len(provider.window_calls) == 2


# ---------------------------------------------------------------------------
# Stage-2 bottom-up TO cluster loop (sequential — plain check_stop at loop top)
# ---------------------------------------------------------------------------
class _FakeEmbed:
    """encode_batch → 4 orthogonal pairs so target-K=4 yields 4 clusters of 2."""

    def encode_batch(self, statements):
        import numpy as np

        basis = np.eye(4, dtype=float)
        return np.array([basis[i // 2] for i in range(len(statements))])


class _ClusterProvider:
    """Records author calls; arms the REAL sentinel after ``arm_after`` calls."""

    def __init__(self, *, arm_after: int = 0) -> None:
        self._model = "test-model-v1"
        self._arm_after = arm_after
        self.author_calls: List[int] = []

    def author_terminal_for_cluster(self, cluster_cos, *, course_name, cluster_index):
        self.author_calls.append(cluster_index)
        if self._arm_after and len(self.author_calls) == self._arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return {"statement": f"TO for cluster {cluster_index}."}


def _cos(n: int = 8) -> List[Dict[str, Any]]:
    return [
        {"statement": f"CO statement {i}.", "source_chunk_ids": [f"c{i}"]}
        for i in range(n)
    ]


def _derive(provider, *, cluster_cp: Path, monkeypatch):
    monkeypatch.setenv("ED4ALL_TO_CLUSTER_K", "4")  # fixed K=4 (existing flag)
    return pt._derive_terminals_bottom_up(
        provider=provider,
        chapter_cos=_cos(8),
        embed=_FakeEmbed(),
        course_name="MATH_101",
        mint_lo_id=_mint,
        checkpoint_path=cluster_cp,
    )


def test_cluster_stop_between_iterations_exact_n(tmp_path, monkeypatch, _armed_env):
    cluster_cp = (
        tmp_path / "01_learning_objectives" / ".stage2_clusters_checkpoint.jsonl"
    )
    provider = _ClusterProvider(arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _derive(provider, cluster_cp=cluster_cp, monkeypatch=monkeypatch)

    # Cluster 3's loop-top check_stop raised: exactly 2 authored + on disk.
    assert len(provider.author_calls) == 2
    store = pt._stage2_cluster_store(cluster_cp)
    assert len(store.load()) == 2


def test_cluster_resume_completes(tmp_path, monkeypatch, _armed_env):
    cluster_cp = (
        tmp_path / "01_learning_objectives" / ".stage2_clusters_checkpoint.jsonl"
    )
    interrupted = _ClusterProvider(arm_after=2)
    with pytest.raises(GracefulStopRequested):
        _derive(interrupted, cluster_cp=cluster_cp, monkeypatch=monkeypatch)
    assert len(interrupted.author_calls) == 2

    stop_control.clear_stop(include_global=True)
    resume = _ClusterProvider()  # never arms
    terminals, _sig = _derive(resume, cluster_cp=cluster_cp, monkeypatch=monkeypatch)
    # Only the 2 un-authored clusters re-run; 4 terminals emitted overall.
    assert len(resume.author_calls) == 2
    assert len(interrupted.author_calls) + len(resume.author_calls) == 4
    assert len(terminals) == 4


def test_cluster_pre_armed_zero_calls(tmp_path, monkeypatch, _armed_env):
    cluster_cp = (
        tmp_path / "01_learning_objectives" / ".stage2_clusters_checkpoint.jsonl"
    )
    stop_control.request_stop(scope="run", reason="test", source="test")
    provider = _ClusterProvider()
    with pytest.raises(GracefulStopRequested):
        _derive(provider, cluster_cp=cluster_cp, monkeypatch=monkeypatch)
    assert provider.author_calls == []
