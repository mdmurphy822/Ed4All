"""Courseforge tier (outline / rewrite) graceful-stop ("checkpoint on command").

An operator ``ed4all stop`` (or a batch-timeout escalation) arms a filesystem
sentinel. The two-pass Courseforge tiers must, at the next unit boundary,
checkpoint every COMPLETED block and raise ``GracefulStopRequested`` — never
losing a completed authoring call, never leaving a half-written sidecar. The
mechanical guarantee asserted throughout: **sidecar records == provider calls**
(stop after call N → exactly N records, exactly N calls; resume → total-N
calls; pre-armed sentinel → 0 calls).

Four dispatch paths are exercised against the SAME per-block sidecars the resume
path replays (``test_tier_checkpoint_fingerprint.py``):

- **Outline** (``.blocks_outline_checkpoint.jsonl``) — sequential per-block loop,
  loop-top ``check_stop`` (three legs).
- **Rewrite sequential** (``.blocks_final_checkpoint.jsonl``) — loop-top
  ``check_stop`` (three legs).
- **Rewrite thread-pool** — futures pre-submitted, so ``_process_one_block``
  returns ``STOP_MARKER`` pre-dispatch (P3.0); the drain loop keeps
  checkpointing already-resolved (non-marker) futures, THEN raises. The
  ``sidecar records == provider calls`` invariant catches a regression where a
  raise inside the worker would skip those checkpoints.
- **Rewrite batched cloud lane** — a pre-flight ``check_stop`` immediately
  before ``route_rewrite_batch`` so a pre-armed sentinel never dispatches the
  batch (per-round hooks INSIDE the router are Wave C).

Hermetic: reuses the ``test_tier_checkpoint_fingerprint`` builders (fake
outline/rewrite providers, tmp project scaffold, LibV2 / capture pins,
``_keep_sidecars`` to simulate a mid-run kill). Sentinel isolation via the
``state_runs_isolated`` fixture (per-test ``ED4ALL_STATE_RUNS_DIR``) + a
synthetic ``ED4ALL_RUN_ID``.
"""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any, List, Optional

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402

# Reuse the fingerprint-test harness verbatim (fake providers + scaffolding).
from MCP.tools.tests.test_tier_checkpoint_fingerprint import (  # noqa: E402
    _FakeProvider,
    _keep_sidecars,
    _patch_router_with_fakes,
    _pin_hermetic_roots,
    _run_outline,
    _run_rewrite,
    _seed_project,
    _seed_validated_blocks,
)

_RUN_ID = "STOP_TIER_TESTRUN"


# --------------------------------------------------------------------------- #
# Arming providers (arm the REAL run-scoped sentinel after the Nth call)
# --------------------------------------------------------------------------- #
class _ArmAfterOutlineProvider(_FakeProvider):
    """Arms the real sentinel AFTER its ``arm_after``-th ``generate_outline``.

    ``route_with_self_consistency`` dispatches exactly one ``generate_outline``
    per block in the hermetic config (1:1 call↔block), so arming after call N
    → block N is authored + checkpointed, then the loop-top ``check_stop``
    raises before block N+1.
    """

    def __init__(self, *, arm_after: int) -> None:
        super().__init__()
        self._arm_after = arm_after

    def generate_outline(self, block, **kw):  # type: ignore[no-untyped-def]
        out = super().generate_outline(block, **kw)
        if len(self.outline_calls) == self._arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return out


class _ArmAfterRewriteProvider(_FakeProvider):
    """Arms the real sentinel AFTER its ``arm_after``-th ``generate_rewrite``."""

    def __init__(self, *, arm_after: int) -> None:
        super().__init__()
        self._arm_after = arm_after

    def generate_rewrite(self, block, **kw):  # type: ignore[no-untyped-def]
        out = super().generate_rewrite(block, **kw)
        if len(self.rewrite_calls) == self._arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return out


class _GatedStopProbe:
    """Deterministic in-flight ``stop_requested`` gate (monkeypatched onto pt).

    The first ``false_for`` probes return ``False`` (those workers dispatch);
    every later probe returns ``True`` (those workers get ``STOP_MARKER``).
    Thread-safe counter → the completed/marker SPLIT is exact regardless of
    pool interleaving, so ``sidecar records == provider calls`` is a
    deterministic assertion under the thread-pool path.
    """

    def __init__(self, false_for: int) -> None:
        self.false_for = false_for
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, run_id: Optional[str] = None) -> bool:
        with self._lock:
            self.calls += 1
            return self.calls > self.false_for


def _sidecar_records(path: Path) -> int:
    """Count checkpointed block entries (block_id-keyed reuse map)."""
    return len(pt._load_block_checkpoint(path))


# Provenance fields the tiers stamp with a fresh per-run value (wall-clock
# timestamp / decision-capture id). A freshly-authored block on the resume leg
# necessarily carries a DIFFERENT value from the uninterrupted oracle, so they
# are stripped before the content byte-equivalence comparison — exactly the
# ``touched_by[].timestamp`` normalization the fingerprint suite uses.
_VOLATILE_KEYS = frozenset(
    {"timestamp", "restamp", "decision_capture_id", "captured_at"}
)


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v)
            for k, v in obj.items()
            if k not in _VOLATILE_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _normalize_jsonl(raw: bytes) -> List[Any]:
    """Parse a blocks JSONL and drop volatile per-run provenance fields."""
    out: List[Any] = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            out.append(_strip_volatile(json.loads(line)))
    return out


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    """Per-test sentinel isolation: tmp state/runs + a synthetic run_id."""
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


# --------------------------------------------------------------------------- #
# Outline tier — three mechanical legs
# --------------------------------------------------------------------------- #
def test_outline_stop_after_n_exact(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_OL_STOP_N"
    project_path = _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    provider = _ArmAfterOutlineProvider(arm_after=3)
    _patch_router_with_fakes(monkeypatch, provider)
    _keep_sidecars(monkeypatch)

    with pytest.raises(GracefulStopRequested):
        _run_outline(project_id)

    # Block 4's loop-top check_stop raised: exactly 3 authored + on disk.
    assert len(provider.outline_calls) == 3
    sidecar = (
        project_path / "01_outline" / pt._OUTLINE_CHECKPOINT_NAME
    )
    assert _sidecar_records(sidecar) == 3


def test_outline_resume_after_stop_byte_equivalent(
    tmp_path, monkeypatch, _armed_env
):
    project_id = "TEST_OL_STOP_RESUME"
    project_path = _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    _keep_sidecars(monkeypatch)
    sidecar = project_path / "01_outline" / pt._OUTLINE_CHECKPOINT_NAME

    # Uninterrupted baseline in the SAME project (identical course name → no
    # course-name drift in the byte oracle), then wipe the emit + sidecar for
    # a clean interrupted-then-resume slate.
    base_provider = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, base_provider)
    base_payload = _run_outline(project_id)
    n_total = len(base_provider.outline_calls)
    assert n_total >= 4
    base_bytes = Path(base_payload["blocks_outline_path"]).read_bytes()
    Path(base_payload["blocks_outline_path"]).unlink()
    sidecar.unlink()

    # Interrupted leg: stop after 3 blocks.
    interrupted = _ArmAfterOutlineProvider(arm_after=3)
    _patch_router_with_fakes(monkeypatch, interrupted)
    with pytest.raises(GracefulStopRequested):
        _run_outline(project_id)
    assert len(interrupted.outline_calls) == 3
    assert _sidecar_records(sidecar) == 3

    # Resume leg: clear the sentinel, rerun against the SAME sidecar → only the
    # un-authored blocks dispatch; total across both legs == the baseline.
    stop_control.clear_stop(include_global=True)
    resume = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, resume)
    resume_payload = _run_outline(project_id)
    assert len(resume.outline_calls) == n_total - 3
    assert len(interrupted.outline_calls) + len(resume.outline_calls) == n_total
    # Content-equivalent to the uninterrupted run modulo per-run provenance
    # (timestamps / capture ids on the freshly-authored resume blocks).
    assert _normalize_jsonl(
        Path(resume_payload["blocks_outline_path"]).read_bytes()
    ) == _normalize_jsonl(base_bytes)


def test_outline_pre_armed_zero_calls(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_OL_STOP_PREARM"
    project_path = _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    provider = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, provider)
    _keep_sidecars(monkeypatch)
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        _run_outline(project_id)

    assert provider.outline_calls == []  # nothing dispatched
    sidecar = project_path / "01_outline" / pt._OUTLINE_CHECKPOINT_NAME
    assert not sidecar.exists()  # no sidecar written


# --------------------------------------------------------------------------- #
# Rewrite tier (sequential) — three mechanical legs
# --------------------------------------------------------------------------- #
def test_rewrite_seq_stop_after_n_exact(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_RW_STOP_N"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    provider = _ArmAfterRewriteProvider(arm_after=2)
    _patch_router_with_fakes(monkeypatch, provider)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path, n=4)

    with pytest.raises(GracefulStopRequested):
        _run_rewrite(project_id, validated_path)

    assert len(provider.rewrite_calls) == 2
    sidecar = _find_rewrite_sidecar(tmp_path, project_id)
    assert _sidecar_records(sidecar) == 2


def test_rewrite_seq_resume_after_stop_byte_equivalent(
    tmp_path, monkeypatch, _armed_env
):
    project_id = "TEST_RW_STOP_RESUME"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path, n=4)

    # Uninterrupted baseline in the SAME project for the byte oracle, then wipe
    # the emit + sidecar for a clean interrupted-then-resume slate.
    base_provider = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, base_provider)
    base_payload = _run_rewrite(project_id, validated_path)
    n_total = len(base_provider.rewrite_calls)
    assert n_total == 4
    base_bytes = Path(base_payload["blocks_final_path"]).read_bytes()
    Path(base_payload["blocks_final_path"]).unlink()
    _find_rewrite_sidecar(tmp_path, project_id).unlink()

    # Interrupted leg: stop after 2 blocks.
    interrupted = _ArmAfterRewriteProvider(arm_after=2)
    _patch_router_with_fakes(monkeypatch, interrupted)
    with pytest.raises(GracefulStopRequested):
        _run_rewrite(project_id, validated_path)
    assert len(interrupted.rewrite_calls) == 2
    sidecar = _find_rewrite_sidecar(tmp_path, project_id)
    assert _sidecar_records(sidecar) == 2

    # Resume leg: only the 2 un-rewritten blocks dispatch; total == 4.
    stop_control.clear_stop(include_global=True)
    resume = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, resume)
    resume_payload = _run_rewrite(project_id, validated_path)
    assert len(resume.rewrite_calls) == n_total - 2
    assert len(interrupted.rewrite_calls) + len(resume.rewrite_calls) == n_total
    # Content-equivalent modulo per-run provenance (timestamps / capture ids).
    assert _normalize_jsonl(
        Path(resume_payload["blocks_final_path"]).read_bytes()
    ) == _normalize_jsonl(base_bytes)


def test_rewrite_seq_pre_armed_zero_calls(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_RW_STOP_PREARM"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    provider = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, provider)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path, n=4)
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        _run_rewrite(project_id, validated_path)

    assert provider.rewrite_calls == []
    # No sidecar written (raise landed at the first loop boundary).
    export_dir = tmp_path / "Courseforge" / "exports" / project_id
    for cand in export_dir.rglob(pt._REWRITE_CHECKPOINT_NAME):
        assert _sidecar_records(cand) == 0


# --------------------------------------------------------------------------- #
# Rewrite tier (thread-pool) — P3.0 drain: resolved futures checkpointed first
# --------------------------------------------------------------------------- #
def test_rewrite_threadpool_completed_checkpointed_before_raise(
    tmp_path, monkeypatch, _armed_env
):
    """A stop armed mid-fanout must not lose the pool's COMPLETED blocks.

    All 4 blocks are submitted to the pool; the deterministic gate lets the
    first 2 workers dispatch (they complete + checkpoint) and refuses the rest
    with STOP_MARKER. If ``_process_one_block`` RAISED instead of returning the
    marker, ``future.result()`` would surface the raise and skip the drain-loop
    checkpoints of the 2 completed blocks. The invariant
    ``sidecar records == provider calls`` catches that regression.
    """
    project_id = "TEST_RW_STOP_POOL"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("COURSEFORGE_REWRITE_CONCURRENCY", "4")
    # Force the per-block thread-pool path (not the batched cloud lane).
    monkeypatch.setenv("COURSEFORGE_REWRITE_BATCH", "0")
    provider = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, provider)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path, n=4)

    # _process_one_block probes pt.stop_requested for the in-flight refusal;
    # the pre-submission check_stop uses the REAL (un-armed) sentinel, so the
    # raise comes from the post-drain marker sweep — never before the 2
    # completed workers are checkpointed.
    gate = _GatedStopProbe(false_for=2)
    monkeypatch.setattr(pt, "stop_requested", gate)

    with pytest.raises(GracefulStopRequested):
        _run_rewrite(project_id, validated_path)

    # Exactly the 2 gate-passed workers completed AND are on disk.
    assert len(provider.rewrite_calls) == 2
    sidecar = _find_rewrite_sidecar(tmp_path, project_id)
    assert _sidecar_records(sidecar) == len(provider.rewrite_calls) == 2


# --------------------------------------------------------------------------- #
# Rewrite tier (batched cloud lane) — pre-flight stop never dispatches the batch
# --------------------------------------------------------------------------- #
def test_rewrite_batched_pre_armed_never_dispatches(
    tmp_path, monkeypatch, _armed_env
):
    """Pre-armed sentinel → the pre-flight check_stop raises before the single
    ``route_rewrite_batch`` dispatch (nothing checkpointed; the batch call is
    never made)."""
    project_id = "TEST_RW_STOP_BATCH"
    _seed_project(tmp_path, project_id)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    provider = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, provider)
    _keep_sidecars(monkeypatch)
    validated_path = _seed_validated_blocks(tmp_path, n=4)

    # Force the batched cloud lane on: mode ON + probe reports a cloud lane.
    from Courseforge.generators import _rewrite_batch as _rb
    from Courseforge.router import router as _rmod

    monkeypatch.setattr(_rb, "resolve_rewrite_batch_mode", lambda: True)
    monkeypatch.setattr(
        _rmod.CourseforgeRouter,
        "_is_cloud_rewrite_lane",
        lambda self, spec: True,
    )

    batch_calls: list = []

    def _recording_batch(self, *a, **k):
        batch_calls.append((a, k))
        return {}

    monkeypatch.setattr(
        _rmod.CourseforgeRouter, "route_rewrite_batch", _recording_batch,
    )

    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        _run_rewrite(project_id, validated_path)

    # The single batched dispatch was never made.
    assert batch_calls == []
    assert provider.rewrite_calls == []


# --------------------------------------------------------------------------- #
# Helper: locate the rewrite sidecar under the export dir
# --------------------------------------------------------------------------- #
def _find_rewrite_sidecar(tmp_path: Path, project_id: str) -> Path:
    export_dir = tmp_path / "Courseforge" / "exports" / project_id
    matches = sorted(export_dir.rglob(pt._REWRITE_CHECKPOINT_NAME))
    assert matches, f"no rewrite sidecar under {export_dir}"
    return matches[0]
