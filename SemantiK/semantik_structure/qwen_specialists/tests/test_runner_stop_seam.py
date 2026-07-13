"""Graceful-stop SEAM (c) — Stage-6 adapter-batch boundary in the runner.

CPU-pinned, no GPU / no GGUF / no endpoint (a MockRuntime spy records
generate_batch calls). Proves the between-adapter-batch cooperative stop:

  * pre-armed sentinel  -> CascadeStopRequested at the FIRST boundary, ZERO
    generate_batch calls (never enters authoring);
  * armed mid-Stage-6   -> stop caught BETWEEN adapter groups (the first group
    generated, later groups did not);
  * no sentinel wired    -> byte-identical: every adapter group generates.

Mirrors the ``test_runner_batched.py`` harness (``_region`` duck type +
MockRuntime subclass).
"""

from __future__ import annotations

import types

import pytest

from semantik_structure import stop_seam
from semantik_structure.qwen_specialists import runner as runner_mod
from semantik_structure.qwen_specialists.runner import run_qwen_specialists
from semantik_structure.qwen_specialists.runtime import MockRuntime
from semantik_structure.stop_seam import STOP_SENTINEL_ENV, CascadeStopRequested, StopPoller


def _region(kind: str, text: str = "x"):
    return types.SimpleNamespace(
        kind=kind,
        payload={"text": text},
        aria_hints=(),
        feature_block_indices=(),
    )


class _BatchSpyRuntime(MockRuntime):
    """Records each generate_batch call; optionally arms a sentinel on first."""

    def __init__(self, arm_path=None) -> None:
        super().__init__()
        self.batch_calls: list[int] = []
        self._arm_path = arm_path

    def generate_batch(self, prompts, *, max_tokens, **kw):  # type: ignore[override]
        self.batch_calls.append(len(prompts))
        if self._arm_path is not None and not self._arm_path.exists():
            # Arm the stop sentinel AFTER the first adapter group generates.
            self._arm_path.write_text("{}", encoding="utf-8")
        return super().generate_batch(prompts, max_tokens=max_tokens, **kw)


def _unthrottled_poller(monkeypatch):
    """Force the runner's StopPoller to a 0s interval so a mid-run arm is
    observed at the very next boundary (defeats the 2s production throttle)."""
    monkeypatch.setattr(
        runner_mod, "StopPoller", lambda: StopPoller(min_interval_s=0.0)
    )


# --------------------------------------------------------------------------- #
def test_prearmed_sentinel_stops_before_any_generation(monkeypatch, tmp_path):
    p = tmp_path / "STOP_REQUESTED"
    p.write_text("{}", encoding="utf-8")
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)

    regions = [_region("paragraph", "a"), _region("math", "b")]
    rt = _BatchSpyRuntime()
    with pytest.raises(CascadeStopRequested) as ei:
        run_qwen_specialists(regions, [], k=2, runtime=rt)
    assert ei.value.site_id.startswith("stage6:phase1:")
    # Stopped at the first adapter boundary — nothing was ever generated.
    assert rt.batch_calls == []


def test_stop_caught_between_adapter_groups(monkeypatch, tmp_path):
    _unthrottled_poller(monkeypatch)
    p = tmp_path / "STOP_REQUESTED"  # armed by the spy after the PROSE group
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(p))
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)

    # PROSE first (generates + arms), then MATH must be skipped by the stop.
    regions = [_region("paragraph", "a"), _region("math", "b")]
    rt = _BatchSpyRuntime(arm_path=p)
    with pytest.raises(CascadeStopRequested):
        run_qwen_specialists(regions, [], k=2, runtime=rt)
    # The PROSE group generated (k=2 slots -> 2 calls); MATH never did.
    assert rt.batch_calls == [1, 1]  # exactly the PROSE group's two K-slots


def test_no_sentinel_is_byte_identical(monkeypatch, tmp_path):
    monkeypatch.delenv(STOP_SENTINEL_ENV, raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)

    regions = [_region("paragraph", "a"), _region("math", "b")]
    rt = _BatchSpyRuntime()
    out = run_qwen_specialists(regions, [], k=2, runtime=rt)
    # Both adapter groups authored: PROSE k-slots + MATH k-slots == 4 calls.
    assert len(rt.batch_calls) == 4
    assert set(out.keys()) == {0, 1}
    for cands in out.values():
        assert len(cands) == 2


def test_env_set_but_absent_file_is_noop(monkeypatch, tmp_path):
    # SEMANTIK_STOP_SENTINEL points at a not-yet-written file -> full run.
    monkeypatch.setenv(STOP_SENTINEL_ENV, str(tmp_path / "STOP_REQUESTED"))
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)

    regions = [_region("paragraph", "a"), _region("math", "b")]
    rt = _BatchSpyRuntime()
    out = run_qwen_specialists(regions, [], k=2, runtime=rt)
    assert len(rt.batch_calls) == 4
    assert set(out.keys()) == {0, 1}


# --------------------------------------------------------------------------- #
# #40 — WITHIN-group pause + resume (per-prompt sidecar).
#
# A runtime whose generate_batch persists each completed prompt via
# on_item_done then raises CascadeStopRequested mid-group (mirrors
# LlamaCppRuntime's per-prompt stop probe). With the resume sidecar ON, the
# resume run regenerates ONLY the un-persisted prompts.
# --------------------------------------------------------------------------- #


class _StopAfterNSpy(MockRuntime):
    """Persists each completed prompt (on_item_done) then raises
    CascadeStopRequested once ``stop_after`` prompts have completed."""

    def __init__(self, stop_after=10**9) -> None:
        super().__init__()
        self.stop_after = stop_after
        self.generated: list[str] = []

    def generate_batch(self, prompts, *, max_tokens, on_item_done=None, **kw):  # type: ignore[override]
        out: list[str] = []
        for i, p in enumerate(prompts):
            if i >= self.stop_after:
                raise CascadeStopRequested("stage6:phase1:gguf-batch")
            res = self.generate(p, n=1, temperature=0.6, top_p=0.95, max_tokens=max_tokens)
            text = res[0]
            out.append(text)
            self.generated.append(p)
            if on_item_done is not None:
                on_item_done(i, text)
        return out


def test_within_group_pause_and_resume_regenerates_only_unpersisted(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("SEMANTIK_SPECIALIST_PROVIDER", raising=False)
    monkeypatch.delenv("SEMANTIK_SPECIALIST_REFINE", raising=False)
    monkeypatch.delenv("ED4ALL_GENERATION_CHECKPOINT", raising=False)
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("SEMANTIK_STAGE6_CHECKPOINT", "1")

    # 3 prose regions -> one PROSE adapter group of 3 prompts, k=1.
    regions = [_region("paragraph", c) for c in ("a", "b", "c")]

    # Run 1 — stop AFTER prompt 0 completes (persisted), raise on prompt 1.
    rt1 = _StopAfterNSpy(stop_after=1)
    with pytest.raises(CascadeStopRequested):
        run_qwen_specialists(regions, [], k=1, runtime=rt1, runtime_mode="real")
    # Exactly ONE prompt completed + persisted before the pause.
    assert len(rt1.generated) == 1
    cache_root = tmp_path / "stage6_candidate_cache"
    assert len(list(cache_root.rglob("*.json"))) == 1

    # Run 2 — resume, no stop: prompt 0 HITS the sidecar, only 2 regenerate.
    rt2 = _StopAfterNSpy(stop_after=10**9)
    out = run_qwen_specialists(regions, [], k=1, runtime=rt2, runtime_mode="real")
    assert len(rt2.generated) == 2  # only the two un-persisted prompts
    assert set(out.keys()) == {0, 1, 2}  # full document assembled on resume
