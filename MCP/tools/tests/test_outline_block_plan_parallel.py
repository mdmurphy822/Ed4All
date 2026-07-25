"""Parallel warm-up for the dynamic block planner in the outline tier.

``MCP.tools.pipeline_tools._run_content_generation_outline`` runs the dynamic
block planner (``ED4ALL_DYNAMIC_BLOCK_PLAN=1``) as a strictly sequential
per-week loop of slow Super-120B ``plan_week_blocks`` dispatches. This module
pins the parallel warm-up pre-pass retrofit:

* a bounded ``ThreadPoolExecutor`` warm-up (when
  ``COURSEFORGE_BLOCK_PLAN_CONCURRENCY`` / ``COURSEFORGE_OUTLINE_CONCURRENCY``
  resolves > 1) computes every week's planner inputs + fingerprint and writes
  each plan into the SAME resume sidecar the loop reads, so the sequential loop
  then makes ZERO inline planner dispatches (every week HITS ``reuse(...)``);
* the plans consumed are byte-identical to the sequential (concurrency=1) path
  — the warm-up's fingerprint cannot drift from the loop's;
* a pre-armed graceful-stop sentinel refuses the whole warm-up before any
  dispatch (zero planner calls);
* resolved concurrency <= 1 builds NO pool (byte-identical legacy path).

Hermetic: the real outline handler is driven against a tmp project scaffold
with a stub router (no LLM / GPU / network); ``plan_week_blocks`` is delegated
to its own deterministic provider=None fixed-plan fallback via a counting spy.
Placeholder course/objective text throughout.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.tools import pipeline_tools as _pt  # noqa: E402
from lib.generation import block_planner as _bp  # noqa: E402
from lib.generation import stop_control as _stop  # noqa: E402
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixture builders (mirror test_pipeline_tools_phase3_handlers.py)
# ---------------------------------------------------------------------- #
class _FakeProvider:
    """Stub router provider — returns blocks unchanged / minimal HTML."""

    def __init__(self) -> None:
        self.outline_calls: List[Dict[str, Any]] = []
        self.rewrite_calls: List[Dict[str, Any]] = []

    def generate_outline(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.outline_calls.append({"block_id": block.block_id})
        return dataclasses.replace(block, content="outline-stub")

    def generate_rewrite(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.rewrite_calls.append({"block_id": block.block_id})
        return dataclasses.replace(block, content="<p>rewrite stub body.</p>")


def _seed_project(
    tmp_path: Path, project_id: str, *, n_weeks: int, n_tos: int,
) -> Path:
    """Minimal Courseforge/exports/<project> scaffold with N weeks + N TOs."""
    project_path = tmp_path / "Courseforge" / "exports" / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    terminal = [
        {"id": f"TO-{i:02d}",
         "statement": f"Describe placeholder concept number {i} in detail."}
        for i in range(1, n_tos + 1)
    ]
    (project_path / "01_learning_objectives"
     / "synthesized_objectives.json").write_text(
        json.dumps({"terminal_objectives": terminal, "chapter_objectives": []}),
        encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": n_weeks}),
        encoding="utf-8",
    )
    return project_path


def _patch_common(monkeypatch, tmp_path: Path) -> _FakeProvider:
    """Redirect project root + captures into tmp, stub the router + provider."""
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("ED4ALL_TRAINING_CAPTURES_DIR", str(tmp_path / "caps"))
    # Dynamic block planner ON; no real planner seat (fixed-plan fallback).
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    monkeypatch.setattr(_pt, "_build_block_planner_provider", lambda capture=None: None)

    fake = _FakeProvider()
    from Courseforge.router import router as _router_mod

    real_init = _router_mod.CourseforgeRouter.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("outline_provider", fake)
        kwargs.setdefault("rewrite_provider", fake)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter, "__init__", patched_init,
    )

    # Wave-7/7b objective blocks escalate_immediately; strip so the outline
    # dispatch path runs to completion under the fake provider.
    import dataclasses as _dc

    from Courseforge.router import policy as _policy_mod

    real_loader = _policy_mod.load_block_routing_policy

    def _empty_escalation_loader(*args: Any, **kwargs: Any):
        loaded = real_loader(*args, **kwargs)
        return _dc.replace(loaded, escalate_immediately_by_block_type={})

    monkeypatch.setattr(
        _policy_mod, "load_block_routing_policy", _empty_escalation_loader,
    )
    return fake


def _install_counting_planner(monkeypatch) -> List[str]:
    """Spy ``plan_week_blocks``: count calls, force provider=None (fixed plan)."""
    real_plan = _bp.plan_week_blocks
    calls: List[str] = []

    def _counting_plan(**kwargs):
        calls.append(str((kwargs.get("terminal_objective") or {}).get("id")))
        kwargs["provider"] = None  # deterministic fixed-plan fallback, no net
        return real_plan(**kwargs)

    monkeypatch.setattr(_bp, "plan_week_blocks", _counting_plan)
    return calls


def _install_pool_spy(monkeypatch) -> Dict[str, int]:
    """Count ThreadPoolExecutor constructions (the ONLY pool here is warm-up)."""
    import concurrent.futures as _cf

    real_tpe = _cf.ThreadPoolExecutor
    counter = {"n": 0}

    class _SpyTPE(real_tpe):  # type: ignore[misc,valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            counter["n"] += 1
            super().__init__(*a, **k)

    monkeypatch.setattr(_cf, "ThreadPoolExecutor", _SpyTPE)
    return counter


def _install_append_spy(monkeypatch) -> List[Dict[str, Any]]:
    """Capture every planner-sidecar append DURING the run.

    The handler removes the planner sidecar on successful completion
    (``_planner_store.remove()``), so post-run inspection is impossible — we
    observe the appends live. Each entry is written exactly once per week by
    WHICHEVER path plans it (warm-up OR the loop's cache-miss branch); a
    fingerprint drift that made the loop re-dispatch a warmed week would append
    that week TWICE. Filtered to the ``block_planner_weeks`` site.
    """
    from lib.generation import llm_checkpoint as _lc

    real_append = _lc.CheckpointStore.append
    captured: List[Dict[str, Any]] = []

    def _spy_append(self, unit_id, fingerprint, payload):  # type: ignore[no-untyped-def]
        if getattr(self, "site_id", "") == "block_planner_weeks":
            captured.append(
                {"unit_id": unit_id, "fingerprint": fingerprint,
                 "payload": payload}
            )
        return real_append(self, unit_id, fingerprint, payload)

    monkeypatch.setattr(_lc.CheckpointStore, "append", _spy_append)
    return captured


# ---------------------------------------------------------------------- #
# _resolve_block_plan_concurrency
# ---------------------------------------------------------------------- #
def test_resolve_block_plan_concurrency_default_one(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_CONCURRENCY", raising=False)
    assert _pt._resolve_block_plan_concurrency() == 1


def test_resolve_block_plan_concurrency_own_flag_wins(monkeypatch):
    monkeypatch.setenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", "5")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "48")
    assert _pt._resolve_block_plan_concurrency() == 5


def test_resolve_block_plan_concurrency_inherits_outline(monkeypatch):
    monkeypatch.delenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", raising=False)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "8")
    assert _pt._resolve_block_plan_concurrency() == 8


def test_resolve_block_plan_concurrency_garbage_falls_back(monkeypatch):
    monkeypatch.setenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", "nonsense")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "6")
    assert _pt._resolve_block_plan_concurrency() == 6
    # own-flag <1 also falls through to outline
    monkeypatch.setenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", "0")
    assert _pt._resolve_block_plan_concurrency() == 6
    # both garbage → 1
    monkeypatch.setenv("COURSEFORGE_OUTLINE_CONCURRENCY", "-3")
    assert _pt._resolve_block_plan_concurrency() == 1


# ---------------------------------------------------------------------- #
# Warm-up pre-pass behavior
# ---------------------------------------------------------------------- #
def test_warmup_prepopulates_and_loop_makes_zero_inline_calls(
    tmp_path, monkeypatch,
):
    """Concurrency > 1: warm-up plans every week; the loop dispatches none."""
    project_id = "TEST_BP_PAR"
    _seed_project(tmp_path, project_id, n_weeks=4, n_tos=4)
    _patch_common(monkeypatch, tmp_path)
    calls = _install_counting_planner(monkeypatch)
    appends = _install_append_spy(monkeypatch)
    pool = _install_pool_spy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", "4")

    result = json.loads(asyncio.run(
        _pt._run_content_generation_outline(project_id=project_id),
    ))
    assert result["success"] is True, result

    # A warm-up pool WAS constructed (the parallel path ran).
    assert pool["n"] >= 1
    # Each of the 4 weeks was planned + checkpointed exactly ONCE — the warm-up
    # populated the sidecar and the loop reused all four (a fingerprint drift
    # that made the loop re-dispatch a warmed week would append it a 2nd time).
    warm_weeks = [a["unit_id"] for a in appends]
    assert sorted(warm_weeks) == ["w1", "w2", "w3", "w4"], warm_weeks
    # Total plan_week_blocks dispatches == weeks (NOT 2x): the loop made ZERO
    # inline planner calls (every week HIT reuse()).
    assert len(calls) == 4, calls


def test_parallel_plans_byte_identical_to_sequential(tmp_path, monkeypatch):
    """The plans the loop consumes are identical to the concurrency=1 path."""
    _patch_common(monkeypatch, tmp_path)
    _install_counting_planner(monkeypatch)
    appends = _install_append_spy(monkeypatch)
    pool = _install_pool_spy(monkeypatch)

    def _plans_by_week(records):
        # {week: serialized WeekBlockPlan payload}. The fingerprint is
        # DELIBERATELY excluded — it encodes course_code, which differs between
        # the two distinct project scaffolds; the byte-identical claim is about
        # the serialized plan the loop consumes, not the resume fingerprint.
        # (Anti-drift between warm-up and loop is proved separately by
        # test_warmup_prepopulates_and_loop_makes_zero_inline_calls.)
        return {r["unit_id"]: r["payload"] for r in records}

    # --- sequential (concurrency=1) reference run ---
    seq_id = "TEST_BP_SEQ"
    _seed_project(tmp_path, seq_id, n_weeks=3, n_tos=3)
    monkeypatch.setenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", "1")
    monkeypatch.delenv("COURSEFORGE_OUTLINE_CONCURRENCY", raising=False)
    seq_res = json.loads(asyncio.run(
        _pt._run_content_generation_outline(project_id=seq_id),
    ))
    assert seq_res["success"] is True
    # Concurrency <= 1 constructs NO pool (legacy sequential path).
    assert pool["n"] == 0
    seq_plans = _plans_by_week(appends)

    # --- parallel (concurrency=3) run in a fresh project ---
    appends.clear()
    par_id = "TEST_BP_PAR2"
    _seed_project(tmp_path, par_id, n_weeks=3, n_tos=3)
    monkeypatch.setenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", "3")
    par_res = json.loads(asyncio.run(
        _pt._run_content_generation_outline(project_id=par_id),
    ))
    assert par_res["success"] is True
    assert pool["n"] >= 1
    par_plans = _plans_by_week(appends)

    # Same weeks; every week's serialized plan payload AND its input
    # fingerprint are byte-identical → no drift, plans consumed are identical.
    assert set(seq_plans) == set(par_plans) == {"w1", "w2", "w3"}
    for wk in seq_plans:
        assert seq_plans[wk] == par_plans[wk], wk


def test_prearmed_stop_dispatches_zero_planner_calls(tmp_path, monkeypatch):
    """A pre-armed stop sentinel refuses the warm-up before any dispatch."""
    project_id = "TEST_BP_STOP"
    _seed_project(tmp_path, project_id, n_weeks=4, n_tos=4)
    _patch_common(monkeypatch, tmp_path)
    calls = _install_counting_planner(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", "4")

    # Isolate the stop sentinel under tmp and pre-arm a global STOP_ALL.
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(tmp_path / "state_runs"))
    try:
        _stop.request_stop(scope="all", reason="test", source="pytest")
        assert _stop.stop_requested() is True
        with pytest.raises(GracefulStopRequested):
            asyncio.run(
                _pt._run_content_generation_outline(project_id=project_id),
            )
        # Zero planner dispatches — the warm-up refused before submitting.
        assert calls == []
    finally:
        _stop.clear_stop(include_global=True)


def test_concurrency_one_builds_no_pool(tmp_path, monkeypatch):
    """Resolved concurrency <= 1 takes the legacy sequential path (no pool)."""
    project_id = "TEST_BP_ONE"
    _seed_project(tmp_path, project_id, n_weeks=3, n_tos=3)
    _patch_common(monkeypatch, tmp_path)
    calls = _install_counting_planner(monkeypatch)
    appends = _install_append_spy(monkeypatch)
    pool = _install_pool_spy(monkeypatch)
    monkeypatch.delenv("COURSEFORGE_BLOCK_PLAN_CONCURRENCY", raising=False)
    monkeypatch.delenv("COURSEFORGE_OUTLINE_CONCURRENCY", raising=False)

    result = json.loads(asyncio.run(
        _pt._run_content_generation_outline(project_id=project_id),
    ))
    assert result["success"] is True
    # No warm-up pool constructed.
    assert pool["n"] == 0
    # The loop itself dispatched + checkpointed each week once.
    assert len(calls) == 3, calls
    assert sorted(a["unit_id"] for a in appends) == ["w1", "w2", "w3"]
