"""Graceful-stop ("checkpoint on command") for two pt.py generation loops.

Covers the two P4 sidecars owned by the pt-C2 finisher:

- **Legacy single-pass content_generation** (P4 item 3) — the
  ``course_generation``-workflow ``_generate_course_content`` per-week emit loop
  (``.content_generation_weeks_checkpoint.jsonl``). A stop at the week-loop
  boundary checkpoints every completed week and raises ``GracefulStopRequested``;
  a resume skips the already-emitted weeks.
- **Dynamic block planner** (P4 item 4 / D5) — the outline tier's per-week
  ``plan_week_blocks`` dispatch (``.block_planner_weeks_checkpoint.jsonl``). A
  stop at the week-loop boundary checkpoints every planned week and raises; a
  resume reuses the planned weeks (zero re-dispatch).

Both assert the canonical invariant set (per-unit write, resume byte-equivalence,
fingerprint-mismatch re-run, torn-trailing-line tolerated, family-flag opt-out)
plus the three stop legs (stop-after-N → sidecar N + provider N; resume →
total-N; pre-armed → 0).

Hermetic: tmp project scaffold, no LLM / GPU / network. The per-week work unit
is a fake (``generate_week`` for the legacy path; a provider-less
``plan_week_blocks`` for the planner path) that arms the REAL run-scoped sentinel
after the Nth call. Sentinel isolation via the ``state_runs_isolated`` fixture
(per-test ``ED4ALL_STATE_RUNS_DIR``) + a synthetic ``ED4ALL_RUN_ID``.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import MCP.tools.pipeline_tools as pt  # noqa: E402
from lib.generation import stop_control  # noqa: E402
from lib.generation.block_planner import (  # noqa: E402
    plan_week_blocks as _REAL_PLAN_WEEK_BLOCKS,
)
from lib.generation.stop_control import GracefulStopRequested  # noqa: E402

# Reuse the outline harness verbatim (fake outline/rewrite providers + roots).
from MCP.tools.tests.test_tier_checkpoint_fingerprint import (  # noqa: E402
    _FakeProvider,
    _patch_router_with_fakes,
    _pin_hermetic_roots,
)

_RUN_ID = "STOP_LEGACY_GEN_TESTRUN"


@pytest.fixture
def _armed_env(state_runs_isolated, monkeypatch):
    """Per-test sentinel isolation: tmp state/runs + a synthetic run_id."""
    monkeypatch.setenv("ED4ALL_RUN_ID", _RUN_ID)
    stop_control.clear_stop(include_global=True)
    yield
    stop_control.clear_stop(include_global=True)


# --------------------------------------------------------------------------- #
# Multi-week project seed (shared)
# --------------------------------------------------------------------------- #
def _seed_multiweek_project(
    tmp_path: Path, project_id: str, *, weeks: int = 3, n_tos: int = 3,
) -> Path:
    project_path = tmp_path / "Courseforge" / "exports" / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    obj_path = (
        project_path / "01_learning_objectives" / "synthesized_objectives.json"
    )
    obj_path.write_text(
        json.dumps({
            "terminal_objectives": [
                {"id": f"TO-{i:02d}",
                 "statement": f"Describe core concept {i} in detail."}
                for i in range(1, n_tos + 1)
            ],
            "chapter_objectives": [],
        }),
        encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({
            "course_name": project_id,
            "duration_weeks": weeks,
            "objectives_path": str(obj_path),
        }),
        encoding="utf-8",
    )
    return project_path


# =========================================================================== #
# P4 item 3 — legacy single-pass content_generation per-week sidecar
# =========================================================================== #
def _fake_generate_week_factory(arm_after: int = 0):
    """Fake ``generate_week``: writes one non-trivial page per week; arms the
    real sentinel after ``arm_after`` calls (0 = never)."""
    state: Dict[str, List[int]] = {"weeks": []}

    def _fake(week_data, content_dir, course_code, **kw):
        wn = int(week_data["week_number"])
        state["weeks"].append(wn)
        week_dir = Path(content_dir) / f"week_{wn:02d}"
        week_dir.mkdir(parents=True, exist_ok=True)
        name = "overview.html"
        (week_dir / name).write_text(
            "<html><body><main><p>" + ("prose " * 40) + "</p></main></body>"
            "</html>",
            encoding="utf-8",
        )
        if arm_after and len(state["weeks"]) == arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return 1, [name]

    _fake.state = state  # type: ignore[attr-defined]
    return _fake


def _run_generate_course_content(project_id: str) -> Dict[str, Any]:
    registry = pt._build_tool_registry()
    fn = registry["generate_course_content"]
    return json.loads(asyncio.run(fn(project_id=project_id)))


def _cg_sidecar(project_path: Path) -> Path:
    return (
        project_path / "03_content_development"
        / pt._LEGACY_CONTENTGEN_CHECKPOINT_NAME
    )


def _patch_generate_week(monkeypatch, fake) -> None:
    from Courseforge.scripts import generate_course as _gen

    monkeypatch.setattr(_gen, "generate_week", fake)


def test_legacy_cg_stop_after_n_exact(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_CG_STOP_N"
    project_path = _seed_multiweek_project(tmp_path, project_id, weeks=3)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    fake = _fake_generate_week_factory(arm_after=2)
    _patch_generate_week(monkeypatch, fake)

    with pytest.raises(GracefulStopRequested):
        _run_generate_course_content(project_id)

    # Week 3's loop-top check_stop raised: exactly 2 weeks emitted + on disk.
    assert fake.state["weeks"] == [1, 2]
    store = pt._legacy_contentgen_store(_cg_sidecar(project_path))
    assert len(store.load()) == 2


def test_legacy_cg_resume_completes(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_CG_STOP_RESUME"
    project_path = _seed_multiweek_project(tmp_path, project_id, weeks=3)
    _pin_hermetic_roots(monkeypatch, tmp_path)

    interrupted = _fake_generate_week_factory(arm_after=2)
    _patch_generate_week(monkeypatch, interrupted)
    with pytest.raises(GracefulStopRequested):
        _run_generate_course_content(project_id)
    assert interrupted.state["weeks"] == [1, 2]
    assert len(pt._legacy_contentgen_store(_cg_sidecar(project_path)).load()) == 2

    # Resume: clear the sentinel; weeks 1+2 are cache hits (record + page files
    # present) → only week 3 re-emits. Total emitted across both legs == 3.
    stop_control.clear_stop(include_global=True)
    resume = _fake_generate_week_factory(arm_after=0)
    _patch_generate_week(monkeypatch, resume)
    payload = _run_generate_course_content(project_id)
    assert resume.state["weeks"] == [3]
    assert payload["success"] is True
    assert payload["weeks_prepared"] == 3
    # Sidecar removed on the successful final write.
    assert not _cg_sidecar(project_path).exists()


def test_legacy_cg_pre_armed_zero_calls(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_CG_STOP_PREARM"
    project_path = _seed_multiweek_project(tmp_path, project_id, weeks=3)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    fake = _fake_generate_week_factory(arm_after=0)
    _patch_generate_week(monkeypatch, fake)
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        _run_generate_course_content(project_id)

    assert fake.state["weeks"] == []                  # nothing emitted
    assert not _cg_sidecar(project_path).exists()     # no sidecar written


def test_legacy_cg_family_flag_off_disables_sidecar(
    tmp_path, monkeypatch, _armed_env
):
    project_id = "TEST_CG_STOP_FAMOFF"
    project_path = _seed_multiweek_project(tmp_path, project_id, weeks=3)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    fake = _fake_generate_week_factory(arm_after=2)
    _patch_generate_week(monkeypatch, fake)

    # The stop check is ALWAYS live (independent of the family flag), so a stop
    # armed after 2 weeks still raises — but no sidecar is written when off.
    with pytest.raises(GracefulStopRequested):
        _run_generate_course_content(project_id)
    assert fake.state["weeks"] == [1, 2]
    assert not _cg_sidecar(project_path).exists()


def test_legacy_cg_fingerprint_axis_sensitive():
    base = dict(
        week_num=1,
        week_objectives=[{"id": "TO-01", "statement": "Alpha."}],
        provider_spec="router=False|provider=|two_pass=False",
        duration_weeks=3,
        course_code="C",
    )
    fp = pt._legacy_contentgen_week_fingerprint(**base)
    # Changing any axis flips the fingerprint (stale record never spliced).
    assert fp != pt._legacy_contentgen_week_fingerprint(
        **{**base, "week_objectives": [{"id": "TO-01", "statement": "Beta."}]}
    )
    assert fp != pt._legacy_contentgen_week_fingerprint(
        **{**base, "provider_spec": "router=True|provider=local|two_pass=True"}
    )
    assert fp != pt._legacy_contentgen_week_fingerprint(
        **{**base, "duration_weeks": 4}
    )
    # Same inputs → same fingerprint (deterministic).
    assert fp == pt._legacy_contentgen_week_fingerprint(**base)


def test_legacy_cg_torn_trailing_line_tolerated(tmp_path):
    cp = tmp_path / pt._LEGACY_CONTENTGEN_CHECKPOINT_NAME
    store = pt._legacy_contentgen_store(cp)
    store.append("w1", "fp1", {"page_paths": ["a.html"]})
    # Simulate a crash mid-append: a torn (unparseable) trailing line.
    with cp.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "unit_id": "w2", "fingerp')
    loaded = store.load()
    assert set(loaded) == {"w1"}          # torn line skipped, w1 survives


# =========================================================================== #
# P4 item 4 — dynamic block planner per-week sidecar
# =========================================================================== #
def _fake_plan_week_blocks_factory(arm_after: int = 0):
    """Fake ``plan_week_blocks``: delegates to the REAL provider-less
    deterministic fixed-plan path (no LLM) and arms the sentinel after N.

    Uses the module-level ``_REAL_PLAN_WEEK_BLOCKS`` captured at import time —
    NOT a fresh ``from ... import`` — so a fake built while a PRIOR fake is
    still patched onto the module does not recursively wrap it."""
    state: Dict[str, int] = {"calls": 0}

    def _fake(**kwargs):
        state["calls"] += 1
        kwargs["provider"] = None  # deterministic fixed fallback plan (no LLM)
        plan = _REAL_PLAN_WEEK_BLOCKS(**kwargs)
        if arm_after and state["calls"] == arm_after:
            stop_control.request_stop(scope="run", reason="test", source="test")
        return plan

    _fake.state = state  # type: ignore[attr-defined]
    return _fake


class _DummyPlannerProvider:
    _model = "fake-planner-model"


def _patch_dynamic_planner(monkeypatch, fake) -> None:
    import lib.generation.block_planner as _bp

    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    monkeypatch.setattr(
        pt, "_build_block_planner_provider",
        lambda capture=None: _DummyPlannerProvider(),
    )
    monkeypatch.setattr(_bp, "plan_week_blocks", fake)


def _run_outline(project_id: str) -> Dict[str, Any]:
    return json.loads(asyncio.run(
        pt._run_content_generation_outline(project_id=project_id),
    ))


def _planner_sidecar(project_path: Path) -> Path:
    return project_path / "01_outline" / pt._BLOCK_PLANNER_CHECKPOINT_NAME


def test_planner_stop_after_n_exact(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_PLAN_STOP_N"
    project_path = _seed_multiweek_project(tmp_path, project_id, weeks=3)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    _patch_router_with_fakes(monkeypatch, _FakeProvider())
    fake = _fake_plan_week_blocks_factory(arm_after=2)
    _patch_dynamic_planner(monkeypatch, fake)

    with pytest.raises(GracefulStopRequested):
        _run_outline(project_id)

    # Week 3's loop-top check_stop raised: exactly 2 weeks planned + on disk.
    assert fake.state["calls"] == 2
    store = pt._block_planner_store(_planner_sidecar(project_path))
    assert len(store.load()) == 2


def test_planner_resume_byte_equivalent(tmp_path, monkeypatch, _armed_env):
    # Uninterrupted oracle in its OWN project (dynamic planner on).
    oracle_id = "TEST_PLAN_ORACLE"
    oracle_path = _seed_multiweek_project(tmp_path, oracle_id, weeks=3)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    _patch_router_with_fakes(monkeypatch, _FakeProvider())
    _patch_dynamic_planner(monkeypatch, _fake_plan_week_blocks_factory(0))
    oracle_payload = _run_outline(oracle_id)
    oracle_bytes = Path(oracle_payload["blocks_outline_path"]).read_bytes()

    # Interrupted leg: stop after planning 2 of 3 weeks (no blocks authored yet).
    project_id = "TEST_PLAN_RESUME"
    project_path = _seed_multiweek_project(tmp_path, project_id, weeks=3)
    interrupted = _fake_plan_week_blocks_factory(arm_after=2)
    _patch_dynamic_planner(monkeypatch, interrupted)
    with pytest.raises(GracefulStopRequested):
        _run_outline(project_id)
    assert interrupted.state["calls"] == 2
    assert len(pt._block_planner_store(_planner_sidecar(project_path)).load()) == 2

    # Resume: clear the sentinel; weeks 1+2 plans are reused (0 re-dispatch),
    # only week 3 re-plans. Then all blocks author → blocks_outline.jsonl.
    stop_control.clear_stop(include_global=True)
    resume = _fake_plan_week_blocks_factory(arm_after=0)
    _patch_dynamic_planner(monkeypatch, resume)
    resume_payload = _run_outline(project_id)
    assert resume.state["calls"] == 1                          # only week 3
    assert interrupted.state["calls"] + resume.state["calls"] == 3
    # Byte-equivalent authored outline to the uninterrupted oracle.
    assert _normalize_jsonl(
        Path(resume_payload["blocks_outline_path"]).read_bytes()
    ) == _normalize_jsonl(oracle_bytes)
    # Planner sidecar removed on the successful final write.
    assert not _planner_sidecar(project_path).exists()


def test_planner_pre_armed_zero_calls(tmp_path, monkeypatch, _armed_env):
    project_id = "TEST_PLAN_PREARM"
    project_path = _seed_multiweek_project(tmp_path, project_id, weeks=3)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    _patch_router_with_fakes(monkeypatch, _FakeProvider())
    fake = _fake_plan_week_blocks_factory(arm_after=0)
    _patch_dynamic_planner(monkeypatch, fake)
    stop_control.request_stop(scope="run", reason="test", source="test")

    with pytest.raises(GracefulStopRequested):
        _run_outline(project_id)

    assert fake.state["calls"] == 0                    # nothing planned
    assert not _planner_sidecar(project_path).exists()  # no sidecar written


def test_planner_family_flag_off_disables_sidecar(
    tmp_path, monkeypatch, _armed_env
):
    project_id = "TEST_PLAN_FAMOFF"
    project_path = _seed_multiweek_project(tmp_path, project_id, weeks=3)
    _pin_hermetic_roots(monkeypatch, tmp_path)
    _patch_router_with_fakes(monkeypatch, _FakeProvider())
    monkeypatch.setenv("ED4ALL_GENERATION_CHECKPOINT", "0")
    fake = _fake_plan_week_blocks_factory(arm_after=2)
    _patch_dynamic_planner(monkeypatch, fake)

    # Stop check stays live; sidecar stays empty when the family flag is off.
    with pytest.raises(GracefulStopRequested):
        _run_outline(project_id)
    assert fake.state["calls"] == 2
    assert not _planner_sidecar(project_path).exists()


def test_planner_fingerprint_axis_sensitive():
    base = dict(
        week_num=1,
        terminal_objective={"id": "TO-01", "statement": "Alpha."},
        chapter_objectives=[{"id": "CO-01", "statement": "A.",
                             "bloom_level": "apply"}],
        source_chunks=[{"id": "c1", "text": "body", "heading": "H"}],
        model="m1",
        budget=(3, 8),
        course_code="C",
    )
    fp = pt._block_planner_week_fingerprint(**base)
    assert fp != pt._block_planner_week_fingerprint(
        **{**base, "terminal_objective": {"id": "TO-01", "statement": "Beta."}}
    )
    assert fp != pt._block_planner_week_fingerprint(**{**base, "model": "m2"})
    assert fp != pt._block_planner_week_fingerprint(
        **{**base, "source_chunks": [{"id": "c1", "text": "OTHER",
                                      "heading": "H"}]}
    )
    assert fp == pt._block_planner_week_fingerprint(**base)


def test_planner_serialize_round_trip():
    from lib.generation.block_planner import plan_week_blocks

    plan = plan_week_blocks(
        terminal_objective={"id": "TO-01", "statement": "Alpha."},
        chapter_objectives=[],
        source_chunks=[],
        provider=None,
    )
    payload = pt._serialize_week_block_plan(plan)
    # JSON round-trip (the sidecar stores JSON) then reconstruct.
    payload = json.loads(json.dumps(payload))
    restored = pt._deserialize_week_block_plan(payload)
    assert restored.fallback_used == plan.fallback_used
    assert restored.terminal_objective_id == plan.terminal_objective_id
    # page_plan tuples survive the list<->tuple round-trip identically.
    for _pt_key, entries in plan.page_plan.items():
        assert [tuple(e) for e in restored.page_plan[_pt_key]] == [
            tuple(e) for e in entries
        ]


def test_planner_torn_trailing_line_tolerated(tmp_path):
    cp = tmp_path / pt._BLOCK_PLANNER_CHECKPOINT_NAME
    store = pt._block_planner_store(cp)
    store.append("w1", "fp1", {"page_plan": {}, "selected": [],
                               "fallback_used": True})
    with cp.open("a", encoding="utf-8") as fh:
        fh.write('{"schema_version": "v1", "unit_id": "w2", "fingerp')
    loaded = store.load()
    assert set(loaded) == {"w1"}


# --------------------------------------------------------------------------- #
# Volatile-field normalization for the byte-equivalence oracle (mirrors
# test_tier_stop._strip_volatile).
# --------------------------------------------------------------------------- #
_VOLATILE_KEYS = frozenset(
    {"timestamp", "restamp", "decision_capture_id", "captured_at"}
)


def _strip_volatile(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items()
                if k not in _VOLATILE_KEYS}
    if isinstance(obj, list):
        return [_strip_volatile(v) for v in obj]
    return obj


def _normalize_jsonl(raw: bytes) -> List[Any]:
    out: List[Any] = []
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            out.append(_strip_volatile(json.loads(line)))
    return out
