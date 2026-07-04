"""Trap 2 — --stop-after persistence + resume-precedence tests.

Hermetic: drives the param-persistence + resume-resolution logic
directly (no real workflow run).

Precedence on ``--resume``:
  * an explicit ``--stop-after`` on the resume command wins;
  * else the value persisted at workflow creation applies;
  * else none (run to completion).
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest
from unittest.mock import AsyncMock, MagicMock

from cli.commands.run import (
    _apply_resume_stop_after_override,
    _resolve_resume_stop_after,
)


# --------------------------------------------------------------------------
# Pure precedence resolver
# --------------------------------------------------------------------------
def test_resolve_explicit_wins():
    assert _resolve_resume_stop_after("course_planning", "packaging") == "course_planning"


def test_resolve_persisted_applies_when_no_explicit():
    assert _resolve_resume_stop_after(None, "imscc_chunking") == "imscc_chunking"


def test_resolve_none_when_neither():
    assert _resolve_resume_stop_after(None, None) is None


def test_resolve_empty_explicit_falls_to_persisted():
    # An empty-string explicit is treated as unset.
    assert _resolve_resume_stop_after("", "packaging") == "packaging"


# --------------------------------------------------------------------------
# State-file override (drives the persistence + resolution IO directly)
# --------------------------------------------------------------------------
def _write_state(tmp_path, monkeypatch, params: dict) -> str:
    import lib.paths as paths_mod

    monkeypatch.setattr(paths_mod, "STATE_PATH", tmp_path / "state")
    workflow_id = "WF-RESUME-TEST"
    wf_dir = tmp_path / "state" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps({"workflow_id": workflow_id, "params": params})
    )
    return workflow_id


def _read_persisted_stop_after(tmp_path, workflow_id: str):
    state = json.loads(
        (tmp_path / "state" / "workflows" / f"{workflow_id}.json").read_text()
    )
    return state["params"].get("stop_after")


def test_override_explicit_patches_persisted(tmp_path, monkeypatch):
    """Explicit --stop-after on resume overrides the persisted value."""
    wid = _write_state(tmp_path, monkeypatch, {"stop_after": "packaging"})
    effective = _apply_resume_stop_after_override(wid, "course_planning")
    assert effective == "course_planning"
    assert _read_persisted_stop_after(tmp_path, wid) == "course_planning"


def test_override_absent_uses_persisted_and_no_write(tmp_path, monkeypatch):
    """No explicit -> persisted value governs; the file is untouched."""
    wid = _write_state(tmp_path, monkeypatch, {"stop_after": "imscc_chunking"})
    effective = _apply_resume_stop_after_override(wid, None)
    assert effective == "imscc_chunking"
    assert _read_persisted_stop_after(tmp_path, wid) == "imscc_chunking"


def test_override_explicit_sets_when_persisted_absent(tmp_path, monkeypatch):
    """Persisted stop_after absent + explicit on resume -> explicit applies."""
    wid = _write_state(tmp_path, monkeypatch, {"course_name": "X"})
    effective = _apply_resume_stop_after_override(wid, "course_planning")
    assert effective == "course_planning"
    assert _read_persisted_stop_after(tmp_path, wid) == "course_planning"


def test_override_none_none_returns_none(tmp_path, monkeypatch):
    wid = _write_state(tmp_path, monkeypatch, {"course_name": "X"})
    assert _apply_resume_stop_after_override(wid, None) is None


# --------------------------------------------------------------------------
# Persistence at creation (create_workflow_impl keeps stop_after verbatim)
# --------------------------------------------------------------------------
def test_create_workflow_persists_stop_after(tmp_path, monkeypatch):
    import lib.paths as paths_mod
    import MCP.tools.orchestrator_tools as ot

    monkeypatch.setattr(paths_mod, "STATE_PATH", tmp_path / "state")
    monkeypatch.setattr(ot, "STATE_PATH", tmp_path / "state")

    params = {
        "course_name": "ZZ_TEST",
        "stop_after": "course_planning",
        "pdf_paths": ["/x.pdf"],
    }

    async def _go():
        return await ot.create_workflow_impl(
            "textbook_to_course", json.dumps(params), "normal"
        )

    res = json.loads(asyncio.run(_go()))
    assert res.get("success")
    state = json.loads(open(res["workflow_path"]).read())
    assert state["params"].get("stop_after") == "course_planning"


# --------------------------------------------------------------------------
# Bug B — --stop-after halts even when the named phase is satisfied by a
# skip-as-already-complete on resume (drives run_workflow directly).
# --------------------------------------------------------------------------
def _run_workflow_with_skipcomplete_stopafter(
    tmp_path,
    monkeypatch,
    *,
    stop_after: str,
    executed: List[str],
) -> Dict[str, Any]:
    """Drive run_workflow where the stop_after phase is pre-completed on
    resume and two downstream phases would run absent the stop.
    """
    from MCP.core import workflow_runner as wr_mod
    from MCP.core.config import WorkflowConfig, WorkflowPhase
    from MCP.core.workflow_runner import WorkflowRunner

    state_root = tmp_path / "state"
    monkeypatch.setattr(wr_mod, "STATE_PATH", state_root)
    monkeypatch.setenv("LOCAL_DISPATCHER_ALLOW_STUB", "1")

    workflow_id = "WF-STOPAFTER-SKIP"
    wf_dir = state_root / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    # ``alpha`` is pre-completed (as after a partial-but-stamped resume); the
    # downstream ``beta`` / ``gamma`` must NOT run because --stop-after=alpha.
    (wf_dir / f"{workflow_id}.json").write_text(
        json.dumps(
            {
                "workflow_id": workflow_id,
                "type": "test_wf",
                "params": {"stop_after": stop_after},
                "phase_outputs": {
                    "alpha": {"_completed": True, "_gates_passed": True, "ok": 1},
                },
                "tasks": [],
            }
        )
    )

    async def side_effect(*_a, **kw):  # noqa: ANN001
        executed.append(kw.get("phase_name") or (_a[1] if len(_a) > 1 else ""))
        return {"T-1": MagicMock(status="COMPLETE", result={"k": "v"})}, True, []

    executor = MagicMock()
    executor.execute_phase = AsyncMock(side_effect=side_effect)

    phases = [
        WorkflowPhase(name="alpha", agents=["content-generator"]),
        WorkflowPhase(
            name="beta", agents=["content-generator"], depends_on=["alpha"],
        ),
        WorkflowPhase(
            name="gamma", agents=["content-generator"], depends_on=["beta"],
        ),
    ]
    config = MagicMock()
    config.get_workflow.return_value = WorkflowConfig(
        description="test", phases=phases
    )

    runner = WorkflowRunner(executor=executor, config=config)
    out = asyncio.run(runner.run_workflow(workflow_id))
    return out


def test_stop_after_halts_on_skipped_as_complete_phase(tmp_path, monkeypatch):
    """--stop-after alpha halts even though alpha was skipped-as-complete."""
    executed: List[str] = []
    out = _run_workflow_with_skipcomplete_stopafter(
        tmp_path, monkeypatch, stop_after="alpha", executed=executed,
    )
    # Neither downstream phase executed — the run halted at alpha's slot.
    assert "beta" not in executed, "stop-after must halt before beta runs"
    assert "gamma" not in executed
    # The deliberate-halt marker is persisted (distinct from a FAILED phase).
    state = json.loads(
        (tmp_path / "state" / "workflows" / "WF-STOPAFTER-SKIP.json").read_text()
    )
    assert state.get("stopped_after") == "alpha"
    assert out["status"] != "FAILED"


def test_no_stop_after_marches_past_skipped_complete_phase(tmp_path, monkeypatch):
    """Control: without --stop-after, a skipped-complete alpha lets beta run."""
    executed: List[str] = []
    _run_workflow_with_skipcomplete_stopafter(
        tmp_path, monkeypatch, stop_after="", executed=executed,
    )
    # alpha is skip-complete; beta + gamma execute (no halt).
    assert "beta" in executed
    assert "gamma" in executed
