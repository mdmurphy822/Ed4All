"""Graceful-stop CLI tests — ``ed4all stop`` + run.py signal/exit plumbing.

Hermetic: every test isolates ``runtime/state/runs`` via ``ED4ALL_STATE_RUNS_DIR`` (so
sentinel writes land in a tmp dir, never the real project state) and
monkeypatches ``lib.paths.STATE_PATH`` for the synthetic ``runtime/state/workflows/``
records. No real workflow is run; the signal-handler and exit-code legs are
driven directly.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.commands import run as run_mod
from cli.commands import stop as stop_mod
from cli.main import cli
from lib.generation import stop_control


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------
@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    """Point runtime/state/runs (sentinels) + runtime/state/workflows (records) at a tmp dir."""
    import lib.paths as paths_mod

    state_root = tmp_path / "state"
    runs_dir = state_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (state_root / "workflows").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(runs_dir))
    monkeypatch.delenv("ED4ALL_RUN_ID", raising=False)
    monkeypatch.delenv("ED4ALL_HOME", raising=False)
    monkeypatch.setattr(paths_mod, "STATE_PATH", state_root)
    return state_root


def _write_workflow(
    state_root: Path,
    workflow_id: str,
    *,
    status: str = "RUNNING",
    run_id: str | None = None,
    age_seconds: float | None = None,
) -> Path:
    params = {}
    if run_id is not None:
        params["run_id"] = run_id
    path = state_root / "workflows" / f"{workflow_id}.json"
    path.write_text(
        json.dumps(
            {
                "id": workflow_id,
                "type": "test_wf",
                "status": status,
                "params": params,
                "updated_at": "2026-07-05T00:00:00+00:00",
            }
        )
    )
    if age_seconds is not None:
        old = time.time() - age_seconds
        os.utime(path, (old, old))
    return path


def _run_sentinel(state_root: Path, run_id: str) -> Path:
    return state_root / "runs" / run_id / "control" / "STOP_REQUESTED"


def _global_sentinel(state_root: Path) -> Path:
    return state_root / "runs" / "STOP_ALL"


# --------------------------------------------------------------------------
# ed4all stop <target>
# --------------------------------------------------------------------------
def test_stop_target_writes_run_sentinel_by_run_id(isolated_state):
    _write_workflow(isolated_state, "WF-AAA", run_id="RUN-AAA")
    result = CliRunner().invoke(cli, ["stop", "RUN-AAA"])
    assert result.exit_code == 0, result.output
    assert _run_sentinel(isolated_state, "RUN-AAA").exists()


def test_stop_target_matches_by_workflow_id(isolated_state):
    # Generic workflow: no params.run_id -> run_id falls back to the workflow id.
    _write_workflow(isolated_state, "WF-BBB", run_id=None)
    result = CliRunner().invoke(cli, ["stop", "WF-BBB"])
    assert result.exit_code == 0, result.output
    assert _run_sentinel(isolated_state, "WF-BBB").exists()


def test_stop_target_maps_workflow_id_to_run_id_sentinel(isolated_state):
    # Target the WORKFLOW id but the sentinel must key on the mapped run_id.
    _write_workflow(isolated_state, "WF-CCC", run_id="RUN-CCC")
    result = CliRunner().invoke(cli, ["stop", "WF-CCC"])
    assert result.exit_code == 0, result.output
    assert _run_sentinel(isolated_state, "RUN-CCC").exists()
    assert not _run_sentinel(isolated_state, "WF-CCC").exists()


def test_stop_unmatched_target_still_writes_best_effort_sentinel(isolated_state):
    result = CliRunner().invoke(cli, ["stop", "RUN-UNKNOWN"])
    assert result.exit_code == 0, result.output
    assert _run_sentinel(isolated_state, "RUN-UNKNOWN").exists()
    assert "no RUNNING workflow matched" in result.output


def test_stop_requires_a_mode(isolated_state):
    result = CliRunner().invoke(cli, ["stop"])
    assert result.exit_code == 2
    assert "provide a workflow" in result.output


def test_stop_rejects_two_modes(isolated_state):
    result = CliRunner().invoke(cli, ["stop", "RUN-X", "--all"])
    assert result.exit_code == 2
    assert "exactly one" in result.output


# --------------------------------------------------------------------------
# ed4all stop --all
# --------------------------------------------------------------------------
def test_stop_all_writes_global_sentinel_and_lists_running(isolated_state):
    _write_workflow(isolated_state, "WF-RUN1", run_id="RUN-1")
    _write_workflow(isolated_state, "WF-DONE", status="COMPLETE", run_id="RUN-2")
    result = CliRunner().invoke(cli, ["stop", "--all"])
    assert result.exit_code == 0, result.output
    assert _global_sentinel(isolated_state).exists()
    # RUNNING is enumerated; COMPLETE is not.
    assert "WF-RUN1" in result.output
    assert "WF-DONE" not in result.output


def test_stop_all_json_shape(isolated_state):
    _write_workflow(isolated_state, "WF-RUN1", run_id="RUN-1")
    result = CliRunner().invoke(cli, ["stop", "--all", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "all"
    assert payload["sentinel"].endswith("STOP_ALL")
    assert [r["workflow_id"] for r in payload["running"]] == ["WF-RUN1"]


# --------------------------------------------------------------------------
# ed4all stop --clear-all
# --------------------------------------------------------------------------
def test_stop_clear_all_removes_global_sentinel(isolated_state):
    stop_control.request_stop(scope="all", reason="t", source="t")
    assert _global_sentinel(isolated_state).exists()
    result = CliRunner().invoke(cli, ["stop", "--clear-all"])
    assert result.exit_code == 0, result.output
    assert not _global_sentinel(isolated_state).exists()
    assert "Cleared the global STOP_ALL" in result.output


def test_stop_clear_all_noop_when_absent(isolated_state):
    result = CliRunner().invoke(cli, ["stop", "--clear-all"])
    assert result.exit_code == 0, result.output
    assert "already absent" in result.output


# --------------------------------------------------------------------------
# Stale-RUNNING heuristic (AMENDMENT #9)
# --------------------------------------------------------------------------
def test_stale_running_workflow_annotated(isolated_state):
    _write_workflow(
        isolated_state, "WF-OLD", run_id="RUN-OLD", age_seconds=48 * 3600
    )
    _write_workflow(
        isolated_state, "WF-FRESH", run_id="RUN-FRESH", age_seconds=60
    )
    running = stop_mod._load_running_workflows()
    by_id = {r["workflow_id"]: r for r in running}
    assert by_id["WF-OLD"]["stale"] is True
    assert by_id["WF-FRESH"]["stale"] is False
    # Surfaced in the --all report text.
    result = CliRunner().invoke(cli, ["stop", "--all"])
    assert "WF-OLD" in result.output
    assert "possibly stale" in result.output


# --------------------------------------------------------------------------
# Exit-code-3 mapping (D8) — _paused_exit_code
# --------------------------------------------------------------------------
class _FakeResult:
    def __init__(self, status):
        self.status = status


def test_paused_exit_code_returns_3_and_hint(capsys):
    code = run_mod._paused_exit_code(
        _FakeResult("paused"), "WF-PAUSE", output_json=False
    )
    assert code == 3
    out = capsys.readouterr().out
    assert "paused at a checkpoint" in out
    assert "--resume WF-PAUSE" in out


def test_paused_exit_code_json_suppresses_hint(capsys):
    code = run_mod._paused_exit_code(
        _FakeResult("paused"), "WF-PAUSE", output_json=True
    )
    assert code == 3
    assert capsys.readouterr().out == ""


def test_paused_exit_code_none_for_ok(capsys):
    assert (
        run_mod._paused_exit_code(_FakeResult("ok"), "WF-OK", output_json=False)
        is None
    )
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# Signal-handler unit (first → sentinel, second → SIG_DFL, no self-kill)
# --------------------------------------------------------------------------
def test_signal_handler_first_writes_sentinel_second_restores_default(
    isolated_state, monkeypatch
):
    # Do NOT actually kill the test process — record the re-raise instead.
    raised: list[int] = []
    monkeypatch.setattr(signal, "raise_signal", lambda s: raised.append(s))

    # Snapshot current dispositions so we can restore them after the test.
    prev_term = signal.getsignal(signal.SIGTERM)
    prev_int = signal.getsignal(signal.SIGINT)
    try:
        handler = run_mod._make_stop_signal_handler("RUN-SIG", "WF-SIG")

        # First signal → writes the run-scoped sentinel, no re-raise.
        handler(signal.SIGTERM, None)
        assert _run_sentinel(isolated_state, "RUN-SIG").exists()
        assert raised == []

        # Second signal → restores DEFAULT disposition on both stop signals
        # and re-raises (recorded, not executed).
        handler(signal.SIGTERM, None)
        assert signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
        assert signal.getsignal(signal.SIGINT) == signal.SIG_DFL
        assert raised == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)


def test_second_signal_marks_running_workflow_interrupted(
    isolated_state, monkeypatch
):
    monkeypatch.setattr(signal, "raise_signal", lambda s: None)
    _write_workflow(isolated_state, "WF-HARD", run_id="RUN-HARD", status="RUNNING")

    prev_term = signal.getsignal(signal.SIGTERM)
    prev_int = signal.getsignal(signal.SIGINT)
    try:
        handler = run_mod._make_stop_signal_handler("RUN-HARD", "WF-HARD")
        handler(signal.SIGTERM, None)  # first → sentinel
        handler(signal.SIGTERM, None)  # second → hard kill path
    finally:
        signal.signal(signal.SIGTERM, prev_term)
        signal.signal(signal.SIGINT, prev_int)

    state = json.loads(
        (isolated_state / "workflows" / "WF-HARD.json").read_text()
    )
    assert state["status"] == "paused"


# --------------------------------------------------------------------------
# run_id resolution for the signal handler
# --------------------------------------------------------------------------
def test_resolve_run_id_prefers_params_run_id(isolated_state):
    _write_workflow(isolated_state, "WF-RID", run_id="RUN-RID")
    assert run_mod._resolve_run_id_for_workflow("WF-RID") == "RUN-RID"


def test_resolve_run_id_falls_back_to_workflow_id(isolated_state):
    _write_workflow(isolated_state, "WF-NORID", run_id=None)
    assert run_mod._resolve_run_id_for_workflow("WF-NORID") == "WF-NORID"


def test_resolve_run_id_missing_file_degrades(isolated_state):
    assert run_mod._resolve_run_id_for_workflow("WF-GONE") == "WF-GONE"
