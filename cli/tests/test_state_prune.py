"""Tests for ``ed4all state prune`` (Wave 74 cleanup)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.commands.state_prune import (
    DEFAULT_KEEP_LAST,
    build_plan,
    state_group,
    _load_run_dirs,
    _load_workflows,
)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


def _write_workflow(
    workflows_dir: Path,
    workflow_id: str,
    *,
    status: str,
    run_id: str | None = None,
    updated_at: str | None = None,
) -> Path:
    """Create a synthetic ``state/workflows/{workflow_id}.json`` file."""
    path = workflows_dir / f"{workflow_id}.json"
    payload = {
        "id": workflow_id,
        "status": status,
        "updated_at": updated_at or datetime.now().isoformat(),
        "params": {"run_id": run_id or workflow_id},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_run_dir(runs_dir: Path, run_id: str, *, files: int = 1) -> Path:
    """Create a synthetic ``state/runs/{run_id}/`` directory with junk."""
    run_path = runs_dir / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    for i in range(files):
        (run_path / f"artifact_{i}.json").write_text(
            json.dumps({"index": i}), encoding="utf-8"
        )
    return run_path


@pytest.fixture
def fake_state_root(tmp_path: Path) -> Path:
    """Build ``tmp_path/state/{workflows,runs}`` skeleton."""
    (tmp_path / "state" / "workflows").mkdir(parents=True)
    (tmp_path / "state" / "runs").mkdir(parents=True)
    return tmp_path / "state"


# ---------------------------------------------------------------------- #
# build_plan unit tests
# ---------------------------------------------------------------------- #


def test_keep_last_n_keeps_exactly_n_complete(fake_state_root):
    workflows_dir = fake_state_root / "workflows"
    runs_dir = fake_state_root / "runs"

    base = datetime(2026, 4, 24, 12, 0, 0)
    # 5 COMPLETE workflows with monotonically-increasing updated_at +
    # matching run dirs.
    for i in range(5):
        wid = f"WF-{i:02d}"
        _write_workflow(
            workflows_dir,
            wid,
            status="COMPLETE",
            run_id=wid,
            updated_at=(base + timedelta(hours=i)).isoformat(),
        )
        _write_run_dir(runs_dir, wid)

    workflows = _load_workflows(workflows_dir)
    runs = _load_run_dirs(runs_dir)
    plan = build_plan(workflows, runs, keep_last=2, keep_statuses=set())

    assert len(plan.keep_workflows) == 2
    assert len(plan.drop_workflows) == 3
    # Most-recent two should be kept (WF-04, WF-03).
    kept_ids = {w.workflow_id for w in plan.keep_workflows}
    assert kept_ids == {"WF-04", "WF-03"}


def test_keep_status_protects_running_workflows(fake_state_root):
    workflows_dir = fake_state_root / "workflows"
    runs_dir = fake_state_root / "runs"

    _write_workflow(workflows_dir, "WF-RUN", status="RUNNING", run_id="WF-RUN")
    _write_run_dir(runs_dir, "WF-RUN")
    for i in range(3):
        wid = f"WF-DONE-{i}"
        _write_workflow(
            workflows_dir, wid, status="COMPLETE", run_id=wid,
            updated_at=datetime(2026, 4, 24, 12, i).isoformat(),
        )
        _write_run_dir(runs_dir, wid)

    workflows = _load_workflows(workflows_dir)
    runs = _load_run_dirs(runs_dir)
    # keep_last=0 — without RUNNING protection, ALL three COMPLETE drop.
    plan = build_plan(
        workflows, runs, keep_last=0, keep_statuses={"RUNNING"},
    )

    kept = {w.workflow_id for w in plan.keep_workflows}
    assert "WF-RUN" in kept
    # And WF-RUN's run dir must survive.
    kept_runs = {r.run_id for r in plan.keep_runs}
    assert "WF-RUN" in kept_runs


def test_orphan_run_dirs_are_pruned_by_default(fake_state_root):
    workflows_dir = fake_state_root / "workflows"
    runs_dir = fake_state_root / "runs"

    # One real workflow + matching run dir.
    _write_workflow(workflows_dir, "WF-LIVE", status="COMPLETE", run_id="WF-LIVE")
    _write_run_dir(runs_dir, "WF-LIVE")
    # Two orphan run dirs (no workflow JSON points at them).
    _write_run_dir(runs_dir, "run_20260424_120000")
    _write_run_dir(runs_dir, "run_20260424_130000")

    workflows = _load_workflows(workflows_dir)
    runs = _load_run_dirs(runs_dir)
    plan = build_plan(
        workflows, runs, keep_last=DEFAULT_KEEP_LAST,
        keep_statuses={"COMPLETE"},
    )

    orphan_ids = {r.run_id for r in plan.orphan_runs}
    assert orphan_ids == {"run_20260424_120000", "run_20260424_130000"}
    # Orphans always go into the drop list.
    drop_ids = {r.run_id for r in plan.drop_runs}
    assert orphan_ids.issubset(drop_ids)
    # Live workflow's run dir is preserved.
    assert "WF-LIVE" in {r.run_id for r in plan.keep_runs}


# ---------------------------------------------------------------------- #
# CLI invocation tests
# ---------------------------------------------------------------------- #


def test_dry_run_does_not_delete_anything(fake_state_root):
    workflows_dir = fake_state_root / "workflows"
    runs_dir = fake_state_root / "runs"

    for i in range(7):
        wid = f"WF-{i:02d}"
        _write_workflow(
            workflows_dir, wid, status="COMPLETE", run_id=wid,
            updated_at=datetime(2026, 4, 24, 10, i).isoformat(),
        )
        _write_run_dir(runs_dir, wid)

    runner = CliRunner()
    result = runner.invoke(
        state_group,
        [
            "prune",
            "--keep-last", "2",
            "--dry-run",
            "--state-root", str(fake_state_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DRY RUN" in result.output

    # Nothing actually deleted.
    assert len(list(workflows_dir.glob("*.json"))) == 7
    assert len([d for d in runs_dir.iterdir() if d.is_dir()]) == 7


def test_keep_last_2_actually_prunes(fake_state_root):
    """``--keep-last 2 --keep-status FAILED`` (override defaults so
    COMPLETE is no longer always-protected) keeps the 2 most-recent
    COMPLETE workflows."""
    workflows_dir = fake_state_root / "workflows"
    runs_dir = fake_state_root / "runs"

    for i in range(5):
        wid = f"WF-{i:02d}"
        _write_workflow(
            workflows_dir, wid, status="COMPLETE", run_id=wid,
            updated_at=datetime(2026, 4, 24, 10, i).isoformat(),
        )
        _write_run_dir(runs_dir, wid)

    runner = CliRunner()
    result = runner.invoke(
        state_group,
        [
            "prune",
            "--keep-last", "2",
            "--keep-status", "FAILED",  # drop COMPLETE from protected set
            "--state-root", str(fake_state_root),
        ],
    )
    assert result.exit_code == 0, result.output

    remaining_wfs = sorted(p.stem for p in workflows_dir.glob("*.json"))
    remaining_runs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
    assert remaining_wfs == ["WF-03", "WF-04"]
    assert remaining_runs == ["WF-03", "WF-04"]


def test_keep_status_running_via_cli(fake_state_root):
    workflows_dir = fake_state_root / "workflows"
    runs_dir = fake_state_root / "runs"

    _write_workflow(workflows_dir, "WF-RUN", status="RUNNING", run_id="WF-RUN")
    _write_run_dir(runs_dir, "WF-RUN")

    for i in range(3):
        wid = f"WF-DONE-{i}"
        _write_workflow(
            workflows_dir, wid, status="COMPLETE", run_id=wid,
            updated_at=datetime(2026, 4, 24, 10, i).isoformat(),
        )
        _write_run_dir(runs_dir, wid)

    runner = CliRunner()
    result = runner.invoke(
        state_group,
        [
            "prune",
            "--keep-last", "0",
            "--keep-status", "RUNNING",
            "--state-root", str(fake_state_root),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (workflows_dir / "WF-RUN.json").exists()
    assert (runs_dir / "WF-RUN").exists()
    # COMPLETE entries should all be gone.
    assert sorted(p.stem for p in workflows_dir.glob("*.json")) == ["WF-RUN"]
    assert sorted(d.name for d in runs_dir.iterdir() if d.is_dir()) == ["WF-RUN"]


def test_orphan_run_dirs_pruned_via_cli(fake_state_root):
    workflows_dir = fake_state_root / "workflows"
    runs_dir = fake_state_root / "runs"

    _write_workflow(workflows_dir, "WF-LIVE", status="COMPLETE", run_id="WF-LIVE")
    _write_run_dir(runs_dir, "WF-LIVE")
    _write_run_dir(runs_dir, "run_20260424_120000")
    _write_run_dir(runs_dir, "run_20260424_130000")

    runner = CliRunner()
    result = runner.invoke(
        state_group,
        ["prune", "--state-root", str(fake_state_root)],
    )
    assert result.exit_code == 0, result.output

    remaining_runs = sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
    assert remaining_runs == ["WF-LIVE"]


def test_gitkeep_marker_is_never_deleted(fake_state_root):
    workflows_dir = fake_state_root / "workflows"
    runs_dir = fake_state_root / "runs"

    (workflows_dir / ".gitkeep").write_text("", encoding="utf-8")
    (runs_dir / ".gitkeep").write_text("", encoding="utf-8")

    # Add a workflow that should drop, plus an orphan run dir.
    _write_workflow(
        workflows_dir, "WF-OLD", status="COMPLETE", run_id="WF-OLD",
        updated_at="2020-01-01T00:00:00",
    )
    _write_run_dir(runs_dir, "orphan_xxx")

    runner = CliRunner()
    result = runner.invoke(
        state_group,
        [
            "prune",
            "--keep-last", "0",
            "--keep-status", "RUNNING",  # drop COMPLETE from protection
            "--state-root", str(fake_state_root),
        ],
    )
    assert result.exit_code == 0, result.output

    assert (workflows_dir / ".gitkeep").exists()
    assert (runs_dir / ".gitkeep").exists()
    assert not (workflows_dir / "WF-OLD.json").exists()
    assert not (runs_dir / "orphan_xxx").exists()
