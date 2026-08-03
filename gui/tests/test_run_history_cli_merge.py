"""Runs-tab surfacing of CLI-launched orchestrator workflows.

Service-level: ``run_service.list_runs`` merges ``runtime/state/workflows/WF-*.json``
records (tagged ``source: "cli"``) into the GUI-registry listing (tagged
``source: "gui"``), de-duped on workflow id, newest-first, staleness-filtered
by the display heuristic (non-terminal ≤ 48 h / terminal ≤ 7 days), tolerant of
mid-write corrupt JSON, and capped by ``limit``.

Static-asset: ``run-history.js`` renders the CLI badge + links every row to the
existing ``#/create/<id>`` build-progress route (which resolves CLI ids via the
unknown_run → /progress fallback), and ``studio.css`` carries the badge tint.
Client-side JS can't be driven headless here, so per ``gui/CLAUDE.md`` those are
asserted against the served static assets (the ``test_embed_widget.py`` pattern).

State isolated via the ``state_dir`` fixture (``ED4ALL_STATE_RUNS_DIR``
redirection); no real run ids or course slugs referenced.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from gui import shared_state
from gui.services import run_service

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_HISTORY_JS = REPO_ROOT / "gui" / "static" / "studio" / "run-history.js"
STUDIO_CSS = REPO_ROOT / "gui" / "static" / "studio" / "studio.css"


# --------------------------------------------------------------- helpers


def _write_workflow_state(
    state_root: Path,
    workflow_id: str,
    *,
    status: str = "RUNNING",
    age: timedelta = timedelta(minutes=5),
    course_name: Optional[str] = "synthetic-course",
    params_as_string: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a minimal orchestrator workflow-state file (naive-local ISO times,
    matching what ``WorkflowRunner`` persists)."""
    stamp = (datetime.now() - age).isoformat()
    params: Any = {"course_name": course_name, "duration_weeks": 8}
    if params_as_string:
        params = json.dumps(params)
    state: Dict[str, Any] = {
        "id": workflow_id,
        "type": "textbook_to_course",
        "params": params,
        "status": status,
        "created_at": stamp,
        "updated_at": stamp,
        "started_at": stamp,
    }
    if extra:
        state.update(extra)
    target = state_root / "workflows" / f"{workflow_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state), encoding="utf-8")
    return target


def _register_gui_run(run_id: str, workflow_id: Optional[str] = None) -> None:
    shared_state.register_run(
        {
            "run_id": run_id,
            "kind": "pipeline",
            "workflow": "textbook_to_course",
            "workflow_id": workflow_id,
            "course_name": "course-alpha",
            "status": "running",
        }
    )


# --------------------------------------------------------- merge + tagging


def test_merge_surfaces_fresh_cli_run_with_mapped_fields(state_dir):
    _register_gui_run("GUI-20260101-aaaa01")
    _write_workflow_state(state_dir, "WF-20260101-cli00001", status="RUNNING")

    runs = run_service.list_runs()
    by_id = {r["run_id"]: r for r in runs}
    assert "GUI-20260101-aaaa01" in by_id
    assert "WF-20260101-cli00001" in by_id

    gui = by_id["GUI-20260101-aaaa01"]
    assert gui["source"] == "gui"

    cli = by_id["WF-20260101-cli00001"]
    assert cli["source"] == "cli"
    assert cli["workflow_id"] == "WF-20260101-cli00001"
    assert cli["workflow"] == "textbook_to_course"
    assert cli["status"] == "running"  # lowercased
    assert cli["course_name"] == "synthetic-course"  # from params
    assert cli["created_at"] and cli["started_at"]
    # No "Run again" ammunition: orchestrator params are shape-incompatible.
    assert cli["params"] is None


def test_cli_terminal_status_normalized_to_frontend_vocabulary(state_dir):
    _write_workflow_state(
        state_dir, "WF-20260101-cli00002", status="COMPLETE", age=timedelta(days=1)
    )
    runs = run_service.list_runs()
    assert [r["status"] for r in runs] == ["completed"]
    assert runs[0]["finished_at"] is not None or runs[0]["status"] == "completed"


def test_cli_params_as_json_string_still_yields_course_name(state_dir):
    _write_workflow_state(
        state_dir, "WF-20260101-cli00003", status="RUNNING", params_as_string=True
    )
    runs = run_service.list_runs()
    assert runs[0]["course_name"] == "synthetic-course"


def test_merge_is_newest_first_across_sources(state_dir):
    # CLI run older than the GUI run -> GUI first.
    _write_workflow_state(
        state_dir, "WF-20260101-cli00004", status="RUNNING", age=timedelta(hours=3)
    )
    _register_gui_run("GUI-20260101-aaaa02")  # created_at = now (UTC-aware)
    runs = run_service.list_runs()
    assert [r["run_id"] for r in runs] == [
        "GUI-20260101-aaaa02",
        "WF-20260101-cli00004",
    ]


# ----------------------------------------------------------------- de-dupe


def test_workflow_id_referenced_by_gui_record_not_listed_twice(state_dir):
    _register_gui_run("GUI-20260101-aaaa03", workflow_id="WF-20260101-cli00005")
    _write_workflow_state(state_dir, "WF-20260101-cli00005", status="RUNNING")

    runs = run_service.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "GUI-20260101-aaaa03"
    assert runs[0]["source"] == "gui"


# ------------------------------------------------------- staleness heuristic


def test_stale_running_cli_record_excluded(state_dir):
    """A zombie RUNNING record from a crashed run days ago must not surface."""
    _write_workflow_state(
        state_dir, "WF-20260101-cli00006", status="RUNNING", age=timedelta(days=5)
    )
    assert run_service.list_runs() == []


def test_fresh_running_cli_record_included(state_dir):
    _write_workflow_state(
        state_dir, "WF-20260101-cli00007", status="RUNNING", age=timedelta(hours=47)
    )
    assert [r["run_id"] for r in run_service.list_runs()] == ["WF-20260101-cli00007"]


def test_terminal_cli_record_within_seven_days_included(state_dir):
    _write_workflow_state(
        state_dir, "WF-20260101-cli00008", status="FAILED", age=timedelta(days=6)
    )
    runs = run_service.list_runs()
    assert [r["run_id"] for r in runs] == ["WF-20260101-cli00008"]
    assert runs[0]["status"] == "failed"


def test_terminal_cli_record_older_than_seven_days_excluded(state_dir):
    _write_workflow_state(
        state_dir, "WF-20260101-cli00009", status="FAILED", age=timedelta(days=10)
    )
    assert run_service.list_runs() == []


# ----------------------------------------------------- corruption + limit


def test_corrupt_mid_write_json_skipped(state_dir):
    target = state_dir / "workflows" / "WF-20260101-cli00010.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text('{"id": "WF-20260101-cli00010", "status": "RUN', encoding="utf-8")
    _write_workflow_state(state_dir, "WF-20260101-cli00011", status="RUNNING")

    runs = run_service.list_runs()
    assert [r["run_id"] for r in runs] == ["WF-20260101-cli00011"]


def test_limit_caps_merged_listing(state_dir):
    _register_gui_run("GUI-20260101-aaaa04")
    _write_workflow_state(
        state_dir, "WF-20260101-cli00012", status="RUNNING", age=timedelta(hours=1)
    )
    _write_workflow_state(
        state_dir, "WF-20260101-cli00013", status="RUNNING", age=timedelta(hours=2)
    )
    assert len(run_service.list_runs()) == 3
    assert len(run_service.list_runs(limit=2)) == 2
    assert run_service.list_runs(limit=0) == []


def test_missing_workflows_dir_yields_gui_only(state_dir):
    _register_gui_run("GUI-20260101-aaaa05")
    # state_dir fixture creates runs/ only; workflows/ does not exist.
    runs = run_service.list_runs()
    assert [r["run_id"] for r in runs] == ["GUI-20260101-aaaa05"]


# ---------------------------------------------------- static-asset contract


def _js() -> str:
    return RUN_HISTORY_JS.read_text(encoding="utf-8")


def test_run_history_renders_cli_badge():
    js = _js()
    assert "r.source === 'cli'" in js, (
        "run-history.js must discriminate CLI-sourced records via source: 'cli'"
    )
    assert "cli-badge" in js and "'CLI'" in js, (
        "run-history.js must render the small CLI badge for CLI-sourced rows"
    )


def test_run_history_links_rows_to_build_progress_route():
    js = _js()
    # Every row links to the existing build-progress route; CLI ids resolve
    # there via create.js's unknown_run -> /progress fallback.
    assert "#/create/${encodeURIComponent(runId)}" in js


def test_run_history_shows_when():
    js = _js()
    assert "fmtWhen(" in js and "r.updated_at || r.created_at" in js


def test_studio_css_carries_cli_badge_tint():
    css = STUDIO_CSS.read_text(encoding="utf-8")
    assert ".card-badge.cli-badge" in css


# ---- stage identity + parent/course grouping ----------------------------
#
# A `courseforge-*` stage subcommand mints its OWN workflow id AND its own
# orchestrator run id, so it surfaces as an unrelated row whose rail/stats come
# from a 2-checkpoint run dir — visually a near-empty build of a course that is
# in fact far along. These pin the association that makes it readable.

def _rec(run_id, *, course, stage=None, created, workflow="textbook_to_course"):
    return {
        "run_id": run_id,
        "workflow": workflow,
        "course_name": course,
        "course_group": course,
        "stage": stage,
        "created_at": created,
        "source": "cli",
    }


def test_stage_run_points_at_preceding_build():
    runs = [  # newest-first, as list_runs sorts them
        _rec("WF-stage", course="c1", stage="courseforge_rewrite",
             created="2026-07-25T08:00:00"),
        _rec("WF-build", course="c1", created="2026-07-24T11:00:00"),
    ]
    run_service._attach_run_lineage(runs)
    by_id = {r["run_id"]: r for r in runs}
    assert by_id["WF-stage"]["stage"] == "courseforge_rewrite"
    assert by_id["WF-stage"]["parent_run_id"] == "WF-build"
    # A full build is never itself a stage, and never gets a parent.
    assert by_id["WF-build"]["stage"] is None
    assert by_id["WF-build"]["parent_run_id"] is None


def test_stage_run_never_adopts_another_courses_build():
    runs = [
        _rec("WF-stage", course="c1", stage="courseforge_rewrite",
             created="2026-07-25T08:00:00"),
        _rec("WF-other", course="c2", created="2026-07-24T11:00:00"),
    ]
    run_service._attach_run_lineage(runs)
    assert runs[0]["parent_run_id"] is None, "no build for c1 — invent nothing"


def test_stage_run_with_no_preceding_build_has_no_parent():
    """Honest-only: a LATER build must not be back-adopted as the parent."""
    runs = [
        _rec("WF-build", course="c1", created="2026-07-25T09:00:00"),
        _rec("WF-stage", course="c1", stage="courseforge_rewrite",
             created="2026-07-25T08:00:00"),
    ]
    run_service._attach_run_lineage(runs)
    by_id = {r["run_id"]: r for r in runs}
    assert by_id["WF-stage"]["parent_run_id"] is None


def test_stage_run_binds_to_the_newest_preceding_build():
    runs = [
        _rec("WF-stage", course="c1", stage="courseforge_rewrite",
             created="2026-07-25T08:00:00"),
        _rec("WF-build-new", course="c1", created="2026-07-24T11:00:00"),
        _rec("WF-build-old", course="c1", created="2026-07-20T11:00:00"),
    ]
    run_service._attach_run_lineage(runs)
    assert runs[0]["parent_run_id"] == "WF-build-new"


def test_gui_stage_run_derives_stage_from_workflow_name():
    """A GUI-launched stage run carries it as the requested workflow name."""
    rec = {
        "run_id": "GUI-1", "workflow": "courseforge-rewrite",
        "course_name": "c1", "created_at": "2026-07-25T08:00:00",
        "source": "gui",
    }
    runs = [rec, _rec("WF-build", course="c1", created="2026-07-24T11:00:00")]
    run_service._attach_run_lineage(runs)
    assert rec["stage"] == "courseforge_rewrite"
    assert rec["parent_run_id"] == "WF-build"


def test_lineage_tolerates_records_missing_identity():
    runs = [{"run_id": None}, {"course_name": None, "run_id": "x"}]
    run_service._attach_run_lineage(runs)  # must not raise
    assert runs[1]["parent_run_id"] is None
