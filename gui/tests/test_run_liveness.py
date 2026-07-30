"""Process-liveness + honest run-state derivation (the incomplete/dead-run defect).

A CLI-launched ``ed4all run`` whose ``runtime/state/workflows/WF-*.json`` still says
``RUNNING`` used to render "Building" forever after the process was stopped or
died (the orchestrator has no reaper). ``gui.services.liveness`` derives an
honest ``effective_status`` from real signals — a ``/proc`` scan for live
``ed4all run`` processes, the graceful-stop sentinel, and workflow-file mtime —
without ever writing pipeline state.

These tests are hermetic: the ``/proc`` scan is fed synthetic cmdlines (never
the real host), the process list is injected into ``effective_status``, and all
state lives under the ``state_dir`` tmp redirect. They cover paused, stopping
(sentinel), incomplete (no process), building (live/attributed), stalled?
(unattributable + stale), terminal passthrough, uppercase normalization, and
the ``list_runs`` / ``run_progress`` integration seams.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from gui import shared_state
from gui.services import liveness, progress_service, run_service


@pytest.fixture(autouse=True)
def _no_seat_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """No registered seats → run_progress never opens a socket in these tests."""
    monkeypatch.delenv("ED4ALL_SEAT_BASE_URLS", raising=False)
    monkeypatch.delenv("COURSEFORGE_TWO_PASS", raising=False)
    progress_service._USAGE_ACCUMULATORS.clear()
    progress_service._UNIT_COUNT_CACHE.clear()
    progress_service._VRAM_CACHE.clear()


# ---------------------------------------------------------------- helpers


def _proc(pid: int, *argv: str) -> Tuple[int, List[str]]:
    return (pid, list(argv))


def _write_cli_state(
    state_root: Path,
    workflow_id: str,
    *,
    status: str = "RUNNING",
    run_id: str = "TTC_liveness_0001",  # synthetic orch run id, slug-guard: allow
    age: timedelta = timedelta(minutes=2),
    corpus: str = "/data/inputs/demo",
    course_name: str = "TEST_COURSE",
) -> Path:
    """Write a minimal orchestrator workflow-state file (naive-local ISO time)."""
    stamp = (datetime.now() - age).isoformat()
    state: Dict[str, Any] = {
        "id": workflow_id,
        "type": "textbook_to_course",
        "params": {
            "course_name": course_name,
            "corpus": corpus,
            "run_id": run_id,
        },
        "status": status,
        "created_at": stamp,
        "updated_at": stamp,
        "started_at": stamp,
        "phase_outputs": {},
    }
    target = state_root / "workflows" / f"{workflow_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(state), encoding="utf-8")
    return target


def _drop_stop_sentinel(state_root: Path, run_id: str) -> None:
    control = state_root / "runs" / run_id / "control"
    control.mkdir(parents=True, exist_ok=True)
    (control / "STOP_REQUESTED").write_text("", encoding="utf-8")


# ----------------------------------------------------- /proc scan adjacency


class _FakeCmdline:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def __enter__(self) -> "_FakeCmdline":
        return self

    def __exit__(self, *_a: Any) -> bool:
        return False

    def read(self) -> bytes:
        return self._data


def test_scan_matches_adjacent_ed4all_run_tokens_only(monkeypatch: pytest.MonkeyPatch):
    """Only an EXACT ``ed4all`` argv token immediately followed by ``run`` counts.

    A ``bash -c 'ed4all run ...'`` wrapper's whole script is ONE argv token, so
    it must never false-positive (the pgrep -f self-match trap).
    """
    procs: Dict[str, bytes] = {
        # live pipeline: /usr/bin/ed4all\0run\0textbook-to-course\0--corpus\0/x
        "1001": b"/usr/bin/ed4all\x00run\x00textbook-to-course\x00--corpus\x00/data/x\x00",
        # bare-name invocation
        "1002": b"ed4all\x00run\x00course_generation\x00",
        # wrapper shell — the whole "ed4all run ..." is a SINGLE token → reject
        "1003": b"bash\x00-c\x00ed4all run textbook-to-course\x00",
        # unrelated process
        "1004": b"python\x00-m\x00pytest\x00",
        # 'ed4all' present but next token isn't 'run'
        "1005": b"ed4all\x00doctor\x00",
        "not-a-pid": b"junk\x00",
    }
    monkeypatch.setattr(liveness.os, "listdir", lambda _p: list(procs.keys()))

    def _fake_open(path: str, *_a: Any, **_k: Any) -> _FakeCmdline:
        pid = str(path).split("/")[2]
        return _FakeCmdline(procs[pid])

    monkeypatch.setattr("builtins.open", _fake_open)

    result = liveness.scan_pipeline_processes()
    pids = {pid for pid, _argv in result}
    assert pids == {1001, 1002}


def test_scan_tolerates_unreadable_proc(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        liveness.os,
        "listdir",
        lambda _p: (_ for _ in ()).throw(OSError("no /proc")),
    )
    assert liveness.scan_pipeline_processes() == []


# --------------------------------------------------- effective_status matrix


def test_paused_is_paused_not_building():
    # Paused wins over the process check — it is resumable, not building.
    assert liveness.effective_status("PAUSED", is_cli=True, processes=[]) == "paused"
    assert liveness.effective_status("paused", is_cli=False, processes=[]) == "paused"


def test_stop_sentinel_yields_stopping(state_dir: Path):
    _drop_stop_sentinel(state_dir, "TTC_stopping")  # slug-guard: allow
    # Sentinel wins even with NO live process (the run is draining to a ckpt).
    assert (
        liveness.effective_status(
            "RUNNING",
            is_cli=True,
            orch_run_id="TTC_stopping",  # slug-guard: allow
            processes=[],
            state_root=state_dir,
        )
        == "stopping"
    )


def test_running_cli_with_no_process_is_incomplete(state_dir: Path):
    assert (
        liveness.effective_status(
            "RUNNING",
            is_cli=True,
            orch_run_id="TTC_gone",  # slug-guard: allow
            processes=[],
            state_root=state_dir,
        )
        == "incomplete"
    )


def test_persisted_incomplete_status_passes_through():
    # A raw status of "incomplete" is a member of _FINAL_STATUSES, so it passes
    # through effective_status unchanged (no re-derivation), regardless of
    # process liveness or is_cli.
    assert "incomplete" in liveness._FINAL_STATUSES
    assert (
        liveness.effective_status("incomplete", is_cli=True, processes=[])
        == "incomplete"
    )
    assert (
        liveness.effective_status("INCOMPLETE", is_cli=False, processes=[])
        == "incomplete"
    )


def test_attributed_live_process_is_building():
    procs = [_proc(4242, "/usr/bin/ed4all", "run", "textbook-to-course", "--corpus", "/data/inputs/demo")]
    assert (
        liveness.effective_status(
            "running",
            is_cli=True,
            attribution_tokens=["/data/inputs/demo"],
            processes=procs,
        )
        == "building"
    )


def test_unattributable_fresh_process_is_building():
    procs = [_proc(4242, "/usr/bin/ed4all", "run", "other-course", "--corpus", "/some/other")]
    now = 1_000_000.0
    assert (
        liveness.effective_status(
            "running",
            is_cli=True,
            attribution_tokens=["/data/inputs/demo"],
            processes=procs,
            wf_mtime=now - 5.0,  # fresh
            now=now,
        )
        == "building"
    )


def test_unattributable_stale_process_is_stalled():
    procs = [_proc(4242, "/usr/bin/ed4all", "run", "other-course")]
    now = 1_000_000.0
    assert (
        liveness.effective_status(
            "running",
            is_cli=True,
            attribution_tokens=["/data/inputs/demo"],
            processes=procs,
            wf_mtime=now - (liveness.STALE_MTIME_SECONDS + 60.0),
            now=now,
        )
        == "stalled?"
    )


def test_resume_process_attributes_via_workflow_id_is_building():
    """The reported defect: a ``--resume WF-<id>`` argv carries NEITHER the
    corpus/project NOR the orch_run_id — only the workflow id. With the WF-id in
    the attribution token set, the run self-attributes and reads ``building``
    even though its workflow file is long-stale (a single long outline phase)."""
    procs = [
        _proc(
            4242, "ed4all", "run", "textbook-to-course", "--resume",
            "WF-20260724-ac29168b",
        )
    ]
    now = 1_000_000.0
    assert (
        liveness.effective_status(
            "running",
            is_cli=True,
            # orch_run_id differs from the workflow id (e.g. TTC_..._113557) and
            # is NOT in the resume argv — only the WF-id token rescues this.
            orch_run_id="TTC_openstax_113557",  # slug-guard: allow
            attribution_tokens=["/data/inputs/demo", "WF-20260724-ac29168b"],
            processes=procs,
            wf_mtime=now - (liveness.STALE_MTIME_SECONDS + 600.0),  # very stale
            now=now,
        )
        == "building"
    )


def test_resume_process_without_workflow_id_token_still_stalls():
    """Companion / control: the SAME resume argv + stale file, but WITHOUT the
    WF-id in the attribution set, cannot self-attribute → ``stalled?``. Proves
    the workflow-id token is precisely what flips the verdict to ``building``."""
    procs = [
        _proc(
            4242, "ed4all", "run", "textbook-to-course", "--resume",
            "WF-20260724-ac29168b",
        )
    ]
    now = 1_000_000.0
    assert (
        liveness.effective_status(
            "running",
            is_cli=True,
            orch_run_id="TTC_openstax_113557",  # slug-guard: allow
            attribution_tokens=["/data/inputs/demo"],  # no WF-id token
            processes=procs,
            wf_mtime=now - (liveness.STALE_MTIME_SECONDS + 600.0),
            now=now,
        )
        == "stalled?"
    )


def test_gui_run_status_unchanged_without_process(state_dir: Path):
    """A GUI run is in-process — no /proc signal — so RUNNING stays running."""
    assert (
        liveness.effective_status(
            "RUNNING",
            is_cli=False,
            orch_run_id="TTC_gui",  # slug-guard: allow
            processes=[],
            state_root=state_dir,
        )
        == "running"
    )


def test_gui_run_honors_stop_sentinel(state_dir: Path):
    _drop_stop_sentinel(state_dir, "TTC_gui_stop")  # slug-guard: allow
    assert (
        liveness.effective_status(
            "running",
            is_cli=False,
            orch_run_id="TTC_gui_stop",  # slug-guard: allow
            processes=[],
            state_root=state_dir,
        )
        == "stopping"
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("COMPLETE", "completed"),
        ("completed", "completed"),
        ("FAILED", "failed"),
        ("Cancelled", "cancelled"),
        ("timeout", "timeout"),
    ],
)
def test_terminal_passthrough_and_normalization(raw: str, expected: str):
    # Terminal statuses pass through (lowercased/folded) regardless of process.
    assert liveness.effective_status(raw, is_cli=True, processes=[]) == expected


def test_attribution_tokens_from_params():
    tokens = liveness.attribution_tokens_from_params(
        {
            "corpus": "/data/x",
            "pdf_paths": ["/data/a.pdf", "/data/b.pdf"],
            "project_path": "/exports/proj",
            "course_name": "PHYS_101",
        },
        "TTC_run_0001",  # slug-guard: allow
    )
    assert tokens == [
        "/data/x",
        "/data/a.pdf",
        "/data/b.pdf",
        "/exports/proj",
        "PHYS_101",
        "TTC_run_0001",  # slug-guard: allow
    ]


def test_attribution_tokens_include_workflow_id_when_passed():
    """A ``--resume WF-<id>`` process carries only the workflow id in its argv,
    so ``wf_id`` MUST be an attribution token (else a resuming build reads
    ``stalled?``)."""
    tokens = liveness.attribution_tokens_from_params(
        {"corpus": "/data/x"},
        "TTC_run_0002",  # slug-guard: allow
        wf_id="WF-20260724-ac29168b",
    )
    assert "WF-20260724-ac29168b" in tokens
    # Order-preserving + de-duplicated, appended after the existing sources.
    assert tokens == ["/data/x", "TTC_run_0002", "WF-20260724-ac29168b"]  # slug-guard: allow
    # A blank/absent wf_id adds nothing (byte-identical for non-resume callers).
    assert liveness.attribution_tokens_from_params({"corpus": "/data/x"}) == ["/data/x"]
    assert liveness.attribution_tokens_from_params(
        {"corpus": "/data/x"}, wf_id="  "
    ) == ["/data/x"]


# ------------------------------------------------------ list_runs integration


def test_list_runs_stamps_incomplete_on_dead_cli_run(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The reported defect: a RUNNING CLI record with no process → incomplete."""
    _write_cli_state(state_dir, "WF-20260722-cli00001", status="RUNNING")
    monkeypatch.setattr(liveness, "scan_pipeline_processes", lambda: [])

    runs = run_service.list_runs()
    rec = {r["run_id"]: r for r in runs}["WF-20260722-cli00001"]
    assert rec["status"] == "running"  # raw status untouched
    assert rec["effective_status"] == "incomplete"


def test_list_runs_stamps_stopping_when_sentinel_present(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_cli_state(
        state_dir, "WF-20260722-cli00002", status="RUNNING", run_id="TTC_stop_2"  # slug-guard: allow
    )
    _drop_stop_sentinel(state_dir, "TTC_stop_2")  # slug-guard: allow
    monkeypatch.setattr(liveness, "scan_pipeline_processes", lambda: [])

    runs = run_service.list_runs()
    rec = {r["run_id"]: r for r in runs}["WF-20260722-cli00002"]
    assert rec["effective_status"] == "stopping"


def test_list_runs_stamps_building_on_attributed_process(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_cli_state(
        state_dir, "WF-20260722-cli00003", status="RUNNING", corpus="/data/inputs/live"
    )
    monkeypatch.setattr(
        liveness,
        "scan_pipeline_processes",
        lambda: [_proc(555, "/usr/bin/ed4all", "run", "textbook-to-course", "--corpus", "/data/inputs/live")],
    )

    runs = run_service.list_runs()
    rec = {r["run_id"]: r for r in runs}["WF-20260722-cli00003"]
    assert rec["effective_status"] == "building"


def test_list_runs_stamps_effective_status_on_gui_paused_run(state_dir: Path):
    shared_state.register_run(
        {
            "run_id": "GUI-20260722-aaaa01",
            "kind": "pipeline",
            "workflow": "textbook_to_course",
            "workflow_id": "WF-20260722-gui00001",
            "course_name": "GUI_COURSE",
            "status": "paused",
        }
    )
    runs = run_service.list_runs()
    rec = {r["run_id"]: r for r in runs}["GUI-20260722-aaaa01"]
    assert rec["effective_status"] == "paused"


# ---------------------------------------------------- run_progress integration


def test_run_progress_effective_status_incomplete_for_dead_cli_run(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_cli_state(state_dir, "WF-20260722-prog0001", status="RUNNING")
    monkeypatch.setattr(liveness, "scan_pipeline_processes", lambda: [])

    payload = progress_service.run_progress("WF-20260722-prog0001")
    assert payload is not None
    assert payload["status"] == "running"
    assert payload["effective_status"] == "incomplete"


def test_run_progress_effective_status_building_when_attributed(
    state_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    _write_cli_state(
        state_dir, "WF-20260722-prog0002", status="RUNNING", corpus="/data/inputs/live2"
    )
    monkeypatch.setattr(
        liveness,
        "scan_pipeline_processes",
        lambda: [_proc(777, "ed4all", "run", "textbook-to-course", "--corpus", "/data/inputs/live2")],
    )
    payload = progress_service.run_progress("WF-20260722-prog0002")
    assert payload is not None
    assert payload["effective_status"] == "building"


def test_run_progress_terminal_status_passes_through(state_dir: Path):
    _write_cli_state(state_dir, "WF-20260722-prog0003", status="COMPLETE")
    payload = progress_service.run_progress("WF-20260722-prog0003")
    assert payload is not None
    assert payload["effective_status"] == "completed"
