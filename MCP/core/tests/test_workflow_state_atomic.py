"""Workflow-state file write/read hardening (2026-07-21 corruption incident).

Two concurrent ``ed4all run --resume`` processes raced non-atomic
``open(path, 'w') + json.dump`` writes to ``runtime/state/workflows/WF-<id>.json``,
interleaving partial documents mid-file (one nearly-complete doc plus the
head of a second appended). Both processes then crashed parsing the result.

Covered here:

- Every writer of the workflow-state file goes through the canonical
  ``lib.state_manager.atomic_write_json`` temp+replace pattern (asserted both
  behaviorally, by intercepting the rename, and via a grep of the writer
  function sources for the old direct-write pattern).
- Two concurrent writers hammering the same path can no longer corrupt it:
  every observed on-disk state parses, including the final one (this test
  FAILS on the pre-fix ``open('w')`` writer).
- A corrupted workflow-state file surfaces on resume as a clear
  ``StateFileCorruptedError`` naming the file, the corruption position, and
  the checkpoint-recovery hint — never a bare ``json.JSONDecodeError``.

Hermetic: ``STATE_PATH`` is monkeypatched into ``tmp_path``; nothing under
the real ``runtime/state/`` is touched.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import MCP.core.executor as executor_mod  # noqa: E402
import MCP.core.workflow_runner as runner_mod  # noqa: E402
import MCP.orchestrator.pipeline_orchestrator as orch_mod  # noqa: E402
import lib.state_manager as state_manager_mod  # noqa: E402
from MCP.core.executor import TaskExecutor  # noqa: E402
from MCP.core.workflow_runner import WorkflowRunner  # noqa: E402
from lib.state_manager import StateFileCorruptedError  # noqa: E402

WF_ID = "WF-20260721-testatom"


def _make_runner() -> WorkflowRunner:
    """A WorkflowRunner whose collaborators are never reached by these tests."""
    return WorkflowRunner(executor=object(), config=object())


def _make_executor(tmp_path: Path) -> TaskExecutor:
    return TaskExecutor(tool_registry={}, run_id="run_test", run_path=tmp_path / "run")


# ---------------------------------------------------------------------------
# (a) atomic write — tmp-then-replace, no direct open(path, 'w') writers left
# ---------------------------------------------------------------------------

@pytest.mark.unit
def test_save_workflow_state_writes_tmp_then_replaces(tmp_path, monkeypatch):
    """_save_workflow_state must route through temp-file + atomic rename."""
    renames = []
    real_rename = state_manager_mod.os.rename

    def recording_rename(src, dst):
        renames.append((str(src), str(dst)))
        return real_rename(src, dst)

    monkeypatch.setattr(state_manager_mod.os, "rename", recording_rename)

    path = tmp_path / "workflows" / f"{WF_ID}.json"
    _make_runner()._save_workflow_state(path, {"id": WF_ID, "status": "RUNNING"})

    assert len(renames) == 1, "expected exactly one atomic rename"
    src, dst = renames[0]
    assert dst == str(path)
    assert src != dst and src.endswith(".tmp")
    # Temp file lives in the SAME directory (same-filesystem rename atomicity).
    assert Path(src).parent == path.parent
    # And the result parses with the payload intact (updated_at stamped).
    loaded = json.loads(path.read_text())
    assert loaded["id"] == WF_ID
    assert "updated_at" in loaded


@pytest.mark.unit
def test_update_task_status_writes_tmp_then_replaces(tmp_path, monkeypatch):
    """executor._update_task_status must route through temp-file + rename."""
    monkeypatch.setattr(executor_mod, "STATE_PATH", tmp_path)
    wf_path = tmp_path / "workflows" / f"{WF_ID}.json"
    wf_path.parent.mkdir(parents=True)
    wf_path.write_text(json.dumps({
        "id": WF_ID,
        "tasks": [{"id": "T1", "status": "PENDING"}],
        "progress": {},
    }))

    renames = []
    real_rename = state_manager_mod.os.rename

    def recording_rename(src, dst):
        renames.append((str(src), str(dst)))
        return real_rename(src, dst)

    monkeypatch.setattr(state_manager_mod.os, "rename", recording_rename)

    ok = _make_executor(tmp_path)._update_task_status(WF_ID, "T1", "COMPLETE")

    assert ok is True
    assert len(renames) == 1
    src, dst = renames[0]
    assert dst == str(wf_path) and src.endswith(".tmp")
    loaded = json.loads(wf_path.read_text())
    assert loaded["tasks"][0]["status"] == "COMPLETE"
    assert loaded["progress"]["completed"] == 1


@pytest.mark.unit
def test_no_direct_write_pattern_remains_in_writer_functions():
    """Grep the writer function sources: the old open(path, 'w') is gone."""
    direct_write = re.compile(r"""open\(\s*(workflow_)?path\s*,\s*['"]w['"]""")
    for fn in (
        WorkflowRunner._save_workflow_state,
        TaskExecutor._update_task_status,
        TaskExecutor._update_task_status_locked,
    ):
        src = inspect.getsource(fn)
        assert not direct_write.search(src), (
            f"{fn.__qualname__} still contains a direct non-atomic "
            f"open(..., 'w') write of the workflow-state file"
        )
        assert "atomic_write_json" in src or "_update_task_status_locked" in src


# ---------------------------------------------------------------------------
# (b) concurrent-writer simulation — the file can never be observed corrupt
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_concurrent_save_cycles_never_corrupt(tmp_path):
    """Two threads doing save cycles on the same path: every observed state
    parses. This FAILS on the pre-fix open('w')+dump interleaving writer."""
    path = tmp_path / "workflows" / f"{WF_ID}.json"
    runner = _make_runner()
    n_cycles = 50
    # Deliberately different-sized payloads so an interleave of partial
    # writes (the incident shape: one doc + a second doc's head) is
    # detectable as a parse failure.
    payload_small = {"id": WF_ID, "status": "RUNNING", "tasks": []}
    payload_large = {
        "id": WF_ID,
        "status": "RUNNING",
        "tasks": [
            {"id": f"T{i}", "status": "PENDING", "note": "x" * 200}
            for i in range(40)
        ],
    }
    errors: list = []
    done = threading.Event()

    def writer(payload):
        try:
            for _ in range(n_cycles):
                runner._save_workflow_state(path, dict(payload))
        except Exception as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    parse_failures = []

    def reader():
        while not done.is_set():
            if path.exists():
                try:
                    json.loads(path.read_text())
                except (json.JSONDecodeError, OSError) as exc:
                    parse_failures.append(str(exc))

    threads = [
        threading.Thread(target=writer, args=(payload_small,)),
        threading.Thread(target=writer, args=(payload_large,)),
    ]
    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    done.set()
    reader_thread.join()

    assert errors == []
    assert parse_failures == [], (
        f"observed {len(parse_failures)} corrupt mid-race states; first: "
        f"{parse_failures[0]}"
    )
    # Final file always parses and is one of the two complete documents.
    final = json.loads(path.read_text())
    assert final["id"] == WF_ID
    assert len(final["tasks"]) in (0, 40)


@pytest.mark.integration
def test_concurrent_task_status_updates_all_land(tmp_path, monkeypatch):
    """Two threads updating DIFFERENT tasks: flock-serialized read-modify-
    write means neither update is lost and the file never corrupts."""
    monkeypatch.setattr(executor_mod, "STATE_PATH", tmp_path)
    wf_path = tmp_path / "workflows" / f"{WF_ID}.json"
    wf_path.parent.mkdir(parents=True)
    wf_path.write_text(json.dumps({
        "id": WF_ID,
        "tasks": [
            {"id": "T1", "status": "PENDING"},
            {"id": "T2", "status": "PENDING"},
        ],
        "progress": {},
    }))
    executor = _make_executor(tmp_path)
    errors: list = []

    def flip(task_id):
        try:
            for _ in range(25):
                assert executor._update_task_status(WF_ID, task_id, "IN_PROGRESS")
                assert executor._update_task_status(WF_ID, task_id, "COMPLETE")
        except Exception as exc:  # pragma: no cover - failure reporting
            errors.append(exc)

    threads = [
        threading.Thread(target=flip, args=("T1",)),
        threading.Thread(target=flip, args=("T2",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    final = json.loads(wf_path.read_text())
    statuses = {t["id"]: t["status"] for t in final["tasks"]}
    assert statuses == {"T1": "COMPLETE", "T2": "COMPLETE"}
    assert final["progress"]["completed"] == 2


# ---------------------------------------------------------------------------
# (c) corrupted-file read raises the enriched error
# ---------------------------------------------------------------------------

def _write_incident_shape_corruption(path: Path) -> None:
    """One nearly-complete document plus the head of a second appended —
    the exact shape the 2026-07-21 interleave left on disk."""
    doc = json.dumps({"id": WF_ID, "type": "textbook_to_course",
                      "params": {}, "tasks": []}, indent=2)
    second_head = json.dumps(
        {"id": WF_ID, "type": "textbook_to_course", "status": "RUNNING",
         "params": {"course_name": "X" * 400}}, indent=2
    )[:487]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc + second_head)


@pytest.mark.unit
def test_run_workflow_corrupted_state_raises_enriched_error(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "STATE_PATH", tmp_path)
    wf_path = tmp_path / "workflows" / f"{WF_ID}.json"
    _write_incident_shape_corruption(wf_path)

    with pytest.raises(StateFileCorruptedError) as excinfo:
        asyncio.run(_make_runner().run_workflow(WF_ID))

    msg = str(excinfo.value)
    assert str(wf_path) in msg                        # names the file
    assert re.search(r"char \d+", msg)                # names the position
    assert ".tmp sibling" in msg                      # recovery hint 1
    assert "runtime/state/runs/<run_id>/checkpoints/" in msg  # recovery hint 2
    # Chained from the underlying decode error, not swallowing it.
    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


@pytest.mark.unit
def test_orchestrator_load_corrupted_state_raises_enriched_error(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(orch_mod, "STATE_PATH", tmp_path)
    wf_path = tmp_path / "workflows" / f"{WF_ID}.json"
    _write_incident_shape_corruption(wf_path)

    with pytest.raises(StateFileCorruptedError) as excinfo:
        orch_mod.PipelineOrchestrator._load_workflow_state(object(), WF_ID)

    msg = str(excinfo.value)
    assert str(wf_path) in msg
    assert "checkpoints" in msg
