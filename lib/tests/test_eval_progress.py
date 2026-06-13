"""Tests for the best-effort incremental eval progress writer
(``lib.retrieval.eval_progress.EvalProgressWriter``).

The writer is ADVISORY: it lays down run-scoped scratch files
(``eval_progress.jsonl`` + ``eval_progress.json``) so a long eval can be tailed,
but a write failure must NEVER raise into the eval. These tests pin:

  * one jsonl line appended per recorded item (flushed to disk);
  * the snapshot atomically rewritten with correct done / total / percent / eta;
  * run-start truncation (a new writer overwrites a stale jsonl);
  * best-effort swallow of a write error (patched ``open`` → no exception);
  * the ``close()`` finalize state (complete / failed) + context-manager use.
"""
from __future__ import annotations

import builtins
import json

import pytest

from lib.retrieval.eval_progress import (
    PROGRESS_JSONL_FILENAME,
    PROGRESS_SNAPSHOT_FILENAME,
    EvalProgressWriter,
)


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _snapshot(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_files_land_with_distinct_names_in_eval_dir(tmp_path):
    w = EvalProgressWriter(
        tmp_path, run_id="RUN-1", arm_totals={"base": 2}
    )
    assert w.jsonl_path == tmp_path / PROGRESS_JSONL_FILENAME
    assert w.snapshot_path == tmp_path / PROGRESS_SNAPSHOT_FILENAME
    # Distinct from any canonical artifact name.
    assert "scorecard" not in PROGRESS_JSONL_FILENAME
    assert "grounded_answer_eval" not in PROGRESS_SNAPSHOT_FILENAME
    # An initial snapshot exists before any item (planned totals visible).
    snap = _snapshot(w.snapshot_path)
    assert snap["run_id"] == "RUN-1"
    assert snap["state"] == "running"
    assert snap["arms"]["base"]["total"] == 2
    assert snap["overall"]["total"] == 2
    assert snap["overall"]["done"] == 0
    # ETA / rate are None until ≥1 item is done (guard div-by-zero).
    assert snap["overall"]["items_per_s"] is None
    assert snap["overall"]["eta_s"] is None
    assert snap["overall"]["percent"] in (0.0, None)


def test_one_jsonl_line_per_item_with_flush(tmp_path):
    w = EvalProgressWriter(
        tmp_path, run_id="RUN-2", arm_totals={"base": 2, "retrieval": 1}
    )
    w.start_arm("base", 2)
    w.record_item("base", "gq-1", "answered", answered=True,
                  key_points_total=2, key_points_covered=1)
    # File is visible immediately (flush + fsync) — read it back NOW.
    lines = _read_jsonl(w.jsonl_path)
    assert len(lines) == 1
    assert lines[0]["arm"] == "base"
    assert lines[0]["item_id"] == "gq-1"
    assert lines[0]["index"] == 1
    assert lines[0]["status"] == "answered"
    assert lines[0]["overall_total"] == 3
    assert lines[0]["overall_done"] == 1
    assert lines[0]["answered_so_far"] == 1
    assert lines[0]["key_point_coverage_so_far"] == pytest.approx(0.5)

    w.record_item("base", "gq-2", "answered", answered=True,
                  key_points_total=2, key_points_covered=2)
    w.record_item("retrieval", "gq-1", "retrieved",
                  key_points_total=2, key_points_covered=2)
    lines = _read_jsonl(w.jsonl_path)
    assert len(lines) == 3
    assert [l["arm"] for l in lines] == ["base", "base", "retrieval"]
    # Per-arm index resets; overall_done is monotonic.
    assert [l["index"] for l in lines] == [1, 2, 1]
    assert [l["overall_done"] for l in lines] == [1, 2, 3]


def test_snapshot_done_total_percent_and_eta(tmp_path):
    w = EvalProgressWriter(tmp_path, run_id="RUN-3", arm_totals={"base": 4})
    w.start_arm("base", 4)
    w.record_item("base", "gq-1", "answered", answered=True)
    snap = _snapshot(w.snapshot_path)
    assert snap["overall"]["done"] == 1
    assert snap["overall"]["total"] == 4
    assert snap["overall"]["percent"] == pytest.approx(25.0)
    # Rate + ETA become available after ≥1 item (guarded, may be None if the
    # clock granularity made elapsed 0 — assert the type contract instead).
    assert snap["overall"]["items_per_s"] is None or snap[
        "overall"
    ]["items_per_s"] > 0
    assert snap["overall"]["eta_s"] is None or snap["overall"]["eta_s"] >= 0
    # Per-arm rollup tracks done.
    assert snap["arms"]["base"]["done"] == 1
    assert snap["arms"]["base"]["answered"] == 1


def test_start_arm_refines_total(tmp_path):
    # Constructed with a coarse estimate; start_arm corrects it.
    w = EvalProgressWriter(tmp_path, run_id="RUN-4", arm_totals={"base": 10})
    assert _snapshot(w.snapshot_path)["overall"]["total"] == 10
    w.start_arm("base", 3)
    snap = _snapshot(w.snapshot_path)
    assert snap["arms"]["base"]["total"] == 3
    assert snap["overall"]["total"] == 3
    assert snap["arms"]["base"]["started"] is True


def test_finish_arm_and_close_complete(tmp_path):
    w = EvalProgressWriter(tmp_path, run_id="RUN-5", arm_totals={"base": 1})
    w.record_item("base", "gq-1", "answered", answered=True)
    w.finish_arm("base")
    assert _snapshot(w.snapshot_path)["arms"]["base"]["finished"] is True
    w.close()
    snap = _snapshot(w.snapshot_path)
    assert snap["state"] == "complete"
    # Idempotent — a second close is a no-op (still complete).
    w.close()
    assert _snapshot(w.snapshot_path)["state"] == "complete"


def test_close_failed_marks_failed_state(tmp_path):
    w = EvalProgressWriter(tmp_path, run_id="RUN-6", arm_totals={"base": 1})
    w.close(failed=True)
    assert _snapshot(w.snapshot_path)["state"] == "failed"


def test_context_manager_marks_failed_on_exception(tmp_path):
    with pytest.raises(ValueError):
        with EvalProgressWriter(
            tmp_path, run_id="RUN-7", arm_totals={"base": 1}
        ) as w:
            w.record_item("base", "gq-1", "answered", answered=True)
            raise ValueError("boom")
    # The body's exception is NOT suppressed, and the snapshot is "failed".
    assert _snapshot(tmp_path / PROGRESS_SNAPSHOT_FILENAME)["state"] == "failed"


def test_run_start_truncates_stale_files(tmp_path):
    # Pre-seed a stale jsonl + snapshot from a "prior run".
    (tmp_path / PROGRESS_JSONL_FILENAME).write_text(
        '{"stale": true}\n{"stale": 2}\n', encoding="utf-8"
    )
    (tmp_path / PROGRESS_SNAPSHOT_FILENAME).write_text(
        '{"run_id": "OLD", "state": "complete"}', encoding="utf-8"
    )
    # A fresh writer must overwrite both at run start (no append onto stale).
    w = EvalProgressWriter(tmp_path, run_id="RUN-NEW", arm_totals={"base": 1})
    assert _read_jsonl(w.jsonl_path) == []  # jsonl truncated
    snap = _snapshot(w.snapshot_path)
    assert snap["run_id"] == "RUN-NEW"  # snapshot overwritten
    assert snap["state"] == "running"
    # After one item only the NEW run's line is present.
    w.record_item("base", "gq-1", "answered", answered=True)
    lines = _read_jsonl(w.jsonl_path)
    assert len(lines) == 1
    assert lines[0]["run_id"] == "RUN-NEW"


def test_write_error_is_swallowed_not_raised(tmp_path, monkeypatch):
    """A write failure must never raise into the eval (best-effort)."""
    w = EvalProgressWriter(tmp_path, run_id="RUN-8", arm_totals={"base": 2})
    real_open = builtins.open

    def _boom(path, *args, **kwargs):
        # Fail only for the progress files; let everything else through.
        if str(path).endswith((PROGRESS_JSONL_FILENAME, ".json.tmp")):
            raise OSError("disk full")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", _boom)
    # None of these may raise despite the IO error.
    w.start_arm("base", 2)
    w.record_item("base", "gq-1", "answered", answered=True)
    w.record_item("base", "gq-2", "error")
    w.finish_arm("base")
    w.close()
    # The writer kept counting in-memory even though disk writes failed.
    # (No assertion on file contents — the point is no exception escaped.)
