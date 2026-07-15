"""Wave E / plan P6 — auto-reuse of completed conversions on a paused resume.

``dart_conversion`` is one SemantiK task per PDF (``max_concurrent: 1``, no
per-unit sidecar — a chapter is either fully converted or not). A graceful stop
mid-phase leaves the finished chapters' artifacts on disk. On ``--resume`` the
runner must AUTO-ENABLE per-PDF conversion reuse for the finished chapters so
the run does not re-convert (hours of 7B/OCR work) what already succeeded —
gated on a paused/partial resume (never a fresh run) and on the completeness
signal BOTH ``{stem}_accessible.html`` AND ``{stem}_accessible.quality.json``.

Pins ``WorkflowRunner._create_phase_tasks`` / ``_resume_reusable_conversion_stems``:

  * fresh run (no prior dart_conversion phase_output) → NO reuse flag on any
    PDF, even if a stale artifact exists (self-gated against stale reuse).
  * paused resume (phase_output present, not ``_completed``): a PDF with BOTH
    artifacts gets ``reuse_conversion=True`` (skipped); a PDF missing the
    quality sidecar (torn write) or the HTML is re-converted (no flag).
  * a ``_completed`` dart_conversion phase_output → no reuse (defensive; the
    phase would be skipped anyway).

CPU-only, stdlib-only, no models, synthetic tmp artifacts (``SEMANTIK_PATH``
monkeypatched to a tmp dir — never the repo ``SemantiK/output``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import MagicMock

from MCP.core import workflow_runner as wr_mod
from MCP.core.config import WorkflowConfig, WorkflowPhase
from MCP.core.workflow_runner import WorkflowRunner


def _runner():
    return WorkflowRunner(
        executor=MagicMock(),
        config=WorkflowConfig(description="t", phases=[]),
    )


def _dart_phase():
    return WorkflowPhase(name="dart_conversion", agents=["dart-converter"])


def _write_complete_conversion(conv_dir: Path, stem: str, *, quality: bool = True):
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / f"{stem}_accessible.html").write_text("<html></html>", "utf-8")
    if quality:
        (conv_dir / f"{stem}_accessible.quality.json").write_text("{}", "utf-8")


def _reuse_flags(tasks):
    """Map pdf stem -> whether the task carries reuse_conversion=True."""
    out = {}
    for t in tasks:
        stem = Path(t["params"]["pdf_path"]).stem
        out[stem] = bool(t["params"].get("reuse_conversion"))
    return out


def test_fresh_run_no_reuse_even_with_stale_artifact(monkeypatch, tmp_path):
    conv_dir = tmp_path / "SemantiK" / "output"
    monkeypatch.setattr(wr_mod, "SEMANTIK_PATH", tmp_path / "SemantiK")
    # A stale complete artifact from an unrelated prior run.
    _write_complete_conversion(conv_dir, "ch01")

    tasks = _runner()._create_phase_tasks(
        "WF-1",
        _dart_phase(),
        routed_params={},
        workflow_params={"pdf_paths": ["a/ch01.pdf", "a/ch02.pdf"]},
        phase_outputs={},  # fresh run: no dart_conversion entry
    )
    assert _reuse_flags(tasks) == {"ch01": False, "ch02": False}


def test_paused_resume_reuses_complete_reconverts_incomplete(monkeypatch, tmp_path):
    conv_dir = tmp_path / "SemantiK" / "output"
    monkeypatch.setattr(wr_mod, "SEMANTIK_PATH", tmp_path / "SemantiK")
    _write_complete_conversion(conv_dir, "ch01")                 # complete → reuse
    _write_complete_conversion(conv_dir, "ch02", quality=False)  # torn → reconvert
    # ch03 has no artifacts at all → reconvert

    tasks = _runner()._create_phase_tasks(
        "WF-1",
        _dart_phase(),
        routed_params={},
        workflow_params={
            "pdf_paths": ["a/ch01.pdf", "a/ch02.pdf", "a/ch03.pdf"]
        },
        phase_outputs={"dart_conversion": {"_paused": True, "_completed": False}},
    )
    assert _reuse_flags(tasks) == {
        "ch01": True,
        "ch02": False,
        "ch03": False,
    }


def test_completed_phase_output_no_reuse(monkeypatch, tmp_path):
    conv_dir = tmp_path / "SemantiK" / "output"
    monkeypatch.setattr(wr_mod, "SEMANTIK_PATH", tmp_path / "SemantiK")
    _write_complete_conversion(conv_dir, "ch01")

    tasks = _runner()._create_phase_tasks(
        "WF-1",
        _dart_phase(),
        routed_params={},
        workflow_params={"pdf_paths": ["a/ch01.pdf"]},
        phase_outputs={"dart_conversion": {"_completed": True}},
    )
    assert _reuse_flags(tasks) == {"ch01": False}


def test_global_reuse_flag_still_wins_on_fresh_run(monkeypatch, tmp_path):
    """The explicit --reuse-conversion path is untouched: every PDF gets the
    flag regardless of the resume gate."""
    monkeypatch.setattr(wr_mod, "SEMANTIK_PATH", tmp_path / "SemantiK")

    tasks = _runner()._create_phase_tasks(
        "WF-1",
        _dart_phase(),
        routed_params={},
        workflow_params={
            "pdf_paths": ["a/ch01.pdf", "a/ch02.pdf"],
            "reuse_conversion": True,
        },
        phase_outputs={},
    )
    assert _reuse_flags(tasks) == {"ch01": True, "ch02": True}


def test_resume_reusable_stems_helper_direct(monkeypatch, tmp_path):
    conv_dir = tmp_path / "SemantiK" / "output"
    monkeypatch.setattr(wr_mod, "SEMANTIK_PATH", tmp_path / "SemantiK")
    _write_complete_conversion(conv_dir, "ch01")

    r = _runner()
    # Not a resume (no entry) → empty.
    assert r._resume_reusable_conversion_stems({}, ["a/ch01.pdf"]) == set()
    # Paused resume → the complete stem.
    assert r._resume_reusable_conversion_stems(
        {"dart_conversion": {"_paused": True}}, ["a/ch01.pdf", "a/ch09.pdf"]
    ) == {"ch01"}
    # None phase_outputs → empty (defensive).
    assert r._resume_reusable_conversion_stems(None, ["a/ch01.pdf"]) == set()
