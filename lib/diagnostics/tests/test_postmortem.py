"""Tests for the forensic run post-mortem :mod:`lib.diagnostics.postmortem`.

Covers the redesigned OOM-inference + fail-closed contract:

* missing run dir → a single FAIL not-found (no raise);
* a failed phase → a FAIL naming the phase + error;
* **OOM honesty** — a failed phase at the NORMAL 8 GB free-VRAM steady state
  (~150 MiB) does NOT yield a 'LIKELY OOM' FAIL; it emits the phase-failed FAIL
  plus honest free-VRAM CONTEXT (INFO, no OOM assertion);
* **confirmed OOM is the only FAIL-level OOM claim** — read word-awarely from a
  failed checkpoint's error ('CUDA out of memory'), scanned across ALL failed
  phases (not just the last), never matching 'bloom'/'roommate';
* **fail-closed** — an empty / all-unreadable checkpoint set is INDETERMINATE
  (non-zero exit), never a clean "completed all phases";
* all-completed checkpoints → an honest "recorded, not terminal" summary;
* dropped (corrupt) checkpoint files → a 'N skipped' WARN;
* an unreadable trajectory file → the 'exists but unreadable' WARN (not the
  'doctor was off' INFO);
* a 'started' last phase → the ambiguous WARN (not a definitive crash FAIL);
* malformed checkpoint JSON / bad JSONL line → skipped, never raises;
* ``ctx.run_id is None`` → the defensive WARN.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lib.diagnostics.core import CheckContext, Severity, resolve_exit_code
from lib.diagnostics.postmortem import postmortem_checks


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #


@pytest.fixture
def runs_root(tmp_path, monkeypatch):
    """Isolate state/runs under tmp_path via ED4ALL_STATE_RUNS_DIR."""
    root = tmp_path / "runs"
    root.mkdir()
    monkeypatch.setenv("ED4ALL_STATE_RUNS_DIR", str(root))
    return root


def _write_checkpoint(run_dir: Path, phase_name: str, phase_index: int, **kw) -> None:
    """Write a ``<run>/checkpoints/<phase>_checkpoint.json`` (PhaseCheckpoint shape)."""
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "run_id": run_dir.name,
        "workflow_id": "wf",
        "phase_name": phase_name,
        "phase_index": phase_index,
        "status": "completed",
        "started_at": "2026-06-26T00:00:00",
        "completed_at": None,
        "tasks_completed": [],
        "tasks_failed": [],
        "tasks_pending": [],
        "last_event_seq": 0,
        "artifacts_produced": [],
        "validation_results": {},
        "error": None,
        "error_details": None,
    }
    doc.update(kw)
    (ckpt_dir / f"{phase_name}_checkpoint.json").write_text(json.dumps(doc))


def _write_trajectory(run_dir: Path, rows: list) -> None:
    """Write ``<run>/vram_trajectory.jsonl``."""
    lines = "\n".join(json.dumps(r) for r in rows) + "\n"
    (run_dir / "vram_trajectory.jsonl").write_text(lines)


def _by_name(results, name):
    return [r for r in results if r.name == name]


def _severities(results):
    return {r.severity for r in results}


# --------------------------------------------------------------------------- #
# Defensive / not-found
# --------------------------------------------------------------------------- #


def test_no_run_id_is_defensive_warn(runs_root):
    results = postmortem_checks(CheckContext(run_id=None))
    assert len(results) == 1
    assert results[0].severity is Severity.WARN
    assert "no run_id" in results[0].summary.lower()


def test_missing_run_dir_fails_not_found(runs_root):
    results = postmortem_checks(CheckContext(run_id="does-not-exist"))
    assert len(results) == 1
    r = results[0]
    assert r.severity is Severity.FAIL
    assert "not found" in r.summary
    assert "does-not-exist" in r.summary
    assert r.remediation  # carries the where-runs-live hint
    # exit code reflects the FAIL.
    assert resolve_exit_code(results) == 2


# --------------------------------------------------------------------------- #
# Failed phases
# --------------------------------------------------------------------------- #


def test_failed_phase_names_phase_and_error(runs_root):
    run_dir = runs_root / "run-fail"
    run_dir.mkdir()
    _write_checkpoint(run_dir, "dart_conversion", 0, status="completed")
    _write_checkpoint(
        run_dir,
        "concept_extraction",
        1,
        status="failed",
        error="validation gate blocked the phase",
        tasks_failed=["t1", "t2"],
    )

    results = postmortem_checks(CheckContext(run_id="run-fail"))

    phase = _by_name(results, "phase_concept_extraction")
    assert phase, "expected a result for the failed phase"
    r = phase[0]
    assert r.severity is Severity.FAIL
    assert "concept_extraction" in r.summary
    assert "validation gate blocked" in r.summary
    assert r.data["tasks_failed"] == ["t1", "t2"]
    assert resolve_exit_code(results) == 2
    # A non-OOM validation failure must NOT yield a confirmed-OOM FAIL.
    assert not _by_name(results, "oom_correlation")


# --------------------------------------------------------------------------- #
# OOM honesty — findings #4 / #5 / #6 (the false-positive guard)
# --------------------------------------------------------------------------- #


def test_failed_phase_at_normal_free_vram_is_not_likely_oom(runs_root):
    """A FAILED phase at the NORMAL ~150 MiB 8 GB steady state must NOT be
    labelled a LIKELY-OOM FAIL — it gets the phase-failed FAIL + honest VRAM
    context only (no OOM assertion). Inverts the old 'fabricated low after row
    asserts LIKELY OOM' test."""
    run_dir = runs_root / "run-normal-vram"
    run_dir.mkdir()
    _write_checkpoint(
        run_dir,
        "concept_extraction",
        0,
        status="failed",
        error="Task content_generation failed",
    )
    _write_trajectory(
        run_dir,
        [
            {"phase": "concept_extraction", "when": "before", "free_mib": 200,
             "resident_models": [{"name": "qwen2.5:7b", "vram_mib": 5300}],
             "cuda_available": True},
            {"phase": "concept_extraction", "when": "after", "free_mib": 150,
             "resident_models": [{"name": "qwen2.5:7b", "vram_mib": 5300}],
             "cuda_available": True},
        ],
    )

    results = postmortem_checks(CheckContext(run_id="run-normal-vram"))

    # No LIKELY-OOM FAIL anywhere.
    for r in results:
        assert "LIKELY OOM" not in r.summary
    # No confirmed-OOM result (the error doesn't name OOM).
    assert not _by_name(results, "oom_correlation")

    # The phase failure is the FAIL.
    phase = _by_name(results, "phase_concept_extraction")
    assert phase and phase[0].severity is Severity.FAIL

    # Honest VRAM context: INFO, shows 150 MiB, with the 'not OOM evidence' caveat.
    ctx = _by_name(results, "vram_context")
    assert ctx, "expected a free-VRAM context result"
    c = ctx[0]
    assert c.severity is Severity.INFO
    assert "150 MiB" in c.summary
    assert "NOT OOM evidence" in c.summary
    assert c.data["oom"] == "not_asserted"
    # The crash phase still drives exit 2 via its FAIL — but NOT via an OOM claim.
    assert resolve_exit_code(results) == 2


def test_started_phase_with_low_vram_is_context_not_oom_fail(runs_root):
    """A 'started' (incomplete) crash phase with a cratered free-VRAM sample is
    still only INFO context — never a FAIL OOM verdict (#6: a real OOM often
    never writes the after row, so a low sample is not proof)."""
    run_dir = runs_root / "run-started-lowvram"
    run_dir.mkdir()
    _write_checkpoint(run_dir, "dart_conversion", 0, status="completed")
    _write_checkpoint(
        run_dir,
        "concept_extraction",
        1,
        status="started",
        error="phase did not return",
    )
    _write_trajectory(
        run_dir,
        [
            {"phase": "concept_extraction", "when": "before", "free_mib": 150,
             "resident_models": [{"name": "qwen2.5:7b", "vram_mib": 5300}],
             "cuda_available": True},
        ],
    )

    results = postmortem_checks(CheckContext(run_id="run-started-lowvram"))

    ctx = _by_name(results, "vram_context")
    assert ctx and ctx[0].severity is Severity.INFO
    assert ctx[0].data["oom"] == "not_asserted"
    # The 'started' phase is an ambiguous WARN, not a definitive crash FAIL.
    crash = _by_name(results, "phase_concept_extraction")
    assert crash and crash[0].severity is Severity.WARN


# --------------------------------------------------------------------------- #
# Confirmed OOM — findings #8 (word boundary) / #9 (all failed phases)
# --------------------------------------------------------------------------- #


def test_confirmed_oom_from_error_string(runs_root):
    run_dir = runs_root / "run-confirmed"
    run_dir.mkdir()
    _write_checkpoint(
        run_dir,
        "concept_extraction",
        0,
        status="failed",
        error="RuntimeError: CUDA out of memory. Tried to allocate 512 MiB",
    )

    results = postmortem_checks(CheckContext(run_id="run-confirmed"))

    corr = _by_name(results, "oom_correlation")
    assert corr, "expected a confirmed-OOM result"
    r = corr[0]
    assert r.severity is Severity.FAIL
    assert "confirmed OOM" in r.summary
    assert "concept_extraction" in r.summary
    assert r.data["oom"] == "confirmed"
    assert resolve_exit_code(results) == 2


def test_confirmed_oom_scans_non_last_failed_phase(runs_root):
    """An OOM in an EARLIER failed phase (not the last crash phase) is still
    confirmed (#9)."""
    run_dir = runs_root / "run-early-oom"
    run_dir.mkdir()
    _write_checkpoint(
        run_dir,
        "objective_extraction",
        0,
        status="failed",
        error="torch.cuda.OutOfMemoryError: CUDA out of memory",
    )
    # A LATER failed phase whose error does NOT name OOM.
    _write_checkpoint(
        run_dir,
        "content_generation",
        1,
        status="failed",
        error="downstream validation gate blocked",
    )

    results = postmortem_checks(CheckContext(run_id="run-early-oom"))

    corr = _by_name(results, "oom_correlation")
    assert corr, "expected a confirmed-OOM result from the earlier phase"
    r = corr[0]
    assert r.severity is Severity.FAIL
    assert "confirmed OOM" in r.summary
    # The earlier failed phase is named.
    assert "objective_extraction" in r.summary
    assert "objective_extraction" in r.data["phases"]


def test_oom_word_boundary_does_not_match_bloom_or_roommate(runs_root):
    """'bloom' / 'roommate' must NOT trip the word-aware 'oom' token (#8)."""
    run_dir = runs_root / "run-bloom"
    run_dir.mkdir()
    _write_checkpoint(
        run_dir,
        "concept_extraction",
        0,
        status="failed",
        error="Bloom taxonomy alignment failed for a roommate scenario block",
    )

    results = postmortem_checks(CheckContext(run_id="run-bloom"))

    # No confirmed OOM — the substring 'oom' inside bloom/roommate is excluded.
    assert not _by_name(results, "oom_correlation")
    # Still a real phase failure.
    phase = _by_name(results, "phase_concept_extraction")
    assert phase and phase[0].severity is Severity.FAIL


# --------------------------------------------------------------------------- #
# Fail-closed — finding #1 (indeterminate)
# --------------------------------------------------------------------------- #


def test_empty_checkpoint_set_is_indeterminate_not_ok(runs_root):
    """No checkpoints at all → INDETERMINATE (non-zero exit), never a clean
    'completed all phases' / exit 0 (#1)."""
    run_dir = runs_root / "run-empty"
    run_dir.mkdir()  # exists, but no checkpoints/ contents

    results = postmortem_checks(CheckContext(run_id="run-empty"))

    # Not a phantom success.
    assert not _by_name(results, "postmortem_summary")
    indet = _by_name(results, "postmortem_indeterminate")
    assert indet, "expected an indeterminate result"
    assert indet[0].severity in (Severity.WARN, Severity.FAIL)
    assert "indeterminate" in indet[0].summary.lower()
    # Non-zero exit — anti-silent-degradation.
    assert resolve_exit_code(results) != 0


def test_all_corrupt_checkpoints_indeterminate_but_trajectory_surfaced(runs_root):
    """All checkpoint files unreadable → indeterminate, but a VRAM trajectory
    crater is still surfaced (#1)."""
    run_dir = runs_root / "run-allcorrupt"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "a_checkpoint.json").write_text("{not valid")
    (ckpt_dir / "b_checkpoint.json").write_text("also broken")
    _write_trajectory(
        run_dir,
        [
            {"phase": "concept_extraction", "when": "before", "free_mib": 4000,
             "resident_models": []},
            {"phase": "concept_extraction", "when": "after", "free_mib": 120,
             "resident_models": [{"name": "qwen2.5:7b", "vram_mib": 5300}]},
        ],
    )

    results = postmortem_checks(CheckContext(run_id="run-allcorrupt"))

    # Indeterminate (not a success).
    assert not _by_name(results, "postmortem_summary")
    assert _by_name(results, "postmortem_indeterminate")
    # Dropped-evidence WARN counts the corrupt files.
    dropped = _by_name(results, "postmortem_dropped_checkpoints")
    assert dropped and dropped[0].data["skipped"] == 2
    # The trajectory timeline still surfaces the crater.
    tl = _by_name(results, "vram_timeline")
    assert tl and tl[0].data["trajectory_present"] is True
    assert "concept_extraction" in tl[0].summary
    assert resolve_exit_code(results) != 0


# --------------------------------------------------------------------------- #
# Finding #10 — dropped evidence visible
# --------------------------------------------------------------------------- #


def test_corrupt_checkpoint_alongside_good_warns_skipped(runs_root):
    run_dir = runs_root / "run-onecorrupt"
    run_dir.mkdir()
    _write_checkpoint(run_dir, "good_phase", 0, status="completed")
    ckpt_dir = run_dir / "checkpoints"
    (ckpt_dir / "broken_checkpoint.json").write_text("{not valid json")

    results = postmortem_checks(CheckContext(run_id="run-onecorrupt"))

    dropped = _by_name(results, "postmortem_dropped_checkpoints")
    assert dropped, "expected a dropped-evidence WARN"
    r = dropped[0]
    assert r.severity is Severity.WARN
    assert "1 checkpoint file" in r.summary
    assert r.data["skipped"] == 1
    # Dropped evidence makes the run non-clean.
    assert resolve_exit_code(results) != 0


# --------------------------------------------------------------------------- #
# Finding #3 — trajectory absent vs unreadable
# --------------------------------------------------------------------------- #


def test_no_trajectory_file_info_and_phases_analyzed(runs_root):
    run_dir = runs_root / "run-notraj"
    run_dir.mkdir()
    _write_checkpoint(
        run_dir,
        "concept_extraction",
        0,
        status="failed",
        error="some non-OOM failure",
    )
    # No vram_trajectory.jsonl written.

    results = postmortem_checks(CheckContext(run_id="run-notraj"))

    tl = _by_name(results, "vram_timeline")
    assert tl and tl[0].severity is Severity.INFO
    assert "no VRAM trajectory" in tl[0].summary
    assert tl[0].data["trajectory_present"] is False

    # Phase still analyzed as a FAIL.
    phase = _by_name(results, "phase_concept_extraction")
    assert phase and phase[0].severity is Severity.FAIL


def test_unreadable_trajectory_warns_not_doctor_off(runs_root):
    """A vram_trajectory.jsonl that EXISTS but cannot be read → the 'exists but
    unreadable' WARN, NOT the 'doctor was off' INFO (#3)."""
    run_dir = runs_root / "run-unreadable-traj"
    run_dir.mkdir()
    _write_checkpoint(run_dir, "good_phase", 0, status="completed")
    # Put a DIRECTORY where the trajectory file should be → open() raises.
    (run_dir / "vram_trajectory.jsonl").mkdir()

    results = postmortem_checks(CheckContext(run_id="run-unreadable-traj"))

    tl = _by_name(results, "vram_timeline")
    assert tl, "expected a timeline result"
    r = tl[0]
    assert r.severity is Severity.WARN
    assert "exists but could not be read" in r.summary
    assert "no VRAM trajectory" not in r.summary
    assert r.data["trajectory_present"] is False
    assert r.data["trajectory_status"] == "unreadable"
    assert resolve_exit_code(results) != 0


# --------------------------------------------------------------------------- #
# Finding #7 — 'started' phase ambiguity
# --------------------------------------------------------------------------- #


def test_started_last_phase_is_ambiguous_warn(runs_root):
    run_dir = runs_root / "run-started"
    run_dir.mkdir()
    _write_checkpoint(run_dir, "dart_conversion", 0, status="completed")
    _write_checkpoint(
        run_dir,
        "concept_extraction",
        1,
        status="started",
        error=None,
    )

    results = postmortem_checks(CheckContext(run_id="run-started"))

    phase = _by_name(results, "phase_concept_extraction")
    assert phase, "expected a result for the started phase"
    r = phase[0]
    assert r.severity is Severity.WARN  # not a definitive crash FAIL
    assert "started" in r.summary
    assert "still in progress" in r.summary
    # No success summary (a crash phase is present).
    assert not _by_name(results, "postmortem_summary")
    # WARN → exit 1 (non-zero) but not a hard FAIL.
    assert resolve_exit_code(results) == 1


# --------------------------------------------------------------------------- #
# Finding #2 — all-completed honesty
# --------------------------------------------------------------------------- #


def test_all_completed_is_honest_not_terminal_claim(runs_root):
    run_dir = runs_root / "run-clean"
    run_dir.mkdir()
    _write_checkpoint(run_dir, "dart_conversion", 0, status="completed")
    _write_checkpoint(run_dir, "staging", 1, status="completed")
    _write_checkpoint(run_dir, "packaging", 2, status="completed")

    results = postmortem_checks(CheckContext(run_id="run-clean"))

    summary = _by_name(results, "postmortem_summary")
    assert summary and summary[0].severity is Severity.OK
    s = summary[0].summary
    # Honest phrasing: recorded, not terminal.
    assert "recorded" in s
    assert "not that the run reached its terminal phase" in s
    assert summary[0].data["last_recorded_phase"] == "packaging"

    # No FAILs / WARNs → exit 0 for a genuinely clean recorded run.
    assert Severity.FAIL not in _severities(results)
    assert Severity.WARN not in _severities(results)
    assert resolve_exit_code(results) == 0


def test_all_completed_with_trajectory_gap_warns(runs_root):
    """An all-completed checkpoint set whose trajectory shows a phase with no
    checkpoint surfaces that gap as a WARN (#2 OS-OOM-kill-in-the-gap)."""
    run_dir = runs_root / "run-gap"
    run_dir.mkdir()
    _write_checkpoint(run_dir, "dart_conversion", 0, status="completed")
    _write_checkpoint(run_dir, "staging", 1, status="completed")
    _write_trajectory(
        run_dir,
        [
            {"phase": "dart_conversion", "when": "before", "free_mib": 6000,
             "resident_models": []},
            {"phase": "staging", "when": "before", "free_mib": 5800,
             "resident_models": []},
            # A 'before' row for a phase that never got a checkpoint.
            {"phase": "concept_extraction", "when": "before", "free_mib": 5500,
             "resident_models": []},
        ],
    )

    results = postmortem_checks(CheckContext(run_id="run-gap"))

    gap = _by_name(results, "postmortem_trajectory_gap")
    assert gap, "expected a trajectory-gap WARN"
    assert gap[0].severity is Severity.WARN
    assert "concept_extraction" in gap[0].summary
    assert resolve_exit_code(results) != 0


# --------------------------------------------------------------------------- #
# Robustness / never-raises
# --------------------------------------------------------------------------- #


def test_malformed_checkpoint_and_jsonl_are_skipped(runs_root):
    run_dir = runs_root / "run-malformed"
    run_dir.mkdir()
    _write_checkpoint(run_dir, "good_phase", 0, status="completed")
    # Malformed checkpoint JSON.
    ckpt_dir = run_dir / "checkpoints"
    (ckpt_dir / "broken_checkpoint.json").write_text("{not valid json")
    # Trajectory with one bad line + one good.
    (run_dir / "vram_trajectory.jsonl").write_text(
        "{bad line\n"
        + json.dumps({"phase": "good_phase", "when": "after", "free_mib": 4000,
                      "resident_models": []})
        + "\n"
    )

    # Must not raise.
    results = postmortem_checks(CheckContext(run_id="run-malformed"))
    assert results  # produced something
    # The good phase survived.
    assert _by_name(results, "phase_good_phase")
    # The corrupt one is surfaced as dropped evidence.
    assert _by_name(results, "postmortem_dropped_checkpoints")
    # Timeline parsed the good line.
    tl = _by_name(results, "vram_timeline")
    assert tl and tl[0].data["trajectory_present"] is True


def test_register_adds_to_registry():
    from lib.diagnostics import core
    from lib.diagnostics.postmortem import postmortem_checks, register_postmortem_checks

    core.clear_registry()
    try:
        register_postmortem_checks()
        pairs = core.registered_checks()
        assert ("postmortem", postmortem_checks) in pairs
    finally:
        core.clear_registry()
