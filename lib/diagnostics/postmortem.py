"""Forensic run post-mortem — the ``ed4all doctor --run-id`` check group.

Given a *past* ``run_id``, this check group reads what the run persisted to
disk (phase checkpoints + the optional VRAM trajectory) and produces a
forensic narrative: WHICH phase failed, with WHAT error, and — where the
evidence actually supports it — the cause.

The single most important contract here is **inference honesty**, and the
honesty bar is deliberately high for one claim in particular: *out of memory*.

* **Confirmed OOM is the ONLY FAIL-level OOM claim.** It is asserted strictly
  from a failed checkpoint's own flat ``error`` string naming the OOM
  (word-aware: ``"CUDA out of memory"`` / ``"OutOfMemoryError"`` / the bare
  token ``oom`` only on a word boundary — never inside ``bloom``/``roommate``).
  ALL failed phases are scanned, not just the last crash phase.
* **Free VRAM is honest CONTEXT, never an asserted OOM.** On the canonical
  shared 8 GB box ~150-200 MiB free is the NORMAL steady state while the local
  7B is resident, so low free VRAM ALONE is NOT evidence of OOM. Worse, a hard
  mid-phase OOM crash never even writes the post-phase trajectory row, so the
  cratered sample usually does not exist. The post-mortem therefore presents
  the observed free VRAM around the crash phase as INFO context with that
  caveat — it never escalates a low free-VRAM reading to a FAIL OOM verdict.

Anti-silent-degradation: the post-mortem fails CLOSED rather than reporting a
phantom success.

* An empty / all-unreadable checkpoint set is reported as *indeterminate*
  (≥ WARN), never as a clean exit-0 "completed all phases".
* An all-``completed`` checkpoint set is reported honestly — it confirms the
  RECORDED checkpoints, NOT that the run reached its terminal phase (an
  OS-OOM-kill in the gap before the next ``started`` checkpoint leaves only
  ``completed`` ones).
* Dropped evidence is surfaced: if fewer checkpoints loaded than there are
  ``*_checkpoint.json`` files on disk, the skipped/corrupted count is WARNed.

Design contract (inherited from the doctor foundation): a doctor that crashes
is worse than no doctor. :func:`postmortem_checks` NEVER raises — every disk
read, JSON parse, and correlation step is wrapped; a malformed checkpoint or
trajectory line is skipped, not fatal. Pure lib (no ``cli`` import); the
heavier deps (``MCP.hardening.checkpoint``, ``lib.paths``) are lazily imported
inside the function so a bare ``import lib.diagnostics.postmortem`` stays cheap.

Exit-code semantics fall out of :func:`lib.diagnostics.core.resolve_exit_code`:
a failed OR indeterminate run emits FAIL/WARN → non-zero exit; a genuinely
clean run emits only OK/INFO → exit 0. That is the desired semantic — the
post-mortem's exit reflects the OUTCOME (or our uncertainty about it) of the
run being analyzed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from lib.diagnostics.core import CheckContext, CheckResult, Severity, register

logger = logging.getLogger(__name__)

#: Word-aware OOM markers matched (case-insensitively) against a failed
#: checkpoint's flat ``error`` string. Only an explicit match here yields the
#: CONFIRMED-OOM FAIL — the run itself named the OOM. The bare token ``oom``
#: is matched ONLY on a word boundary (``\boom\b``) so ``bloom`` / ``roommate``
#: never trip it. ``torch.cuda.OutOfMemoryError`` is covered by the
#: ``outofmemoryerror`` alternative once lower-cased.
_OOM_ERROR_REGEX = re.compile(
    r"cuda out of memory|out of memory|outofmemoryerror|\boom\b",
    re.IGNORECASE,
)

#: Human caveat appended to the free-VRAM context so a low reading is never
#: mistaken for an OOM verdict.
_VRAM_CONTEXT_CAVEAT = (
    "NOTE on a shared 8GB box ~150 MiB free is normal with a resident model, "
    "so this alone is NOT OOM evidence — check the run log for the executor's "
    "loud 'GPU OUT OF MEMORY' line (the authoritative confirmed signal)."
)


# --------------------------------------------------------------------------- #
# Disk readers (best-effort, never raise)
# --------------------------------------------------------------------------- #


def _resolve_run_dir(run_id: str):
    """Resolve ``<state/runs>/<run_id>`` (honors ``ED4ALL_STATE_RUNS_DIR``).

    Returns a ``pathlib.Path`` or ``None`` on any failure. Reads the runs
    parent at CALL time via :func:`lib.paths.get_state_runs_dir` so a test's
    monkeypatched ``ED4ALL_STATE_RUNS_DIR`` is honored.
    """
    try:
        from lib.paths import get_state_runs_dir  # lazy

        return get_state_runs_dir() / run_id
    except Exception as exc:  # noqa: BLE001 — never crash the doctor
        logger.warning("postmortem: could not resolve run dir for %r: %s", run_id, exc)
        return None


def _load_checkpoints(run_dir) -> List[Any]:
    """Load every phase checkpoint for the run (sorted by phase index).

    Returns a list of ``PhaseCheckpoint`` objects (or ``[]`` on any failure).
    Best-effort — a missing checkpoints dir, an import failure, or a malformed
    checkpoint file all degrade to a shorter list, never an exception. A
    malformed checkpoint file is dropped by ``get_all_checkpoints`` (see
    :func:`_count_checkpoint_files` for surfacing that dropped evidence).
    """
    try:
        from MCP.hardening.checkpoint import CheckpointManager  # lazy

        manager = CheckpointManager(run_dir)
        return manager.get_all_checkpoints()
    except Exception as exc:  # noqa: BLE001
        logger.warning("postmortem: checkpoint load failed for %s: %s", run_dir, exc)
        return []


def _count_checkpoint_files(run_dir) -> int:
    """Count ``*_checkpoint.json`` files actually on disk in ``<run>/checkpoints/``.

    Globs the SAME pattern ``CheckpointManager.get_all_checkpoints`` globs, so
    comparing this against the loaded count reveals checkpoints that were
    silently dropped because they were unreadable/corrupted (finding #10).
    Returns ``0`` on any failure (never raises).
    """
    try:
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            return 0
        return sum(1 for _ in ckpt_dir.glob("*_checkpoint.json"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("postmortem: could not count checkpoint files for %s: %s", run_dir, exc)
        return 0


def _load_trajectory(run_dir) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Load ``<run>/vram_trajectory.jsonl`` → ``(status, rows)``.

    ``status`` distinguishes the two failure modes that the old None-return
    conflated (finding #3):

    * ``"absent"`` — the file does not exist (the trajectory was never written;
      ``ED4ALL_VRAM_DOCTOR`` was off for that run). ``rows`` is ``None``.
    * ``"unreadable"`` — the file EXISTS but could not be opened/read (e.g. a
      directory in its place, a permission error). ``rows`` is ``None`` — the
      VRAM evidence is unavailable and that is WARN-worthy, not "doctor off".
    * ``"present"`` — the file was read. ``rows`` is the list of parsed row
      dicts (possibly empty). A malformed JSONL LINE is skipped (logged at
      debug), not fatal.

    Never raises.
    """
    target = run_dir / "vram_trajectory.jsonl"
    try:
        if not target.exists():
            return ("absent", None)
    except Exception as exc:  # noqa: BLE001 — stat failure → treat as absent
        logger.warning("postmortem: trajectory stat failed for %s: %s", run_dir, exc)
        return ("absent", None)

    try:
        rows: List[Dict[str, Any]] = []
        with open(target, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                except Exception as exc:  # noqa: BLE001 — skip the bad line
                    logger.debug("postmortem: skipping malformed trajectory line: %s", exc)
                    continue
                if isinstance(parsed, dict):
                    rows.append(parsed)
        return ("present", rows)
    except Exception as exc:  # noqa: BLE001 — file exists but unreadable
        logger.warning("postmortem: trajectory exists but unreadable for %s: %s", run_dir, exc)
        return ("unreadable", None)


# --------------------------------------------------------------------------- #
# Trajectory correlation helpers
# --------------------------------------------------------------------------- #


def _phase_rows(rows: List[Dict[str, Any]], phase: str) -> List[Dict[str, Any]]:
    """Return the trajectory rows whose ``phase`` matches ``phase``."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        try:
            if row.get("phase") == phase:
                out.append(row)
        except Exception:  # noqa: BLE001 — a weird row is just skipped
            continue
    return out


def _min_free_mib(rows: List[Dict[str, Any]]) -> Optional[int]:
    """Lowest non-null ``free_mib`` across ``rows`` (``None`` if none present).

    Considers BOTH the ``before`` and ``after`` samples (finding #6) — a hard
    mid-phase OOM usually never writes the ``after`` row, so the ``before``
    sample may be all we have.
    """
    lowest: Optional[int] = None
    for row in rows:
        try:
            value = row.get("free_mib")
            if value is None:
                continue
            value = int(value)
        except Exception:  # noqa: BLE001
            continue
        if lowest is None or value < lowest:
            lowest = value
    return lowest


def _resident_label(residents: List[Dict[str, Any]]) -> str:
    """One-line ``name (~N MiB)`` label for the resident model set."""
    parts: List[str] = []
    for model in residents:
        try:
            parts.append(f"{model.get('name', '?')} (~{model.get('vram_mib', 0)} MiB)")
        except Exception:  # noqa: BLE001
            parts.append(repr(model))
    return ", ".join(parts) if parts else "none"


def _residents_at(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten the resident models named across ``rows`` (de-duped by name)."""
    seen: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        try:
            models = row.get("resident_models") or []
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(models, list):
            continue
        for model in models:
            if not isinstance(model, dict):
                continue
            name = str(model.get("name", "?"))
            if name not in seen:
                seen[name] = model
    return list(seen.values())


def _error_names_oom(error: Optional[str]) -> bool:
    """True iff the flat checkpoint ``error`` string word-awarely names an OOM."""
    if not error:
        return False
    try:
        return bool(_OOM_ERROR_REGEX.search(str(error)))
    except Exception:  # noqa: BLE001
        return False


def _trajectory_phase_order(trajectory: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Distinct phase names seen in the trajectory, in first-seen order."""
    order: List[str] = []
    seen = set()
    for row in trajectory or []:
        try:
            phase = str(row.get("phase", "?"))
        except Exception:  # noqa: BLE001
            continue
        if phase not in seen:
            seen.add(phase)
            order.append(phase)
    return order


# --------------------------------------------------------------------------- #
# Per-section emitters
# --------------------------------------------------------------------------- #


def _phase_results(checkpoints: List[Any]) -> List[CheckResult]:
    """One :class:`CheckResult` per phase checkpoint (in phase order).

    completed → OK; failed → FAIL (carries ``tasks_failed`` + ``error``);
    started-but-not-completed → WARN (ambiguous: the run either DIED here OR is
    still in progress executing this phase — finding #7; not asserted as a
    confirmed crash).
    """
    results: List[CheckResult] = []
    for cp in checkpoints:
        try:
            phase = getattr(cp, "phase_name", "?")
            status = getattr(cp, "status", "?")
            error = getattr(cp, "error", None)
            tasks_failed = list(getattr(cp, "tasks_failed", []) or [])
            data = {
                "phase": phase,
                "status": status,
                "error": error,
                "tasks_failed": tasks_failed,
            }

            if status == "completed":
                results.append(
                    CheckResult(
                        name=f"phase_{phase}",
                        group="postmortem",
                        severity=Severity.OK,
                        summary=f"phase {phase} completed",
                        data=data,
                    )
                )
            elif status == "failed":
                results.append(
                    CheckResult(
                        name=f"phase_{phase}",
                        group="postmortem",
                        severity=Severity.FAIL,
                        summary=f"phase {phase} FAILED: {error or 'no error recorded'}",
                        detail=(
                            f"{len(tasks_failed)} task(s) failed"
                            if tasks_failed
                            else ""
                        ),
                        remediation="inspect the phase error above; see the VRAM context note if present",
                        data=data,
                    )
                )
            elif status == "started":
                # Ambiguous: a 'started' checkpoint that never advanced to
                # completed/failed means EITHER the process died in this phase
                # OR the run is still in flight executing it. We do NOT assert a
                # crash — demote to a WARN that states the ambiguity (finding #7).
                started_at = getattr(cp, "started_at", None)
                results.append(
                    CheckResult(
                        name=f"phase_{phase}",
                        group="postmortem",
                        severity=Severity.WARN,
                        summary=(
                            f"phase {phase} is in 'started' state — the run "
                            "either died here OR is still in progress"
                        ),
                        detail=(
                            f"phase started_at={started_at}" if started_at else ""
                        ),
                        remediation="check whether the run is still executing; if not, this is the last phase reached — inspect its logs and the VRAM context below",
                        data=data,
                    )
                )
            else:
                results.append(
                    CheckResult(
                        name=f"phase_{phase}",
                        group="postmortem",
                        severity=Severity.WARN,
                        summary=f"phase {phase} has unrecognized status={status!r}",
                        data=data,
                    )
                )
        except Exception as exc:  # noqa: BLE001 — one bad checkpoint is skipped
            logger.warning("postmortem: skipping a malformed checkpoint: %s", exc)
            continue
    return results


def _dropped_evidence_result(files_on_disk: int, loaded: int) -> Optional[CheckResult]:
    """WARN when fewer checkpoints loaded than ``*_checkpoint.json`` on disk.

    ``CheckpointManager.get_all_checkpoints`` silently drops a checkpoint file
    it cannot parse; counting the files on disk vs the loaded count makes that
    dropped evidence visible (finding #10). Returns ``None`` when nothing was
    dropped.
    """
    try:
        skipped = files_on_disk - loaded
        if skipped <= 0:
            return None
        return CheckResult(
            name="postmortem_dropped_checkpoints",
            group="postmortem",
            severity=Severity.WARN,
            summary=(
                f"{skipped} checkpoint file(s) were unreadable/corrupted and "
                f"skipped ({loaded} of {files_on_disk} loaded)"
            ),
            remediation="the dropped checkpoints may have held the crash phase; inspect <run>/checkpoints/ for malformed *_checkpoint.json files",
            data={"files_on_disk": files_on_disk, "loaded": loaded, "skipped": skipped},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("postmortem: dropped-evidence check failed: %s", exc)
        return None


def _confirmed_oom_result(checkpoints: List[Any]) -> Optional[CheckResult]:
    """Scan ALL failed checkpoints for an OOM-naming error → one CONFIRMED FAIL.

    The ONLY FAIL-level OOM claim. Scans every failed phase, not just the last
    crash phase (finding #9), so an OOM in an earlier failed phase is still
    confirmed. Returns ``None`` when no failed phase's error names an OOM.
    """
    try:
        matched: List[str] = []
        primary_error: Optional[str] = None
        for cp in checkpoints:
            try:
                if getattr(cp, "status", None) != "failed":
                    continue
                error = getattr(cp, "error", None)
                if _error_names_oom(error):
                    phase = getattr(cp, "phase_name", "?")
                    matched.append(phase)
                    if primary_error is None:
                        primary_error = error
            except Exception:  # noqa: BLE001
                continue

        if not matched:
            return None

        primary = matched[0]
        others = matched[1:]
        suffix = f" (also: {', '.join(others)})" if others else ""
        return CheckResult(
            name="oom_correlation",
            group="postmortem",
            severity=Severity.FAIL,
            summary=f"confirmed OOM at phase {primary} (read from the checkpoint error){suffix}",
            detail=f"checkpoint error: {primary_error}" if primary_error else "",
            remediation="reduce the resident-model VRAM footprint or shrink the phase batch; the run itself reported the OOM",
            data={
                "phase": primary,
                "phases": matched,
                "error": primary_error,
                "oom": "confirmed",
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("postmortem: confirmed-OOM scan failed: %s", exc)
        return None


def _timeline_result(
    status: str, trajectory: Optional[List[Dict[str, Any]]]
) -> CheckResult:
    """A VRAM-timeline result keyed on the trajectory load ``status``.

    * ``"absent"`` → INFO 'no VRAM trajectory recorded (doctor was off)';
    * ``"unreadable"`` → WARN 'exists but could not be read' (finding #3);
    * ``"present"`` → INFO per-phase ``free_mib`` before→after summary.
    """
    if status == "absent":
        return CheckResult(
            name="vram_timeline",
            group="postmortem",
            severity=Severity.INFO,
            summary="no VRAM trajectory recorded (ED4ALL_VRAM_DOCTOR was off for this run)",
            detail="VRAM context is unavailable; only the phase errors above are forensic.",
            data={"trajectory_present": False, "trajectory_status": status},
        )

    if status == "unreadable":
        return CheckResult(
            name="vram_timeline",
            group="postmortem",
            severity=Severity.WARN,
            summary="vram_trajectory.jsonl exists but could not be read — VRAM evidence unavailable",
            remediation="inspect <run>/vram_trajectory.jsonl; it is present but unreadable (wrong type, permissions, or corruption)",
            data={"trajectory_present": False, "trajectory_status": status},
        )

    # status == "present"
    rows = trajectory or []

    # Build a per-phase free_mib before→after summary in first-seen order.
    phase_order: List[str] = []
    by_phase: Dict[str, Dict[str, Optional[int]]] = {}
    for row in rows:
        try:
            phase = str(row.get("phase", "?"))
            when = row.get("when")
            free = row.get("free_mib")
            free = int(free) if free is not None else None
        except Exception:  # noqa: BLE001
            continue
        if phase not in by_phase:
            by_phase[phase] = {"before": None, "after": None}
            phase_order.append(phase)
        if when in ("before", "after"):
            by_phase[phase][when] = free
        else:
            # Unknown 'when' — record as a best-effort 'after' if unset.
            if by_phase[phase]["after"] is None:
                by_phase[phase]["after"] = free

    segments: List[str] = []
    for phase in phase_order:
        before = by_phase[phase]["before"]
        after = by_phase[phase]["after"]
        if before is not None and after is not None:
            segments.append(f"{phase} {before}→{after}")
        elif after is not None:
            segments.append(f"{phase} {after}")
        elif before is not None:
            segments.append(f"{phase} {before}")
        else:
            segments.append(f"{phase} ?")

    summary = (
        "VRAM: " + ", ".join(segments) + " MiB"
        if segments
        else "VRAM trajectory present but empty"
    )
    return CheckResult(
        name="vram_timeline",
        group="postmortem",
        severity=Severity.INFO,
        summary=summary,
        data={
            "trajectory_present": True,
            "trajectory_status": status,
            "phase_free_mib": by_phase,
        },
    )


def _vram_context_result(
    crash_cp: Any, status: str, trajectory: Optional[List[Dict[str, Any]]]
) -> Optional[CheckResult]:
    """Honest free-VRAM CONTEXT around the crash phase — NEVER an OOM verdict.

    Demoted from the old 'LIKELY OOM' FAIL heuristic (findings #4/#5/#6): low
    free VRAM is normal on the shared 8 GB box and a hard OOM often never writes
    the cratered ``after`` sample, so the free-VRAM reading is presented as INFO
    context with the explicit caveat that it is NOT OOM evidence on its own. No
    resident-model requirement (finding #5). Returns ``None`` when there is no
    trajectory or no rows for the crash phase (the timeline already covers an
    absent/unreadable trajectory).
    """
    try:
        if status != "present" or not trajectory:
            return None

        phase = getattr(crash_cp, "phase_name", "?")

        # A confirmed-OOM FAIL (error-string) already owns the OOM narrative for
        # this phase; the heuristic context would only muddy it.
        if _error_names_oom(getattr(crash_cp, "error", None)):
            return None

        rows = _phase_rows(trajectory, phase)
        if not rows:
            return None

        free_mib = _min_free_mib(rows)
        if free_mib is None:
            return None

        residents = _residents_at(rows)
        return CheckResult(
            name="vram_context",
            group="postmortem",
            severity=Severity.INFO,
            summary=(
                f"free VRAM at {phase} was {free_mib} MiB "
                f"(resident: {_resident_label(residents)}); {_VRAM_CONTEXT_CAVEAT}"
            ),
            data={
                "phase": phase,
                "free_mib": free_mib,
                "resident": residents,
                "oom": "not_asserted",
            },
        )
    except Exception as exc:  # noqa: BLE001 — context must never crash
        logger.warning("postmortem: VRAM context failed: %s", exc)
        return None


def _trajectory_gap_result(
    checkpoints: List[Any], status: str, trajectory: Optional[List[Dict[str, Any]]]
) -> Optional[CheckResult]:
    """WARN when the trajectory shows phase activity past the last checkpoint.

    For an all-``completed`` checkpoint set, a trajectory phase that has NO
    corresponding checkpoint is a gap — the run started a phase (VRAM row
    written) but no checkpoint advanced, the OS-OOM-kill-in-the-gap signature
    (finding #2). Returns ``None`` when there is no such gap.
    """
    try:
        if status != "present" or not trajectory:
            return None
        known_phases = set()
        for cp in checkpoints:
            try:
                known_phases.add(getattr(cp, "phase_name", None))
            except Exception:  # noqa: BLE001
                continue
        extra = [p for p in _trajectory_phase_order(trajectory) if p not in known_phases]
        if not extra:
            return None
        return CheckResult(
            name="postmortem_trajectory_gap",
            group="postmortem",
            severity=Severity.WARN,
            summary=(
                "VRAM trajectory shows activity for phase(s) "
                f"{', '.join(extra)} with no recorded checkpoint — the run may "
                "have died in the gap after the last completed checkpoint"
            ),
            remediation="treat the recorded checkpoints as a floor, not proof the run reached its terminal phase; inspect the run log for the trailing phase(s)",
            data={"phases_without_checkpoint": extra},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("postmortem: trajectory-gap check failed: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# Public check
# --------------------------------------------------------------------------- #


def postmortem_checks(ctx: CheckContext) -> List[CheckResult]:
    """Forensic post-mortem of a past run (``group='postmortem'``).

    Reads ``ctx.run_id`` and the run's persisted checkpoints + VRAM trajectory
    to narrate which phase failed, with what error, and — only where the
    evidence supports it — a confirmed OOM. Fails CLOSED on indeterminate
    evidence (no readable checkpoints) rather than claiming success. NEVER
    raises — every disk read / parse / correlation is wrapped; a malformed file
    is skipped.
    """
    results: List[CheckResult] = []
    try:
        run_id = ctx.run_id if ctx is not None else None

        # --- Defensive: the CLI only routes here with a run_id ------------
        if not run_id:
            return [
                CheckResult(
                    name="postmortem_no_run_id",
                    group="postmortem",
                    severity=Severity.WARN,
                    summary="no run_id provided",
                    remediation="pass --run-id <id> to analyze a past run",
                    data={},
                )
            ]

        # --- Resolve the run dir -----------------------------------------
        run_dir = _resolve_run_dir(run_id)
        if run_dir is None or not run_dir.exists():
            shown = str(run_dir) if run_dir is not None else "(unresolved)"
            return [
                CheckResult(
                    name="postmortem_run_not_found",
                    group="postmortem",
                    severity=Severity.FAIL,
                    summary=f"run {run_id} not found at {shown}",
                    remediation="check the run id; runs live under state/runs/ or $ED4ALL_STATE_RUNS_DIR",
                    data={"run_id": run_id, "run_dir": shown},
                )
            ]

        # --- Phase checkpoints -------------------------------------------
        checkpoints = _load_checkpoints(run_dir)
        files_on_disk = _count_checkpoint_files(run_dir)
        loaded = len(checkpoints)

        results.extend(_phase_results(checkpoints))

        # --- #10: dropped (unreadable) checkpoint evidence ---------------
        dropped = _dropped_evidence_result(files_on_disk, loaded)
        if dropped is not None:
            results.append(dropped)

        # --- #9: confirmed OOM from ANY failed phase's error -------------
        confirmed = _confirmed_oom_result(checkpoints)
        if confirmed is not None:
            results.append(confirmed)

        # --- Identify the crash phase (failed or started-incomplete) -----
        crash_cp = None
        for cp in checkpoints:
            try:
                if getattr(cp, "status", None) in ("failed", "started"):
                    crash_cp = cp  # last such phase in index order = the crash
            except Exception:  # noqa: BLE001
                continue

        # --- VRAM trajectory timeline ------------------------------------
        status, trajectory = _load_trajectory(run_dir)
        results.append(_timeline_result(status, trajectory))

        # --- Honest free-VRAM context for the crash phase (never a FAIL) --
        if crash_cp is not None:
            ctx_result = _vram_context_result(crash_cp, status, trajectory)
            if ctx_result is not None:
                results.append(ctx_result)

        # --- Outcome summary (fail-closed) -------------------------------
        if loaded == 0:
            # #1: empty / all-unreadable checkpoint set → INDETERMINATE, never
            # a clean exit-0 "completed all phases". WARN (non-zero exit). The
            # timeline above still surfaces any VRAM trajectory crater.
            results.append(
                CheckResult(
                    name="postmortem_indeterminate",
                    group="postmortem",
                    severity=Severity.WARN,
                    summary=(
                        f"indeterminate: no readable checkpoints for run {run_id} "
                        "— cannot determine the run outcome (checkpoints missing "
                        "or corrupted)"
                    ),
                    remediation="check <run>/checkpoints/ exists and holds valid *_checkpoint.json files; a doctor cannot certify a run with no readable checkpoints",
                    data={
                        "run_id": run_id,
                        "checkpoints_loaded": 0,
                        "checkpoint_files_on_disk": files_on_disk,
                    },
                )
            )
        elif crash_cp is None:
            # #2: every recorded phase is 'completed'. Do NOT assert the RUN
            # succeeded — only that the RECORDED checkpoints completed. Surface
            # a trajectory gap (activity past the last checkpoint) as a WARN.
            last_phase = getattr(checkpoints[-1], "phase_name", "?")
            results.append(
                CheckResult(
                    name="postmortem_summary",
                    group="postmortem",
                    severity=Severity.OK,
                    summary=(
                        f"all {loaded} recorded phase(s) completed (last recorded: "
                        f"{last_phase}) — note: this confirms the recorded "
                        "checkpoints, not that the run reached its terminal phase"
                    ),
                    data={
                        "run_id": run_id,
                        "phases": loaded,
                        "last_recorded_phase": last_phase,
                    },
                )
            )
            gap = _trajectory_gap_result(checkpoints, status, trajectory)
            if gap is not None:
                results.append(gap)
        # else: a failed/started crash phase already emitted its FAIL/WARN
        # (plus any confirmed-OOM FAIL) — no success summary.

        return results
    except Exception as exc:  # noqa: BLE001 — the whole check must never raise
        logger.warning("postmortem: check raised (degrading to a WARN): %s", exc)
        return [
            CheckResult(
                name="postmortem_error",
                group="postmortem",
                severity=Severity.WARN,
                summary=f"post-mortem errored: {exc}",
                detail=f"{type(exc).__name__}: {exc}",
                remediation="this is a doctor bug — the post-mortem should never raise",
                data={"error": str(exc), "error_type": type(exc).__name__},
            )
        ]


def register_postmortem_checks() -> None:
    """Register the post-mortem check group (NOT at import time)."""
    register("postmortem", postmortem_checks)


__all__ = ["postmortem_checks", "register_postmortem_checks"]
