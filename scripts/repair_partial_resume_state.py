#!/usr/bin/env python3
"""Repair a workflow whose phase_outputs were poisoned by a partial-completion
resume trap (the Bug A / Bug B live-fire).

Symptom this repairs
--------------------
A multi-task phase (canonically ``dart_conversion``: one task per corpus PDF)
partially completed — some tasks COMPLETE-d, others timed out — but the pre-fix
workflow runner stamped the phase ``_completed=True, _gates_passed=True`` on
disk anyway (Bug A). A subsequent ``--resume`` then treated that phase as
satisfied and marched downstream (Bug B), stamping ``staging`` /
``objective_extraction`` / ``course_planning`` / etc. ``_completed=True`` on top
of outputs derived from only the ONE converted chapter.

What this does
--------------
Rewrites ``state/workflows/<workflow_id>.json`` so the next ``--resume`` re-runs
the incomplete phase AND every downstream phase fresh, WITHOUT discarding the
legitimately-completed task result(s):

  * The ``--incomplete-phase`` (default ``dart_conversion``) has its
    ``_completed`` flag flipped to ``False`` (and ``_resume_restored`` cleared)
    so the runner re-dispatches it. Its recorded output keys (e.g. the ch09
    ``output_paths``) and the workflow ``tasks`` array are left untouched — the
    already-converted units survive as inputs / per-task-checkpoint reuse.
  * Every OTHER phase present in ``phase_outputs`` (the poisoned downstream set)
    is REMOVED entirely so those phases run fresh on the full corpus. Override
    the reset set with ``--reset-phases a,b,c`` if you want to keep some.
  * Workflow-level ``status`` is reset to ``CREATED``, and the stale
    ``failed_phase`` / ``failure_reason`` / ``stopped_after`` markers are
    cleared, so the resume starts clean.

Per-run executor checkpoints under ``state/runs/<run_id>/checkpoints/`` live in
a per-invocation timestamped dir and are NOT consulted across a fresh resume
(each run gets its own run dir), so they are intentionally left alone.

Safety
------
Prints a full plan by default (``--dry-run`` is the default). Nothing is written
unless ``--apply`` is passed.

Usage
-----
    python scripts/repair_partial_resume_state.py --workflow-id WF-XXXX            # dry-run
    python scripts/repair_partial_resume_state.py --workflow-id WF-XXXX --apply     # write
    python scripts/repair_partial_resume_state.py --workflow-id WF-XXXX \
        --incomplete-phase dart_conversion --reset-phases staging,course_planning
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _state_path(workflow_id: str) -> Path:
    """Resolve the workflow-state JSON path, honouring ED4ALL_HOME via lib.paths."""
    try:
        from lib.paths import STATE_PATH  # noqa: PLC0415
        base = Path(STATE_PATH)
    except Exception:  # noqa: BLE001 — fall back to in-tree default
        base = Path(__file__).resolve().parent.parent / "state"
    return base / "workflows" / f"{workflow_id}.json"


def _load_params(state: Dict[str, Any]) -> Dict[str, Any]:
    params = state.get("params", {})
    if isinstance(params, str):
        try:
            params = json.loads(params)
        except ValueError:
            params = {}
    return params if isinstance(params, dict) else {}


def build_plan(
    state: Dict[str, Any],
    *,
    incomplete_phase: str,
    reset_phases: Optional[List[str]],
) -> Dict[str, Any]:
    """Compute the repair plan (no mutation). Returns a describable dict."""
    phase_outputs: Dict[str, Any] = state.get("phase_outputs", {}) or {}

    # Default reset set: every phase_output except the incomplete phase.
    if reset_phases is None:
        to_reset = [p for p in phase_outputs.keys() if p != incomplete_phase]
    else:
        to_reset = [p for p in reset_phases if p in phase_outputs]

    incomplete_present = incomplete_phase in phase_outputs
    incomplete_flag = (
        bool(phase_outputs.get(incomplete_phase, {}).get("_completed"))
        if incomplete_present
        else None
    )

    clear_workflow_markers = {
        k: state.get(k)
        for k in ("status", "failed_phase", "failure_reason", "stopped_after")
        if k in state
    }

    return {
        "incomplete_phase": incomplete_phase,
        "incomplete_present": incomplete_present,
        "incomplete_completed_flag_before": incomplete_flag,
        "reset_phases": to_reset,
        "kept_phases": [p for p in phase_outputs.keys() if p not in to_reset],
        "workflow_markers_before": clear_workflow_markers,
    }


def apply_plan(state: Dict[str, Any], plan: Dict[str, Any]) -> None:
    """Mutate ``state`` in place per the plan."""
    phase_outputs: Dict[str, Any] = state.get("phase_outputs", {}) or {}

    # Un-complete the incomplete phase (keep its recorded outputs + tasks).
    ip = plan["incomplete_phase"]
    if plan["incomplete_present"] and isinstance(phase_outputs.get(ip), dict):
        phase_outputs[ip]["_completed"] = False
        phase_outputs[ip].pop("_resume_restored", None)

    # Remove poisoned downstream phase outputs so they re-run fresh.
    for phase in plan["reset_phases"]:
        phase_outputs.pop(phase, None)

    state["phase_outputs"] = phase_outputs
    state["status"] = "CREATED"
    state.pop("failed_phase", None)
    state.pop("failure_reason", None)
    state.pop("stopped_after", None)


def _print_plan(workflow_id: str, path: Path, plan: Dict[str, Any], apply: bool) -> None:
    print(f"workflow_id : {workflow_id}")
    print(f"state file  : {path}")
    print(f"mode        : {'APPLY (writing)' if apply else 'DRY-RUN (no writes)'}")
    print("-" * 68)
    ip = plan["incomplete_phase"]
    if not plan["incomplete_present"]:
        print(f"[warn] incomplete-phase {ip!r} not present in phase_outputs")
    else:
        print(
            f"un-complete : {ip}  "
            f"(_completed {plan['incomplete_completed_flag_before']} -> False; "
            f"recorded outputs + tasks preserved)"
        )
    if plan["reset_phases"]:
        print(f"reset (drop): {', '.join(plan['reset_phases'])}")
    else:
        print("reset (drop): <none>")
    kept = [p for p in plan["kept_phases"] if p != ip]
    if kept:
        print(f"untouched   : {', '.join(kept)}")
    print(
        "workflow    : status -> CREATED; clearing "
        f"{sorted(plan['workflow_markers_before'].keys())}"
    )
    print("-" * 68)
    if not apply:
        print("No changes written. Re-run with --apply to write.")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workflow-id", required=True, help="e.g. WF-00000000-00000001")
    ap.add_argument(
        "--incomplete-phase",
        default="dart_conversion",
        help="Multi-task phase to un-complete (default: dart_conversion).",
    )
    ap.add_argument(
        "--reset-phases",
        default=None,
        help=(
            "Comma-separated downstream phases to drop from phase_outputs. "
            "Default: every phase_output except --incomplete-phase."
        ),
    )
    ap.add_argument("--apply", action="store_true", help="Write changes (default: dry-run).")
    args = ap.parse_args(argv)

    path = _state_path(args.workflow_id)
    if not path.exists():
        print(f"error: workflow state not found: {path}", file=sys.stderr)
        return 2
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read/parse {path}: {exc}", file=sys.stderr)
        return 2

    reset_phases = (
        [p.strip() for p in args.reset_phases.split(",") if p.strip()]
        if args.reset_phases is not None
        else None
    )

    plan = build_plan(
        state,
        incomplete_phase=args.incomplete_phase,
        reset_phases=reset_phases,
    )
    _print_plan(args.workflow_id, path, plan, args.apply)

    if args.apply:
        apply_plan(state, plan)
        path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        print(f"applied: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
