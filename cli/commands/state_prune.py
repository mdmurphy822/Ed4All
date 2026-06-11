"""
``ed4all state prune`` — garbage-collect ``state/runs/`` and
``state/workflows/`` artifacts (Wave 74 cleanup).

Both directories accumulate per-run dirs/files indefinitely:

* ``state/workflows/WF-{date}-{hash}.json`` — the workflow record.
* ``state/runs/{run_id}/`` — per-run mailbox/checkpoints/decisions tree.

This command keeps the most recent N COMPLETED workflows (and their
matching ``state/runs/{run_id}/`` dirs) plus any workflows in a status
the operator wants to preserve (``--keep-status``, default ``COMPLETE``;
repeat the flag to add ``RUNNING``, etc.). Anything older — and any
``state/runs/`` dir with no matching workflow record (orphan) — is
dropped. Workflows whose status is in ``--keep-status`` are NEVER
pruned regardless of ``--keep-last``; this is what protects an
in-progress run from being clobbered while the dispatcher is still
working.

Use ``--dry-run`` to preview before deleting.

Guardrails
----------

* ``.gitkeep`` markers under either directory are never touched.
* A workflow whose status is in ``--keep-status`` is treated as a
  protected entry and its run dir is preserved.
* Orphan ``state/runs/`` dirs (no workflow JSON references the run_id)
  are pruned by default — but if the dir name matches an active
  workflow's run_id even one whose status is RUNNING, it survives.
"""

from __future__ import annotations

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import click

from lib.paths import STATE_PATH, get_training_captures_dir


# Default ``--keep-last`` value. Conservative: keep the 5 most-recent
# COMPLETE workflows so an operator who runs ``ed4all state prune``
# without thinking about it doesn't lose recent debug context.
DEFAULT_KEEP_LAST = 5

# Default ``--keep-status`` set. COMPLETE workflows count toward the
# ``--keep-last`` quota and may be pruned beyond it; statuses passed
# via ``--keep-status`` (e.g. ``RUNNING``) are *always* preserved.
DEFAULT_KEEP_STATUS = ("COMPLETE",)

# Default age threshold (days) for ``--training-captures``. Conservative: only
# captures older than this are eligible, and the whole sweep is opt-in (never
# pruned without the explicit flag — training captures are the SLM-training
# audit trail, so default-preserve is the safe posture).
DEFAULT_TRAINING_CAPTURES_OLDER_THAN = 30


@dataclass
class WorkflowRecord:
    """One ``state/workflows/*.json`` entry."""

    path: Path
    workflow_id: str
    run_id: Optional[str]
    status: str
    updated_at: Optional[str]
    raw: Dict
    size_bytes: int

    @property
    def updated_dt(self) -> datetime:
        """Sortable timestamp; falls back to mtime if updated_at missing."""
        if self.updated_at:
            try:
                return datetime.fromisoformat(self.updated_at)
            except ValueError:
                pass
        return datetime.fromtimestamp(self.path.stat().st_mtime)


@dataclass
class RunDir:
    """One ``state/runs/{run_id}/`` entry."""

    path: Path
    run_id: str
    size_bytes: int


@dataclass
class PrunePlan:
    """What ``ed4all state prune`` will delete vs keep."""

    keep_workflows: List[WorkflowRecord] = field(default_factory=list)
    drop_workflows: List[WorkflowRecord] = field(default_factory=list)
    keep_runs: List[RunDir] = field(default_factory=list)
    drop_runs: List[RunDir] = field(default_factory=list)
    orphan_runs: List[RunDir] = field(default_factory=list)

    @property
    def workflow_bytes_freed(self) -> int:
        return sum(w.size_bytes for w in self.drop_workflows)

    @property
    def run_bytes_freed(self) -> int:
        return sum(r.size_bytes for r in self.drop_runs)

    @property
    def total_bytes_freed(self) -> int:
        return self.workflow_bytes_freed + self.run_bytes_freed


@dataclass
class CaptureFile:
    """One ``training-captures/**/*.jsonl`` decision-capture file."""

    path: Path
    mtime: float
    size_bytes: int


@dataclass
class CapturePrunePlan:
    """Opt-in age-based prune over ``training-captures/``."""

    older_than_days: int
    drop_captures: List[CaptureFile] = field(default_factory=list)
    keep_count: int = 0

    @property
    def capture_bytes_freed(self) -> int:
        return sum(c.size_bytes for c in self.drop_captures)


def _scan_capture_files(captures_dir: Path) -> List[CaptureFile]:
    """List every ``*.jsonl`` decision-capture file under ``captures_dir``.

    ``.gitkeep`` markers are skipped (mirrors the state-prune guardrail).
    """
    out: List[CaptureFile] = []
    if not captures_dir.exists():
        return out
    for path in sorted(captures_dir.rglob("*.jsonl")):
        if path.name == ".gitkeep" or not path.is_file():
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        out.append(CaptureFile(path=path, mtime=st.st_mtime, size_bytes=st.st_size))
    return out


def build_capture_plan(
    captures: List[CaptureFile],
    *,
    older_than_days: int,
    now: Optional[float] = None,
) -> CapturePrunePlan:
    """Select capture files whose mtime is older than ``older_than_days``.

    Age is measured from ``now`` (defaults to wall-clock); a file exactly at the
    boundary is KEPT (strict ``<`` cutoff comparison). Never prunes anything
    newer than the threshold — the whole point of the opt-in age gate.
    """
    if now is None:
        now = time.time()
    cutoff = now - (older_than_days * 86400)
    plan = CapturePrunePlan(older_than_days=older_than_days)
    for cap in captures:
        if cap.mtime < cutoff:
            plan.drop_captures.append(cap)
        else:
            plan.keep_count += 1
    return plan


def _execute_capture_plan(plan: CapturePrunePlan, captures_dir: Path) -> None:
    """Delete every dropped capture file, then prune emptied subdirs.

    Only directories that become empty as a result are removed; the
    ``captures_dir`` root and any ``.gitkeep`` markers are preserved.
    """
    for cap in plan.drop_captures:
        try:
            cap.path.unlink()
        except FileNotFoundError:
            pass
    # Bottom-up sweep of now-empty subdirs (never the root itself).
    captures_dir = captures_dir.resolve()
    for sub in sorted(
        (p for p in captures_dir.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        try:
            if not any(sub.iterdir()):
                sub.rmdir()
        except OSError:
            continue


def _dir_size_bytes(path: Path) -> int:
    """Recursively sum file sizes under ``path``. Robust to broken symlinks."""
    total = 0
    if not path.exists():
        return 0
    for sub in path.rglob("*"):
        try:
            if sub.is_file():
                total += sub.stat().st_size
        except (OSError, ValueError):
            continue
    return total


def _load_workflows(workflows_dir: Path) -> List[WorkflowRecord]:
    """Read every ``state/workflows/*.json`` into a ``WorkflowRecord``.

    Files that fail to parse are still returned (with ``status="UNKNOWN"``)
    so the operator can choose to drop them via ``--keep-status``.
    """
    records: List[WorkflowRecord] = []
    if not workflows_dir.exists():
        return records
    for path in sorted(workflows_dir.glob("*.json")):
        if path.name == ".gitkeep":
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        records.append(
            WorkflowRecord(
                path=path,
                workflow_id=str(raw.get("id") or path.stem),
                run_id=(raw.get("params") or {}).get("run_id") or raw.get("run_id"),
                status=str(raw.get("status") or "UNKNOWN").upper(),
                updated_at=raw.get("updated_at") or raw.get("created_at"),
                raw=raw if isinstance(raw, dict) else {},
                size_bytes=path.stat().st_size if path.exists() else 0,
            )
        )
    return records


def _load_run_dirs(runs_dir: Path) -> List[RunDir]:
    """List every ``state/runs/{run_id}/`` directory."""
    out: List[RunDir] = []
    if not runs_dir.exists():
        return out
    for entry in sorted(runs_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == ".gitkeep":
            continue
        out.append(
            RunDir(
                path=entry,
                run_id=entry.name,
                size_bytes=_dir_size_bytes(entry),
            )
        )
    return out


def build_plan(
    workflows: List[WorkflowRecord],
    runs: List[RunDir],
    *,
    keep_last: int,
    keep_statuses: Set[str],
) -> PrunePlan:
    """Decide which workflows + run dirs to keep vs drop.

    Rules:

    1. Workflows with ``status`` in ``keep_statuses`` are always kept.
    2. Of the remaining workflows, the ``keep_last`` most-recently
       updated COMPLETE entries are kept (sorted by ``updated_at``).
    3. ``state/runs/{run_id}/`` dirs are kept iff their ``run_id``
       matches a kept workflow's run_id (or workflow_id, for back-compat).
    4. ``state/runs/`` dirs with no matching workflow record are
       classified as orphans and dropped (unless an associated kept
       workflow shares the run_id — which step 3 already covers).
    """
    plan = PrunePlan()

    protected = [w for w in workflows if w.status in keep_statuses]
    candidate = [w for w in workflows if w.status not in keep_statuses]

    # Sort candidates by recency (most-recent first); keep first N.
    candidate.sort(key=lambda w: w.updated_dt, reverse=True)
    keep_from_candidates = candidate[:keep_last] if keep_last > 0 else []
    drop_from_candidates = candidate[keep_last:] if keep_last > 0 else candidate

    plan.keep_workflows = protected + keep_from_candidates
    plan.drop_workflows = drop_from_candidates

    # Build the set of run_ids referenced by kept workflows. Match on
    # both ``run_id`` (canonical) and ``workflow_id`` (legacy) so we
    # don't accidentally drop a run dir whose name is the workflow_id.
    kept_run_keys: Set[str] = set()
    for w in plan.keep_workflows:
        if w.run_id:
            kept_run_keys.add(w.run_id)
        kept_run_keys.add(w.workflow_id)

    referenced_run_keys: Set[str] = set()
    for w in workflows:
        if w.run_id:
            referenced_run_keys.add(w.run_id)
        referenced_run_keys.add(w.workflow_id)

    for run in runs:
        if run.run_id in kept_run_keys:
            plan.keep_runs.append(run)
        elif run.run_id not in referenced_run_keys:
            # Orphan — no workflow JSON points at it. Always prunable
            # unless it belongs to a kept workflow above (handled).
            plan.orphan_runs.append(run)
            plan.drop_runs.append(run)
        else:
            plan.drop_runs.append(run)

    return plan


def _format_bytes(n: int) -> str:
    """Render an integer byte count for human-readable summaries."""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _execute_plan(plan: PrunePlan) -> None:
    """Actually delete every ``drop_*`` entry in the plan."""
    for w in plan.drop_workflows:
        try:
            w.path.unlink()
        except FileNotFoundError:
            pass
    for r in plan.drop_runs:
        if r.path.exists():
            shutil.rmtree(r.path, ignore_errors=True)


@click.group(name="state")
def state_group() -> None:
    """Inspect and prune Ed4All run state."""


@state_group.command("prune")
@click.option(
    "--keep-last",
    "-n",
    type=click.IntRange(min=0),
    default=DEFAULT_KEEP_LAST,
    show_default=True,
    help=(
        "Keep the N most-recent workflow records (and their associated "
        "state/runs dirs) regardless of status. 0 prunes everything not "
        "covered by --keep-status."
    ),
)
@click.option(
    "--keep-status",
    "keep_statuses",
    multiple=True,
    default=DEFAULT_KEEP_STATUS,
    show_default=True,
    help=(
        "Workflow statuses to always preserve. Repeat to add more "
        "(e.g. --keep-status COMPLETE --keep-status RUNNING)."
    ),
)
@click.option("--dry-run", is_flag=True, help="Print the plan without deleting.")
@click.option(
    "--state-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override state root (tests only). Defaults to the project state/.",
)
@click.option(
    "--training-captures",
    "prune_training_captures",
    is_flag=True,
    help=(
        "ALSO prune training-captures/ decision-capture files older than "
        "--older-than days. OPT-IN: training captures are never pruned without "
        "this flag (they are the SLM-training audit trail)."
    ),
)
@click.option(
    "--older-than",
    "older_than_days",
    type=click.IntRange(min=0),
    default=DEFAULT_TRAINING_CAPTURES_OLDER_THAN,
    show_default=True,
    help=(
        "Age threshold in days for --training-captures: capture files with an "
        "mtime older than this are eligible. No effect without "
        "--training-captures."
    ),
)
@click.option(
    "--training-captures-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=(
        "Override training-captures root (tests only). Defaults to the "
        "ED4ALL_TRAINING_CAPTURES_DIR / ED4ALL_HOME resolver."
    ),
)
def prune_command(
    keep_last: int,
    keep_statuses: Tuple[str, ...],
    dry_run: bool,
    state_root: Optional[Path],
    prune_training_captures: bool,
    older_than_days: int,
    training_captures_root: Optional[Path],
) -> None:
    """GC ``state/runs/`` and ``state/workflows/`` per the keep policy.

    With ``--training-captures`` ALSO age-prunes ``training-captures/`` (opt-in;
    files older than ``--older-than`` days). The training-captures sweep is
    never run by default.
    """
    root = Path(state_root) if state_root else STATE_PATH
    workflows_dir = root / "workflows"
    runs_dir = root / "runs"

    workflows = _load_workflows(workflows_dir)
    runs = _load_run_dirs(runs_dir)

    keep_set = {s.upper() for s in keep_statuses} if keep_statuses else set()
    plan = build_plan(
        workflows,
        runs,
        keep_last=keep_last,
        keep_statuses=keep_set,
    )

    # ----- Output a summary table --------------------------------------
    click.echo("ed4all state prune")
    click.echo("-" * 60)
    click.echo(
        f"  Keep last: {keep_last}    Keep statuses: "
        f"{', '.join(sorted(keep_set)) or '(none)'}"
    )
    click.echo(f"  State root: {root}")
    click.echo()
    click.echo(
        f"  Workflows: kept={len(plan.keep_workflows):<3} "
        f"dropped={len(plan.drop_workflows):<3} "
        f"({_format_bytes(plan.workflow_bytes_freed)} freed)"
    )
    click.echo(
        f"  Run dirs:  kept={len(plan.keep_runs):<3} "
        f"dropped={len(plan.drop_runs):<3} "
        f"(orphans={len(plan.orphan_runs)}, "
        f"{_format_bytes(plan.run_bytes_freed)} freed)"
    )
    # ----- Optional training-captures age-prune (opt-in) ----------------
    capture_plan: Optional[CapturePrunePlan] = None
    captures_dir: Optional[Path] = None
    if prune_training_captures:
        captures_dir = (
            Path(training_captures_root)
            if training_captures_root
            else get_training_captures_dir()
        )
        captures = _scan_capture_files(captures_dir)
        capture_plan = build_capture_plan(captures, older_than_days=older_than_days)
        click.echo()
        click.echo(
            f"  Training captures (root: {captures_dir}):"
        )
        click.echo(
            f"    older-than={older_than_days}d  "
            f"kept={capture_plan.keep_count:<4} "
            f"dropped={len(capture_plan.drop_captures):<4} "
            f"({_format_bytes(capture_plan.capture_bytes_freed)} freed)"
        )

    click.echo(
        f"  Total to free: {_format_bytes(plan.total_bytes_freed)}"
    )

    if dry_run:
        click.echo()
        click.secho("DRY RUN — no files were deleted.", fg="yellow")
        if plan.drop_workflows:
            click.echo()
            click.echo("Would delete workflows:")
            for w in plan.drop_workflows:
                click.echo(f"  - {w.path.name} (status={w.status})")
        if plan.drop_runs:
            click.echo()
            click.echo("Would delete run dirs:")
            for r in plan.drop_runs:
                tag = "(orphan)" if r in plan.orphan_runs else ""
                click.echo(f"  - {r.run_id} {tag}".rstrip())
        if capture_plan is not None and capture_plan.drop_captures:
            click.echo()
            click.echo("Would delete training-capture files:")
            for c in capture_plan.drop_captures:
                click.echo(f"  - {c.path}")
        return

    # ----- Execute --------------------------------------------------------
    _execute_plan(plan)
    if capture_plan is not None and captures_dir is not None:
        _execute_capture_plan(capture_plan, captures_dir)
    click.secho("Prune complete.", fg="green")


def register_state_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all state`` command group to the top-level CLI group."""
    cli_group.add_command(state_group)
