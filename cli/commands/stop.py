"""``ed4all stop`` — request a graceful, checkpoint-on-command stop.

Writes a filesystem stop sentinel (``lib.generation.stop_control``) that every
long-running Ed4All stage polls at unit boundaries. The running process finishes
its in-flight unit, checkpoints it, and pauses (phase status ``paused``, exit
code 3) — worst-case loss is ONE in-flight LLM call. This command NEVER touches
the running process directly; it only drops the sentinel and reports.

Three modes (exactly one per invocation):

* ``ed4all stop <workflow_id|run_id>`` — run-scoped stop. Resolves the target
  against the ``RUNNING`` workflows in ``state/workflows/*.json`` (matching
  either the workflow id or its ``params.run_id``; generic workflows fall back
  to the workflow id) and writes the run-scoped
  ``<state_runs>/<run_id>/control/STOP_REQUESTED`` sentinel.
* ``ed4all stop --all`` — global stop. Writes ``<state_runs>/STOP_ALL`` (stops
  every run) and enumerates the currently-``RUNNING`` workflows in the report.
* ``ed4all stop --clear-all`` — remove the global ``STOP_ALL`` sentinel
  (operator-owned; never auto-cleared).

Stale-``RUNNING`` heuristic: a hard-killed run leaves ``status=="RUNNING"``
forever, so candidates whose state file has not been touched in
:data:`_STALE_AFTER_SECONDS` are annotated ``possibly stale`` rather than
asserted live.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import click

from lib.generation import stop_control

#: A ``RUNNING`` workflow whose state file has not been touched in this many
#: seconds is annotated "possibly stale" (a hard kill never clears the status).
_STALE_AFTER_SECONDS = 24 * 3600


def _load_running_workflows() -> List[Dict[str, Any]]:
    """Return the ``RUNNING`` workflow records from ``state/workflows/*.json``.

    Each record carries ``workflow_id`` (the file stem / ``id``), ``run_id``
    (``params.run_id`` when present — textbook pipelines mint it — else the
    workflow id, mirroring the orchestrator's run_id resolution), ``status``,
    ``updated_at``, and a ``stale`` flag (state file older than
    :data:`_STALE_AFTER_SECONDS`). ``STATE_PATH`` is imported per call so tests
    can monkeypatch it.
    """
    from lib.paths import STATE_PATH  # noqa: PLC0415 — per-call for test isolation

    wf_dir = STATE_PATH / "workflows"
    out: List[Dict[str, Any]] = []
    if not wf_dir.is_dir():
        return out
    now = time.time()
    for path in sorted(wf_dir.glob("*.json")):
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(state, dict):
            continue
        if str(state.get("status", "")).upper() != "RUNNING":
            continue
        params = state.get("params")
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except ValueError:
                params = {}
        if not isinstance(params, dict):
            params = {}
        workflow_id = state.get("id") or state.get("workflow_id") or path.stem
        run_id = params.get("run_id") or workflow_id
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        out.append(
            {
                "workflow_id": workflow_id,
                "run_id": run_id,
                "status": state.get("status"),
                "updated_at": state.get("updated_at"),
                "mtime": mtime,
                "stale": bool(mtime and (now - mtime) > _STALE_AFTER_SECONDS),
            }
        )
    return out


def _resolve_target(
    target: str, running: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Match ``target`` against a RUNNING workflow by workflow_id or run_id.

    Returns the matching record, or ``None`` when no RUNNING workflow matches
    (the caller still writes a best-effort sentinel treating ``target`` as a
    raw run_id — a stray sentinel is a harmless no-op per risk R1).
    """
    for rec in running:
        if target in (rec.get("workflow_id"), rec.get("run_id")):
            return rec
    return None


def _emit(payload: Dict[str, Any], *, output_json: bool) -> None:
    """Render the report as JSON or as a short human-readable block."""
    if output_json:
        click.echo(json.dumps(payload, indent=2, default=str))
        return
    mode = payload.get("mode")
    if payload.get("error"):
        click.secho(f"ed4all stop: {payload['error']}", fg="red", err=True)
        return
    if mode == "clear-all":
        if payload.get("cleared"):
            click.secho("Cleared the global STOP_ALL sentinel.", fg="green")
        else:
            click.secho(
                "No global STOP_ALL sentinel to clear (already absent).",
                fg="yellow",
            )
        return
    if mode == "all":
        click.secho("Global stop requested — wrote STOP_ALL.", fg="yellow")
        click.echo(f"  Sentinel: {payload.get('sentinel')}")
    elif mode == "run":
        click.secho(
            f"Stop requested for run {payload.get('run_id')!r}.", fg="yellow"
        )
        click.echo(f"  Sentinel: {payload.get('sentinel')}")
        if not payload.get("matched"):
            click.secho(
                "  (no RUNNING workflow matched this target — sentinel written "
                "anyway; it is a harmless no-op if no such run is live.)",
                fg="yellow",
            )
    running = payload.get("running") or []
    if running:
        click.echo()
        click.secho("Currently RUNNING workflows:", fg="cyan")
        for rec in running:
            stale = "  [possibly stale]" if rec.get("stale") else ""
            click.echo(
                f"  {rec.get('workflow_id')}  (run_id={rec.get('run_id')}, "
                f"updated={rec.get('updated_at') or 'N/A'}){stale}"
            )
    elif mode in ("all", "run"):
        click.echo()
        click.echo("  (no RUNNING workflows found in state/workflows/)")


@click.command("stop")
@click.argument("target", required=False)
@click.option(
    "--all",
    "stop_all",
    is_flag=True,
    default=False,
    help="Request a GLOBAL stop (STOP_ALL) — pauses every run at its next unit.",
)
@click.option(
    "--clear-all",
    "clear_all",
    is_flag=True,
    default=False,
    help="Remove the global STOP_ALL sentinel (operator-owned; never auto-cleared).",
)
@click.option("--json", "output_json", is_flag=True, help="Machine-readable JSON output.")
def stop_command(
    target: Optional[str],
    stop_all: bool,
    clear_all: bool,
    output_json: bool,
) -> None:
    """Request a graceful stop for a run (or all runs).

    The stopped run finishes its in-flight unit, checkpoints it, and pauses
    (exit code 3); resume it later with ``--resume``. Send SIGTERM/SIGINT to a
    running ``ed4all run`` for the same effect (signal again to force-quit).
    """
    # Exactly one mode.
    modes = sum(bool(x) for x in (bool(target), stop_all, clear_all))
    if modes == 0:
        _emit(
            {
                "error": "provide a workflow_id / run_id, or --all / --clear-all.",
            },
            output_json=output_json,
        )
        raise SystemExit(2)
    if modes > 1:
        _emit(
            {
                "error": "pass exactly one of <target> / --all / --clear-all.",
            },
            output_json=output_json,
        )
        raise SystemExit(2)

    if clear_all:
        existed = stop_control._global_sentinel_path().exists()
        stop_control.clear_stop(include_global=True)
        _emit(
            {"mode": "clear-all", "cleared": bool(existed)},
            output_json=output_json,
        )
        return

    running = _load_running_workflows()

    if stop_all:
        path = stop_control.request_stop(
            scope="all", reason="operator", source="cli:stop --all"
        )
        _emit(
            {
                "mode": "all",
                "sentinel": str(path) if path else None,
                "running": running,
            },
            output_json=output_json,
        )
        return

    # Run-scoped stop.
    assert target is not None  # guarded by the mode count above
    matched = _resolve_target(target, running)
    run_id = matched["run_id"] if matched else target
    path = stop_control.request_stop(
        run_id=run_id, reason="operator", source="cli:stop"
    )
    _emit(
        {
            "mode": "run",
            "target": target,
            "run_id": run_id,
            "matched": bool(matched),
            "sentinel": str(path) if path else None,
            "running": running,
        },
        output_json=output_json,
    )


def register_stop_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all stop`` command to the top-level CLI group."""
    cli_group.add_command(stop_command)
