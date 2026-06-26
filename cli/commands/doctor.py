"""``ed4all doctor`` — GPU/VRAM-contention preflight check.

A standalone, NEVER-raising preflight that snapshots the GPU's true
free/total VRAM (NVML-first, WSL2-correct), lists the local ollama models
resident on the card, and predicts whether each in-process GPU consumer
(the NLI classifier and the embedding provider) will FIT, EVICT-then-fit,
fall back to CPU, or OOM *before* a long build kicks off.

The whole command is a thin CLI shell over the
:mod:`lib.llm.vram_doctor` foundation — it owns ZERO new device policy and
only renders/exit-codes what the foundation reports. The body is wrapped
defensively so the command can never raise to the operator (the foundation
already never raises; this is belt-and-suspenders).

Exit codes (for preflight scripting / build gating):

* ``2`` — DANGER: at least one consumer is predicted to ``would_oom``
  (the build would likely crash mid-run); abort before launching.
* ``1`` — DEGRADED: at least one cuda-requested consumer is predicted to
  ``would_fallback_cpu`` (safe but slow — CPU NLI/embeddings are ~20-50x
  slower); proceed knowingly.
* ``0`` — OK: every consumer fits, runs on CPU by request, or would
  evict-then-fit.
"""

from __future__ import annotations

import dataclasses
import json as _json
import sys

import click

from lib.llm.vram_doctor import (
    fit_check,
    format_doctor_report,
    snapshot_vram,
)


def _resolve_exit(verdicts) -> tuple[int, str]:
    """Map fit-check verdicts to (exit_code, one-line summary).

    ``would_oom`` anywhere → exit 2 (DANGER). A cuda-requested
    ``would_fallback_cpu`` → exit 1 (DEGRADED). Otherwise exit 0 (OK).
    Kept deliberately small and obvious — the gating contract.
    """
    oom = [v for v in verdicts if getattr(v, "outcome", None) == "would_oom"]
    if oom:
        names = ", ".join(v.consumer for v in oom)
        return 2, f"DANGER: would OOM ({names}) — abort the build, free VRAM first."

    degraded = [
        v
        for v in verdicts
        if getattr(v, "outcome", None) == "would_fallback_cpu"
        and getattr(v, "device_requested", "cpu") != "cpu"
    ]
    if degraded:
        names = ", ".join(v.consumer for v in degraded)
        return 1, f"DEGRADED: cuda requested but CPU fallback ({names}) — safe but slow."

    return 0, "OK"


@click.command("doctor")
@click.option(
    "--base-url",
    "base_url",
    default=None,
    help=(
        "Base URL for the ollama resident-model probe. Default: unset → "
        "resolves LOCAL_SYNTHESIS_BASE_URL."
    ),
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Emit the snapshot + verdicts as JSON instead of the formatted report.",
)
def doctor_command(base_url: str | None, output_json: bool) -> None:
    """Report GPU/VRAM-contention state before a build (preflight).

    Snapshots free/total VRAM + resident ollama models, then predicts
    placement (fits / evict-then-fit / CPU-fallback / OOM) for the NLI and
    embedding consumers. Never raises; exit code gates a preflight script
    (2=DANGER would-OOM, 1=DEGRADED cuda→CPU fallback, 0=OK).
    """
    try:
        snapshot = snapshot_vram(base_url=base_url)
        verdicts = fit_check(snapshot)
        exit_code, summary = _resolve_exit(verdicts)

        if output_json:
            payload = {
                "snapshot": dataclasses.asdict(snapshot),
                "verdicts": [dataclasses.asdict(v) for v in verdicts],
                "exit_code": exit_code,
                "summary": summary,
            }
            click.echo(_json.dumps(payload, indent=2))
        else:
            click.echo(format_doctor_report(snapshot, verdicts))
            click.echo("")
            color = {2: "red", 1: "yellow", 0: "green"}.get(exit_code, "white")
            click.secho(summary, fg=color)

        sys.exit(exit_code)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — the doctor must never crash the operator
        click.secho(f"ed4all doctor: unexpected error: {exc}", fg="red", err=True)
        sys.exit(1)


def register_doctor_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all doctor`` command to the top-level CLI group."""
    cli_group.add_command(doctor_command)
