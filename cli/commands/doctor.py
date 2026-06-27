"""``ed4all doctor`` — multi-check preflight diagnostics.

A standalone, NEVER-raising preflight that iterates the pluggable
diagnostics registry (:mod:`lib.diagnostics`) and renders / exit-codes
whatever the registered checks report. The command owns ZERO diagnostic
policy — it bootstraps the registry, runs the checks for a
:class:`~lib.diagnostics.CheckContext`, and formats the results.

Registered groups (each contributed by a check module that does NOT
self-register — the bootstrap wires them in explicitly):

* ``gpu`` — GPU/VRAM-contention fit prediction (wraps
  :mod:`lib.llm.vram_doctor`; the historical VRAM-only doctor).
* ``window`` — serving-window / context-budget checks.
* ``environment`` — environment / provider / dependency checks.

The body is wrapped defensively so the command can never raise to the
operator (the foundation already never raises; this is
belt-and-suspenders).

Exit codes (for preflight scripting / build gating) — owned by
:func:`lib.diagnostics.resolve_exit_code`:

* ``2`` — DANGER: at least one check FAILed (e.g. a consumer predicted to
  OOM); abort before launching.
* ``1`` — DEGRADED: at least one check WARNed (e.g. a cuda→CPU fallback —
  safe but slow); proceed knowingly.
* ``0`` — OK: every check passed.
"""

from __future__ import annotations

import json as _json
import sys

import click

from lib.diagnostics import (
    CheckContext,
    clear_registry,
    format_report,
    registered_checks,
    resolve_exit_code,
    resolve_verdict,
    results_to_json,
    run_checks,
)

# The three check modules do NOT self-register — importing them is
# side-effect-free; the bootstrap calls their register_* fns explicitly.
from lib.diagnostics.environment import register_environment_checks
from lib.diagnostics.serving_window import register_serving_window_checks
from lib.diagnostics.vram import register_gpu_checks


def _gpu_backcompat_payload(results) -> tuple[dict | None, list[dict]]:
    """Reconstruct the pre-Phase-1 ``snapshot`` + ``verdicts`` from gpu results.

    Best-effort (NEVER raises): derives the two legacy VRAM-only ``--json``
    keys from whatever the ``gpu`` group reported, so scripts reading
    ``payload["snapshot"]["free_mib"]`` or iterating ``payload["verdicts"]``
    keep working. Returns ``(None, [])`` when the gpu group did not run
    (e.g. ``--group window`` only) — it does NOT probe a second time.

    * ``snapshot`` — the ``gpu_snapshot`` result's ``data`` (already the
      ``{free_mib,total_mib,cuda_available,probe_source,resident_models,error}``
      shape; skipped for the import/probe-error fallback that carries only
      ``error``).
    * ``verdicts`` — one dict per ``gpu_fit_*`` result, mapped back to the
      prior ``FitVerdict`` asdict shape
      (``consumer/device_requested/need_mib/free_mib/outcome/detail``).
    """
    snapshot: dict | None = None
    verdicts: list[dict] = []
    try:
        for r in results:
            if getattr(r, "group", None) != "gpu":
                continue
            data = getattr(r, "data", None) or {}
            if r.name == "gpu_snapshot" and "free_mib" in data:
                snapshot = {
                    "free_mib": data.get("free_mib"),
                    "total_mib": data.get("total_mib"),
                    "cuda_available": data.get("cuda_available"),
                    "probe_source": data.get("probe_source"),
                    "resident_models": data.get("resident_models", []),
                    "error": data.get("error"),
                }
            elif "consumer" in data and "outcome" in data:
                verdicts.append(
                    {
                        "consumer": data.get("consumer"),
                        "device_requested": data.get("device"),
                        "need_mib": data.get("need_mib"),
                        "free_mib": data.get("free_mib"),
                        "outcome": data.get("outcome"),
                        "detail": getattr(r, "detail", ""),
                    }
                )
    except Exception:  # noqa: BLE001 — reconstruction is best-effort only
        return snapshot, verdicts
    return snapshot, verdicts


def _bootstrap_checks() -> None:
    """Register the built-in check groups (idempotent).

    Clears the registry first so repeated command invocations / tests do
    NOT double-register, then wires the three explicit per-group helpers
    (``gpu`` / ``window`` / ``environment``). Safe to call on every
    command invocation.
    """
    clear_registry()
    register_gpu_checks()
    register_serving_window_checks()
    register_environment_checks()


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
    help="Emit the check results + verdict as JSON instead of the formatted report.",
)
@click.option(
    "--group",
    "-g",
    "groups",
    multiple=True,
    help=(
        "Run only the named check group(s) (e.g. -g gpu -g window). "
        "Repeatable. Default (none given): run ALL groups."
    ),
)
def doctor_command(base_url: str | None, output_json: bool, groups: tuple[str, ...]) -> None:
    """Report build-preflight diagnostics across all registered check groups.

    Bootstraps the diagnostics registry, runs every check (optionally
    filtered to ``--group``) for the given ``--base-url``, then renders the
    grouped report (or JSON). Never raises; exit code gates a preflight
    script (2=DANGER any-FAIL, 1=DEGRADED any-WARN, 0=OK).
    """
    try:
        _bootstrap_checks()

        # Finding #2: validate every requested --group against the set of
        # registered group names. An unknown/typo'd group must NOT silently
        # run zero checks and exit 0/OK (a build-gating script would treat
        # the typo as a passing preflight) — fail loud with exit 2.
        valid_groups = {g for g, _ in registered_checks()}
        unknown = [g for g in groups if g not in valid_groups]
        if unknown:
            click.secho(
                "ed4all doctor: unknown --group value(s): "
                f"{', '.join(sorted(set(unknown)))}. "
                f"Valid groups: {', '.join(sorted(valid_groups))}.",
                fg="red",
                err=True,
            )
            sys.exit(2)

        ctx = CheckContext(base_url=base_url)
        results = run_checks(ctx, groups=groups or None)
        exit_code = resolve_exit_code(results)

        if output_json:
            # snapshot/verdicts are back-compat with the pre-Phase-1
            # VRAM-only --json shape ({snapshot, verdicts, exit_code,
            # summary}); derived from the gpu group when it ran, else
            # null/empty (no second probe).
            snapshot, verdicts = _gpu_backcompat_payload(results)
            payload = {
                "results": results_to_json(results),
                "snapshot": snapshot,
                "verdicts": verdicts,
                "exit_code": exit_code,
                "summary": resolve_verdict(results),
            }
            click.echo(_json.dumps(payload, indent=2))
        else:
            click.echo(format_report(results))

        sys.exit(exit_code)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — the doctor must never crash the operator
        click.secho(f"ed4all doctor: unexpected error: {exc}", fg="red", err=True)
        sys.exit(1)


def register_doctor_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all doctor`` command to the top-level CLI group."""
    cli_group.add_command(doctor_command)
