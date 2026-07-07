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
* ``provider`` — provider/seat run-preflight (which provider every
  authoring/synthesis/answer/embedding seat resolves to for a run, and is
  it correct + reachable). EXCLUDED from the bare default — it is a
  run-preflight concern, surfaced only via ``--run`` / ``--ping`` /
  ``--group provider`` so the default doctor stays focused.

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
import os
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

# The check modules do NOT self-register — importing them is
# side-effect-free; the bootstrap calls their register_* fns explicitly.
from lib.diagnostics.environment import register_environment_checks
from lib.diagnostics.postmortem import register_postmortem_checks
from lib.diagnostics.provider import register_provider_checks
from lib.diagnostics.run_env import applied_run_env
from lib.diagnostics.serving_window import register_serving_window_checks
from lib.diagnostics.vram import register_gpu_checks

# Reused, pure, side-effect-free helpers from the sibling ``run`` command
# (same CLI layer — importing across cli.commands is fine).
# ``_nvidia_preflight`` (the deprecated compat alias for
# ``_cloud_seat_preflight``) resolves+asserts the hosted-large routing off env
# (no dispatch); the workflow normaliser + supported set validate ``--run``.
# Imported as module globals so tests can monkeypatch
# ``doctor_mod._nvidia_preflight`` / ``applied_run_env``.
from cli.commands.run import (
    COURSEFORGE_STAGE_SUBCOMMANDS,
    SUPPORTED_WORKFLOWS,
    _nvidia_preflight,
    _normalize_workflow,
)

#: The three always-on groups. ``provider`` is intentionally NOT here — it is
#: a run-preflight concern, opted into via ``--run`` / ``--ping`` /
#: ``--group provider`` (keeps the bare default free of class-default seat
#: noise).
_DEFAULT_GROUPS = ("gpu", "window", "environment")
_RUN_GROUPS = ("gpu", "window", "environment", "provider")

#: Post-mortem mode (``--run-id``) is a DISTINCT mode — it analyzes a PAST run
#: off disk (checkpoints + VRAM trajectory) and does ZERO live-env probing. It
#: runs ONLY this group; it is mutually exclusive with the preflight flags.
_POSTMORTEM_GROUPS = ("postmortem",)


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
    NOT double-register, then wires the five explicit per-group helpers
    (``gpu`` / ``window`` / ``environment`` / ``provider`` / ``postmortem``).
    Safe to call on every command invocation. NB: registering ``provider``
    makes ``--group provider`` valid; it is still EXCLUDED from the bare
    default group selection (run-preflight only). Likewise ``postmortem`` is
    reachable via ``--group postmortem`` / ``--run-id`` only (past-run forensic
    mode), never in the bare default.
    """
    clear_registry()
    register_gpu_checks()
    register_serving_window_checks()
    register_environment_checks()
    register_provider_checks()
    register_postmortem_checks()


def _compute_cloud_seat_preflight(workflow: str, provider: str | None) -> dict | None:
    """Best-effort hosted-large routing preflight UNDER the run fanout.

    Only meaningful when the run's effective provider could be ``nvidia`` (the
    vendor endpoint-registry key — the ``--provider`` hint or ``LLM_PROVIDER``
    env) — ``_cloud_seat_preflight`` always reports a cloud-seat verdict off
    env, so computing it for a non-nvidia run would attach misleading FAILs.
    Computed inside :func:`applied_run_env` so it reflects the env a real
    ``ed4all run`` would apply. NEVER raises — any failure degrades to ``None``
    (the provider check is fine without it).
    """
    effective = (provider or os.environ.get("LLM_PROVIDER") or "").strip().lower()
    if effective != "nvidia":
        return None
    try:
        # _cloud_seat_preflight reads only env + flags; params are nominal.
        params = {"course_name": "doctor-preflight", "provider": provider or ""}
        with applied_run_env(workflow, provider or ""):
            return _nvidia_preflight(params)
    except Exception:  # noqa: BLE001 — preflight must never break the command
        return None


# Deprecated compat alias for the renamed helper.
_compute_nvidia_preflight = _compute_cloud_seat_preflight


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
        "Run only the named check group(s) (e.g. -g gpu -g window -g provider). "
        "Repeatable. Default (none given): gpu/window/environment (provider is "
        "opt-in via --run/--ping/-g provider)."
    ),
)
@click.option(
    "--run",
    "-r",
    "run_workflow",
    default=None,
    help=(
        "Model the provider/seat preflight for a specific workflow run (e.g. "
        "--run textbook_to_course). Adds the ``provider`` group and resolves "
        "every authoring/synthesis/answer/embedding seat UNDER the run fanout. "
        "Unknown workflow → error + exit 2."
    ),
)
@click.option(
    "--provider",
    "provider",
    default=None,
    help=(
        "The run's --provider hint (e.g. nvidia / anthropic / local) used when "
        "modelling the provider/seat preflight. Only consulted with --run / "
        "--ping. Default: unset (the fanout's single-switch local default)."
    ),
)
@click.option(
    "--mode",
    "mode",
    type=click.Choice(["local", "api"]),
    default=None,
    help="The run's execution mode for the provider preflight. Default: local.",
)
@click.option(
    "--ping",
    "ping",
    is_flag=True,
    default=False,
    help=(
        "Opt-in reachability probe: do a real 1-token network call per distinct "
        "OpenAI-compatible provider seat. Adds the ``provider`` group. Off by "
        "default (the rest of the doctor makes NO network calls)."
    ),
)
@click.option(
    "--run-id",
    "run_id",
    default=None,
    help=(
        "Post-mortem mode: forensically analyze a PAST run by its id (reads the "
        "run's persisted checkpoints + VRAM trajectory off disk). Runs ONLY the "
        "``postmortem`` group — does NO live-env probing (no GPU / ollama / "
        "network). Mutually exclusive with --run / --ping (post-mortem vs "
        "preflight are different modes). Exit 2 if the analyzed run failed."
    ),
)
def doctor_command(
    base_url: str | None,
    output_json: bool,
    groups: tuple[str, ...],
    run_workflow: str | None,
    provider: str | None,
    mode: str | None,
    ping: bool,
    run_id: str | None,
) -> None:
    """Report build-preflight diagnostics across the registered check groups.

    Bootstraps the diagnostics registry, runs each check (optionally filtered
    to ``--group``) for the given ``--base-url``, then renders the grouped
    report (or JSON). With ``--run <workflow>`` (or ``--ping``) it also runs
    the ``provider`` group — modelling which provider every seat resolves to
    for that run. With ``--run-id <id>`` it switches to POST-MORTEM mode:
    runs ONLY the ``postmortem`` group, analyzing a past run off disk (no
    live-env probing); the exit code then reflects that run's OUTCOME (2 if it
    failed). Never raises; exit code gates a preflight script
    (2=DANGER any-FAIL, 1=DEGRADED any-WARN, 0=OK).
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

        # Post-mortem mode (--run-id): a DISTINCT mode that analyzes a PAST
        # run off disk. It runs ONLY the ``postmortem`` group and does ZERO
        # live-env probing (no GPU / ollama / network). It is mutually
        # exclusive with the preflight flags (--run / --ping), and forces the
        # postmortem group (an explicit conflicting --group is a loud error).
        if run_id is not None:
            if run_workflow or ping:
                click.secho(
                    "ed4all doctor: --run-id (post-mortem of a past run) is "
                    "mutually exclusive with --run / --ping (live-env "
                    "preflight). Pass one mode or the other.",
                    fg="red",
                    err=True,
                )
                sys.exit(2)
            conflicting = [g for g in groups if g != "postmortem"]
            if conflicting:
                click.secho(
                    "ed4all doctor: --run-id forces the 'postmortem' group; "
                    "remove the conflicting --group value(s): "
                    f"{', '.join(sorted(set(conflicting)))}.",
                    fg="red",
                    err=True,
                )
                sys.exit(2)

            ctx = CheckContext(base_url=base_url, run_id=run_id)
            results = run_checks(ctx, groups=_POSTMORTEM_GROUPS)
            exit_code = resolve_exit_code(results)

            if output_json:
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

        # Validate --run against the supported workflow set (mirror the
        # unknown-group pattern: fail loud, exit 2, run nothing).
        workflow: str | None = None
        if run_workflow:
            normalized = _normalize_workflow(run_workflow)
            supported = {_normalize_workflow(w) for w in SUPPORTED_WORKFLOWS}
            if normalized not in supported:
                click.secho(
                    "ed4all doctor: unknown --run workflow: "
                    f"{run_workflow!r}. "
                    f"Valid workflows: {', '.join(sorted(supported))}.",
                    fg="red",
                    err=True,
                )
                sys.exit(2)
            # Finding #7: mirror the run command's aliasing. ``ed4all run``
            # aliases every courseforge-* stage subcommand back to
            # ``textbook_to_course`` BEFORE the fanout fires (see
            # cli/commands/run.py: ``if workflow in COURSEFORGE_STAGE_SUBCOMMANDS:
            # workflow = "textbook_to_course"``). The doctor must model the
            # SAME canonical workflow's fanout, so the value fed to run_config /
            # applied_run_env is the aliased one — not the un-aliased subcommand.
            if normalized in COURSEFORGE_STAGE_SUBCOMMANDS:
                workflow = "textbook_to_course"
            else:
                workflow = normalized

        # Build the provider-check run_config when modelling a run (--run)
        # or probing reachability (--ping). Bare (neither) → None, and the
        # provider group resolves seats off the CURRENT env (caveated).
        run_config: dict | None = None
        if run_workflow or ping:
            run_config = {
                "workflow": workflow,  # None in bare+ping mode
                "provider_hint": provider or "",
                "mode": mode or "local",
                "ping": bool(ping),
            }
            if run_workflow:
                # Best-effort, never breaks the command (→ None on failure).
                # Emit under the new ``cloud_seat_preflight`` key AND the legacy
                # ``nvidia_preflight`` key for one release (provider checks read
                # new-then-old; old post-mortem sidecars carry the legacy key).
                _pf = _compute_cloud_seat_preflight(workflow, provider)
                run_config["cloud_seat_preflight"] = _pf
                run_config["nvidia_preflight"] = _pf

        # Group selection: explicit --group wins; else --run/--ping add the
        # provider group; else the bare default (provider EXCLUDED).
        # Finding #9: --run/--ping NEED the provider group (it owns the only
        # network probe). When an explicit --group list omits ``provider`` we
        # union it in rather than silently filtering the ping/run preflight out
        # to a no-op — the operator asked for a probe, they get one.
        if groups:
            selected_groups = groups
            if (run_workflow or ping) and "provider" not in selected_groups:
                selected_groups = (*selected_groups, "provider")
        elif run_workflow or ping:
            selected_groups = _RUN_GROUPS
        else:
            selected_groups = _DEFAULT_GROUPS

        ctx = CheckContext(base_url=base_url, run_config=run_config)
        results = run_checks(ctx, groups=selected_groups)
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
