"""``ed4all assistant`` — sandboxed operator-assistant chat (REPL + one-shot).

A THIN consumer of :class:`lib.assistant.engine.AssistantEngine` — the
engine owns the sandbox (system prompt, typed tool whitelist, loopback
guard, seat lazy-start policy, DecisionCapture, 6-round tool-loop cap), so
this module only does terminal I/O and exit codes. The Control-Plane GUI
consumes the same engine, inheriting the identical sandbox.

Backed by the LOCAL Nemotron nano vLLM seat (``ED4ALL_ASSISTANT_BASE_URL``,
loopback-only). When the seat is down:

* ``--no-seat-start`` → exit 1 with the start instructions (autostart
  forbidden even if ``ED4ALL_ASSISTANT_AUTOSTART`` is truthy).
* ``ED4ALL_ASSISTANT_AUTOSTART`` truthy → lazy-start the ``spark-nano``
  seat via the lifecycle lib (liveness ceiling + coherence probes).
* otherwise → exit 1 telling the operator how to start it.

DEBUG MODE (``--debug [--run WF-...]``): builds a bounded diagnostic
context for the given failed run (or the most recent failed run when
``--run`` is omitted — the campaign driver's ``last_failure.json`` pointer
wins, then the newest FAILED state/workflows entry), prints a one-paragraph
failure-summary banner, and starts the session with the root-cause-diagnosis
system prompt. Same tool whitelist; no new capabilities.

CAMPAIGN MODE (``--campaign``): starts the REPL/one-shot session with the
campaign engine (``AssistantEngine(mode="campaign")``) — the bounded
organize/arrange/monitor tool surface for the multi-book campaign. Mutually
exclusive with ``--debug``. The one-shot (``--campaign-tick``) drives one
deterministic pilot tick and prints its JSON result; it does NOT chat and
takes no question.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import click


def _run_campaign_tick(pilot_path: Path) -> Any:
    """Load the operator campaign harness' ``pilot.py`` off disk and run
    one tick.

    Kept a module-level helper (never a model-reachable capability) so tests
    can monkeypatch it or point ``CAMPAIGN_DIR`` at a tmp checkout with a stub
    ``pilot.py``. The loader mirrors the debug-context lazy-import discipline:
    the pilot is dev-authored deterministic code, not something the assistant
    ever writes.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "ed4all_campaign_pilot", str(pilot_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load campaign pilot module at {pilot_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run_tick()


@click.command("assistant")
@click.argument("question", required=False, default=None)
@click.option(
    "--once",
    "once_text",
    default=None,
    help="One-shot: answer this question and exit (same as passing QUESTION).",
)
@click.option(
    "--no-seat-start",
    is_flag=True,
    help=(
        "Forbid lazy seat start: if the assistant seat is down, exit with "
        "instructions instead of autostarting it (overrides "
        "ED4ALL_ASSISTANT_AUTOSTART)."
    ),
)
@click.option(
    "--debug",
    "debug_mode",
    is_flag=True,
    help=(
        "Start in DEBUG MODE: seed the session with a failed run's "
        "diagnostic context (run report, failing-phase gates, log tail, "
        "doctor post-mortem) and shift the assistant to root-cause "
        "diagnosis."
    ),
)
@click.option(
    "--run",
    "debug_run_id",
    default=None,
    help=(
        "With --debug: the failed run id (WF-YYYYMMDD-xxxxxxxx) to debug. "
        "Omitted → the most recent failed run is auto-resolved."
    ),
)
@click.option(
    "--campaign",
    "campaign_mode",
    is_flag=True,
    help=(
        "Start in CAMPAIGN MODE: the bounded organize/arrange/monitor tool "
        "surface for the multi-book campaign (assistant organizes, Claude "
        "fixes). Mutually exclusive with --debug."
    ),
)
@click.option(
    "--campaign-tick",
    "campaign_tick",
    is_flag=True,
    help=(
        "Run ONE deterministic campaign pilot tick (non-interactive) and "
        "print its JSON result. Takes no question and cannot combine with "
        "--once/--campaign/--debug."
    ),
)
def assistant_command(
    question: Optional[str],
    once_text: Optional[str],
    no_seat_start: bool,
    debug_mode: bool,
    debug_run_id: Optional[str],
    campaign_mode: bool,
    campaign_tick: bool,
) -> None:
    """Chat with the Ed4All operator assistant (local nano seat, sandboxed).

    With QUESTION (or --once), answers once and exits; otherwise starts an
    interactive REPL. The assistant can report campaign/run/seat status,
    read run/gate/cost/aggregator reports and bounded log tails, answer
    grounded questions about library courses, and perform whitelisted
    actions (start/resume/stop runs, seat lifecycle, support bundle) —
    nothing else (no shell, no file access).
    """
    # --campaign-tick is a standalone non-interactive verb: no chat, no
    # engine, mutually exclusive with everything else. Handle it first so the
    # heavy chat imports never load on this path.
    if campaign_tick:
        if (
            question is not None
            or once_text is not None
            or campaign_mode
            or debug_mode
        ):
            click.secho(
                "--campaign-tick runs a single non-interactive tick: it takes "
                "no question and cannot combine with --once/--campaign/--debug.",
                fg="red",
                err=True,
            )
            sys.exit(2)
        import json

        # CAMPAIGN_DIR is the only campaign path constant this file touches;
        # imported lazily (cli/ convention) — no campaign slugs/books here.
        from lib.assistant.campaign_tools import CAMPAIGN_DIR

        pilot_path = CAMPAIGN_DIR / "pilot.py"
        if not pilot_path.exists():
            click.secho(
                "no campaign pilot on this checkout "
                f"(expected {pilot_path}).",
                fg="red",
                err=True,
            )
            sys.exit(2)
        try:
            result = _run_campaign_tick(pilot_path)
        except Exception as exc:  # noqa: BLE001 — surface any tick failure loudly
            click.secho(f"campaign tick failed: {exc}", fg="red", err=True)
            sys.exit(1)
        click.echo(json.dumps(result, indent=2, default=str))
        return

    # Heavy imports inside the body per cli/ convention (`--help` stays fast).
    from lib.assistant.client import (
        AssistantProviderNotLocal,
        AssistantSeatUnavailable,
    )
    from lib.assistant.engine import AssistantEngine

    if campaign_mode and debug_mode:
        click.secho(
            "--campaign and --debug are mutually exclusive.", fg="red", err=True
        )
        sys.exit(2)

    if debug_run_id is not None and not debug_mode:
        click.secho("--run requires --debug.", fg="red", err=True)
        sys.exit(2)

    debug_context_text: Optional[str] = None
    banner: Optional[str] = None
    if debug_mode:
        from lib.assistant.debug_context import (
            DebugContextUnavailable,
            build_debug_context,
        )

        try:
            context = build_debug_context(debug_run_id)
        except DebugContextUnavailable as exc:
            click.secho(f"debug mode unavailable: {exc}", fg="red", err=True)
            sys.exit(2)
        debug_context_text = context["text"]
        banner = context["summary"]

    if campaign_mode:
        engine_mode = "campaign"
    elif debug_mode:
        engine_mode = "debug"
    else:
        engine_mode = "operator"

    try:
        engine = AssistantEngine(
            mode=engine_mode,
            debug_context=debug_context_text,
        )
    except AssistantProviderNotLocal as exc:
        click.secho(str(exc), fg="red", err=True)
        sys.exit(2)

    try:
        engine.ensure_seat(allow_autostart=not no_seat_start)
    except AssistantSeatUnavailable as exc:
        click.secho(str(exc), fg="red", err=True)
        sys.exit(1)

    if banner:
        click.secho("FAILURE SUMMARY: " + banner, fg="yellow")

    one_shot = once_text if once_text is not None else question
    if one_shot is not None:
        try:
            turn = engine.run_turn(one_shot)
        except AssistantSeatUnavailable as exc:
            click.secho(str(exc), fg="red", err=True)
            sys.exit(1)
        click.echo(turn.reply)
        return

    # Interactive REPL.
    if campaign_mode:
        mode_tag = "[CAMPAIGN MODE] "
    elif debug_mode:
        mode_tag = "[DEBUG MODE] "
    else:
        mode_tag = ""
    click.echo(
        "Ed4All operator assistant "
        + mode_tag
        + f"(model {engine.client.model} @ {engine.client.base_url}). "
        "Type 'exit' or Ctrl-D to quit."
    )
    history = None
    while True:
        try:
            user_text = click.prompt("ed4all", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt, click.Abort):
            click.echo("bye")
            return
        if user_text.strip().lower() in {"exit", "quit", "q"}:
            click.echo("bye")
            return
        if not user_text.strip():
            continue
        try:
            turn = engine.run_turn(user_text, history=history)
        except AssistantSeatUnavailable as exc:
            click.secho(str(exc), fg="red", err=True)
            sys.exit(1)
        history = turn.messages
        for call in turn.tool_calls:
            click.secho(f"  [tool] {call['tool']}", fg="cyan", err=True)
        # Campaign mode surfaces which seat/model answered (dynamic resolver),
        # mirroring the [tool] echo style — dim so it stays out of the way.
        if campaign_mode and getattr(turn, "model", None):
            click.secho(
                f"  [seat] {turn.model} @ {getattr(turn, 'seat_name', None)}",
                fg="cyan",
                dim=True,
                err=True,
            )
        click.echo(turn.reply)


def register_assistant_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all assistant`` command to the top-level CLI group."""
    cli_group.add_command(assistant_command)
