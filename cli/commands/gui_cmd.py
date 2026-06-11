"""``ed4all gui`` — launch the control-plane GUI web server.

Runs uvicorn against the FastAPI app factory ``gui.app:create_app``
(``factory=True``), serving the vanilla-JS SPA + ``/api/*`` routers + the
``/ws/runs/{run_id}`` log stream described in the GUI build spec.

The ``uvicorn`` import is deferred into the command callback so this module is
import-safe on a default install (the ``gui`` extra ships fastapi/uvicorn). The
command surfaces a typed, actionable error if the extra is missing rather than
crashing at import time.
"""

from __future__ import annotations

import os

import click

from gui import DEFAULT_HOST, DEFAULT_PORT


@click.command("gui")
@click.option("--host", default=DEFAULT_HOST, show_default=True, help="Bind host.")
@click.option("--port", default=DEFAULT_PORT, show_default=True, type=int, help="Bind port.")
@click.option(
    "--reload/--no-reload",
    "reload",
    default=False,
    show_default=True,
    help="Enable uvicorn autoreload (development only).",
)
@click.option(
    "--learner",
    "learner",
    is_flag=True,
    default=False,
    show_default=True,
    help=(
        "Serve ONLY the learner answer surface (/learn/ + /api/learn/*); the "
        "operator SPA and settings/uploads/runs/courses/retrieval APIs are not "
        "mounted. Env fallback: ED4ALL_GUI_LEARNER=1. For moderated pilot "
        "sessions only — the GUI has no auth, so keep the loopback bind. "
        "Equivalent to --mode learner; --mode wins if both are passed."
    ),
)
@click.option(
    "--mode",
    "mode",
    type=click.Choice(["full", "studio", "learner"]),
    default=None,
    help=(
        "Serve mode. 'full' (default): operator + Studio + learner. 'studio': "
        "the end-user Library + IMSCC course viewer (/studio/ + /api/library + "
        "/api/courses/* + /api/learn/*); operator APIs NOT mounted. 'learner': "
        "the answer surface only. Env fallback: ED4ALL_GUI_MODE (wins over the "
        "legacy ED4ALL_GUI_LEARNER). The GUI has no auth — keep the loopback bind."
    ),
)
@click.pass_context
def gui_command(
    ctx: click.Context, host: str, port: int, reload: bool, learner: bool,
    mode: str | None,
) -> None:
    """Launch the Ed4All control-plane GUI (FastAPI + uvicorn).

    Serves the SPA at ``http://<host>:<port>/`` and the REST/WebSocket API under
    ``/api`` and ``/ws``. Requires the ``gui`` extra::

        pip install 'ed4all[gui]'
        ed4all gui --host 127.0.0.1 --port 8077

    If ``--port`` is left at its default and that port is busy, the next free
    port is chosen automatically (same scan ``gui/launch.py`` uses). If an
    explicit ``--port`` is passed and it is busy, the command fails with an
    actionable message rather than a raw bind traceback.
    """
    try:
        import uvicorn  # noqa: PLC0415
    except ImportError:
        raise click.ClickException(
            "The 'gui' extra is not installed. Install it with: "
            "pip install 'ed4all[gui]'"
        ) from None

    # Reuse the launcher's stdlib-only free-port scanner so ``ed4all gui`` and
    # ``gui/launch.py`` agree on port selection.
    from gui.launch import _port_is_free, pick_port  # noqa: PLC0415

    port_was_explicit = ctx.get_parameter_source("port") != click.core.ParameterSource.DEFAULT

    if port_was_explicit:
        if not _port_is_free(host, port):
            raise click.ClickException(
                f"Port {port} on {host} is already in use. Free it, or omit "
                "--port to let Ed4All pick the next free port automatically."
            )
        chosen = port
    else:
        chosen = pick_port(host, port)

    # ``ed4all gui`` always launches via the import-string factory
    # (``gui.app:create_app``, ``factory=True``) so uvicorn can reload it; the
    # factory takes no args from an import string, so the serve-mode choice rides
    # the environment. ``ED4ALL_GUI_MODE`` is canonical (full|studio|learner);
    # ``ED4ALL_GUI_LEARNER`` stays honoured for backward compat. The CLI --mode
    # wins; --learner is the legacy alias for --mode learner.
    env_mode = os.environ.get("ED4ALL_GUI_MODE", "").strip().lower()
    env_learner = os.environ.get("ED4ALL_GUI_LEARNER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if mode:
        resolved = mode
    elif learner:
        resolved = "learner"
    elif env_mode in {"full", "studio", "learner"}:
        resolved = env_mode
    elif env_learner:
        resolved = "learner"
    else:
        resolved = "full"

    # Export the resolved mode so the reloaded factory workers honor it.
    os.environ["ED4ALL_GUI_MODE"] = resolved
    if resolved == "learner":
        os.environ["ED4ALL_GUI_LEARNER"] = "1"

    url = f"http://{host}:{chosen}/"
    surface = {
        "learner": "learner-only",
        "studio": "studio (end-user library + course viewer)",
        "full": "operator + learner",
    }[resolved]
    click.echo(f"Starting Ed4All control-plane GUI ({surface}) on {url}")
    uvicorn.run(
        "gui.app:create_app",
        host=host,
        port=chosen,
        reload=reload,
        factory=True,
    )


def register_gui_command(cli_group: click.Group) -> None:
    """Attach the ``ed4all gui`` command to the top-level CLI group."""
    cli_group.add_command(gui_command)
