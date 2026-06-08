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
@click.pass_context
def gui_command(ctx: click.Context, host: str, port: int, reload: bool) -> None:
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

    url = f"http://{host}:{chosen}/"
    click.echo(f"Starting Ed4All control-plane GUI on {url}")
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
