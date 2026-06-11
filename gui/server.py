"""Uvicorn entrypoint for the Ed4All control-plane GUI.

Run with ``python -m gui.server`` (or via ``ed4all gui``). Exposes
``create_app`` (re-exported from :mod:`gui.app`) and ``main()``.
"""

from __future__ import annotations

import argparse
import ipaddress
import logging
import os
from typing import List, Optional

from gui import DEFAULT_HOST, DEFAULT_PORT
from gui.app import create_app

logger = logging.getLogger("gui.server")


def _is_loopback_host(host: str) -> bool:
    """True if ``host`` is a loopback bind (no non-local exposure).

    ``localhost`` + the IPv4/IPv6 loopback ranges count as loopback. An empty
    host or ``0.0.0.0`` / ``::`` (bind-all) is NOT loopback. Unparseable hosts
    are treated as non-loopback (fail safe → warn).
    """
    h = (host or "").strip().lower()
    if h in {"localhost", ""}:
        return h == "localhost"
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _warn_if_exposed_without_token(host: str, resolved_mode: Optional[str]) -> None:
    """Log a WARNING when binding non-loopback in full mode without a token.

    The operator token gate (``gui.auth``) only protects the full-mode operator
    surface, and only when a token is configured. Binding the full operator
    surface to a non-loopback address with no ``ED4ALL_GUI_TOKEN`` leaves env /
    API-key management + run-launching open to the LAN — warn loudly. Studio /
    learner modes don't mount the operator surface, so they're exempt.
    """
    from gui.auth import _settings_token, resolve_token  # noqa: PLC0415

    mode = resolved_mode or "full"
    if mode != "full":
        return
    if _is_loopback_host(host):
        return
    if resolve_token(_settings_token()) is not None:
        return
    logger.warning(
        "GUI bound to non-loopback host %s in full (operator) mode with no "
        "%s set: the operator surface (env / API keys / run-launching) is "
        "UNAUTHENTICATED and reachable from the network. Set %s to require a "
        "bearer token, or bind to a loopback host.",
        host,
        "ED4ALL_GUI_TOKEN",
        "ED4ALL_GUI_TOKEN",
    )


def _env_learner_default() -> bool:
    """Resolve the ``ED4ALL_GUI_LEARNER`` env fallback for ``--learner``.

    Truthy values (``1``/``true``/``yes``/``on``, case-insensitive) default the
    serve mode to learner-only when ``--learner`` is not passed on the CLI.
    """
    return os.environ.get("ED4ALL_GUI_LEARNER", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Parse args and run the uvicorn server. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="gui.server",
        description="Ed4All control-plane GUI server.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("ED4ALL_GUI_HOST", DEFAULT_HOST),
        help=f"Bind host (default {DEFAULT_HOST}).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ED4ALL_GUI_PORT", DEFAULT_PORT)),
        help=f"Bind port (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only).",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        help="Uvicorn log level (default info).",
    )
    parser.add_argument(
        "--learner",
        action="store_true",
        default=_env_learner_default(),
        help=(
            "Serve ONLY the learner answer surface (/learn/ + /api/learn/*); the "
            "operator settings/uploads/runs/courses/retrieval APIs and SPA are not "
            "mounted. Env fallback: ED4ALL_GUI_LEARNER=1. Use for moderated pilot "
            "sessions; the GUI has no authentication, so keep the default loopback "
            "bind (never expose the operator surface where a learner can browse). "
            "Equivalent to --mode learner; --mode wins if both are passed."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["full", "studio", "learner"],
        default=None,
        help=(
            "Serve mode. 'full' (default): operator + Studio + learner. 'studio': "
            "the end-user Library + IMSCC course viewer (/studio/ + /api/library + "
            "/api/courses/* + /api/learn/*); operator APIs NOT mounted. 'learner': "
            "the answer surface only. Env fallback: ED4ALL_GUI_MODE (wins over the "
            "legacy ED4ALL_GUI_LEARNER). The GUI has no auth — keep the loopback bind."
        ),
    )
    args = parser.parse_args(argv)

    # Import uvicorn lazily so importing this module (e.g. to grab create_app)
    # doesn't hard-require the server runtime.
    import uvicorn  # noqa: PLC0415

    # Resolve the serve mode: --mode wins; else --learner (or its env default)
    # maps to learner; else full. The factory re-resolves from the env on the
    # reload path, so propagate the choice through ED4ALL_GUI_MODE there.
    resolved_mode = args.mode or ("learner" if args.learner else None)

    # Loud warning if the full operator surface is exposed beyond loopback with
    # no operator token configured (Marketable-v1 D2). Mode resolution mirrors
    # the factory: only full mode mounts the operator surface, so only full mode
    # warns. A None resolved_mode here means full (the factory default).
    _warn_if_exposed_without_token(args.host, resolved_mode)

    if args.reload:
        # Reload mode needs an import string, not an app instance — uvicorn
        # re-imports + calls the factory in each worker with no args, so the
        # mode choice rides the environment (the factory honors ED4ALL_GUI_MODE,
        # and ED4ALL_GUI_LEARNER for backward compat). Propagate the CLI choice
        # through the env so a reloaded worker keeps the same surface.
        if resolved_mode:
            os.environ["ED4ALL_GUI_MODE"] = resolved_mode
            if resolved_mode == "learner":
                os.environ["ED4ALL_GUI_LEARNER"] = "1"
        uvicorn.run(
            "gui.app:create_app",
            factory=True,
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
    else:
        uvicorn.run(
            create_app(learner_only=args.learner, mode=args.mode),
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
