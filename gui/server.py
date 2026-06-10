"""Uvicorn entrypoint for the Ed4All control-plane GUI.

Run with ``python -m gui.server`` (or via ``ed4all gui``). Exposes
``create_app`` (re-exported from :mod:`gui.app`) and ``main()``.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

from gui import DEFAULT_HOST, DEFAULT_PORT
from gui.app import create_app


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
            "bind (never expose the operator surface where a learner can browse)."
        ),
    )
    args = parser.parse_args(argv)

    # Import uvicorn lazily so importing this module (e.g. to grab create_app)
    # doesn't hard-require the server runtime.
    import uvicorn  # noqa: PLC0415

    if args.reload:
        # Reload mode needs an import string, not an app instance — uvicorn
        # re-imports + calls the factory in each worker with no args, so the
        # learner-only choice rides the environment (the factory honors
        # ED4ALL_GUI_LEARNER). Propagate the CLI flag through the env so a
        # reloaded worker keeps the same surface.
        if args.learner:
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
            create_app(learner_only=args.learner),
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
