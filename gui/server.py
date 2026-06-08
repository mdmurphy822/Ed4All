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
    args = parser.parse_args(argv)

    # Import uvicorn lazily so importing this module (e.g. to grab create_app)
    # doesn't hard-require the server runtime.
    import uvicorn  # noqa: PLC0415

    if args.reload:
        # Reload mode needs an import string, not an app instance.
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
            create_app(),
            host=args.host,
            port=args.port,
            log_level=args.log_level,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
