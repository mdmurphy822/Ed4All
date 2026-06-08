#!/usr/bin/env python3
"""One-click setup-and-run launcher for the Ed4All control-plane GUI.

Dependency-light, stdlib-only bootstrapper. It runs on a *bare* ``python3``
with NO project dependencies installed, then:

1. Locates the repo root (the parent of this ``gui/`` directory).
2. Creates a dedicated venv at ``<repo>/.venv-gui`` (reusing an existing one).
3. Uses that venv's pip to ``pip install -e '.[gui]' '.[server]'`` so both the
   FastAPI/uvicorn web stack and the MCP tools import cleanly.
4. Starts uvicorn (``gui.app:create_app --factory``) on a free port.
5. Polls ``/api/health`` until the server answers, then opens a browser tab.
6. Keeps the server in the foreground; Ctrl-C stops it cleanly.

Because this module must execute before the project is installed, it imports
ONLY the Python standard library. Do not add third-party imports here.
"""

from __future__ import annotations

import argparse
import http.client
import os
import signal
import socket
import subprocess
import sys
import time
import venv
import webbrowser
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MIN_PYTHON = (3, 9)
VENV_DIRNAME = ".venv-gui"
INSTALLED_MARKER = ".gui-installed"

# Bind defaults. The canonical source of truth is ``gui/__init__.py``'s
# DEFAULT_HOST / DEFAULT_PORT, but this launcher runs on a bare ``python3``
# (pre-install) and is typically invoked as a script (``python3 gui/launch.py``),
# where ``import gui`` is not on the path. So we keep stdlib-only literals here
# and prefer the package values when (and only when) the package imports cleanly.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8077
try:  # best-effort: stay in sync with the package when it's importable
    from gui import DEFAULT_HOST as _PKG_HOST, DEFAULT_PORT as _PKG_PORT

    DEFAULT_HOST, DEFAULT_PORT = _PKG_HOST, _PKG_PORT
except Exception:  # noqa: BLE001 — pre-install / run-as-script: use the literals
    pass

PORT_ATTEMPTS = 10
HEALTH_PATH = "/api/health"
HEALTH_TIMEOUT_SECONDS = 60.0
HEALTH_POLL_INTERVAL = 0.5


# --------------------------------------------------------------------------- #
# Console helpers
# --------------------------------------------------------------------------- #


def _say(message: str) -> None:
    """Print a friendly status line, flushed so it interleaves with subprocess output."""
    print(f"[ed4all-gui] {message}", flush=True)


def _warn(message: str) -> None:
    print(f"[ed4all-gui] WARNING: {message}", file=sys.stderr, flush=True)


def _die(message: str, *, hint: str | None = None, code: int = 1):  # -> NoReturn
    print(f"[ed4all-gui] ERROR: {message}", file=sys.stderr, flush=True)
    if hint:
        print(f"[ed4all-gui]        {hint}", file=sys.stderr, flush=True)
    raise SystemExit(code)


# --------------------------------------------------------------------------- #
# Path / environment discovery
# --------------------------------------------------------------------------- #


def repo_root() -> Path:
    """Return the repo root: the parent of the ``gui/`` directory holding this file."""
    return Path(__file__).resolve().parent.parent


def venv_dir(root: Path) -> Path:
    return root / VENV_DIRNAME


def venv_python(vdir: Path) -> Path:
    """Return the path to the venv's python interpreter (cross-platform)."""
    if os.name == "nt":
        return vdir / "Scripts" / "python.exe"
    return vdir / "bin" / "python"


def marker_path(vdir: Path) -> Path:
    return vdir / INSTALLED_MARKER


# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        have = ".".join(str(p) for p in sys.version_info[:3])
        want = ".".join(str(p) for p in MIN_PYTHON)
        _die(
            f"Python {want}+ is required, but this interpreter is {have}.",
            hint="Install a newer Python and re-run, e.g. `python3.11 gui/launch.py`.",
        )


# --------------------------------------------------------------------------- #
# Venv creation + dependency install
# --------------------------------------------------------------------------- #


def ensure_venv(vdir: Path) -> Path:
    """Create the venv if missing; return the venv python path."""
    py = venv_python(vdir)
    if py.exists():
        _say(f"Reusing existing venv at {vdir}")
        return py

    _say(f"Creating virtual environment at {vdir} ...")
    try:
        venv.EnvBuilder(with_pip=True, clear=False).create(str(vdir))
    except Exception as exc:  # noqa: BLE001 - surface any venv failure as actionable
        _die(
            f"Failed to create virtual environment at {vdir}: {exc}",
            hint=(
                "Ensure the stdlib `venv` module works (on Debian/Ubuntu you may "
                "need `sudo apt install python3-venv`), then re-run."
            ),
        )

    if not py.exists():
        _die(
            f"Virtual environment was created but no interpreter was found at {py}.",
            hint=f"Delete {vdir} and re-run, or recreate it with `python -m venv {vdir}`.",
        )
    return py


def deps_installed(vdir: Path) -> bool:
    return marker_path(vdir).exists()


def install_deps(py: Path, root: Path, vdir: Path) -> None:
    """Install the project in editable mode with the gui + server extras."""
    _say("Upgrading pip in the venv ...")
    _run_streamed(
        [str(py), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=root,
        what="pip upgrade",
        fatal=False,
    )

    _say("Installing Ed4All with the [gui] and [server] extras (this may take a minute) ...")
    # Two extras in one install resolves the dependency graph once.
    _run_streamed(
        [str(py), "-m", "pip", "install", "-e", ".[gui]", "-e", ".[server]"],
        cwd=root,
        what="dependency install",
        fatal=True,
    )

    try:
        marker_path(vdir).write_text(
            f"installed-by gui/launch.py at {time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        _warn(f"Could not write install marker ({exc}); reinstall will run next time.")
    _say("Dependencies installed.")


def _run_streamed(cmd: list[str], *, cwd: Path, what: str, fatal: bool) -> None:
    """Run a subprocess, streaming its output to this console."""
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    except FileNotFoundError as exc:
        if fatal:
            _die(f"Command not found during {what}: {exc}")
        _warn(f"Command not found during {what}: {exc}")
        return
    if proc.returncode != 0:
        if fatal:
            _die(
                f"{what} failed (exit code {proc.returncode}).",
                hint="Scroll up for the pip output; fix the reported error and re-run.",
            )
        _warn(f"{what} returned non-zero exit code {proc.returncode}; continuing.")


# --------------------------------------------------------------------------- #
# Port selection
# --------------------------------------------------------------------------- #


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
        return True


def pick_port(host: str, preferred: int, attempts: int = PORT_ATTEMPTS) -> int:
    """Return the first free port at or above ``preferred``."""
    for offset in range(attempts):
        candidate = preferred + offset
        if candidate > 65535:
            break
        if _port_is_free(host, candidate):
            if offset:
                _say(f"Port {preferred} busy; using {candidate} instead.")
            return candidate
    _die(
        f"No free port found in range {preferred}-{preferred + attempts - 1} on {host}.",
        hint="Free a port or pass `--port <N>` to choose a different starting point.",
    )


# --------------------------------------------------------------------------- #
# Health polling + browser
# --------------------------------------------------------------------------- #


def _health_ok(host: str, port: int) -> bool:
    conn = http.client.HTTPConnection(host, port, timeout=2.0)
    try:
        conn.request("GET", HEALTH_PATH)
        resp = conn.getresponse()
        resp.read()
        return 200 <= resp.status < 300
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def wait_for_health(host: str, port: int, server: subprocess.Popen, timeout: float) -> bool:
    """Poll the health endpoint until it responds or the server exits / times out."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if server.poll() is not None:
            return False  # server died; caller reports the exit code
        if _health_ok(host, port):
            return True
        time.sleep(HEALTH_POLL_INTERVAL)
    return False


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


def start_server(py: Path, root: Path, host: str, port: int) -> subprocess.Popen:
    cmd = [
        str(py),
        "-m",
        "uvicorn",
        "gui.app:create_app",
        "--factory",
        "--host",
        host,
        "--port",
        str(port),
    ]
    _say(f"Starting GUI server: uvicorn gui.app:create_app --factory @ {host}:{port}")
    try:
        return subprocess.Popen(cmd, cwd=str(root))
    except FileNotFoundError as exc:
        _die(
            f"Could not launch the venv interpreter to start uvicorn: {exc}",
            hint="Try `--reinstall` to rebuild the venv.",
        )


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    _say("Stopping GUI server ...")
    try:
        if os.name == "nt":
            server.terminate()
        else:
            server.send_signal(signal.SIGTERM)
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _warn("Server did not stop in time; killing it.")
        server.kill()
    except Exception as exc:  # noqa: BLE001
        _warn(f"Error while stopping server: {exc}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="gui/launch.py",
        description="Set up a venv, install the GUI deps, and launch the Ed4All control-plane GUI.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host (default {DEFAULT_HOST}).")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Preferred port; next free one is used if busy (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser tab once the server is ready.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip the pip install step (assume the venv already has the deps).",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Force a fresh dependency install even if the install marker is present.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    check_python_version()

    root = repo_root()
    if not (root / "pyproject.toml").exists():
        _die(
            f"Could not find pyproject.toml at the expected repo root ({root}).",
            hint="Run this from a full Ed4All checkout (gui/launch.py inside the repo).",
        )
    _say(f"Repo root: {root}")

    vdir = venv_dir(root)
    py = ensure_venv(vdir)

    if args.no_install:
        if not deps_installed(vdir):
            _warn("--no-install set but no install marker found; the server may fail to import.")
        else:
            _say("Skipping install (--no-install).")
    elif args.reinstall or not deps_installed(vdir):
        if args.reinstall and marker_path(vdir).exists():
            marker_path(vdir).unlink()
        install_deps(py, root, vdir)
    else:
        _say(
            "Dependencies already installed (marker present); skipping install. "
            "Use --reinstall to force."
        )

    host = args.host
    port = pick_port(host, args.port)

    server = start_server(py, root, host, port)
    url = f"http://{host}:{port}/"

    try:
        _say(f"Waiting for the server to become healthy (up to {int(HEALTH_TIMEOUT_SECONDS)}s) ...")
        if wait_for_health(host, port, server, HEALTH_TIMEOUT_SECONDS):
            _say(f"GUI is ready: {url}")
            if args.no_browser:
                _say("Browser auto-open disabled (--no-browser). Open the URL above manually.")
            else:
                _say("Opening your default browser ...")
                try:
                    webbrowser.open(url)
                except Exception as exc:  # noqa: BLE001
                    _warn(f"Could not open a browser automatically ({exc}); open {url} manually.")
            _say("Server is running. Press Ctrl-C to stop.")
        else:
            if server.poll() is not None:
                _die(
                    f"Server exited before becoming healthy (exit code {server.returncode}).",
                    hint="Scroll up for the uvicorn traceback; try `--reinstall` if imports broke.",
                )
            _warn(
                f"Server did not answer {HEALTH_PATH} within "
                f"{int(HEALTH_TIMEOUT_SECONDS)}s; it may still come up. Try {url} in your browser."
            )

        # Foreground: block until the server exits or Ctrl-C.
        server.wait()
        return server.returncode or 0
    except KeyboardInterrupt:
        print(flush=True)
        _say("Ctrl-C received.")
        return 0
    finally:
        stop_server(server)


if __name__ == "__main__":
    raise SystemExit(main())
