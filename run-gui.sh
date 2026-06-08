#!/bin/sh
# One-click launcher for the Ed4All control-plane GUI (macOS / Linux).
#
# Double-click this file (or run `./run-gui.sh`) to set up a venv, install the
# GUI dependencies, start the server, and open it in your browser. All flags are
# forwarded to gui/launch.py (e.g. `./run-gui.sh --port 9000 --no-browser`).
set -e

# Resolve the directory this script lives in, so it works regardless of cwd.
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 was not found on your PATH." >&2
    echo "       Install Python 3.9+ (https://www.python.org/downloads/) and re-run." >&2
    exit 1
fi

exec python3 "$SCRIPT_DIR/gui/launch.py" "$@"
