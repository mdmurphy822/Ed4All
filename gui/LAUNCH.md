# One-Click GUI Launcher

Go from a fresh checkout to the Ed4All control-plane GUI running in your browser
with a single double-click — no manual venv or `pip` steps.

## One-click flow

| Platform | Do this |
|----------|---------|
| **macOS / Linux** | Double-click **`run-gui.sh`**, or run `./run-gui.sh` from the repo root. |
| **Windows** | Double-click **`run-gui.bat`**. |
| **Any (manual)** | `python3 gui/launch.py` from the repo root. |

## What it does

The launcher (`gui/launch.py`, stdlib-only — it runs before anything is
installed) performs every setup step for you:

1. **Venv** — creates `.venv-gui/` at the repo root (reuses it if present).
2. **Install** — runs `pip install -e '.[gui]' '.[server]'` into that venv so the
   FastAPI/uvicorn web stack *and* the MCP tools import cleanly. A
   `.venv-gui/.gui-installed` marker means subsequent launches skip the install.
3. **Serve** — starts `uvicorn gui.app:create_app --factory` on a free port
   (default `127.0.0.1:8077`; if busy, it picks the next free port and tells you).
4. **Open** — polls `/api/health` until the server is ready (up to ~60s), then
   opens your default browser to the GUI. The server stays in the foreground;
   press **Ctrl-C** to stop it cleanly.

## Requirements

- **Python ≥ 3.9** on your `PATH` (`python3` on macOS/Linux; `py` or `python` on
  Windows). Nothing else — the launcher builds its own venv.
- On some Debian/Ubuntu systems you may need `sudo apt install python3-venv`.

## Flags

All flags work on `run-gui.sh`, `run-gui.bat`, and `python gui/launch.py`:

| Flag | Effect |
|------|--------|
| `--host <addr>` | Bind host (default `127.0.0.1`). |
| `--port <n>` | Preferred port; the next free port is used if it's busy (default `8077`). |
| `--no-browser` | Start the server but don't auto-open a browser tab. |
| `--no-install` | Skip the `pip install` step (assume `.venv-gui/` is already set up). |
| `--reinstall` | Force a fresh dependency install even if the marker is present. |

Examples:

```sh
./run-gui.sh --port 9000          # start on a different port
./run-gui.sh --no-browser         # headless / remote box
./run-gui.sh --reinstall          # rebuild deps after a pull
```

## Already installed the `gui` extra?

If you've already run `pip install -e '.[gui]'` in your own environment, you can
skip the launcher entirely and use the CLI:

```sh
ed4all gui --host 127.0.0.1 --port 8077
```

## Troubleshooting

- **Port in use** — the launcher auto-advances to the next free port; or pass
  `--port <n>` to choose a starting point.
- **Browser didn't open** — open the printed `http://<host>:<port>/` URL
  manually, or re-run with `--no-browser` and copy the URL.
- **Imports fail / stale deps after a pull** — run with `--reinstall`.
- **`python3-venv` missing (Linux)** — install it (`sudo apt install
  python3-venv`) and re-run.
- **Local models** — local-provider routing (the `local` provider) expects a
  local OpenAI-compatible server — a vLLM seat, [Ollama](https://ollama.com),
  llama.cpp, LM Studio, etc. — running and the relevant model served; configure
  the base URL/model from the GUI's Settings → Model Routing tab.

## Secrets

API keys you enter in the GUI are stored locally in `state/gui/settings.json`,
written owner-only (`0600`) and never committed (`state/gui/` is gitignored).
They stay on your machine; nothing is sent anywhere except to the provider you
explicitly route a task to.

The full GUI feature reference lives in `gui/README.md`.
