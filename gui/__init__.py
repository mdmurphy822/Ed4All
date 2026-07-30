"""Ed4All control-plane GUI package.

A real, no-stubs web GUI for full human management of the Ed4All pipeline:
file upload, env/API-key management, per-task model routing, course
topics/subjects, the retrieval system, per-component runs, and a Claude<->GUI
control surface.

The foundation-core modules (``shared_state``, ``env_catalog``,
``settings_store``, ``models``) import WITHOUT FastAPI/uvicorn installed, so
MCP tools can read/write the same ``runtime/state/gui/`` store without pulling web
deps. The web surface (``app``, ``server``, ``routers/*``) lives behind the
opt-in ``gui`` extra.

``DEFAULT_HOST`` / ``DEFAULT_PORT`` are the single source of truth for the GUI
bind address, shared by ``gui.server`` and the ``ed4all gui`` command. They live
here (not in ``gui.server``) so importing them never drags in FastAPI/uvicorn.
``gui.launch`` (stdlib-only, runs pre-install) keeps its own literal copies and
cannot import this module when run as a script, so keep the values in sync.
"""

# Single source of truth for the GUI bind defaults. Kept lightweight (no heavy
# imports) so any consumer can read these without the web stack installed.
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8077

__all__ = [
    "shared_state",
    "env_catalog",
    "settings_store",
    "models",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
]
