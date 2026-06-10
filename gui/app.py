"""FastAPI application factory for the Ed4All control-plane GUI.

``create_app()`` mounts the five API routers under ``/api/...``, serves the
vanilla-JS SPA from ``gui/static/`` via ``StaticFiles`` at ``/`` (``html=True``),
and exposes a ``GET /api/health`` liveness probe.

Web deps (fastapi/uvicorn) are imported HERE — never in the foundation-core
modules (``shared_state`` / ``env_catalog`` / ``settings_store`` / ``models``),
so MCP tools can import the store without web deps.

Routers are imported under a per-router try/except so the app comes up cleanly
even while a sibling router module is mid-build. Each router module exposes a
module-level ``router = APIRouter()``; the intended final state wires all five.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("gui.app")

# (module attribute name, mount prefix) for each router. The attribute is the
# ``router`` object exported by ``gui.routers.<name>``.
_ROUTER_MOUNTS = [
    ("settings", "/api/settings"),
    ("uploads", "/api/uploads"),
    ("runs", "/api"),  # runs.py owns /api/workflows + /api/runs/* + /ws/runs/*
    ("courses", "/api/courses"),
    ("retrieval", "/api/retrieval"),
    ("learn", "/api/learn"),  # learner answer surface (ask, courses, source)
]

# The learner-only serve mode (``ed4all gui --learner`` / ``ED4ALL_GUI_LEARNER=1``)
# mounts ONLY this router — the operator routers (settings/uploads/runs/courses/
# retrieval, which expose env / API keys / run-launching) are NOT mounted, so a
# learner on a shared appliance never reaches the operator surface.
_LEARNER_ROUTER_MOUNTS = [
    ("learn", "/api/learn"),
]


def _static_dir() -> Path:
    """Return ``gui/static/`` (created lazily so the mount never errors)."""
    static = Path(__file__).resolve().parent / "static"
    static.mkdir(parents=True, exist_ok=True)
    return static


def _learn_static_dir() -> Path:
    """Return ``gui/static/learn/`` (the learner page subtree).

    Built by a sibling executor; created lazily here so the mount never errors
    even before the page assets land (the dir is tolerated absent — an empty
    dir simply serves nothing until ``index.html`` ships).
    """
    learn = _static_dir() / "learn"
    learn.mkdir(parents=True, exist_ok=True)
    return learn


def _include_routers(app: FastAPI, mounts: list) -> None:
    """Import + mount each router in ``mounts``; tolerate a router mid-build.

    Routers carry their own ``@router.<method>`` paths; ``prefix`` here only
    namespaces them. ``runs`` mounts at the bare ``/api`` prefix because it owns
    several distinct path families (``/api/workflows``, ``/api/runs/*``, and the
    ``/ws/runs/*`` WebSocket).
    """
    import importlib  # noqa: PLC0415

    for name, prefix in mounts:
        module_path = f"gui.routers.{name}"
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            logger.warning(
                "gui router %s not available yet (%s); skipping mount", module_path, exc
            )
            continue
        router = getattr(module, "router", None)
        if router is None:
            logger.warning("gui router %s exposes no module-level 'router'", module_path)
            continue
        app.include_router(router, prefix=prefix)
        logger.info("mounted gui router %s at %s", module_path, prefix)


def create_app(learner_only: bool = False) -> FastAPI:
    """Build and return the configured FastAPI app.

    ``learner_only=True`` mounts ONLY the learner answer surface — the learn
    router (``/api/learn/*``) plus the ``gui/static/learn/`` page subtree served
    at ``/`` (so ``/`` IS the learner page in learner mode) — and NONE of the
    operator routers or the operator SPA. The liveness probe (``/api/health``)
    stays. This is the access-control posture for moderated pilot sessions on a
    shared appliance (the operator surface that exposes env / API keys / run
    launching is not even mounted).

    ``ED4ALL_GUI_LEARNER`` (truthy: 1/true/yes/on) also forces learner-only
    mode. uvicorn's import-string factory paths (``ed4all gui`` and
    ``--reload``) call ``create_app()`` with no arguments, so the env var is
    the only channel through which those entry points can request the learner
    surface — without this check they would silently serve the full operator
    surface to a learner session.
    """
    if not learner_only:
        learner_only = os.environ.get(
            "ED4ALL_GUI_LEARNER", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
    app = FastAPI(
        title="Ed4All Control-Plane GUI",
        description="Human management surface for the Ed4All pipeline.",
        version="1.0.0",
    )

    # Permissive CORS for the local single-origin SPA + MCP-driven tooling.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health() -> dict:  # noqa: D401 — simple liveness probe
        """Liveness probe."""
        return {"status": "ok"}

    @app.on_event("startup")
    async def _reconcile_orphans_on_boot() -> None:  # noqa: D401
        """Reconcile orphaned runs left by a previous process on boot.

        Background driver tasks live only in-process, so a uvicorn restart leaves
        any ``queued``/``running`` run stuck forever (no task drives it, WS clients
        poll indefinitely). Flip them to a terminal ``interrupted`` marker once at
        startup. Best-effort: a failure here is logged but never blocks boot.
        """
        try:
            from gui.services.run_service import reconcile_orphans  # noqa: PLC0415

            reconciled = reconcile_orphans()
            if reconciled:
                logger.info("startup reconciled %d orphaned run(s)", len(reconciled))
        except Exception:  # noqa: BLE001 — startup reconciliation is best-effort
            logger.exception("orphan reconciliation failed on startup")

    if learner_only:
        _include_routers(app, _LEARNER_ROUTER_MOUNTS)
        # Serve the learner page subtree at ``/`` — ``/`` IS the learner page in
        # learner mode (no operator SPA mounted).
        app.mount(
            "/",
            StaticFiles(directory=str(_learn_static_dir()), html=True),
            name="learn-static",
        )
        return app

    _include_routers(app, _ROUTER_MOUNTS)

    # SPA mount LAST so it doesn't shadow the /api routes. html=True serves
    # index.html for the SPA root. The existing ``html=True`` mount at ``/``
    # auto-serves any ``gui/static/learn/index.html`` at ``/learn/`` with no
    # extra mount (the learner page rides the operator app in the full mode).
    app.mount("/", StaticFiles(directory=str(_static_dir()), html=True), name="static")

    return app


__all__ = ["create_app"]
