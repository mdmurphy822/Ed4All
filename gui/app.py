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


class _NoCacheStaticFiles(StaticFiles):
    """StaticFiles that forces revalidation on every request.

    The SPA ships no build step and no fingerprinted filenames, so a plain
    static mount lets browsers cache ``*.js``/``*.css`` indefinitely — a
    redeployed container then serves new code that returning browsers never
    fetch (observed live: a UI fix invisible until a hard refresh).
    ``no-cache`` keeps ETag/304 revalidation (cheap on loopback) while
    guaranteeing a normal refresh always picks up redeployed assets.
    """

    async def get_response(self, path, scope):  # noqa: ANN001 — upstream sig
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


logger = logging.getLogger("gui.app")

#: Env var carrying the iframe-embed allowlist for the learner/Studio surfaces.
#: Its value is the *source list* of a CSP ``frame-ancestors`` directive — one or
#: more space-separated origins (e.g. ``http://lms.example:8080`` or
#: ``'self' http://lms.example:8080``). When UNSET/blank the GUI sends NO
#: ``frame-ancestors`` header at all (today's byte-identical, framed-by-anyone
#: posture the cookieless ask surface was designed for); when set, only the
#: listed origins may frame the GUI. Parse-with-fallback: garbage/blank → unset.
FRAME_ANCESTORS_ENV = "ED4ALL_GUI_FRAME_ANCESTORS"


def resolve_frame_ancestors() -> str | None:
    """Resolve the ``Content-Security-Policy: frame-ancestors ...`` value, or None.

    Reads ``ED4ALL_GUI_FRAME_ANCESTORS``. Returns the full CSP directive value
    (``"frame-ancestors <sources>"``) when the env is set to a non-blank value,
    else ``None`` (no header emitted → byte-identical to today's open-framing
    default). Header-injection safety: CR/LF are scrubbed and interior
    whitespace collapsed to single spaces, so a hostile value can't smuggle a
    second header line. A value that scrubs to empty resolves to ``None``.
    """
    raw = os.environ.get(FRAME_ANCESTORS_ENV)
    if not raw or not raw.strip():
        return None
    cleaned = " ".join(raw.replace("\r", " ").replace("\n", " ").split())
    if not cleaned:
        return None
    return "frame-ancestors " + cleaned


class FrameAncestorsMiddleware:
    """Pure-ASGI middleware that stamps a ``frame-ancestors`` CSP on HTTP responses.

    Opt-in via ``ED4ALL_GUI_FRAME_ANCESTORS`` (resolved PER-REQUEST so an operator
    edit takes effect without a restart). When the env is unset the middleware is
    a transparent pass-through — no header is added, so the response is
    byte-identical to today's open-framing behaviour. When set, the resolved
    ``Content-Security-Policy: frame-ancestors <allowlist>`` header is appended to
    every HTTP response that does NOT already carry a ``Content-Security-Policy``
    (so the archived-source viewer's own restrictive CSP is never clobbered).

    Installed in ALL serve modes (learner / studio / full) so the embeddable ask
    widget — served from any of them — honours the allowlist. Header-only, so it
    ignores non-HTTP (websocket / lifespan) scopes.
    """

    def __init__(self, app, value_provider=None) -> None:  # noqa: ANN001
        self.app = app
        # Injectable for tests; defaults to the env resolver (per-request).
        self._value_provider = value_provider or resolve_frame_ancestors

    async def __call__(self, scope, receive, send) -> None:  # noqa: ANN001
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        value = self._value_provider()
        if not value:
            await self.app(scope, receive, send)
            return
        header_value = value.encode("latin-1")

        async def send_wrapper(message):  # noqa: ANN001, ANN202
            if message.get("type") == "http.response.start":
                headers = message.setdefault("headers", [])
                has_csp = any(
                    name.lower() == b"content-security-policy" for name, _ in headers
                )
                if not has_csp:
                    headers.append((b"content-security-policy", header_value))
            await send(message)

        await self.app(scope, receive, send_wrapper)

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

# The end-user "Studio" serve mode (``ED4ALL_GUI_MODE=studio``) mounts the
# read-only Library + IMSCC course-viewer API (``/api/library`` +
# ``/api/courses/{id}/...``) PLUS the learner answer surface (the C2 ask drawer
# rides ``/api/learn/*``). The Studio page subtree (``gui/static/studio/``) is
# served at ``/`` so ``/`` IS the Studio shell.
#
# Marketable-v1 C3 adds the Create wizard + Studio settings page to the Studio
# surface, so a non-developer can upload a textbook, configure minimally, and
# launch a course build from Studio itself. That needs the upload + run + a
# scoped settings surface, so studio mode now ALSO mounts ``uploads`` (PDF
# intake), ``runs`` (launch + run registry + the ``/ws/runs/*`` progress
# stream), and ``settings`` (the credentials/answer/global subset the Studio
# settings page edits). The full env catalog is still rendered by the settings
# router, but the Studio settings UI only surfaces the Studio-relevant subset;
# learner mode remains the locked-down answer-only surface (it mounts none of
# these). The Courseforge stage / training routers stay operator-only.
_STUDIO_ROUTER_MOUNTS = [
    ("library", "/api"),  # /api/library + /api/courses/{id}/manifest|page|asset
    ("learn", "/api/learn"),  # learner answer surface (C2 ask drawer hooks here)
    ("uploads", "/api/uploads"),  # C3 Create wizard step 1 (PDF intake)
    ("runs", "/api"),  # C3 launch + run registry + /ws/runs/* progress stream
    ("settings", "/api/settings"),  # C3 Studio settings page (scoped subset)
]


def _resolve_mode(mode: str | None, learner_only: bool) -> str:
    """Resolve the canonical serve mode (``full`` | ``studio`` | ``learner``).

    Precedence (high → low):

    1. An explicit ``mode`` kwarg (``create_app(mode="studio")``).
    2. ``learner_only=True`` (the frozen back-compat kwarg) → ``"learner"``.
    3. ``ED4ALL_GUI_MODE`` env (``studio`` | ``learner`` | ``full``).
    4. ``ED4ALL_GUI_LEARNER`` truthy (the LEGACY env) → ``"learner"``.
    5. Default ``"full"`` (operator + learner surface).

    The legacy ``ED4ALL_GUI_LEARNER`` is honoured for backward compatibility but
    ``ED4ALL_GUI_MODE`` wins when both are set (a deliberate ``studio`` request
    must not be silently downgraded to learner by a stale learner env). Unknown
    ``ED4ALL_GUI_MODE`` values fall through to the learner-env check then default.
    """
    valid = {"full", "studio", "learner"}
    if mode and mode.strip().lower() in valid:
        return mode.strip().lower()
    if learner_only:
        return "learner"
    env_mode = os.environ.get("ED4ALL_GUI_MODE", "").strip().lower()
    if env_mode in valid:
        return env_mode
    if os.environ.get("ED4ALL_GUI_LEARNER", "").strip().lower() in {"1", "true", "yes", "on"}:
        return "learner"
    return "full"


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


def _studio_static_dir() -> Path:
    """Return ``gui/static/studio/`` (the end-user Studio page subtree).

    Created lazily so the mount never errors even before the page assets land.
    """
    studio = _static_dir() / "studio"
    studio.mkdir(parents=True, exist_ok=True)
    return studio


def _dev_static_dir() -> Path:
    """Return ``gui/static/dev/`` (the developer-console pane shell, C4).

    Served at ``/dev/`` in FULL mode only (operator-classified, so the D2 token
    gates it). Created lazily so the mount never errors even before the page
    assets land.
    """
    dev = _static_dir() / "dev"
    dev.mkdir(parents=True, exist_ok=True)
    return dev


def _shared_static_dir() -> Path:
    """Return ``gui/static/shared/`` (the reusable ES-module toolkit).

    Served at ``/shared/`` so every surface's ``import ... from '/shared/...'``
    resolves regardless of which subtree owns ``/``. In full mode the SPA's ``/``
    mount already serves ``gui/static/shared/`` at ``/shared/``; in studio mode
    the ``/`` root is the studio subtree, so ``/shared`` needs its own mount.
    """
    shared = _static_dir() / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    return shared


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


def create_app(learner_only: bool = False, mode: str | None = None) -> FastAPI:
    """Build and return the configured FastAPI app.

    Three serve modes (resolved by ``_resolve_mode``):

    * ``full`` (default) — operator routers + SPA + the learner/Studio surfaces
      riding the operator app (``/learn/`` and ``/studio/`` served by the SPA's
      ``html=True`` static mount).
    * ``studio`` — the end-user Studio surface ONLY: the Library + IMSCC
      course-viewer API (``/api/library`` + ``/api/courses/{id}/...``) plus the
      learner answer API (``/api/learn/*`` — the C2 ask drawer), with the
      ``gui/static/studio/`` page subtree served at ``/``. The operator routers
      (env / API keys / run launching) are NOT mounted.
    * ``learner`` — the learner answer surface ONLY (``/api/learn/*`` +
      ``gui/static/learn/`` at ``/``); the original moderated-pilot posture.

    ``learner_only=True`` is the frozen back-compat kwarg → ``mode="learner"``.
    ``mode=`` lets a caller request any mode directly. uvicorn's import-string
    factory paths (``ed4all gui`` / ``--reload``) call ``create_app()`` with no
    arguments, so the env vars (``ED4ALL_GUI_MODE`` canonical, ``ED4ALL_GUI_LEARNER``
    legacy) are the only channel those entry points use to request a non-full
    surface — without the env resolution here they would silently serve the full
    operator surface. ``ED4ALL_GUI_MODE`` wins over the legacy ``ED4ALL_GUI_LEARNER``.
    """
    resolved_mode = _resolve_mode(mode, learner_only)
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

    # Opt-in iframe-embed allowlist (ED4ALL_GUI_FRAME_ANCESTORS). Installed in
    # EVERY serve mode so the embeddable ask widget honours the allowlist
    # regardless of which shell serves it. Transparent pass-through (no header,
    # byte-identical) when the env is unset — the ask surface stays framed-by-
    # anyone by default, matching the cookieless embed design.
    app.add_middleware(FrameAncestorsMiddleware)

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

    if resolved_mode == "learner":
        _include_routers(app, _LEARNER_ROUTER_MOUNTS)
        # The learner page references the shared design-token + component-kit
        # assets via ``/shared/...`` (tokens.css, components.css, and the
        # presentation-only ring.js ES module the thinking ring imports). Mount
        # that subtree FIRST so those resolve in pure learner mode (it carries
        # NO router/endpoint — the kit is presentation-only by contract, so the
        # locked-down learner appliance still cannot reach a run-control route).
        app.mount(
            "/shared",
            _NoCacheStaticFiles(directory=str(_shared_static_dir())),
            name="shared-static",
        )
        # Serve the learner page subtree at ``/`` — ``/`` IS the learner page in
        # learner mode (no operator SPA mounted).
        app.mount(
            "/",
            _NoCacheStaticFiles(directory=str(_learn_static_dir()), html=True),
            name="learn-static",
        )
        return app

    if resolved_mode == "studio":
        _include_routers(app, _STUDIO_ROUTER_MOUNTS)

        # The Studio shell's index.html references its assets via ``/studio/...``
        # and the shared toolkit via ``/shared/...``; serve those subtrees at
        # their own prefixes, and redirect the bare ``/`` to the Studio shell.
        # No operator SPA is mounted. The learn page subtree (C2 ask drawer)
        # rides ``/learn/``.
        from fastapi.responses import RedirectResponse  # noqa: PLC0415

        @app.get("/", include_in_schema=False)
        async def _studio_root() -> RedirectResponse:  # noqa: D401
            return RedirectResponse(url="/studio/")

        app.mount(
            "/learn",
            _NoCacheStaticFiles(directory=str(_learn_static_dir()), html=True),
            name="learn-static",
        )
        app.mount(
            "/shared",
            _NoCacheStaticFiles(directory=str(_shared_static_dir())),
            name="shared-static",
        )
        app.mount(
            "/studio",
            _NoCacheStaticFiles(directory=str(_studio_static_dir()), html=True),
            name="studio-static",
        )
        return app

    # Full mode is the only surface that mounts the operator routers + SPA, so
    # it's the only mode that installs the operator token gate (Marketable-v1 D2).
    # When no token is configured the middleware is a transparent pass-through
    # (current open LAN/loopback default); when ED4ALL_GUI_TOKEN (or the settings
    # store's secrets.gui_token) is set, operator-classified paths require a
    # bearer token. Studio/learner modes don't mount the operator surface, so
    # they need no gate. Added last among middleware so it wraps the whole app.
    from gui.auth import OperatorTokenMiddleware  # noqa: PLC0415

    app.add_middleware(OperatorTokenMiddleware)

    _include_routers(app, _ROUTER_MOUNTS)
    # Studio Library + viewer API rides the full operator app too, so an operator
    # can preview the end-user Studio at ``/studio/`` without a separate process.
    _include_routers(app, [("library", "/api")])

    # GUI-redesign Phase 2 (the auth remount). In FULL mode the OPEN product is
    # now the Studio shell, served at the bare ``/``; the operator six-tab SPA +
    # the developer console move BEHIND the token gate at ``/advanced/``. The
    # operator SPA lives at ``gui/static/`` (``_static_dir()``), whose ``dev/``
    # subdir is the C4 console — so a single ``/advanced`` mount over the whole
    # operator-static tree serves BOTH the SPA (``/advanced/`` index +
    # ``/advanced/app.js`` + ``/advanced/styles.css``) AND the dev console
    # (``/advanced/dev/``); no separate ``/dev`` mount is needed. The whole
    # ``/advanced`` prefix is operator-classified in ``gui.auth`` (the D2 token
    # gates it), except ``/advanced/styles.css`` which stays open (mirrors the
    # historical styles.css-stays-open posture).
    app.mount(
        "/advanced",
        _NoCacheStaticFiles(directory=str(_static_dir()), html=True),
        name="advanced-static",
    )

    # The Studio / learner / shared static subtrees previously rode the catch-all
    # ``/`` operator-static mount (which served ``gui/static/`` whole, including
    # its ``studio/ learn/ shared/`` subdirs). With ``/`` now the Studio shell
    # those subtrees need their own explicit mounts. The shared ES toolkit
    # (``/shared/...``) and the learner page (``/learn/``) ride the open surface.
    app.mount(
        "/studio",
        _NoCacheStaticFiles(directory=str(_studio_static_dir()), html=True),
        name="studio-static",
    )
    app.mount(
        "/learn",
        _NoCacheStaticFiles(directory=str(_learn_static_dir()), html=True),
        name="learn-static",
    )
    app.mount(
        "/shared",
        _NoCacheStaticFiles(directory=str(_shared_static_dir())),
        name="shared-static",
    )

    # Studio shell at the bare ``/`` LAST so it doesn't shadow the ``/api/...``
    # routers or the ``/advanced /studio /learn /shared`` mounts above. ``/`` IS
    # the Studio shell now (THE PRODUCT) — open, no operator token required.
    # ``html=True`` serves ``gui/static/studio/index.html`` at ``/``.
    app.mount(
        "/",
        _NoCacheStaticFiles(directory=str(_studio_static_dir()), html=True),
        name="studio-root",
    )

    return app


__all__ = [
    "create_app",
    "FRAME_ANCESTORS_ENV",
    "FrameAncestorsMiddleware",
    "resolve_frame_ancestors",
]
