"""Shared-secret bearer-token auth for the operator surface (Marketable-v1 D2).

Minimum-viable auth: a single shared secret (``ED4ALL_GUI_TOKEN``, or the
settings store's ``secrets.gui_token`` with env precedence) gates the
OPERATOR-classified routes so the full operator surface can be exposed beyond
loopback without leaving env/API-key management and run-launching wide open.
Studio / learner surfaces stay open for LAN appliance deploys — in those serve
modes the operator routers are not mounted at all, and the middleware is only
installed in ``full`` mode.

Boundary (operator-classified, enforced only in ``full`` mode when a token is
configured):

* the six-tab operator SPA root (``/`` index + ``/app.js``),
* the developer-console subtree (``/dev/`` — the Marketable-v1 C4 pane shell +
  its ``dev.js`` / ``dev.css`` assets), a power-user surface mounted in full
  mode only,
* the OpenAPI docs surface (``/docs`` / ``/redoc`` / ``/openapi.json``),
* the operator-only API routers Studio never calls: ``/api/courses``,
  ``/api/retrieval``, ``/api/activity``,
* the operator SPA's run-log WebSocket ``/api/ws/runs/*`` (token via the
  ``?token=`` query param — browser JS can't set WS request headers).

Deliberately NOT gated (shared with Studio flows / public by contract):

* ``/api/health`` — Docker healthcheck must stay unauthenticated,
* the Studio-shared API routers ``/api/settings``, ``/api/uploads``,
  ``/api/runs`` (REST), ``/api/learn``, ``/api/library`` — the Studio Create
  wizard + settings page legitimately use these,
* static asset subtrees ``/shared/``, ``/studio/``, ``/learn/``,
  ``/styles.css`` — the open studio/learner shells load these.

When no token is configured the middleware is a pass-through (the current open
LAN/loopback default); a non-loopback bind without a token logs a startup
WARNING (see ``gui.server``).
"""

from __future__ import annotations

import hmac
import logging
import os
from typing import Optional

logger = logging.getLogger("gui.auth")

#: Env var carrying the shared operator token (Docker-friendly primary channel).
TOKEN_ENV = "ED4ALL_GUI_TOKEN"

#: WS query-param carrying the token (browser JS can't set WS request headers).
WS_TOKEN_PARAM = "token"

# Path prefixes that REQUIRE the token in full mode. Matched against the request
# path. ``/`` and ``/app.js`` are the operator SPA root; the rest are the
# operator-only API families + the operator run-log WebSocket.
_OPERATOR_API_PREFIXES = (
    "/api/courses",
    "/api/retrieval",
    "/api/activity",
)
_OPERATOR_WS_PREFIX = "/api/ws/runs"
_OPERATOR_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")
# Method-gated destructive routes on otherwise-open (Studio-shared) families.
# ``DELETE /api/library/{slug}`` permanently removes a course (D5): destructive
# ops get the stricter operator treatment even though GET ``/api/library`` stays
# open for the Studio shell. Matched on (method, path-prefix).
_OPERATOR_DESTRUCTIVE_ROUTES = (
    ("DELETE", "/api/library"),
)
# Operator SPA shell files served at the root mount (exact paths). The shared /
# studio / learn subtrees + styles.css stay open for the studio/learner shells.
_OPERATOR_ROOT_PATHS = ("/", "/index.html", "/app.js")
# The developer-console subtree (Marketable-v1 C4). The whole prefix is
# operator-classified: the pane shell + its dev.js / dev.css assets are a
# power-user surface a learner / studio appliance never reaches.
_OPERATOR_DEV_PREFIX = "/dev"


def resolve_token(settings_token: Optional[str] = None) -> Optional[str]:
    """Return the configured operator token, or ``None`` when unset.

    Resolution chain (high → low): ``ED4ALL_GUI_TOKEN`` env > the passed
    ``settings_token`` (the settings store's ``secrets.gui_token``). Env wins so
    a Docker ``-e ED4ALL_GUI_TOKEN=...`` override always takes precedence over a
    persisted settings secret. Blank / whitespace-only values resolve to
    ``None`` (treated as unset → open behaviour).
    """
    env_val = os.environ.get(TOKEN_ENV)
    if env_val and env_val.strip():
        return env_val.strip()
    if settings_token and str(settings_token).strip():
        return str(settings_token).strip()
    return None


def _settings_token() -> Optional[str]:
    """Best-effort read of ``secrets.gui_token`` from the settings store.

    Tolerant of an absent / malformed store (returns ``None``); the env channel
    is the primary, so a settings-read failure must never break auth resolution.
    """
    try:
        from gui import settings_store  # noqa: PLC0415

        doc = settings_store.load_settings()
        secrets = doc.get("secrets") if isinstance(doc, dict) else None
        if isinstance(secrets, dict):
            tok = secrets.get("gui_token")
            if tok and str(tok).strip():
                return str(tok).strip()
    except Exception:  # noqa: BLE001 — settings store is best-effort here
        return None
    return None


def is_operator_path(path: str, method: Optional[str] = None) -> bool:
    """True if ``(method, path)`` is operator-classified (token-required, full mode).

    The operator SPA root + ``/app.js``, the OpenAPI docs surface, the
    operator-only API routers (courses / retrieval / activity), and the operator
    run-log WebSocket. Health, the Studio-shared API routers, and the
    static studio/learn/shared subtrees are NOT operator-classified.

    ``method`` (when supplied) additionally gates destructive verbs on otherwise
    open route families: ``DELETE /api/library/<slug>`` is operator-classified
    even though ``GET /api/library`` stays open for the Studio shell.
    """
    if path in _OPERATOR_ROOT_PATHS:
        return True
    if path in _OPERATOR_DOCS_PATHS:
        return True
    if path.startswith(_OPERATOR_WS_PREFIX):
        return True
    # The /dev/ developer console (C4) — gate the whole subtree (shell + assets).
    if path == _OPERATOR_DEV_PREFIX or path.startswith(_OPERATOR_DEV_PREFIX + "/"):
        return True
    if any(path == p or path.startswith(p + "/") for p in _OPERATOR_API_PREFIXES):
        return True
    # Method-gated destructive routes (e.g. DELETE on the otherwise-open Library).
    if method is not None:
        verb = method.upper()
        for dverb, prefix in _OPERATOR_DESTRUCTIVE_ROUTES:
            if verb == dverb and (path == prefix or path.startswith(prefix + "/")):
                return True
    return False


def tokens_match(presented: Optional[str], expected: str) -> bool:
    """Constant-time compare of a presented token against the expected one.

    Uses ``hmac.compare_digest`` so a timing side-channel can't leak the token
    one character at a time. A missing / empty presented token is a non-match.
    """
    if not presented:
        return False
    return hmac.compare_digest(presented, expected)


def _bearer_token(header_value: Optional[str]) -> Optional[str]:
    """Extract the bearer token from an ``Authorization`` header value."""
    if not header_value:
        return None
    parts = header_value.split(None, 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def _unauthorized(send) -> None:
    """Emit a 401 ASGI response with a ``WWW-Authenticate: Bearer`` challenge."""
    body = b'{"error":"unauthorized","detail":"operator token required"}'
    await send(
        {
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"www-authenticate", b'Bearer realm="ed4all-operator"'),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


class OperatorTokenMiddleware:
    """Pure-ASGI middleware gating operator-classified paths with a bearer token.

    Installed only in ``full`` serve mode (``gui.app.create_app``). When no token
    is configured it is a transparent pass-through (current open behaviour).
    When a token IS configured:

    * HTTP operator paths require ``Authorization: Bearer <token>`` (401 +
      ``WWW-Authenticate`` otherwise),
    * the operator WebSocket (``/api/ws/runs/*``) requires ``?token=<token>`` in
      the query string (4401 close otherwise — browser JS can't set WS headers),
    * non-operator paths (health, Studio-shared APIs, static studio/learn/shared
      assets) pass through untouched.

    Pure ASGI (not ``BaseHTTPMiddleware``) so it can gate WebSocket scopes too;
    ``BaseHTTPMiddleware`` only sees HTTP. The token is resolved per-request so a
    settings-store write takes effect without a restart (env still wins).
    """

    def __init__(self, app, token_provider=None) -> None:
        self.app = app
        # Injectable for tests; defaults to env > settings-store resolution.
        self._token_provider = token_provider or (lambda: resolve_token(_settings_token()))

    async def __call__(self, scope, receive, send) -> None:
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        # HTTP carries a method (used for method-gated destructive routes);
        # WebSocket scopes have none, so pass None (path-only classification).
        method = scope.get("method") if scope_type == "http" else None
        if not is_operator_path(path, method):
            await self.app(scope, receive, send)
            return

        token = self._token_provider()
        if not token:
            # No token configured → open behaviour (LAN/loopback default).
            await self.app(scope, receive, send)
            return

        if scope_type == "websocket":
            presented = self._ws_token(scope)
            if not tokens_match(presented, token):
                # Reject before accept: an ASGI WS close with a 4401 app code.
                await send({"type": "websocket.close", "code": 4401})
                return
            await self.app(scope, receive, send)
            return

        # HTTP operator path.
        presented = self._http_token(scope)
        if not tokens_match(presented, token):
            await _unauthorized(send)
            return
        await self.app(scope, receive, send)

    @staticmethod
    def _http_token(scope) -> Optional[str]:
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                return _bearer_token(value.decode("latin-1"))
        return None

    @staticmethod
    def _ws_token(scope) -> Optional[str]:
        from urllib.parse import parse_qs  # noqa: PLC0415

        qs = scope.get("query_string", b"").decode("latin-1")
        if not qs:
            return None
        values = parse_qs(qs).get(WS_TOKEN_PARAM)
        if values:
            return values[0]
        return None


__all__ = [
    "TOKEN_ENV",
    "WS_TOKEN_PARAM",
    "OperatorTokenMiddleware",
    "resolve_token",
    "is_operator_path",
    "tokens_match",
]
