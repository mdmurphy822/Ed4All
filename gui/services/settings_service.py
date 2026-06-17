"""Thin orchestration over ``settings_store`` + ``env_catalog`` (no web deps).

This module holds the SETTINGS-vertical business logic so the FastAPI router
stays thin. It imports cleanly WITHOUT FastAPI installed — only ``gui`` foundation
modules + stdlib at import time. The heavier ``urllib`` / ``anthropic`` imports
are deferred into the functions that need them so module import stays light.

Two public surfaces:

* ``build_settings_payload()`` — the masked settings doc enriched with the
  catalog / provider registry / base-model list for the GET endpoint.
* ``test_provider(provider)`` — a REAL reachability probe per provider family
  (never a hardcoded ``ok``). See the function docstring for the per-provider
  contract.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from gui import env_catalog, settings_store

# Default base URL for the local OpenAI-compatible server (Ollama et al.) when
# neither os.environ nor the settings env block names one.
_DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"

# Short network timeout (seconds) for live reachability probes so a stuck
# server can't hang the request.
_PROBE_TIMEOUT_S = 4.0


# --------------------------------------------------------------------- payload


def build_settings_payload() -> Dict[str, Any]:
    """Return the masked settings doc enriched for the GET endpoint.

    Shape::

        { ...mask_secrets(load_settings()),
          "catalog": CATALOG,
          "providers": PROVIDERS,
          "base_models": BASE_MODELS }

    Secrets are masked (``"set"`` / ``None``) — raw key values never leave here.
    """
    doc = settings_store.mask_secrets(settings_store.load_settings())
    doc["catalog"] = env_catalog.CATALOG
    doc["providers"] = env_catalog.PROVIDERS
    doc["base_models"] = env_catalog.BASE_MODELS
    return doc


# The env-catalog categories a non-developer Studio user needs to see + edit:
# their cloud-provider keys, the global LLM mode/provider/model, the
# grounded-answer (ask) backend, and the local server wiring (so an air-gapped
# Ollama lattice can be pointed at). The full operator catalog (DART / per-tier
# Courseforge / Trainforge / embedding knobs) is intentionally hidden from
# Studio. ``model_routing`` tasks the page edits map to the same canonical env
# vars via ``ROUTING_ENV_MAP`` — the page writes routing, never raw env, so a
# secret never round-trips through it.
_STUDIO_CATEGORIES = ("credentials", "global", "answer", "local")

# The model_routing tasks the Studio settings page surfaces (authoring +
# answer). Mirrors ``env_catalog.ROUTING_ENV_MAP`` task names; the page reads /
# patches ``model_routing.<task>``. Both Courseforge two-pass authoring tiers
# (outline + rewrite) are surfaced so a Studio user can route the full
# generation pipeline; they sit adjacent here for parity with the operator tab.
_STUDIO_ROUTING_TASKS = (
    "global",
    "courseplanner",
    "courseforge_outline",
    "courseforge_rewrite",
    "answer",
)


def build_studio_settings_payload() -> Dict[str, Any]:
    """Return the masked settings doc scoped to the Studio user subset.

    Shape (a strict subset of ``build_settings_payload`` so the Studio settings
    page renders the SAME shapes the operator settings tab does, just filtered)::

        { "version", "updated_at",
          "env":          { <only Studio-category keys>: "set"|<value>|None },
          "model_routing": { <only Studio tasks>: {...} },
          "catalog":      [ <only Studio-category entries> ],
          "providers":    PROVIDERS,           # full list (provider picker)
          "host":         "<gui host>",        # read-only display
          "port":         <gui port> }         # read-only display

    Secrets stay masked (``mask_secrets`` runs first). The host/port are a
    read-only echo for the page (the Studio user can't change the bind address
    from the browser). The full operator catalog is NOT returned here.
    """
    masked = settings_store.mask_secrets(settings_store.load_settings())
    catalog = [e for e in env_catalog.CATALOG if e.get("category") in _STUDIO_CATEGORIES]
    studio_keys = {e["key"] for e in catalog}

    env = masked.get("env") if isinstance(masked, dict) else None
    env = {k: v for k, v in env.items() if k in studio_keys} if isinstance(env, dict) else {}

    routing = masked.get("model_routing") if isinstance(masked, dict) else None
    routing = (
        {k: v for k, v in routing.items() if k in _STUDIO_ROUTING_TASKS}
        if isinstance(routing, dict)
        else {}
    )

    return {
        "version": masked.get("version") if isinstance(masked, dict) else None,
        "updated_at": masked.get("updated_at") if isinstance(masked, dict) else None,
        "env": env,
        "model_routing": routing,
        "catalog": catalog,
        "providers": env_catalog.PROVIDERS,
        "host": os.environ.get("ED4ALL_GUI_HOST", "127.0.0.1"),
        "port": _gui_port(),
    }


def _gui_port() -> int:
    """Return the configured GUI port for read-only display (default 8077)."""
    raw = os.environ.get("ED4ALL_GUI_PORT", "8077")
    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return 8077


# ----------------------------------------------------------------- env lookup


def _settings_env() -> Dict[str, Any]:
    """Return the raw (unmasked) ``env`` block of the persisted settings doc."""
    doc = settings_store.load_settings()
    env = doc.get("env") if isinstance(doc, dict) else None
    return env if isinstance(env, dict) else {}


def _resolve_env_value(key: str, settings_env: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """Resolve an env value, preferring ``os.environ`` then the settings doc.

    ``os.environ`` wins because ``apply_env`` writes resolved settings there, so
    a freshly-applied value is reflected immediately. Empty strings are treated
    as unset. Returns the stripped value or ``None``.
    """
    raw = os.environ.get(key)
    if raw is None or not str(raw).strip():
        env = settings_env if settings_env is not None else _settings_env()
        raw = env.get(key)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _has_key(key: str, settings_env: Optional[Dict[str, Any]] = None) -> bool:
    """True when a non-empty value for ``key`` exists in env or settings."""
    return _resolve_env_value(key, settings_env) is not None


# ------------------------------------------------------------- provider probes


def test_provider(provider_name: str) -> Dict[str, Any]:
    """Run a REAL reachability check for ``provider_name``.

    Returns ``{"ok": bool, "status": str, "detail": str}`` plus a ``provider``
    echo. Per-provider contract:

    * ``anthropic`` — if ``ANTHROPIC_API_KEY`` is present (os.environ or settings
      env) AND the ``anthropic`` package is importable AND LLM mode allows api
      calls, attempt a minimal 1-token ping. Otherwise report key-presence only.
      Any exception is caught → ``ok=False`` with the error detail.
    * ``local`` — HTTP GET ``<base_url>/models`` (base_url from
      ``LOCAL_SYNTHESIS_BASE_URL`` or the default) with a short timeout; ``ok``
      on HTTP 200.
    * ``together`` / ``groq`` / ``fireworks`` / ``deepseek`` — report ``ok`` based
      on the provider's ``api_key_env`` presence (these need a key; no live
      billable call is made).
    * ``mock`` — always reachable (no credential, no network).

    Unknown provider names → ``ok=False`` / ``status="unknown_provider"``.
    """
    name = (provider_name or "").strip()
    entry = env_catalog.provider(name)
    if entry is None:
        return _result(
            name,
            ok=False,
            status="unknown_provider",
            detail=(
                f"{name!r} is not a registered provider "
                f"(valid: {env_catalog.provider_names()})"
            ),
        )

    settings_env = _settings_env()

    if name == "anthropic":
        return _test_anthropic(name, entry, settings_env)
    if name == "local":
        return _test_local(name, entry, settings_env)
    if name == "mock":
        return _result(name, ok=True, status="ready", detail="Mock backend; no network required.")
    # Hosted OpenAI-compatible providers — key-presence only (no billable call).
    return _test_key_presence(name, entry, settings_env)


def _test_anthropic(
    name: str, entry: Dict[str, Any], settings_env: Dict[str, Any]
) -> Dict[str, Any]:
    """Anthropic probe: key-presence, then a 1-token ping when wiring allows."""
    key = _resolve_env_value("ANTHROPIC_API_KEY", settings_env)
    if not key:
        return _result(
            name,
            ok=False,
            status="missing_key",
            detail="ANTHROPIC_API_KEY is not set (os.environ or settings env).",
        )

    # LLM mode: only attempt a live ping when api mode is selected; in local mode
    # the session is the LLM and an SDK ping is not the configured path.
    mode = _resolve_env_value("LLM_MODE", settings_env)
    if mode is not None and mode != "api":
        return _result(
            name,
            ok=True,
            status="key_present",
            detail=(
                f"ANTHROPIC_API_KEY is set; LLM_MODE={mode!r} (not 'api'), "
                "so no live ping was attempted."
            ),
        )

    try:
        import anthropic  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001 — package may be absent on a light install
        return _result(
            name,
            ok=True,
            status="key_present",
            detail=(
                "ANTHROPIC_API_KEY is set; the 'anthropic' package is not "
                f"importable ({exc}), so no live ping was attempted."
            ),
        )

    try:
        client = anthropic.Anthropic(api_key=key)
        model = (
            _resolve_env_value("LLM_MODEL", settings_env)
            or entry.get("model_default")
            or "claude-opus-4-7"
        )
        client.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
    except Exception as exc:  # noqa: BLE001 — any SDK/network/auth error → ok=False
        return _result(
            name,
            ok=False,
            status="ping_failed",
            detail=f"Anthropic 1-token ping failed: {type(exc).__name__}: {exc}",
        )
    return _result(
        name,
        ok=True,
        status="reachable",
        detail=f"Anthropic 1-token ping succeeded (model={model}).",
    )


def _test_local(
    name: str, entry: Dict[str, Any], settings_env: Dict[str, Any]
) -> Dict[str, Any]:
    """Local probe: HTTP GET ``<base_url>/models`` with a short timeout."""
    base_url = (
        _resolve_env_value("LOCAL_SYNTHESIS_BASE_URL", settings_env)
        or entry.get("base_url_default")
        or _DEFAULT_LOCAL_BASE_URL
    )
    url = base_url.rstrip("/") + "/models"

    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(url, method="GET")
    key = _resolve_env_value("LOCAL_SYNTHESIS_API_KEY", settings_env)
    if key:
        request.add_header("Authorization", f"Bearer {key}")

    try:
        with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_S) as response:  # noqa: S310
            code = getattr(response, "status", None) or response.getcode()
    except urllib.error.HTTPError as exc:
        return _result(
            name,
            ok=False,
            status="http_error",
            detail=f"GET {url} returned HTTP {exc.code}.",
        )
    except Exception as exc:  # noqa: BLE001 — connection refused / timeout / DNS
        return _result(
            name,
            ok=False,
            status="unreachable",
            detail=f"GET {url} failed: {type(exc).__name__}: {exc}",
        )
    if code == 200:
        return _result(
            name,
            ok=True,
            status="reachable",
            detail=f"GET {url} returned HTTP 200.",
        )
    return _result(
        name,
        ok=False,
        status="http_error",
        detail=f"GET {url} returned HTTP {code} (expected 200).",
    )


def _test_key_presence(
    name: str, entry: Dict[str, Any], settings_env: Dict[str, Any]
) -> Dict[str, Any]:
    """Hosted-provider probe: report on the provider's api_key_env presence."""
    api_key_env = entry.get("api_key_env")
    if not api_key_env:
        # No credential declared — treat as ready (e.g. an unauthenticated proxy).
        return _result(
            name,
            ok=True,
            status="ready",
            detail=f"Provider {name!r} declares no API key env; assumed reachable.",
        )
    if _has_key(api_key_env, settings_env):
        return _result(
            name,
            ok=True,
            status="key_present",
            detail=(
                f"{api_key_env} is set; no live billable call is made for "
                f"{name!r} (key-presence check only)."
            ),
        )
    return _result(
        name,
        ok=False,
        status="missing_key",
        detail=f"{api_key_env} is not set (os.environ or settings env).",
    )


# ----------------------------------------------------------------- ollama models


def list_ollama_models() -> Dict[str, Any]:
    """Discover models installed on the local Ollama server (REAL, live probe).

    The Ollama host is derived from ``LOCAL_SYNTHESIS_BASE_URL`` (the ``local``
    provider's base URL, e.g. ``http://localhost:11434/v1``) by stripping a
    trailing ``/v1`` — Ollama's native tags API lives at ``/api/tags`` on the
    bare host, NOT under the OpenAI-compat ``/v1`` prefix. Issues a short-timeout
    ``GET <host>/api/tags`` and returns::

        {"available": bool, "models": [<name>, ...], "detail": str, "host": str}

    Graceful on any connection failure (refused / timeout / DNS / HTTP error /
    bad JSON): ``available=False``, ``models=[]`` and a human-readable
    ``detail``. This powers the live model dropdowns in the GUI — a real
    discovery, never a hardcoded list.
    """
    settings_env = _settings_env()
    base_url = (
        _resolve_env_value("LOCAL_SYNTHESIS_BASE_URL", settings_env)
        or _DEFAULT_LOCAL_BASE_URL
    )
    # Strip a trailing ``/v1`` (Ollama OpenAI-compat prefix) to reach the host
    # root that serves ``/api/tags``.
    host = base_url.rstrip("/")
    if host.endswith("/v1"):
        host = host[: -len("/v1")]
    url = host + "/api/tags"

    import json as _json  # noqa: PLC0415
    import urllib.error  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(url, method="GET")
    key = _resolve_env_value("LOCAL_SYNTHESIS_API_KEY", settings_env)
    if key:
        request.add_header("Authorization", f"Bearer {key}")

    try:
        with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_S) as response:  # noqa: S310
            raw = response.read()
    except urllib.error.HTTPError as exc:
        return {
            "available": False,
            "models": [],
            "detail": f"GET {url} returned HTTP {exc.code}.",
            "host": host,
        }
    except Exception as exc:  # noqa: BLE001 — connection refused / timeout / DNS
        return {
            "available": False,
            "models": [],
            "detail": f"GET {url} failed: {type(exc).__name__}: {exc}",
            "host": host,
        }

    try:
        payload = _json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 — non-JSON body
        return {
            "available": False,
            "models": [],
            "detail": f"GET {url} returned non-JSON body: {exc}",
            "host": host,
        }

    # Ollama ``/api/tags`` shape: {"models": [{"name": "llava:13b", ...}, ...]}.
    raw_models = payload.get("models") if isinstance(payload, dict) else None
    models = []
    if isinstance(raw_models, list):
        for item in raw_models:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model")
            else:
                name = item
            if name:
                models.append(str(name))
    return {
        "available": True,
        "models": models,
        "detail": f"GET {url} returned {len(models)} model(s).",
        "host": host,
    }


# --------------------------------------------------------------------- helpers


def _result(provider_name: str, *, ok: bool, status: str, detail: str) -> Dict[str, Any]:
    """Build the canonical test-provider response shape."""
    return {"provider": provider_name, "ok": bool(ok), "status": status, "detail": detail}


__all__ = [
    "build_settings_payload",
    "build_studio_settings_payload",
    "test_provider",
    "list_ollama_models",
]
