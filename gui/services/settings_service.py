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

# Default strict OpenAI-compatible local endpoint. TRT-LLM/vLLM deployments
# may override the host/port through LOCAL_SYNTHESIS_BASE_URL.
_DEFAULT_LOCAL_BASE_URL = "http://localhost:8000/v1"

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
# local model server — vLLM / Ollama / llama.cpp — can be pointed at). The full
# operator catalog (conversion /
# per-tier Courseforge / Trainforge / embedding knobs) is intentionally hidden from
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

# The flags the Studio surface reads (read-only echo). Only the Courseforge
# two-pass flag is needed today: the Create-wizard AI-tier flow tree greys the
# Validate/Rewrite node when it is off. NOT a secret; never patched here.
_STUDIO_FLAG_KEYS = ("COURSEFORGE_TWO_PASS",)


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

    # Read-only flag echo scoped to the flags the Studio surface needs (the
    # Create-wizard AI-tier flow tree greys the Validate/Rewrite node off the
    # two-pass flag). NOT a secret; the page renders it, never patches it here.
    all_flags = masked.get("flags") if isinstance(masked, dict) else None
    flags = (
        {k: v for k, v in all_flags.items() if k in _STUDIO_FLAG_KEYS}
        if isinstance(all_flags, dict)
        else {}
    )

    return {
        "version": masked.get("version") if isinstance(masked, dict) else None,
        "updated_at": masked.get("updated_at") if isinstance(masked, dict) else None,
        "env": env,
        "model_routing": routing,
        "flags": flags,
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


# ----------------------------------------------------- local model discovery


def _local_server_root(settings_env: Dict[str, Any]) -> str:
    """Resolve the local model server's ROOT host URL (no trailing ``/`` or ``/v1``).

    Base URL from ``LOCAL_SYNTHESIS_BASE_URL`` (or the default). The seat-registry
    convention is a bare host root (``http://localhost:8001``) while some envs
    carry the OpenAI-client ``/v1`` form (``http://localhost:11434/v1``). Both
    resolve to the same ROOT here so the caller can compose ``/v1/models`` and
    ``/api/tags`` from it without ever doubling the ``/v1`` suffix.
    """
    base_url = (
        _resolve_env_value("LOCAL_SYNTHESIS_BASE_URL", settings_env)
        or _DEFAULT_LOCAL_BASE_URL
    )
    root = str(base_url).rstrip("/")
    if root.endswith("/v1"):
        root = root[: -len("/v1")].rstrip("/")
    return root


def _http_get_json(url: str, key: Optional[str]) -> Any:
    """Bounded ``GET url`` returning parsed JSON, or raising on any failure.

    Never used at module scope — every caller wraps it in try/except and degrades
    gracefully. Raises so the caller can distinguish reachable-but-bad from
    unreachable in its own vendor-neutral detail message.
    """
    import json as _json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415

    request = urllib.request.Request(url, method="GET")
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_S) as response:  # noqa: S310
        return _json.loads(response.read().decode("utf-8"))


def _probe_openai_models(root: str, key: Optional[str]) -> Dict[str, Any]:
    """Probe the OpenAI-compatible ``GET {root}/v1/models`` (vLLM / Ollama / …).

    Returns ``{"ok": bool, "models": [<id>, ...], "detail": str}``. ``ok`` is True
    when the endpoint answered with a parseable body (even if the model list is
    empty). Never raises. The ``/v1`` suffix is composed from the ROOT here, so it
    is never doubled.
    """
    url = root + "/v1/models"
    try:
        payload = _http_get_json(url, key)
    except Exception as exc:  # noqa: BLE001 — refused / timeout / DNS / HTTP / bad JSON
        return {"ok": False, "models": [], "detail": f"GET {url} failed: {type(exc).__name__}: {exc}"}
    # OpenAI ``/v1/models`` shape: {"data": [{"id": "nemotron-3-super", ...}, ...]}.
    data = payload.get("data") if isinstance(payload, dict) else None
    models: list = []
    if isinstance(data, list):
        for item in data:
            ident = item.get("id") if isinstance(item, dict) else item
            if ident:
                models.append(str(ident))
    return {"ok": True, "models": models, "detail": f"GET {url} returned {len(models)} model(s)."}


def _probe_ollama_tags(root: str, key: Optional[str]) -> Dict[str, Any]:
    """Probe Ollama's native ``GET {root}/api/tags`` (Ollama-specific fallback).

    Returns ``{"ok": bool, "models": [<name>, ...], "detail": str}``. Ollama's
    tags API lives on the bare host root, NOT under the OpenAI-compat ``/v1``
    prefix. Never raises.
    """
    url = root + "/api/tags"
    try:
        payload = _http_get_json(url, key)
    except Exception as exc:  # noqa: BLE001 — refused / timeout / DNS / HTTP / bad JSON
        return {"ok": False, "models": [], "detail": f"GET {url} failed: {type(exc).__name__}: {exc}"}
    # Ollama ``/api/tags`` shape: {"models": [{"name": "llava:13b", ...}, ...]}.
    raw_models = payload.get("models") if isinstance(payload, dict) else None
    models: list = []
    if isinstance(raw_models, list):
        for item in raw_models:
            name = (item.get("name") or item.get("model")) if isinstance(item, dict) else item
            if name:
                models.append(str(name))
    return {"ok": True, "models": models, "detail": f"GET {url} returned {len(models)} model(s)."}


def list_local_models() -> Dict[str, Any]:
    """Discover models on the local model server (protocol-first, REAL live probe).

    Backend-agnostic. The local server may be vLLM, Ollama, llama.cpp, LM Studio,
    or any other OpenAI-compatible server. Discovery is PROTOCOL-FIRST, not
    vendor-first: it probes the OpenAI-compatible ``GET {root}/v1/models`` FIRST
    (every one of the above serves it) and falls back to Ollama's native
    ``GET {root}/api/tags`` only when the OpenAI path fails or returns nothing.

    The host root comes from ``LOCAL_SYNTHESIS_BASE_URL`` (or the default). A
    trailing ``/v1`` on that base URL is detected and never doubled into
    ``/v1/v1/models`` (the seat-registry convention is a bare host root, but some
    envs carry the OpenAI-client ``/v1`` form). Returns::

        {"available": bool, "models": [<id>, ...], "detail": str,
         "host": str, "backend": "openai-compatible"|"ollama"|None}

    ``backend`` names what actually ANSWERED so the UI can report it. Graceful on
    any connection failure (refused / timeout / DNS / HTTP error / bad JSON):
    ``available=False``, ``models=[]``, ``backend=None`` and a human-readable,
    vendor-neutral ``detail``. Powers the live model dropdowns in the GUI — a
    real discovery, never a hardcoded list.
    """
    settings_env = _settings_env()
    root = _local_server_root(settings_env)
    key = _resolve_env_value("LOCAL_SYNTHESIS_API_KEY", settings_env)

    # Protocol-first: the OpenAI-compatible endpoint (served by vLLM AND Ollama).
    v1 = _probe_openai_models(root, key)
    if v1["ok"] and v1["models"]:
        return {
            "available": True,
            "models": v1["models"],
            "detail": v1["detail"],
            "host": root,
            "backend": "openai-compatible",
        }

    # Vendor fallback: Ollama's native tags API (only when the OpenAI path failed
    # or returned nothing).
    tags = _probe_ollama_tags(root, key)
    if tags["ok"] and tags["models"]:
        return {
            "available": True,
            "models": tags["models"],
            "detail": tags["detail"],
            "host": root,
            "backend": "ollama",
        }

    # Reachable but empty: prefer the protocol-first backend when it answered.
    if v1["ok"]:
        return {
            "available": True,
            "models": [],
            "detail": v1["detail"],
            "host": root,
            "backend": "openai-compatible",
        }
    if tags["ok"]:
        return {
            "available": True,
            "models": [],
            "detail": tags["detail"],
            "host": root,
            "backend": "ollama",
        }

    # Neither endpoint answered — a genuinely unreachable local server. The
    # detail names both probes and the base URL, blaming no specific vendor.
    return {
        "available": False,
        "models": [],
        "detail": (
            f"Local model server not reachable at {root} "
            f"({v1['detail']}; {tags['detail']})."
        ),
        "host": root,
        "backend": None,
    }


def list_ollama_models() -> Dict[str, Any]:
    """Deprecated alias for :func:`list_local_models` (kept for external callers).

    Backend discovery is now protocol-first and vendor-neutral; this alias simply
    delegates so nothing that imported the old name breaks.
    """
    return list_local_models()


# --------------------------------------------------------------------- helpers


def _result(provider_name: str, *, ok: bool, status: str, detail: str) -> Dict[str, Any]:
    """Build the canonical test-provider response shape."""
    return {"provider": provider_name, "ok": bool(ok), "status": status, "detail": detail}


__all__ = [
    "build_settings_payload",
    "build_studio_settings_payload",
    "test_provider",
    "list_local_models",
    "list_ollama_models",
]
