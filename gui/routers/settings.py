"""Settings REST router — ``/api/settings/*`` (env, keys, providers, routing).

Mounted at prefix ``/api/settings`` by ``gui.app``, so paths here are relative
(``""``, ``"/apply"``, ``"/test-provider"``). The router is thin: every bit of
logic lives in ``gui.services.settings_service`` and ``gui.settings_store``.

Endpoint contract (frontend-facing):

* ``GET    ""``              → masked settings doc + catalog + providers + base_models
* ``PUT    ""``             → save a full settings doc, return masked doc
* ``PATCH  ""``             → deep-merge a partial patch, return masked doc
* ``POST   "/apply"``       → render+apply settings env, return applied key NAMES
* ``POST   "/test-provider"`` → REAL per-provider reachability check

Validation / store errors surface as a typed ``{error, detail}`` body with the
right status code (422 for bad input, 500 for the unexpected) — never a faked
success.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from gui import settings_store
from gui.services import settings_service

router = APIRouter()


# ----------------------------------------------------------------- request models


class SettingsPatch(BaseModel):
    """Partial settings patch for ``PATCH ""`` (deep-merged into the stored doc).

    All sections optional; only the present sections are merged. Unknown extra
    keys are tolerated and forwarded to the store's deep-merge (the store drops
    unknown ``env`` keys on validation).
    """

    env: Optional[Dict[str, Any]] = None
    flags: Optional[Dict[str, Any]] = None
    model_routing: Optional[Dict[str, Any]] = None
    retrieval: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class TestProviderRequest(BaseModel):
    """Body for ``POST "/test-provider"`` — the provider name to probe."""

    provider: str


# ------------------------------------------------------------------- error helper


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    """Return a typed ``{error, detail}`` JSON error with ``status_code``."""
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


# ----------------------------------------------------------------------- endpoints


@router.get("")
async def get_settings() -> Any:
    """Return the masked settings doc enriched with catalog / providers / models."""
    try:
        return settings_service.build_settings_payload()
    except Exception as exc:  # noqa: BLE001 — unexpected store/catalog failure
        return _error(500, "settings_load_failed", str(exc))


@router.put("")
async def put_settings(doc: Dict[str, Any]) -> Any:
    """Persist a full settings doc (validated), returning the masked stored doc."""
    try:
        stored = settings_store.save_settings(doc)
    except ValueError as exc:
        return _error(422, "settings_invalid", str(exc))
    except Exception as exc:  # noqa: BLE001 — unexpected write failure
        return _error(500, "settings_save_failed", str(exc))
    return settings_store.mask_secrets(stored)


@router.patch("")
async def patch_settings(patch: SettingsPatch) -> Any:
    """Deep-merge a partial patch into the stored doc; return the masked doc."""
    payload = patch.model_dump(exclude_none=True)
    try:
        stored = settings_store.update_settings(payload)
    except ValueError as exc:
        return _error(422, "settings_invalid", str(exc))
    except Exception as exc:  # noqa: BLE001 — unexpected merge/write failure
        return _error(500, "settings_patch_failed", str(exc))
    return settings_store.mask_secrets(stored)


@router.post("/apply")
async def apply_settings() -> Any:
    """Render + apply the stored settings into ``os.environ``.

    Returns ``{"applied": [<key names>]}`` — env var NAMES only, never the
    resolved values (so a secret never leaves through this endpoint).
    """
    try:
        applied = settings_store.apply_env(settings_store.load_settings())
    except Exception as exc:  # noqa: BLE001 — unexpected render/apply failure
        return _error(500, "settings_apply_failed", str(exc))
    return {"applied": sorted(applied.keys())}


@router.get("/studio")
async def get_studio_settings() -> Any:
    """Return the masked settings doc scoped to the Studio-user subset.

    Strict subset of ``GET ""`` (credentials / global / answer / local
    categories + the authoring + answer routing tasks) plus a read-only
    host/port echo. The Studio settings page renders from this so a
    non-developer never sees the full operator catalog. Writes still go through
    ``PATCH ""`` (deep-patch dotted paths) — this endpoint is read-only.
    """
    try:
        return settings_service.build_studio_settings_payload()
    except Exception as exc:  # noqa: BLE001 — unexpected store/catalog failure
        return _error(500, "studio_settings_load_failed", str(exc))


@router.get("/ollama-models")
async def ollama_models() -> Any:
    """Discover models installed on the local Ollama server (live probe).

    Delegates to ``settings_service.list_ollama_models``; returns
    ``{available, models, detail, host}``. Graceful on connection failure
    (``available=False`` / empty ``models``) — never raises for an offline
    Ollama, so the service handles its own failures. A genuinely unexpected
    error surfaces as a typed 500.
    """
    try:
        return settings_service.list_ollama_models()
    except Exception as exc:  # noqa: BLE001 — service is graceful; this is a backstop
        return _error(500, "ollama_models_failed", str(exc))


@router.post("/test-provider")
async def test_provider(req: TestProviderRequest) -> Any:
    """Run a REAL reachability check for the named provider.

    Delegates to ``settings_service.test_provider``; returns
    ``{provider, ok, status, detail}``. Service-level exceptions are caught
    inside the service, so any failure here is genuinely unexpected.
    """
    try:
        return settings_service.test_provider(req.provider)
    except Exception as exc:  # noqa: BLE001 — service catches its own; this is a backstop
        return _error(500, "test_provider_failed", str(exc))


__all__ = ["router"]
