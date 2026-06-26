"""VRAM reclaim helper — evict a resident local ollama generation model.

The single-GPU coexistence problem: an 8 GB card shared between a
resident local ollama generation model (e.g. ``qwen2.5:7b-instruct``,
~5.3 GB) and the in-process NLI classifier
(:mod:`lib.classifiers.nli_classifier`, ~0.4 GB fp16 + several hundred
MiB forward-pass activations). When both want the card at once, the NLI
batch-8 x 512 forward pass spikes VRAM past the free budget and raises an
uncaught ``RuntimeError: CUDA out of memory`` mid-validation that the
orchestrator swallows tracebackless ("silent death").

The better design (the user directive this module realizes): generation
(7B via ollama) and validation (NLI) do NOT run simultaneously within a
phase. So instead of demoting NLI to CPU, EVICT the resident ollama
generation model to free the card, then run NLI on cuda for speed. The
7B lazy-reloads on the next generation HTTP request (ollama auto-loads on
demand), so no explicit reload is needed — eviction is a temporary
hand-off of the card, not a teardown.

This module owns ONLY the ollama-specific reclaim knowledge so the NLI
scoring path stays clean (it calls :func:`evict_local_llm` and re-probes
free VRAM; it never speaks the ollama wire protocol itself).

Eviction mechanism: ollama unloads a model when it receives a request
carrying ``"keep_alive": 0`` (documented — the model is unloaded after
that request returns). We discover which model(s) are actually resident
via ``GET {base_url}/api/ps`` (ollama's "list running models" endpoint)
and unload each one, so we evict what's really on the card rather than
guessing a single configured model name.

Base-URL resolution mirrors the local synthesis client
(:mod:`Trainforge.generators._local_provider`): the same
``LOCAL_SYNTHESIS_BASE_URL`` env (default ``http://localhost:11434/v1``).
That env carries the OpenAI-compatible ``/v1`` suffix; ollama's native
``/api/*`` endpoints live at the server ROOT, so we strip a trailing
``/v1`` to recover the native root.

Fully graceful: ANY error (ollama not running, HTTP failure, no models
resident, missing httpx) returns ``False`` and never raises — the caller
treats ``False`` as "could not free VRAM" and falls back to CPU.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Env var carrying the local OpenAI-compatible model server base URL —
#: the SAME env the local synthesis provider reads
#: (:data:`Trainforge.generators._local_provider.ENV_BASE_URL`). Default
#: is the Ollama default. We strip a trailing ``/v1`` to reach the native
#: ``/api/*`` endpoints.
ENV_BASE_URL = "LOCAL_SYNTHESIS_BASE_URL"

#: Default base URL (Ollama default), matching the local synthesis client.
DEFAULT_BASE_URL = "http://localhost:11434/v1"

#: Per-request timeout (seconds) for the reclaim HTTP calls. Short — this
#: is a fire-and-return hand-off, not a generation call; we don't want to
#: stall NLI loading on a wedged server.
_RECLAIM_TIMEOUT_SECONDS = 30.0


def resolve_ollama_root(base_url: Optional[str] = None) -> str:
    """Resolve the ollama server ROOT (no ``/v1`` suffix).

    Resolution chain (mirrors the local synthesis client): explicit
    ``base_url`` arg > ``LOCAL_SYNTHESIS_BASE_URL`` env > the Ollama
    default. The OpenAI-compatible ``/v1`` suffix is stripped because
    ollama's native ``/api/ps`` + ``/api/generate`` endpoints live at the
    server root, not under ``/v1``.
    """
    resolved = (
        base_url
        or os.environ.get(ENV_BASE_URL)
        or DEFAULT_BASE_URL
    ).strip() or DEFAULT_BASE_URL
    resolved = resolved.rstrip("/")
    if resolved.endswith("/v1"):
        resolved = resolved[: -len("/v1")].rstrip("/")
    return resolved


def _fetch_resident_models(client: "Any", root: str) -> List[Dict[str, Any]]:  # noqa: F821
    """Parse ``GET {root}/api/ps`` into resident-model entries.

    The SINGLE shared ollama ``/api/ps`` parser — both this module's evictor
    (:func:`_list_running_models`, which needs only the names) and the VRAM
    doctor (``lib.llm.vram_doctor._query_resident_models``, which needs the
    name + ``size_vram`` bytes → MiB) call this so the guard sequence
    (get → raise_for_status → json → dict guard → models-list → per-entry
    name) lives in exactly one place.

    ``GET {root}/api/ps`` returns ``{"models": [{"name": "...",
    "size_vram": <bytes>, ...}, ...]}``. Returns a list of
    ``{"name": str, "size_vram": <raw>}`` dicts — one per resident model
    that carries a usable name (``"name"`` or ``"model"``). A response of an
    unexpected-but-non-erroring shape yields ``[]``.

    NOTE: this helper does NOT swallow transport/JSON errors — it lets a
    failed ``client.get`` / ``raise_for_status`` / ``json`` propagate so each
    caller can decide how to report it. Both call sites wrap it best-effort
    (never raise; ollama-down → ``[]`` / a degraded value).
    """
    resp = client.get(f"{root}/api/ps")
    resp.raise_for_status()
    body = resp.json()
    models = body.get("models") if isinstance(body, dict) else None
    if not isinstance(models, list):
        return []
    entries: List[Dict[str, Any]] = []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        # ollama uses "name" (and sometimes "model"); accept either.
        name = entry.get("name") or entry.get("model")
        if isinstance(name, str) and name.strip():
            entries.append({"name": name.strip(), "size_vram": entry.get("size_vram")})
    return entries


def _list_running_models(client: "Any", root: str) -> List[str]:  # noqa: F821
    """Return the model names ollama reports as currently resident.

    Thin wrapper over the shared :func:`_fetch_resident_models` parser.
    Best-effort: any error / unexpected shape returns an empty list.
    """
    try:
        entries = _fetch_resident_models(client, root)
    except Exception as exc:  # noqa: BLE001 — server down / bad JSON / etc.
        logger.debug("vram_reclaim: /api/ps probe failed: %s", exc)
        return []
    return [entry["name"] for entry in entries]


def _unload_model(client: "Any", root: str, model: str) -> bool:  # noqa: F821
    """Unload one resident ollama model via ``keep_alive: 0``.

    POSTs an empty-prompt generate with ``keep_alive=0``; ollama unloads
    the model after the request returns (documented). Best-effort —
    returns False on any error.
    """
    try:
        resp = client.post(
            f"{root}/api/generate",
            json={"model": model, "keep_alive": 0},
        )
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("vram_reclaim: unload of %r failed: %s", model, exc)
        return False


def evict_local_llm(base_url: Optional[str] = None) -> bool:
    """Evict resident ollama generation model(s) to free VRAM.

    Discovers the resident model(s) via ``GET {root}/api/ps`` and unloads
    each via a ``keep_alive: 0`` generate request. The model lazy-reloads
    on the next generation HTTP request (ollama auto-loads on demand), so
    eviction is a temporary card hand-off — no explicit reload is needed.

    Returns ``True`` when at least one resident model was successfully
    asked to unload (VRAM is being freed), ``False`` otherwise — including
    when no model is resident (nothing to free → the caller's VRAM probe
    will see whatever is actually free), when ollama is not running, or on
    any HTTP / dependency error. NEVER raises: any failure degrades to
    ``False`` so the NLI caller can fall back to CPU.
    """
    try:
        import httpx  # type: ignore
    except Exception as exc:  # noqa: BLE001 — httpx absent / broken
        logger.debug("vram_reclaim: httpx unavailable (%s); cannot evict.", exc)
        return False

    root = resolve_ollama_root(base_url)
    try:
        with httpx.Client(timeout=_RECLAIM_TIMEOUT_SECONDS) as client:
            resident = _list_running_models(client, root)
            if not resident:
                logger.debug(
                    "vram_reclaim: no resident ollama models at %s "
                    "(nothing to evict).",
                    root,
                )
                return False
            unloaded = [m for m in resident if _unload_model(client, root, m)]
    except Exception as exc:  # noqa: BLE001 — client construction / network
        logger.debug("vram_reclaim: eviction failed against %s: %s", root, exc)
        return False

    if unloaded:
        logger.info(
            "vram_reclaim: evicted %d ollama model(s) %s at %s to free VRAM "
            "for NLI; they lazy-reload on the next generation request.",
            len(unloaded), unloaded, root,
        )
        return True
    logger.debug(
        "vram_reclaim: %d resident model(s) at %s but none unloaded "
        "successfully.",
        len(resident), root,
    )
    return False


__all__ = [
    "evict_local_llm",
    "resolve_ollama_root",
    "ENV_BASE_URL",
    "DEFAULT_BASE_URL",
]
