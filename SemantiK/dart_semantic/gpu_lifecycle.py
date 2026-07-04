"""Deterministic GPU-lifecycle lease helper — SemantiK cross-venv TWIN.

This is a SELF-CONTAINED twin of ``lib/gpu_lifecycle.py`` (the Ed4All-venv
original). The out-of-process SemantiK cascade bridge runs in its OWN venv and
CANNOT import Ed4All's ``lib/`` — so the ollama ``/api/ps`` + ``keep_alive:0``
sweep and the torch release are re-implemented here with ZERO Ed4All import
(this module imports only the stdlib + a lazy ``httpx`` / ``torch``).

**Twin-drift contract:** any change to the ``/api/ps`` sweep shape (the
enumerate → per-model ``keep_alive:0`` unload) MUST land in BOTH this module
and ``lib/gpu_lifecycle.py`` (which delegates to ``lib/llm/vram_reclaim.py``).
The two are intentionally parallel, not shared.

Lease semantics + flag (``ED4ALL_GPU_LIFECYCLE``, **default ON**, only an
explicit falsey token disables) are identical to the original: a model stays
resident within a cascade STAGE's batch and the release fires only at a stage
SEAM (post-Stage-5e / pre-Stage-6 ollama hand-off, post-captioner torch, post
theta/Stage-12 torch, post second-pass+ocr_repair ollama). Every release is
best-effort / NEVER-raises so it can never crash a cascade stage.

Base-URL resolution mirrors the SemantiK endpoint seat
(``endpoint_runtime._resolve_base_url``): ``SEMANTIK_SPECIALIST_BASE_URL`` >
``NVIDIA_BASE_URL`` > the local ollama default — so this twin sweeps the
endpoint the cascade itself dispatched to (the orchestrator side sweeps its own
``LOCAL_SYNTHESIS_BASE_URL`` root; a box running two ollama endpoints has each
side sweep only the one it knows — documented base-URL split-brain).
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

#: Master switch — TWIN of ``lib.gpu_lifecycle.ENV_GPU_LIFECYCLE``. Default ON;
#: only an explicit falsey token disables. Root-owned ``ED4ALL_*`` flag (read,
#: not owned, by SemantiK — no ``SEMANTIK_*`` row).
ENV_GPU_LIFECYCLE = "ED4ALL_GPU_LIFECYCLE"

_FALSEY = frozenset({"0", "false", "no", "off"})

#: Default ollama root (native ``/api/*`` server root, NO ``/v1`` suffix).
_DEFAULT_OLLAMA_ROOT = "http://localhost:11434"

#: Per-request HTTP timeout (seconds) for the sweep calls.
_SWEEP_TIMEOUT_SECONDS = 30.0


def resolve_gpu_lifecycle_mode(value: Optional[str] = None) -> bool:
    """True iff the deterministic GPU-lifecycle sweep is enabled (default ON).

    Explicit ``value`` arg → ``ED4ALL_GPU_LIFECYCLE`` env → default ON. Only an
    explicit falsey token (``0`` / ``false`` / ``no`` / ``off``) disables;
    unset / blank / garbage / truthy → on. Byte-parallel to the twin.
    """
    raw = value if value is not None else os.environ.get(ENV_GPU_LIFECYCLE)
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in _FALSEY


def resolve_ollama_root(base_url: Optional[str] = None) -> str:
    """Resolve the ollama server ROOT (no ``/v1`` suffix) for the SemantiK seat.

    Resolution: explicit ``base_url`` arg > ``SEMANTIK_SPECIALIST_BASE_URL`` >
    ``NVIDIA_BASE_URL`` > the local ollama default. A trailing ``/v1``
    (OpenAI-compatible suffix) is stripped so the native ``/api/ps`` +
    ``/api/generate`` endpoints resolve.
    """
    resolved = (
        base_url
        or os.environ.get("SEMANTIK_SPECIALIST_BASE_URL")
        or os.environ.get("NVIDIA_BASE_URL")
        or _DEFAULT_OLLAMA_ROOT
    ).strip() or _DEFAULT_OLLAMA_ROOT
    resolved = resolved.rstrip("/")
    if resolved.endswith("/v1"):
        resolved = resolved[: -len("/v1")].rstrip("/")
    return resolved


def _fetch_resident_models(client: Any, root: str) -> List[Dict[str, Any]]:
    """Parse ``GET {root}/api/ps`` into resident-model entries (twin parser).

    Byte-parallel to ``lib.llm.vram_reclaim._fetch_resident_models`` — returns a
    list of ``{"name": str}`` dicts, one per resident model carrying a usable
    name. Lets transport / JSON errors propagate; the caller wraps it
    best-effort.
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
        name = entry.get("name") or entry.get("model")
        if isinstance(name, str) and name.strip():
            entries.append({"name": name.strip()})
    return entries


def _unload_model(client: Any, root: str, model: str) -> bool:
    """Unload one resident ollama model via ``keep_alive:0`` (twin unloader).

    Best-effort — returns False on any error.
    """
    try:
        resp = client.post(
            f"{root}/api/generate",
            json={"model": model, "keep_alive": 0},
        )
        resp.raise_for_status()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("gpu_lifecycle(twin): unload of %r failed: %s", model, exc)
        return False


def release_ollama_models(
    base_url: Optional[str] = None, *, stage: Optional[str] = None
) -> List[str]:
    """Release the resident ollama model(s); return evicted names (twin arm).

    Enumerate via ``GET {root}/api/ps`` + unload each via ``keep_alive:0``.
    Best-effort — NEVER raises: httpx absent, ollama down, bad shape, or an
    unload failure all degrade to an empty / partial list. Models lazy-reload
    on the next generation request (card hand-off, not a teardown).
    """
    try:
        import httpx  # type: ignore
    except Exception as exc:  # noqa: BLE001 — httpx absent / broken
        logger.debug("gpu_lifecycle(twin): httpx unavailable (%s); no sweep.", exc)
        return []

    root = resolve_ollama_root(base_url)
    evicted: List[str] = []
    try:
        with httpx.Client(timeout=_SWEEP_TIMEOUT_SECONDS) as client:
            try:
                entries = _fetch_resident_models(client, root)
            except Exception as exc:  # noqa: BLE001 — server down / bad JSON
                logger.debug(
                    "gpu_lifecycle(twin): /api/ps probe failed at %s: %s",
                    root, exc,
                )
                return []
            for entry in entries:
                name = entry.get("name")
                if name and _unload_model(client, root, name):
                    evicted.append(name)
    except Exception as exc:  # noqa: BLE001 — client construction / network
        logger.debug("gpu_lifecycle(twin): ollama sweep failed at %s: %s", root, exc)
        return []

    if evicted:
        logger.info(
            "gpu_lifecycle(twin): released %d ollama model(s) %s at %s%s "
            "(card hand-off; lazy-reload on next request).",
            len(evicted),
            evicted,
            root,
            f" after {stage}" if stage else "",
        )
    return evicted


def release_torch(*, stage: Optional[str] = None) -> None:
    """``gc.collect()`` + ``torch.cuda.empty_cache()`` (twin torch arm).

    Lazy torch import; best-effort — NEVER raises. HONEST LIMIT (same as the
    original): frees the allocator cache, not weights held by a live module
    singleton. The cascade's canonical stage releases (``release_council_gpu``,
    ``LlamaCppRuntime.free``, theta scope-drop) own the real weight teardown;
    this arm is the belt-and-suspenders allocator reclaim at the seam.
    """
    try:
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        logger.debug("gpu_lifecycle(twin): gc.collect() failed (%s).", exc)

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 — torch absent / no CUDA / broken
        logger.debug("gpu_lifecycle(twin): torch empty_cache skipped (%s).", exc)

    if stage:
        logger.debug("gpu_lifecycle(twin): torch release fired after %s.", stage)


@contextlib.contextmanager
def stage_lease(
    stage_name: str, *, ollama: bool = True, torch: bool = True
):
    """Deterministic GPU lease around a cascade stage (twin).

    Yields; on exit (incl. exception) releases the requested arms. Pure no-op
    when the lifecycle mode is off (zero release calls → byte-identical control
    flow). All releases fail-soft.
    """
    enabled = resolve_gpu_lifecycle_mode()
    try:
        yield
    finally:
        if enabled:
            if ollama:
                try:
                    release_ollama_models(stage=stage_name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "gpu_lifecycle(twin): ollama release at %s failed: %s",
                        stage_name, exc,
                    )
            if torch:
                try:
                    release_torch(stage=stage_name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "gpu_lifecycle(twin): torch release at %s failed: %s",
                        stage_name, exc,
                    )


__all__ = [
    "ENV_GPU_LIFECYCLE",
    "resolve_gpu_lifecycle_mode",
    "resolve_ollama_root",
    "release_ollama_models",
    "release_torch",
    "stage_lease",
]
