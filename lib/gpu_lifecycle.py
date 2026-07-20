"""Deterministic GPU-lifecycle lease helper (Ed4All venv).

GOAL: **deterministic lease semantics** — every GPU model
loads, does its job, releases the card, and hands it to the next stage. This
is NOT a contention heuristic (that is what
:mod:`lib.classifiers.nli_classifier` eviction + :mod:`lib.llm.vram_doctor`
already provide). A model stays RESIDENT for the whole of a stage's batch;
release fires only at a stage / phase BOUNDARY.

The three arms:

* :func:`release_ollama_models` — a thin delegate over the EXISTING
  :mod:`lib.llm.vram_reclaim` primitives (``resolve_ollama_root`` /
  ``_fetch_resident_models`` / ``_unload_model``). That module already does the
  ``GET /api/ps`` enumerate + ``keep_alive:0``-per-resident-model unload
  fail-soft; we do NOT re-implement (or fork) the parser here — we only add
  stage-context logging and RETURN the evicted names. Models lazy-reload on the
  next generation request (ollama auto-loads on demand), so a release is a card
  HAND-OFF, not a teardown.
* :func:`release_torch` — ``gc.collect()`` + ``torch.cuda.empty_cache()`` with a
  lazy torch import (precedent: ``release_council_gpu`` tail /
  ``LlamaCppRuntime.free`` tail). HONEST LIMIT: ``empty_cache`` frees the
  allocator CACHE, not weights held by a live in-process module singleton (the
  NLI classifier, the ``st`` embedding provider). So this arm ALSO runs a small
  registry of releaser callables (:func:`register_releaser`) — a heavy
  singleton can register an unloader so the sweep genuinely evicts it. v1 ships
  the registry + the torch/ollama arms; wiring the NLI-singleton releaser is a
  LISTED FOLLOW-UP, not silently claimed here.
* :func:`stage_lease` — a context manager that yields, then releases on exit
  (including on exception). A pure no-op when the lifecycle mode is off, so a
  flag-off run makes ZERO release calls (byte-identical control flow).

Everything is best-effort / NEVER-raises (mirrors ``vram_reclaim``'s
never-raise contract): a release failure logs a warning and never crashes the
phase / stage it is observing.

Flag: ``ED4ALL_GPU_LIFECYCLE`` — **default ON** (a documented deviation from the
default-off house convention; see :func:`resolve_gpu_lifecycle_mode` +
``docs/operations/behavior-flags.md``). It changes model RESIDENCY / timing
only, never an output byte, so the byte-identical-when-unset contract (which
protects ARTIFACTS) is not in scope; the opt-out
(``ED4ALL_GPU_LIFECYCLE=0/false/no/off``) exists for perf-sensitive runs where
the per-boundary cold reload of a same-seat model hurts.

Cross-venv twin: :mod:`SemantiK.semantik_structure.gpu_lifecycle` is a
self-contained ~parallel implementation of the ollama + torch arms, because the
out-of-process SemantiK cascade bridge cannot import Ed4All's ``lib/``. Any
change to the ``/api/ps`` sweep shape must land in BOTH modules — the twin
carries the same note.
"""

from __future__ import annotations

import contextlib
import gc
import logging
import os
from typing import Callable, List, Optional

from lib.llm.vram_reclaim import (
    _fetch_resident_models,
    _unload_model,
    resolve_ollama_root,
)

logger = logging.getLogger(__name__)

#: Master switch for the deterministic GPU-lifecycle sweep. **Default ON.**
#: Only an explicit falsey token (``0`` / ``false`` / ``no`` / ``off``,
#: case-insensitive) disables; unset / blank / truthy / garbage → on (the exact
#: parse semantics of ``ED4ALL_NLI_EVICT_FOR_CUDA``). Documented in root
#: CLAUDE.md + docs/operations/behavior-flags.md.
ENV_GPU_LIFECYCLE = "ED4ALL_GPU_LIFECYCLE"

_FALSEY = frozenset({"0", "false", "no", "off"})

#: Per-request HTTP timeout (seconds) for the sweep's ``/api/ps`` enumerate +
#: ``keep_alive:0`` unload calls. Matches the evictor's reclaim timeout — a
#: boundary sweep is a fire-and-return hand-off, not latency-critical, but must
#: not hang forever on a wedged server (each call is individually fail-soft).
_SWEEP_TIMEOUT_SECONDS = 30.0

#: In-process singleton unloaders. A heavy module that holds weights on the GPU
#: (NLI classifier, ``st`` embedding provider) can register an unloader so
#: :func:`release_torch` genuinely evicts it rather than only freeing the
#: allocator cache. Empty in v1 (the NLI wiring is a listed follow-up).
_RELEASERS: List[Callable[[], None]] = []


def resolve_gpu_lifecycle_mode(value: Optional[str] = None) -> bool:
    """Resolve whether the deterministic GPU-lifecycle sweep is enabled.

    Resolution chain: explicit ``value`` arg → ``ED4ALL_GPU_LIFECYCLE`` env →
    **default ON**. Parse-with-fallback: only the explicit falsey tokens
    (``0`` / ``false`` / ``no`` / ``off``, case-insensitive) disable; anything
    else — including unset / blank / garbage / truthy — keeps the default-ON
    lease behavior (mirrors :func:`lib.classifiers.nli_classifier.
    resolve_evict_for_cuda`).
    """
    raw = value if value is not None else os.environ.get(ENV_GPU_LIFECYCLE)
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() not in _FALSEY


def register_releaser(fn: Callable[[], None]) -> None:
    """Register an in-process singleton unloader run by :func:`release_torch`.

    Idempotent (a callable already registered is not added twice). Best-effort:
    a non-callable is ignored. The registered callables should themselves be
    fail-soft — :func:`release_torch` wraps each call, so a raising releaser
    does not stop the others, but a well-behaved unloader logs its own errors.
    """
    if callable(fn) and fn not in _RELEASERS:
        _RELEASERS.append(fn)


def release_ollama_models(
    base_url: Optional[str] = None, *, stage: Optional[str] = None
) -> List[str]:
    """Release the resident ollama generation model(s); return evicted names.

    A thin delegate over the EXISTING :mod:`lib.llm.vram_reclaim` primitives:
    enumerate the resident models via ``GET {root}/api/ps``
    (:func:`_fetch_resident_models`) and unload each via a ``keep_alive:0``
    generate request (:func:`_unload_model`). We do NOT re-implement / fork that
    parser — this function only adds the stage-context log and returns the list
    of names that were successfully asked to unload.

    ``stage`` is an optional label (e.g. ``"phase:content_generation"`` or
    ``"post-Stage-5e"``) threaded into the log so a trajectory reader can see
    WHERE the hand-off fired. Best-effort — NEVER raises: httpx absent, ollama
    down, a bad ``/api/ps`` shape, or an unload failure all degrade to an empty
    / partial list.
    """
    try:
        import httpx  # type: ignore
    except Exception as exc:  # noqa: BLE001 — httpx absent / broken
        logger.debug("gpu_lifecycle: httpx unavailable (%s); no ollama sweep.", exc)
        return []

    root = resolve_ollama_root(base_url)
    evicted: List[str] = []
    try:
        with httpx.Client(timeout=_SWEEP_TIMEOUT_SECONDS) as client:
            try:
                entries = _fetch_resident_models(client, root)
            except Exception as exc:  # noqa: BLE001 — server down / bad JSON
                logger.debug(
                    "gpu_lifecycle: /api/ps probe failed at %s: %s", root, exc
                )
                return []
            for entry in entries:
                name = entry.get("name")
                if name and _unload_model(client, root, name):
                    evicted.append(name)
    except Exception as exc:  # noqa: BLE001 — client construction / network
        logger.debug("gpu_lifecycle: ollama sweep failed at %s: %s", root, exc)
        return []

    if evicted:
        logger.info(
            "gpu_lifecycle: released %d ollama model(s) %s at %s%s "
            "(card hand-off; they lazy-reload on the next generation request).",
            len(evicted),
            evicted,
            root,
            f" after {stage}" if stage else "",
        )
    return evicted


def release_torch(*, stage: Optional[str] = None) -> None:
    """Run registered singleton unloaders, then ``gc`` + ``empty_cache``.

    HONEST LIMIT: ``torch.cuda.empty_cache()`` frees the ALLOCATOR CACHE, not
    weights held by a live in-process module singleton. Those weights are freed
    only by the registered unloaders (:func:`register_releaser`) — the registry
    runs FIRST so a freed singleton's memory is then reclaimed into the
    allocator/OS by the ``gc`` + ``empty_cache`` tail. Best-effort — NEVER
    raises: torch absent / a raising registered releaser / an ``empty_cache``
    failure all degrade to a debug log.
    """
    for fn in list(_RELEASERS):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — one releaser must not stop the rest
            logger.warning(
                "gpu_lifecycle: registered releaser %r raised: %s", fn, exc
            )

    try:
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        logger.debug("gpu_lifecycle: gc.collect() failed (%s).", exc)

    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 — torch absent / no CUDA / broken
        logger.debug("gpu_lifecycle: torch empty_cache skipped (%s).", exc)

    if stage:
        logger.debug("gpu_lifecycle: torch release fired after %s.", stage)


@contextlib.contextmanager
def stage_lease(
    stage_name: str, *, ollama: bool = True, torch: bool = True
):
    """Deterministic GPU lease around a stage's batch.

    Yields immediately; on exit (INCLUDING when the body raises) releases the
    requested arms so the card is handed to the next stage. When the lifecycle
    mode is OFF this is a pure no-op — the body runs and NO release fires, so a
    flag-off path is byte-identical control flow.

    All releases are fail-soft: a release failure logs a warning and never
    propagates, so the lease never converts a stage's own exception into a
    different one (the body's exception, if any, still propagates after the
    releases run).

    ``ollama`` / ``torch`` select which arms fire — e.g. a captioner stage that
    only used torch weights can pass ``ollama=False`` to skip a pointless
    ``/api/ps`` probe. Designed so the future Phase-4 VLM extraction arm can
    adopt this directly (``with stage_lease("vlm-extract"): ...`` releases the
    ``qwen2.5vl:7b`` seat before the council loads — the deterministic
    replacement for the plan's "serialize behind gpu_guard" note).
    """
    enabled = resolve_gpu_lifecycle_mode()
    try:
        yield
    finally:
        if enabled:
            if ollama:
                try:
                    release_ollama_models(stage=stage_name)
                except Exception as exc:  # noqa: BLE001 — lease must never crash the stage
                    logger.warning(
                        "gpu_lifecycle: ollama release at %s failed: %s",
                        stage_name, exc,
                    )
            if torch:
                try:
                    release_torch(stage=stage_name)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "gpu_lifecycle: torch release at %s failed: %s",
                        stage_name, exc,
                    )


__all__ = [
    "ENV_GPU_LIFECYCLE",
    "resolve_gpu_lifecycle_mode",
    "register_releaser",
    "release_ollama_models",
    "release_torch",
    "stage_lease",
]
