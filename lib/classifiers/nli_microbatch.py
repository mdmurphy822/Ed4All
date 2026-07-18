"""Opt-in NLI micro-batching dispatcher (``ED4ALL_NLI_MICROBATCH``).

Problem this solves: the Courseforge rewrite/outline two-pass tiers fan per-block
LLM dispatch across a thread pool (``COURSEFORGE_REWRITE_CONCURRENCY`` /
``COURSEFORGE_OUTLINE_CONCURRENCY``). Each block's best-of-N candidate selection
scores candidates with an NLI entailment pass
(``Courseforge/router/router.py::score_candidate`` → ``score_groundedness`` →
``NliClassifier.score_batch``), and the shared process-wide torch NLI model is
NOT safe under concurrent forward passes. The historical guard is a global mutex
(``Courseforge/router/router.py::_NLI_SCORE_LOCK``): only ONE thread inside the
GPU forward pass at a time. Measured live, that mutex caps effective in-flight
LLM requests at ~28 at 64 rewrite threads — concurrency >32 buys nothing because
the workers queue on the lock.

This module replaces the mutex with a **micro-batching dispatcher**: a single
background scorer thread owns the model; concurrent callers ENQUEUE their
``(premise, hypothesis)`` pair-batches and wait on a per-request event; the
scorer drains the queue (coalescing up to ``ED4ALL_NLI_MICROBATCH_MAX_PAIRS``
pairs, or whatever is queued after ``ED4ALL_NLI_MICROBATCH_WINDOW_MS``) and runs
ONE batched tokenizer+forward pass, then resolves each caller with exactly its
slice of the scores. The scorer thread IS the serialization point for the model,
so the ``_NLI_SCORE_LOCK`` mutex is redundant on the dispatcher path and the
router bypasses it (see ``score_candidate``).

Correctness contract:

* **Drop-in for ``nli``** — the dispatcher exposes ``score_batch(pairs=...)`` /
  ``score_pair(...)`` plus ``_revision`` / ``device`` / ``dtype`` so it can be
  passed as ``score_groundedness(..., nli=dispatcher)`` unchanged.
* **Semantically identical scores** — the underlying ``NliClassifier`` tokenizes
  with ``padding=True`` + per-pair ``attention_mask`` + ``truncation``, so each
  pair's score is independent of its batch-mates. Batched vs sequential scores
  can differ only in low-order floating-point bits (batched matmul reductions and
  padding change the reduction order); the verdict thresholds (0.70 / 0.50) are
  robust to that magnitude. See the flag row in
  ``docs/operations/behavior-flags.md``.
* **Never deadlock, never fabricate** — every dequeued request is resolved with
  either its scores OR the scorer's exception (surfaced to the caller, which
  re-raises). A scorer-side failure NEVER resolves a caller with fabricated
  scores.

Default OFF (``ED4ALL_NLI_MICROBATCH`` unset) → nothing here is constructed and
the router keeps the exact ``_NLI_SCORE_LOCK`` path (byte-identical behavior).
"""

from __future__ import annotations

import inspect
import logging
import os
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from lib.classifiers.nli_classifier import estimate_pair_tokens

logger = logging.getLogger(__name__)

#: Master gate. Default OFF → the dispatcher is never constructed and the router
#: keeps the ``_NLI_SCORE_LOCK`` mutex path.
ENV_MICROBATCH = "ED4ALL_NLI_MICROBATCH"

#: Max pairs coalesced into ONE drain/forward pass. Once a drain has accumulated
#: at least this many pairs it stops collecting and scores immediately.
ENV_MICROBATCH_MAX_PAIRS = "ED4ALL_NLI_MICROBATCH_MAX_PAIRS"
_DEFAULT_MAX_PAIRS = 64

#: Collection window (milliseconds). After the first request arrives the scorer
#: waits up to this long for more requests before scoring what it has.
ENV_MICROBATCH_WINDOW_MS = "ED4ALL_NLI_MICROBATCH_WINDOW_MS"
_DEFAULT_WINDOW_MS = 10.0

#: Cross-block gate NLI concurrency master flag. Default OFF → the
#: block_prose_entailment / claim_support / block_objective_delivery gates score
#: SERIALLY on the raw NLI singleton exactly as before (byte-identical). When
#: truthy, the SCORABLE blocks are fanned across a ``ThreadPoolExecutor`` whose
#: workers submit their pair-batches to THIS module's coalescing dispatcher (one
#: scorer thread owns the single GPU model — no extra CUDA contexts, unlike the
#: retired ``ED4ALL_NLI_MICROBATCH_VALIDATORS`` spawn-ProcessPool path that
#: livelocked 4 CUDA contexts on one card).
ENV_CROSSBLOCK = "ED4ALL_NLI_CROSSBLOCK"

#: Satellite of ``ED4ALL_NLI_CROSSBLOCK`` — the ThreadPoolExecutor worker count
#: fanning cache-miss blocks at the dispatcher. Default 16. Garbage / non-positive
#: → the default. No-op when the master flag is off.
ENV_CROSSBLOCK_THREADS = "ED4ALL_NLI_CROSSBLOCK_THREADS"
_DEFAULT_CROSSBLOCK_THREADS = 16

_TRUTHY = {"1", "true", "yes", "on"}


def resolve_microbatch_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``ED4ALL_NLI_MICROBATCH`` is set to a truthy token.

    Parse-with-fallback (mirrors the repo's other opt-in gates): only the
    explicit truthy tokens (``1``/``true``/``yes``/``on``, case-insensitive)
    enable it; anything else (unset / falsey / garbage) leaves it OFF so the
    default ``_NLI_SCORE_LOCK`` path is byte-identical.
    """
    src = env if env is not None else os.environ
    return str(src.get(ENV_MICROBATCH, "")).strip().lower() in _TRUTHY


def resolve_microbatch_max_pairs(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve the per-drain pair cap. Garbage / non-positive → the default."""
    src = env if env is not None else os.environ
    raw = src.get(ENV_MICROBATCH_MAX_PAIRS)
    if raw is None or not str(raw).strip():
        return _DEFAULT_MAX_PAIRS
    try:
        iv = int(str(raw).strip())
        return iv if iv > 0 else _DEFAULT_MAX_PAIRS
    except (TypeError, ValueError):
        return _DEFAULT_MAX_PAIRS


def resolve_microbatch_window_ms(env: Optional[Dict[str, str]] = None) -> float:
    """Resolve the collection window (ms). Garbage / negative → the default.

    ``0`` is honored (score as soon as the queue drains — no wait for coalescing
    beyond what is already enqueued).
    """
    src = env if env is not None else os.environ
    raw = src.get(ENV_MICROBATCH_WINDOW_MS)
    if raw is None or not str(raw).strip():
        return _DEFAULT_WINDOW_MS
    try:
        fv = float(str(raw).strip())
        return fv if fv >= 0 else _DEFAULT_WINDOW_MS
    except (TypeError, ValueError):
        return _DEFAULT_WINDOW_MS


def resolve_crossblock_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True when ``ED4ALL_NLI_CROSSBLOCK`` is set to a truthy token.

    Parse-with-fallback: only the explicit truthy tokens enable it; anything else
    (unset / falsey / garbage) leaves it OFF so the gate NLI path is byte-identical
    serial scoring.
    """
    src = env if env is not None else os.environ
    return str(src.get(ENV_CROSSBLOCK, "")).strip().lower() in _TRUTHY


def resolve_crossblock_threads(env: Optional[Dict[str, str]] = None) -> int:
    """Resolve the cross-block fan-out thread count. Garbage / ≤0 → the default."""
    src = env if env is not None else os.environ
    raw = src.get(ENV_CROSSBLOCK_THREADS)
    if raw is None or not str(raw).strip():
        return _DEFAULT_CROSSBLOCK_THREADS
    try:
        iv = int(str(raw).strip())
        return iv if iv > 0 else _DEFAULT_CROSSBLOCK_THREADS
    except (TypeError, ValueError):
        return _DEFAULT_CROSSBLOCK_THREADS


def _accepts_batch_size(fn: Any) -> bool:
    """True when ``fn`` accepts a ``batch_size`` keyword (or ``**kwargs``).

    The real :meth:`NliClassifier.score_batch` accepts ``batch_size`` so the
    dispatcher can request one big forward pass per drain; simple test fakes may
    not, so the dispatcher introspects once and calls accordingly.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    for param in sig.parameters.values():
        if param.name == "batch_size":
            return True
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return False


def _is_cuda_oom(exc: BaseException) -> bool:
    """True when ``exc`` looks like a CUDA out-of-memory error.

    Matches by type name / message so we don't hard-import torch (the dispatcher
    stays usable with stub scorers). ``torch.cuda.OutOfMemoryError`` and the
    generic ``RuntimeError('CUDA out of memory')`` both match.
    """
    name = type(exc).__name__
    if "OutOfMemory" in name:
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda oom" in msg


def _empty_cuda_cache() -> None:
    """Best-effort ``torch.cuda.empty_cache()`` before an OOM retry (never raises)."""
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — best-effort only
        pass


class _Request:
    """One caller's pending pair-batch + its completion slot."""

    __slots__ = ("pairs", "event", "result", "exception")

    def __init__(self, pairs: List[Tuple[str, str]]) -> None:
        self.pairs = pairs
        self.event = threading.Event()
        self.result: Optional[List[Any]] = None
        self.exception: Optional[BaseException] = None

    def set_result(self, result: List[Any]) -> None:
        self.result = result
        self.event.set()

    def set_exception(self, exc: BaseException) -> None:
        self.exception = exc
        self.event.set()


class _Shutdown:
    """Sentinel enqueued to stop the scorer thread."""


class NliMicrobatchDispatcher:
    """Coalesces concurrent NLI ``score_batch`` calls onto one scorer thread.

    A single daemon thread owns the underlying :class:`NliClassifier` (or any
    object exposing ``score_batch(pairs=...) -> [NliScore]``). Callers submit via
    :meth:`score_batch`; the scorer drains + coalesces + scores + routes each
    caller its own slice.
    """

    def __init__(
        self,
        nli: Any,
        *,
        max_pairs: Optional[int] = None,
        window_ms: Optional[float] = None,
    ) -> None:
        self._nli = nli
        self._max_pairs = (
            max_pairs if (max_pairs and max_pairs > 0) else resolve_microbatch_max_pairs()
        )
        self._window_s = (
            (window_ms if (window_ms is not None and window_ms >= 0)
             else resolve_microbatch_window_ms()) / 1000.0
        )
        self._supports_batch_size = _accepts_batch_size(getattr(nli, "score_batch", None))
        self._queue: "queue.Queue[Any]" = queue.Queue()
        #: Observability: number of requests coalesced per drain (thread-safe
        #: append from the single scorer thread; read after join in tests).
        self.drain_request_counts: List[int] = []
        self.drain_pair_counts: List[int] = []
        #: Periodic throughput stats (INFO log every ~30s from the scorer
        #: thread) so a long silent NLI region is diagnosable live.
        self._stats_last_ts = time.monotonic()
        self._stats_drains = 0
        self._stats_pairs = 0
        self._stats_tokens = 0
        self._thread = threading.Thread(
            target=self._run,
            name="nli-microbatch-scorer",
            daemon=True,
        )
        self._thread.start()

    # -- drop-in NLI surface -------------------------------------------------- #

    @property
    def device(self) -> Optional[str]:
        return getattr(self._nli, "device", None)

    @property
    def dtype(self) -> Optional[str]:
        return getattr(self._nli, "dtype", None)

    @property
    def _revision(self) -> Optional[str]:
        return getattr(self._nli, "_revision", None)

    def score_pair(self, *, premise: str, hypothesis: str) -> Any:
        return self.score_batch(pairs=[(premise, hypothesis)])[0]

    def score_batch(
        self,
        *,
        pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,  # accepted for drop-in parity; ignored
    ) -> List[Any]:
        """Enqueue ``pairs``, block until the scorer resolves this request.

        Empty input is a no-op that returns ``[]`` without touching the queue
        (mirrors :meth:`NliClassifier.score_batch`). A scorer-side exception is
        re-raised here so the caller never sees fabricated scores.
        """
        if not pairs:
            return []
        req = _Request(list(pairs))
        self._queue.put(req)
        req.event.wait()
        if req.exception is not None:
            raise req.exception
        return req.result if req.result is not None else []

    # -- scorer thread -------------------------------------------------------- #

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if isinstance(first, _Shutdown):
                return
            batch: List[_Request] = [first]
            total = len(first.pairs)
            # Coalesce further requests until the pair cap is hit OR the
            # collection window elapses. window_s == 0 → no wait, take only what
            # is already queued.
            deadline = time.monotonic() + self._window_s
            while total < self._max_pairs:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    nxt = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if isinstance(nxt, _Shutdown):
                    # Drain-and-die: score what we have, then stop. Re-enqueue
                    # the sentinel so any collected-but-unscored siblings after
                    # this loop still get processed, then exit next iteration.
                    self._drain_batch(batch)
                    return
                batch.append(nxt)
                total += len(nxt.pairs)
            self._drain_batch(batch)

    def _score_combined(self, combined: List[Tuple[str, str]]) -> List[Any]:
        """Score ``combined`` in ONE forward pass, halving the slice on CUDA OOM.

        Risk #3 (VRAM OOM at a large ``MAX_PAIRS``): the W2.1 in-gate OOM handler
        would otherwise convert a validator OOM into a warning auto-pass (silent
        degrade). The dispatcher instead catches the CUDA OOM here, halves the
        forward-pass slice size, and re-scores in sub-slices so the gate keeps
        REAL scores. Only re-raises when even a single-pair slice OOMs (a genuine
        hardware limit the caller must see, not a vacuous pass).
        """
        if not combined:
            return []
        # First attempt: the original single-call shape (byte-identical to the
        # pre-OOM-guard path — one score_batch per drain). Only a CUDA OOM drops
        # into the halving sub-slice loop.
        try:
            if self._supports_batch_size:
                return self._nli.score_batch(pairs=combined, batch_size=self._max_pairs)
            return self._nli.score_batch(pairs=combined)
        except BaseException as exc:  # noqa: BLE001
            if not (_is_cuda_oom(exc) and len(combined) > 1):
                raise
            _empty_cuda_cache()
        slice_size = max(1, (self._max_pairs if self._max_pairs > 0 else len(combined)) // 2)
        while True:
            try:
                scores: List[Any] = []
                for start in range(0, len(combined), slice_size):
                    sub = combined[start : start + slice_size]
                    if self._supports_batch_size:
                        part = self._nli.score_batch(pairs=sub, batch_size=slice_size)
                    else:
                        part = self._nli.score_batch(pairs=sub)
                    scores.extend(part)
                return scores
            except BaseException as exc:  # noqa: BLE001
                if _is_cuda_oom(exc) and slice_size > 1:
                    _empty_cuda_cache()
                    slice_size = max(1, slice_size // 2)
                    logger.warning(
                        "NLI micro-batch drain hit CUDA OOM; halving forward-pass "
                        "slice to %d pairs and retrying.", slice_size,
                    )
                    continue
                raise

    def _drain_batch(self, batch: List[_Request]) -> None:
        """Run ONE combined forward pass over ``batch`` and route each slice."""
        combined: List[Tuple[str, str]] = []
        for req in batch:
            combined.extend(req.pairs)
        self.drain_request_counts.append(len(batch))
        self.drain_pair_counts.append(len(combined))
        self._stats_drains += 1
        self._stats_pairs += len(combined)
        # Token-length estimate (tokenizer-free, capped at the 512 truncation)
        # so throughput is reported in tokens as well as pairs — a drain of long
        # ~512-token pairs moves far fewer pairs/s than a short-pair drain, and
        # tok/s is the length-invariant signal for spotting the slow long-pair
        # regions the bucketing knob targets.
        self._stats_tokens += sum(
            estimate_pair_tokens(p, h) for p, h in combined
        )
        _now = time.monotonic()
        _elapsed = _now - self._stats_last_ts
        if _elapsed >= 30.0:
            logger.info(
                "NLI microbatch: %d drains / %d pairs / %d tok~ in %.0fs "
                "(%.0f pairs/s, %.0f tok/s, mean fill %.1f pairs/drain, cap %d)",
                self._stats_drains, self._stats_pairs, self._stats_tokens,
                _elapsed,
                self._stats_pairs / _elapsed,
                self._stats_tokens / _elapsed,
                self._stats_pairs / max(1, self._stats_drains),
                self._max_pairs,
            )
            self._stats_last_ts = _now
            self._stats_drains = 0
            self._stats_pairs = 0
            self._stats_tokens = 0
        try:
            scores = self._score_combined(combined)
            if len(scores) != len(combined):
                raise RuntimeError(
                    "NLI micro-batch scorer returned "
                    f"{len(scores)} scores for {len(combined)} pairs; "
                    "refusing to route a length-mismatched result."
                )
        except BaseException as exc:  # noqa: BLE001 — surface to every caller
            for req in batch:
                req.set_exception(exc)
            return
        offset = 0
        for req in batch:
            n = len(req.pairs)
            req.set_result(list(scores[offset : offset + n]))
            offset += n

    def shutdown(self) -> None:
        """Stop the scorer thread (best-effort; used by tests)."""
        self._queue.put(_Shutdown())
        self._thread.join(timeout=5.0)


# --------------------------------------------------------------------------- #
# Process-level dispatcher registry (keyed by the underlying nli's identity).
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[int, NliMicrobatchDispatcher] = {}
#: Strong refs so an ``id()`` key can never be reused by a GC'd underlying nli.
_REGISTRY_REFS: Dict[int, Any] = {}
_REGISTRY_LOCK = threading.Lock()


def get_dispatcher(nli: Any) -> Optional[NliMicrobatchDispatcher]:
    """Return the process-shared dispatcher for ``nli`` (constructing once).

    ``nli`` is the resolved NLI classifier (the process singleton in production,
    an injected fake in tests). Returns ``None`` when ``nli`` is falsy (no model
    available → the caller degrades exactly as it would without the dispatcher).
    Keyed on ``id(nli)`` so distinct classifiers get distinct scorer threads; a
    strong reference is held so the key can never collide.
    """
    if nli is None:
        return None
    key = id(nli)
    with _REGISTRY_LOCK:
        existing = _REGISTRY.get(key)
        if existing is not None:
            return existing
        dispatcher = NliMicrobatchDispatcher(nli)
        _REGISTRY[key] = dispatcher
        _REGISTRY_REFS[key] = nli
        return dispatcher


def _reset_for_tests() -> None:
    """Test-only: shut down + clear the dispatcher registry."""
    with _REGISTRY_LOCK:
        for dispatcher in _REGISTRY.values():
            try:
                dispatcher.shutdown()
            except Exception:  # noqa: BLE001 — best-effort teardown
                pass
        _REGISTRY.clear()
        _REGISTRY_REFS.clear()


__all__ = [
    "ENV_MICROBATCH",
    "ENV_MICROBATCH_MAX_PAIRS",
    "ENV_MICROBATCH_WINDOW_MS",
    "ENV_CROSSBLOCK",
    "ENV_CROSSBLOCK_THREADS",
    "NliMicrobatchDispatcher",
    "get_dispatcher",
    "resolve_microbatch_enabled",
    "resolve_microbatch_max_pairs",
    "resolve_microbatch_window_ms",
    "resolve_crossblock_enabled",
    "resolve_crossblock_threads",
]
