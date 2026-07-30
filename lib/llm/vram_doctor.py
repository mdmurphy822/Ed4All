"""VRAM-contention doctor — best-effort GPU-memory observability.

Shared foundation for the "VRAM doctor" feature: a passive, NEVER-raising
observability layer that snapshots the GPU's true free/total VRAM, lists
the local ollama models actually resident on the card, and predicts
whether each in-process GPU consumer (the NLI classifier and the
embedding provider) will FIT, EVICT-then-fit, fall back to CPU, or OOM.

This module owns ZERO new device policy — it reuses the existing
primitives and only *reports* what they would decide:

* :func:`lib.classifiers.nli_classifier.probe_free_vram_mib` — the
  NVML-first (WSL2-correct) global-free-VRAM probe.
* :func:`lib.classifiers.nli_classifier.resolve_nli_device`,
  :func:`resolve_min_free_vram_mib`, :func:`resolve_evict_for_cuda` —
  the NLI device / floor / eviction config resolvers.
* :func:`lib.llm.vram_reclaim.resolve_ollama_root` — the ``/v1``-stripped
  ollama native root; the resident-model list comes from
  ``GET {root}/api/ps`` (``size_vram`` bytes per model).

Design contract (the whole reason this module exists): a doctor that
crashes the run is worse than no doctor. Every public function is
best-effort — externals are wrapped in ``try/except``, failures log via
the module logger and return a degraded value, and ``torch`` / ``httpx``
are imported lazily so the module imports cleanly on a box without them.

The ``ED4ALL_VRAM_DOCTOR`` flag (default OFF, parse-with-fallback, mirrors
``ED4ALL_CLOUD_RATE_LIMIT`` / the NLI knobs) gates whether a consumer runs
the doctor at all; the functions here do not read it themselves (a caller
checks :func:`vram_doctor_enabled` before invoking).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from lib.classifiers.nli_classifier import (
    probe_free_vram_mib_with_source,
    resolve_evict_for_cuda,
    resolve_min_free_vram_mib,
    resolve_nli_device,
)
from lib.llm.vram_reclaim import _fetch_resident_models, resolve_ollama_root

logger = logging.getLogger(__name__)

#: Master switch for the VRAM-contention doctor. Default OFF — a default
#: run never snapshots VRAM (mirrors ``ED4ALL_CLOUD_RATE_LIMIT`` /
#: ``LOCAL_DISPATCHER_ALLOW_STUB``). Truthy tokens: ``1`` / ``true`` /
#: ``yes`` / ``on`` (case-insensitive). Documented in root CLAUDE.md.
ENV_VRAM_DOCTOR = "ED4ALL_VRAM_DOCTOR"

#: Env var selecting the embedding provider's torch device. Read directly
#: here (rather than importing ``lib.embedding.providers``) to avoid
#: pulling the heavy sentence-transformers import chain into the doctor.
#: Mirrors ``lib.embedding.providers.ENV_DEVICE``.
ENV_EMBEDDING_DEVICE = "ED4ALL_EMBEDDING_DEVICE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Per-consumer VRAM need estimates (MiB).
#:
#: NLI: fp16 DeBERTa-v3-base weights ~400 MiB + the batch-8 × 512-token
#: forward-pass activation spike ~500 MiB. The fit-check floors the
#: NLI-config ``resolve_min_free_vram_mib`` against this so a misconfigured
#: floor of 0 doesn't make the doctor report "fits" on a near-full card.
_NLI_FORWARD_SPIKE_FLOOR_MIB = 900

#: Embedding (sentence-transformers bge-* class) resident footprint, MiB.
_EMBEDDING_NEED_MIB = 500

#: Per-request HTTP timeout (seconds) for the doctor's ``/api/ps`` probe.
#: SHORT on purpose — the doctor runs at preflight + every phase boundary, so
#: it must NOT inherit the evictor's 30s reclaim timeout and stall the run on
#: a wedged ollama server. A failed/slow probe just degrades to ``[]``.
_DOCTOR_PS_TIMEOUT_SECONDS = 2.0


def vram_doctor_enabled(value: Optional[str] = None) -> bool:
    """True iff the VRAM doctor is enabled (``ED4ALL_VRAM_DOCTOR``).

    Resolution chain: explicit ``value`` arg → ``ED4ALL_VRAM_DOCTOR`` env →
    default OFF. Parse-with-fallback (mirrors
    :func:`lib.llm.rate_limiter.is_rate_limit_enabled`): only the truthy
    tokens (``1`` / ``true`` / ``yes`` / ``on``, case-insensitive) enable;
    anything else — including garbage / unset — is OFF.
    """
    raw = value if value is not None else os.environ.get(ENV_VRAM_DOCTOR, "")
    return str(raw).strip().lower() in _TRUTHY


@dataclass
class VramSnapshot:
    """A best-effort point-in-time read of GPU VRAM + resident ollama models.

    All numeric fields are ``Optional`` because the snapshot NEVER raises —
    a probe failure (no torch, no CUDA, NVML missing) degrades to ``None``
    rather than crashing the caller.
    """

    free_mib: Optional[int]
    total_mib: Optional[int]
    cuda_available: bool
    probe_source: str  # "nvml" | "torch" | "unavailable"
    resident_models: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class FitVerdict:
    """Predicted placement outcome for one in-process GPU consumer."""

    consumer: str  # "nli" | "embedding"
    device_requested: str  # "cuda" / "cuda:N" / "cpu"
    need_mib: int
    free_mib: Optional[int]
    outcome: str  # see the per-outcome enum in fit_check's docstring
    detail: str


def _import_torch() -> Any:
    """Lazy-import torch; return the module or ``None`` (never raises)."""
    try:
        import torch  # type: ignore

        return torch
    except Exception as exc:  # noqa: BLE001 — torch absent / broken build
        logger.debug("vram_doctor: torch unavailable (%s).", exc)
        return None


def _probe_total_mib(torch_module: Any, device: str) -> Optional[int]:
    """Total VRAM (MiB) on ``device`` via ``torch.cuda.mem_get_info``.

    ``mem_get_info`` returns ``(free, total)`` in bytes; only the total is
    read here (it is correct even on WSL2, unlike the free leg). Best-effort:
    any failure returns ``None``.
    """
    try:
        dev_arg = None if device == "cuda" else device
        _free_bytes, total_bytes = torch_module.cuda.mem_get_info(dev_arg)
        return int(total_bytes) // (1024 * 1024)
    except Exception as exc:  # noqa: BLE001
        logger.debug("vram_doctor: total-VRAM probe failed on %r: %s", device, exc)
        return None


def _query_resident_models(base_url: Optional[str] = None) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """List ollama-resident models + their VRAM via ``GET {root}/api/ps``.

    Returns ``(models, error)`` where ``models`` is a list of
    ``{"name": str, "vram_mib": int}`` (``size_vram`` bytes → MiB) and
    ``error`` is a short string when the probe failed (ollama down, httpx
    missing, bad JSON). Best-effort — never raises; on failure returns
    ``([], "<reason>")``.
    """
    try:
        import httpx  # type: ignore
    except Exception as exc:  # noqa: BLE001 — httpx absent / broken
        logger.debug("vram_doctor: httpx unavailable (%s).", exc)
        return [], f"httpx unavailable: {exc}"

    root = resolve_ollama_root(base_url)
    if not root:
        return [], None
    try:
        # SHORT timeout — the doctor is latency-sensitive (preflight +
        # per-phase); it must NOT inherit the evictor's 30s reclaim timeout.
        with httpx.Client(timeout=_DOCTOR_PS_TIMEOUT_SECONDS) as client:
            entries = _fetch_resident_models(client, root)
    except Exception as exc:  # noqa: BLE001 — server down / bad JSON / network
        logger.debug("vram_doctor: /api/ps probe failed at %s: %s", root, exc)
        return [], f"ollama /api/ps unreachable: {exc}"

    resident: List[Dict[str, Any]] = []
    for entry in entries:
        size_vram = entry.get("size_vram")
        try:
            vram_mib = int(size_vram) // (1024 * 1024) if size_vram else 0
        except (TypeError, ValueError):
            vram_mib = 0
        resident.append({"name": entry["name"], "vram_mib": vram_mib})
    return resident, None


def snapshot_vram(base_url: Optional[str] = None) -> VramSnapshot:
    """Best-effort point-in-time VRAM + resident-model snapshot (NEVER raises).

    Lazy-imports torch (absent → ``cuda_available=False``). When CUDA is
    available, free VRAM comes from
    :func:`lib.classifiers.nli_classifier.probe_free_vram_mib` (NVML-first,
    WSL2-correct) and total from ``torch.cuda.mem_get_info``. The
    resident-model list comes from ``GET {root}/api/ps`` regardless of CUDA
    state (an ollama-down probe just yields ``[]`` + an ``error`` string).
    """
    resident, ollama_error = _query_resident_models(base_url)

    torch_module = _import_torch()
    if torch_module is None:
        return VramSnapshot(
            free_mib=None,
            total_mib=None,
            cuda_available=False,
            probe_source="unavailable",
            resident_models=resident,
            error=ollama_error,
        )

    try:
        cuda_available = bool(torch_module.cuda.is_available())
    except Exception as exc:  # noqa: BLE001 — torch probe is best-effort
        logger.debug("vram_doctor: torch.cuda.is_available() raised: %s", exc)
        cuda_available = False

    if not cuda_available:
        return VramSnapshot(
            free_mib=None,
            total_mib=None,
            cuda_available=False,
            probe_source="unavailable",
            resident_models=resident,
            error=ollama_error,
        )

    # Probe the physical card (index 0 / current device). The NLI/embedding
    # device *config* is a fit-check concern; the snapshot reports the card.
    # The PRIMITIVE reports which backend produced the value (nvml vs the
    # torch fallback) so the stamped ``probe_source`` can never mislabel a
    # torch-derived (cross-process-blind) number as ``"nvml"``.
    device = "cuda"
    free_mib, probe_source = probe_free_vram_mib_with_source(torch_module, device)
    total_mib = _probe_total_mib(torch_module, device)
    if free_mib is None:
        # Invariant: no value → unavailable (defends against a future probe
        # returning a non-"unavailable" tag alongside a None value).
        probe_source = "unavailable"

    return VramSnapshot(
        free_mib=free_mib,
        total_mib=total_mib,
        cuda_available=True,
        probe_source=probe_source,
        resident_models=resident,
        error=ollama_error,
    )


def _build_verdict(
    consumer: str,
    device_requested: str,
    need_mib: int,
    snapshot: VramSnapshot,
    allow_evict: bool,
) -> FitVerdict:
    """Shared placement-outcome logic for both in-process GPU consumers.

    Encodes the branch ladder common to the NLI and embedding fit-checks
    (cpu_requested → cuda-unavailable → probe-unknown → fits → near-full)
    once. The two consumers differ only in:

    * the consumer-specific detail strings (NLI cites ``ED4ALL_NLI_DEVICE``
      and loads ``(fp16)``; embedding cites ``ED4ALL_EMBEDDING_DEVICE``);
    * ``need_mib`` (resolved by the caller); and
    * the near-full tail: NLI (``allow_evict=True``) consults
      :func:`resolve_evict_for_cuda` + the resident models for the
      evict-then-fit / fallback / OOM split; embedding (``allow_evict=False``)
      has no eviction seam → straight to ``would_oom``.

    The outcomes + detail text are byte-identical to the prior pair of
    hand-written functions (the existing fit-check tests are the guard).
    """
    device = device_requested
    free_mib = snapshot.free_mib

    if device == "cpu":
        detail = (
            "ED4ALL_NLI_DEVICE=cpu — NLI scores on CPU; no VRAM needed."
            if consumer == "nli"
            else "ED4ALL_EMBEDDING_DEVICE=cpu — embeddings run on CPU; no VRAM needed."
        )
        return FitVerdict(
            consumer=consumer,
            device_requested=device,
            need_mib=need_mib,
            free_mib=free_mib,
            outcome="cpu_requested",
            detail=detail,
        )
    if not snapshot.cuda_available:
        return FitVerdict(
            consumer=consumer,
            device_requested=device,
            need_mib=need_mib,
            free_mib=free_mib,
            outcome="would_fallback_cpu",
            detail=f"cuda requested ({device}) but CUDA unavailable → CPU fallback.",
        )
    if free_mib is None:
        return FitVerdict(
            consumer=consumer,
            device_requested=device,
            need_mib=need_mib,
            free_mib=None,
            outcome="unknown",
            detail="free-VRAM probe failed → cannot predict placement.",
        )
    if free_mib >= need_mib:
        suffix = " (fp16)" if consumer == "nli" else ""
        return FitVerdict(
            consumer=consumer,
            device_requested=device,
            need_mib=need_mib,
            free_mib=free_mib,
            outcome="fits",
            detail=f"{free_mib} MiB free ≥ {need_mib} MiB need → loads on {device}{suffix}.",
        )

    # free < need. The embedding provider has no evict-then-retry seam → OOM.
    if not allow_evict:
        return FitVerdict(
            consumer=consumer,
            device_requested=device,
            need_mib=need_mib,
            free_mib=free_mib,
            outcome="would_oom",
            detail=(
                f"{free_mib} MiB free < {need_mib} MiB need (no eviction path for the "
                "embedding provider) → CUDA OOM risk."
            ),
        )

    # NLI eviction tail.
    evict = resolve_evict_for_cuda()
    reclaimable = sum(int(m.get("vram_mib", 0) or 0) for m in snapshot.resident_models)
    if evict and snapshot.resident_models and (free_mib + reclaimable) >= need_mib:
        return FitVerdict(
            consumer=consumer,
            device_requested=device,
            need_mib=need_mib,
            free_mib=free_mib,
            outcome="would_evict_then_fit",
            detail=(
                f"{free_mib} MiB free < {need_mib} MiB; evicting {len(snapshot.resident_models)} "
                f"resident model(s) frees ~{reclaimable} MiB → fits on {device}."
            ),
        )
    if evict:
        return FitVerdict(
            consumer=consumer,
            device_requested=device,
            need_mib=need_mib,
            free_mib=free_mib,
            outcome="would_fallback_cpu",
            detail=(
                f"{free_mib} MiB free + ~{reclaimable} MiB reclaimable < {need_mib} MiB need "
                "→ eviction can't free enough → CPU fallback (fp32)."
            ),
        )
    # Eviction OFF + free < need → the "silent death" config: the floor is
    # disabled or eviction is off and the forward pass would OOM on load.
    return FitVerdict(
        consumer=consumer,
        device_requested=device,
        need_mib=need_mib,
        free_mib=free_mib,
        outcome="would_oom",
        detail=(
            f"{free_mib} MiB free < {need_mib} MiB need and eviction is OFF "
            f"(ED4ALL_NLI_EVICT_FOR_CUDA) → forward-pass CUDA OOM risk."
        ),
    )


def _fit_check_nli(snapshot: VramSnapshot) -> FitVerdict:
    """Predict the NLI classifier's placement outcome from a snapshot."""
    device = resolve_nli_device()
    need_mib = max(resolve_min_free_vram_mib(), _NLI_FORWARD_SPIKE_FLOOR_MIB)
    return _build_verdict("nli", device, need_mib, snapshot, allow_evict=True)


def _fit_check_embedding(snapshot: VramSnapshot) -> FitVerdict:
    """Predict the embedding provider's placement outcome from a snapshot.

    Same shape as the NLI check minus the eviction path — the embedding
    provider has no evict-then-retry seam, so a near-full card is reported
    as ``would_oom`` rather than ``would_evict_then_fit``.
    """
    device = (os.environ.get(ENV_EMBEDDING_DEVICE) or "cpu").strip() or "cpu"
    need_mib = _EMBEDDING_NEED_MIB
    return _build_verdict("embedding", device, need_mib, snapshot, allow_evict=False)


def fit_check(snapshot: VramSnapshot) -> List[FitVerdict]:
    """Predict placement for every in-process GPU consumer (NEVER raises).

    Returns one :class:`FitVerdict` per consumer (``nli`` then ``embedding``).
    Outcomes:

    * ``cpu_requested`` — the consumer's device is CPU; no VRAM needed.
    * ``fits`` — free VRAM ≥ the consumer's need; loads on GPU.
    * ``would_evict_then_fit`` — (NLI only) free < need but evicting the
      resident ollama model(s) frees enough; eviction is enabled.
    * ``would_fallback_cpu`` — CUDA unavailable, or (NLI) eviction enabled
      but can't free enough → the consumer degrades to CPU.
    * ``would_oom`` — free < need with no remedy (NLI: floor/eviction off;
      embedding: no evict path) → forward-pass CUDA OOM risk.
    * ``unknown`` — the free-VRAM probe failed; placement can't be predicted.
    """
    verdicts: List[FitVerdict] = []
    for builder in (_fit_check_nli, _fit_check_embedding):
        try:
            verdicts.append(builder(snapshot))
        except Exception as exc:  # noqa: BLE001 — best-effort observability
            logger.warning("vram_doctor: fit-check %s raised: %s", builder.__name__, exc)
    return verdicts


def write_trajectory_row(
    run_id: str,
    phase: str,
    when: str,
    snapshot: VramSnapshot,
    event: str = "phase_boundary",
    extra: Optional[dict] = None,
) -> None:
    """Append one JSONL row to ``runtime/state/runs/<run_id>/vram_trajectory.jsonl``.

    Row shape: ``{run_id, phase, when, ts, event, free_mib, total_mib,
    probe_source, resident_models, cuda_available, **extra}``. ``when`` is the
    boundary LABEL (``before`` / ``after``); ``ts`` is the wall-clock UTC
    ISO-8601 instant the row was written, so residency-hours can be joined
    across rows later. The parent directory is created on first use.
    Best-effort: ANY failure (bad run_id, unwritable path, serialization error)
    is logged and swallowed — a trajectory write must NEVER crash the run it
    observes.
    """
    try:
        from datetime import datetime, timezone
        from lib.paths import get_state_runs_dir

        runs_dir = get_state_runs_dir()
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        target = run_dir / "vram_trajectory.jsonl"

        row: Dict[str, Any] = {
            "run_id": run_id,
            "phase": phase,
            "when": when,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "free_mib": snapshot.free_mib,
            "total_mib": snapshot.total_mib,
            "probe_source": snapshot.probe_source,
            "resident_models": snapshot.resident_models,
            "cuda_available": snapshot.cuda_available,
        }
        if extra:
            row.update(extra)

        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:  # noqa: BLE001 — observability must never crash the run
        logger.warning(
            "vram_doctor: failed to append trajectory row for run %r phase %r: %s",
            run_id, phase, exc,
        )


_OUTCOME_MARKERS = {
    "fits": "✓",
    "cpu_requested": "✓",
    "would_evict_then_fit": "✓",
    "would_fallback_cpu": "⚠",
    "unknown": "⚠",
    "would_oom": "✗",
}


def format_doctor_report(snapshot: VramSnapshot, verdicts: List[FitVerdict]) -> str:
    """Render a human-readable multi-line doctor report for the CLI.

    A free/total/source header, the resident-model list, then one marked
    (``✓`` / ``⚠`` / ``✗``) verdict line per consumer. Best-effort — never
    raises; a malformed verdict degrades to a plain line.
    """
    lines: List[str] = ["VRAM doctor report"]

    free = "unknown" if snapshot.free_mib is None else f"{snapshot.free_mib} MiB"
    total = "unknown" if snapshot.total_mib is None else f"{snapshot.total_mib} MiB"
    lines.append(
        f"  GPU: free {free} / total {total} "
        f"(source: {snapshot.probe_source}, cuda_available={snapshot.cuda_available})"
    )
    if snapshot.error:
        lines.append(f"  ollama probe: {snapshot.error}")

    if snapshot.resident_models:
        lines.append("  resident ollama models:")
        for model in snapshot.resident_models:
            try:
                lines.append(
                    f"    - {model.get('name', '?')} (~{model.get('vram_mib', 0)} MiB VRAM)"
                )
            except Exception:  # noqa: BLE001
                lines.append(f"    - {model!r}")
    else:
        lines.append("  resident ollama models: none")

    lines.append("  fit checks:")
    for verdict in verdicts:
        try:
            marker = _OUTCOME_MARKERS.get(verdict.outcome, "?")
            lines.append(
                f"    {marker} {verdict.consumer} "
                f"(device={verdict.device_requested}, need={verdict.need_mib} MiB): "
                f"{verdict.outcome} — {verdict.detail}"
            )
        except Exception:  # noqa: BLE001
            lines.append(f"    ? {verdict!r}")

    return "\n".join(lines)


__all__ = [
    "ENV_VRAM_DOCTOR",
    "ENV_EMBEDDING_DEVICE",
    "VramSnapshot",
    "FitVerdict",
    "vram_doctor_enabled",
    "snapshot_vram",
    "fit_check",
    "write_trajectory_row",
    "format_doctor_report",
]
