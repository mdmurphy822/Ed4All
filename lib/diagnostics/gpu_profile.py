"""Big-memory GPU profile checks — the ``gpu_profile`` doctor group.

The pipeline's GPU-contention defaults are tuned for a small (8GB-class)
dev box where exactly ONE model can be resident at a time: the GPU
lifecycle sweep cold-evicts the resident model at every phase boundary,
the NLI classifier evicts the generation model when free VRAM is low, and
the ``gpu_guard`` wrapper refuses to start while a model holds the card.
Every one of those defaults actively FIGHTS a big-memory (unified-memory,
tens-to-hundreds of GiB) endpoint configured for concurrent serving, where
a large model is meant to stay resident and serve many requests at once.

This check group is a purely ADVISORY preflight: when it detects a
big-memory GPU it WARNs (never FAILs) for each default that would sabotage
a concurrent-serving endpoint, naming the flag, why it fights, and the
one-line remediation. On the small dev box (or a box with no probeable
GPU) it is a silent no-op — it emits ZERO results, so the doctor output is
byte-identical to before this group existed.

Design contract (inherited from :mod:`lib.diagnostics.core`): a doctor
that crashes the run is worse than no doctor. ``gpu_profile_checks`` NEVER
raises and NEVER errors — any probe/resolver failure degrades to "not
applicable" (an empty result list), never to a WARN or an exception. The
whole check is deterministic and makes NO LLM calls.

VRAM detection REUSES the existing never-raising probe
:func:`lib.llm.vram_doctor.snapshot_vram` (the ``ED4ALL_VRAM_DOCTOR`` work)
rather than writing a new one — its ``total_mib`` field is the total VRAM
of the current CUDA device, resolved via ``torch.cuda.mem_get_info`` /
NVML. A ``None`` total (no torch / no CUDA / probe unavailable) is treated
as "not applicable", so the group skips cleanly on any box where the GPU
cannot be probed.

The per-flag states are read through their CANONICAL resolvers — the GPU
lifecycle mode via :func:`lib.gpu_lifecycle.resolve_gpu_lifecycle_mode`,
the NLI eviction strategy via
:func:`lib.classifiers.nli_classifier.resolve_evict_for_cuda`, and the
cloud admission gate via :func:`lib.llm.rate_limiter.is_rate_limit_enabled`
— never by re-parsing the env here (so the doctor and the runtime can
never disagree on what a flag resolves to).

Registration is explicit (:func:`register_gpu_profile_checks`), called by
the CLI bootstrap — NEVER at import time.
"""

from __future__ import annotations

import logging
import os
from typing import List

from lib.diagnostics.core import CheckContext, CheckResult, Severity, register

logger = logging.getLogger(__name__)

#: Threshold env: a GPU whose total VRAM is at or above this many MiB is
#: treated as "big memory" (a concurrent-serving endpoint host). Default
#: 49152 MiB (48 GiB) — comfortably above every consumer card, below the
#: unified-memory serving boxes this profile targets.
ENV_BIG_MEMORY_MIN_MIB = "ED4ALL_BIG_MEMORY_MIN_MIB"
_DEFAULT_BIG_MEMORY_MIN_MIB = 49152

#: The gpu_guard busy-ceiling env (read by ``scripts/gpu_guard.sh``, default
#: 1500 MiB there). There is no Python resolver for it — the wrapper is a
#: shell script — so this group reads the raw env directly and only warns
#: when it is actually set to a value below half the detected total VRAM.
ENV_GPU_MAX_USED_MB = "ED4ALL_GPU_MAX_USED_MB"

_GROUP = "gpu_profile"


def resolve_big_memory_min_mib(value: object = None) -> int:
    """Resolve the big-memory VRAM threshold (MiB).

    Resolution chain: explicit ``value`` arg → ``ED4ALL_BIG_MEMORY_MIN_MIB``
    env → the 49152 MiB default. Parse-with-fallback: garbage / non-positive
    / non-integer → the default (mirrors the house resolver convention).
    """
    raw = value if value is not None else os.environ.get(ENV_BIG_MEMORY_MIN_MIB)
    if raw is None or not str(raw).strip():
        return _DEFAULT_BIG_MEMORY_MIN_MIB
    try:
        parsed = int(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_BIG_MEMORY_MIN_MIB
    return parsed if parsed > 0 else _DEFAULT_BIG_MEMORY_MIN_MIB


def _probe_total_mib(base_url) -> object:
    """Return the current CUDA device's total VRAM (MiB) or ``None``.

    REUSES the canonical never-raising :func:`snapshot_vram` probe. Any
    failure (import error / degraded snapshot) degrades to ``None`` →
    "not applicable" (the group skips silently). NEVER raises.
    """
    try:
        from lib.llm.vram_doctor import snapshot_vram

        snapshot = snapshot_vram(base_url)
    except Exception as exc:  # noqa: BLE001 — probe must never crash the doctor
        logger.debug("gpu_profile: total-VRAM probe failed: %s", exc)
        return None
    return getattr(snapshot, "total_mib", None)


def _lifecycle_warning() -> List[CheckResult]:
    """WARN when the GPU-lifecycle phase-boundary sweep is enabled."""
    try:
        from lib.gpu_lifecycle import resolve_gpu_lifecycle_mode

        enabled = resolve_gpu_lifecycle_mode()
    except Exception as exc:  # noqa: BLE001
        logger.debug("gpu_profile: lifecycle resolver failed: %s", exc)
        return []
    if not enabled:
        return []
    return [
        CheckResult(
            name="gpu_profile_lifecycle_sweep",
            group=_GROUP,
            severity=Severity.WARN,
            summary=(
                "ED4ALL_GPU_LIFECYCLE is on — the per-phase-boundary sweep "
                "cold-evicts the resident model (keep_alive:0) at every stage "
                "seam, defeating an always-resident big-memory endpoint"
            ),
            remediation="set ED4ALL_GPU_LIFECYCLE=0 for a big-memory serving endpoint",
            data={"flag": "ED4ALL_GPU_LIFECYCLE", "resolved": True},
        )
    ]


def _nli_evict_warning() -> List[CheckResult]:
    """WARN when NLI-evict-for-CUDA is enabled."""
    try:
        from lib.classifiers.nli_classifier import resolve_evict_for_cuda

        enabled = resolve_evict_for_cuda()
    except Exception as exc:  # noqa: BLE001
        logger.debug("gpu_profile: nli-evict resolver failed: %s", exc)
        return []
    if not enabled:
        return []
    return [
        CheckResult(
            name="gpu_profile_nli_evict",
            group=_GROUP,
            severity=Severity.WARN,
            summary=(
                "ED4ALL_NLI_EVICT_FOR_CUDA is on — the NLI classifier evicts "
                "the resident generation model to free VRAM, which is wrong on "
                "a big-memory box where generation and NLI can co-reside"
            ),
            remediation="set ED4ALL_NLI_EVICT_FOR_CUDA=0 for a big-memory serving endpoint",
            data={"flag": "ED4ALL_NLI_EVICT_FOR_CUDA", "resolved": True},
        )
    ]


def _rate_limit_warning() -> List[CheckResult]:
    """WARN when the process-global cloud admission gate is enabled."""
    try:
        from lib.llm.rate_limiter import is_rate_limit_enabled

        enabled = is_rate_limit_enabled()
    except Exception as exc:  # noqa: BLE001
        logger.debug("gpu_profile: rate-limit resolver failed: %s", exc)
        return []
    if not enabled:
        return []
    return [
        CheckResult(
            name="gpu_profile_cloud_rate_limit",
            group=_GROUP,
            severity=Severity.WARN,
            summary=(
                "ED4ALL_CLOUD_RATE_LIMIT is on — the admission gate is "
                "process-global (it wraps every OpenAI-compatible seat), so it "
                "throttles the LOCAL endpoint too, not just cloud seats"
            ),
            remediation=(
                "leave ED4ALL_CLOUD_RATE_LIMIT unset for a self-hosted endpoint "
                "(there is no cloud rate cap to respect)"
            ),
            data={"flag": "ED4ALL_CLOUD_RATE_LIMIT", "resolved": True},
        )
    ]


def _gpu_guard_warning(total_mib: int) -> List[CheckResult]:
    """WARN when ED4ALL_GPU_MAX_USED_MB is set below half the total VRAM.

    Only fires when the env is ACTUALLY set to a valid positive int below
    half of the detected total VRAM — a value that would keep the
    ``gpu_guard`` wrapper blocked while a large model is resident. Unset /
    blank / garbage / non-positive → no warning (the wrapper is not in play,
    or the value is not evaluable).
    """
    raw = os.environ.get(ENV_GPU_MAX_USED_MB)
    if raw is None or not str(raw).strip():
        return []
    try:
        ceiling = int(str(raw).strip())
    except (TypeError, ValueError):
        return []
    if ceiling <= 0:
        return []
    half_total = total_mib // 2
    if ceiling >= half_total:
        return []
    return [
        CheckResult(
            name="gpu_profile_gpu_guard",
            group=_GROUP,
            severity=Severity.WARN,
            summary=(
                f"ED4ALL_GPU_MAX_USED_MB={ceiling} is far below the detected "
                f"{total_mib} MiB total VRAM — the gpu_guard used-VRAM gate "
                "would block forever while a large model is resident"
            ),
            remediation=(
                "do not wrap big-memory runs in scripts/gpu_guard.sh, or raise "
                "ED4ALL_GPU_MAX_USED_MB above the total VRAM"
            ),
            data={
                "flag": ENV_GPU_MAX_USED_MB,
                "ceiling_mib": ceiling,
                "total_mib": total_mib,
            },
        )
    ]


def gpu_profile_checks(ctx: CheckContext) -> List[CheckResult]:
    """Advise on concurrency-hostile defaults when a big-memory GPU is seen.

    Returns an EMPTY list (silent, not-applicable) unless a big-memory GPU
    is detected (total VRAM ≥ ``ED4ALL_BIG_MEMORY_MIN_MIB``). When one is
    detected it emits an OK summary line plus one WARN per concurrency-hostile
    default currently in effect. NEVER raises and NEVER errors — any failure
    degrades to the empty (not-applicable) list.
    """
    try:
        total = _probe_total_mib(ctx.base_url)
        if total is None:
            # No probeable GPU (no torch / no CUDA / probe unavailable) →
            # not applicable; skip cleanly.
            return []
        threshold = resolve_big_memory_min_mib()
        if total < threshold:
            # Small dev-box GPU — this profile does not apply; stay silent so
            # the dev-box doctor output is byte-identical.
            return []

        results: List[CheckResult] = [
            CheckResult(
                name="gpu_profile_detected",
                group=_GROUP,
                severity=Severity.OK,
                summary=(
                    f"big-memory GPU detected ({total} MiB total ≥ {threshold} "
                    "MiB threshold) — checking concurrent-serving profile"
                ),
                detail=(
                    "See the big-memory concurrent-serving operator runbook for "
                    "the full flag recipe. Any warnings below are advisory."
                ),
                data={"total_mib": total, "threshold_mib": threshold},
            )
        ]
        results.extend(_lifecycle_warning())
        results.extend(_nli_evict_warning())
        results.extend(_rate_limit_warning())
        results.extend(_gpu_guard_warning(total))
        return results
    except Exception as exc:  # noqa: BLE001 — advisory check must never crash/degrade to WARN
        logger.debug("gpu_profile: check degraded to not-applicable: %s", exc)
        return []


def register_gpu_profile_checks() -> None:
    """Register :func:`gpu_profile_checks` under ``group="gpu_profile"``.

    Called explicitly from the CLI bootstrap — NEVER at import time.
    """
    register(_GROUP, gpu_profile_checks)


__all__ = [
    "ENV_BIG_MEMORY_MIN_MIB",
    "ENV_GPU_MAX_USED_MB",
    "resolve_big_memory_min_mib",
    "gpu_profile_checks",
    "register_gpu_profile_checks",
]
