"""VRAM check ADAPTER — wraps :mod:`lib.llm.vram_doctor` as a doctor check.

This is the bridge between the existing (never-raising) VRAM-doctor
foundation and the pluggable :mod:`lib.diagnostics.core` registry. It owns
ZERO new probing / device policy — it calls
:func:`lib.llm.vram_doctor.snapshot_vram` +
:func:`lib.llm.vram_doctor.fit_check` and maps each
:class:`~lib.llm.vram_doctor.FitVerdict` onto a
:class:`~lib.diagnostics.core.CheckResult` under ``group="gpu"``.

Registration is explicit (:func:`register_gpu_checks`), called by the CLI
bootstrap — never at import time.
"""

from __future__ import annotations

import logging
from typing import List

from lib.diagnostics.core import CheckContext, CheckResult, Severity, register

logger = logging.getLogger(__name__)

#: ``FitVerdict.outcome`` → :class:`Severity`. ``would_oom`` and
#: ``would_fail_closed`` are the hard fails (the build would crash /
#: stop mid-run); a cuda→CPU fallback is a warning; ``unknown`` (the
#: free-VRAM probe could not determine a value — NVML +
#: torch.cuda.mem_get_info both unavailable, the documented WSL2 quirk) is
#: purely INFO — it is NOT evidence of a problem and must NOT push the exit
#: code above 0 (matching the prior committed VRAM doctor's exit-0
#: treatment); everything else fits / is fine. An unrecognized/garbage
#: outcome falls through to WARN (a genuinely unexpected value is worth
#: flagging, unlike the well-defined ``unknown``).
#:
#: ``would_fallback_cpu`` and ``would_fail_closed`` are deliberately SEPARATE
#: outcomes with separate severities. They describe the same trigger (cuda
#: requested, cuda absent) on consumers with opposite contracts: NLI has a
#: blessed CPU fallback and merely gets slower (WARN); the embedding stack
#: raises (``EmbeddingModelUnavailable`` / ``EmbeddingBackendUnavailable``)
#: and stops the run (FAIL). Rendering the second as the first told the
#: operator the run was safe when it was not.
_OUTCOME_SEVERITY = {
    "would_oom": Severity.FAIL,
    "would_fail_closed": Severity.FAIL,
    "would_fallback_cpu": Severity.WARN,
    "unknown": Severity.INFO,
    "fits": Severity.OK,
    "cpu_requested": Severity.OK,
    "would_evict_then_fit": Severity.OK,
}

#: Per-outcome remediation hints for the WARN/FAIL cases. ``unknown`` is INFO
#: (no remediation rendered), so it carries no entry here.
_OUTCOME_REMEDIATION = {
    "would_oom": (
        "free VRAM below need — evict the resident model or run uncontended, "
        "or set ED4ALL_NLI_DEVICE=cpu / ED4ALL_EMBEDDING_DEVICE=cpu to run on CPU"
    ),
    "would_fallback_cpu": (
        "cuda requested but the consumer will fall back to CPU (safe but ~20-50x "
        "slower) — free VRAM or accept the slowdown"
    ),
    "would_fail_closed": (
        "cuda requested but unavailable, and this consumer has NO CPU fallback — "
        "it raises and stops the run. Provision the requested device, or set "
        "ED4ALL_EMBEDDING_DEVICE=cpu to run the embedder on CPU as an explicit "
        "operator choice (there is no automatic CUDA→CPU downgrade)"
    ),
}


def gpu_checks(ctx: CheckContext) -> List[CheckResult]:
    """Snapshot VRAM + fit-check every in-process GPU consumer (NEVER raises).

    Calls :func:`snapshot_vram` (passing ``ctx.base_url``) then
    :func:`fit_check`, emits one OK/info :class:`CheckResult` summarizing the
    snapshot (free/total/source/resident models), and one result per
    :class:`FitVerdict` mapped through :data:`_OUTCOME_SEVERITY`. Each
    fit-check result carries ``need_mib`` / ``free_mib`` / ``outcome`` /
    ``device`` in ``data`` and a populated ``remediation`` for WARN/FAIL.
    """
    try:
        from lib.llm.vram_doctor import fit_check, snapshot_vram
    except Exception as exc:  # noqa: BLE001 — foundation import must not crash the doctor
        logger.warning("diagnostics.vram: vram_doctor import failed: %s", exc)
        return [
            CheckResult(
                name="gpu_snapshot",
                group="gpu",
                severity=Severity.WARN,
                summary="VRAM doctor unavailable (import failed)",
                detail=f"{type(exc).__name__}: {exc}",
                remediation="verify lib.llm.vram_doctor imports on this box",
                data={"error": str(exc)},
            )
        ]

    results: List[CheckResult] = []

    try:
        snapshot = snapshot_vram(ctx.base_url)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders (foundation never raises)
        logger.warning("diagnostics.vram: snapshot_vram raised: %s", exc)
        return [
            CheckResult(
                name="gpu_snapshot",
                group="gpu",
                severity=Severity.WARN,
                summary=f"VRAM snapshot failed: {exc}",
                detail=f"{type(exc).__name__}: {exc}",
                remediation="verify the GPU/NVML/ollama stack",
                data={"error": str(exc)},
            )
        ]

    free = "unknown" if snapshot.free_mib is None else f"{snapshot.free_mib} MiB"
    total = "unknown" if snapshot.total_mib is None else f"{snapshot.total_mib} MiB"
    resident_names = [m.get("name", "?") for m in snapshot.resident_models]
    summary = (
        f"GPU: free {free} / total {total} "
        f"(source: {snapshot.probe_source}, cuda_available={snapshot.cuda_available}, "
        f"resident: {len(resident_names)})"
    )
    results.append(
        CheckResult(
            name="gpu_snapshot",
            group="gpu",
            severity=Severity.OK,
            summary=summary,
            detail=(f"ollama probe: {snapshot.error}" if snapshot.error else ""),
            data={
                "free_mib": snapshot.free_mib,
                "total_mib": snapshot.total_mib,
                "probe_source": snapshot.probe_source,
                "cuda_available": snapshot.cuda_available,
                "resident_models": list(snapshot.resident_models),
                "error": snapshot.error,
            },
        )
    )

    try:
        verdicts = fit_check(snapshot)
    except Exception as exc:  # noqa: BLE001 — belt-and-suspenders
        logger.warning("diagnostics.vram: fit_check raised: %s", exc)
        results.append(
            CheckResult(
                name="gpu_fit_error",
                group="gpu",
                severity=Severity.WARN,
                summary=f"fit-check failed: {exc}",
                detail=f"{type(exc).__name__}: {exc}",
                remediation="verify lib.llm.vram_doctor.fit_check on this box",
                data={"error": str(exc)},
            )
        )
        return results

    for verdict in verdicts:
        severity = _OUTCOME_SEVERITY.get(verdict.outcome, Severity.WARN)
        remediation = (
            _OUTCOME_REMEDIATION.get(verdict.outcome, "")
            if severity in (Severity.WARN, Severity.FAIL)
            else ""
        )
        free_str = "unknown" if verdict.free_mib is None else f"{verdict.free_mib} MiB"
        if verdict.outcome == "unknown":
            summary = (
                f"{verdict.consumer} (device={verdict.device_requested}, "
                f"need={verdict.need_mib} MiB): could not probe free VRAM "
                "(NVML + torch unavailable) — placement unknown, not necessarily "
                "a problem"
            )
        else:
            summary = (
                f"{verdict.consumer} (device={verdict.device_requested}, "
                f"need={verdict.need_mib} MiB, free={free_str}): {verdict.outcome}"
            )
        results.append(
            CheckResult(
                name=f"gpu_fit_{verdict.consumer}",
                group="gpu",
                severity=severity,
                summary=summary,
                detail=verdict.detail,
                remediation=remediation,
                data={
                    "consumer": verdict.consumer,
                    "device": verdict.device_requested,
                    "need_mib": verdict.need_mib,
                    "free_mib": verdict.free_mib,
                    "outcome": verdict.outcome,
                },
            )
        )

    return results


def register_gpu_checks() -> None:
    """Register :func:`gpu_checks` under ``group="gpu"``.

    Called explicitly from the CLI bootstrap (a later worker), NEVER at
    import time.
    """
    register("gpu", gpu_checks)


__all__ = ["gpu_checks", "register_gpu_checks"]
