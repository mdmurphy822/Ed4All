"""Pluggable ``ed4all doctor`` foundation.

A small check-registry layer (:mod:`lib.diagnostics.core`) plus the VRAM
check adapter (:mod:`lib.diagnostics.vram`) that wraps the existing
:mod:`lib.llm.vram_doctor` foundation. Generalizes the (currently
hardcoded, VRAM-only) doctor command into a pluggable multi-check doctor.

Registration is explicit (the CLI bootstrap calls the per-group
``register_*`` helpers) — there is NO import-time auto-registration, so
importing this package has no side effects.
"""

from __future__ import annotations

from lib.diagnostics.core import (
    CheckContext,
    CheckFn,
    CheckResult,
    Severity,
    clear_registry,
    format_report,
    register,
    registered_checks,
    resolve_exit_code,
    resolve_verdict,
    results_to_json,
    run_checks,
)
from lib.diagnostics.vram import gpu_checks, register_gpu_checks

__all__ = [
    "Severity",
    "CheckResult",
    "CheckContext",
    "CheckFn",
    "register",
    "registered_checks",
    "clear_registry",
    "run_checks",
    "resolve_exit_code",
    "resolve_verdict",
    "format_report",
    "results_to_json",
    "gpu_checks",
    "register_gpu_checks",
]
