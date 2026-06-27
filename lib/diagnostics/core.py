"""Pluggable doctor foundation — a check registry + result model.

This module generalizes the (currently VRAM-only, hardcoded) ``ed4all
doctor`` command into a pluggable multi-check doctor. A *check* is a
callable that takes a :class:`CheckContext` and returns a list of
:class:`CheckResult` (one check may emit several results — e.g. one per
GPU consumer). Checks register themselves into a module-level registry
via :func:`register`; the CLI bootstrap (a later worker) calls the
per-group ``register_*`` helpers explicitly so there is NO import-time
magic / auto-registration to couple import order or muddy test isolation.

Design contract (inherited from :mod:`lib.llm.vram_doctor`): a doctor
that crashes the run is worse than no doctor. Every public function here
is best-effort — :func:`run_checks` isolates each check so a raising check
becomes a single WARN result rather than propagating, and the registry /
formatting / serialization helpers never raise on well-formed input.

The exit-code + verdict wording is kept compatible with the existing
``cli/commands/doctor.py`` contract: a FAIL (``would_oom``) → exit 2 /
"DANGER"; a WARN (cuda→CPU fallback) → exit 1 / "DEGRADED"; all-OK →
exit 0 / "OK".
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


class Severity(enum.Enum):
    """Outcome severity for a single :class:`CheckResult`.

    Ordered (informally) INFO < OK < WARN < FAIL for advisory purposes —
    :func:`resolve_exit_code` escalates the process exit code to the worst
    *escalating* severity present. INFO is BELOW WARN: it is purely
    informational — shown to the operator, but it NEVER affects the exit
    code or the overall verdict (an INFO-only result set exits 0 / "OK").
    """

    OK = "ok"
    INFO = "info"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class CheckResult:
    """One diagnostic finding emitted by a check.

    A check may emit several of these (e.g. one per GPU consumer plus a
    snapshot-summary line). ``data`` carries structured extras surfaced in
    ``--json`` mode; ``remediation`` should be populated for WARN/FAIL so
    the operator knows what to do.
    """

    name: str  # short stable id, e.g. "gpu_fit_nli"
    group: str  # e.g. "gpu" | "environment" | "window" | "provider"
    severity: Severity
    summary: str  # one-line human summary
    detail: str = ""  # optional longer text
    remediation: str = ""  # what to do (populate for WARN/FAIL)
    data: Dict = field(default_factory=dict)  # structured extras for --json


@dataclass
class CheckContext:
    """Inputs handed to every check.

    ``base_url`` is the ollama base from the CLI ``--base-url`` flag.
    ``run_config`` is populated only in ``--run`` mode (a later phase); it
    is ``None`` for plain environment checks.
    ``run_id`` is populated only in ``--run-id`` post-mortem mode (the
    forensic run-analysis check group); it is ``None`` for every other
    check. Additive + default ``None`` → fully backward-compatible with
    existing ``CheckContext(base_url=..., run_config=...)`` call sites.
    """

    base_url: Optional[str] = None
    run_config: Optional[dict] = None
    run_id: Optional[str] = None


#: A check is a callable taking a :class:`CheckContext` and returning a
#: list of :class:`CheckResult`. Checks MUST NEVER raise (and
#: :func:`run_checks` isolates them belt-and-suspenders if they do).
CheckFn = Callable[[CheckContext], List[CheckResult]]


# --------------------------------------------------------------------- #
# Registry — module-level, explicit registration (NO import-time magic).
# --------------------------------------------------------------------- #

_REGISTRY: List[Tuple[str, CheckFn]] = []


def register(group: str, fn: CheckFn) -> None:
    """Register one check ``fn`` under ``group`` (appended → runs in order).

    Explicit registration only — there is no import-time auto-registration.
    The CLI bootstrap calls the per-group ``register_*`` helpers (e.g.
    :func:`lib.diagnostics.vram.register_gpu_checks`).
    """
    _REGISTRY.append((group, fn))


def registered_checks() -> List[Tuple[str, CheckFn]]:
    """Return the registered ``(group, fn)`` pairs in registration order."""
    return list(_REGISTRY)


def clear_registry() -> None:
    """Empty the registry (test isolation / re-bootstrap)."""
    _REGISTRY.clear()


def run_checks(
    context: CheckContext, groups: Optional[Iterable[str]] = None
) -> List[CheckResult]:
    """Run every registered check (optionally filtered to ``groups``).

    Runs in registration order. NEVER raises: if a check fn raises, the
    exception is captured as a single WARN :class:`CheckResult`
    (``name=f'{group}_error'``, ``summary='check errored: <exc>'``) and the
    run continues. Returns the flat list of all results.
    """
    wanted = set(groups) if groups is not None else None
    results: List[CheckResult] = []
    for group, fn in registered_checks():
        if wanted is not None and group not in wanted:
            continue
        try:
            produced = fn(context)
        except Exception as exc:  # noqa: BLE001 — a doctor must never crash
            logger.warning("diagnostics: check in group %r raised: %s", group, exc)
            results.append(
                CheckResult(
                    name=f"{group}_error",
                    group=group,
                    severity=Severity.WARN,
                    summary=f"check errored: {exc}",
                    detail=f"{type(exc).__name__}: {exc}",
                    remediation="this is a doctor bug — the check should never raise",
                    data={"error": str(exc), "error_type": type(exc).__name__},
                )
            )
            continue
        if produced:
            results.extend(produced)
    return results


def resolve_exit_code(results: List[CheckResult]) -> int:
    """Return ``2`` if any FAIL, else ``1`` if any WARN, else ``0``.

    Matches the existing doctor exit contract: ``would_oom`` → FAIL → 2;
    a cuda→CPU fallback → WARN → 1; all-ok (or empty) → 0. INFO is below
    WARN and contributes 0 (like OK), so an INFO-only / INFO+OK result set
    returns 0 — INFO never escalates the exit code.
    """
    severities = {r.severity for r in results}
    if Severity.FAIL in severities:
        return 2
    if Severity.WARN in severities:
        return 1
    return 0


def resolve_verdict(results: List[CheckResult]) -> str:
    """One-line overall verdict, mirroring ``cli/commands/doctor.py``.

    FAIL → ``"DANGER: ..."``; WARN → ``"DEGRADED: ..."``; else ``"OK"``.
    The named-consumer suffix lists the offending result summaries' names.
    INFO is below WARN: it never triggers DEGRADED and is never listed in
    the DEGRADED/DANGER summary (an INFO-only set verdicts ``"OK"``).
    """
    fails = [r for r in results if r.severity is Severity.FAIL]
    if fails:
        names = ", ".join(r.name for r in fails)
        return f"DANGER: {names} — abort the build, fix before launching."
    warns = [r for r in results if r.severity is Severity.WARN]
    if warns:
        names = ", ".join(r.name for r in warns)
        return f"DEGRADED: {names} — safe but degraded; proceed knowingly."
    return "OK"


_SEVERITY_MARKERS = {
    Severity.OK: "✓",
    Severity.INFO: "ℹ",
    Severity.WARN: "⚠",
    Severity.FAIL: "✗",
}


def format_report(results: List[CheckResult]) -> str:
    """Render a human-readable report, GROUPED by ``group``.

    Each result is a line prefixed with a marker (OK ``✓`` / INFO ``ℹ`` /
    WARN ``⚠`` / FAIL ``✗``) and its summary; a non-empty ``remediation``
    on a WARN/FAIL is shown indented underneath. INFO lines render like any
    other (marker + summary) and generally carry no remediation, so none is
    shown (it never crashes if one is present). The report ends with a
    one-line overall verdict (``OK`` / ``DEGRADED: ...`` / ``DANGER: ...``).
    Best-effort — never raises; a malformed result degrades to a plain line.
    """
    lines: List[str] = ["ed4all doctor report"]

    if not results:
        lines.append("  (no checks registered)")

    # Preserve first-seen group order while grouping.
    group_order: List[str] = []
    by_group: Dict[str, List[CheckResult]] = {}
    for result in results:
        group = getattr(result, "group", "?") or "?"
        if group not in by_group:
            by_group[group] = []
            group_order.append(group)
        by_group[group].append(result)

    for group in group_order:
        lines.append(f"  [{group}]")
        for result in by_group[group]:
            try:
                marker = _SEVERITY_MARKERS.get(result.severity, "?")
                lines.append(f"    {marker} {result.summary}")
                if result.severity in (Severity.WARN, Severity.FAIL) and result.remediation:
                    lines.append(f"        → {result.remediation}")
            except Exception:  # noqa: BLE001 — render must never crash the doctor
                lines.append(f"    ? {result!r}")

    lines.append("")
    lines.append(resolve_verdict(results))
    return "\n".join(lines)


def results_to_json(results: List[CheckResult]) -> List[dict]:
    """Serialize results to plain dicts (``Severity`` → its ``.value``).

    ``dataclasses.asdict``-style, but with the ``Severity`` enum flattened
    to its string value so the payload is JSON-serializable.
    """
    payload: List[dict] = []
    for result in results:
        payload.append(
            {
                "name": result.name,
                "group": result.group,
                "severity": result.severity.value,
                "summary": result.summary,
                "detail": result.detail,
                "remediation": result.remediation,
                "data": dict(result.data),
            }
        )
    return payload


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
]
