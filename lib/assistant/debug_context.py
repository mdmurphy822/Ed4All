"""Debug-mode context builder for the operator assistant.

:func:`build_debug_context` assembles a bounded, secret-filtered diagnostic
snapshot of a FAILED workflow run — run report + failing-phase gate report +
log tail + doctor post-mortem — for injection into an
``AssistantEngine(mode="debug")`` session and for the CLI failure banner.

Everything flows through the SAME summarizers / redaction as the tool
whitelist (:mod:`lib.assistant.tools`), so debug mode grants the model no
capability beyond the registered tools; the total context is hard-capped at
:data:`MAX_DEBUG_CONTEXT_CHARS` (~6000 chars) for the 32k-context seat.

Run selection when ``run_id`` is ``None``:

1. ``<campaign dir>/last_failure.json`` — the pointer the operator campaign
   driver writes under ``ED4ALL_ASSISTANT_DEBUG_ON_FAILURE`` (validated
   before use; never trusted as a path). The campaign dir is resolved by
   ``lib.paths.campaign_dir`` (``ED4ALL_CAMPAIGN_DIR``, default
   ``plans/campaign``).
2. Otherwise the newest ``state/workflows/WF-*.json`` whose status is
   FAILED (by file mtime).

No failed run found → :class:`DebugContextUnavailable` (loud, never a
fabricated context).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

from lib.assistant import tools as assistant_tools
from lib.assistant.tools import (
    CAMPAIGN_DIR,
    RUN_ID_RE,
    SLUG_RE,
    STATE_PATH,
    _clip,
    _redact,
)
from lib.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

#: Hard cap on the composed debug-context text.
MAX_DEBUG_CONTEXT_CHARS = 6000

#: Log lines pulled into the context (~80 per spec).
DEBUG_LOG_TAIL_LINES = 80

#: Pointer file the campaign driver writes on a failed book
#: (ED4ALL_ASSISTANT_DEBUG_ON_FAILURE).
LAST_FAILURE_POINTER = CAMPAIGN_DIR / "last_failure.json"


class DebugContextUnavailable(RuntimeError):
    """No failed run could be resolved for debug mode."""


def _read_pointer() -> Tuple[Optional[str], Optional[str]]:
    """(run_id, slug) from last_failure.json — both validated, else None."""
    if not LAST_FAILURE_POINTER.is_file():
        return None, None
    try:
        doc = json.loads(LAST_FAILURE_POINTER.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(doc, dict):
        return None, None
    run_id = str(doc.get("run_id") or "").strip()
    slug = str(doc.get("slug") or "").strip()
    return (
        run_id if RUN_ID_RE.match(run_id) else None,
        slug if SLUG_RE.match(slug) else None,
    )


def _newest_failed_run_id() -> Optional[str]:
    """Newest FAILED run in state/workflows (by mtime); id shape-validated."""
    workflows_dir = STATE_PATH / "workflows"
    if not workflows_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in workflows_dir.glob("WF-*.json") if RUN_ID_RE.match(p.stem)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and str(doc.get("status", "")).upper() == "FAILED":
            return path.stem
    return None


def _failed_phase_and_slug(run_id: str) -> Tuple[Optional[str], Optional[str]]:
    """(failed_phase, course slug) straight from the workflow state doc."""
    path = STATE_PATH / "workflows" / f"{run_id}.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, None
    if not isinstance(doc, dict):
        return None, None
    phase = doc.get("failed_phase") or doc.get("paused_phase")
    params = doc.get("params") or {}
    slug = str(params.get("course_name") or "").strip()
    return (
        str(phase) if phase else None,
        slug if SLUG_RE.match(slug) else None,
    )


def _doctor_postmortem(run_id: str) -> str:
    """Bounded ``ed4all doctor --run-id`` post-mortem (fixed argv, redacted).

    The doctor CLI never raises; a non-runnable post-mortem is reported as
    text. The run id is the pre-validated WF id — the only model-adjacent
    value that reaches the argv.
    """
    try:
        proc = subprocess.run(  # noqa: S603 — fixed argv, validated run id
            [sys.executable, "-m", "cli.main", "doctor", "--run-id", run_id],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(PROJECT_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"doctor post-mortem unavailable ({type(exc).__name__}: {exc})"
    output = (proc.stdout or "").strip() or (proc.stderr or "").strip()
    return _clip(_redact(output or f"doctor exited {proc.returncode} with no output"), 1200)


def _one_paragraph_summary(
    run_id: str, phase: Optional[str], slug: Optional[str], run_report_text: str
) -> str:
    reason = ""
    match = re.search(r"failure_reason: (.+)", run_report_text)
    if match:
        reason = _clip(match.group(1).strip(), 220)
    bits = [f"Run {run_id}"]
    if slug:
        bits.append(f"(course {slug})")
    bits.append("FAILED")
    if phase:
        bits.append(f"in phase {phase}")
    summary = " ".join(bits) + "."
    if reason:
        summary += f" Reason: {reason}"
    return summary


def build_debug_context(run_id: Optional[str] = None) -> Dict[str, Any]:
    """Build the bounded debug context for a failed run.

    Args:
        run_id: An explicit WF run id, or ``None`` to auto-resolve (pointer
            file first, then the newest FAILED state/workflows entry).

    Returns:
        Dict with ``run_id``, ``source`` (``explicit`` /
        ``last_failure_pointer`` / ``newest_failed``), ``slug``,
        ``failed_phase``, ``summary`` (one paragraph, banner-ready), and
        ``text`` (the composed, redacted, ≤ ~6000-char context block).

    Raises:
        DebugContextUnavailable: invalid explicit id, or no failed run found.
    """
    slug: Optional[str] = None
    if run_id is not None:
        candidate = str(run_id).strip()
        if not RUN_ID_RE.match(candidate):
            raise DebugContextUnavailable(
                f"{candidate[:80]!r} is not a valid run id (expected WF-YYYYMMDD-xxxxxxxx)."
            )
        resolved, source = candidate, "explicit"
    else:
        pointer_run, pointer_slug = _read_pointer()
        if pointer_run is not None:
            resolved, source, slug = pointer_run, "last_failure_pointer", pointer_slug
        else:
            # Pointer absent — or present without an extractable run id (the
            # driver could not find one in the log): fall back to the newest
            # FAILED state entry, keeping the pointer's slug for log lookup.
            slug = pointer_slug
            newest = _newest_failed_run_id()
            if newest is None:
                raise DebugContextUnavailable(
                    "No failed run found: no usable last_failure.json pointer "
                    "and no FAILED entry under state/workflows/. Nothing to "
                    "debug."
                )
            resolved, source = newest, "newest_failed"

    failed_phase, state_slug = _failed_phase_and_slug(resolved)
    slug = slug or state_slug

    # All sections go through the SAME summarizer/redaction path the tools
    # use — debug mode adds no capability beyond the whitelist.
    run_report_text = _clip(assistant_tools.run_report(resolved), 1400)
    gate_report_text = _clip(
        assistant_tools.gate_report(resolved, failed_phase or ""), 1800
    )
    log_target = slug if slug and (CAMPAIGN_DIR / "logs" / f"{slug}.log").is_file() else resolved
    log_tail_text = _clip(
        assistant_tools.tail_log(log_target, DEBUG_LOG_TAIL_LINES), 1800
    )
    doctor_text = _doctor_postmortem(resolved)

    summary = _one_paragraph_summary(resolved, failed_phase, slug, run_report_text)

    sections: List[str] = [
        f"=== FAILED RUN DEBUG CONTEXT ({source}) ===",
        summary,
        "--- run report ---",
        run_report_text,
        "--- gate report (failing phase) ---",
        gate_report_text,
        f"--- log tail ({log_target}) ---",
        log_tail_text,
        "--- doctor post-mortem ---",
        doctor_text,
    ]
    text = _clip(_redact("\n".join(sections)), MAX_DEBUG_CONTEXT_CHARS)

    return {
        "run_id": resolved,
        "source": source,
        "slug": slug,
        "failed_phase": failed_phase,
        "summary": summary,
        "run_report": run_report_text,
        "gate_report": gate_report_text,
        "log_tail": log_tail_text,
        "doctor": doctor_text,
        "text": text,
    }


__all__ = [
    "DEBUG_LOG_TAIL_LINES",
    "DebugContextUnavailable",
    "LAST_FAILURE_POINTER",
    "MAX_DEBUG_CONTEXT_CHARS",
    "build_debug_context",
]
