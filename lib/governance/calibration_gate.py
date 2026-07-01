"""Calibration-gated severity-flip resolver (Wave 3 Worker W3.A).

A reusable helper that reads the most recent
``LibV2/courses/<course_slug>/quality/trainforge_assessment_quality_report.json``
calibration report, extracts a named summary signal, and decides
whether a calibration-gated severity flip should fire.

Why a helper rather than inline at the call site
================================================

Multiple validators in Waves 3+ will adopt the same pattern: a gate
that ships at warning severity, then promotes to critical once a
calibration corpus signal proves the gate is stable enough to fail
closed without false positives in CI. Centralising the
threshold-readback logic here keeps the audit trail consistent across
flips and ensures every flip emits the same shape of decision-capture
event so a downstream eval surface can audit "which gates flipped on
which calibration runs" with a single grep.

The helper itself does NOT emit decision-capture events — that is the
caller's responsibility, since the caller owns the
:class:`DecisionCapture` instance and knows the right ``operation`` /
``run_id`` / ``course_id`` to interpolate. The helper returns the
event payload pre-built so the caller can hand it directly to
``capture.log_decision(**payload)``.

References
----------

* ``plans/gpt-feedback-2-wave3-wiring-telemetry-2026-05.md`` § Worker
  W3.A — calibration-gated severity flip on
  ``AssessmentRetrievalGroundingValidator``.
* ``lib/aggregators/trainforge_assessment_quality_report.py`` — the
  canonical W2.B report this helper reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CALIBRATION_COURSE_SLUG",
    "DEFAULT_LIBV2_ROOT",
    "resolve_severity_flip",
]


#: Default reference course slug for calibration runs. The Wave 2 W2.B
#: deliverable produces this corpus's
#: ``trainforge_assessment_quality_report.json`` from
#: ``tests/fixtures/calibration_textbook.pdf``.
DEFAULT_CALIBRATION_COURSE_SLUG: str = "calibration_textbook"

#: Default LibV2 root. Resolved relative to the repo root so a clean
#: checkout that has never run a calibration build resolves to a
#: non-existent path and the flip silently defers.
DEFAULT_LIBV2_ROOT: Path = Path(__file__).resolve().parents[2] / "LibV2"


def _resolve_report_path(course_slug: str, libv2_root: Path) -> Path:
    """Compute the canonical calibration-report path."""
    return (
        libv2_root
        / "courses"
        / course_slug
        / "quality"
        / "trainforge_assessment_quality_report.json"
    )


def _read_calibration_report(report_path: Path) -> Optional[Dict[str, Any]]:
    """Read + JSON-decode the calibration report.

    Returns ``None`` on missing-file / decode-error / non-mapping —
    the caller treats every failure mode as "calibration source
    missing" and defers the flip.
    """
    if not report_path.exists():
        return None
    try:
        with report_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        logger.debug(
            "Calibration report at %s failed to load: %s", report_path, exc
        )
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def resolve_severity_flip(
    *,
    calibration_signal: str,
    threshold: float,
    course_slug: str = DEFAULT_CALIBRATION_COURSE_SLUG,
    libv2_root: Optional[Path] = None,
    fallback_signals: Sequence[str] = (),
) -> Tuple[bool, Dict[str, Any]]:
    """Decide whether a calibration-gated severity flip should fire.

    Reads
    ``<libv2_root>/courses/<course_slug>/quality/trainforge_assessment_quality_report.json``,
    extracts ``summary[<calibration_signal>]``, and returns
    ``(apply_flip, decision_payload)`` where ``apply_flip`` is True
    iff the observed value is a finite numeric ``>= threshold``.

    The returned ``decision_payload`` is a kwargs dict suitable for
    ``DecisionCapture.log_decision(**payload)``. The caller is
    responsible for actually emitting the event (the helper has no
    capture handle of its own).

    Decision-event shape:

    * On flip applied (``apply_flip=True``)::

        {
            "decision_type": "severity_flip_applied",
            "decision": "critical",
            "rationale": "Calibration signal <name> = <value> >= "
                         "threshold <threshold> at "
                         "<calibration_source>; flipping severity "
                         "warning -> critical.",
            "ml_features": {
                "calibration_source": "<absolute path>",
                "calibration_signal": "<name>",
                "observed_rate": <float>,
                "threshold": <float>,
            },
        }

    * On flip deferred (``apply_flip=False``)::

        {
            "decision_type": "severity_flip_deferred",
            "decision": "warning",
            "rationale": "<reason>; holding severity at warning.",
            "ml_features": {
                "calibration_source": "<absolute path | None>",
                "calibration_signal": "<name>",
                "observed_rate": <float | None>,
                "threshold": <float>,
                "reason": "calibration_report_missing"
                          | "calibration_signal_missing"
                          | "alignment_rate_below_floor",
            },
        }

    Args:
        calibration_signal: Name of the field under
            ``trainforge_assessment_quality_report.json::summary``
            whose value drives the flip.
        threshold: Floor at or above which the flip fires.
        course_slug: Reference calibration-run course slug. Defaults
            to ``calibration_textbook`` (the W2.B canonical fixture).
        libv2_root: Override the LibV2 root path. Test-only; production
            callers use the default.
        fallback_signals: Ordered sibling summary fields to try when
            ``calibration_signal`` is absent / non-numeric. W8.1: the
            ``trainforge_assessment_quality_report.json`` schema
            (``schemas/aggregators/…`` — ``summary.additionalProperties:
            false``) has no ``answerability_alignment_rate`` field, so a
            real calibration build cannot carry it directly; the
            equivalent value is emitted under the schema-present
            ``answerable_rate`` (itself sourced verbatim from the
            ``AssessmentRetrievalGroundingValidator`` answer↔source
            alignment score). Passing ``("answerable_rate",)`` lets the
            flip read that REAL input instead of the never-written
            canonical field, so the flip is no longer dead. Defaults to
            empty (byte-identical to the pre-W8.1 single-signal read).

    Returns:
        ``(apply_flip, decision_payload)``. The caller fans out:
        when ``apply_flip`` is True, downstream emits use critical
        severity; when False, severity stays at warning.
    """
    root = libv2_root or DEFAULT_LIBV2_ROOT
    report_path = _resolve_report_path(course_slug, root)
    payload = _read_calibration_report(report_path)

    threshold_f = float(threshold)
    base_features: Dict[str, Any] = {
        "calibration_source": str(report_path),
        "calibration_signal": calibration_signal,
        "threshold": threshold_f,
    }

    if payload is None:
        rationale = (
            f"Calibration report missing at {report_path}; "
            "holding severity at warning until a calibration build lands."
        )
        return False, {
            "decision_type": "severity_flip_deferred",
            "decision": "warning",
            "rationale": rationale,
            "ml_features": {
                **base_features,
                "calibration_source": str(report_path),
                "observed_rate": None,
                "reason": "calibration_report_missing",
            },
        }

    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        rationale = (
            f"Calibration signal {calibration_signal!r} not present in "
            f"{report_path}::summary; holding severity at warning."
        )
        return False, {
            "decision_type": "severity_flip_deferred",
            "decision": "warning",
            "rationale": rationale,
            "ml_features": {
                **base_features,
                "observed_rate": None,
                "reason": "calibration_signal_missing",
            },
        }

    # W8.1 — try the canonical signal first, then any fallback siblings.
    # A fallback that resolves records BOTH the field actually read
    # (``calibration_signal``) and the canonical name requested
    # (``requested_signal``) so the audit trail is honest about which
    # summary field drove the flip.
    resolved_signal: Optional[str] = None
    observed: Any = None
    for candidate in (calibration_signal, *fallback_signals):
        val = summary.get(candidate)
        if isinstance(val, bool):
            # bool is an int subclass but never a valid rate.
            continue
        if isinstance(val, (int, float)):
            resolved_signal = candidate
            observed = val
            break

    if resolved_signal is None:
        tried = ", ".join(repr(s) for s in (calibration_signal, *fallback_signals))
        rationale = (
            f"Calibration signal(s) {tried} not present / not numeric in "
            f"{report_path}::summary; holding severity at warning."
        )
        return False, {
            "decision_type": "severity_flip_deferred",
            "decision": "warning",
            "rationale": rationale,
            "ml_features": {
                **base_features,
                "observed_rate": None,
                "reason": "calibration_signal_missing",
            },
        }

    # Reflect the field actually read (may be a fallback sibling) while
    # preserving the requested canonical name for provenance.
    base_features = {
        **base_features,
        "calibration_signal": resolved_signal,
        "requested_signal": calibration_signal,
    }
    observed_f = float(observed)
    if observed_f < threshold_f:
        rationale = (
            f"Calibration signal {calibration_signal} = {observed_f:.4f} < "
            f"threshold {threshold_f:.4f} at {report_path}; "
            "holding severity at warning."
        )
        return False, {
            "decision_type": "severity_flip_deferred",
            "decision": "warning",
            "rationale": rationale,
            "ml_features": {
                **base_features,
                "observed_rate": observed_f,
                "reason": "alignment_rate_below_floor",
            },
        }

    rationale = (
        f"Calibration signal {calibration_signal} = {observed_f:.4f} >= "
        f"threshold {threshold_f:.4f} at {report_path}; "
        "flipping severity warning -> critical."
    )
    return True, {
        "decision_type": "severity_flip_applied",
        "decision": "critical",
        "rationale": rationale,
        "ml_features": {
            **base_features,
            "observed_rate": observed_f,
        },
    }
