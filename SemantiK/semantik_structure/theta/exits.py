"""Stage 13 — exit-decision table over (WCAG × lane × theta × floors).

Implements the table in ``architecture.md:587-597``. Reads a
:class:`ThetaReport` from Stage 12 and stamps the final ``action``.

Two architecture rows route to the offline-Qwen lane:

  * ``retry_offline`` — pass | fast | theta < 0.70.
  * ``offline_qwen_lane`` — fail | fast | any.

Both are now implemented in
:func:`semantik_structure.theta.offline_retry.maybe_offline_retry`. The
cascade driver MUST call ``maybe_offline_retry(fast_report, run_lane=...)``
BEFORE calling :func:`decide_exit` — that orchestrator decides whether
to retry, dispatches the offline pass, reconciles the two reports,
and returns the FINAL report. By the time :func:`decide_exit` runs,
the only valid combinations are:

  * fast | pass | any theta — either ``theta >= TAU_THETA_RETRY`` (no
    retry was needed) OR the orchestrator ran the offline lane and
    decided to keep fast (retry failed to beat ``DELTA_THETA_IMPROVE``).
  * offline | pass | any theta — retry succeeded and beat the delta.
  * offline | failed — both lanes were tried; genuine non-certifiable.
  * fast | failed — DEFENSIVE ONLY. Should never be reached when the
    cascade driver wires up ``maybe_offline_retry`` correctly. We
    still raise :class:`StageThirteenStubRequired` here because
    silently stamping ``non_certified_stamp`` on a never-retried doc
    would conflate two different semantics ("we tried both lanes" vs
    "we never tried the offline lane"). See
    feedback_no_silent_fallbacks.md.

OCR-confusable repair-stats amendment (channel 3, SEMANTIK_OCR_CONFUSABLE_REPAIR)
---------------------------------------------------------------------------------

When the repair pass ran, the cascade driver applies
:func:`semantik_structure.theta.evaluator.apply_repair_stats` to the FINAL report
AFTER :func:`semantik_structure.theta.offline_retry.maybe_offline_retry` and BEFORE
:func:`decide_exit`. The amendment replaces the stub_v1 semantic-preservation
placeholder with the pass's ``repair_stats_score`` and re-derives the composite
+ ``MEANING_PRESERVATION_LOW`` floor. It is applied POST-retry deliberately: the
repair score informs the exit STAMP but does NOT gate the offline retry (the
retry stays stub-skipped, exactly today's ``_needs_retry`` semantics) — minimal
blast radius. Because the amended report is no longer ``theta_is_stubbed`` (its
method is ``ocr_repair_stats_v1``, not ``stub_v1``), :func:`decide_exit` needs
ZERO change here: it naturally takes the tau_confidence / tau_retry threshold
path instead of the unconditional SHIP_WITH_FLAG + THETA_UNVERIFIED_STUB bypass.
When theta is a REAL model, or the repair pass is off / produced no stats, the
report is untouched and this seam is byte-stable.

The :class:`StageThirteenStubRequired` exception class is retained
(deprecated) for callers that don't yet wire the retry orchestrator;
they will hit it on any fast-lane WCAG-failed input and can degrade
honestly. The legacy flag enums (``THETA_LOW_NO_RETRY``,
``OFFLINE_LANE_UNAVAILABLE_V1``) stay in :mod:`semantik_structure.theta.types`
for backward compatibility with old eval JSON schemas.
"""

from __future__ import annotations

import dataclasses

from .evaluator import theta_is_stubbed
from .types import (
    ConfidenceAction,
    ThetaFlag,
    ThetaReport,
    load_theta_config,
)

# The four action-changing "floor breach" flags (architecture.md §7). A
# breach forces SHIP_WITH_FLAG. The remaining ThetaFlags
# (TITLE_FABRICATED, TABLE_HEADERS_INFERRED, READING_ORDER_AT_RISK,
# MEANING_PRESERVATION_BORDERLINE) are documented in types.py as
# "Observational only — does NOT change the Stage 13 action"; they ride
# along on report.flags for the audit sidecar but must NOT trip a breach.
_FLOOR_FLAGS = frozenset(
    {
        ThetaFlag.BROKEN_REFS_PRESENT.value,
        ThetaFlag.GAP_FILL_REVIEW_RECOMMENDED.value,
        ThetaFlag.MEANING_PRESERVATION_LOW.value,
        ThetaFlag.COGNITIVE_LOAD_HIGH.value,
    }
)


class StageThirteenStubRequired(RuntimeError):
    """Raised when ``decide_exit`` is reached without the retry orchestrator wired.

    .. deprecated:: Stage 13 v1.1
        The offline-lane retry is now implemented in
        :func:`semantik_structure.theta.offline_retry.maybe_offline_retry`.
        Cascade drivers SHOULD call that orchestrator before
        :func:`decide_exit`; only the legacy "no retry wired" path
        triggers this exception now. It is retained so callers that
        haven't migrated still fail loudly rather than silently
        stamping a benign-looking action on a never-retried doc.

    Carries the partial :class:`ThetaReport` so callers can log
    diagnostics, plus the row that fired (``"retry_offline"`` or
    ``"offline_qwen_lane"``) and a human-readable reason. **Callers
    must not silently catch and coerce to a default action** — the
    point of the exception is to surface that the offline lane was
    never attempted. Catch and either (a) record as an honest
    failure and move on, or (b) propagate.
    """

    def __init__(self, report: ThetaReport, *, lane_required: str, reason: str):
        super().__init__(
            f"Stage 13 routes to '{lane_required}' lane (not implemented in v1): {reason}"
        )
        self.report = report
        self.lane_required = lane_required
        self.reason = reason


def decide_exit(report: ThetaReport) -> ThetaReport:
    """Apply the §7 decision table to populate ``report.action``.

    The returned report is a new frozen instance — the input is not
    mutated. Floors that fired in Stage 12 (already on
    ``report.flags``) drive the "floor breach" column of the table.

    Expects to be called AFTER
    :func:`semantik_structure.theta.offline_retry.maybe_offline_retry`,
    which decides whether to dispatch the offline retry and reconciles
    the two lane reports into a single FINAL report. The combinations
    that reach ``decide_exit`` in normal operation are:

      * fast | passed | theta >= TAU_THETA_RETRY   (no retry needed)
      * fast | passed | theta <  TAU_THETA_RETRY   (retry ran, fast won)
      * offline | passed | any theta                (retry ran, offline won)
      * offline | failed                            (both lanes tried)

    Defensive: if a fast | failed report arrives here, the cascade
    driver did NOT wire ``maybe_offline_retry`` first — silently
    stamping ``non_certified_stamp`` would conflate "both lanes tried"
    with "never tried offline lane", so we raise
    :class:`StageThirteenStubRequired`. Callers must wire the retry
    orchestrator. See feedback_no_silent_fallbacks.md.
    """
    flags: list = list(report.flags or [])
    floor_breach = bool(set(flags) & _FLOOR_FLAGS)  # only the 4 floor flags

    if report.wcag_status == "failed":
        if report.lane == "fast":
            # Defensive — the retry orchestrator should have been called
            # before us. If we reach here, the cascade driver did not
            # wire maybe_offline_retry. Refuse to silently stamp.
            raise StageThirteenStubRequired(
                report,
                lane_required="offline_qwen_lane",
                reason=(
                    "WCAG failed on fast lane and decide_exit was reached "
                    "without the offline-retry orchestrator being invoked first. "
                    "Wire semantik_structure.theta.offline_retry.maybe_offline_retry "
                    "into the cascade driver. Refusing to silently stamp "
                    "'non_certified_stamp' — that label must mean 'tried both "
                    "lanes and could not certify', not 'gave up before trying'."
                ),
            )
        # fail | offline → genuinely non-certifiable (we tried both lanes).
        return _replace(
            report,
            action=ConfidenceAction.NON_CERTIFIED_STAMP,
            flags=flags,
            theta_score=None,
        )

    # WCAG passed, but the semantic-preservation cross-encoder was
    # STUBBED (mode-collapsed / unavailable; DART_ALLOW_THETA_STUB=1).
    # The composite theta_score includes a flat 0.7 placeholder and is
    # therefore meaningless — it must NOT decide ship_with_confidence
    # vs ship_with_flag. Stamp SHIP_WITH_FLAG with the explicit
    # THETA_UNVERIFIED_STUB flag so a WCAG-clean doc (e.g. the 70B fast
    # lane) ships honestly without claiming a confidence the broken
    # model could not support. Applies to both lanes (a stubbed offline
    # lane is equally unverified). A genuine floor breach still rides
    # along on the flag list. Byte-stable when theta is not stubbed.
    if theta_is_stubbed(report):
        if ThetaFlag.THETA_UNVERIFIED_STUB.value not in flags:
            flags.append(ThetaFlag.THETA_UNVERIFIED_STUB.value)
        return _replace(
            report,
            action=ConfidenceAction.SHIP_WITH_FLAG,
            flags=flags,
        )

    # WCAG passed. Thresholds come from the same cached config load
    # that supplied the evaluator's composite weights (theta-config-2.0
    # calibration discipline) — not from import-time snapshots.
    cfg = load_theta_config()
    tau_confidence = cfg.tau_theta_ship
    tau_retry = cfg.tau_theta_retry
    theta = report.theta_score if report.theta_score is not None else 0.0

    if report.lane == "fast":
        if floor_breach:
            action = ConfidenceAction.SHIP_WITH_FLAG
        elif theta >= tau_confidence:
            action = ConfidenceAction.SHIP_WITH_CONFIDENCE
        elif theta >= tau_retry:
            action = ConfidenceAction.SHIP_WITH_FLAG
            if ThetaFlag.MEANING_PRESERVATION_BORDERLINE.value not in flags:
                flags.append(ThetaFlag.MEANING_PRESERVATION_BORDERLINE.value)
        else:  # theta < tau_retry
            # Reachable when the retry orchestrator ran the offline
            # lane and decided the offline result did NOT beat fast by
            # DELTA_THETA_IMPROVE — so we keep the fast (low-theta)
            # report. Surface as ship_with_flag + borderline, the same
            # treatment used in the borderline band; the difference is
            # ``retry_history`` is non-empty.
            action = ConfidenceAction.SHIP_WITH_FLAG
            if ThetaFlag.MEANING_PRESERVATION_BORDERLINE.value not in flags:
                flags.append(ThetaFlag.MEANING_PRESERVATION_BORDERLINE.value)
    else:
        # offline lane.
        if floor_breach:
            action = ConfidenceAction.SHIP_WITH_FLAG
        elif theta >= tau_confidence:
            action = ConfidenceAction.SHIP_WITH_CONFIDENCE
        else:
            action = ConfidenceAction.SHIP_WITH_FLAG

    return _replace(report, action=action, flags=flags)


def _replace(report: ThetaReport, **changes) -> ThetaReport:
    """:func:`dataclasses.replace` shim. ``ThetaReport`` is frozen."""
    return dataclasses.replace(report, **changes)


__all__ = ["decide_exit", "StageThirteenStubRequired"]
