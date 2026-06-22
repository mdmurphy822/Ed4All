"""Stage 13 — offline-lane retry orchestration.

Implements the architecture.md §6.3 / §7 retry policy. The fast lane's
``ThetaReport`` is fed in along with a ``run_lane`` callable; this
module decides whether a retry is required and, if so, dispatches it
and reconciles the two reports into a single FINAL ``ThetaReport`` for
:func:`dart_semantic.theta.exits.decide_exit` to stamp.

The orchestrator owns NONE of the cascade machinery — the caller
(``dart_semantic/cascade.py:run_full_cascade`` or any future
production driver) injects ``run_lane: Callable[[str], ThetaReport]``
which knows how to re-run Stages 6-12 against a given lane label
("fast" or "offline"). This keeps the orchestrator pure and testable
without HtmlValidator / PDF / pipeline imports.

Reconciliation rules (architecture.md §6.3 / §7)
------------------------------------------------

* fast.wcag_status == "passed" AND fast.theta_score >= TAU_THETA_RETRY
    → no retry. Return fast verbatim.
* fast.wcag_status == "passed" AND fast.theta_score <  TAU_THETA_RETRY
    → retry on offline lane.
        - offline.theta_score >= fast.theta_score + DELTA_THETA_IMPROVE
            → keep offline.
        - else
            → keep fast (retry didn't clear the improvement bar).
* fast.wcag_status == "failed"
    → retry on offline lane.
        - offline.wcag_status == "passed"  → keep offline.
        - else                              → keep offline (still
          ``wcag=failed``, lane="offline") so a downstream
          ``non_certified_stamp`` is honest: BOTH lanes were tried.

The chosen report carries the not-chosen one on ``retry_history``
for the eval driver to inspect.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable

from .types import (
    DELTA_THETA_IMPROVE,
    TAU_THETA_RETRY,
    ThetaReport,
)


def _needs_retry(report: ThetaReport) -> bool:
    """True iff the architecture rows route this report to the offline lane.

    * ``wcag_status == "failed"`` on the fast lane: always retry.
    * ``wcag_status == "passed"`` on the fast lane with theta below the
      retry threshold: retry.
    * Any report already on the offline lane: NEVER retry (single-shot
      offline lane is a hard architecture rule — see §7).
    """
    if report.lane != "fast":
        return False
    if report.wcag_status == "failed":
        return True
    theta = report.theta_score if report.theta_score is not None else 0.0
    return theta < TAU_THETA_RETRY


def _reconcile(fast: ThetaReport, offline: ThetaReport) -> ThetaReport:
    """Pick the keep-report per the rules in the module docstring.

    Always attaches the not-chosen report to ``retry_history`` so
    downstream consumers (eval JSON, dashboards) can see both attempts.
    """
    fast_passed = fast.wcag_status == "passed"
    offline_passed = offline.wcag_status == "passed"

    # WCAG-failed branch (architecture row "fail | fast → offline_qwen_lane").
    if not fast_passed:
        # Whether or not offline cleared WCAG, we KEEP the offline report
        # so a NON_CERTIFIED_STAMP downstream is honest about having tried
        # both lanes. If offline passed, even better — Stage 13 will then
        # route "pass | offline" through the normal exit table.
        return dataclasses.replace(offline, retry_history=[fast])

    # WCAG-passed branch (architecture row "pass | fast | theta < 0.70").
    fast_theta = fast.theta_score if fast.theta_score is not None else 0.0
    offline_theta = offline.theta_score if offline.theta_score is not None else 0.0

    # Offline must clear WCAG (it might have regressed) AND beat fast by delta.
    if offline_passed and (offline_theta >= fast_theta + DELTA_THETA_IMPROVE):
        return dataclasses.replace(offline, retry_history=[fast])

    # Otherwise keep fast — the retry didn't clear the improvement bar
    # (or regressed WCAG). Attach offline to history so the eval driver
    # can see what the offline attempt produced.
    return dataclasses.replace(fast, retry_history=[offline])


def maybe_offline_retry(
    fast_report: ThetaReport,
    *,
    run_lane: Callable[[str], ThetaReport],
) -> ThetaReport:
    """Decide if retry is needed; if yes, run offline lane and reconcile.

    Parameters
    ----------
    fast_report
        The fast-lane :class:`ThetaReport` produced by Stages 6-12.
    run_lane
        Callable that re-runs Stages 6-12 for a given lane label and
        returns a fresh :class:`ThetaReport`. The cascade driver
        injects this so the orchestrator stays pure (no HtmlValidator
        / PDF / pipeline imports).

    Returns
    -------
    ThetaReport
        The FINAL report — caller passes to
        :func:`dart_semantic.theta.exits.decide_exit`. When no retry
        was needed, this is ``fast_report`` verbatim. When retry
        fired, ``retry_history`` carries the not-chosen report.
    """
    if not _needs_retry(fast_report):
        return fast_report
    offline_report = run_lane("offline")
    return _reconcile(fast_report, offline_report)


__all__ = ["maybe_offline_retry"]
