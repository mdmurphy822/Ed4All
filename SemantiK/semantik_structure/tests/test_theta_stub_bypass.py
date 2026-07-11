"""Unit tests — theta-stub bypass at Stage 13 (offline retry + exit).

When the semantic-preservation cross-encoder is mode-collapsed / missing
and the run substitutes the 0.7 ``stub_v1`` placeholder
(``DART_ALLOW_THETA_STUB=1``), the composite ``theta_score`` is
meaningless. These tests pin the bypass:

  * ``theta_is_stubbed`` keys off the report's own
    ``dimensions["semantic_preservation"]["breakdown"]["method"]``.
  * ``offline_retry._needs_retry`` does NOT retry a stubbed + WCAG-passed
    fast lane (no silently discarding the 70B fast lane), but STILL
    retries a stubbed + WCAG-FAILED fast lane.
  * ``exits.decide_exit`` stamps SHIP_WITH_FLAG + THETA_UNVERIFIED_STUB
    for a stubbed + WCAG-passed report (any lane).
  * Real-model reports (non-stub breakdown) are byte-stable — the
    threshold-driven path is unchanged.
"""

from __future__ import annotations

from semantik_structure.theta.evaluator import (
    STUB_SEMANTIC_METHOD,
    theta_is_stubbed,
)
from semantik_structure.theta.exits import decide_exit
from semantik_structure.theta.offline_retry import _needs_retry, maybe_offline_retry
from semantik_structure.theta.types import (
    TAU_THETA_RETRY,
    ConfidenceAction,
    ThetaDimension,
    ThetaFlag,
    ThetaReport,
)

_SEM = ThetaDimension.SEMANTIC_PRESERVATION.value


def _dims(method: str) -> dict:
    """Minimal dimensions dict carrying a semantic_preservation breakdown."""
    return {
        _SEM: {
            "dimension": _SEM,
            "score": 0.7,
            "breakdown": {"method": method},
        }
    }


def _report(
    *,
    lane: str = "fast",
    wcag: str = "passed",
    theta: float | None = 0.7,
    method: str = STUB_SEMANTIC_METHOD,
    flags: list | None = None,
) -> ThetaReport:
    return ThetaReport(
        schema_version="theta/1.0",
        wcag_status=wcag,
        lane=lane,
        theta_score=theta,
        theta_version="test",
        dimensions=_dims(method),
        flags=list(flags or []),
        action=ConfidenceAction.SHIP_WITH_CONFIDENCE,
    )


# ---------------------------------------------------------------------------
# theta_is_stubbed detector
# ---------------------------------------------------------------------------


def test_detector_true_on_stub_breakdown():
    assert theta_is_stubbed(_report(method=STUB_SEMANTIC_METHOD)) is True


def test_detector_false_on_real_breakdown():
    assert theta_is_stubbed(_report(method="cross_encoder_v8")) is False


def test_detector_false_on_none():
    assert theta_is_stubbed(None) is False


def test_detector_false_on_empty_dimensions():
    rep = ThetaReport(
        schema_version="theta/1.0",
        wcag_status="failed",
        lane="fast",
        theta_score=None,
        theta_version="test",
        dimensions={},
        flags=[],
        action=ConfidenceAction.NON_CERTIFIED_STAMP,
    )
    assert theta_is_stubbed(rep) is False


# ---------------------------------------------------------------------------
# _needs_retry bypass
# ---------------------------------------------------------------------------


def test_stubbed_passed_low_theta_does_not_retry():
    # theta below the retry threshold, but stubbed -> NO retry (would
    # otherwise discard the fast lane for the local lane).
    rep = _report(wcag="passed", theta=0.50, method=STUB_SEMANTIC_METHOD)
    assert rep.theta_score < TAU_THETA_RETRY
    assert _needs_retry(rep) is False


def test_stubbed_failed_still_retries():
    # A REAL WCAG failure still routes to the offline lane even when
    # theta is stubbed — the WCAG hard gate is independent of theta.
    rep = _report(wcag="failed", theta=None, method=STUB_SEMANTIC_METHOD)
    assert _needs_retry(rep) is True


def test_real_low_theta_still_retries():
    # Byte-stable: a real-model low-theta passed report retries as before.
    rep = _report(wcag="passed", theta=0.50, method="cross_encoder_v8")
    assert _needs_retry(rep) is True


def test_real_high_theta_no_retry():
    rep = _report(wcag="passed", theta=0.90, method="cross_encoder_v8")
    assert _needs_retry(rep) is False


def test_maybe_offline_retry_keeps_stubbed_fast_lane():
    """End-to-end: stubbed + passed fast lane is returned verbatim (no
    offline lane dispatched)."""
    fast = _report(wcag="passed", theta=0.50, method=STUB_SEMANTIC_METHOD)
    dispatched: list[str] = []

    def _run_lane(lane: str) -> ThetaReport:
        dispatched.append(lane)
        return _report(lane="offline", wcag="passed", theta=0.95, method="cross_encoder_v8")

    out = maybe_offline_retry(fast, run_lane=_run_lane)
    assert dispatched == []  # offline lane NEVER dispatched
    assert out is fast


# ---------------------------------------------------------------------------
# decide_exit bypass
# ---------------------------------------------------------------------------


def test_decide_exit_stubbed_passed_ships_with_flag():
    rep = _report(lane="fast", wcag="passed", theta=0.50, method=STUB_SEMANTIC_METHOD)
    out = decide_exit(rep)
    assert out.action == ConfidenceAction.SHIP_WITH_FLAG
    assert ThetaFlag.THETA_UNVERIFIED_STUB.value in out.flags


def test_decide_exit_stubbed_offline_lane_ships_with_flag():
    rep = _report(lane="offline", wcag="passed", theta=0.50, method=STUB_SEMANTIC_METHOD)
    out = decide_exit(rep)
    assert out.action == ConfidenceAction.SHIP_WITH_FLAG
    assert ThetaFlag.THETA_UNVERIFIED_STUB.value in out.flags


def test_decide_exit_real_high_theta_ships_with_confidence():
    # Byte-stable: real-model high theta still ships with confidence
    # (no THETA_UNVERIFIED_STUB flag).
    rep = _report(lane="fast", wcag="passed", theta=0.95, method="cross_encoder_v8")
    out = decide_exit(rep)
    assert out.action == ConfidenceAction.SHIP_WITH_CONFIDENCE
    assert ThetaFlag.THETA_UNVERIFIED_STUB.value not in out.flags


def test_decide_exit_failed_offline_still_non_certified():
    # A genuine both-lanes-failed report is unaffected by the stub bypass.
    rep = _report(lane="offline", wcag="failed", theta=None, method=STUB_SEMANTIC_METHOD)
    out = decide_exit(rep)
    assert out.action == ConfidenceAction.NON_CERTIFIED_STAMP
