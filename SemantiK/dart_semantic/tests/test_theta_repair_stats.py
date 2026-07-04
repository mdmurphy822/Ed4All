"""Unit tests — theta channel-3 repair-stats amendment (apply_repair_stats).

When the OCR-confusable repair pass ran on a STUBBED theta box, the proxy
``repair_stats_score`` replaces the flat-0.7 ``stub_v1`` semantic-preservation
placeholder, the composite is recomputed with the config weights, and the
amended report is no longer ``theta_is_stubbed`` — so ``decide_exit`` takes the
real tau threshold path instead of the unconditional SHIP_WITH_FLAG +
THETA_UNVERIFIED_STUB bypass. Non-stubbed / no-stats reports are byte-stable.

Extends the ``test_theta_stub_bypass.py`` idiom.
"""

from __future__ import annotations

from dart_semantic.theta.evaluator import (
    OCR_REPAIR_STATS_METHOD,
    STUB_SEMANTIC_METHOD,
    apply_repair_stats,
    theta_is_stubbed,
)
from dart_semantic.theta.exits import decide_exit
from dart_semantic.theta.types import (
    ConfidenceAction,
    ThetaDimension,
    ThetaFlag,
    ThetaReport,
    load_theta_config,
    _WEIGHT_KEYS,
)

_SEM = ThetaDimension.SEMANTIC_PRESERVATION.value


def _full_dims(*, sem_score: float, sem_method: str, other_score: float) -> dict:
    """All 8 dimensions (so the composite recompute is realistic)."""
    dims = {}
    for name in _WEIGHT_KEYS:
        if name == _SEM:
            dims[name] = {
                "dimension": name,
                "score": sem_score,
                "breakdown": {"method": sem_method},
            }
        else:
            dims[name] = {
                "dimension": name,
                "score": other_score,
                "breakdown": {"method": "deterministic"},
            }
    return dims


def _report(*, sem_method=STUB_SEMANTIC_METHOD, sem_score=0.7, other=0.95,
            lane="fast", wcag="passed", flags=None) -> ThetaReport:
    return ThetaReport(
        schema_version="theta/1.0",
        wcag_status=wcag,
        lane=lane,
        theta_score=0.7,
        theta_version="test",
        dimensions=_full_dims(
            sem_score=sem_score, sem_method=sem_method, other_score=other
        ),
        flags=list(flags or []),
        action=ConfidenceAction.SHIP_WITH_CONFIDENCE,
    )


def _stats(score: float, *, density=0.05, flagged=3, accepted=4) -> dict:
    return {
        "repair_stats_score": score,
        "unrepairable_defect_density": density,
        "flagged_blocks": flagged,
        "accepted": accepted,
    }


# ---------------------------------------------------------------------------
# Amendment replaces the stub dimension + recomputes.
# ---------------------------------------------------------------------------


def test_amendment_replaces_stub_and_recomputes():
    report = _report(sem_method=STUB_SEMANTIC_METHOD, other=0.95)
    assert theta_is_stubbed(report) is True

    amended = apply_repair_stats(report, _stats(0.90))
    # Dimension replaced with the repair-stats method + score.
    sem = amended.dimensions[_SEM]
    assert sem["score"] == 0.90
    assert sem["breakdown"]["method"] == OCR_REPAIR_STATS_METHOD
    # No longer stubbed → the exit-decider will take the threshold path.
    assert theta_is_stubbed(amended) is False

    # Composite recomputed with the config weights (all dims 0.95 except sem
    # 0.90 → a weighted value in that band).
    cfg = load_theta_config()
    expected = round(
        sum(
            cfg.weights[n] * (0.90 if n == _SEM else 0.95)
            for n in _WEIGHT_KEYS
        ),
        4,
    )
    assert amended.theta_score == expected


def test_amendment_below_floor_sets_meaning_preservation_low():
    report = _report(sem_method=STUB_SEMANTIC_METHOD, other=0.95)
    cfg = load_theta_config()
    below = max(0.0, cfg.floor_semantic_preservation - 0.2)
    amended = apply_repair_stats(report, _stats(below))
    assert ThetaFlag.MEANING_PRESERVATION_LOW.value in amended.flags


def test_amendment_above_floor_clears_meaning_preservation_low():
    report = _report(
        sem_method=STUB_SEMANTIC_METHOD, other=0.95,
        flags=[ThetaFlag.MEANING_PRESERVATION_LOW.value],
    )
    cfg = load_theta_config()
    above = min(1.0, cfg.floor_semantic_preservation + 0.2)
    amended = apply_repair_stats(report, _stats(above))
    assert ThetaFlag.MEANING_PRESERVATION_LOW.value not in amended.flags


# ---------------------------------------------------------------------------
# Byte-stability — no stats / non-stubbed / bad stats.
# ---------------------------------------------------------------------------


def test_no_stats_returns_unchanged():
    report = _report(sem_method=STUB_SEMANTIC_METHOD)
    assert apply_repair_stats(report, None) is report
    assert apply_repair_stats(report, {}) is report
    assert apply_repair_stats(report, {"repair_stats_score": None}) is report


def test_real_model_report_untouched():
    report = _report(sem_method="cross_encoder_v8")
    assert theta_is_stubbed(report) is False
    amended = apply_repair_stats(report, _stats(0.90))
    assert amended is report  # real model never overridden


# ---------------------------------------------------------------------------
# decide_exit on an amended report takes the threshold path.
# ---------------------------------------------------------------------------


def test_decide_exit_amended_takes_threshold_path():
    report = _report(sem_method=STUB_SEMANTIC_METHOD, other=0.95)
    amended = apply_repair_stats(report, _stats(0.95))
    decided = decide_exit(amended)
    # A high recomputed composite ships with confidence — NOT the stub bypass.
    assert ThetaFlag.THETA_UNVERIFIED_STUB.value not in decided.flags
    assert decided.action in {
        ConfidenceAction.SHIP_WITH_CONFIDENCE,
        ConfidenceAction.SHIP_WITH_FLAG,
    }


def test_decide_exit_stubbed_unamended_keeps_bypass():
    # Sanity: an UN-amended stubbed report still takes the stub bypass.
    report = _report(sem_method=STUB_SEMANTIC_METHOD)
    decided = decide_exit(report)
    assert ThetaFlag.THETA_UNVERIFIED_STUB.value in decided.flags
    assert decided.action == ConfidenceAction.SHIP_WITH_FLAG
