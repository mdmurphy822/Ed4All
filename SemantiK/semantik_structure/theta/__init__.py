"""Theta — post-WCAG meaning-preservation diagnostic (Phase 11).

Theta is a deterministic composite over per-dimension scorers, with a
single narrow learned cross-encoder for semantic preservation. It scores
*absolute quality* of the chosen DocCandidate, separately from Stage
11's "pick the best" objective.

Theta NEVER overrides the WCAG hard gate; it can only recommend retry
or set a `ThetaFlag` on the sidecar. See
Plans/03_theta_investigation.md §10.

Phase 0 landed the typed report contract + the YAML config skeleton.
Stage 12 (``evaluate``) and Stage 13 (``decide_exit``) land here.
"""

from .evaluator import (
    OCR_REPAIR_STATS_METHOD,
    STUB_SEMANTIC_METHOD,
    apply_repair_stats,
    evaluate,
    theta_is_stubbed,
)
from .exits import StageThirteenStubRequired, decide_exit
from .offline_retry import maybe_offline_retry
from .types import (
    DELTA_THETA_IMPROVE,
    DIM_FLOORS,
    TAU_THETA_CONFIDENCE,
    TAU_THETA_RETRY,
    CognitiveLoadRisk,
    ConfidenceAction,
    DimensionScore,
    SectionScore,
    ThetaConfig,
    ThetaDimension,
    ThetaFlag,
    ThetaReport,
)

__all__ = [
    "CognitiveLoadRisk",
    "ConfidenceAction",
    "DELTA_THETA_IMPROVE",
    "DIM_FLOORS",
    "DimensionScore",
    "OCR_REPAIR_STATS_METHOD",
    "SectionScore",
    "STUB_SEMANTIC_METHOD",
    "apply_repair_stats",
    "TAU_THETA_CONFIDENCE",
    "TAU_THETA_RETRY",
    "ThetaConfig",
    "ThetaDimension",
    "ThetaFlag",
    "ThetaReport",
    "StageThirteenStubRequired",
    "decide_exit",
    "evaluate",
    "maybe_offline_retry",
    "theta_is_stubbed",
]
