"""Theta (post-WCAG meaning-preservation) typed report.

The shape of this module is locked by Plans/03_theta_investigation.md §9.

As of theta-config-2.0 (Plan 12 A3) the composite weights, exit
thresholds (TAU_*), and per-dimension floors are CALIBRATED values
loaded from ``theta/config.yaml`` via :func:`load_theta_config` — the
module-level constants below are import-time snapshots kept for
backward compatibility with existing importers. New code should call
:func:`load_theta_config` so monkeypatched/alternate configs take
effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ThetaDimension(str, Enum):
    SEMANTIC_PRESERVATION = "semantic_preservation"
    STRUCTURAL_COHERENCE = "structural_coherence"
    NAVIGATION_CLARITY = "navigation_clarity"
    CONTEXT_CONTINUITY = "context_continuity"
    REFERENCE_INTEGRITY = "reference_integrity"
    COGNITIVE_LOAD_REDUCTION = "cognitive_load_reduction"
    FRAGMENTATION_PENALTY = "fragmentation_penalty"
    HALLUCINATED_STRUCTURE_PENALTY = "hallucinated_structure_penalty"


class CognitiveLoadRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConfidenceAction(str, Enum):
    SHIP_WITH_CONFIDENCE = "ship_with_confidence"
    SHIP_WITH_FLAG = "ship_with_flag"
    RETRY_OFFLINE = "retry_offline"
    NON_CERTIFIED = "non_certified"
    # Stage 13 — full §7 vocabulary. ``NON_CERTIFIED`` is retained as
    # a legacy alias for ``NON_CERTIFIED_STAMP`` to keep existing
    # test_theta_types fixtures working.
    OFFLINE_QWEN_LANE = "offline_qwen_lane"
    NON_CERTIFIED_STAMP = "non_certified_stamp"


class ThetaFlag(str, Enum):
    BROKEN_REFS_PRESENT = "broken_refs_present"
    GAP_FILL_REVIEW_RECOMMENDED = "gap_fill_review_recommended"
    MEANING_PRESERVATION_LOW = "meaning_preservation_low"
    MEANING_PRESERVATION_BORDERLINE = "meaning_preservation_borderline"
    COGNITIVE_LOAD_HIGH = "cognitive_load_high"
    # Title is a hardcoded fallback ("Untitled document") — no
    # doc_role:title source AND no h1 in raw heading inputs AND no Qwen
    # gap-fill survivor. Observational only: surfaced for downstream
    # auditing per the no-silent-fallbacks policy; does not change the
    # Stage 13 action.
    TITLE_FABRICATED = "title_fabricated"
    # At least one ``<table>`` in the assembled doc carries
    # ``data-dart-cell-roles="qwen-inferred"`` — the BERT-TableSpecialist
    # signal was missing for that table region (didn't run, didn't cover
    # the candidate, or partial failure) so Qwen-Table inferred the
    # header roles on its own. Stage 7's per-region gate does not
    # structurally validate ``<th scope=...>`` presence (only HTML5
    # well-formedness, text-preservation [skipped for tables], MathML,
    # and axe-core under WCAG 2.2 AA), so a missing-header table can
    # ship to assembled.html without failing the gate. Observational
    # only — does not change the Stage 13 action; surfaced so eval JSON
    # / sidecar dashboards can see when table header semantics were
    # best-effort. See feedback_no_silent_fallbacks.md.
    TABLE_HEADERS_INFERRED = "table_headers_inferred"
    # At least one region fell back to its deterministic per-kind
    # emission (all K Stage-6 candidates died at the Stage-7 gate) while
    # being, or sitting adjacent to, a complex region (table / figure /
    # math) in reading order. Degraded or missing context around a
    # complex region puts SC 1.3.2 (Meaningful Sequence) at risk in a
    # way no post-hoc gate can verify. Observational only — does not
    # change the Stage 13 action; surfaced for the conformance-audit
    # sidecar (Plan 12 B3).
    READING_ORDER_AT_RISK = "reading_order_at_risk"
    # v1-stub flags — these mark architecture-spec actions that are not
    # yet wired (offline-Qwen lane, theta retry). The flag carries the
    # degradation reason on the sidecar.
    THETA_LOW_NO_RETRY = "theta_low_no_retry"
    OFFLINE_LANE_UNAVAILABLE_V1 = "offline_lane_unavailable_v1"
    # The semantic-preservation cross-encoder could not be loaded
    # (mode-collapsed / missing) and the 0.7 stub placeholder was
    # substituted (DART_ALLOW_THETA_STUB=1). The composite theta_score
    # is therefore UNVERIFIED — Stage 13 ships the doc WITH this flag
    # rather than letting the meaningless score gate ship/retry. Honest
    # degradation per feedback_no_silent_fallbacks: a WCAG-clean doc is
    # not held hostage by a broken theta model, but the flag records
    # that meaning-preservation was not actually measured.
    THETA_UNVERIFIED_STUB = "theta_unverified_stub"


# ---------------------------------------------------------------------------
# Config loading (theta/config.yaml — single source of truth since 2.0).
# ---------------------------------------------------------------------------

#: Canonical on-disk location of the versioned theta config.
THETA_CONFIG_PATH: Path = Path(__file__).resolve().parent / "config.yaml"

#: All 8 dimension names the weights map must cover, exactly.
_WEIGHT_KEYS: tuple = (
    "semantic_preservation",
    "structural_coherence",
    "navigation_clarity",
    "context_continuity",
    "reference_integrity",
    "cognitive_load_reduction",
    "fragmentation_penalty",
    "hallucinated_structure_penalty",
)


class ThetaConfigError(RuntimeError):
    """Raised when theta/config.yaml is missing or structurally invalid.

    Deliberately loud (feedback_no_silent_fallbacks): the composite
    weights and exit thresholds drive ship/retry decisions — silently
    substituting defaults would un-calibrate the number we ship.
    """


# Cache keyed on resolved path so tests can load alternates without
# clobbering the production config entry.
_config_cache: dict = {}


def load_theta_config(path: Path | str | None = None) -> "ThetaConfig":
    """Load + validate the versioned theta config (cached per path).

    Raises :class:`ThetaConfigError` on a missing file, unparseable
    YAML, missing keys, weights that don't cover exactly the 8
    dimensions or don't sum to 1.0, or thresholds outside (0, 1) /
    mis-ordered (retry must be below confidence). There is NO fallback
    to built-in defaults — see :class:`ThetaConfigError`.
    """
    resolved = Path(path) if path is not None else THETA_CONFIG_PATH
    key = str(resolved)
    if key in _config_cache:
        return _config_cache[key]

    if not resolved.exists():
        raise ThetaConfigError(f"theta config not found at {resolved}")

    import yaml  # deferred: keep module import light for non-config users

    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 — re-raise typed, never default
        raise ThetaConfigError(f"theta config at {resolved} is unparseable: {exc}") from exc
    if not isinstance(raw, dict):
        raise ThetaConfigError(f"theta config at {resolved} is not a mapping")

    required = (
        "version",
        "weights",
        "tau_theta_ship",
        "tau_theta_retry",
        "delta_theta_improve",
        "floor_reference_integrity",
        "floor_hallucinated_structure",
        "floor_semantic_preservation",
    )
    missing = [k for k in required if k not in raw]
    if missing:
        raise ThetaConfigError(f"theta config at {resolved} missing keys: {missing}")

    weights = raw["weights"]
    if not isinstance(weights, dict) or set(weights) != set(_WEIGHT_KEYS):
        raise ThetaConfigError(
            f"theta config weights must cover exactly the 8 dimensions "
            f"{sorted(_WEIGHT_KEYS)}; got {sorted(weights) if isinstance(weights, dict) else weights!r}"
        )
    weights = {k: float(v) for k, v in weights.items()}
    if any(v < 0.0 for v in weights.values()):
        raise ThetaConfigError("theta config weights must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-4:
        raise ThetaConfigError(f"theta config weights must sum to 1.0 (got {total:.6f})")

    tau_ship = float(raw["tau_theta_ship"])
    tau_retry = float(raw["tau_theta_retry"])
    if not (0.0 < tau_retry < tau_ship < 1.0):
        raise ThetaConfigError(
            f"theta thresholds mis-ordered: need 0 < tau_theta_retry "
            f"({tau_retry}) < tau_theta_ship ({tau_ship}) < 1"
        )

    floors = {
        "floor_reference_integrity": float(raw["floor_reference_integrity"]),
        "floor_hallucinated_structure": float(raw["floor_hallucinated_structure"]),
        "floor_semantic_preservation": float(raw["floor_semantic_preservation"]),
    }
    for name, val in floors.items():
        if not (0.0 <= val <= 1.0):
            raise ThetaConfigError(f"theta config {name} out of [0, 1]: {val}")

    calibration = raw.get("calibration")
    if calibration is not None and not isinstance(calibration, dict):
        raise ThetaConfigError("theta config calibration block must be a mapping")

    cfg = ThetaConfig(
        version=str(raw["version"]),
        weights=weights,
        tau_theta_ship=tau_ship,
        tau_theta_retry=tau_retry,
        delta_theta_improve=float(raw["delta_theta_improve"]),
        floor_reference_integrity=floors["floor_reference_integrity"],
        floor_hallucinated_structure=floors["floor_hallucinated_structure"],
        floor_semantic_preservation=floors["floor_semantic_preservation"],
        semantic_model_version=str(raw.get("semantic_model_version", "")),
        calibration=calibration,
    )
    _config_cache[key] = cfg
    return cfg


def _reset_theta_config_cache() -> None:
    """Test hook — clears the cache so the next load re-reads the YAML."""
    _config_cache.clear()


def floors_from_config(cfg: "ThetaConfig") -> dict:
    """Per-dimension floor map (dimension name -> floor) from a config."""
    return {
        "reference_integrity": cfg.floor_reference_integrity,
        "hallucinated_structure_penalty": cfg.floor_hallucinated_structure,
        "semantic_preservation": cfg.floor_semantic_preservation,
    }


@dataclass(frozen=True)
class SectionScore:
    section_id: str
    score: float


@dataclass(frozen=True)
class DimensionScore:
    dimension: ThetaDimension
    score: float  # always [0.0, 1.0]
    breakdown: dict = field(default_factory=dict)  # dimension-specific fields


@dataclass(frozen=True)
class ThetaConfig:
    """Versioned thresholds + weights. Loaded from theta/config.yaml.

    Locked at model-release time. See Plans/03_theta_investigation.md §7.3.
    """

    version: str  # e.g. "theta-config-2.0"
    weights: dict  # {dimension name: float}; sum to 1.0
    tau_theta_ship: float = 0.80
    tau_theta_retry: float = 0.70
    delta_theta_improve: float = 0.05
    floor_reference_integrity: float = 0.80
    floor_hallucinated_structure: float = 0.85
    floor_semantic_preservation: float = 0.65
    semantic_model_version: str = "theta-semantic-deberta-1.0"
    #: Calibration provenance (Plan 12 A3): {report, date, n_docs, ...}.
    #: None only for legacy/uncalibrated configs.
    calibration: dict | None = None


@dataclass(frozen=True)
class ThetaReport:
    """The canonical theta report. JSON shape in Plans/03 §5."""

    schema_version: str  # e.g. "theta/1.0"
    wcag_status: str  # "passed" | "failed"
    lane: str  # "fast" | "offline"
    theta_score: float | None
    theta_version: str
    dimensions: dict  # {ThetaDimension: DimensionScore}
    flags: list  # list[ThetaFlag]
    action: ConfidenceAction
    retry_history: list = field(default_factory=list)  # list[ThetaReport]


# ---------------------------------------------------------------------------
# Stage 12 / 13 thresholds (architecture.md §6.2 / §7).
#
# Import-time snapshots of theta/config.yaml, kept for backward
# compatibility with existing importers (exits, offline_retry,
# conformance_audit, eval scripts). New code should prefer
# load_theta_config() so alternate/monkeypatched configs apply. This
# block sits below ThetaConfig because load_theta_config constructs one.
# ---------------------------------------------------------------------------

_cfg = load_theta_config()

#: Score at or above which a passing-WCAG fast-lane doc ships with
#: confidence (architecture.md §7). Calibrated — see config.yaml
#: ``calibration:`` block for provenance.
TAU_THETA_CONFIDENCE: float = _cfg.tau_theta_ship

#: Score below which the architecture calls for a single offline-lane
#: retry (see :func:`semantik_structure.theta.exits.decide_exit`). Calibrated.
TAU_THETA_RETRY: float = _cfg.tau_theta_retry

#: Minimum theta improvement required to prefer the offline-lane retry
#: output over the fast-lane output (architecture.md §6.3).
DELTA_THETA_IMPROVE: float = _cfg.delta_theta_improve

#: Per-dimension floor map. Breaching any of these triggers a flag
#: independently of the composite (architecture.md §6.2). Calibrated.
DIM_FLOORS: dict = floors_from_config(_cfg)

del _cfg
