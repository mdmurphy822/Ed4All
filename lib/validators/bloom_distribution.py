"""Course-level Bloom-distribution-vs-target-curve gate.

Ed4All emits a PER-PAGE ``BloomDistribution`` (``_build_bloom_distribution`` +
JSON-LD ``$defs.BloomDistribution``) but NO validator aggregates the COURSE-level
objective Bloom mix and compares it to a target curve. The four existing bloom
validators all do per-objective / per-block alignment (verb<->level, BERT
disagreement, structural floors, per-type ceiling) — none audits the
DISTRIBUTION. A recall-only course (every objective ``remember``/``understand``)
or a top-heavy course (mostly ``evaluate``/``create``) is a genuinely
unvalidated pedagogical defect.

This validator COMPOSES (never re-scores) a course-level Bloom mix by counting
the declared Bloom level of every resolved synthesized objective onto the
canonical six-level :data:`lib.ontology.bloom.BLOOM_LEVELS` axis, then compares
the observed per-level shares against a checked-in GENERIC target curve
(``schemas/taxonomies/bloom_target_distribution.json``, operator-overridable via
``ED4ALL_BLOOM_DISTRIBUTION_TARGET``).

Primary signal = synthesized-objectives Bloom level (the course's INTENDED
cognitive demand). The rewrite-tier blocks' ``target_bloom`` is a SECONDARY
corroborating signal surfaced in metadata only — it never drives the gate.

Anti-fabrication
----------------
Counts only an objective's ALREADY-DECLARED Bloom signal — an explicit
``bloom_level``/``bloomLevel`` field, or the canonical level of its declared
``bloom_verb`` (a deterministic lookup of a real field, NOT imputation). An
objective that declares NEITHER is SKIPPED, never imputed (the validator never
runs free-text Bloom detection over the statement prose). It never invents a
level, never adds an objective, and graceful-skips (``passed=True`` + a
structured skip code) when no objectives resolve.

Issue codes (all WARNING; disjoint from bloom_type_range / bloom_classifier_*)
-----------------------------------------------------------------------------
  * ``BLOOM_DISTRIBUTION_DISABLED`` (info) — flag off; byte-stable no-op.
  * ``BLOOM_DISTRIBUTION_NO_OBJECTIVES`` (info) — no resolvable Bloom signal;
    structured skip.
  * ``BLOOM_DISTRIBUTION_SMALL_N`` (info) — fewer than the small-N floor of
    objectives; share-based OFF_TARGET / RECALL_HEAVY / TOP_HEAVY are SKIPPED to
    avoid noise on tiny courses (NO_HIGHER_ORDER still fires — a recall-only
    course is a defect at any size).
  * ``BLOOM_DISTRIBUTION_NO_HIGHER_ORDER`` — zero apply+ objectives (recall-only
    course).
  * ``BLOOM_DISTRIBUTION_RECALL_HEAVY`` — remember+understand share above the
    curve's ``recall_ceiling``.
  * ``BLOOM_DISTRIBUTION_TOP_HEAVY`` — evaluate+create share above the curve's
    ``top_heavy_ceiling``.
  * ``BLOOM_DISTRIBUTION_OFF_TARGET`` — summed absolute per-level share deviation
    (L1) from the target curve above ``ED4ALL_BLOOM_DISTRIBUTION_TOLERANCE``.

Posture
-------
Warning-day-1 (``passed=True`` unconditionally) with a ``# TODO(calibration)``
deferred critical-flip — the generic curve is an opinion, not a measured
constant; do NOT flip critical until scripts/calibration_harness.py confirms the
FP rate on >=2 corpora. Byte-stable no-op when ``ED4ALL_BLOOM_DISTRIBUTION`` is
unset (returns a ``BLOOM_DISTRIBUTION_DISABLED`` info issue, writes nothing,
fires no decision event). Structurally cloned from
:class:`lib.validators.course_level_qa.CourseLevelQaValidator`.

References:
    - ``lib/ontology/bloom.py::BLOOM_LEVELS`` — canonical six-level axis.
    - ``schemas/taxonomies/bloom_target_distribution.json`` — generic curve.
    - ``lib/validators/course_level_qa.py`` — structural template.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from MCP.hardening.validation_gates import GateIssue, GateResult
from lib.ontology.bloom import BLOOM_LEVELS, get_verbs

logger = logging.getLogger(__name__)

ENV_BLOOM_DISTRIBUTION = "ED4ALL_BLOOM_DISTRIBUTION"
ENV_BLOOM_TARGET = "ED4ALL_BLOOM_DISTRIBUTION_TARGET"
ENV_BLOOM_TOLERANCE = "ED4ALL_BLOOM_DISTRIBUTION_TOLERANCE"
ENV_BLOOM_MIN_LOS = "ED4ALL_BLOOM_DISTRIBUTION_MIN_LOS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: Default L1-deviation tolerance for the OFF_TARGET decision.
DEFAULT_TOLERANCE: float = 0.20
#: Default small-N objective floor — below this, share-based verdicts skip.
DEFAULT_MIN_LOS: int = 6

#: The two LOWER-ORDER (recall) levels and the four HIGHER-ORDER (apply+) levels.
_RECALL_LEVELS: Tuple[str, ...] = ("remember", "understand")
_HIGHER_ORDER_LEVELS: Tuple[str, ...] = ("apply", "analyze", "evaluate", "create")
_TOP_HEAVY_LEVELS: Tuple[str, ...] = ("evaluate", "create")

_TARGET_CURVE_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "bloom_target_distribution.json"
)


# ---------------------------------------------------------------------------
# Flag / target / tolerance resolvers (parse-with-fallback)
# ---------------------------------------------------------------------------


def resolve_bloom_distribution(value: object = None) -> bool:
    """Return True iff the course-level Bloom-distribution gate is enabled.

    ``value`` (optional) overrides the env when not ``None``. Falsey / garbage /
    unset → False (parse-with-fallback).
    """
    if value is None:
        raw = os.environ.get(ENV_BLOOM_DISTRIBUTION, "")
    else:
        raw = str(value)
    return raw.strip().lower() in _TRUTHY


@lru_cache(maxsize=1)
def _load_default_curve() -> Dict[str, Any]:
    """Read + cache the checked-in generic target curve."""
    try:
        return json.loads(_TARGET_CURVE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # pragma: no cover - shipped file
        logger.error("bloom_distribution: default curve unreadable: %s", exc)
        # Hard-fallback so the validator never crashes if the file vanishes.
        return {
            "target_shares": {
                "remember": 0.15,
                "understand": 0.25,
                "apply": 0.25,
                "analyze": 0.20,
                "evaluate": 0.10,
                "create": 0.05,
            },
            "recall_ceiling": 0.50,
            "higher_order_floor": 0.30,
            "top_heavy_ceiling": 0.35,
        }


def _validate_curve_doc(doc: Any) -> Optional[Dict[str, Any]]:
    """Return a normalized curve dict iff ``doc`` is a valid target curve.

    A valid curve has a ``target_shares`` map whose keys are a subset of
    BLOOM_LEVELS and whose values are floats in [0,1]. Returns ``None`` on any
    shape violation (caller falls back to the canonical curve).
    """
    if not isinstance(doc, dict):
        return None
    shares = doc.get("target_shares")
    if not isinstance(shares, dict) or not shares:
        return None
    out_shares: Dict[str, float] = {}
    for level, share in shares.items():
        if level not in BLOOM_LEVELS:
            return None
        if not isinstance(share, (int, float)) or isinstance(share, bool):
            return None
        if not (0.0 <= float(share) <= 1.0):
            return None
        out_shares[level] = float(share)
    norm = {"target_shares": out_shares}
    for key in ("recall_ceiling", "higher_order_floor", "top_heavy_ceiling"):
        val = doc.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if 0.0 <= float(val) <= 1.0:
                norm[key] = float(val)
    return norm


def resolve_bloom_target(value: object = None) -> Dict[str, Any]:
    """Resolve the target Bloom curve (path / inline JSON / canonical default).

    ``ED4ALL_BLOOM_DISTRIBUTION_TARGET`` may be a path to a JSON file OR an inline
    JSON object of ``{level: share}`` (or a full ``{target_shares: {...}}`` doc).
    Unset / unreadable / schema-invalid / garbage → the checked-in generic curve
    (parse-with-fallback). The default curve's band knobs (recall_ceiling /
    higher_order_floor / top_heavy_ceiling) backfill any an override omits.
    """
    default = _load_default_curve()
    raw = os.environ.get(ENV_BLOOM_TARGET, "") if value is None else str(value)
    raw = (raw or "").strip()
    if not raw:
        return default

    doc: Any = None
    # Inline JSON object first (starts with '{'); else treat as a file path.
    if raw.startswith("{"):
        try:
            doc = json.loads(raw)
        except ValueError:
            doc = None
    else:
        try:
            p = Path(raw)
            if p.exists() and p.is_file():
                doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            doc = None

    # Accept either a full {target_shares: {...}} doc or a bare {level: share}.
    if isinstance(doc, dict) and "target_shares" not in doc:
        doc = {"target_shares": doc}

    norm = _validate_curve_doc(doc)
    if norm is None:
        logger.warning(
            "bloom_distribution: ED4ALL_BLOOM_DISTRIBUTION_TARGET invalid "
            "(%r) — falling back to the canonical curve.", raw[:80],
        )
        return default
    # Backfill band knobs the override omitted from the canonical defaults.
    for key in ("recall_ceiling", "higher_order_floor", "top_heavy_ceiling"):
        norm.setdefault(key, default.get(key))
    return norm


def resolve_tolerance(value: object = None) -> float:
    """Resolve the L1 OFF_TARGET tolerance. Garbage / non-positive → default."""
    raw = os.environ.get(ENV_BLOOM_TOLERANCE, "") if value is None else str(value)
    try:
        f = float(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_TOLERANCE
    if not (f > 0.0):
        return DEFAULT_TOLERANCE
    return f


def resolve_min_los(value: object = None) -> int:
    """Resolve the small-N objective floor. Garbage / non-positive → default."""
    raw = os.environ.get(ENV_BLOOM_MIN_LOS, "") if value is None else str(value)
    try:
        n = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_MIN_LOS
    if n < 1:
        return DEFAULT_MIN_LOS
    return n


# ---------------------------------------------------------------------------
# Objective Bloom-level resolution (anti-fabrication: declared signal only)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _verb_to_level() -> Dict[str, str]:
    """Reverse map: canonical Bloom verb (lowercase) -> level."""
    out: Dict[str, str] = {}
    verbs = get_verbs()
    for level in BLOOM_LEVELS:
        for v in verbs.get(level, set()):
            out.setdefault(v.lower(), level)
    return out


def _level_of_objective(obj: Dict[str, Any]) -> Optional[str]:
    """Resolve an objective's DECLARED Bloom level, or None (skip — no impute).

    Order: explicit ``bloom_level``/``bloomLevel`` field → canonical level of the
    declared ``bloom_verb``. NEVER runs free-text detection over the statement
    (that would be imputation). Returns None when the objective declares neither.
    """
    if not isinstance(obj, dict):
        return None
    for key in ("bloom_level", "bloomLevel", "bloom"):
        v = obj.get(key)
        if isinstance(v, str) and v.strip().lower() in BLOOM_LEVELS:
            return v.strip().lower()
    verb = obj.get("bloom_verb")
    if isinstance(verb, str) and verb.strip():
        return _verb_to_level().get(verb.strip().lower())
    return None


def _iter_objectives(obj_doc: Any) -> List[Dict[str, Any]]:
    """Yield every objective dict from a synthesized objectives doc.

    Handles the Courseforge synthesized form (``terminal_objectives`` flat +
    ``chapter_objectives`` grouped as ``[{chapter, objectives:[...]}]`` OR flat),
    the LibV2 archive form (``terminal_outcomes`` / ``component_objectives``),
    and a flat ``learning_outcomes`` list. Never raises.
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(obj_doc, dict):
        return out

    def _add_list(items: Any) -> None:
        if not isinstance(items, list):
            return
        for it in items:
            if isinstance(it, dict) and "objectives" in it and isinstance(
                it.get("objectives"), list
            ):
                # Grouped form: {chapter, objectives: [...]}.
                for sub in it["objectives"]:
                    if isinstance(sub, dict):
                        out.append(sub)
            elif isinstance(it, dict):
                out.append(it)

    for key in (
        "terminal_objectives",
        "terminal_outcomes",
        "chapter_objectives",
        "component_objectives",
        "learning_outcomes",
    ):
        _add_list(obj_doc.get(key))
    return out


def _load_json(path: Optional[str]) -> Optional[Any]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists() or not p.is_file():
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        logger.debug("bloom_distribution: failed to load %s: %s", path, exc)
        return None


def _block_target_bloom(block: Any) -> Optional[str]:
    """Resolve a rewrite-tier block's declared ``target_bloom`` (secondary)."""
    from lib.validators._block_rubric_helpers import block_attr

    v = block_attr(block, "target_bloom")
    if isinstance(v, str) and v.strip().lower() in BLOOM_LEVELS:
        return v.strip().lower()
    return None


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------


class BloomDistributionValidator:
    """Course-level Bloom-distribution-vs-target-curve gate."""

    name = "bloom_distribution"
    version = "0.1.0"

    def __init__(self, *, decision_capture: Optional[Any] = None) -> None:
        self._decision_capture = decision_capture

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        gate_id = inputs.get("gate_id", self.name)
        capture = inputs.get("decision_capture") or self._decision_capture

        enabled = inputs.get("bloom_distribution_enabled")
        if enabled is None:
            enabled = resolve_bloom_distribution()
        if not enabled:
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="BLOOM_DISTRIBUTION_DISABLED",
                        message=(
                            "ED4ALL_BLOOM_DISTRIBUTION unset — course-level Bloom "
                            "distribution gate skipped (byte-stable no-op)."
                        ),
                    )
                ],
                action=None,
                metadata={"bloom_distribution": {"enabled": False}},
            )

        target = resolve_bloom_target(inputs.get("bloom_target"))
        tolerance = resolve_tolerance(inputs.get("bloom_tolerance"))
        min_los = resolve_min_los(inputs.get("bloom_min_los"))

        # --- Primary signal: synthesized-objectives Bloom level ---------------
        obj_doc = _load_json(inputs.get("synthesized_objectives_path"))
        counts: Dict[str, int] = {lvl: 0 for lvl in BLOOM_LEVELS}
        n_objectives = 0
        n_skipped = 0
        for obj in _iter_objectives(obj_doc):
            lvl = _level_of_objective(obj)
            if lvl is None:
                n_skipped += 1
                continue
            counts[lvl] += 1
            n_objectives += 1

        # --- Secondary corroborating signal: block target_bloom (metadata) ----
        block_counts: Dict[str, int] = {lvl: 0 for lvl in BLOOM_LEVELS}
        for block in inputs.get("blocks") or []:
            lvl = _block_target_bloom(block)
            if lvl is not None:
                block_counts[lvl] += 1
        n_block_signal = sum(block_counts.values())

        if n_objectives == 0:
            self._emit_decision(
                capture,
                observed={},
                target=target,
                l1=None,
                n_objectives=0,
                n_skipped=n_skipped,
                issue_count=0,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=True,
                issues=[
                    GateIssue(
                        severity="info",
                        code="BLOOM_DISTRIBUTION_NO_OBJECTIVES",
                        message=(
                            "no synthesized objective declared a Bloom level / "
                            "verb (counted 0, skipped %d); course-level Bloom "
                            "distribution not auditable (structured skip)."
                            % n_skipped
                        ),
                    )
                ],
                action=None,
                metadata={
                    "bloom_distribution": {
                        "enabled": True,
                        "n_objectives": 0,
                        "n_skipped": n_skipped,
                        "block_target_bloom_counts": block_counts,
                    }
                },
            )

        observed = {lvl: counts[lvl] / n_objectives for lvl in BLOOM_LEVELS}
        target_shares: Dict[str, float] = {
            lvl: float(target.get("target_shares", {}).get(lvl, 0.0))
            for lvl in BLOOM_LEVELS
        }
        l1 = sum(abs(observed[lvl] - target_shares[lvl]) for lvl in BLOOM_LEVELS)

        recall_share = sum(observed[lvl] for lvl in _RECALL_LEVELS)
        higher_share = sum(observed[lvl] for lvl in _HIGHER_ORDER_LEVELS)
        top_share = sum(observed[lvl] for lvl in _TOP_HEAVY_LEVELS)
        recall_ceiling = float(target.get("recall_ceiling", 0.5))
        top_heavy_ceiling = float(target.get("top_heavy_ceiling", 0.35))

        issues: List[GateIssue] = []
        small_n = n_objectives < min_los
        if small_n:
            issues.append(
                GateIssue(
                    severity="info",
                    code="BLOOM_DISTRIBUTION_SMALL_N",
                    message=(
                        f"only {n_objectives} objective(s) carry a Bloom level "
                        f"(< small-N floor {min_los}); share-based OFF_TARGET / "
                        f"RECALL_HEAVY / TOP_HEAVY suppressed to avoid noise on a "
                        f"tiny course (NO_HIGHER_ORDER still applies)."
                    ),
                )
            )

        # NO_HIGHER_ORDER fires at ANY size — a recall-only course is a defect.
        if higher_share <= 0.0:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="BLOOM_DISTRIBUTION_NO_HIGHER_ORDER",
                    message=(
                        f"zero of {n_objectives} objectives reach apply+ "
                        f"(recall-only course): the entire Bloom mix sits in "
                        f"{sorted(_RECALL_LEVELS)}. The framework requires "
                        f"higher-order cognitive demand (apply / analyze / "
                        f"evaluate / create), not recall alone."
                    ),
                    suggestion=(
                        "Re-author at least some terminal objectives to an "
                        "apply+ Bloom level so the course demands transfer, not "
                        "only recall."
                    ),
                )
            )

        if not small_n:
            if recall_share > recall_ceiling:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="BLOOM_DISTRIBUTION_RECALL_HEAVY",
                        message=(
                            f"remember+understand share {recall_share:.2f} exceeds "
                            f"the curve's recall_ceiling {recall_ceiling:.2f}; the "
                            f"course is recall-heavy (low-order-dominant)."
                        ),
                        suggestion=(
                            "Shift some objectives from remember/understand to "
                            "apply+ so the cognitive demand isn't recall-dominant."
                        ),
                    )
                )
            if top_share > top_heavy_ceiling:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="BLOOM_DISTRIBUTION_TOP_HEAVY",
                        message=(
                            f"evaluate+create share {top_share:.2f} exceeds the "
                            f"curve's top_heavy_ceiling {top_heavy_ceiling:.2f}; the "
                            f"course is top-heavy (insufficient apply/analyze "
                            f"scaffolding under the highest-order demand)."
                        ),
                        suggestion=(
                            "Add apply/analyze objectives so evaluate/create "
                            "demand is scaffolded rather than front-loaded."
                        ),
                    )
                )
            if l1 > tolerance:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="BLOOM_DISTRIBUTION_OFF_TARGET",
                        message=(
                            f"observed Bloom mix deviates from the target curve "
                            f"by L1={l1:.3f} (> tolerance {tolerance:.2f}). "
                            f"observed={ {k: round(v, 2) for k, v in observed.items()} } "
                            f"target={ {k: round(v, 2) for k, v in target_shares.items()} }."
                        ),
                        suggestion=(
                            "Re-balance objective Bloom levels toward the target "
                            "curve, or set ED4ALL_BLOOM_DISTRIBUTION_TARGET to a "
                            "curve matching this course's intended profile."
                        ),
                    )
                )

        self._emit_decision(
            capture,
            observed=observed,
            target=target_shares,
            l1=l1,
            n_objectives=n_objectives,
            n_skipped=n_skipped,
            issue_count=len([i for i in issues if i.severity == "warning"]),
        )

        # Warning-day-1: ``passed`` stays True.
        #
        # TODO(calibration): when scripts/calibration_harness.py confirms the
        # FP rate on >=2 corpora, flip the gate row to ``severity: critical`` +
        # ``behavior.on_fail: block`` in config/workflows.yaml AND set ``passed``
        # to ``"BLOOM_DISTRIBUTION_NO_HIGHER_ORDER" not in {i.code for i in issues}``
        # so a recall-only course blocks promotion.
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=True,
            score=round(max(0.0, 1.0 - l1), 4),
            issues=issues,
            action=None,
            metadata={
                "bloom_distribution": {
                    "enabled": True,
                    "n_objectives": n_objectives,
                    "n_skipped": n_skipped,
                    "small_n": small_n,
                    "min_los": min_los,
                    "tolerance": tolerance,
                    "observed_counts": counts,
                    "observed_shares": {k: round(v, 4) for k, v in observed.items()},
                    "target_shares": {k: round(v, 4) for k, v in target_shares.items()},
                    "l1_deviation": round(l1, 4),
                    "recall_share": round(recall_share, 4),
                    "higher_order_share": round(higher_share, 4),
                    "top_heavy_share": round(top_share, 4),
                    "block_target_bloom_counts": block_counts,
                    "n_block_signal": n_block_signal,
                }
            },
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _emit_decision(
        capture: Any,
        *,
        observed: Dict[str, float],
        target: Dict[str, Any],
        l1: Optional[float],
        n_objectives: int,
        n_skipped: int,
        issue_count: int,
    ) -> None:
        if capture is None:
            return
        try:
            obs_round = {k: round(float(v), 3) for k, v in observed.items()}
            l1_str = "n/a" if l1 is None else f"{l1:.3f}"
            capture.log_decision(
                decision_type="validation_result",
                decision=(
                    f"bloom_distribution n_objectives={n_objectives}, "
                    f"L1={l1_str}, warnings={issue_count}"
                ),
                rationale=(
                    "Course-level Bloom-distribution-vs-target audit: counted the "
                    f"declared Bloom level of {n_objectives} objective(s) "
                    f"(skipped {n_skipped} with no declared level/verb — never "
                    f"imputed) onto the canonical six-level axis. observed shares "
                    f"{obs_round} vs target curve; L1 deviation {l1_str}; "
                    f"{issue_count} warning(s). Warning-day-1: enumerated but does "
                    "not yet block (calibration-deferred critical-flip)."
                ),
                ml_features={
                    "n_objectives": int(n_objectives),
                    "n_skipped": int(n_skipped),
                    "l1_deviation": (None if l1 is None else round(float(l1), 4)),
                    "warning_count": int(issue_count),
                },
            )
        except Exception as exc:  # noqa: BLE001 — audit emit is best-effort
            logger.debug(
                "DecisionCapture.log_decision raised on bloom_distribution: %s",
                exc,
            )


__all__ = [
    "BloomDistributionValidator",
    "resolve_bloom_distribution",
    "resolve_bloom_target",
    "resolve_tolerance",
    "resolve_min_los",
    "ENV_BLOOM_DISTRIBUTION",
    "ENV_BLOOM_TARGET",
    "ENV_BLOOM_TOLERANCE",
    "ENV_BLOOM_MIN_LOS",
    "DEFAULT_TOLERANCE",
    "DEFAULT_MIN_LOS",
]
