"""Course-level final-status helpers.

  * :func:`load_course_status_schema` — cached loader for
    ``schemas/governance/course_status.schema.json``. Same single-shot cached
    pattern as :func:`lib.ontology.taxonomy.load_taxonomy`.

  * :func:`compose_course_status` — partial composer over a
    ``{arrow_name: promotion_decision}`` mapping. Implements only the ``failed``
    and ``non_certified_archive`` branches; the three ``certified_*`` branches
    raise :class:`NotImplementedError` because tier selection needs gate-by-gate
    evidence that a bare per-arrow decision map does not carry. Callers wanting
    full enum routing use :func:`derive_course_status` instead.

  * :func:`derive_course_status` — full 5-branch composer over the 9-arrow
    promotion-chain rows produced by
    :class:`lib.aggregators.promotion_chain_report.PromotionChainAggregator`.

The enum values are the five members of
``schemas/governance/course_status.schema.json``:

    certified_accessible
    certified_instructional
    certified_trainable
    non_certified_archive
    failed

``schemas/ONTOLOGY.md`` covers the broader governance posture.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

__all__ = [
    "load_course_status_schema",
    "compose_course_status",
    "derive_course_status",
    "ACCESSIBILITY_GATE_IDS",
    "INSTRUCTIONAL_GATE_IDS",
    "TRAINABLE_GATE_IDS",
]


# ---------------------------------------------------------------------------
# Schema path + cache
# ---------------------------------------------------------------------------

_COURSE_STATUS_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "governance"
    / "course_status.schema.json"
)


@lru_cache(maxsize=1)
def _load_course_status_schema_cached() -> Dict[str, Any]:
    """Cached single-shot loader for the course-status schema."""
    if not _COURSE_STATUS_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Course-status schema not found at {_COURSE_STATUS_SCHEMA_PATH}. "
            "Expected canonical copy from Wave 1 (Worker W1.C)."
        )
    with open(_COURSE_STATUS_SCHEMA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "enum" not in data:
        raise ValueError(
            f"Malformed course-status schema at {_COURSE_STATUS_SCHEMA_PATH}: "
            "missing top-level 'enum' member."
        )
    return data


def load_course_status_schema() -> Dict[str, Any]:
    """Load and return the course-status JSON-Schema dict.

    Returns a defensive deep copy of the cached schema so callers may
    mutate the returned dict (including its nested ``enum`` list) without
    polluting the cache. Mirrors the pattern in
    :func:`lib.ontology.taxonomy.load_taxonomy`.

    Raises:
        FileNotFoundError: when the schema file is missing.
        ValueError: when the schema shape is invalid (missing ``enum``).
    """
    return copy.deepcopy(_load_course_status_schema_cached())


# ---------------------------------------------------------------------------
# Partial composer over a per-arrow decision map
# ---------------------------------------------------------------------------

# Boundary between the "accessible+instructional" arrows (1-5) and the
# "trainable" arrows (6+). Arrows up to this index inclusive form the
# "accessible+instructional" slice.
_ACCESSIBLE_INSTRUCTIONAL_ARROW_BOUNDARY = 5

# Promotion-decision values that count as a passing arrow. Every other value
# (including ``"missing"`` / absent / ``"fail"``) is not-passing.
_PASSING_DECISIONS = frozenset({"pass", "warn"})

# Promotion-decision values that count as a definite failure for the
# course-level rollup. Any arrow with one of these values short-circuits the
# course to ``"failed"``.
_FAILING_DECISIONS = frozenset({"fail"})


def compose_course_status(per_arrow_decisions: Mapping[str, str]) -> str:
    """Compose a course-level final-status enum from per-arrow decisions.

    Implements only the two branches decidable from per-arrow decisions alone:

      * Returns ``"failed"`` when ANY arrow has
        ``promotion_decision == "fail"``.
      * Returns ``"non_certified_archive"`` when every arrow up to a
        5-arrow boundary passes (``"pass"`` or ``"warn"``) and arrows
        beyond that boundary are missing or non-passing.

    The three ``certified_*`` branches need gate-by-gate evidence from the
    promotion-chain aggregator output, which this mapping does not carry, so
    they raise :class:`NotImplementedError`. Use :func:`derive_course_status`
    for full enum routing.

    Arrow naming convention: keys are the per-arrow names (informally
    ``"arrow1"`` … ``"arrow9"``); values are the per-arrow promotion
    decisions, drawn from the same enum the PhaseOutput
    ``promotion_decision`` field carries (``"pass" | "warn" | "fail" |
    "escalate"``) plus a ``"missing"`` sentinel for arrows that have not yet
    emitted a decision. Mapping order does NOT matter — keys are partitioned
    by their numeric suffix, not by iteration order.

    Args:
        per_arrow_decisions: mapping of ``{arrow_name: promotion_decision}``.

    Returns:
        One of the five enum values from
        ``schemas/governance/course_status.schema.json``.

    Raises:
        NotImplementedError: for the three ``certified_*`` branches.
    """
    # Defensive copy so callers can mutate the input afterward.
    decisions: Dict[str, str] = dict(per_arrow_decisions)

    # --- Branch 1: any explicit fail short-circuits to "failed". -----------
    if any(value in _FAILING_DECISIONS for value in decisions.values()):
        return "failed"

    # --- Branch 2: arrows 1-5 pass, arrows 6+ missing/failed ---------------
    def _arrow_index(name: str) -> int:
        # Strip the "arrow" prefix when present and parse the trailing digits.
        # A non-parseable key lands BEYOND the 1-5 boundary on purpose: a
        # non-canonical key must not accidentally satisfy the
        # "non_certified_archive" precondition by counting as a passing
        # in-slice arrow.
        suffix = name[len("arrow"):] if name.startswith("arrow") else name
        try:
            return int(suffix)
        except (TypeError, ValueError):
            return _ACCESSIBLE_INSTRUCTIONAL_ARROW_BOUNDARY + 1

    accessible_slice = {
        name: value
        for name, value in decisions.items()
        if _arrow_index(name) <= _ACCESSIBLE_INSTRUCTIONAL_ARROW_BOUNDARY
    }
    trainable_slice = {
        name: value
        for name, value in decisions.items()
        if _arrow_index(name) > _ACCESSIBLE_INSTRUCTIONAL_ARROW_BOUNDARY
    }

    accessible_all_pass = bool(accessible_slice) and all(
        value in _PASSING_DECISIONS for value in accessible_slice.values()
    )
    trainable_any_pass = any(
        value in _PASSING_DECISIONS for value in trainable_slice.values()
    )

    if accessible_all_pass and not trainable_any_pass:
        return "non_certified_archive"

    # --- Branches 3-5: certified_* ----------------------------------------
    # When arrows 1-5 pass AND any arrow 6+ also passes, the course is a
    # candidate for one of the three certified_* tiers. Tier selection needs
    # gate-by-gate evidence from the promotion-chain aggregator (arrow-6 alone
    # => certified_accessible vs arrows 6+7 => certified_instructional vs full
    # chain => certified_trainable), which a per-arrow decision map cannot
    # supply. Raise rather than guess a tier.
    raise NotImplementedError(
        "compose_course_status: certified_* branch routing is deferred to "
        "Wave 3 (governance G1 — promotion-chain aggregator). Wave 1 only "
        "implements the 'failed' and 'non_certified_archive' branches. See "
        "plans/gpt-feedback-2-wave1-schemas-2026-05.md § 'Worker W1.C' for "
        "the deferral rationale."
    )


# ---------------------------------------------------------------------------
# Full 5-branch composer over the promotion-chain aggregator's per-arrow rows.
# ---------------------------------------------------------------------------


# Canonical gate cohorts. The composer walks every arrow row's
# ``validator_set[]`` looking for these gate IDs to decide the certified_*
# tier. They are module constants rather than inline literals so test code has
# a single import surface.

#: Gates whose pass status promotes a course past arrow 5 (IMSCC chunks)
#: into the ``certified_accessible`` tier. This is the WCAG / accessibility
#: floor — without it a course can only land in ``non_certified_archive``.
ACCESSIBILITY_GATE_IDS: frozenset = frozenset({
    "wcag_compliance",
    "wcag_aa_compliance",
})

#: Gates whose joint pass status promotes ``certified_accessible`` to
#: ``certified_instructional``. Surfaces the content-quality dimensions
#: that make a course teach something coherently: per-claim NLI entailment
#: against cited chunks, source grounding, OSCQR course-quality scoring,
#: per-page LO coverage, and per-emit source-ref integrity.
#:
#: ``claim_support`` — block-side per-claim NLI entailment gate fired at the
#: ``post_rewrite_validation`` phase (Arrow 3 cohort). Sibling to
#: ``content_grounding`` and ``source_refs``. Warning-severity until operator
#: calibration, so this cohort entry is dormant; when severity flips to
#: critical, a critical-fail must cap the instructional tier here rather than
#: short-circuit to ``"failed"`` via the critical-cohort branch in
#: :func:`derive_course_status`.
INSTRUCTIONAL_GATE_IDS: frozenset = frozenset({
    "claim_support",
    "content_grounding",
    "oscqr_score",
    "page_objectives",
    "source_refs",
})

#: Gates whose joint pass status promotes ``certified_instructional`` to
#: ``certified_trainable``. Surfaces the training-corpus quality
#: dimensions: KG edge density, synthesis-template diversity,
#: post-training eval-gating thresholds, per-family CURIE-coverage
#: symmetry, and per-pair per-claim NLI entailment.
#:
#: ``pair_claim_support`` — pair-side per-claim NLI entailment gate fired at
#: the ``training_synthesis`` phase (Arrow 7 cohort). Sibling to
#: ``synthesis_diversity`` and ``synthesis_leakage``. Warning-severity until
#: operator calibration, so this cohort entry is dormant; when severity flips
#: to critical, a critical-fail must cap the trainable tier here rather than
#: short-circuit to ``"failed"`` via the critical-cohort branch in
#: :func:`derive_course_status`.
TRAINABLE_GATE_IDS: frozenset = frozenset({
    "eval_gating",
    "family_completeness",
    "min_edge_count",
    "pair_claim_support",
    "synthesis_diversity",
})

# Last arrow of the archivable core pipeline, used by the ``failed`` /
# ``non_certified_archive`` decision branches.
_NON_CERTIFIED_ARCHIVE_BOUNDARY_ARROW = 5

# Anti-silent-degradation sentinel gate ID. A module constant so the
# non-training-run relaxation below can recognise a legitimately-skipped stage
# (an arrow whose ONLY failure signal is this sentinel) without importing the
# aggregator.
_MISSING_STAGE_REPORT = "missing_stage_report"

# First arrow of the assessment / training cohort. Arrows at or beyond this
# ordinal (6: assessment items, 7: training pairs, 8: adapter, 9: eval) only
# run on a TRAINING build. On a non-training run (``--skip-training`` /
# ``--stop-after imscc_chunking``) they are legitimately absent, so a
# ``missing_stage_report`` sentinel on them must NOT short-circuit the whole
# course to ``failed``.
_TRAINING_COHORT_MIN_ARROW = 6

# Promotion-decision values that signal a hard failure at the arrow level.
_HARD_FAIL_DECISIONS = frozenset({"fail"})

# Promotion-decision values that count as "not blocking" (the arrow's gates
# either passed or only fired warnings). Used by every cohort-pass check.
_NON_BLOCKING_DECISIONS = frozenset({"pass", "warn"})


def _arrow_passes(arrow: Mapping[str, Any]) -> bool:
    """Return True when an arrow's per-row promotion_decision is non-blocking."""
    return arrow.get("promotion_decision") in _NON_BLOCKING_DECISIONS


def _validator_set(arrow: Mapping[str, Any]) -> Set[str]:
    """Coerce an arrow's ``validator_set`` field to a set of gate IDs."""
    raw = arrow.get("validator_set") or []
    if not isinstance(raw, (list, tuple, set, frozenset)):
        return set()
    out: Set[str] = set()
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.add(item.strip())
    return out


def _arrows_in_chain_order(arrows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """Sort arrows by arrow_id so the helper is order-stable across callers."""
    out: List[Mapping[str, Any]] = []
    for a in arrows or []:
        if isinstance(a, Mapping):
            out.append(a)
    out.sort(key=lambda a: int(a.get("arrow_id") or 0))
    return out


def _cohort_passes(
    arrows: Sequence[Mapping[str, Any]],
    cohort_gate_ids: frozenset,
) -> bool:
    """Return True when the cohort's gate evidence supports certification.

    A gate ID "passes" when at least one arrow row carries the gate ID in
    its ``validator_set`` AND that arrow's ``promotion_decision`` is
    non-blocking (``"pass"`` or ``"warn"``); it "fails" when at least one
    arrow row carries the gate ID AND that arrow's promotion_decision is
    ``"fail"``.

    Cohort-pass rule:

    * **Any failure disqualifies the cohort.** A single fail attached to
      ANY cohort gate flips the cohort to non-certifying. This closes the
      silent-degradation class where a partial pass + a single fail still
      certifies.
    * **At least one cohort gate must pass.** A cohort with zero
      passing-gate evidence in the chain cannot certify — silently
      crediting an empty cohort would also be a silent-degradation class.
    * Cohort gates that NEVER appear in the chain are treated as "not
      asserted": they don't count as failures (the composer can't
      penalise a phase that didn't run), but they also don't count as
      passes. The cohort certifies as long as the asserted gates all pass
      AND at least one gate is asserted.
    """
    any_pass = False
    any_fail = False
    for gate_id in cohort_gate_ids:
        for arrow in arrows:
            if gate_id not in _validator_set(arrow):
                continue
            if _arrow_passes(arrow):
                any_pass = True
            else:
                any_fail = True
    if any_fail:
        return False
    return any_pass


def derive_course_status(
    arrows: Iterable[Mapping[str, Any]],
    *,
    critical_gate_ids: Optional[Iterable[str]] = None,
    training_expected: Optional[bool] = None,
) -> str:
    """Walk the per-arrow rows and return the canonical course_status enum.

    ``training_expected``: three-valued run-scope hint.

      * ``None`` (default) — strict: a ``missing_stage_report`` sentinel on ANY
        arrow (including the assessment/training cohort, arrows 6-9) is a
        critical failure and short-circuits to ``failed``. This is the
        behaviour every caller that omits the hint gets.
      * ``False`` — a NON-training run (``--skip-training`` /
        ``--stop-after imscc_chunking``): the assessment/training arrows
        (``arrow_id >= 6``) were never dispatched, so a
        ``missing_stage_report`` sentinel on them is a legitimate skip, NOT a
        broken audit trail. Those arrows no longer short-circuit to
        ``failed``; the course certifies at whatever tier the completed
        arrows (1-5) support (e.g. ``certified_instructional``). A genuine
        failure in the core pipeline (a missing/failed arrow 1-5, or a real
        critical-gate fail) STILL returns ``failed``.
      * ``True`` — a training run: identical to the strict ``None`` path (a
        missing training report on a build that WAS supposed to train is a
        real regression).

    Decision logic (the 5-branch tree):

      1. ``failed`` — any arrow's ``promotion_decision == "fail"`` AND the
         arrow's failure attaches to a critical-cohort gate. Critical
         cohort defaults to :data:`ACCESSIBILITY_GATE_IDS` plus the
         ``"missing_stage_report"`` sentinel — the two cohorts that
         legitimately gate the entire chain (a WCAG fail makes the
         course unshippable; a missing per-stage report breaks the audit
         trail). Failures attached only to instructional / trainable
         cohort gates do NOT short-circuit to ``failed`` — they just
         prevent the corresponding tier from certifying. Callers that
         want broader critical membership pass ``critical_gate_ids=``.
         A failing arrow with NO validator_set rows is treated as
         critical (the composer can't otherwise tell which gate fired).
      2. ``non_certified_archive`` — every arrow up to arrow 5 (IMSCC
         chunks) lands cleanly (no fail), but every arrow 6+ either
         skipped or non-blocking-failed. This is the "the course is
         archivable but never reached the assessment / training cohort"
         tier.
      3. ``certified_accessible`` — :data:`ACCESSIBILITY_GATE_IDS`
         passes; the instructional + trainable cohorts may be partial.
      4. ``certified_instructional`` — :data:`ACCESSIBILITY_GATE_IDS`
         AND :data:`INSTRUCTIONAL_GATE_IDS` both pass.
      5. ``certified_trainable`` — all three cohorts
         (:data:`ACCESSIBILITY_GATE_IDS` + :data:`INSTRUCTIONAL_GATE_IDS`
         + :data:`TRAINABLE_GATE_IDS`) pass.

    Anti-silent-degradation: a course with NO matching cohort gates in
    the chain falls through to ``non_certified_archive`` (when arrows
    1-5 pass) OR ``failed`` (when any arrow has hard-failed). The
    helper never returns ``None`` — every input maps to one of the five
    enum values.
    """
    rows = _arrows_in_chain_order(arrows)
    critical_set: Set[str] = set(critical_gate_ids or []) | (
        ACCESSIBILITY_GATE_IDS | {"missing_stage_report"}
    )

    # --- Branch 1: any hard fail attached to a critical gate -> failed ---
    for arrow in rows:
        if arrow.get("promotion_decision") not in _HARD_FAIL_DECISIONS:
            continue
        vs = _validator_set(arrow)
        # Non-training run relaxation: an assessment/training arrow
        # (``arrow_id >= 6``) whose ONLY failure signal is the
        # missing-stage-report sentinel was legitimately skipped (the
        # training phases were never dispatched). It must NOT force the
        # whole course to ``failed`` — the completed core-pipeline arrows
        # decide the certified tier below. Any OTHER failing gate on those
        # arrows (a real assessment/eval fail) falls through to the normal
        # criticality test, which does not mark them critical either (they
        # are not in the accessibility cohort) — it just caps the tier.
        if (
            training_expected is False
            and int(arrow.get("arrow_id") or 0) >= _TRAINING_COHORT_MIN_ARROW
            and vs <= {_MISSING_STAGE_REPORT}
        ):
            continue
        # A failing arrow with no validator_set rows is treated as
        # critical by construction (the missing-stage-report sentinel
        # falls into this bucket because the composer can't tell which
        # gate would have fired). Otherwise the failing arrow is critical
        # iff at least one of its validator_set entries appears in the
        # critical cohort.
        if not vs or vs & critical_set:
            return "failed"

    # --- Branch 5 candidate: full trainable cohort passes ----------------
    accessibility_ok = _cohort_passes(rows, ACCESSIBILITY_GATE_IDS)
    instructional_ok = _cohort_passes(rows, INSTRUCTIONAL_GATE_IDS)
    trainable_ok = _cohort_passes(rows, TRAINABLE_GATE_IDS)

    if accessibility_ok and instructional_ok and trainable_ok:
        return "certified_trainable"

    # --- Branch 4: certified_instructional -------------------------------
    if accessibility_ok and instructional_ok:
        return "certified_instructional"

    # --- Branch 3: certified_accessible ----------------------------------
    if accessibility_ok:
        return "certified_accessible"

    # --- Branch 2: non_certified_archive ---------------------------------
    # Every arrow up to and including arrow 5 must pass for a course to
    # earn the archive tier. Below arrow 5 (i.e. arrows 6/7/8/9) is
    # allowed to be missing or non-blocking-failed.
    archive_slice = [
        a for a in rows
        if int(a.get("arrow_id") or 0) <= _NON_CERTIFIED_ARCHIVE_BOUNDARY_ARROW
    ]
    if archive_slice and all(_arrow_passes(a) for a in archive_slice):
        return "non_certified_archive"

    # --- Fallback: the chain is messy enough that none of the strict
    # branches matched, and no critical fail was raised. Treat as
    # ``failed`` so the composer never silently downgrades a degraded
    # chain to a certified tier.
    return "failed"
