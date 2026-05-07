"""Per-question-type quality thresholds + diversity floors.

W-D7 T7.4: extracted from :mod:`lib.validators.assessment` to keep the
calibration-coupled threshold tables in one auditable file. The
canonical module re-exports every name so existing imports
(``from lib.validators.assessment import _PER_QUESTION_TYPE_THRESHOLDS``)
keep resolving.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.4.
"""
from __future__ import annotations

from typing import Any, Dict

__all__ = [
    "STEM_DIVERSITY_THRESHOLD",
    "CORRECT_ANSWER_DIVERSITY_THRESHOLD",
    "DISTRACTOR_TEMPLATE_MAX_RATIO",
    "_PER_QUESTION_TYPE_THRESHOLDS",
    "_DEFAULT_QUESTION_TYPE_THRESHOLDS",
    "_resolve_per_type_thresholds",
    "_thresholds_for_type",
]


# Wave 26 real-failure-mode thresholds
STEM_DIVERSITY_THRESHOLD = 0.7
CORRECT_ANSWER_DIVERSITY_THRESHOLD = 0.6
DISTRACTOR_TEMPLATE_MAX_RATIO = 0.30


#: Per-question-type quality thresholds (Wave 6 W6.A calibration).
#: Day-1 starting points; calibrate against the rdf-shacl-551-2 corpus
#: rebuild before promoting severity from warning to critical.
#:
#: Mirrors :data:`lib.validators.block_objective_delivery._PER_BLOCK_TYPE_ENTAILMENT_FLOOR`
#: (W1.7.C) and :data:`lib.validators.pair.objective_delivery._PER_PAIR_KIND_ENTAILMENT_FLOOR`
#: (W4.C MEDIUM) shape — closes the assessment-quality side of
#: "validation reflects question shape, not aggregate noise".
#:
#: Rationale per type:
#:   - multiple_choice: highest stem volume per assessment, MC has
#:     well-formed stem grammar requirements (interrogative or
#:     stem-completion form). Tightest distinct-stem floor.
#:   - true_false: minimal stem complexity; relaxed distinct-stem
#:     floor because legitimate T/F can repeat declarative shapes.
#:     Allows verb-less stems (the existing single-exception rule
#:     already accommodates this; per-type relaxation makes it
#:     dispatch-clean).
#:   - short_answer: free-response, moderate diversity required.
#:   - essay: small-N per assessment (1-3 essay questions); diversity
#:     ratio is meaningless on N<3, so the floor relaxes for low-N.
#:   - fill_in_blank: term-recognition format, moderate diversity.
_PER_QUESTION_TYPE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "multiple_choice": {
        "stem_diversity": 0.75,
        "correct_answer_diversity": 0.65,
        "distractor_template_max_ratio": 0.25,
        "min_stem_chars": 12,
    },
    "true_false": {
        "stem_diversity": 0.50,
        "correct_answer_diversity": 0.40,
        "distractor_template_max_ratio": 1.0,  # T/F has no real distractors
        "min_stem_chars": 10,
    },
    "short_answer": {
        "stem_diversity": 0.65,
        "correct_answer_diversity": 0.55,
        "distractor_template_max_ratio": 1.0,  # SA has no distractors
        "min_stem_chars": 12,
    },
    "essay": {
        "stem_diversity": 0.55,
        "correct_answer_diversity": 0.50,
        "distractor_template_max_ratio": 1.0,
        "min_stem_chars": 15,
    },
    "fill_in_blank": {
        "stem_diversity": 0.65,
        "correct_answer_diversity": 0.55,
        "distractor_template_max_ratio": 0.30,
        "min_stem_chars": 10,
    },
}


#: Default thresholds — used when a question_type is unknown
#: (defensive against future type additions or QTI ``matching``).
#: Same values as the existing module-level constants for back-compat.
_DEFAULT_QUESTION_TYPE_THRESHOLDS: Dict[str, float] = {
    "stem_diversity": STEM_DIVERSITY_THRESHOLD,           # 0.7
    "correct_answer_diversity": CORRECT_ANSWER_DIVERSITY_THRESHOLD,  # 0.6
    "distractor_template_max_ratio": DISTRACTOR_TEMPLATE_MAX_RATIO,  # 0.30
    "min_stem_chars": 10,
}


def _resolve_per_type_thresholds(
    inputs: Dict[str, Any],
) -> Dict[str, Dict[str, float]]:
    """Resolve the per-question-type quality threshold table.

    Operators can override per-gate via
    ``gate.config.per_question_type_thresholds`` in ``config/workflows.yaml``;
    the gate runner merges ``gate.config`` into the inputs dict at
    ``MCP/hardening/validation_gates.py:266-271`` (Wave 78 setdefault-merge
    pattern; existing precedent for ``outline_objective_assessment_similarity``
    and similar warning gates).

    Mirrors :func:`lib.validators.block_objective_delivery._resolve_threshold_table`
    at ``block_objective_delivery.py:252-271``. Per-type overlay merges
    on top of the canonical table; entries the operator omits keep the
    day-1 default values.
    """
    raw = inputs.get("per_question_type_thresholds") if isinstance(inputs, dict) else None
    merged: Dict[str, Dict[str, float]] = {
        qt: dict(thresholds)
        for qt, thresholds in _PER_QUESTION_TYPE_THRESHOLDS.items()
    }
    if isinstance(raw, dict):
        for qt, override in raw.items():
            if not isinstance(override, dict):
                continue
            qt_key = str(qt).lower()
            if qt_key not in merged:
                merged[qt_key] = dict(_DEFAULT_QUESTION_TYPE_THRESHOLDS)
            for axis, value in override.items():
                try:
                    merged[qt_key][str(axis)] = float(value)
                except (TypeError, ValueError):
                    continue
    return merged


def _thresholds_for_type(
    q_type: str,
    table: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Return the per-type threshold dict, falling back to defaults.

    A question_type not present in ``table`` (e.g. QTI's ``matching``,
    or a future-added type the operator hasn't calibrated yet) routes
    through :data:`_DEFAULT_QUESTION_TYPE_THRESHOLDS` so the per-type
    matrix degrades gracefully on unknown types rather than KeyError'ing.
    """
    return table.get(q_type, dict(_DEFAULT_QUESTION_TYPE_THRESHOLDS))
