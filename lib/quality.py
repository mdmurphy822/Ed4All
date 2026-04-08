"""
Quality assessment utilities for decision capture.

Provides centralized quality scoring for captured decisions to ensure
training data meets minimum quality standards.
"""

from typing import Any, Dict, List, Optional, Tuple

from .constants import QUALITY_THRESHOLDS


def assess_decision_quality(
    rationale: str,
    inputs_ref: Optional[List[Any]] = None,
    alternatives_considered: Optional[List[str]] = None,
    decision_type: str = ""
) -> str:
    """Assess the quality level of a decision based on its components.

    Args:
        rationale: The rationale text for the decision
        inputs_ref: List of input references used
        alternatives_considered: List of alternatives that were considered
        decision_type: Type of decision (some types have relaxed requirements)

    Returns:
        Quality level: "exemplary", "proficient", "developing", or "inadequate"
    """
    # Some decision types have relaxed requirements
    relaxed_types = {"prompt_response", "file_creation", "source_usage"}

    rationale_length = len(rationale) if rationale else 0
    has_inputs = bool(inputs_ref and len(inputs_ref) > 0)
    has_alternatives = bool(alternatives_considered and len(alternatives_considered) > 0)

    # Exemplary: long rationale + inputs + alternatives
    if (rationale_length >= QUALITY_THRESHOLDS["exemplary"]["rationale_min_length"]
            and has_inputs and has_alternatives):
        return "exemplary"

    # Proficient: medium rationale + (inputs OR alternatives)
    if (rationale_length >= QUALITY_THRESHOLDS["proficient"]["rationale_min_length"]
            and (has_inputs or has_alternatives)):
        return "proficient"

    # Developing: minimum rationale length
    if rationale_length >= QUALITY_THRESHOLDS["developing"]["rationale_min_length"]:
        return "developing"

    # For relaxed types, developing is acceptable with shorter rationale
    if decision_type in relaxed_types and rationale_length > 0:
        return "developing"

    return "inadequate"


def assess_from_inputs_ref(inputs_ref: Optional[List[Any]]) -> Dict[str, Any]:
    """Assess quality contribution from input references.

    Args:
        inputs_ref: List of InputRef objects or dicts

    Returns:
        Dict with assessment metrics
    """
    if not inputs_ref:
        return {
            "has_inputs": False,
            "input_count": 0,
            "has_content_hash": False,
            "source_types": [],
        }

    source_types = set()
    has_hash = False

    for ref in inputs_ref:
        if isinstance(ref, dict):
            source_types.add(ref.get("source_type", "unknown"))
            if ref.get("content_hash"):
                has_hash = True
        elif hasattr(ref, "source_type"):
            source_types.add(ref.source_type)
            if hasattr(ref, "content_hash") and ref.content_hash:
                has_hash = True

    return {
        "has_inputs": True,
        "input_count": len(inputs_ref),
        "has_content_hash": has_hash,
        "source_types": list(source_types),
    }


def calculate_quality_breakdown(
    rationale: str,
    inputs_ref: Optional[List[Any]] = None,
    alternatives_considered: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculate detailed quality breakdown for a decision.

    Args:
        rationale: The rationale text
        inputs_ref: Input references
        alternatives_considered: Alternatives considered

    Returns:
        Dict with detailed quality metrics
    """
    rationale_length = len(rationale) if rationale else 0
    input_assessment = assess_from_inputs_ref(inputs_ref)

    return {
        "rationale_length": rationale_length,
        "rationale_score": min(1.0, rationale_length / 100),  # Cap at 100 chars
        "has_inputs": input_assessment["has_inputs"],
        "input_count": input_assessment["input_count"],
        "has_alternatives": bool(alternatives_considered and len(alternatives_considered) > 0),
        "alternatives_count": len(alternatives_considered) if alternatives_considered else 0,
        "overall_quality": assess_decision_quality(
            rationale, inputs_ref, alternatives_considered
        ),
    }


def check_quality_acceptable(
    quality_level: str,
    minimum_level: str = "developing"
) -> Tuple[bool, str]:
    """Check if a quality level meets minimum requirements.

    Args:
        quality_level: The assessed quality level
        minimum_level: The minimum acceptable level

    Returns:
        Tuple of (is_acceptable, reason)
    """
    levels = ["inadequate", "developing", "proficient", "exemplary"]

    try:
        quality_idx = levels.index(quality_level)
        min_idx = levels.index(minimum_level)
    except ValueError:
        return False, f"Unknown quality level: {quality_level} or {minimum_level}"

    if quality_idx >= min_idx:
        return True, f"Quality '{quality_level}' meets minimum '{minimum_level}'"

    return False, f"Quality '{quality_level}' below minimum '{minimum_level}'"


# =============================================================================
# ENHANCED QUALITY SCORING
# =============================================================================

# Quality level ordering for comparisons
QUALITY_ORDER = {"inadequate": 0, "developing": 1, "proficient": 2, "exemplary": 3}


def score_rationale_depth(rationale: str) -> float:
    """
    Score rationale quality beyond just length.

    Returns 0.0-1.0 composite score based on:
    - Length (0-0.3): Longer rationales up to 300 chars
    - Reasoning indicators (0-0.4): Presence of causal/comparative language
    - Pedagogical specificity (0-0.3): Domain-specific terminology

    Args:
        rationale: The rationale text to score

    Returns:
        Float score from 0.0 to 1.0
    """
    if not rationale:
        return 0.0

    score = 0.0
    rationale_lower = rationale.lower()

    # Length component (0-0.3)
    # Rewards longer rationales up to 300 characters
    score += min(0.3, len(rationale) / 300)

    # Reasoning indicators (0-0.4)
    # Words that indicate causal reasoning or comparison
    reasoning_words = [
        "because", "since", "therefore", "thus", "consequently",
        "allows", "enables", "prevents", "ensures", "supports",
        "aligns with", "chosen over", "preferred", "better than",
        "more appropriate", "rather than", "instead of", "compared to"
    ]
    reasoning_count = sum(1 for word in reasoning_words if word in rationale_lower)
    score += min(0.4, reasoning_count * 0.05)

    # Pedagogical specificity indicators (0-0.3)
    # Domain-specific terminology that indicates substantive reasoning
    pedagogy_terms = [
        ("bloom", "taxonomy", "cognitive level"),  # Bloom's taxonomy
        ("misconception", "distractor", "plausibility"),  # Assessment design
        ("learner", "student", "pedagog"),  # Learner-centered language
        ("objective", "outcome", "competency"),  # Learning objectives
        ("udl", "accessibility", "inclusive"),  # UDL principles
        ("formative", "summative", "assessment"),  # Assessment types
    ]
    for term_group in pedagogy_terms:
        if any(term in rationale_lower for term in term_group):
            score += 0.05

    return min(1.0, score)


def compute_quality_score(record: Dict[str, Any]) -> float:
    """
    Compute 0.0-1.0 composite quality score for a decision record.

    Weights:
    - Rationale depth: 40% (substantive reasoning)
    - Alternatives present: 20% (considered options)
    - Inputs referenced: 15% (grounded in content)
    - Confidence calibration: 10% (self-assessment)
    - Outcome accepted: 15% (validation result)

    Args:
        record: Decision record dictionary with rationale, alternatives_considered,
                inputs_ref, confidence, and outcome fields

    Returns:
        Float score from 0.0 to 1.0
    """
    rationale = record.get("rationale", "")
    alternatives = record.get("alternatives_considered", [])
    inputs_ref = record.get("inputs_ref", [])
    confidence = record.get("confidence", 0.5)
    outcome = record.get("outcome", {})

    # Compute component scores
    scores = {
        "rationale_depth": score_rationale_depth(rationale),
        "alternatives_present": 1.0 if alternatives else 0.0,
        "inputs_referenced": 1.0 if inputs_ref else 0.0,
        "confidence_calibration": confidence if 0 <= confidence <= 1 else 0.5,
        "outcome_accepted": 1.0 if outcome.get("accepted") else 0.5,
    }

    # Apply weights
    weights = {
        "rationale_depth": 0.4,
        "alternatives_present": 0.2,
        "inputs_referenced": 0.15,
        "confidence_calibration": 0.1,
        "outcome_accepted": 0.15,
    }

    return sum(scores[k] * weights[k] for k in scores)


def compute_quality_breakdown(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute detailed quality breakdown for a decision record.

    Args:
        record: Decision record dictionary

    Returns:
        Dict with component scores and overall assessment
    """
    rationale = record.get("rationale", "")
    alternatives = record.get("alternatives_considered", [])
    inputs_ref = record.get("inputs_ref", [])
    confidence = record.get("confidence", 0.5)
    outcome = record.get("outcome", {})

    depth_score = score_rationale_depth(rationale)
    composite_score = compute_quality_score(record)

    return {
        "rationale_length": len(rationale) if rationale else 0,
        "rationale_depth_score": round(depth_score, 3),
        "has_alternatives": bool(alternatives),
        "alternatives_count": len(alternatives) if alternatives else 0,
        "has_inputs_ref": bool(inputs_ref),
        "inputs_count": len(inputs_ref) if inputs_ref else 0,
        "confidence": confidence,
        "outcome_accepted": outcome.get("accepted", False),
        "composite_score": round(composite_score, 3),
        "quality_level": assess_decision_quality(
            rationale,
            inputs_ref,
            alternatives
        ),
    }


def quality_level_to_order(quality_level: str) -> int:
    """Convert quality level string to numeric order for comparison.

    Args:
        quality_level: Quality level string

    Returns:
        Integer order (higher = better quality)
    """
    return QUALITY_ORDER.get(quality_level, -1)


def compare_quality_levels(level_a: str, level_b: str) -> int:
    """Compare two quality levels.

    Args:
        level_a: First quality level
        level_b: Second quality level

    Returns:
        -1 if a < b, 0 if a == b, 1 if a > b
    """
    order_a = quality_level_to_order(level_a)
    order_b = quality_level_to_order(level_b)

    if order_a < order_b:
        return -1
    elif order_a > order_b:
        return 1
    return 0
