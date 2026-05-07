"""Wave 7 W7.A — shared per-question-type segmentation for W3.F evaluators.

Single source of truth for question-type bucketing across the 6
post-training adapter eval surfaces. Mirrors W6.A's
``_normalize_question_type`` resolution chain (generator's
``question_type`` field, defensive QTI ``type`` fallback) and W6.B's
``_bucket_by_type`` projection so the eval-time and gate-time
question-type identities stay aligned.

Note: this module deliberately does NOT import from
``lib.validators.assessment`` — the eval surface should not depend on
validator modules (keeps the eval module DAG clean). The resolution
chain is duplicated by intent.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List


#: Question types where each W3.F metric is structurally meaningful.
#: Consumed by :func:`attach_relevance` so the per-type emit shape carries
#: a ``relevant: bool`` flag per bucket; the W7.D gate validator skips
#: non-relevant buckets.
RELEVANT_QUESTION_TYPES: Dict[str, set] = {
    "answerable_rate": {
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "fill_in_blank",
    },
    "single_correct_rate": {"multiple_choice", "true_false"},
    "distractor_entropy": {"multiple_choice"},
    "bloom_alignment_rate": {
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "fill_in_blank",
    },
    "placeholder_rate": {
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "fill_in_blank",
    },
    "source_support_rate": {
        "multiple_choice",
        "true_false",
        "short_answer",
        "essay",
        "fill_in_blank",
    },
}


def normalize_question_type(value: Any) -> str:
    """Resolve a question_type string from a prompt-dict-like input.

    Mirrors :func:`lib.validators.assessment._normalize_question_type`
    but does not import it — the eval surface should not depend on
    validator modules (keeps the eval module DAG clean).

    Accepts either:

    * a dict — reads ``question_type`` first, falls back to ``type``
      (the QTI alias), lowercased;
    * ``None`` — returns ``""``;
    * any other value — coerced via ``str(...)`` and lowercased.
    """
    if isinstance(value, dict):
        return str(value.get("question_type") or value.get("type") or "").lower()
    if value is None:
        return ""
    return str(value).lower()


def bucket_per_question_records(
    records: List[Dict[str, Any]],
    *,
    question_types: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Group per-question records by their (parallel-indexed) question_type.

    Index alignment is the caller's contract:
    ``len(records) == len(question_types)``.

    Records whose corresponding question_type normalises to the empty
    string bucket under ``""`` — callers may choose to skip that bucket
    when emitting the per-type block (the answerable_rate /
    placeholder_rate extensions skip it; future evaluators may surface
    it as a "type-unknown" bucket).

    Raises:
        ValueError: when records and question_types disagree on length.
    """
    if len(records) != len(question_types):
        raise ValueError(
            f"records / question_types length mismatch: "
            f"{len(records)} vs {len(question_types)}"
        )
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record, qt in zip(records, question_types):
        buckets[normalize_question_type(qt)].append(record)
    return dict(buckets)


def attach_relevance(
    per_question_type: Dict[str, Dict[str, Any]],
    *,
    metric_name: str,
) -> Dict[str, Dict[str, Any]]:
    """Stamp ``relevant: bool`` on each bucket per
    :data:`RELEVANT_QUESTION_TYPES`.

    Mutates the input dict in place AND returns it (chaining
    convenience). Buckets whose key is not in the metric's relevant set
    get ``relevant=False``; buckets whose key IS in the relevant set
    get ``relevant=True``. Unknown ``metric_name`` is treated as the
    empty relevant set (every bucket marked irrelevant — the safe
    default for an evaluator that hasn't declared its relevance table
    yet).
    """
    relevant_set = RELEVANT_QUESTION_TYPES.get(metric_name, set())
    for qt, payload in per_question_type.items():
        payload["relevant"] = qt in relevant_set
    return per_question_type


__all__ = [
    "RELEVANT_QUESTION_TYPES",
    "normalize_question_type",
    "bucket_per_question_records",
    "attach_relevance",
]
