"""Assessment-dimension for ``quality_report.json``.

Wave 26 adds a pedagogical-quality view of the generated assessments to
``quality_report.json`` so a human reviewer can see WHICH question is
broken, not just an aggregate score. Populated from the same validator
calls used at phase gates:

- :class:`lib.validators.assessment.AssessmentQualityValidator` — stem
  diversity, correct-answer diversity, TOC fragments, verb-less stems,
  templated distractors.
- :class:`lib.validators.bloom.BloomAlignmentValidator` — per-question
  Bloom alignment (strict mode: verb-less stems count as UNALIGNED).

The module is callable as a pure function — ``build_assessment_dimension
(assessment_dict)`` — so the caller (``CourseProcessor._write_metadata``
or a test harness) owns when to invoke it.

Shape (see ``Trainforge/tests/test_quality_report_assessment_dimension.py``):

.. code-block:: json

    {
        "total_questions": 10,
        "distinct_stems": 10,
        "distinct_correct_answers": 9,
        "distinct_stem_ratio": 1.0,
        "distinct_correct_answer_ratio": 0.9,
        "avg_distractor_entropy": 0.82,
        "bloom_distribution_observed": {"remember": 3, "understand": 4, ...},
        "objective_coverage_ratio": 0.9,
        "per_question_issues": [
            {"question_id": "q-001", "issues": ["TOC_FRAGMENT_ANSWER", ...]}
        ]
    }
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _correct_answer_for(q: Dict[str, Any]) -> str:
    """Return the canonical correct-answer text for a question, or ""."""
    ca = q.get("correct_answer")
    if ca:
        return _strip_html(ca).lower()
    for c in q.get("choices", []) or []:
        if c.get("is_correct"):
            return _strip_html(c.get("text", "")).lower()
    return ""


def _distractors_for(q: Dict[str, Any]) -> List[str]:
    """Return the list of distractor text strings (lowercased, stripped)."""
    distractors: List[str] = []
    for c in q.get("choices", []) or []:
        if c.get("is_correct"):
            continue
        t = _strip_html(c.get("text", ""))
        if t:
            distractors.append(t.lower())
    return distractors


def _shannon_entropy(counts: Dict[str, int]) -> float:
    """Shannon entropy (base 2) of a discrete distribution."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts.values():
        if c == 0:
            continue
        p = c / total
        entropy -= p * math.log2(p)
    return entropy


def _normalized_entropy(texts: List[str]) -> float:
    """Return normalized Shannon entropy in [0, 1].

    1.0 == all distinct (maximum diversity).
    0.0 == single repeated value.
    """
    if not texts:
        return 0.0
    counts: Counter = Counter(texts)
    if len(counts) <= 1:
        return 0.0
    ent = _shannon_entropy(counts)
    max_ent = math.log2(len(counts)) if len(counts) > 1 else 1.0
    # But for a uniform distribution the max is log2(N)=log2(len(texts))
    # in the best case. Normalize against log2(len(texts)) so a truly
    # uniform distribution scores 1.0.
    max_possible = math.log2(len(texts)) if len(texts) > 1 else 1.0
    if max_possible == 0:
        return 0.0
    return min(1.0, ent / max_possible)


def _per_question_issues(
    assessment: Dict[str, Any],
    validator_result: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Group :class:`GateIssue` codes per question_id.

    Uses the assessment validator's ``message`` field's leading ``"{q_id}: "``
    prefix to re-associate cross-question checks (which don't name a single
    question) with all questions that contributed. For those, we emit a
    pseudo-entry with ``question_id=None`` so the report still surfaces the
    issue.

    If ``validator_result`` is provided, its issues are the source of
    truth. Otherwise we run AssessmentQualityValidator + BloomAlignment
    strict.
    """
    # Import here to avoid circular import at module load time.
    from lib.validators.assessment import AssessmentQualityValidator
    from lib.validators.bloom import BloomAlignmentValidator

    issues_by_qid: Dict[Optional[str], List[str]] = defaultdict(list)

    # 1. Assessment quality validator issues
    if validator_result is None:
        aqv = AssessmentQualityValidator()
        aqv_result = aqv.validate({
            "assessment_data": assessment,
            "min_score": 0.8,
        })
    else:
        aqv_result = validator_result

    for issue in aqv_result.issues:
        msg = issue.message or ""
        m = re.match(r"^([A-Za-z0-9_\-]+):\s", msg)
        qid: Optional[str] = m.group(1) if m else None
        issues_by_qid[qid].append(issue.code)

    # 2. Strict-mode Bloom alignment diagnostics
    bav = BloomAlignmentValidator()
    bav_result = bav.validate({
        "assessment_data": assessment,
        "min_alignment_score": 0.7,
        "permissive_mode": False,
    })
    for issue in bav_result.issues:
        msg = issue.message or ""
        # Bloom validator emits "Question {q_id}: ..."
        m = re.match(r"^Question\s+([A-Za-z0-9_\-]+):", msg)
        qid = m.group(1) if m else None
        if issue.code not in issues_by_qid.get(qid, []):
            issues_by_qid[qid].append(issue.code)

    # Flatten into serializable list, questions first then cross-question
    out: List[Dict[str, Any]] = []
    questions = assessment.get("questions", []) or []
    # Keep original question order.
    known_qids = [q.get("question_id", "") for q in questions]
    for qid in known_qids:
        codes = issues_by_qid.get(qid, [])
        if codes:
            out.append({"question_id": qid, "issues": sorted(set(codes))})
    # Cross-question (qid is None) or unmatched
    cross_codes = []
    for qid, codes in issues_by_qid.items():
        if qid not in known_qids:
            cross_codes.extend(codes)
    if cross_codes:
        out.append({
            "question_id": None,
            "issues": sorted(set(cross_codes)),
        })
    return out


def build_assessment_dimension(
    assessment: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build the ``assessments`` dimension for ``quality_report.json``.

    Args:
        assessment: Parsed assessments.json dict (single-assessment shape,
            with a ``questions`` list). ``None`` or a dict with no
            questions returns ``None`` so the caller can cleanly omit the
            dimension from the report.

    Returns:
        The dimension dict, or ``None`` when no assessments are available.
    """
    if not assessment:
        return None
    questions = assessment.get("questions") or []
    if not questions:
        return None

    total = len(questions)
    stems = [_strip_html(q.get("stem", "")).lower() for q in questions]
    stems = [s for s in stems if s]
    correct_answers = [_correct_answer_for(q) for q in questions]
    correct_answers_nonempty = [a for a in correct_answers if a]

    distinct_stems = len(set(stems))
    distinct_correct_answers = len(set(correct_answers_nonempty))

    distinct_stem_ratio = (
        round(distinct_stems / len(stems), 3) if stems else 0.0
    )
    distinct_correct_answer_ratio = (
        round(distinct_correct_answers / len(correct_answers_nonempty), 3)
        if correct_answers_nonempty else 0.0
    )

    # Average per-question distractor entropy (normalized)
    per_q_entropies = [_normalized_entropy(_distractors_for(q)) for q in questions]
    avg_distractor_entropy = (
        round(sum(per_q_entropies) / len(per_q_entropies), 3)
        if per_q_entropies else 0.0
    )

    # Observed Bloom distribution
    bloom_distribution_observed: Dict[str, int] = defaultdict(int)
    for q in questions:
        lvl = q.get("bloom_level") or "unknown"
        bloom_distribution_observed[lvl] += 1

    # Objective coverage ratio (observed vs targeted)
    objectives_targeted = assessment.get("objectives_targeted") or []
    covered = {q.get("objective_id") for q in questions if q.get("objective_id")}
    if objectives_targeted:
        hit = len(covered & set(objectives_targeted))
        objective_coverage_ratio = round(hit / len(objectives_targeted), 3)
    else:
        objective_coverage_ratio = 1.0 if covered else 0.0

    dimension: Dict[str, Any] = {
        "total_questions": total,
        "distinct_stems": distinct_stems,
        "distinct_correct_answers": distinct_correct_answers,
        "distinct_stem_ratio": distinct_stem_ratio,
        "distinct_correct_answer_ratio": distinct_correct_answer_ratio,
        "avg_distractor_entropy": avg_distractor_entropy,
        "bloom_distribution_observed": dict(bloom_distribution_observed),
        "objective_coverage_ratio": objective_coverage_ratio,
        "per_question_issues": _per_question_issues(assessment),
    }
    return dimension
