"""
Question Quality Validator

Validates individual question quality based on actual content properties
rather than structural checks alone. Assesses:
- Stem grounding in source content
- Distractor plausibility and distinctness
- Answer specificity
- Bloom verb alignment
- Feedback quality

Referenced by: config/workflows.yaml (rag_training question_quality gate)
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, Set

from lib.validators.bloom import detect_bloom_level
from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_question_quality_decision(
    capture: Any,
    *,
    question_id: str,
    stem_grounding: float,
    distractor_pairwise_max: float,
    distractor_count: int,
    score: float,
    passed: bool,
    issue_codes: List[str],
) -> None:
    """Emit one ``question_quality_check`` per question scored.

    Per H3 W5 contract: per-question cardinality. Dynamic signals:
    question_id, stem_grounding_jaccard, distractor_pairwise_jaccard,
    score.
    """
    if capture is None:
        return
    decision = "passed" if passed else "failed:" + (issue_codes[0] if issue_codes else "low_score")
    rationale = (
        f"QuestionQualityValidator scored question {question_id!r}: "
        f"stem_grounding_jaccard={stem_grounding:.4f}, "
        f"distractor_pairwise_jaccard_max={distractor_pairwise_max:.4f}, "
        f"distractor_count={distractor_count}, "
        f"composite_score={score:.4f}, issue_codes={issue_codes!r}, "
        f"per_question_passed={passed}."
    )
    metrics: Dict[str, Any] = {
        "question_id": question_id,
        "stem_grounding_jaccard": float(stem_grounding),
        "distractor_pairwise_jaccard_max": float(distractor_pairwise_max),
        "distractor_count": int(distractor_count),
        "score": float(score),
        "issue_codes": list(issue_codes),
        "passed": bool(passed),
    }
    try:
        capture.log_decision(
            decision_type="question_quality_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on question_quality_check: %s",
            exc,
        )


def _strip_html(text: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_words(text: str) -> Set[str]:
    """Extract meaningful content words (lowercase, no stop words)."""
    STOPS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "this", "that", "it", "its", "as", "if", "not", "no", "do", "does",
        "has", "have", "had", "will", "would", "can", "could", "should",
        "which", "who", "what", "how", "why", "your", "you", "we", "they",
    }
    words = set(re.findall(r"\b[a-z]{3,}\b", text.lower()))
    return words - STOPS


def _jaccard(a: Set[str], b: Set[str]) -> float:
    """Jaccard similarity between two sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class QuestionQualityValidator:
    """Validates question quality based on content properties.

    Scoring dimensions:
      - Stem grounding (0.25): stem overlaps with source content
      - Distractor plausibility (0.25): distractors from content, not placeholders
      - Answer specificity (0.20): answer is substantive
      - Bloom verb alignment (0.15): stem cognitive demand matches declared level
      - Feedback quality (0.15): feedback references actual content
    """

    name = "question_quality"
    version = "1.0.0"

    DIMENSION_WEIGHTS = {
        "stem_grounding": 0.25,
        "distractor_plausibility": 0.25,
        "answer_specificity": 0.20,
        "bloom_alignment": 0.15,
        "feedback_quality": 0.15,
    }

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate question quality.

        Expected inputs:
            assessment_data: Assessment dict with 'questions' list
            source_chunks: Optional list of source chunk dicts
            min_score: Minimum quality score (default 0.6)
        """
        gate_id = inputs.get("gate_id", "question_quality")
        min_score = inputs.get("min_score", 0.6)
        capture = inputs.get("decision_capture")
        start_time = time.time()
        issues: List[GateIssue] = []

        data = inputs.get("assessment_data")
        if not data or not data.get("questions"):
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="NO_QUESTIONS",
                        message="No questions to validate",
                    )
                ],
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        source_chunks = inputs.get("source_chunks", [])
        # Build content word set from all source chunks
        all_content_words: Set[str] = set()
        for chunk in source_chunks:
            text = chunk.get("text", "")
            all_content_words |= _content_words(_strip_html(text))

        questions = data["questions"]
        dimension_scores: Dict[str, List[float]] = {
            dim: [] for dim in self.DIMENSION_WEIGHTS
        }

        # Wave H3-W5 wiring: emit one ``question_quality_check``
        # decision per question scored so post-hoc replay can
        # reconstruct per-question grounding + distractor-pairwise
        # signals.
        for q in questions:
            q_id = q.get("question_id", "unknown")
            q_issues, q_scores = self._score_question(
                q, all_content_words, q_id
            )
            issues.extend(q_issues)
            for dim, score in q_scores.items():
                dimension_scores[dim].append(score)

            # Per-question composite score (weighted average of the
            # five dimensions for THIS question only) for the capture.
            q_composite = sum(
                self.DIMENSION_WEIGHTS[d] * q_scores.get(d, 0.0)
                for d in self.DIMENSION_WEIGHTS
            )

            # Distractor pairwise max jaccard for the dynamic signal
            # (re-derived here to avoid threading it out of
            # _score_question).
            distractor_pairwise_max = 0.0
            distractor_count = 0
            if q.get("question_type") == "multiple_choice":
                choices = q.get("choices", [])
                distractors = [
                    _strip_html(c.get("text", ""))
                    for c in choices
                    if not c.get("is_correct")
                ]
                distractor_count = len(distractors)
                for i in range(distractor_count):
                    t1 = _content_words(distractors[i])
                    for j in range(i + 1, distractor_count):
                        t2 = _content_words(distractors[j])
                        sim = _jaccard(t1, t2)
                        if sim > distractor_pairwise_max:
                            distractor_pairwise_max = sim

            issue_codes = sorted({i.code for i in q_issues if i.code})
            q_passed = q_composite >= min_score
            _emit_question_quality_decision(
                capture,
                question_id=str(q_id),
                stem_grounding=float(q_scores.get("stem_grounding", 0.0)),
                distractor_pairwise_max=distractor_pairwise_max,
                distractor_count=distractor_count,
                score=float(q_composite),
                passed=q_passed,
                issue_codes=issue_codes,
            )

        # Compute weighted average across all questions and dimensions
        overall = 0.0
        for dim, weight in self.DIMENSION_WEIGHTS.items():
            scores = dimension_scores[dim]
            dim_avg = sum(scores) / len(scores) if scores else 0.0
            overall += weight * dim_avg

        passed = overall >= min_score

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=round(overall, 3),
            issues=issues,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )

    def _score_question(
        self,
        q: Dict[str, Any],
        content_words: Set[str],
        q_id: str,
    ) -> tuple:
        """Score a single question across all dimensions.

        Returns (issues, dimension_scores).
        """
        issues: List[GateIssue] = []
        scores: Dict[str, float] = {}

        stem_text = _strip_html(q.get("stem", ""))
        q_type = q.get("question_type", "")
        bloom_level = q.get("bloom_level", "")

        # 1. Stem grounding: overlap with source content
        if content_words:
            stem_words = _content_words(stem_text)
            grounding = _jaccard(stem_words, content_words)
            # Boost: any overlap is better than none
            scores["stem_grounding"] = min(1.0, grounding * 5)
        else:
            # No source chunks to check against — give neutral score
            scores["stem_grounding"] = 0.5

        if scores["stem_grounding"] < 0.1 and content_words:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="UNGROUNDED_STEM",
                    message=f"{q_id}: stem shares no content words with source material",
                )
            )

        # 2. Distractor plausibility (MCQ only)
        if q_type == "multiple_choice":
            choices = q.get("choices", [])
            correct_choices = [c for c in choices if c.get("is_correct")]
            distractor_choices = [c for c in choices if not c.get("is_correct")]

            if not distractor_choices:
                scores["distractor_plausibility"] = 0.0
            else:
                d_scores = []
                correct_text = _strip_html(
                    correct_choices[0].get("text", "")
                ) if correct_choices else ""
                correct_words = _content_words(correct_text)

                for d in distractor_choices:
                    d_text = _strip_html(d.get("text", ""))
                    d_words = _content_words(d_text)
                    d_score = 1.0

                    # Penalize very short distractors
                    if len(d_text.split()) < 3:
                        d_score -= 0.4
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="SHORT_DISTRACTOR",
                                message=f"{q_id}: distractor too short: '{d_text[:50]}'",
                            )
                        )

                    # Penalize distractors too similar to correct answer
                    if correct_words:
                        sim = _jaccard(d_words, correct_words)
                        if sim > 0.9:
                            d_score -= 0.5
                            issues.append(
                                GateIssue(
                                    severity="warning",
                                    code="DUPLICATE_DISTRACTOR",
                                    message=f"{q_id}: distractor nearly identical to correct answer",
                                )
                            )

                    # Penalize distractors with no content overlap at all
                    if content_words and d_words:
                        content_overlap = _jaccard(d_words, content_words)
                        if content_overlap < 0.01:
                            d_score -= 0.3

                    d_scores.append(max(0.0, d_score))

                # Check pairwise distractor distinctness
                for i in range(len(distractor_choices)):
                    for j in range(i + 1, len(distractor_choices)):
                        t1 = _content_words(_strip_html(distractor_choices[i].get("text", "")))
                        t2 = _content_words(_strip_html(distractor_choices[j].get("text", "")))
                        if _jaccard(t1, t2) > 0.8:
                            issues.append(
                                GateIssue(
                                    severity="warning",
                                    code="SIMILAR_DISTRACTORS",
                                    message=f"{q_id}: distractors {i} and {j} are too similar",
                                )
                            )

                scores["distractor_plausibility"] = (
                    sum(d_scores) / len(d_scores) if d_scores else 0.0
                )
        else:
            # Non-MCQ: full marks for distractor dimension
            scores["distractor_plausibility"] = 1.0

        # 3. Answer specificity
        if q_type == "multiple_choice":
            correct = [c for c in q.get("choices", []) if c.get("is_correct")]
            answer_text = _strip_html(correct[0].get("text", "")) if correct else ""
        else:
            answer_text = q.get("correct_answer", "")

        if answer_text:
            word_count = len(answer_text.split())
            if word_count >= 5:
                scores["answer_specificity"] = 1.0
            elif word_count >= 2:
                scores["answer_specificity"] = 0.6
            else:
                scores["answer_specificity"] = 0.3
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="VAGUE_ANSWER",
                        message=f"{q_id}: answer is very short ({word_count} words)",
                    )
                )
        else:
            # Essay/short answer may not have a preset correct_answer
            if q_type in ("essay", "short_answer"):
                scores["answer_specificity"] = 0.8  # neutral
            else:
                scores["answer_specificity"] = 0.3

        # 4. Bloom verb alignment
        detected = detect_bloom_level(stem_text)
        if detected and bloom_level:
            if detected == bloom_level.lower():
                scores["bloom_alignment"] = 1.0
            else:
                scores["bloom_alignment"] = 0.4
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="BLOOM_MISMATCH",
                        message=(
                            f"{q_id}: declared '{bloom_level}' but stem "
                            f"detected as '{detected}'"
                        ),
                    )
                )
        elif detected:
            scores["bloom_alignment"] = 0.7  # detected but no declared level
        else:
            scores["bloom_alignment"] = 0.5  # no verbs detected

        # 5. Feedback quality
        feedback = _strip_html(q.get("feedback", ""))
        if feedback:
            fb_words = _content_words(feedback)
            fb_len = len(feedback.split())
            if fb_len >= 10 and (not content_words or _jaccard(fb_words, content_words) > 0.02):
                scores["feedback_quality"] = 1.0
            elif fb_len >= 5:
                scores["feedback_quality"] = 0.6
            else:
                scores["feedback_quality"] = 0.3
        else:
            scores["feedback_quality"] = 0.0
            issues.append(
                GateIssue(
                    severity="warning",
                    code="NO_FEEDBACK",
                    message=f"{q_id}: question has no feedback",
                )
            )

        return issues, scores
