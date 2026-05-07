"""
Assessment Quality Validators

Validates generated assessments for quality and alignment:

AssessmentQualityValidator:
- Question clarity and unambiguity
- Distractor quality (plausible, misconception-based)
- Answer correctness
- Coverage of learning objectives
- Appropriate difficulty distribution

Placeholder detection (PLACEHOLDER_QUESTION / PLACEHOLDER_CHOICE /
PLACEHOLDER_ANSWER / PLACEHOLDER_FEEDBACK) emits ``severity="critical"``
as fail-closed defense-in-depth. Worker W1 killed the runtime emit
path that produced placeholder strings, so these codes should never
fire on a healthy run; promotion to ``critical`` (Worker W4) means
any external IMSCC import OR a future regression that re-introduces
placeholder content fails the ``assessment_quality`` gate immediately
rather than degrading the score-based pass threshold.

FinalQualityValidator:
- End-to-end quality check after all generation
- Cross-assessment consistency
- No duplicate questions
- Minimum quality score threshold

Referenced by: config/workflows.yaml (rag_training, textbook_to_course)
"""

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from lib.validators.bloom import detect_bloom_level
from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


def _emit_assessment_quality_decision(
    capture: Any,
    *,
    question_id: str,
    passed: bool,
    placeholder_hits: int,
    bloom_level: str,
    is_mcq: bool,
    issue_codes: List[str],
    question_type: str = "",
    per_type_thresholds_used: Optional[Dict[str, float]] = None,
    bucket_size: int = 0,
) -> None:
    """Emit one ``assessment_quality_check`` decision per question audited.

    Per H3 W5 contract: per-question cardinality, dynamic signals
    interpolated (question_id, placeholder_hits, bloom_level, is_mcq,
    issue_codes). Rationale length >= 60 chars to avoid the static-
    rationale regression class.

    Wave 6 W6.A extension: ``question_type``, ``per_type_thresholds_used``,
    and ``bucket_size`` are interpolated into the rationale so post-hoc
    replay can reconstruct the per-bucket dispatch (no new
    ``decision_type`` enum member — Lesson 3 from W5 prefers extending
    the existing rationale over minting a new value when the surface
    is mechanically the same).
    """
    if capture is None:
        return
    decision = "passed" if passed else "failed:" + (issue_codes[0] if issue_codes else "unknown")
    type_used = per_type_thresholds_used or {}
    rationale = (
        f"AssessmentQualityValidator audited question {question_id!r}: "
        f"placeholder_hits={placeholder_hits}, bloom_level={bloom_level or 'n/a'}, "
        f"is_mcq={is_mcq}, issue_codes={issue_codes!r}, "
        f"per_question_passed={passed}, "
        f"question_type={question_type or 'unknown'!r}, "
        f"bucket_size={bucket_size}, "
        f"per_type_thresholds_used={type_used!r}."
    )
    metrics: Dict[str, Any] = {
        "question_id": question_id,
        "placeholder_hits": int(placeholder_hits),
        "bloom_level": bloom_level or "",
        "is_mcq": bool(is_mcq),
        "issue_codes": list(issue_codes),
        "passed": bool(passed),
        "question_type": question_type or "",
        "per_type_thresholds_used": dict(type_used),
        "bucket_size": int(bucket_size),
    }
    try:
        capture.log_decision(
            decision_type="assessment_quality_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001 — capture must not break the gate
        logger.debug(
            "DecisionCapture.log_decision raised on assessment_quality_check: %s",
            exc,
        )


def _emit_final_quality_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    total_questions: int,
    n_assessments: int,
    duplicate_count: int,
    score: float,
    min_score: float,
) -> None:
    """Emit one corpus-wide ``final_quality_check`` per validate() call.

    Per H3 W5 contract: corpus-wide cardinality. Dynamic signals:
    total_questions, passed_count proxy via score, error_rate proxy
    via duplicate_count.
    """
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rationale = (
        f"FinalQualityValidator corpus-wide verdict: "
        f"n_assessments={n_assessments}, total_questions={total_questions}, "
        f"duplicate_stem_count={duplicate_count}, score={score:.4f}, "
        f"min_score={min_score:.4f}, failure_code={code or 'none'}."
    )
    metrics: Dict[str, Any] = {
        "n_assessments": int(n_assessments),
        "total_questions": int(total_questions),
        "duplicate_count": int(duplicate_count),
        "score": float(score),
        "min_score": float(min_score),
        "passed": bool(passed),
        "failure_code": code,
    }
    try:
        capture.log_decision(
            decision_type="final_quality_check",
            decision=decision,
            rationale=rationale,
            context=str(metrics),
            metrics=metrics,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on final_quality_check: %s",
            exc,
        )

ASSESSMENT_PLACEHOLDER_PATTERNS = [
    re.compile(r"Correct answer based on content", re.IGNORECASE),
    re.compile(r"Plausible distractor [A-C]", re.IGNORECASE),
    re.compile(r"Statement about .+ content\.", re.IGNORECASE),
    re.compile(r"The key concept from .+ is _______", re.IGNORECASE),
    re.compile(r"the concept from (?:LO-|INT|[A-Z]{2,})", re.IGNORECASE),
    re.compile(r"^Briefly \w+ the key points from ", re.IGNORECASE),
    re.compile(r"concepts from .+ and provide examples\.", re.IGNORECASE),
    re.compile(r"^concept term$", re.IGNORECASE),
    re.compile(r"Review content for objective ", re.IGNORECASE),
    re.compile(r"This statement is accurate based on ", re.IGNORECASE),
    re.compile(r"The correct term is found in .+ content", re.IGNORECASE),
    re.compile(r"A complete response should address all aspects of ", re.IGNORECASE),
    re.compile(r"Your response should cover the main concepts from ", re.IGNORECASE),
]


# Wave 26 real-failure-mode thresholds
STEM_DIVERSITY_THRESHOLD = 0.7
CORRECT_ANSWER_DIVERSITY_THRESHOLD = 0.6
DISTRACTOR_TEMPLATE_MAX_RATIO = 0.30
# TOC fragment: three standalone integers inline ("1.1 Something 14 1.7 ...")
_TOC_THREE_INTS_RE = re.compile(r"\b\d+\b\s+\S+.*\b\d+\b.*\b\d+\b", re.DOTALL)
_CHAPTER_HEADING_RE = re.compile(r"\b\d+\.\d+\b")
_INTEGER_TOKEN_RE = re.compile(r"\b\d+\b")


#: Per-question-type quality thresholds (Wave 6 W6.A calibration).
#: Day-1 starting points; calibrate against the rdf-shacl-551-2 corpus
#: rebuild before promoting severity from warning to critical.
#:
#: Mirrors :data:`lib.validators.block_objective_delivery._PER_BLOCK_TYPE_ENTAILMENT_FLOOR`
#: (W1.7.C) and :data:`lib.validators.pair_objective_delivery._PER_PAIR_KIND_ENTAILMENT_FLOOR`
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


def _normalize_question_type(q: Dict[str, Any]) -> str:
    """Resolve question_type, tolerating QTI-parsed ``type`` field.

    W4.B Lesson 1 (pair-schema field-name divergence):
    ``Trainforge/parsers/qti_parser.py:24`` uses ``type:`` while
    ``Trainforge/generators/assessment_generator.py:101`` uses
    ``question_type:``. Mirrors W4.B's ``lo_refs`` resolution chain
    (``pair.get("lo_refs") or pair.get("learning_outcome_refs")``).
    Generator surface always emits ``question_type``; QTI surface
    always emits ``type``. Day-1 only generator-emit reaches this
    validator; this helper is defense-in-depth against a future
    call site that routes QTI-parsed dicts directly.
    """
    if not isinstance(q, dict):
        return ""
    return str(q.get("question_type") or q.get("type") or "").lower()


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


def _strip_html_text(s: str) -> str:
    """Helper: strip HTML tags and normalize whitespace."""
    if not s:
        return ""
    return re.sub(r"<[^>]+>", "", s).strip()


def _looks_like_toc_fragment(answer_text: str) -> bool:
    """Return True if answer_text looks like a raw TOC fragment.

    Matches when the string contains either:
      - Three standalone integers inline (page numbers), OR
      - Is > 500 chars AND has >= 3 integers AND >= 2 dotted-numeric
        headings like ``1.1`` / ``4.2``.
    """
    if not answer_text:
        return False
    text = _strip_html_text(answer_text)
    if _TOC_THREE_INTS_RE.search(text):
        return True
    if len(text) > 500:
        int_count = len(_INTEGER_TOKEN_RE.findall(text))
        heading_count = len(_CHAPTER_HEADING_RE.findall(text))
        if int_count >= 3 and heading_count >= 2:
            return True
    return False


class AssessmentQualityValidator:
    """Validates individual assessment quality."""

    name = "assessment_quality"
    version = "1.2.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate assessment quality.

        Expected inputs:
            assessment_path: Path to assessment JSON
            assessment_data: Assessment dict (alternative to path)
            learning_objectives: List of target objectives (optional)
            min_score: Minimum quality score (default 0.8)
        """
        gate_id = inputs.get("gate_id", "assessment_quality")
        issues: List[GateIssue] = []
        min_score = inputs.get("min_score", 0.8)
        capture = inputs.get("decision_capture")

        # Load assessment data
        data = inputs.get("assessment_data")
        if not data and inputs.get("assessment_path"):
            path = Path(inputs["assessment_path"])
            if not path.exists():
                return GateResult(
                    gate_id=gate_id,
                    validator_name=self.name,
                    validator_version=self.version,
                    passed=False,
                    issues=[
                        GateIssue(
                            severity="error",
                            code="FILE_NOT_FOUND",
                            message=f"Assessment not found: {path}",
                        )
                    ],
                )
            data = json.loads(path.read_text(encoding="utf-8"))

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
            )

        questions = data["questions"]

        # Wave 6 W6.A: resolve per-question-type threshold table once
        # (operator override merges into the canonical table inside
        # ``_resolve_per_type_thresholds``). The same table threads
        # through ``_check_question`` (per-type ``min_stem_chars``)
        # and ``_check_cross_question_failures`` (per-type bucket
        # diversity sub-checks).
        per_type_thresholds = _resolve_per_type_thresholds(inputs)

        # Pre-compute bucket sizes for decision-capture rationale
        # interpolation (the per-question emit needs to surface which
        # bucket the question landed in, plus the bucket's size, so
        # post-hoc replay can reconstruct the per-bucket dispatch).
        bucket_sizes: Dict[str, int] = Counter(
            _normalize_question_type(q) for q in questions
        )

        # Check each question (per-question issues). Wave H3-W5 wiring:
        # emit one ``assessment_quality_check`` capture event per
        # question audited so post-hoc replay can reconstruct the per-
        # question pass/fail trail (placeholder hits, Bloom level,
        # MCQ flag).
        for q in questions:
            q_type = _normalize_question_type(q)
            q_thresholds = _thresholds_for_type(q_type, per_type_thresholds)
            q_issues = self._check_question(q, q_thresholds)
            issues.extend(q_issues)
            q_id = str(q.get("question_id", "unknown"))
            placeholder_codes = {
                "PLACEHOLDER_QUESTION", "PLACEHOLDER_CHOICE",
                "PLACEHOLDER_ANSWER", "PLACEHOLDER_FEEDBACK",
            }
            placeholder_hits = sum(
                1 for i in q_issues if i.code in placeholder_codes
            )
            issue_codes = sorted({i.code for i in q_issues if i.code})
            q_passed = not any(
                i.severity in ("critical", "error") for i in q_issues
            )
            _emit_assessment_quality_decision(
                capture,
                question_id=q_id,
                passed=q_passed,
                placeholder_hits=placeholder_hits,
                bloom_level=str(q.get("bloom_level") or ""),
                is_mcq=q_type == "multiple_choice",
                issue_codes=issue_codes,
                question_type=q_type,
                per_type_thresholds_used=q_thresholds,
                bucket_size=int(bucket_sizes.get(q_type, 0)),
            )

        # Wave 26: cross-question real-failure-mode checks
        issues.extend(
            self._check_cross_question_failures(questions, per_type_thresholds)
        )

        # Check objective coverage
        target_objectives = inputs.get("learning_objectives", [])
        if target_objectives:
            issues.extend(
                self._check_objective_coverage(questions, target_objectives)
            )

        # Compute score. Critical issues (Wave 26) hard-fail the gate and
        # deduct the most aggressively; legacy "error" severity remains for
        # placeholder regex hits to preserve back-compat score behavior.
        critical_count = sum(1 for i in issues if i.severity == "critical")
        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        score = max(
            0.0,
            1.0
            - critical_count * 0.15
            - error_count * 0.15
            - warning_count * 0.05,
        )
        # Wave 26: any critical flips passed to False regardless of score.
        passed = score >= min_score and critical_count == 0

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    def _check_question(
        self,
        q: Dict[str, Any],
        type_thresholds: Optional[Dict[str, float]] = None,
    ) -> List[GateIssue]:
        """Check a single question for quality issues.

        Wave 6 W6.A: ``type_thresholds`` is the per-question-type
        threshold dict resolved by the validator's ``validate()`` body
        via :func:`_resolve_per_type_thresholds` + :func:`_thresholds_for_type`.
        Currently consumed for the ``min_stem_chars`` floor only — every
        other per-question check (placeholder regex, MCQ-specific
        min-3-choices, TOC-fragment, verb-less, placeholder-in-feedback)
        is structural, not calibration-sensitive, so they stay
        type-agnostic. Callers that don't pass a table (legacy direct
        invocations from tests) fall back to the default floor.
        """
        if type_thresholds is None:
            type_thresholds = dict(_DEFAULT_QUESTION_TYPE_THRESHOLDS)
        issues = []
        q_id = q.get("question_id", "unknown")
        stem = q.get("stem", "")
        q_type = _normalize_question_type(q)

        # Check stem is non-empty. Wave 6 W6.A: per-type ``min_stem_chars``
        # floor (essay = 15, MC = 12, T/F + fill_in_blank = 10) replaces
        # the prior hardcoded 10.
        min_stem_chars = int(
            type_thresholds.get(
                "min_stem_chars",
                _DEFAULT_QUESTION_TYPE_THRESHOLDS["min_stem_chars"],
            )
        )
        text = re.sub(r"<[^>]+>", "", stem).strip()
        if len(text) < min_stem_chars:
            issues.append(
                GateIssue(
                    severity="error",
                    code="SHORT_STEM",
                    message=(
                        f"{q_id}: question stem too short ({len(text)} chars; "
                        f"floor={min_stem_chars} for type={q_type or 'unknown'!r})"
                    ),
                )
            )

        # Check for placeholder content in stem
        for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
            if pattern.search(text):
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="PLACEHOLDER_QUESTION",
                        message=f"{q_id}: stem contains placeholder text matching '{pattern.pattern}'",
                    )
                )
                break  # One placeholder hit per question is enough

        # Check MCQ-specific issues
        if q_type == "multiple_choice":
            choices = q.get("choices", [])
            if len(choices) < 3:
                issues.append(
                    GateIssue(
                        severity="error",
                        code="FEW_CHOICES",
                        message=f"{q_id}: MCQ has only {len(choices)} choices (min 3)",
                    )
                )
            correct = [c for c in choices if c.get("is_correct")]
            if len(correct) != 1:
                issues.append(
                    GateIssue(
                        severity="error",
                        code="WRONG_CORRECT_COUNT",
                        message=f"{q_id}: MCQ has {len(correct)} correct answers (need 1)",
                    )
                )

            # Check for placeholder content in choices
            for choice in choices:
                choice_text = re.sub(r"<[^>]+>", "", choice.get("text", "")).strip()
                for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
                    if pattern.search(choice_text):
                        issues.append(
                            GateIssue(
                                severity="critical",
                                code="PLACEHOLDER_CHOICE",
                                message=f"{q_id}: choice contains placeholder text: '{choice_text}'",
                            )
                        )
                        break

        # Check for placeholder in correct_answer (fill-in-blank, T/F)
        correct_answer = q.get("correct_answer", "")
        if correct_answer:
            for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
                if pattern.search(correct_answer):
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="PLACEHOLDER_ANSWER",
                            message=f"{q_id}: correct_answer is placeholder text: '{correct_answer}'",
                        )
                    )
                    break

        # Wave 26: TOC-fragment correct answer check (critical). Applies to
        # correct_answer (fill-in-blank / T/F) AND to any MCQ choice flagged
        # is_correct. Catches raw TOC text like
        # "1.1 Structural changes in the economy 14 1.7 From the periphery".
        candidates: List[str] = []
        if correct_answer:
            candidates.append(correct_answer)
        for c in q.get("choices", []):
            if c.get("is_correct"):
                candidates.append(_strip_html_text(c.get("text", "")))
        for cand in candidates:
            if _looks_like_toc_fragment(cand):
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="TOC_FRAGMENT_ANSWER",
                        message=(
                            f"{q_id}: correct answer looks like a raw TOC "
                            f"fragment (page numbers + chapter headings): "
                            f"'{cand[:120]}{'...' if len(cand) > 120 else ''}'"
                        ),
                    )
                )
                break

        # Wave 26: verb-less stem (warning). T/F questions are allowed one
        # verb-less stem per-assessment — the cross-question pass enforces
        # that cap. Here we just record the finding per question.
        if text and detect_bloom_level(text) is None:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="VERB_LESS_STEM",
                    message=(
                        f"{q_id}: stem has no detectable Bloom verb: "
                        f"'{text[:80]}{'...' if len(text) > 80 else ''}'"
                    ),
                )
            )

        # Check for placeholder in feedback
        feedback = re.sub(r"<[^>]+>", "", q.get("feedback", "")).strip()
        if feedback:
            for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
                if pattern.search(feedback):
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="PLACEHOLDER_FEEDBACK",
                            message=f"{q_id}: feedback contains placeholder text",
                        )
                    )
                    break

        return issues

    def _check_cross_question_failures(
        self,
        questions: List[Dict[str, Any]],
        per_type_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[GateIssue]:
        """Wave 26: cross-question real-failure-mode checks.

        Emits critical issues for:
          - LOW_STEM_DIVERSITY: distinct stem ratio < STEM_DIVERSITY_THRESHOLD
          - LOW_ANSWER_DIVERSITY: distinct correct-answer ratio
            < CORRECT_ANSWER_DIVERSITY_THRESHOLD
          - TEMPLATED_DISTRACTORS: a single distractor string appears on
            >= 30% of questions

        The per-question VERB_LESS_STEM warnings are capped at 1 per
        assessment (allowing a single T/F-style verb-less stem); anything
        above the cap is escalated here.

        Wave 6 W6.A extension: when ``per_type_thresholds`` is supplied,
        questions are ALSO bucketed by ``_normalize_question_type`` and
        per-type diversity sub-checks fire at ``severity="warning"``
        with type-suffixed issue codes
        (``LOW_STEM_DIVERSITY_MULTIPLE_CHOICE``, ``LOW_ANSWER_DIVERSITY_ESSAY``,
        ``TEMPLATED_DISTRACTORS_FILL_IN_BLANK``, …). The legacy
        corpus-wide critical checks remain unchanged so existing
        ``LOW_STEM_DIVERSITY`` / ``LOW_ANSWER_DIVERSITY`` /
        ``TEMPLATED_DISTRACTORS`` regression coverage stays load-bearing
        (``test_assessment_validator_real_failures.py``). Per-type
        warnings are calibration-deferred per plan §5; flip warning →
        critical in a follow-up micro-wave once per-type pass-rate
        distributions are calibrated against the post-Wave-5 corpus
        rebuild.
        """
        if per_type_thresholds is None:
            per_type_thresholds = {
                qt: dict(thresholds)
                for qt, thresholds in _PER_QUESTION_TYPE_THRESHOLDS.items()
            }
        issues: List[GateIssue] = []
        total = len(questions)
        if total == 0:
            return issues

        # 1. Distinct-stem ratio
        stems = []
        for q in questions:
            s = _strip_html_text(q.get("stem", "")).lower()
            if s:
                stems.append(s)
        if stems:
            distinct_ratio = len(set(stems)) / len(stems)
            if distinct_ratio < STEM_DIVERSITY_THRESHOLD:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="LOW_STEM_DIVERSITY",
                        message=(
                            f"Distinct stem ratio {distinct_ratio:.2f} "
                            f"below threshold {STEM_DIVERSITY_THRESHOLD} "
                            f"({len(set(stems))}/{len(stems)} unique)"
                        ),
                    )
                )

        # 2. Distinct correct-answer ratio
        correct_answers: List[str] = []
        for q in questions:
            ca = q.get("correct_answer")
            if ca:
                correct_answers.append(_strip_html_text(ca).lower())
                continue
            for c in q.get("choices", []):
                if c.get("is_correct"):
                    correct_answers.append(
                        _strip_html_text(c.get("text", "")).lower()
                    )
                    break
        if correct_answers:
            distinct_answer_ratio = (
                len(set(correct_answers)) / len(correct_answers)
            )
            if distinct_answer_ratio < CORRECT_ANSWER_DIVERSITY_THRESHOLD:
                issues.append(
                    GateIssue(
                        severity="critical",
                        code="LOW_ANSWER_DIVERSITY",
                        message=(
                            f"Distinct correct-answer ratio "
                            f"{distinct_answer_ratio:.2f} below threshold "
                            f"{CORRECT_ANSWER_DIVERSITY_THRESHOLD} "
                            f"({len(set(correct_answers))}/"
                            f"{len(correct_answers)} unique)"
                        ),
                    )
                )

        # 3. Templated distractors: any single distractor appearing on
        # >= 30% of questions is a template leak.
        distractor_counts: Counter = Counter()
        q_has_distractor: Counter = Counter()
        for q in questions:
            seen_in_q: Set[str] = set()
            for c in q.get("choices", []):
                if c.get("is_correct"):
                    continue
                d = _strip_html_text(c.get("text", "")).lower()
                if d and d not in seen_in_q:
                    seen_in_q.add(d)
                    distractor_counts[d] += 1
            for d in seen_in_q:
                q_has_distractor[d] += 1
        # We count per-question occurrences (q_has_distractor) so a
        # distractor repeated within the same question only counts once.
        questions_with_choices = sum(
            1 for q in questions if q.get("choices")
        )
        if questions_with_choices > 0:
            threshold = DISTRACTOR_TEMPLATE_MAX_RATIO * questions_with_choices
            for template_text, occurrences in q_has_distractor.items():
                if occurrences >= threshold and occurrences >= 2:
                    ratio = occurrences / questions_with_choices
                    issues.append(
                        GateIssue(
                            severity="critical",
                            code="TEMPLATED_DISTRACTORS",
                            message=(
                                f"Distractor template repeated on "
                                f"{occurrences}/{questions_with_choices} "
                                f"({ratio:.0%}) of questions: "
                                f"'{template_text[:80]}"
                                f"{'...' if len(template_text) > 80 else ''}'"
                            ),
                        )
                    )

        # 4. Verb-less cap: allow at most one verb-less stem per assessment
        # (T/F exception). Escalate the rest if needed.
        verbless_q_ids: List[str] = []
        tf_verbless_q_ids: List[str] = []
        for q in questions:
            s = _strip_html_text(q.get("stem", ""))
            if not s:
                continue
            if detect_bloom_level(s) is None:
                q_id = q.get("question_id", "unknown")
                if q.get("question_type") == "true_false":
                    tf_verbless_q_ids.append(q_id)
                else:
                    verbless_q_ids.append(q_id)
        # Allow a single exception total. If both T/F-verbless and
        # non-T/F-verbless exist beyond the budget, escalate a critical.
        total_verbless = len(verbless_q_ids) + len(tf_verbless_q_ids)
        if total_verbless > 1 and len(verbless_q_ids) >= 1:
            issues.append(
                GateIssue(
                    severity="critical",
                    code="PERVASIVE_VERBLESS_STEMS",
                    message=(
                        f"{total_verbless} questions have verb-less stems "
                        f"(of {total} total). Single-exception rule "
                        f"exhausted."
                    ),
                )
            )

        # Wave 6 W6.A: per-question-type bucketing + diversity sub-checks.
        # Buckets with <2 questions skip diversity computation (N=1
        # ratios are mathematically undefined; emitting a "1/1 = 1.0"
        # passes trivially and a "0/1 = 0" trips a meaningless warning).
        # Issue codes carry the type suffix so the wave-end gate can
        # disambiguate from the legacy corpus-wide critical codes.
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for q in questions:
            qt = _normalize_question_type(q)
            buckets.setdefault(qt, []).append(q)
        for qt, bucket_qs in buckets.items():
            if len(bucket_qs) < 2:
                continue
            type_suffix = (qt or "unknown").upper()
            thresholds = _thresholds_for_type(qt, per_type_thresholds)
            issues.extend(
                self._check_diversity_for_bucket(
                    bucket_qs, thresholds, type_suffix
                )
            )

        return issues

    def _check_diversity_for_bucket(
        self,
        bucket_qs: List[Dict[str, Any]],
        type_thresholds: Dict[str, float],
        type_suffix: str,
    ) -> List[GateIssue]:
        """Per-bucket diversity sub-checks (Wave 6 W6.A).

        Mirrors the legacy corpus-wide diversity passes inside
        :meth:`_check_cross_question_failures` but scoped to a single
        ``question_type`` bucket and emitting ``severity="warning"``
        with type-suffixed issue codes. Calibration-deferred per
        plan §5; severity flip to critical happens in a follow-up
        micro-wave once per-type warning-fire rates stabilize on the
        rdf-shacl-551-2 rebuild.

        ``type_suffix`` is the upper-cased question_type for the issue
        code suffix (``MULTIPLE_CHOICE`` / ``TRUE_FALSE`` / etc.).
        """
        issues: List[GateIssue] = []
        n = len(bucket_qs)
        if n < 2:
            return issues

        stem_floor = float(
            type_thresholds.get(
                "stem_diversity",
                _DEFAULT_QUESTION_TYPE_THRESHOLDS["stem_diversity"],
            )
        )
        answer_floor = float(
            type_thresholds.get(
                "correct_answer_diversity",
                _DEFAULT_QUESTION_TYPE_THRESHOLDS["correct_answer_diversity"],
            )
        )
        distractor_max = float(
            type_thresholds.get(
                "distractor_template_max_ratio",
                _DEFAULT_QUESTION_TYPE_THRESHOLDS["distractor_template_max_ratio"],
            )
        )

        # 1. Distinct-stem ratio (per bucket).
        stems = [
            _strip_html_text(q.get("stem", "")).lower()
            for q in bucket_qs
        ]
        stems = [s for s in stems if s]
        if stems:
            ratio = len(set(stems)) / len(stems)
            if ratio < stem_floor:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code=f"LOW_STEM_DIVERSITY_{type_suffix}",
                        message=(
                            f"[{type_suffix}] Distinct stem ratio {ratio:.2f} "
                            f"below per-type floor {stem_floor:.2f} "
                            f"({len(set(stems))}/{len(stems)} unique)"
                        ),
                    )
                )

        # 2. Distinct correct-answer ratio (per bucket).
        correct_answers: List[str] = []
        for q in bucket_qs:
            ca = q.get("correct_answer")
            if ca:
                correct_answers.append(_strip_html_text(ca).lower())
                continue
            for c in q.get("choices", []):
                if c.get("is_correct"):
                    correct_answers.append(
                        _strip_html_text(c.get("text", "")).lower()
                    )
                    break
        if correct_answers:
            ratio = len(set(correct_answers)) / len(correct_answers)
            if ratio < answer_floor:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code=f"LOW_ANSWER_DIVERSITY_{type_suffix}",
                        message=(
                            f"[{type_suffix}] Distinct correct-answer ratio "
                            f"{ratio:.2f} below per-type floor "
                            f"{answer_floor:.2f} ({len(set(correct_answers))}/"
                            f"{len(correct_answers)} unique)"
                        ),
                    )
                )

        # 3. Templated distractors (per bucket). Skip when the type's
        # distractor floor is 1.0 (T/F / SA / essay have no real
        # distractors, so the check is always vacuous).
        if distractor_max < 1.0:
            distractor_per_q: Counter = Counter()
            for q in bucket_qs:
                seen: Set[str] = set()
                for c in q.get("choices", []):
                    if c.get("is_correct"):
                        continue
                    d = _strip_html_text(c.get("text", "")).lower()
                    if d and d not in seen:
                        seen.add(d)
                for d in seen:
                    distractor_per_q[d] += 1
            qs_with_choices = sum(1 for q in bucket_qs if q.get("choices"))
            if qs_with_choices > 0:
                threshold = distractor_max * qs_with_choices
                for template_text, occurrences in distractor_per_q.items():
                    if occurrences >= threshold and occurrences >= 2:
                        ratio = occurrences / qs_with_choices
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code=f"TEMPLATED_DISTRACTORS_{type_suffix}",
                                message=(
                                    f"[{type_suffix}] Distractor template "
                                    f"repeated on {occurrences}/{qs_with_choices} "
                                    f"({ratio:.0%}) of bucket questions: "
                                    f"'{template_text[:80]}"
                                    f"{'...' if len(template_text) > 80 else ''}'"
                                ),
                            )
                        )

        return issues

    def _check_objective_coverage(
        self, questions: List[Dict], targets: List[str]
    ) -> List[GateIssue]:
        """Check that all target objectives are covered."""
        covered: Set[str] = set()
        for q in questions:
            obj = q.get("objective_id", "")
            if obj:
                covered.add(obj)

        missing = set(targets) - covered
        issues = []
        for obj_id in sorted(missing):
            issues.append(
                GateIssue(
                    severity="warning",
                    code="OBJECTIVE_UNCOVERED",
                    message=f"Objective {obj_id} has no assessment items",
                )
            )
        return issues


class FinalQualityValidator:
    """Validates final assessment quality after all generation."""

    name = "final_quality"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate final assessment quality.

        Expected inputs:
            assessments_dir: Path to directory of all assessments
            assessments: List of assessment dicts (alternative)
            min_score: Minimum final quality score (default 0.85)
        """
        gate_id = inputs.get("gate_id", "final_quality")
        issues: List[GateIssue] = []
        min_score = inputs.get("min_score", 0.85)
        capture = inputs.get("decision_capture")

        # Load assessments
        assessments = inputs.get("assessments", [])
        if not assessments and inputs.get("assessments_dir"):
            adir = Path(inputs["assessments_dir"])
            if adir.exists():
                for f in adir.glob("*.json"):
                    try:
                        assessments.append(
                            json.loads(f.read_text(encoding="utf-8"))
                        )
                    except (json.JSONDecodeError, OSError):
                        issues.append(
                            GateIssue(
                                severity="warning",
                                code="LOAD_ERROR",
                                message=f"Failed to load {f.name}",
                            )
                        )

        if not assessments:
            _emit_final_quality_decision(
                capture,
                passed=False,
                code="NO_ASSESSMENTS",
                total_questions=0,
                n_assessments=0,
                duplicate_count=0,
                score=0.0,
                min_score=min_score,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="NO_ASSESSMENTS",
                        message="No assessments to validate",
                    )
                ],
            )

        # Check for duplicate questions across assessments
        all_stems: List[str] = []
        for a in assessments:
            for q in a.get("questions", []):
                stem = re.sub(r"<[^>]+>", "", q.get("stem", "")).strip().lower()
                if stem:
                    all_stems.append(stem)

        stem_counts = Counter(all_stems)
        dupes = {s: c for s, c in stem_counts.items() if c > 1}
        if dupes:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="DUPLICATE_QUESTIONS",
                    message=f"{len(dupes)} duplicate question stems found",
                )
            )

        # Check total question count
        total = sum(
            len(a.get("questions", [])) for a in assessments
        )
        if total < 5:
            issues.append(
                GateIssue(
                    severity="warning",
                    code="FEW_QUESTIONS",
                    message=f"Only {total} total questions across all assessments",
                )
            )

        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        score = max(0.0, 1.0 - error_count * 0.2 - warning_count * 0.05)
        passed = score >= min_score

        # Wave H3-W5 wiring: emit one corpus-wide
        # ``final_quality_check`` decision per validate() call.
        failure_code: Optional[str] = None
        if not passed:
            if dupes:
                failure_code = "DUPLICATE_QUESTIONS"
            elif total < 5:
                failure_code = "FEW_QUESTIONS"
            else:
                failure_code = "BELOW_MIN_SCORE"
        _emit_final_quality_decision(
            capture,
            passed=passed,
            code=failure_code,
            total_questions=total,
            n_assessments=len(assessments),
            duplicate_count=len(dupes),
            score=score,
            min_score=min_score,
        )

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )
