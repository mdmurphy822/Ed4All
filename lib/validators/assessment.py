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
# Imported from the module rather than the package ``__init__`` so the cloze
# helper stays co-located with ``detect_bloom_level`` — the two are always used
# together (see ``_stem_lacks_task_verb``).
from lib.validators.bloom.alignment import (
    stem_is_cloze,
    stem_is_declarative_proposition,
)
from MCP.hardening.validation_gates import GateIssue, GateResult

# W-D7 T7.4: thresholds + placeholder helpers extracted into the
# ``_assessment_helpers`` private subpackage so the calibration table +
# regex set live in one auditable file. Re-exported here so existing
# cross-file imports (training_pair_promotion, assessment_objective_alignment,
# schemas/tests/test_wave{6,7}*.py) keep resolving without change.
from lib.validators._assessment_helpers.placeholders import (  # noqa: F401
    ASSESSMENT_PLACEHOLDER_PATTERNS,
    _CHAPTER_HEADING_RE,
    _INTEGER_TOKEN_RE,
    _TOC_THREE_INTS_RE,
    _looks_like_toc_fragment,
    _normalize_question_type,
    _strip_html_text,
)
from lib.validators._assessment_helpers.thresholds import (  # noqa: F401
    CORRECT_ANSWER_DIVERSITY_THRESHOLD,
    DISTRACTOR_TEMPLATE_MAX_RATIO,
    STEM_DIVERSITY_THRESHOLD,
    _DEFAULT_QUESTION_TYPE_THRESHOLDS,
    _PER_QUESTION_TYPE_THRESHOLDS,
    _resolve_per_type_thresholds,
    _thresholds_for_type,
)

logger = logging.getLogger(__name__)


# Assessment-quality overhaul (Phase 2) — item subtypes that legitimately carry
# more than one correct key on the ``multiple_choice``-typed surface. Mirrors
# ``Trainforge.generators.assessment.generator`` item_subtype universe; kept as
# a local constant so the validator never needs the generator at import time.
_MULTI_KEY_SUBTYPES: frozenset = frozenset({"mc_multiple_response"})


def _item_allows_multiple_keys(q: Dict[str, Any]) -> bool:
    """True when a question legitimately carries >1 correct key.

    A genuine multiple-response / select-all item is exempt from the
    single-key ``WRONG_CORRECT_COUNT`` hard-fail. Detected via ANY of: a
    ``question_type`` that normalizes to ``multiple_response``; an
    ``item_subtype`` in the multi-key set (``mc_multiple_response``); or a
    populated plural ``correct_answers`` list (the diversified tier's
    self-describing shape). Two-tier answer/reason items keep a single key by
    construction, so they are NOT force-exempted here — they only qualify if
    they actually declare plural keys.
    """
    if _normalize_question_type(q) == "multiple_response":
        return True
    subtype = q.get("item_subtype")
    if isinstance(subtype, str) and subtype.strip() in _MULTI_KEY_SUBTYPES:
        return True
    plural = q.get("correct_answers")
    if isinstance(plural, list) and len(plural) > 1:
        return True
    return False


def _stem_lacks_task_verb(
    stem_text: str, question: Optional[Dict[str, Any]] = None
) -> bool:
    """True when a stem genuinely carries no task verb (the VERB_LESS defect).

    Three conditions, all required:

    1. :func:`detect_bloom_level` finds no Bloom verb, and
    2. the stem is not a **cloze** (fill-in-the-blank) sentence, and
    3. the item is not a **true/false** whose stem is a well-formed
       declarative proposition.

    Conditions 2 and 3 are the same category correction applied to the two
    item types whose cognitive demand rides on the TYPE rather than on a stem
    imperative. A cloze stem is a sentence with a gap in it
    (``Complete the following: _______ is a multiple of 4.``); a true/false
    stem is a claim to judge (``The opposite of -10 is 10 because it is the
    same distance from 0 but on the opposite side.``). Neither can carry a
    Bloom verb, so flagging them was a false positive that, at scale,
    escalated into a build-blocking ``PERVASIVE_VERBLESS_STEMS`` critical —
    which is why the exemption must live at the per-question rule, not at the
    cap.

    Neither exemption is a blanket amnesty:

    * Cloze detection keys on stem SHAPE, not on ``question_type``, because an
      imported cartridge routinely mislabels a cloze item — and a verb-less
      non-cloze stem carries no gap marker, so it is still reported.
    * The true/false exemption requires BOTH the item type AND a structurally
      complete proposition (see
      :func:`~lib.validators.bloom.alignment.stem_is_declarative_proposition`),
      so a true/false whose stem is apparatus (``Step 2: Since -9 is 9 units
      from 0, |-9| = 9.``), a dangling anaphor (``This is determined by its
      position…``) or an exercise directive (``Find three consecutive integers
      whose sum is -36.``) keeps warning. Those are real defects, fixed at the
      mining layer rather than suppressed here.

    ``question`` is optional so bare-stem callers keep working; without it only
    the cloze exemption can apply. ``stem_text`` must already be tag-stripped.
    Domain-agnostic by construction — closed function-word lists only, no
    subject vocabulary and no publisher phrase list.
    """
    if not stem_text:
        return False
    if detect_bloom_level(stem_text) is not None:
        return False
    if stem_is_cloze(stem_text):
        return False
    if (
        question is not None
        and _normalize_question_type(question) == "true_false"
        and stem_is_declarative_proposition(stem_text)
    ):
        return False
    return True


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

# W-D7 T7.4: ASSESSMENT_PLACEHOLDER_PATTERNS, STEM_DIVERSITY_THRESHOLD,
# CORRECT_ANSWER_DIVERSITY_THRESHOLD, DISTRACTOR_TEMPLATE_MAX_RATIO,
# _TOC_THREE_INTS_RE, _CHAPTER_HEADING_RE, _INTEGER_TOKEN_RE moved to
# lib.validators._assessment_helpers.{placeholders,thresholds}; imported
# at the top of this file.


# W-D7 T7.4: _PER_QUESTION_TYPE_THRESHOLDS, _DEFAULT_QUESTION_TYPE_THRESHOLDS,
# _normalize_question_type, _resolve_per_type_thresholds,
# _thresholds_for_type, _strip_html_text, _looks_like_toc_fragment moved
# to lib.validators._assessment_helpers.{placeholders,thresholds}; all
# names imported at the top of this file so existing
# ``from lib.validators.assessment import <name>`` calls keep resolving.


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
            # Assessment-quality overhaul (Phase 2) — RELAX the single-key
            # hard-fail for genuine multiple-response / multi-key items. The
            # deterministic diversified tier types a select-all item as
            # item_subtype=mc_multiple_response (question_type stays a portable
            # CC primitive, sometimes "multiple_choice") and carries the plural
            # keys in ``correct_answers`` (each key ALSO flagged is_correct in
            # ``choices``, so the item is self-describing). For such items a
            # correct-count > 1 is CORRECT, not a defect — only a zero-key item
            # is a real fault. Single-answer items keep the exact-one contract.
            if _item_allows_multiple_keys(q):
                if len(correct) < 1:
                    issues.append(
                        GateIssue(
                            severity="error",
                            code="WRONG_CORRECT_COUNT",
                            message=(
                                f"{q_id}: multiple-response item has "
                                f"{len(correct)} correct answers (need >=1)"
                            ),
                        )
                    )
            elif len(correct) != 1:
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
        # that cap. Here we just record the finding per question. Cloze
        # (fill-in-the-blank) stems and well-formed declarative true/false
        # propositions are exempt: see ``_stem_lacks_task_verb``.
        if _stem_lacks_task_verb(text, q):
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
        above the cap is escalated here. Cloze (fill-in-the-blank) stems
        never count toward the cap — see ``_stem_lacks_task_verb``; both
        sites share that helper so the per-question warning and the
        cross-question escalation can never disagree about what counts.

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
            if _stem_lacks_task_verb(s, q):
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
        RDF/SHACL calibration corpus rebuild.

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
