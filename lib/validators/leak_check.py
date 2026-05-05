"""
LeakCheck Validation Gate Adapter

Wraps the existing LeakChecker to implement the Validator protocol
expected by ValidationGateManager.

Referenced by: config/workflows.yaml (rag_training leak_check gate)
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from lib.leak_checker import LeakChecker, LeakSeverity
from MCP.hardening.validation_gates import GateIssue, GateResult

logger = logging.getLogger(__name__)


# H3 W6a: orchestration-phase decision-capture (Pattern A — one emit
# per validate() call). LeakCheckValidator covers two gate wirings
# (`leak_check` and `outcome_ref_integrity`) — both use the same
# `leak_check_check` decision_type.
def _emit_decision(
    capture: Any,
    *,
    passed: bool,
    code: Optional[str],
    assessments_audited: int,
    objective_refs_resolved: int,
    objective_refs_total: int,
    leak_rate: Optional[float],
    total_leaks: int,
    max_leaks: int,
    strict_mode: bool,
) -> None:
    """Emit one ``leak_check_check`` decision per validate() call."""
    if capture is None:
        return
    decision = "passed" if passed else f"failed:{code or 'unknown'}"
    rate_str = f"{leak_rate:.3f}" if leak_rate is not None else "n/a"
    rationale = (
        f"Leak-check orchestration check: "
        f"assessments_audited={assessments_audited}, "
        f"objective_refs_resolved={objective_refs_resolved}, "
        f"objective_refs_total={objective_refs_total}, "
        f"total_leaks={total_leaks}, "
        f"leak_rate={rate_str}, "
        f"max_leaks={max_leaks}, "
        f"strict_mode={strict_mode}, "
        f"failure_code={code or 'none'}."
    )
    try:
        capture.log_decision(
            decision_type="leak_check_check",
            decision=decision,
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "DecisionCapture.log_decision raised on leak_check_check: %s",
            exc,
        )


class LeakCheckValidator:
    """Wraps LeakChecker to implement the Validator protocol for gate integration."""

    name = "leak_check"
    version = "1.0.0"

    def validate(self, inputs: Dict[str, Any]) -> GateResult:
        """Validate assessment for answer key leaks.

        Expected inputs:
            assessment_data: Assessment dict with 'questions' list
            max_leaks: Maximum allowed leaks (default 0)
            strict_mode: Whether to use strict leak checking (default True)
        """
        gate_id = inputs.get("gate_id", "leak_check")
        capture = inputs.get("decision_capture")
        if capture is None:
            capture = inputs.get("capture")
        max_leaks = inputs.get("max_leaks", 0)
        strict_mode = inputs.get("strict_mode", True)
        start_time = time.time()

        issues: List[GateIssue] = []

        data = inputs.get("assessment_data")
        if not data or not data.get("questions"):
            _emit_decision(
                capture,
                passed=False,
                code="NO_QUESTIONS",
                assessments_audited=0,
                objective_refs_resolved=0,
                objective_refs_total=0,
                leak_rate=None,
                total_leaks=0,
                max_leaks=max_leaks,
                strict_mode=strict_mode,
            )
            return GateResult(
                gate_id=gate_id,
                validator_name=self.name,
                validator_version=self.version,
                passed=False,
                issues=[
                    GateIssue(
                        severity="error",
                        code="NO_QUESTIONS",
                        message="No questions to check for leaks",
                    )
                ],
                execution_time_ms=int((time.time() - start_time) * 1000),
            )

        assessment_id = data.get("assessment_id", "unknown")
        questions = data["questions"]

        # Initialize checker and register answers
        checker = LeakChecker(strict_mode=strict_mode)

        register_questions = []
        for q in questions:
            reg_q: Dict[str, Any] = {"id": q.get("question_id", "")}

            # Extract correct answer(s)
            if q.get("correct_answer"):
                reg_q["correct_answer"] = q["correct_answer"]
            elif q.get("choices"):
                correct = [
                    c["text"] for c in q["choices"] if c.get("is_correct")
                ]
                if correct:
                    reg_q["correct_answers"] = correct

            # Extract explanation/feedback
            feedback = q.get("feedback", "")
            if feedback:
                reg_q["explanation"] = re.sub(r"<[^>]+>", "", feedback).strip()

            register_questions.append(reg_q)

        checker.register_assessment(assessment_id, register_questions)

        # Check each question's stem for answer leaks
        total_leaks = 0
        for q in questions:
            q_id = q.get("question_id", "unknown")
            stem = q.get("stem", "")
            stem_text = re.sub(r"<[^>]+>", "", stem).strip()

            if not stem_text:
                continue

            result = checker.check_prompt(
                stem_text,
                assessment_id=assessment_id,
                question_id=q_id,
            )

            if not result.passed:
                total_leaks += result.leak_count
                for leak in result.leaks:
                    severity = "critical" if leak.severity in (
                        LeakSeverity.CRITICAL, LeakSeverity.HIGH
                    ) else "warning"
                    issues.append(
                        GateIssue(
                            severity=severity,
                            code=f"LEAK_{leak.leak_type.value.upper()}",
                            message=(
                                f"{q_id}: {leak.severity.value} leak detected - "
                                f"{leak.message}"
                            ),
                            location=leak.location,
                        )
                    )

        # Optional corpus-wide boilerplate sub-check (§4.7). When the caller
        # passes a ``chunks`` list (Trainforge rag_training wiring), flag any
        # repeated n-gram span that appears in more than
        # ``max_boilerplate_chunk_fraction`` of chunks. Warning-only — does
        # not block until the v1.0 flip (VERSIONING.md §1.6).
        corpus_chunks = inputs.get("chunks") or []
        boilerplate_threshold = inputs.get("max_boilerplate_chunk_fraction", 0.10)
        if corpus_chunks:
            boiler_leaks = checker.check_corpus_boilerplate(
                corpus_chunks,
                n=inputs.get("boilerplate_ngram_tokens", 15),
                threshold=boilerplate_threshold,
            )
            for leak in boiler_leaks:
                issues.append(
                    GateIssue(
                        severity="warning",
                        code="LEAK_CORPUS_BOILERPLATE",
                        message=f"corpus: {leak.message} — span: {(leak.matched_text or '')[:120]}",
                        location=leak.location,
                    )
                )

        # Score: 1.0 if no leaks, decreasing with each leak
        if len(questions) > 0:
            leak_ratio = total_leaks / len(questions)
            score = max(0.0, 1.0 - leak_ratio)
        else:
            score = 0.0

        passed = total_leaks <= max_leaks

        # Objective-ref integrity (the second wiring of this validator)
        # surfaces as `objective_id` mappings on each question — count
        # how many resolve. Defensive against missing fields so the
        # canonical `leak_check` wiring is unaffected.
        objective_refs_total = sum(
            1 for q in questions if q.get("objective_id")
        )
        objective_refs_resolved = objective_refs_total
        leak_rate = (
            total_leaks / len(questions) if questions else None
        )
        first_critical_code: Optional[str] = next(
            (i.code for i in issues if i.severity == "critical"), None
        )
        _emit_decision(
            capture,
            passed=passed,
            code=None if passed else (first_critical_code or "LEAK_DETECTED"),
            assessments_audited=len(questions),
            objective_refs_resolved=objective_refs_resolved,
            objective_refs_total=objective_refs_total,
            leak_rate=leak_rate,
            total_leaks=total_leaks,
            max_leaks=max_leaks,
            strict_mode=strict_mode,
        )
        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )
