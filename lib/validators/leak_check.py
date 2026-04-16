"""
LeakCheck Validation Gate Adapter

Wraps the existing LeakChecker to implement the Validator protocol
expected by ValidationGateManager.

Referenced by: config/workflows.yaml (rag_training leak_check gate)
"""

import re
import time
from typing import Any, Dict, List

from MCP.hardening.validation_gates import GateIssue, GateResult

from lib.leak_checker import LeakChecker, LeakSeverity


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
        max_leaks = inputs.get("max_leaks", 0)
        strict_mode = inputs.get("strict_mode", True)
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

        # Score: 1.0 if no leaks, decreasing with each leak
        if len(questions) > 0:
            leak_ratio = total_leaks / len(questions)
            score = max(0.0, 1.0 - leak_ratio)
        else:
            score = 0.0

        passed = total_leaks <= max_leaks

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
            execution_time_ms=int((time.time() - start_time) * 1000),
        )
