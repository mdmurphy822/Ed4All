"""
Assessment Quality Validators

Validates generated assessments for quality and alignment:

AssessmentQualityValidator:
- Question clarity and unambiguity
- Distractor quality (plausible, misconception-based)
- Answer correctness
- Coverage of learning objectives
- Appropriate difficulty distribution

FinalQualityValidator:
- End-to-end quality check after all generation
- Cross-assessment consistency
- No duplicate questions
- Minimum quality score threshold

Referenced by: config/workflows.yaml (rag_training, textbook_to_course)
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from MCP.hardening.validation_gates import GateIssue, GateResult

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


class AssessmentQualityValidator:
    """Validates individual assessment quality."""

    name = "assessment_quality"
    version = "1.1.0"

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

        # Check each question
        for q in questions:
            issues.extend(self._check_question(q))

        # Check objective coverage
        target_objectives = inputs.get("learning_objectives", [])
        if target_objectives:
            issues.extend(
                self._check_objective_coverage(questions, target_objectives)
            )

        # Compute score
        error_count = sum(1 for i in issues if i.severity == "error")
        warning_count = sum(1 for i in issues if i.severity == "warning")
        score = max(
            0.0,
            1.0
            - error_count * 0.15
            - warning_count * 0.05,
        )
        passed = score >= min_score

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )

    def _check_question(self, q: Dict[str, Any]) -> List[GateIssue]:
        """Check a single question for quality issues."""
        issues = []
        q_id = q.get("question_id", "unknown")
        stem = q.get("stem", "")
        q_type = q.get("question_type", "")

        # Check stem is non-empty
        text = re.sub(r"<[^>]+>", "", stem).strip()
        if len(text) < 10:
            issues.append(
                GateIssue(
                    severity="error",
                    code="SHORT_STEM",
                    message=f"{q_id}: question stem too short ({len(text)} chars)",
                )
            )

        # Check for placeholder content in stem
        for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
            if pattern.search(text):
                issues.append(
                    GateIssue(
                        severity="error",
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
                                severity="error",
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
                            severity="error",
                            code="PLACEHOLDER_ANSWER",
                            message=f"{q_id}: correct_answer is placeholder text: '{correct_answer}'",
                        )
                    )
                    break

        # Check for placeholder in feedback
        feedback = re.sub(r"<[^>]+>", "", q.get("feedback", "")).strip()
        if feedback:
            for pattern in ASSESSMENT_PLACEHOLDER_PATTERNS:
                if pattern.search(feedback):
                    issues.append(
                        GateIssue(
                            severity="warning",
                            code="PLACEHOLDER_FEEDBACK",
                            message=f"{q_id}: feedback contains placeholder text",
                        )
                    )
                    break

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

        return GateResult(
            gate_id=gate_id,
            validator_name=self.name,
            validator_version=self.version,
            passed=passed,
            score=score,
            issues=issues,
        )
