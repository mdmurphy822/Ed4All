"""
Answer Key Leak Checker

Prevents training data contamination by detecting answer leakage.
Ensures prompts don't inadvertently contain correct answers.

Phase 0 Hardening - Requirement 8: Training Capture Quality Controls
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class LeakSeverity(Enum):
    """Severity of detected leak."""
    CRITICAL = "critical"    # Exact answer in prompt
    HIGH = "high"            # Answer substring in prompt
    MEDIUM = "medium"        # Answer pattern detected
    LOW = "low"              # Potential indirect leak
    INFO = "info"            # Informational only


class LeakType(Enum):
    """Type of leak detected."""
    EXACT_MATCH = "exact_match"
    SUBSTRING_MATCH = "substring_match"
    PATTERN_MATCH = "pattern_match"
    EXPLANATION_LEAK = "explanation_leak"
    DISTRACTOR_REVEALED = "distractor_revealed"
    ANSWER_IN_CONTEXT = "answer_in_context"
    INDIRECT_HINT = "indirect_hint"


@dataclass
class LeakDetection:
    """Single detected answer leak."""
    leak_type: LeakType
    severity: LeakSeverity
    question_id: str
    assessment_id: str
    location: str
    message: str
    matched_text: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "leak_type": self.leak_type.value,
            "severity": self.severity.value,
            "question_id": self.question_id,
            "assessment_id": self.assessment_id,
            "location": self.location,
            "message": self.message,
            "matched_text": self.matched_text,
            "suggestion": self.suggestion
        }


@dataclass
class LeakCheckResult:
    """Result of leak check operation."""
    passed: bool
    checked_count: int
    leak_count: int
    leaks: List[LeakDetection] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def critical_count(self) -> int:
        """Count critical severity leaks."""
        return sum(1 for l in self.leaks if l.severity == LeakSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        """Count high severity leaks."""
        return sum(1 for l in self.leaks if l.severity == LeakSeverity.HIGH)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "checked_count": self.checked_count,
            "leak_count": self.leak_count,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "leaks": [l.to_dict() for l in self.leaks],
            "timestamp": self.timestamp
        }


@dataclass
class RegisteredAnswer:
    """Registered answer for leak detection."""
    question_id: str
    assessment_id: str
    answer_hash: str
    answer_normalized: str
    is_explanation: bool = False
    is_distractor: bool = False


class LeakChecker:
    """Checks for answer key leaks in training data."""

    # Patterns that indicate answer revelation
    ANSWER_REVEAL_PATTERNS = [
        (r'(?:the\s+)?(?:correct|right)\s+answer\s+(?:is|would\s+be|should\s+be)', "answer_reveal"),
        (r'answer\s*[:=]\s*', "answer_assignment"),
        (r'(?:this|that)\s+is\s+(?:the\s+)?(?:correct|right)', "correctness_indicator"),
        (r'(?:should|must|will)\s+(?:be|equal|return)', "result_indicator"),
        (r'(?:always|never)\s+(?:returns?|equals?|is)', "absolute_result"),
    ]

    # Patterns for explanation context (may be intentional)
    EXPLANATION_PATTERNS = [
        r'(?:the\s+)?explanation\s*[:=]',
        r'(?:this|that)\s+is\s+because',
        r'the\s+reason\s+(?:is|being)',
        r'to\s+understand\s+why',
        r'(?:here\'s|here\s+is)\s+why',
    ]

    def __init__(self, strict_mode: bool = True):
        """
        Initialize leak checker.

        Args:
            strict_mode: If True, any detected leak fails the check
        """
        self.strict_mode = strict_mode
        self._answer_registry: Dict[str, RegisteredAnswer] = {}  # hash -> RegisteredAnswer
        self._question_answers: Dict[str, Set[str]] = {}  # question_id -> set of answer hashes
        self._assessment_questions: Dict[str, Set[str]] = {}  # assessment_id -> set of question_ids

        # Compile patterns
        self._reveal_patterns = [
            (re.compile(p, re.IGNORECASE), name)
            for p, name in self.ANSWER_REVEAL_PATTERNS
        ]
        self._explanation_re = re.compile(
            '|'.join(self.EXPLANATION_PATTERNS),
            re.IGNORECASE
        )

    def register_assessment(
        self,
        assessment_id: str,
        questions: List[Dict[str, Any]]
    ) -> int:
        """
        Register an assessment's answers for leak detection.

        Args:
            assessment_id: Unique assessment identifier
            questions: List of question dicts with 'id', 'correct_answer'/'correct_answers',
                      optional 'explanation', 'distractors'

        Returns:
            Number of answers registered
        """
        count = 0
        self._assessment_questions[assessment_id] = set()

        for q in questions:
            q_id = q.get('id', f"q_{count}")
            self._assessment_questions[assessment_id].add(q_id)
            self._question_answers[q_id] = set()

            # Register correct answer(s)
            correct = q.get('correct_answer') or q.get('correct_answers', [])
            if isinstance(correct, str):
                correct = [correct]

            for answer in correct:
                reg = self._register_answer(
                    question_id=q_id,
                    assessment_id=assessment_id,
                    answer_text=answer,
                    is_explanation=False,
                    is_distractor=False
                )
                if reg:
                    count += 1

            # Register explanation (shouldn't appear in prompts)
            if q.get('explanation'):
                reg = self._register_answer(
                    question_id=q_id,
                    assessment_id=assessment_id,
                    answer_text=q['explanation'],
                    is_explanation=True,
                    is_distractor=False
                )
                if reg:
                    count += 1

            # Register distractors with incorrect reason (shouldn't be revealed)
            for distractor in q.get('distractors', []):
                if isinstance(distractor, dict) and distractor.get('why_incorrect'):
                    reg = self._register_answer(
                        question_id=q_id,
                        assessment_id=assessment_id,
                        answer_text=distractor['why_incorrect'],
                        is_explanation=False,
                        is_distractor=True
                    )
                    if reg:
                        count += 1

        logger.debug(f"Registered {count} answers for assessment {assessment_id}")
        return count

    def _register_answer(
        self,
        question_id: str,
        assessment_id: str,
        answer_text: str,
        is_explanation: bool,
        is_distractor: bool
    ) -> Optional[RegisteredAnswer]:
        """Register a single answer."""
        if not answer_text or len(answer_text.strip()) < 3:
            return None

        normalized = self._normalize(answer_text)
        answer_hash = self._compute_hash(normalized)

        reg = RegisteredAnswer(
            question_id=question_id,
            assessment_id=assessment_id,
            answer_hash=answer_hash,
            answer_normalized=normalized,
            is_explanation=is_explanation,
            is_distractor=is_distractor
        )

        self._answer_registry[answer_hash] = reg
        self._question_answers.setdefault(question_id, set()).add(answer_hash)

        return reg

    def check_prompt(
        self,
        prompt_text: str,
        assessment_id: Optional[str] = None,
        question_id: Optional[str] = None
    ) -> LeakCheckResult:
        """
        Check if prompt contains answer leaks.

        Args:
            prompt_text: The prompt text to check
            assessment_id: Optional - limit check to specific assessment
            question_id: Optional - limit check to specific question

        Returns:
            LeakCheckResult with detected leaks
        """
        leaks = []
        checked_count = 0

        # Determine which answers to check against
        if question_id and question_id in self._question_answers:
            hashes_to_check = self._question_answers[question_id]
        elif assessment_id and assessment_id in self._assessment_questions:
            hashes_to_check = set()
            for q_id in self._assessment_questions[assessment_id]:
                hashes_to_check.update(self._question_answers.get(q_id, set()))
        else:
            hashes_to_check = set(self._answer_registry.keys())

        # Check for exact matches
        segments = self._extract_segments(prompt_text)
        for segment in segments:
            checked_count += 1
            normalized = self._normalize(segment)
            segment_hash = self._compute_hash(normalized)

            if segment_hash in hashes_to_check:
                reg = self._answer_registry[segment_hash]
                leak = LeakDetection(
                    leak_type=LeakType.EXACT_MATCH,
                    severity=LeakSeverity.CRITICAL,
                    question_id=reg.question_id,
                    assessment_id=reg.assessment_id,
                    location=self._truncate(segment, 60),
                    message="Exact answer text found in prompt",
                    matched_text=segment[:100],
                    suggestion="Remove or paraphrase the answer content"
                )
                leaks.append(leak)

        # Check for substring matches
        prompt_normalized = self._normalize(prompt_text)
        for answer_hash in hashes_to_check:
            reg = self._answer_registry[answer_hash]

            # Skip short answers for substring matching (too many false positives)
            if len(reg.answer_normalized) < 15:
                continue

            if reg.answer_normalized in prompt_normalized:
                # Check if we already detected this as exact match
                already_detected = any(
                    l.question_id == reg.question_id and
                    l.leak_type == LeakType.EXACT_MATCH
                    for l in leaks
                )
                if not already_detected:
                    leak = LeakDetection(
                        leak_type=LeakType.SUBSTRING_MATCH,
                        severity=LeakSeverity.HIGH,
                        question_id=reg.question_id,
                        assessment_id=reg.assessment_id,
                        location="(substring match)",
                        message="Answer appears as substring in prompt",
                        matched_text=reg.answer_normalized[:100],
                        suggestion="Ensure answer content is not embedded in context"
                    )
                    leaks.append(leak)

        # Check for answer reveal patterns
        for pattern, pattern_name in self._reveal_patterns:
            matches = pattern.finditer(prompt_text)
            for match in matches:
                # Check context around match
                context_start = max(0, match.start() - 50)
                context_end = min(len(prompt_text), match.end() + 100)
                context = prompt_text[context_start:context_end]

                # Skip if in explanation context
                if self._explanation_re.search(context):
                    continue

                leak = LeakDetection(
                    leak_type=LeakType.PATTERN_MATCH,
                    severity=LeakSeverity.MEDIUM,
                    question_id=question_id or "unknown",
                    assessment_id=assessment_id or "unknown",
                    location=f"pattern:{pattern_name}",
                    message=f"Answer revelation pattern detected: {pattern_name}",
                    matched_text=context,
                    suggestion="Review context to ensure answer is not revealed"
                )
                leaks.append(leak)

        # Determine pass/fail
        if self.strict_mode:
            passed = len(leaks) == 0
        else:
            # Non-strict: fail only on critical/high
            passed = all(
                l.severity not in (LeakSeverity.CRITICAL, LeakSeverity.HIGH)
                for l in leaks
            )

        return LeakCheckResult(
            passed=passed,
            checked_count=checked_count,
            leak_count=len(leaks),
            leaks=leaks
        )

    def check_response(
        self,
        response_text: str,
        question_id: str,
        correct_answer: str,
        context: str = "generation"
    ) -> LeakCheckResult:
        """
        Check if model response inappropriately reveals answer.

        This is for checking generated content (e.g., hints, feedback)
        to ensure they don't give away the answer.

        Args:
            response_text: The generated response to check
            question_id: Question being addressed
            correct_answer: The correct answer
            context: Context type ('generation', 'hint', 'feedback')

        Returns:
            LeakCheckResult
        """
        leaks = []

        answer_normalized = self._normalize(correct_answer)
        response_normalized = self._normalize(response_text)

        # Check for answer presence
        if answer_normalized in response_normalized:
            # Determine severity based on context
            if context == 'hint':
                severity = LeakSeverity.CRITICAL
                message = "Hint contains the correct answer"
            elif context == 'feedback':
                # Feedback may legitimately contain answer
                severity = LeakSeverity.LOW
                message = "Feedback contains answer (may be intentional)"
            else:
                severity = LeakSeverity.HIGH
                message = "Response contains correct answer"

            leak = LeakDetection(
                leak_type=LeakType.ANSWER_IN_CONTEXT,
                severity=severity,
                question_id=question_id,
                assessment_id="response_check",
                location=context,
                message=message,
                matched_text=correct_answer[:100]
            )
            leaks.append(leak)

        # Check for reveal patterns
        for pattern, pattern_name in self._reveal_patterns:
            if pattern.search(response_text):
                # In hint context, this is more severe
                severity = LeakSeverity.MEDIUM if context != 'hint' else LeakSeverity.HIGH

                leak = LeakDetection(
                    leak_type=LeakType.PATTERN_MATCH,
                    severity=severity,
                    question_id=question_id,
                    assessment_id="response_check",
                    location=f"pattern:{pattern_name}",
                    message="Response contains answer revelation pattern"
                )
                leaks.append(leak)

        passed = len(leaks) == 0 or (
            not self.strict_mode and
            all(l.severity not in (LeakSeverity.CRITICAL, LeakSeverity.HIGH) for l in leaks)
        )

        return LeakCheckResult(
            passed=passed,
            checked_count=1,
            leak_count=len(leaks),
            leaks=leaks
        )

    def check_training_example(
        self,
        prompt: str,
        response: str,
        assessment_id: Optional[str] = None
    ) -> LeakCheckResult:
        """
        Check a complete training example (prompt + response) for leaks.

        Args:
            prompt: The training prompt
            response: The expected/actual response
            assessment_id: Optional assessment ID

        Returns:
            Combined LeakCheckResult
        """
        # Check prompt
        prompt_result = self.check_prompt(prompt, assessment_id)

        # For training, response should not reveal answers either
        # (unless it's explicitly an answer-providing example)
        response_leaks = []
        response_normalized = self._normalize(response)

        for _answer_hash, reg in self._answer_registry.items():
            if assessment_id and reg.assessment_id != assessment_id:
                continue

            if reg.answer_normalized in response_normalized:
                # Training responses containing answers might be intentional
                # Mark as INFO for review
                leak = LeakDetection(
                    leak_type=LeakType.ANSWER_IN_CONTEXT,
                    severity=LeakSeverity.INFO,
                    question_id=reg.question_id,
                    assessment_id=reg.assessment_id,
                    location="response",
                    message="Training response contains answer (verify if intentional)"
                )
                response_leaks.append(leak)

        # Combine results
        all_leaks = prompt_result.leaks + response_leaks

        return LeakCheckResult(
            passed=prompt_result.passed and len([
                l for l in response_leaks
                if l.severity in (LeakSeverity.CRITICAL, LeakSeverity.HIGH)
            ]) == 0,
            checked_count=prompt_result.checked_count + 1,
            leak_count=len(all_leaks),
            leaks=all_leaks
        )

    def scan_batch(
        self,
        examples: List[Dict[str, str]],
        assessment_id: Optional[str] = None
    ) -> Tuple[LeakCheckResult, List[int]]:
        """
        Scan a batch of training examples for leaks.

        Args:
            examples: List of dicts with 'prompt' and 'response' keys
            assessment_id: Optional assessment ID to check against

        Returns:
            Tuple of (aggregate result, list of failing example indices)
        """
        all_leaks = []
        failing_indices = []
        checked_count = 0

        for i, example in enumerate(examples):
            prompt = example.get('prompt', '')
            response = example.get('response', '')

            result = self.check_training_example(prompt, response, assessment_id)
            checked_count += result.checked_count

            if not result.passed:
                failing_indices.append(i)
                all_leaks.extend(result.leaks)

        return LeakCheckResult(
            passed=len(failing_indices) == 0,
            checked_count=checked_count,
            leak_count=len(all_leaks),
            leaks=all_leaks
        ), failing_indices

    def clear_registry(self) -> None:
        """Clear all registered answers."""
        self._answer_registry.clear()
        self._question_answers.clear()
        self._assessment_questions.clear()

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Remove punctuation, extra whitespace, lowercase
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = ' '.join(text.split())
        return text

    def _compute_hash(self, normalized_text: str) -> str:
        """Compute hash for normalized text."""
        return hashlib.sha256(normalized_text.encode()).hexdigest()[:16]

    def _extract_segments(self, text: str) -> List[str]:
        """Extract meaningful segments from text."""
        # Split on sentence boundaries and list items
        segments = re.split(r'[.!?]\s+|\n+|[•\-*]\s+|\d+[.)]\s+', text)
        # Filter short segments (unlikely to be full answers)
        return [s.strip() for s in segments if len(s.strip()) >= 10]

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text with ellipsis."""
        if len(text) <= max_len:
            return text
        return text[:max_len - 3] + "..."


def check_corpus_boilerplate(
    self,
    chunks: List[Dict[str, Any]],
    n: int = 15,
    threshold: float = 0.10,
) -> List[LeakDetection]:
    """Detect corpus-wide repeated n-gram boilerplate across chunk text.

    Flags when any repeated span appears in more than ``threshold`` of the
    chunks — typically footers, copyright, or template chrome that escaped
    stripping at the chunker stage. Returns one LeakDetection per span at
    ``LOW`` severity so existing leak-dashboard plumbing can surface it
    without blocking the pipeline.
    """
    try:
        from Trainforge.rag.boilerplate_detector import (
            contamination_rate,
            detect_repeated_ngrams,
        )
    except Exception:
        return []

    if not chunks:
        return []

    texts = [c.get("text", "") or "" for c in chunks]
    spans = detect_repeated_ngrams(texts, n=n, min_doc_frac=threshold)
    if not spans:
        return []

    rate = contamination_rate(chunks, spans)
    details: List[LeakDetection] = []
    for span in spans:
        details.append(LeakDetection(
            leak_type=LeakType.PATTERN_MATCH,
            severity=LeakSeverity.LOW,
            question_id="corpus",
            assessment_id="corpus",
            location="corpus.boilerplate",
            message=(
                f"Corpus boilerplate above {threshold:.0%} threshold "
                f"(contamination={rate:.0%})"
            ),
            matched_text=span[:200],
            suggestion=(
                "Strip at source (Courseforge template chrome) or defensively "
                "via Trainforge boilerplate_detector. See VERSIONING.md §4.7."
            ),
        ))
    return details


# Attach to LeakChecker as a bound method. Defined at module scope to
# keep the class body tight; the binding makes `LeakChecker.check_corpus_boilerplate`
# available to callers that hold a checker instance.
LeakChecker.check_corpus_boilerplate = check_corpus_boilerplate


# Global checker instance
_global_checker: Optional[LeakChecker] = None


def get_leak_checker(strict_mode: bool = True) -> LeakChecker:
    """Get global leak checker instance."""
    global _global_checker
    if _global_checker is None:
        _global_checker = LeakChecker(strict_mode=strict_mode)
    return _global_checker


def check_for_leaks(
    prompt: str,
    assessment_id: Optional[str] = None
) -> LeakCheckResult:
    """
    Convenience function to check for leaks.

    Args:
        prompt: Prompt text to check
        assessment_id: Optional assessment ID

    Returns:
        LeakCheckResult
    """
    return get_leak_checker().check_prompt(prompt, assessment_id)


def register_answers(
    assessment_id: str,
    questions: List[Dict[str, Any]]
) -> int:
    """
    Convenience function to register answers.

    Args:
        assessment_id: Assessment identifier
        questions: List of question dicts

    Returns:
        Number of answers registered
    """
    return get_leak_checker().register_assessment(assessment_id, questions)
