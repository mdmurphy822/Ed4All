#!/usr/bin/env python3
"""
Trainforge Question Factory

Factory for creating different question types aligned with Bloom's taxonomy.

Supports:
- Multiple Choice (MCQ)
- Multiple Response (MRQ)
- True/False
- Fill-in-the-Blank
- Short Answer
- Essay
"""

import logging
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

logger = logging.getLogger(__name__)


@dataclass
class QuestionChoice:
    """A choice option for MCQ/MRQ questions."""
    text: str
    is_correct: bool = False
    feedback: Optional[str] = None


@dataclass
class Question:
    """Base question structure."""
    question_id: str
    question_type: str
    stem: str
    bloom_level: str
    objective_id: str
    points: float = 1.0
    feedback: Optional[str] = None
    choices: List[QuestionChoice] = field(default_factory=list)
    correct_answers: List[str] = field(default_factory=list)
    case_sensitive: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "stem": self.stem,
            "bloom_level": self.bloom_level,
            "objective_id": self.objective_id,
            "points": self.points,
            "feedback": self.feedback,
            "choices": [
                {"text": c.text, "is_correct": c.is_correct, "feedback": c.feedback}
                for c in self.choices
            ],
            "correct_answers": self.correct_answers,
            "case_sensitive": self.case_sensitive,
        }


class QuestionFactory:
    """
    Factory for creating assessment questions.

    Centralizes question creation with:
    - Type-specific validation
    - Bloom's alignment checks
    - Decision capture logging
    """

    VALID_TYPES = [
        "multiple_choice",
        "multiple_response",
        "true_false",
        "fill_in_blank",
        "short_answer",
        "essay",
        "matching",
    ]

    BLOOM_QUESTION_MAP = {
        "remember": ["multiple_choice", "true_false", "fill_in_blank", "matching"],
        "understand": ["multiple_choice", "short_answer", "fill_in_blank", "matching"],
        "apply": ["multiple_choice", "short_answer", "essay"],
        "analyze": ["multiple_choice", "short_answer", "essay", "matching"],
        "evaluate": ["essay", "short_answer", "multiple_choice"],
        "create": ["essay", "short_answer"],
    }

    def __init__(self, capture: Optional["DecisionCapture"] = None):
        """Initialize factory with optional decision capture."""
        self.capture = capture

    def create_multiple_choice(
        self,
        stem: str,
        choices: List[Dict[str, Any]],
        bloom_level: str = "understand",
        objective_id: str = "",
        points: float = 2.0,
        feedback: Optional[str] = None,
    ) -> Question:
        """
        Create a multiple choice question.

        Args:
            stem: Question text (HTML)
            choices: List of {"text": str, "is_correct": bool, "feedback": str}
            bloom_level: Bloom's taxonomy level
            objective_id: Learning objective ID
            points: Point value
            feedback: General feedback

        Returns:
            Question object
        """
        question_id = f"MCQ-{str(uuid.uuid4())[:8]}"

        # Validate exactly one correct answer
        correct_count = sum(1 for c in choices if c.get("is_correct", False))
        if correct_count != 1:
            logger.warning(f"MCQ {question_id}: Expected 1 correct answer, found {correct_count}")

        question_choices = [
            QuestionChoice(
                text=c["text"],
                is_correct=c.get("is_correct", False),
                feedback=c.get("feedback"),
            )
            for c in choices
        ]

        if self.capture:
            self.capture.log_decision(
                decision_type="question_creation",
                decision=f"Created MCQ {question_id} with {len(choices)} choices",
                rationale=f"Bloom level: {bloom_level}, Objective: {objective_id}",
            )

        return Question(
            question_id=question_id,
            question_type="multiple_choice",
            stem=stem,
            bloom_level=bloom_level,
            objective_id=objective_id,
            points=points,
            feedback=feedback,
            choices=question_choices,
        )

    def create_multiple_response(
        self,
        stem: str,
        choices: List[Dict[str, Any]],
        bloom_level: str = "analyze",
        objective_id: str = "",
        points: float = 3.0,
        feedback: Optional[str] = None,
    ) -> Question:
        """
        Create a multiple response question (select all that apply).

        Args:
            stem: Question text (HTML)
            choices: List of {"text": str, "is_correct": bool}
            bloom_level: Bloom's taxonomy level
            objective_id: Learning objective ID
            points: Point value
            feedback: General feedback

        Returns:
            Question object
        """
        question_id = f"MRQ-{str(uuid.uuid4())[:8]}"

        question_choices = [
            QuestionChoice(
                text=c["text"],
                is_correct=c.get("is_correct", False),
                feedback=c.get("feedback"),
            )
            for c in choices
        ]

        if self.capture:
            correct_count = sum(1 for c in choices if c.get("is_correct", False))
            self.capture.log_decision(
                decision_type="question_creation",
                decision=f"Created MRQ {question_id} with {correct_count} correct answers",
                rationale=f"Multiple response for {bloom_level} level assessment",
            )

        return Question(
            question_id=question_id,
            question_type="multiple_response",
            stem=stem,
            bloom_level=bloom_level,
            objective_id=objective_id,
            points=points,
            feedback=feedback,
            choices=question_choices,
        )

    def create_true_false(
        self,
        stem: str,
        correct_answer: bool = True,
        bloom_level: str = "remember",
        objective_id: str = "",
        points: float = 1.0,
        feedback: Optional[str] = None,
    ) -> Question:
        """
        Create a true/false question.

        Args:
            stem: Statement to evaluate (HTML)
            correct_answer: True if statement is true
            bloom_level: Bloom's taxonomy level
            objective_id: Learning objective ID
            points: Point value
            feedback: General feedback

        Returns:
            Question object
        """
        question_id = f"TF-{str(uuid.uuid4())[:8]}"

        choices = [
            QuestionChoice(text="True", is_correct=correct_answer),
            QuestionChoice(text="False", is_correct=not correct_answer),
        ]

        if self.capture:
            self.capture.log_decision(
                decision_type="question_creation",
                decision=f"Created T/F {question_id}, correct={correct_answer}",
                rationale=f"Binary question for {bloom_level} level",
            )

        return Question(
            question_id=question_id,
            question_type="true_false",
            stem=stem,
            bloom_level=bloom_level,
            objective_id=objective_id,
            points=points,
            feedback=feedback,
            choices=choices,
            correct_answers=["True" if correct_answer else "False"],
        )

    def create_fill_in_blank(
        self,
        stem: str,
        correct_answers: List[str],
        bloom_level: str = "remember",
        objective_id: str = "",
        points: float = 1.0,
        case_sensitive: bool = False,
        feedback: Optional[str] = None,
    ) -> Question:
        """
        Create a fill-in-the-blank question.

        Args:
            stem: Question with blank indicated (HTML)
            correct_answers: List of acceptable answers
            bloom_level: Bloom's taxonomy level
            objective_id: Learning objective ID
            points: Point value
            case_sensitive: Whether matching is case-sensitive
            feedback: General feedback

        Returns:
            Question object
        """
        question_id = f"FIB-{str(uuid.uuid4())[:8]}"

        if self.capture:
            self.capture.log_decision(
                decision_type="question_creation",
                decision=f"Created FIB {question_id} with {len(correct_answers)} accepted answers",
                rationale=f"Case sensitive: {case_sensitive}",
            )

        return Question(
            question_id=question_id,
            question_type="fill_in_blank",
            stem=stem,
            bloom_level=bloom_level,
            objective_id=objective_id,
            points=points,
            feedback=feedback,
            correct_answers=correct_answers,
            case_sensitive=case_sensitive,
        )

    def create_short_answer(
        self,
        stem: str,
        bloom_level: str = "apply",
        objective_id: str = "",
        points: float = 5.0,
        feedback: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> Question:
        """
        Create a short answer question.

        Args:
            stem: Question text (HTML)
            bloom_level: Bloom's taxonomy level
            objective_id: Learning objective ID
            points: Point value
            feedback: General feedback
            rubric: Grading rubric

        Returns:
            Question object
        """
        question_id = f"SA-{str(uuid.uuid4())[:8]}"

        if self.capture:
            self.capture.log_decision(
                decision_type="question_creation",
                decision=f"Created short answer {question_id}",
                rationale=f"Open response for {bloom_level} assessment",
            )

        question = Question(
            question_id=question_id,
            question_type="short_answer",
            stem=stem,
            bloom_level=bloom_level,
            objective_id=objective_id,
            points=points,
            feedback=feedback,
        )

        return question

    def create_essay(
        self,
        stem: str,
        bloom_level: str = "evaluate",
        objective_id: str = "",
        points: float = 10.0,
        feedback: Optional[str] = None,
        word_limit: Optional[int] = None,
    ) -> Question:
        """
        Create an essay question.

        Args:
            stem: Essay prompt (HTML)
            bloom_level: Bloom's taxonomy level
            objective_id: Learning objective ID
            points: Point value
            feedback: General feedback
            word_limit: Optional word limit

        Returns:
            Question object
        """
        question_id = f"ESS-{str(uuid.uuid4())[:8]}"

        if self.capture:
            self.capture.log_decision(
                decision_type="question_creation",
                decision=f"Created essay {question_id} worth {points} points",
                rationale=f"Extended response for {bloom_level} level",
            )

        return Question(
            question_id=question_id,
            question_type="essay",
            stem=stem,
            bloom_level=bloom_level,
            objective_id=objective_id,
            points=points,
            feedback=feedback,
        )

    def validate_bloom_alignment(
        self,
        question_type: str,
        bloom_level: str,
    ) -> bool:
        """
        Validate that question type is appropriate for Bloom's level.

        Args:
            question_type: Type of question
            bloom_level: Target Bloom's level

        Returns:
            True if aligned, False otherwise
        """
        valid_types = self.BLOOM_QUESTION_MAP.get(bloom_level, [])
        is_aligned = question_type in valid_types

        if not is_aligned and self.capture:
            self.capture.log_decision(
                decision_type="alignment_warning",
                decision=f"{question_type} may not be optimal for {bloom_level}",
                rationale=f"Recommended types: {valid_types}",
            )

        return is_aligned


# Convenience functions
def create_mcq(
    stem: str,
    choices: List[Dict[str, Any]],
    **kwargs,
) -> Question:
    """Create multiple choice question."""
    factory = QuestionFactory()
    return factory.create_multiple_choice(stem, choices, **kwargs)


def create_tf(stem: str, correct: bool, **kwargs) -> Question:
    """Create true/false question."""
    factory = QuestionFactory()
    return factory.create_true_false(stem, correct, **kwargs)


def create_fib(stem: str, answers: List[str], **kwargs) -> Question:
    """Create fill-in-blank question."""
    factory = QuestionFactory()
    return factory.create_fill_in_blank(stem, answers, **kwargs)


def create_essay(stem: str, **kwargs) -> Question:
    """Create essay question."""
    factory = QuestionFactory()
    return factory.create_essay(stem, **kwargs)


if __name__ == "__main__":
    # Test factory
    factory = QuestionFactory()

    mcq = factory.create_multiple_choice(
        stem="<p>What is the primary purpose of X?</p>",
        choices=[
            {"text": "<p>Answer A</p>", "is_correct": True},
            {"text": "<p>Answer B</p>", "is_correct": False},
            {"text": "<p>Answer C</p>", "is_correct": False},
            {"text": "<p>Answer D</p>", "is_correct": False},
        ],
        bloom_level="understand",
        objective_id="LO-001",
    )
    print(f"MCQ: {mcq.question_id}")

    tf = factory.create_true_false(
        stem="<p>Statement X is always true.</p>",
        correct_answer=False,
        objective_id="LO-002",
    )
    print(f"T/F: {tf.question_id}")

    fib = factory.create_fill_in_blank(
        stem="<p>The term for X is _______.</p>",
        correct_answers=["answer", "Answer"],
        objective_id="LO-003",
    )
    print(f"FIB: {fib.question_id}")

    essay = factory.create_essay(
        stem="<p>Evaluate the impact of X on Y.</p>",
        bloom_level="evaluate",
        objective_id="LO-004",
        points=20.0,
    )
    print(f"Essay: {essay.question_id}")
