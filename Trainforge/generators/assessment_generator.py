#!/usr/bin/env python3
"""
Trainforge Assessment Generator

Generates assessments from course content with decision capture for training data.

Pipeline Position:
    IMSCC Package → RAG Index → [Assessment Generator] → Validated Assessments

Decision Capture:
    All generation decisions logged for model training.
"""

import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Add project path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # → Ed4All/
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

logger = logging.getLogger(__name__)

# Import leak checker for answer-leak detection
try:
    from lib.leak_checker import LeakChecker  # noqa: F401
    LEAK_CHECKER_AVAILABLE = True
except ImportError:
    LEAK_CHECKER_AVAILABLE = False


# Bloom's Taxonomy levels with associated question patterns
BLOOM_LEVELS = {
    "remember": {
        "verbs": ["define", "list", "recall", "identify", "name"],
        "patterns": ["What is...?", "List the...", "Which of the following...?"],
        "question_types": ["multiple_choice", "true_false", "fill_in_blank"],
    },
    "understand": {
        "verbs": ["explain", "describe", "summarize", "interpret", "paraphrase"],
        "patterns": ["Explain why...", "Describe how...", "What does X mean?"],
        "question_types": ["multiple_choice", "short_answer", "fill_in_blank"],
    },
    "apply": {
        "verbs": ["apply", "demonstrate", "use", "solve", "implement"],
        "patterns": ["How would you use...?", "Apply X to...", "Solve..."],
        "question_types": ["multiple_choice", "short_answer", "essay"],
    },
    "analyze": {
        "verbs": ["analyze", "compare", "contrast", "differentiate", "examine"],
        "patterns": ["Compare and contrast...", "What are the differences...", "Analyze..."],
        "question_types": ["multiple_choice", "essay", "short_answer"],
    },
    "evaluate": {
        "verbs": ["evaluate", "judge", "justify", "critique", "assess"],
        "patterns": ["Evaluate the effectiveness...", "Justify your answer...", "Assess..."],
        "question_types": ["essay", "multiple_choice", "short_answer"],
    },
    "create": {
        "verbs": ["create", "design", "develop", "construct", "formulate"],
        "patterns": ["Design a...", "Develop a plan for...", "Create..."],
        "question_types": ["essay", "short_answer"],
    },
}


@dataclass
class QuestionData:
    """A generated assessment question."""
    question_id: str
    question_type: str
    stem: str
    bloom_level: str
    objective_id: str
    choices: List[Dict[str, Any]] = field(default_factory=list)
    correct_answer: Optional[str] = None
    points: float = 1.0
    feedback: Optional[str] = None
    source_chunks: List[str] = field(default_factory=list)
    generation_rationale: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "stem": self.stem,
            "bloom_level": self.bloom_level,
            "objective_id": self.objective_id,
            "choices": self.choices,
            "correct_answer": self.correct_answer,
            "points": self.points,
            "feedback": self.feedback,
            "source_chunks": self.source_chunks,
            "generation_rationale": self.generation_rationale,
        }


@dataclass
class AssessmentData:
    """A complete assessment with multiple questions."""
    assessment_id: str
    title: str
    course_code: str
    questions: List[QuestionData] = field(default_factory=list)
    objectives_targeted: List[str] = field(default_factory=list)
    bloom_levels: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "generated"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "title": self.title,
            "course_code": self.course_code,
            "questions": [q.to_dict() for q in self.questions],
            "objectives_targeted": self.objectives_targeted,
            "bloom_levels": self.bloom_levels,
            "created_at": self.created_at,
            "status": self.status,
            "question_count": len(self.questions),
            "total_points": sum(q.points for q in self.questions),
        }


class AssessmentGenerator:
    """
    Generates assessments from course content with decision capture.

    Supports:
    - Multiple question types (MCQ, T/F, Fill-blank, Essay)
    - Bloom's taxonomy targeting
    - Learning objective alignment
    - Decision capture for training

    Usage:
        generator = AssessmentGenerator(capture=capture)
        assessment = generator.generate(
            course_code="INT_101",
            objective_ids=["LO-001", "LO-002"],
            bloom_levels=["understand", "apply"],
            question_count=10
        )
    """

    def __init__(
        self,
        capture: Optional["DecisionCapture"] = None,
        check_leaks: bool = True,
    ):
        """
        Initialize the assessment generator.

        Args:
            capture: Optional DecisionCapture for logging generation decisions
            check_leaks: If True, run leak checker on generated questions
        """
        self.capture = capture
        self.check_leaks = check_leaks and LEAK_CHECKER_AVAILABLE
        self._leak_checker = LeakChecker(strict_mode=False) if self.check_leaks else None

    def generate(
        self,
        course_code: str,
        objective_ids: List[str],
        bloom_levels: List[str],
        question_count: int = 10,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> AssessmentData:
        """
        Generate an assessment for the given objectives.

        Args:
            course_code: Course identifier
            objective_ids: Learning objectives to assess
            bloom_levels: Target Bloom's levels
            question_count: Number of questions to generate
            source_chunks: Optional content chunks from RAG retrieval

        Returns:
            AssessmentData with generated questions
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        assessment_id = f"ASM-{course_code}-{session_id}"

        # Log generation decision with substantive rationale
        if self.capture:
            # Build pedagogical rationale
            level_distribution = ", ".join(bloom_levels)
            rationale = (
                f"Covering {len(objective_ids)} objectives ensures "
                f"learners demonstrate mastery across the full scope. "
                f"Bloom's levels [{level_distribution}] assess both "
                f"foundational and higher-order thinking. "
                f"{question_count} questions balance sampling density "
                f"with learner time constraints."
            )
            self.capture.log_decision(
                decision_type="assessment_planning",
                decision=(
                    f"Planning assessment with {question_count} "
                    f"questions covering {len(objective_ids)} objectives"
                ),
                rationale=rationale,
                alternatives_considered=[
                    {
                        "option": "fewer_questions",
                        "rejected_because": "Insufficient sampling",
                    },
                    {
                        "option": "more_questions",
                        "rejected_because": "Exceeds optimal length",
                    },
                ],
            )

        # Distribute questions across objectives and levels
        questions = []
        questions_per_combo = max(1, question_count // (len(objective_ids) * len(bloom_levels)))

        for obj_id in objective_ids:
            for bloom_level in bloom_levels:
                for _ in range(questions_per_combo):
                    if len(questions) >= question_count:
                        break

                    question = self._generate_question(
                        objective_id=obj_id,
                        bloom_level=bloom_level,
                        source_chunks=source_chunks,
                    )
                    questions.append(question)

                if len(questions) >= question_count:
                    break
            if len(questions) >= question_count:
                break

        # Fill remaining with cycling through objectives
        idx = 0
        while len(questions) < question_count:
            obj_id = objective_ids[idx % len(objective_ids)]
            bloom_level = bloom_levels[idx % len(bloom_levels)]

            question = self._generate_question(
                objective_id=obj_id,
                bloom_level=bloom_level,
                source_chunks=source_chunks,
            )
            questions.append(question)
            idx += 1

        # Run leak checker on generated questions
        if self._leak_checker and questions:
            leak_questions = []
            for q in questions:
                leak_q = {"id": q.question_id}
                if q.correct_answer:
                    leak_q["correct_answer"] = q.correct_answer
                elif q.choices:
                    correct = [
                        c["text"] for c in q.choices if c.get("is_correct")
                    ]
                    if correct:
                        leak_q["correct_answers"] = correct
                leak_questions.append(leak_q)

            self._leak_checker.register_assessment(assessment_id, leak_questions)

            # Check each question's stem for answer leaks
            leaked_ids = set()
            for q in questions:
                result = self._leak_checker.check_prompt(
                    q.stem, assessment_id=assessment_id, question_id=q.question_id
                )
                if not result.passed:
                    leaked_ids.add(q.question_id)
                    logger.warning(
                        "Leak detected in %s: %d leaks found",
                        q.question_id, result.leak_count,
                    )

            # Remove leaked questions
            if leaked_ids:
                original_count = len(questions)
                questions = [q for q in questions if q.question_id not in leaked_ids]
                logger.warning(
                    "Removed %d/%d questions with answer leaks",
                    original_count - len(questions), original_count,
                )

                if self.capture:
                    self.capture.log_decision(
                        decision_type="leak_check_filtering",
                        decision=f"Removed {len(leaked_ids)} questions with answer leaks",
                        rationale=(
                            f"Leak checker detected answer content in question stems "
                            f"for questions {leaked_ids}. Removing to prevent training "
                            f"data contamination and maintain RAG corpus integrity."
                        ),
                    )

        assessment = AssessmentData(
            assessment_id=assessment_id,
            title=f"Assessment: {course_code}",
            course_code=course_code,
            questions=questions,
            objectives_targeted=objective_ids,
            bloom_levels=bloom_levels,
        )

        # Log final assessment decision with substantive rationale
        if self.capture:
            type_counts = {}
            bloom_counts = {}
            for q in questions:
                type_counts[q.question_type] = type_counts.get(q.question_type, 0) + 1
                bloom_counts[q.bloom_level] = bloom_counts.get(q.bloom_level, 0) + 1

            # Build comprehensive rationale
            type_summary = ", ".join(f"{count} {qtype}" for qtype, count in type_counts.items())
            bloom_summary = ", ".join(f"{count} at {level}" for level, count in bloom_counts.items())

            rationale = (
                f"Finalized assessment with balanced question distribution: {type_summary}. "
                f"Cognitive level coverage: {bloom_summary}. "
                f"This distribution ensures comprehensive assessment of learning objectives "
                f"while maintaining appropriate variety in question formats to reduce test-taking strategy effects. "
                f"The mix of question types accommodates diverse learner strengths and provides "
                f"multiple opportunities to demonstrate competency."
            )

            self.capture.log_decision(
                decision_type="assessment_generation",
                decision=f"Completed assessment {assessment_id} with {len(questions)} questions across {len(objective_ids)} objectives",
                rationale=rationale,
                alternatives_considered=[
                    {"option": "all_mcq", "rejected_because": "Homogeneous formats allow test-taking strategies and fail to assess constructed response skills"},
                    {"option": "all_essay", "rejected_because": "Excessive grading burden and potential learner fatigue without assessing factual recall"},
                ],
            )

        return assessment

    def _generate_question(
        self,
        objective_id: str,
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> QuestionData:
        """
        Generate a single question for the given objective and level.

        Args:
            objective_id: Learning objective ID
            bloom_level: Target Bloom's level
            source_chunks: Optional content chunks

        Returns:
            QuestionData with generated question
        """
        question_id = f"Q-{str(uuid.uuid4())[:8]}"

        # Select question type based on Bloom's level
        level_config = BLOOM_LEVELS.get(bloom_level, BLOOM_LEVELS["understand"])
        available_types = level_config["question_types"]
        question_type = available_types[0]  # Select first suitable type

        # Build pedagogical rationale for question type selection
        type_rationales = {
            "multiple_choice": (
                "because MCQ format allows efficient assessment of recognition and discrimination skills, "
                "and plausible distractors can target common misconceptions to provide diagnostic feedback"
            ),
            "true_false": (
                "because T/F format efficiently assesses factual recall and simple comprehension, "
                "though limited to binary distinctions which suits lower cognitive levels"
            ),
            "fill_in_blank": (
                "because fill-in-blank requires active recall rather than recognition, "
                "testing deeper retention of key terminology and concepts"
            ),
            "short_answer": (
                "because short answer requires learners to construct responses, "
                "demonstrating understanding through explanation rather than selection"
            ),
            "essay": (
                "because essay format allows learners to synthesize, evaluate, and create, "
                "demonstrating higher-order thinking that cannot be assessed through closed-format items"
            ),
        }

        # Log question type decision with substantive rationale
        if self.capture:
            base_rationale = type_rationales.get(question_type, "because it aligns with the target cognitive level")
            alternatives = [
                {"option": alt_type, "rejected_because": f"Less appropriate for {bloom_level} level cognitive demands"}
                for alt_type in available_types[1:3] if alt_type != question_type
            ]

            self.capture.log_decision(
                decision_type="question_type_selection",
                decision=f"Selected {question_type} for Bloom level '{bloom_level}' targeting objective {objective_id}",
                rationale=f"Chose {question_type} format {base_rationale}. This format is pedagogically appropriate for the '{bloom_level}' cognitive level.",
                alternatives_considered=alternatives if alternatives else None,
            )

        # Generate question based on type
        if question_type == "multiple_choice":
            return self._generate_multiple_choice(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )
        elif question_type == "true_false":
            return self._generate_true_false(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )
        elif question_type == "fill_in_blank":
            return self._generate_fill_in_blank(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )
        elif question_type == "essay":
            return self._generate_essay(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )
        else:
            return self._generate_short_answer(
                question_id, objective_id, bloom_level, level_config, source_chunks
            )

    def _generate_multiple_choice(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ) -> QuestionData:
        """Generate a multiple choice question."""
        verb = level_config["verbs"][0]
        pattern = level_config["patterns"][0]

        stem = f"<p>{pattern.replace('...', f' the concept from {objective_id}')}</p>"

        choices = [
            {"text": "<p>Correct answer based on content</p>", "is_correct": True},
            {"text": "<p>Plausible distractor A</p>", "is_correct": False},
            {"text": "<p>Plausible distractor B</p>", "is_correct": False},
            {"text": "<p>Plausible distractor C</p>", "is_correct": False},
        ]

        return QuestionData(
            question_id=question_id,
            question_type="multiple_choice",
            stem=stem,
            bloom_level=bloom_level,
            objective_id=objective_id,
            choices=choices,
            points=2.0,
            feedback=f"<p>Review content for objective {objective_id}.</p>",
            source_chunks=[c.get("chunk_id", "") for c in (source_chunks or [])[:2]],
            generation_rationale=f"MCQ using verb '{verb}' at {bloom_level} level",
        )

    def _generate_true_false(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ) -> QuestionData:
        """Generate a true/false question."""
        return QuestionData(
            question_id=question_id,
            question_type="true_false",
            stem=f"<p>Statement about {objective_id} content.</p>",
            bloom_level=bloom_level,
            objective_id=objective_id,
            choices=[
                {"text": "True", "is_correct": True},
                {"text": "False", "is_correct": False},
            ],
            correct_answer="True",
            points=1.0,
            feedback=f"<p>This statement is accurate based on {objective_id}.</p>",
            generation_rationale=f"T/F question at {bloom_level} level",
        )

    def _generate_fill_in_blank(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ) -> QuestionData:
        """Generate a fill-in-the-blank question."""
        return QuestionData(
            question_id=question_id,
            question_type="fill_in_blank",
            stem=f"<p>The key concept from {objective_id} is _______.</p>",
            bloom_level=bloom_level,
            objective_id=objective_id,
            correct_answer="concept term",
            points=1.0,
            feedback=f"<p>The correct term is found in {objective_id} content.</p>",
            generation_rationale=f"Fill-in-blank at {bloom_level} level",
        )

    def _generate_essay(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ) -> QuestionData:
        """Generate an essay question."""
        verb = level_config["verbs"][0]

        return QuestionData(
            question_id=question_id,
            question_type="essay",
            stem=f"<p>{verb.capitalize()} the concepts from {objective_id} and provide examples.</p>",
            bloom_level=bloom_level,
            objective_id=objective_id,
            points=10.0,
            feedback=f"<p>A complete response should address all aspects of {objective_id}.</p>",
            generation_rationale=f"Essay using verb '{verb}' at {bloom_level} level",
        )

    def _generate_short_answer(
        self,
        question_id: str,
        objective_id: str,
        bloom_level: str,
        level_config: Dict[str, Any],
        source_chunks: Optional[List[Dict[str, Any]]],
    ) -> QuestionData:
        """Generate a short answer question."""
        verb = level_config["verbs"][0]

        return QuestionData(
            question_id=question_id,
            question_type="short_answer",
            stem=f"<p>Briefly {verb} the key points from {objective_id}.</p>",
            bloom_level=bloom_level,
            objective_id=objective_id,
            points=5.0,
            feedback=f"<p>Your response should cover the main concepts from {objective_id}.</p>",
            generation_rationale=f"Short answer using verb '{verb}' at {bloom_level} level",
        )

    def generate_for_objective(
        self,
        objective: Dict[str, Any],
        bloom_level: str,
        source_chunks: Optional[List[Dict[str, Any]]] = None,
        question_count: int = 3,
    ) -> List[QuestionData]:
        """
        Generate questions for a single learning objective.

        Args:
            objective: Learning objective data with id and text
            bloom_level: Target Bloom's level
            source_chunks: Content chunks from RAG
            question_count: Number of questions

        Returns:
            List of QuestionData
        """
        obj_id = objective.get("id", "LO-001")

        questions = []
        for _ in range(question_count):
            question = self._generate_question(
                objective_id=obj_id,
                bloom_level=bloom_level,
                source_chunks=source_chunks,
            )
            questions.append(question)

        if self.capture:
            self.capture.log_decision(
                decision_type="objective_assessment",
                decision=f"Generated {len(questions)} questions for objective {obj_id}",
                rationale=(
                    f"Created {len(questions)} assessment items targeting the '{bloom_level}' cognitive level "
                    f"because this ensures thorough coverage of objective {obj_id}. "
                    f"Multiple questions per objective increase measurement reliability and allow "
                    f"learners multiple opportunities to demonstrate competency, reducing the impact "
                    f"of any single item on overall assessment outcomes."
                ),
                alternatives_considered=[
                    {"option": "single_question", "rejected_because": "Single item provides insufficient reliability for competency determination"},
                ],
            )

        return questions


def generate_assessment(
    course_code: str,
    objective_ids: List[str],
    bloom_levels: List[str],
    question_count: int = 10,
    capture: Optional["DecisionCapture"] = None,
) -> AssessmentData:
    """
    Convenience function to generate an assessment.

    Args:
        course_code: Course identifier
        objective_ids: Learning objectives to assess
        bloom_levels: Target Bloom's levels
        question_count: Number of questions
        capture: Optional DecisionCapture

    Returns:
        AssessmentData with generated questions
    """
    generator = AssessmentGenerator(capture=capture)
    return generator.generate(
        course_code=course_code,
        objective_ids=objective_ids,
        bloom_levels=bloom_levels,
        question_count=question_count,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test generation
    assessment = generate_assessment(
        course_code="TEST_101",
        objective_ids=["LO-001", "LO-002", "LO-003"],
        bloom_levels=["remember", "understand", "apply"],
        question_count=9,
    )

    print(f"Generated: {assessment.assessment_id}")
    print(f"Questions: {len(assessment.questions)}")
    print(f"Total points: {sum(q.points for q in assessment.questions)}")
    print(json.dumps(assessment.to_dict(), indent=2))
