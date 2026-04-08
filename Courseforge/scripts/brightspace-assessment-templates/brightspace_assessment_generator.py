#!/usr/bin/env python3
"""
Brightspace Assessment Generator
Generates IMSCC-compliant assessment XML files for Brightspace import.

This script creates properly formatted assessment files using the correct
namespaces and formats verified against actual Brightspace exports.

CORRECT NAMESPACES (from real Brightspace exports):
- Quiz: http://www.imsglobal.org/xsd/ims_qtiasiv1p2
- Assignment: http://www.imsglobal.org/xsd/imscc_extensions/assignment
- Discussion: http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3

RESOURCE TYPES:
- Quiz: imsqti_xmlv1p2/imscc_xmlv1p3/assessment
- Assignment: assignment_xmlv1p0
- Discussion: imsdt_xmlv1p3
"""

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Add Ed4All lib to path for decision capture
ED4ALL_ROOT = Path(__file__).resolve().parents[3]  # scripts/brightspace-assessment-templates/... → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

from generators import (  # noqa: E402
    AssignmentGenerator,
    Choice,
    DiscussionGenerator,
    ManifestGenerator,
    QuestionType,
    QuizGenerator,
    QuizQuestion,
)
from generators.quiz_generator import (  # noqa: E402
    AssessmentType,
    create_essay_question,
    create_fill_in_blank_question,
    create_multiple_choice_question,
    create_true_false_question,
)
from validators import (  # noqa: E402
    AssignmentValidator,
    DiscussionValidator,
    QTIValidator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BrightspaceAssessmentGenerator:
    """
    Generator for Brightspace-compatible IMSCC assessment files.

    Uses verified correct namespaces from actual Brightspace exports.
    Supports all 5 QTI question types:
    - Multiple Choice
    - Multiple Response
    - True/False
    - Fill in the Blank
    - Essay

    Resource types generated:
    - assignment_xmlv1p0 for assignments
    - imsdt_xmlv1p3 for discussions
    - imsqti_xmlv1p2/imscc_xmlv1p3/assessment for quizzes
    """

    def __init__(
        self,
        validate_output: bool = True,
        capture: Optional["DecisionCapture"] = None,
    ):
        """
        Initialize generator with validators.

        Args:
            validate_output: If True, validate all generated XML
            capture: Optional DecisionCapture for logging assessment decisions
        """
        self.validate_output = validate_output
        self.assignment_gen = AssignmentGenerator()
        self.discussion_gen = DiscussionGenerator()
        self.quiz_gen = QuizGenerator()
        self.manifest_gen = ManifestGenerator()
        self.capture = capture

        if validate_output:
            self.assignment_validator = AssignmentValidator()
            self.discussion_validator = DiscussionValidator()
            self.qti_validator = QTIValidator()

    def generate_assignment(self,
                            title: str,
                            instructions: str,
                            points: float = 100.0,
                            submission_types: List[str] = None,
                            validate: bool = None) -> str:
        """
        Generate assignment XML with correct namespace.

        Args:
            title: Assignment title
            instructions: HTML instructions
            points: Points possible
            submission_types: List of ['file', 'text', 'url']
            validate: Override default validation setting

        Returns:
            Valid assignment XML string
        """
        xml = self.assignment_gen.generate(
            title=title,
            instructions=instructions,
            points=points,
            submission_types=submission_types
        )

        should_validate = validate if validate is not None else self.validate_output
        if should_validate:
            result = self.assignment_validator.validate(xml)
            if not result.is_valid:
                logger.error(f"Assignment validation failed: {result.errors}")
                raise ValueError(f"Generated assignment failed validation: {result.errors}")

        return xml

    def generate_discussion(self,
                            title: str,
                            prompt: str,
                            validate: bool = None) -> str:
        """
        Generate discussion topic XML with correct namespace.

        IMPORTANT: Root element is <topic>, NOT <discussion>.

        Args:
            title: Discussion title
            prompt: HTML discussion prompt
            validate: Override default validation setting

        Returns:
            Valid discussion XML string
        """
        xml = self.discussion_gen.generate(
            title=title,
            prompt=prompt
        )

        should_validate = validate if validate is not None else self.validate_output
        if should_validate:
            result = self.discussion_validator.validate(xml)
            if not result.is_valid:
                logger.error(f"Discussion validation failed: {result.errors}")
                raise ValueError(f"Generated discussion failed validation: {result.errors}")

        return xml

    def generate_quiz(self,
                      title: str,
                      questions: List[QuizQuestion],
                      max_attempts: int = 1,
                      time_limit: int = 0,
                      assessment_type: AssessmentType = AssessmentType.EXAM,
                      validate: bool = None) -> str:
        """
        Generate QTI assessment XML with all 5 question types.

        Supported question types:
        - QuestionType.MULTIPLE_CHOICE
        - QuestionType.MULTIPLE_RESPONSE
        - QuestionType.TRUE_FALSE
        - QuestionType.FILL_IN_BLANK
        - QuestionType.ESSAY

        Args:
            title: Quiz title
            questions: List of QuizQuestion objects
            max_attempts: Maximum attempts (0 = unlimited)
            time_limit: Time limit in seconds (0 = no limit)
            assessment_type: Type of assessment
            validate: Override default validation setting

        Returns:
            Valid QTI XML string
        """
        xml = self.quiz_gen.generate(
            title=title,
            questions=questions,
            max_attempts=max_attempts,
            time_limit=time_limit,
            assessment_type=assessment_type
        )

        should_validate = validate if validate is not None else self.validate_output
        if should_validate:
            result = self.qti_validator.validate(xml)
            if not result.is_valid:
                logger.error(f"Quiz validation failed: {result.errors}")
                raise ValueError(f"Generated quiz failed validation: {result.errors}")

        return xml

    def generate_week_quiz(self, week_num: int, topic: str = "Course Content") -> str:
        """
        Generate a sample quiz for a course week.

        Creates a quiz with various question types demonstrating capabilities.

        Args:
            week_num: Week number
            topic: Topic name for question context

        Returns:
            QTI quiz XML string
        """
        questions = [
            create_multiple_choice_question(
                f"<p>Which concept from Week {week_num} {topic} is most fundamental?</p>",
                [
                    {"text": "<p>Core concept A</p>", "is_correct": True},
                    {"text": "<p>Supporting concept B</p>", "is_correct": False},
                    {"text": "<p>Related concept C</p>", "is_correct": False},
                    {"text": "<p>Advanced concept D</p>", "is_correct": False},
                ],
                points=2.0,
                feedback="<p>Review the week's materials for more details.</p>"
            ),
            create_true_false_question(
                f"<p>Week {week_num} introduces concepts that build on previous weeks.</p>",
                correct_answer=True,
                points=1.0,
                feedback="<p>Course content is designed to build progressively.</p>"
            ),
            create_fill_in_blank_question(
                f"<p>The main topic of Week {week_num} is _______.</p>",
                correct_answers=[topic, topic.lower()],
                case_sensitive=False,
                points=1.0,
                feedback=f"<p>The correct answer is {topic}.</p>"
            ),
        ]

        return self.generate_quiz(
            title=f"Week {week_num} Knowledge Check",
            questions=questions,
            max_attempts=2,
            time_limit=0,
            assessment_type=AssessmentType.QUIZ
        )

    def generate_week_assignment(self, week_num: int, topic: str = "Course Content") -> str:
        """
        Generate a sample assignment for a course week.

        Args:
            week_num: Week number
            topic: Topic name

        Returns:
            Assignment XML string
        """
        return self.generate_assignment(
            title=f"Week {week_num} Application Assignment",
            instructions=f"""<p>Complete the Week {week_num} assignment focusing on {topic} applications.</p>
<p><strong>Requirements:</strong></p>
<ul>
<li>Submit your work as a file (PDF or Word document preferred)</li>
<li>Include all calculations and show your work</li>
<li>Explain your reasoning clearly</li>
</ul>""",
            points=100.0,
            submission_types=['file', 'text']
        )

    def generate_week_discussion(self, week_num: int, topic: str = "Course Content") -> str:
        """
        Generate a sample discussion for a course week.

        Args:
            week_num: Week number
            topic: Topic name

        Returns:
            Discussion XML string
        """
        return self.generate_discussion(
            title=f"Week {week_num} Discussion Forum",
            prompt=f"""<p>Discuss how Week {week_num} {topic} concepts apply to your field of study or career interests.</p>
<p><strong>Discussion Guidelines:</strong></p>
<ul>
<li>Provide specific examples from your experience</li>
<li>Reference concepts from this week's materials</li>
<li>Respond thoughtfully to at least two classmates</li>
</ul>"""
        )

    def generate_all_assessments(self, output_dir: str, weeks: int = 4,
                                 topic: str = "Course Content") -> str:
        """
        Generate all assessment files for the specified number of weeks.

        Args:
            output_dir: Path to output directory
            weeks: Number of weeks to generate
            topic: Topic name for context

        Returns:
            Success message with count

        Raises:
            PermissionError: If directory cannot be created
            IOError: If files cannot be written
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        generated_count = 0

        for week in range(1, weeks + 1):
            try:
                # Generate quiz
                quiz_xml = self.generate_week_quiz(week, topic)
                quiz_path = output_path / f"quiz_week_{week:02d}.xml"
                quiz_path.write_text(quiz_xml, encoding='utf-8')
                generated_count += 1
                logger.info(f"Generated quiz for week {week}")

                # Generate assignment
                assignment_xml = self.generate_week_assignment(week, topic)
                assignment_path = output_path / f"assignment_week_{week:02d}.xml"
                assignment_path.write_text(assignment_xml, encoding='utf-8')
                generated_count += 1
                logger.info(f"Generated assignment for week {week}")

                # Generate discussion
                discussion_xml = self.generate_week_discussion(week, topic)
                discussion_path = output_path / f"discussion_week_{week:02d}.xml"
                discussion_path.write_text(discussion_xml, encoding='utf-8')
                generated_count += 1
                logger.info(f"Generated discussion for week {week}")

            except Exception as e:
                logger.error(f"Error generating assessments for week {week}: {e}")
                raise

        return f"Generated {generated_count} Brightspace-compatible assessment files"

    # Backward compatibility methods (deprecated)

    def generate_brightspace_quiz(self, week_num: int, quiz_data: dict = None) -> str:
        """DEPRECATED: Use generate_week_quiz instead."""
        logger.warning("generate_brightspace_quiz is deprecated, use generate_week_quiz")
        return self.generate_week_quiz(week_num)

    def generate_qti_quiz(self, week_num: int, quiz_data: dict = None) -> str:
        """DEPRECATED: Use generate_week_quiz instead."""
        logger.warning("generate_qti_quiz is deprecated, use generate_week_quiz")
        return self.generate_week_quiz(week_num)

    def generate_brightspace_assignment(self, week_num: int, assignment_data: dict = None) -> str:
        """DEPRECATED: Use generate_week_assignment instead."""
        logger.warning("generate_brightspace_assignment is deprecated, use generate_week_assignment")
        return self.generate_week_assignment(week_num)

    def generate_d2l_assignment(self, week_num: int, assignment_data: dict = None) -> str:
        """DEPRECATED: Use generate_week_assignment instead."""
        logger.warning("generate_d2l_assignment is deprecated, use generate_week_assignment")
        return self.generate_week_assignment(week_num)

    def generate_brightspace_discussion(self, week_num: int, discussion_data: dict = None) -> str:
        """DEPRECATED: Use generate_week_discussion instead."""
        logger.warning("generate_brightspace_discussion is deprecated, use generate_week_discussion")
        return self.generate_week_discussion(week_num)

    def generate_d2l_discussion(self, week_num: int, discussion_data: dict = None) -> str:
        """DEPRECATED: Use generate_week_discussion instead."""
        logger.warning("generate_d2l_discussion is deprecated, use generate_week_discussion")
        return self.generate_week_discussion(week_num)

    # Resource type helpers

    @staticmethod
    def get_quiz_resource_type() -> str:
        """Get the correct manifest resource type for quizzes."""
        return "imsqti_xmlv1p2/imscc_xmlv1p3/assessment"

    @staticmethod
    def get_assignment_resource_type() -> str:
        """Get the correct manifest resource type for assignments."""
        return "assignment_xmlv1p0"

    @staticmethod
    def get_discussion_resource_type() -> str:
        """Get the correct manifest resource type for discussions."""
        return "imsdt_xmlv1p3"


# Export key classes and functions for easy import
__all__ = [
    'BrightspaceAssessmentGenerator',
    'QuestionType',
    'QuizQuestion',
    'Choice',
    'AssessmentType',
    'create_multiple_choice_question',
    'create_true_false_question',
    'create_fill_in_blank_question',
    'create_essay_question',
]


if __name__ == "__main__":
    generator = BrightspaceAssessmentGenerator()
    result = generator.generate_all_assessments("/tmp/brightspace_assessments")
    print(f"SUCCESS: {result}")
