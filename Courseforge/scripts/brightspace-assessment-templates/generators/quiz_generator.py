"""
QTI Quiz XML Generator

Generates IMSCC QTI 1.2 assessment XML files with support for all 5 question types.
"""

import uuid
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from .base_generator import BaseGenerator, escape_for_cdata, escape_xml_attribute, generate_brightspace_id
from .constants import (
    NAMESPACES,
    SCHEMA_LOCATIONS,
    RESOURCE_TYPES,
    QTI_QUESTION_PROFILES,
    MAX_POINTS,
    MIN_POINTS,
    MAX_TITLE_LENGTH,
)


class QuestionType(Enum):
    """Supported QTI question types with their cc_profile values."""
    MULTIPLE_CHOICE = "cc.multiple_choice.v0p1"
    MULTIPLE_RESPONSE = "cc.multiple_response.v0p1"
    TRUE_FALSE = "cc.true_false.v0p1"
    FILL_IN_BLANK = "cc.fib.v0p1"
    ESSAY = "cc.essay.v0p1"


class AssessmentType(Enum):
    """Assessment profile types."""
    EXAM = ("cc.exam.v0p1", "Examination")
    QUIZ = ("cc.quiz.v0p1", "Quiz")
    SURVEY = ("cc.survey.v0p1", "Survey")
    GRADED_SURVEY = ("cc.graded_survey.v0p1", "Survey")


@dataclass
class Choice:
    """Answer choice for MC, TF, or MR questions."""
    text: str
    is_correct: bool = False
    feedback: str = ""
    identifier: str = field(default_factory=generate_brightspace_id)


@dataclass
class QuizQuestion:
    """Data class representing a quiz question."""
    question_type: QuestionType
    question_text: str
    points: float = 1.0
    choices: List[Choice] = field(default_factory=list)  # For MC, TF, MR
    correct_answers: List[str] = field(default_factory=list)  # For FIB
    case_sensitive: bool = False  # For FIB
    feedback: str = ""
    solution: str = ""  # For essay - model answer
    identifier: str = field(default_factory=generate_brightspace_id)


class QuizGenerator(BaseGenerator):
    """
    Generator for IMSCC QTI 1.2 assessment XML files.

    Namespace: http://www.imsglobal.org/xsd/ims_qtiasiv1p2
    Manifest resource type: imsqti_xmlv1p2/imscc_xmlv1p3/assessment

    Supports all 5 question types:
    - Multiple Choice (single select)
    - Multiple Response (multi-select)
    - True/False
    - Fill in the Blank
    - Essay (manual grading)
    """

    # QTI namespace - sourced from constants for single source of truth
    NAMESPACE = NAMESPACES['qti']
    SCHEMA_LOCATION = "http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd"

    # Manifest resource type - sourced from constants
    RESOURCE_TYPE = RESOURCE_TYPES['quiz']

    def generate(self,
                 title: str,
                 questions: List[QuizQuestion],
                 max_attempts: int = 1,
                 time_limit: int = 0,
                 assessment_type: AssessmentType = AssessmentType.EXAM,
                 identifier: str = None) -> str:
        """
        Generate complete QTI assessment XML.

        Args:
            title: Assessment title
            questions: List of QuizQuestion objects
            max_attempts: Maximum attempts allowed (0 = unlimited)
            time_limit: Time limit in seconds (0 = no limit)
            assessment_type: Type of assessment (exam, quiz, survey)
            identifier: Unique identifier (auto-generated if not provided)

        Returns:
            Valid QTI XML string

        Raises:
            ValueError: If validation fails (empty title, invalid points, etc.)
        """
        # Validate title
        if not title or not title.strip():
            raise ValueError("Assessment title is required")
        if len(title) > MAX_TITLE_LENGTH:
            raise ValueError(f"Assessment title exceeds maximum length ({MAX_TITLE_LENGTH} chars)")

        # Validate questions exist
        if not questions:
            raise ValueError("At least one question is required")

        # Validate each question's points
        for i, q in enumerate(questions):
            if q.points < MIN_POINTS:
                raise ValueError(f"Question {i+1}: Points must be non-negative (got {q.points})")
            if q.points > MAX_POINTS:
                raise ValueError(f"Question {i+1}: Points exceed maximum ({MAX_POINTS})")

        if identifier is None:
            identifier = self.generate_id()

        section_id = self.generate_id()
        cc_profile, qmd_type = assessment_type.value

        # Generate question items
        question_items = '\n'.join(
            self._generate_question(q) for q in questions
        )

        xml = f'''<?xml version="1.0" encoding="utf-8"?>
<questestinterop xmlns="{self.NAMESPACE}"
                 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                 xsi:schemaLocation="{self.NAMESPACE} {self.SCHEMA_LOCATION}">
  <assessment ident="{escape_xml_attribute(identifier)}" title="{escape_for_cdata(title)}">
    <qtimetadata>
      <qtimetadatafield>
        <fieldlabel>cc_profile</fieldlabel>
        <fieldentry>{cc_profile}</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>qmd_assessmenttype</fieldlabel>
        <fieldentry>{qmd_type}</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>cc_maxattempts</fieldlabel>
        <fieldentry>{max_attempts}</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>qmd_timelimit</fieldlabel>
        <fieldentry>{time_limit}</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="{escape_xml_attribute(section_id)}">
{question_items}
    </section>
  </assessment>
</questestinterop>'''

        return xml

    def _generate_question(self, question: QuizQuestion) -> str:
        """Route to appropriate question generator based on type."""
        if question.question_type == QuestionType.MULTIPLE_CHOICE:
            return self._generate_multiple_choice(question)
        elif question.question_type == QuestionType.TRUE_FALSE:
            return self._generate_true_false(question)
        elif question.question_type == QuestionType.FILL_IN_BLANK:
            return self._generate_fill_in_blank(question)
        elif question.question_type == QuestionType.MULTIPLE_RESPONSE:
            return self._generate_multiple_response(question)
        elif question.question_type == QuestionType.ESSAY:
            return self._generate_essay(question)
        else:
            raise ValueError(f"Unknown question type: {question.question_type}")

    def _generate_multiple_choice(self, q: QuizQuestion) -> str:
        """Generate multiple choice question XML."""
        # Validate input
        if not q.choices:
            raise ValueError("Multiple choice question requires at least one choice")

        correct_choices = [c for c in q.choices if c.is_correct]
        if len(correct_choices) == 0:
            raise ValueError("Multiple choice question requires exactly one correct answer")
        if len(correct_choices) > 1:
            raise ValueError("Multiple choice question must have exactly one correct answer (use multiple_response for multiple correct answers)")

        response_id = self.generate_id()

        # Build choices
        choices_xml = []
        correct_id = correct_choices[0].identifier
        for choice in q.choices:
            choices_xml.append(
                f'''              <response_label ident="{escape_xml_attribute(choice.identifier)}">
                <material><mattext texttype="text/html">{escape_for_cdata(choice.text)}</mattext></material>
              </response_label>'''
            )

        choices_str = '\n'.join(choices_xml)

        feedback_id = f"fb_{q.identifier}"

        return f'''      <item ident="{escape_xml_attribute(q.identifier)}">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>{QuestionType.MULTIPLE_CHOICE.value}</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>cc_weighting</fieldlabel>
              <fieldentry>{q.points}</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material>
            <mattext texttype="text/html">{escape_for_cdata(q.question_text)}</mattext>
          </material>
          <response_lid ident="{escape_xml_attribute(response_id)}" rcardinality="Single">
            <render_choice>
{choices_str}
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar minvalue="0" maxvalue="100" varname="SCORE" vartype="Decimal"/>
          </outcomes>
          <respcondition continue="Yes">
            <conditionvar><other/></conditionvar>
            <displayfeedback feedbacktype="Response" linkrefid="{escape_xml_attribute(feedback_id)}"/>
          </respcondition>
          <respcondition continue="No">
            <conditionvar>
              <varequal respident="{escape_xml_attribute(response_id)}">{escape_for_cdata(correct_id)}</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">100</setvar>
          </respcondition>
        </resprocessing>
        <itemfeedback ident="{escape_xml_attribute(feedback_id)}">
          <flow_mat>
            <material><mattext texttype="text/html">{escape_for_cdata(q.feedback)}</mattext></material>
          </flow_mat>
        </itemfeedback>
      </item>'''

    def _generate_true_false(self, q: QuizQuestion) -> str:
        """Generate true/false question XML."""
        # Validate input - must have exactly 2 choices (True and False)
        if len(q.choices) != 2:
            raise ValueError("True/false question must have exactly 2 choices")

        correct_choices = [c for c in q.choices if c.is_correct]
        if len(correct_choices) != 1:
            raise ValueError("True/false question requires exactly one correct answer")

        response_id = self.generate_id()
        true_id = self.generate_id()
        false_id = self.generate_id()

        # Determine correct answer by checking which choice (index 0 = True, index 1 = False)
        # This relies on the convention that True is listed first
        # The create_true_false_question() function enforces this order
        correct_id = true_id if q.choices[0].is_correct else false_id

        feedback_id = f"fb_{q.identifier}"

        return f'''      <item ident="{escape_xml_attribute(q.identifier)}">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>{QuestionType.TRUE_FALSE.value}</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>cc_weighting</fieldlabel>
              <fieldentry>{q.points}</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material>
            <mattext texttype="text/html">{escape_for_cdata(q.question_text)}</mattext>
          </material>
          <response_lid ident="{escape_xml_attribute(response_id)}" rcardinality="Single">
            <render_choice>
              <response_label ident="{escape_xml_attribute(true_id)}">
                <material><mattext>True</mattext></material>
              </response_label>
              <response_label ident="{escape_xml_attribute(false_id)}">
                <material><mattext>False</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar minvalue="0" maxvalue="100" varname="SCORE" vartype="Decimal"/>
          </outcomes>
          <respcondition continue="Yes">
            <conditionvar><other/></conditionvar>
            <displayfeedback feedbacktype="Response" linkrefid="{escape_xml_attribute(feedback_id)}"/>
          </respcondition>
          <respcondition continue="No">
            <conditionvar>
              <varequal respident="{escape_xml_attribute(response_id)}">{escape_for_cdata(correct_id)}</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">100</setvar>
          </respcondition>
        </resprocessing>
        <itemfeedback ident="{escape_xml_attribute(feedback_id)}">
          <flow_mat>
            <material><mattext texttype="text/html">{escape_for_cdata(q.feedback)}</mattext></material>
          </flow_mat>
        </itemfeedback>
      </item>'''

    def _generate_fill_in_blank(self, q: QuizQuestion) -> str:
        """Generate fill in the blank question XML."""
        # Validate input
        if not q.correct_answers:
            raise ValueError("Fill-in-blank question requires at least one correct answer")

        # Use full UUID for FIB response_id to avoid collision (birthday paradox with short hashes)
        response_id = f"fib_{uuid.uuid4().hex}"
        answer_label_id = self.generate_id()

        # Build correct answer conditions with OR logic for multiple answers
        case_attr = "Yes" if q.case_sensitive else "No"
        conditions = []
        for answer in q.correct_answers:
            conditions.append(
                f'                <varequal case="{case_attr}" respident="{escape_xml_attribute(response_id)}">{escape_for_cdata(answer)}</varequal>'
            )

        # If multiple answers, wrap in <or> tag; otherwise just use the single condition
        if len(conditions) > 1:
            conditions_str = '              <or>\n' + '\n'.join(conditions) + '\n              </or>'
        else:
            # Single answer - use it directly (adjust indentation)
            conditions_str = conditions[0].replace('                ', '              ')

        feedback_id = f"fb_{q.identifier}"

        return f'''      <item ident="{escape_xml_attribute(q.identifier)}">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>{QuestionType.FILL_IN_BLANK.value}</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>cc_weighting</fieldlabel>
              <fieldentry>{q.points}</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material>
            <mattext texttype="text/html">{escape_for_cdata(q.question_text)}</mattext>
          </material>
          <response_str ident="{escape_xml_attribute(response_id)}">
            <render_fib>
              <response_label ident="{escape_xml_attribute(answer_label_id)}"/>
            </render_fib>
          </response_str>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar minvalue="0" maxvalue="100" varname="SCORE" vartype="Decimal"/>
          </outcomes>
          <respcondition continue="No">
            <conditionvar>
{conditions_str}
            </conditionvar>
            <setvar action="Set" varname="SCORE">100</setvar>
          </respcondition>
          <respcondition continue="Yes">
            <conditionvar><other/></conditionvar>
            <setvar action="Set" varname="SCORE">0</setvar>
          </respcondition>
          <respcondition continue="Yes">
            <conditionvar><other/></conditionvar>
            <displayfeedback feedbacktype="Response" linkrefid="{escape_xml_attribute(feedback_id)}"/>
          </respcondition>
        </resprocessing>
        <itemfeedback ident="{escape_xml_attribute(feedback_id)}">
          <flow_mat>
            <material><mattext texttype="text/html">{escape_for_cdata(q.feedback)}</mattext></material>
          </flow_mat>
        </itemfeedback>
      </item>'''

    def _generate_multiple_response(self, q: QuizQuestion) -> str:
        """Generate multiple response (multi-select) question XML."""
        # Validate input
        if not q.choices:
            raise ValueError("Multiple response question requires at least one choice")

        correct_choices = [c for c in q.choices if c.is_correct]
        if len(correct_choices) == 0:
            raise ValueError("Multiple response question requires at least one correct answer")

        response_id = self.generate_id()

        # Build choices
        choices_xml = []
        correct_ids = []
        incorrect_ids = []
        for choice in q.choices:
            if choice.is_correct:
                correct_ids.append(choice.identifier)
            else:
                incorrect_ids.append(choice.identifier)
            choices_xml.append(
                f'''              <response_label ident="{escape_xml_attribute(choice.identifier)}">
                <material><mattext texttype="text/html">{escape_for_cdata(choice.text)}</mattext></material>
              </response_label>'''
            )

        choices_str = '\n'.join(choices_xml)

        # Build condition: all correct AND not any incorrect
        conditions = []
        for cid in correct_ids:
            conditions.append(f'                <varequal respident="{escape_xml_attribute(response_id)}">{escape_for_cdata(cid)}</varequal>')
        for iid in incorrect_ids:
            conditions.append(f'                <not><varequal respident="{escape_xml_attribute(response_id)}">{escape_for_cdata(iid)}</varequal></not>')
        conditions_str = '\n'.join(conditions)

        feedback_id = f"fb_{q.identifier}"

        return f'''      <item ident="{escape_xml_attribute(q.identifier)}">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>{QuestionType.MULTIPLE_RESPONSE.value}</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>cc_weighting</fieldlabel>
              <fieldentry>{q.points}</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material>
            <mattext texttype="text/html">{escape_for_cdata(q.question_text)}</mattext>
          </material>
          <response_lid ident="{escape_xml_attribute(response_id)}" rcardinality="Multiple">
            <render_choice>
{choices_str}
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar minvalue="0" maxvalue="100" varname="SCORE" vartype="Decimal"/>
          </outcomes>
          <respcondition continue="No">
            <conditionvar>
              <and>
{conditions_str}
              </and>
            </conditionvar>
            <setvar action="Set" varname="SCORE">100</setvar>
          </respcondition>
          <respcondition continue="Yes">
            <conditionvar><other/></conditionvar>
            <displayfeedback feedbacktype="Response" linkrefid="{escape_xml_attribute(feedback_id)}"/>
          </respcondition>
        </resprocessing>
        <itemfeedback ident="{escape_xml_attribute(feedback_id)}">
          <flow_mat>
            <material><mattext texttype="text/html">{escape_for_cdata(q.feedback)}</mattext></material>
          </flow_mat>
        </itemfeedback>
      </item>'''

    def _generate_essay(self, q: QuizQuestion) -> str:
        """Generate essay question XML (manual grading)."""
        response_id = self.generate_id()
        answer_label_id = self.generate_id()
        feedback_id = f"fb_{q.identifier}"
        solution_id = f"sol_{q.identifier}"

        return f'''      <item ident="{escape_xml_attribute(q.identifier)}">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>{QuestionType.ESSAY.value}</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>qmd_computerscored</fieldlabel>
              <fieldentry>No</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>cc_weighting</fieldlabel>
              <fieldentry>{q.points}</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material>
            <mattext texttype="text/html">{escape_for_cdata(q.question_text)}</mattext>
          </material>
          <response_str ident="{escape_xml_attribute(response_id)}">
            <render_fib>
              <response_label ident="{escape_xml_attribute(answer_label_id)}"/>
            </render_fib>
          </response_str>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar minvalue="0" maxvalue="100" varname="SCORE" vartype="Decimal"/>
          </outcomes>
          <respcondition>
            <conditionvar><other/></conditionvar>
            <displayfeedback feedbacktype="Response" linkrefid="{escape_xml_attribute(feedback_id)}"/>
            <displayfeedback feedbacktype="Solution" linkrefid="{escape_xml_attribute(solution_id)}"/>
          </respcondition>
        </resprocessing>
        <itemfeedback ident="{escape_xml_attribute(solution_id)}">
          <solution>
            <solutionmaterial>
              <flow_mat>
                <material><mattext texttype="text/html">{escape_for_cdata(q.solution)}</mattext></material>
              </flow_mat>
            </solutionmaterial>
          </solution>
        </itemfeedback>
        <itemfeedback ident="{escape_xml_attribute(feedback_id)}">
          <flow_mat>
            <material><mattext texttype="text/html">{escape_for_cdata(q.feedback)}</mattext></material>
          </flow_mat>
        </itemfeedback>
      </item>'''

    def get_resource_type(self) -> str:
        """Return the manifest resource type for QTI assessments."""
        return self.RESOURCE_TYPE

    def get_namespace(self) -> str:
        """Return the XML namespace for QTI."""
        return self.NAMESPACE


# Convenience functions

def create_multiple_choice_question(
    question_text: str,
    choices: List[Dict[str, Any]],
    points: float = 1.0,
    feedback: str = ""
) -> QuizQuestion:
    """
    Create a multiple choice question.

    Args:
        question_text: The question text
        choices: List of dicts with 'text' and 'is_correct' keys
        points: Point value
        feedback: General feedback

    Returns:
        QuizQuestion object
    """
    choice_objs = []
    for c in choices:
        if 'text' not in c:
            raise ValueError("Each choice must have a 'text' key")
        choice_objs.append(Choice(text=c['text'], is_correct=c.get('is_correct', False)))
    return QuizQuestion(
        question_type=QuestionType.MULTIPLE_CHOICE,
        question_text=question_text,
        choices=choice_objs,
        points=points,
        feedback=feedback
    )


def create_true_false_question(
    question_text: str,
    correct_answer: bool,
    points: float = 1.0,
    feedback: str = ""
) -> QuizQuestion:
    """
    Create a true/false question.

    Args:
        question_text: The question text
        correct_answer: True if the answer is True, False if False
        points: Point value
        feedback: General feedback

    Returns:
        QuizQuestion object
    """
    choices = [
        Choice(text="True", is_correct=correct_answer),
        Choice(text="False", is_correct=not correct_answer)
    ]
    return QuizQuestion(
        question_type=QuestionType.TRUE_FALSE,
        question_text=question_text,
        choices=choices,
        points=points,
        feedback=feedback
    )


def create_multiple_response_question(
    question_text: str,
    choices: List[Dict[str, Any]],
    points: float = 1.0,
    feedback: str = ""
) -> QuizQuestion:
    """
    Create a multiple response (multi-select) question.

    Args:
        question_text: The question text
        choices: List of dicts with 'text' and 'is_correct' keys
                 (multiple choices can be marked correct)
        points: Point value
        feedback: General feedback

    Returns:
        QuizQuestion object
    """
    choice_objs = []
    for c in choices:
        if 'text' not in c:
            raise ValueError("Each choice must have a 'text' key")
        choice_objs.append(Choice(text=c['text'], is_correct=c.get('is_correct', False)))
    return QuizQuestion(
        question_type=QuestionType.MULTIPLE_RESPONSE,
        question_text=question_text,
        choices=choice_objs,
        points=points,
        feedback=feedback
    )


def create_fill_in_blank_question(
    question_text: str,
    correct_answers: List[str],
    case_sensitive: bool = False,
    points: float = 1.0,
    feedback: str = ""
) -> QuizQuestion:
    """
    Create a fill in the blank question.

    Args:
        question_text: The question text (use _______ for blank)
        correct_answers: List of acceptable answers
        case_sensitive: Whether matching is case-sensitive
        points: Point value
        feedback: General feedback

    Returns:
        QuizQuestion object
    """
    return QuizQuestion(
        question_type=QuestionType.FILL_IN_BLANK,
        question_text=question_text,
        correct_answers=correct_answers,
        case_sensitive=case_sensitive,
        points=points,
        feedback=feedback
    )


def create_essay_question(
    question_text: str,
    points: float = 10.0,
    solution: str = "",
    feedback: str = ""
) -> QuizQuestion:
    """
    Create an essay question (manual grading).

    Args:
        question_text: The question/prompt text
        points: Point value
        solution: Model answer or rubric information
        feedback: General feedback shown after submission

    Returns:
        QuizQuestion object
    """
    return QuizQuestion(
        question_type=QuestionType.ESSAY,
        question_text=question_text,
        points=points,
        solution=solution,
        feedback=feedback
    )
