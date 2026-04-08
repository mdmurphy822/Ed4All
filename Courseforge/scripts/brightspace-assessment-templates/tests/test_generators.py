#!/usr/bin/env python3
"""
Unit tests for Brightspace assessment generators.

Tests all generator classes for correct namespace usage, XML structure,
and Brightspace compatibility.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from generators import (
    AssignmentGenerator,
    DiscussionGenerator,
    QuizGenerator,
    ManifestGenerator,
    ResourceEntry,
    QuestionType,
    QuizQuestion,
    Choice,
    generate_brightspace_id,
)
from generators.base_generator import (
    escape_xml_attribute,
    escape_for_cdata,
    escape_xml_content,
)
from generators.quiz_generator import (
    create_multiple_choice_question,
    create_multiple_response_question,
    create_true_false_question,
    create_fill_in_blank_question,
    create_essay_question,
    AssessmentType,
)
from generators.constants import (
    MAX_TITLE_LENGTH,
    MAX_POINTS,
    MIN_POINTS,
    MAX_CONTENT_LENGTH,
)


class TestAssignmentGenerator(unittest.TestCase):
    """Test assignment XML generation."""

    def setUp(self):
        self.generator = AssignmentGenerator()

    def test_correct_namespace(self):
        """Verify assignment uses correct IMSCC namespace."""
        xml = self.generator.generate(
            title="Test Assignment",
            instructions="<p>Test instructions</p>",
            points=100.0
        )
        self.assertIn("http://www.imsglobal.org/xsd/imscc_extensions/assignment", xml)
        # Should NOT contain deprecated d2l_2p0 namespace
        self.assertNotIn("d2l_2p0", xml)
        self.assertNotIn("desire2learn", xml)

    def test_correct_root_element(self):
        """Verify assignment uses <assignment> root element."""
        xml = self.generator.generate(
            title="Test Assignment",
            instructions="<p>Test instructions</p>",
            points=100.0
        )
        self.assertIn("<assignment xmlns=", xml)

    def test_required_elements_present(self):
        """Verify all required elements are present."""
        xml = self.generator.generate(
            title="Test Assignment",
            instructions="<p>Test instructions</p>",
            points=100.0
        )
        self.assertIn("<title>", xml)
        self.assertIn("<instructor_text texttype=\"text/html\">", xml)
        self.assertIn("<submission_formats>", xml)
        self.assertIn("<gradable points_possible=", xml)

    def test_points_formatting(self):
        """Verify points formatted with 9 decimal places."""
        xml = self.generator.generate(
            title="Test Assignment",
            instructions="<p>Test</p>",
            points=100.0
        )
        self.assertIn('points_possible="100.000000000"', xml)

    def test_submission_types(self):
        """Verify submission types are correctly formatted."""
        xml = self.generator.generate(
            title="Test Assignment",
            instructions="<p>Test</p>",
            points=100.0,
            submission_types=['file', 'text', 'url']
        )
        self.assertIn('<format type="file" />', xml)
        self.assertIn('<format type="text" />', xml)
        self.assertIn('<format type="url" />', xml)

    def test_html_escaping(self):
        """Verify HTML content is properly escaped."""
        xml = self.generator.generate(
            title="Test & Title",
            instructions="<p>Instructions with <strong>tags</strong></p>",
            points=100.0
        )
        # Title should be escaped
        self.assertIn("Test &amp; Title", xml)

    def test_resource_type(self):
        """Verify correct resource type is returned."""
        self.assertEqual(self.generator.get_resource_type(), "assignment_xmlv1p0")


class TestDiscussionGenerator(unittest.TestCase):
    """Test discussion XML generation."""

    def setUp(self):
        self.generator = DiscussionGenerator()

    def test_correct_namespace(self):
        """Verify discussion uses correct IMSCC namespace."""
        xml = self.generator.generate(
            title="Test Discussion",
            prompt="<p>Discussion prompt</p>"
        )
        self.assertIn("http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3", xml)

    def test_topic_root_element(self):
        """Verify discussion uses <topic> root element, NOT <discussion>."""
        xml = self.generator.generate(
            title="Test Discussion",
            prompt="<p>Discussion prompt</p>"
        )
        self.assertIn("<topic xmlns=", xml)
        # Should NOT start with <discussion>
        self.assertNotIn("<discussion", xml.split("</topic>")[0])

    def test_required_elements_present(self):
        """Verify all required elements are present."""
        xml = self.generator.generate(
            title="Test Discussion",
            prompt="<p>Discussion prompt</p>"
        )
        self.assertIn("<title>", xml)
        self.assertIn("<text texttype=\"text/html\">", xml)

    def test_attachments(self):
        """Verify attachments are correctly formatted."""
        xml = self.generator.generate(
            title="Test Discussion",
            prompt="<p>Prompt</p>",
            attachments=["file1.pdf", "file2.doc"]
        )
        self.assertIn("<attachments>", xml)
        self.assertIn('href="file1.pdf"', xml)
        self.assertIn('href="file2.doc"', xml)

    def test_resource_type(self):
        """Verify correct resource type is returned."""
        self.assertEqual(self.generator.get_resource_type(), "imsdt_xmlv1p3")


class TestQuizGenerator(unittest.TestCase):
    """Test QTI quiz XML generation."""

    def setUp(self):
        self.generator = QuizGenerator()

    def test_correct_namespace(self):
        """Verify quiz uses correct QTI namespace."""
        questions = [
            create_multiple_choice_question(
                "<p>Test question?</p>",
                [
                    {"text": "<p>A</p>", "is_correct": True},
                    {"text": "<p>B</p>", "is_correct": False},
                ],
                points=1.0
            )
        ]
        xml = self.generator.generate(
            title="Test Quiz",
            questions=questions
        )
        self.assertIn("http://www.imsglobal.org/xsd/ims_qtiasiv1p2", xml)

    def test_root_element(self):
        """Verify quiz uses <questestinterop> root element."""
        questions = [
            create_true_false_question("<p>True or false?</p>", True, 1.0)
        ]
        xml = self.generator.generate(
            title="Test Quiz",
            questions=questions
        )
        self.assertIn("<questestinterop", xml)

    def test_multiple_choice_question(self):
        """Test multiple choice question generation."""
        questions = [
            create_multiple_choice_question(
                "<p>Which is correct?</p>",
                [
                    {"text": "<p>Answer A</p>", "is_correct": True},
                    {"text": "<p>Answer B</p>", "is_correct": False},
                    {"text": "<p>Answer C</p>", "is_correct": False},
                ],
                points=2.0
            )
        ]
        xml = self.generator.generate(title="MC Quiz", questions=questions)
        self.assertIn("cc.multiple_choice.v0p1", xml)
        self.assertIn('rcardinality="Single"', xml)

    def test_true_false_question(self):
        """Test true/false question generation."""
        questions = [
            create_true_false_question(
                "<p>The sky is blue.</p>",
                correct_answer=True,
                points=1.0
            )
        ]
        xml = self.generator.generate(title="TF Quiz", questions=questions)
        self.assertIn("cc.true_false.v0p1", xml)

    def test_fill_in_blank_question(self):
        """Test fill-in-the-blank question generation."""
        questions = [
            create_fill_in_blank_question(
                "<p>The capital of France is _______.</p>",
                correct_answers=["Paris", "paris"],
                case_sensitive=False,
                points=1.0
            )
        ]
        xml = self.generator.generate(title="FIB Quiz", questions=questions)
        self.assertIn("cc.fib.v0p1", xml)
        self.assertIn("<render_fib", xml)

    def test_essay_question(self):
        """Test essay question generation."""
        questions = [
            create_essay_question(
                "<p>Explain the concept in your own words.</p>",
                points=10.0
            )
        ]
        xml = self.generator.generate(title="Essay Quiz", questions=questions)
        self.assertIn("cc.essay.v0p1", xml)
        self.assertIn("qmd_computerscored", xml)

    def test_multiple_response_question(self):
        """Test multiple response (multi-select) question generation."""
        questions = [
            create_multiple_response_question(
                "<p>Select all that apply:</p>",
                [
                    {"text": "<p>Correct A</p>", "is_correct": True},
                    {"text": "<p>Correct B</p>", "is_correct": True},
                    {"text": "<p>Wrong C</p>", "is_correct": False},
                ],
                points=3.0
            )
        ]
        xml = self.generator.generate(title="MR Quiz", questions=questions)
        self.assertIn("cc.multiple_response.v0p1", xml)
        self.assertIn('rcardinality="Multiple"', xml)

    def test_fib_multiple_answers_or_logic(self):
        """Test fill-in-blank with multiple answers uses OR logic."""
        questions = [
            create_fill_in_blank_question(
                "<p>The capital of France is _______.</p>",
                correct_answers=["Paris", "paris", "PARIS"],
                case_sensitive=False,
                points=1.0
            )
        ]
        xml = self.generator.generate(title="FIB Quiz", questions=questions)
        # Should contain <or> tag when multiple answers exist
        self.assertIn("<or>", xml)

    def test_mc_validation_no_correct(self):
        """Test that MC question with no correct answer raises error."""
        questions = [
            QuizQuestion(
                question_type=QuestionType.MULTIPLE_CHOICE,
                question_text="<p>Test?</p>",
                choices=[
                    Choice(text="A", is_correct=False),
                    Choice(text="B", is_correct=False),
                ],
                points=1.0
            )
        ]
        with self.assertRaises(ValueError):
            self.generator.generate(title="Test Quiz", questions=questions)

    def test_mc_validation_multiple_correct(self):
        """Test that MC question with multiple correct answers raises error."""
        questions = [
            QuizQuestion(
                question_type=QuestionType.MULTIPLE_CHOICE,
                question_text="<p>Test?</p>",
                choices=[
                    Choice(text="A", is_correct=True),
                    Choice(text="B", is_correct=True),
                ],
                points=1.0
            )
        ]
        with self.assertRaises(ValueError):
            self.generator.generate(title="Test Quiz", questions=questions)

    def test_fib_validation_empty_answers(self):
        """Test that FIB question with empty answers raises error."""
        questions = [
            QuizQuestion(
                question_type=QuestionType.FILL_IN_BLANK,
                question_text="<p>Answer: _______</p>",
                correct_answers=[],  # Empty!
                points=1.0
            )
        ]
        with self.assertRaises(ValueError):
            self.generator.generate(title="Test Quiz", questions=questions)

    def test_mr_validation_empty_choices(self):
        """Test that MR question with empty choices raises error."""
        questions = [
            QuizQuestion(
                question_type=QuestionType.MULTIPLE_RESPONSE,
                question_text="<p>Select all:</p>",
                choices=[],  # Empty!
                points=1.0
            )
        ]
        with self.assertRaises(ValueError):
            self.generator.generate(title="Test Quiz", questions=questions)

    def test_resource_type(self):
        """Verify correct resource type is returned."""
        self.assertEqual(
            self.generator.get_resource_type(),
            "imsqti_xmlv1p2/imscc_xmlv1p3/assessment"
        )


class TestManifestGenerator(unittest.TestCase):
    """Test manifest XML generation."""

    def setUp(self):
        self.generator = ManifestGenerator()

    def test_correct_namespace(self):
        """Verify manifest uses correct IMSCC 1.3 namespace."""
        xml = self.generator.generate(
            course_title="Test Course",
            resources=[]
        )
        self.assertIn("http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1", xml)

    def test_schema_version(self):
        """Verify manifest has correct schema version."""
        xml = self.generator.generate(
            course_title="Test Course",
            resources=[]
        )
        self.assertIn("<schemaversion>1.3.0</schemaversion>", xml)

    def test_resource_types(self):
        """Verify resource types are correctly mapped."""
        types = ManifestGenerator.MANIFEST_RESOURCE_TYPES
        self.assertEqual(types['assignment'], "assignment_xmlv1p0")
        self.assertEqual(types['discussion'], "imsdt_xmlv1p3")
        self.assertEqual(types['quiz'], "imsqti_xmlv1p2/imscc_xmlv1p3/assessment")

    def test_resource_creation(self):
        """Test resource entry creation."""
        resource = self.generator.create_assignment_resource(
            href="assignment/assignment_01.xml",
            title="Week 1 Assignment"
        )
        self.assertEqual(resource.resource_type, "assignment_xmlv1p0")
        self.assertEqual(resource.title, "Week 1 Assignment")

    def test_manifest_with_resources(self):
        """Test manifest generation with resources."""
        resources = [
            self.generator.create_assignment_resource(
                href="assignment_01.xml",
                title="Assignment 1"
            ),
            self.generator.create_discussion_resource(
                href="discussion_01.xml",
                title="Discussion 1"
            ),
            self.generator.create_quiz_resource(
                href="quiz_01.xml",
                title="Quiz 1"
            ),
        ]
        xml = self.generator.generate(
            course_title="Test Course",
            resources=resources
        )
        self.assertIn("assignment_xmlv1p0", xml)
        self.assertIn("imsdt_xmlv1p3", xml)
        self.assertIn("imsqti_xmlv1p2/imscc_xmlv1p3/assessment", xml)


class TestBrightspaceIdGeneration(unittest.TestCase):
    """Test Brightspace ID generation."""

    def test_id_format(self):
        """Verify ID format matches Brightspace pattern."""
        id = generate_brightspace_id()
        # Should start with 'i' prefix
        self.assertTrue(id.startswith('i'))
        # Should be 33 characters total (i + 32 hex chars)
        self.assertEqual(len(id), 33)
        # Should be lowercase hex after prefix
        self.assertTrue(id[1:].islower() or id[1:].isdigit())

    def test_id_uniqueness(self):
        """Verify generated IDs are unique."""
        ids = [generate_brightspace_id() for _ in range(100)]
        self.assertEqual(len(ids), len(set(ids)))


class TestEscapeXmlAttribute(unittest.TestCase):
    """Test XML attribute escaping function."""

    def test_escape_double_quotes(self):
        """Test that double quotes are escaped."""
        result = escape_xml_attribute('test"value')
        self.assertEqual(result, 'test&quot;value')

    def test_escape_single_quotes(self):
        """Test that single quotes are escaped."""
        result = escape_xml_attribute("test'value")
        self.assertEqual(result, 'test&apos;value')

    def test_escape_ampersand(self):
        """Test that ampersands are escaped."""
        result = escape_xml_attribute('test&value')
        self.assertEqual(result, 'test&amp;value')

    def test_escape_less_than(self):
        """Test that less-than is escaped."""
        result = escape_xml_attribute('test<value')
        self.assertEqual(result, 'test&lt;value')

    def test_escape_greater_than(self):
        """Test that greater-than is escaped."""
        result = escape_xml_attribute('test>value')
        self.assertEqual(result, 'test&gt;value')

    def test_escape_combined_special_chars(self):
        """Test escaping multiple special characters."""
        result = escape_xml_attribute('<test>&"value\'')
        self.assertEqual(result, '&lt;test&gt;&amp;&quot;value&apos;')

    def test_escape_none_returns_empty(self):
        """Test that None returns empty string."""
        result = escape_xml_attribute(None)
        self.assertEqual(result, '')

    def test_escape_empty_string(self):
        """Test that empty string returns empty string."""
        result = escape_xml_attribute('')
        self.assertEqual(result, '')

    def test_escape_normal_text_unchanged(self):
        """Test that normal text is not changed."""
        result = escape_xml_attribute('normal text 123')
        self.assertEqual(result, 'normal text 123')

    def test_escaped_output_is_xml_safe(self):
        """Test that escaped output can be safely embedded in XML."""
        dangerous_input = 'id="break" onclick="alert(1)"'
        escaped = escape_xml_attribute(dangerous_input)
        # Verify the escaped output is XML-safe
        self.assertIn('&quot;', escaped)
        # Verify we can create valid XML with it
        xml = f'<?xml version="1.0"?><root attr="{escaped}"/>'
        from xml.etree import ElementTree as ET
        elem = ET.fromstring(xml)
        self.assertEqual(elem.get('attr'), dangerous_input)


class TestBoundaryLimits(unittest.TestCase):
    """Test boundary conditions for validation limits."""

    def setUp(self):
        self.assignment_gen = AssignmentGenerator()
        self.discussion_gen = DiscussionGenerator()

    def test_title_at_max_length(self):
        """Test title at exactly MAX_TITLE_LENGTH."""
        title = 'A' * MAX_TITLE_LENGTH
        # Should succeed
        xml = self.assignment_gen.generate(
            title=title,
            instructions="<p>Test</p>",
            points=100.0
        )
        self.assertIn(title, xml)

    def test_title_exceeds_max_length(self):
        """Test title exceeding MAX_TITLE_LENGTH raises error."""
        title = 'A' * (MAX_TITLE_LENGTH + 1)
        with self.assertRaises(ValueError) as ctx:
            self.assignment_gen.generate(
                title=title,
                instructions="<p>Test</p>",
                points=100.0
            )
        self.assertIn('maximum length', str(ctx.exception).lower())

    def test_points_at_max(self):
        """Test points at exactly MAX_POINTS."""
        xml = self.assignment_gen.generate(
            title="Test",
            instructions="<p>Test</p>",
            points=MAX_POINTS
        )
        self.assertIn(f'{MAX_POINTS:.9f}', xml)

    def test_points_exceeds_max(self):
        """Test points exceeding MAX_POINTS raises error."""
        with self.assertRaises(ValueError) as ctx:
            self.assignment_gen.generate(
                title="Test",
                instructions="<p>Test</p>",
                points=MAX_POINTS + 1
            )
        self.assertIn('maximum', str(ctx.exception).lower())

    def test_points_at_min(self):
        """Test points at exactly MIN_POINTS (0)."""
        xml = self.assignment_gen.generate(
            title="Test",
            instructions="<p>Test</p>",
            points=MIN_POINTS
        )
        self.assertIn(f'{MIN_POINTS:.9f}', xml)

    def test_points_below_min(self):
        """Test negative points raises error."""
        with self.assertRaises(ValueError) as ctx:
            self.assignment_gen.generate(
                title="Test",
                instructions="<p>Test</p>",
                points=-1.0
            )
        self.assertIn('non-negative', str(ctx.exception).lower())


class TestErrorPaths(unittest.TestCase):
    """Test error handling for invalid inputs."""

    def setUp(self):
        self.assignment_gen = AssignmentGenerator()
        self.discussion_gen = DiscussionGenerator()

    def test_empty_title_raises_error(self):
        """Test that empty title raises ValueError."""
        with self.assertRaises(ValueError):
            self.assignment_gen.generate(
                title="",
                instructions="<p>Test</p>",
                points=100.0
            )

    def test_whitespace_only_title_raises_error(self):
        """Test that whitespace-only title raises ValueError."""
        with self.assertRaises(ValueError):
            self.assignment_gen.generate(
                title="   ",
                instructions="<p>Test</p>",
                points=100.0
            )

    def test_path_traversal_in_attachment_raises_error(self):
        """Test that path traversal in attachments raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.assignment_gen.generate_with_attachments(
                title="Test Assignment",
                instructions="<p>Test</p>",
                points=100.0,
                attachments=["../../../etc/passwd"]
            )
        self.assertIn('..', str(ctx.exception))

    def test_absolute_path_in_attachment_raises_error(self):
        """Test that absolute paths in attachments raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.assignment_gen.generate_with_attachments(
                title="Test Assignment",
                instructions="<p>Test</p>",
                points=100.0,
                attachments=["/etc/passwd"]
            )
        self.assertIn('absolute', str(ctx.exception).lower())

    def test_discussion_path_traversal_raises_error(self):
        """Test that path traversal in discussion attachments raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            self.discussion_gen.generate(
                title="Test Discussion",
                prompt="<p>Test</p>",
                attachments=["../../secret.txt"]
            )
        self.assertIn('..', str(ctx.exception))


if __name__ == '__main__':
    unittest.main()
