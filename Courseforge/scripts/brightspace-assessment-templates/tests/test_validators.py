#!/usr/bin/env python3
"""
Unit tests for Brightspace assessment validators.

Tests all validator classes for correct namespace detection, XML validation,
and error handling.
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validators import (
    AssignmentValidator,
    DiscussionValidator,
    QTIValidator,
    ManifestValidator,
    ValidationResult,
)


class TestAssignmentValidator(unittest.TestCase):
    """Test assignment XML validation."""

    def setUp(self):
        self.validator = AssignmentValidator()

    def test_valid_assignment(self):
        """Test validation of correctly formatted assignment."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="http://www.imsglobal.org/xsd/imscc_extensions/assignment http://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd"
            identifier="test_assignment">
  <title>Test Assignment</title>
  <instructor_text texttype="text/html">&lt;p&gt;Instructions&lt;/p&gt;</instructor_text>
  <submission_formats>
    <format type="file" />
  </submission_formats>
  <gradable points_possible="100.000000000">true</gradable>
</assignment>'''
        result = self.validator.validate(xml)
        # Check that no CRITICAL namespace errors occurred
        critical_errors = [e for e in result.errors if 'namespace' in e.lower() or 'CRITICAL' in e]
        self.assertEqual(len(critical_errors), 0, f"Critical errors: {critical_errors}")

    def test_invalid_namespace(self):
        """Test detection of deprecated d2l_2p0 namespace."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<assignment xmlns="http://www.desire2learn.com/xsd/d2l_2p0"
            identifier="test_assignment">
  <title>Test Assignment</title>
</assignment>'''
        result = self.validator.validate(xml)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("deprecated" in e.lower() or "namespace" in e.lower()
                          for e in result.errors))

    def test_missing_title(self):
        """Test detection of missing title element."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment">
  <instructor_text texttype="text/html">Instructions</instructor_text>
  <gradable points_possible="100">true</gradable>
</assignment>'''
        result = self.validator.validate(xml)
        self.assertFalse(result.is_valid)

    def test_missing_gradable(self):
        """Test that assignments without gradable element are acceptable.

        Note: gradable is optional in IMSCC assignments, so this should pass.
        """
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment">
  <title>Test Assignment</title>
  <instructor_text texttype="text/html">Instructions</instructor_text>
</assignment>'''
        result = self.validator.validate(xml)
        # gradable is optional, so this should not cause critical errors
        critical_namespace_errors = [e for e in result.errors if 'namespace' in e.lower()]
        self.assertEqual(len(critical_namespace_errors), 0)

    def test_malformed_xml(self):
        """Test handling of malformed XML."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment">
  <title>Test
</assignment>'''
        result = self.validator.validate(xml)
        self.assertFalse(result.is_valid)


class TestDiscussionValidator(unittest.TestCase):
    """Test discussion XML validation."""

    def setUp(self):
        self.validator = DiscussionValidator()

    def test_valid_discussion(self):
        """Test validation of correctly formatted discussion."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <title>Test Discussion</title>
  <text texttype="text/html">&lt;p&gt;Discussion prompt&lt;/p&gt;</text>
</topic>'''
        result = self.validator.validate(xml)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_wrong_root_element(self):
        """Test detection of <discussion> instead of <topic>."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<discussion xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3">
  <title>Test Discussion</title>
  <text texttype="text/html">Prompt</text>
</discussion>'''
        result = self.validator.validate(xml)
        self.assertFalse(result.is_valid)
        self.assertTrue(any("topic" in e.lower() for e in result.errors))

    def test_missing_title(self):
        """Test detection of missing title element."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3">
  <text texttype="text/html">Prompt</text>
</topic>'''
        result = self.validator.validate(xml)
        self.assertFalse(result.is_valid)

    def test_missing_text(self):
        """Test detection of missing text element."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3">
  <title>Test Discussion</title>
</topic>'''
        result = self.validator.validate(xml)
        self.assertFalse(result.is_valid)


class TestQTIValidator(unittest.TestCase):
    """Test QTI quiz XML validation."""

    def setUp(self):
        self.validator = QTIValidator()

    def test_valid_quiz(self):
        """Test validation of correctly formatted quiz."""
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
  <assessment ident="test_quiz" title="Test Quiz">
    <qtimetadata>
      <qtimetadatafield>
        <fieldlabel>cc_profile</fieldlabel>
        <fieldentry>cc.exam.v0p1</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="root_section">
      <item ident="q1" title="Question 1">
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.multiple_choice.v0p1</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material>
            <mattext texttype="text/html">Question text</mattext>
          </material>
          <response_lid ident="response" rcardinality="Single">
            <render_choice>
              <response_label ident="a1">
                <material><mattext>Answer</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar maxvalue="1" minvalue="0" varname="SCORE" vartype="Decimal"/>
          </outcomes>
        </resprocessing>
      </item>
    </section>
  </assessment>
</questestinterop>'''
        result = self.validator.validate(xml)
        self.assertTrue(result.is_valid, f"Errors: {result.errors}")

    def test_missing_assessment(self):
        """Test detection of missing assessment element."""
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
</questestinterop>'''
        result = self.validator.validate(xml)
        self.assertFalse(result.is_valid)

    def test_missing_section(self):
        """Test detection of missing section element."""
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2">
  <assessment ident="test" title="Test">
  </assessment>
</questestinterop>'''
        result = self.validator.validate(xml)
        self.assertFalse(result.is_valid)

    def test_question_type_validation(self):
        """Test that question types are validated."""
        # Valid question types should be recognized
        valid_types = [
            "cc.multiple_choice.v0p1",
            "cc.multiple_response.v0p1",
            "cc.true_false.v0p1",
            "cc.fib.v0p1",
            "cc.essay.v0p1",
        ]
        for qtype in valid_types:
            self.assertIn(qtype, self.validator.VALID_QUESTION_PROFILES)


class TestManifestValidator(unittest.TestCase):
    """Test manifest XML validation."""

    def setUp(self):
        self.validator = ManifestValidator()

    def test_valid_manifest(self):
        """Test validation of correctly formatted manifest."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<manifest identifier="test_manifest"
          xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.3.0</schemaversion>
  </metadata>
  <organizations>
    <organization identifier="org1" structure="rooted-hierarchy">
      <item identifier="root">
        <title>Root</title>
      </item>
    </organization>
  </organizations>
  <resources>
    <resource identifier="res1" type="webcontent" href="content.html">
      <file href="content.html" />
    </resource>
  </resources>
</manifest>'''
        result = self.validator.validate(xml)
        self.assertTrue(result.is_valid, f"Errors: {result.errors}")

    def test_wrong_namespace(self):
        """Test detection of incorrect namespace version."""
        xml = '''<?xml version="1.0" encoding="utf-8"?>
<manifest identifier="test_manifest"
          xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.2.0</schemaversion>
  </metadata>
  <organizations/>
  <resources/>
</manifest>'''
        result = self.validator.validate(xml)
        # Should warn about older version
        self.assertTrue(len(result.warnings) > 0 or not result.is_valid)

    def test_valid_resource_types(self):
        """Verify resource type validation."""
        valid_types = self.validator.VALID_RESOURCE_TYPES
        self.assertIn("assignment_xmlv1p0", valid_types)
        self.assertIn("imsdt_xmlv1p3", valid_types)
        self.assertIn("imsqti_xmlv1p2/imscc_xmlv1p3/assessment", valid_types)


class TestValidationResult(unittest.TestCase):
    """Test ValidationResult class."""

    def test_valid_result(self):
        """Test creation of valid result."""
        result = ValidationResult(is_valid=True)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(len(result.warnings), 0)

    def test_invalid_result(self):
        """Test creation of invalid result."""
        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"]
        )
        self.assertFalse(result.is_valid)
        self.assertEqual(len(result.errors), 2)

    def test_result_with_warnings(self):
        """Test result with warnings."""
        result = ValidationResult(
            is_valid=True,
            warnings=["Warning 1"]
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.warnings), 1)


if __name__ == '__main__':
    unittest.main()
