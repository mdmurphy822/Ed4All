"""
Assignment XML Validator

Validates IMSCC assignment XML files for correct format and Brightspace compatibility.
"""

from typing import List, Optional
from .xml_validator import IMSCCValidator, ValidationResult, ValidationLevel

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    import xml.etree.ElementTree as etree


class AssignmentValidator(IMSCCValidator):
    """
    Validator for IMSCC assignment XML files.

    Validates against the correct namespace:
    http://www.imsglobal.org/xsd/imscc_extensions/assignment

    NOT the deprecated d2l_2p0 namespace.
    """

    # Correct namespace for assignments
    ASSIGNMENT_NAMESPACE = 'http://www.imsglobal.org/xsd/imscc_extensions/assignment'

    # Deprecated/incorrect namespace (should NOT be used)
    DEPRECATED_NAMESPACE = 'http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0'

    # Required elements for a valid assignment
    REQUIRED_ELEMENTS = ['title', 'instructor_text']

    # Valid submission format types
    VALID_SUBMISSION_TYPES = ['text', 'html', 'url', 'file']

    def validate(self, xml_content: str) -> ValidationResult:
        """
        Perform comprehensive validation of assignment XML.

        Args:
            xml_content: Assignment XML string

        Returns:
            ValidationResult with all validation findings
        """
        result = ValidationResult(is_valid=True)

        # Check for deprecated namespace
        deprecated_result = self._check_deprecated_namespace(xml_content)
        result.merge(deprecated_result)

        # Validate namespace
        ns_result = self.validate_namespace(xml_content, 'assignment')
        result.merge(ns_result)

        # Validate root element
        root_result = self.validate_root_element(xml_content, 'assignment')
        result.merge(root_result)

        # Validate required elements
        req_result = self.validate_required_elements(
            xml_content,
            self.REQUIRED_ELEMENTS,
            self.ASSIGNMENT_NAMESPACE
        )
        result.merge(req_result)

        # Validate structure
        struct_result = self._validate_structure(xml_content)
        result.merge(struct_result)

        # Schema validation
        schema_result = self.validate_xml_string(xml_content, 'assignment')
        result.merge(schema_result)

        return result

    def _check_deprecated_namespace(self, xml_content: str) -> ValidationResult:
        """Check if XML uses deprecated d2l_2p0 namespace."""
        result = ValidationResult(is_valid=True)

        if self.DEPRECATED_NAMESPACE in xml_content:
            result.add_error(
                f"Assignment uses DEPRECATED namespace '{self.DEPRECATED_NAMESPACE}'. "
                f"Use '{self.ASSIGNMENT_NAMESPACE}' instead.",
                ValidationLevel.CRITICAL
            )

        return result

    def _validate_structure(self, xml_content: str) -> ValidationResult:
        """Validate assignment-specific structure requirements."""
        result = ValidationResult(is_valid=True)

        if not LXML_AVAILABLE:
            result.add_warning("lxml not available - detailed structure validation skipped")
            return result

        try:
            doc = etree.fromstring(xml_content.encode('utf-8'))
            ns = {'a': self.ASSIGNMENT_NAMESPACE}

            # Check title is not empty
            title = doc.find('a:title', ns)
            if title is not None and not title.text:
                result.add_error("Assignment title is empty", ValidationLevel.HIGH)

            # Check instructor_text has texttype attribute
            instructor_text = doc.find('a:instructor_text', ns)
            if instructor_text is not None:
                texttype = instructor_text.get('texttype')
                if texttype not in ['text/plain', 'text/html']:
                    result.add_warning(
                        f"instructor_text texttype should be 'text/plain' or 'text/html', got '{texttype}'"
                    )

            # Check submission_formats if present
            sub_formats = doc.find('a:submission_formats', ns)
            if sub_formats is not None:
                formats = sub_formats.findall('a:format', ns)
                if not formats:
                    result.add_warning("submission_formats element is empty")
                else:
                    for fmt in formats:
                        fmt_type = fmt.get('type')
                        if fmt_type not in self.VALID_SUBMISSION_TYPES:
                            result.add_error(
                                f"Invalid submission format type: '{fmt_type}'. "
                                f"Valid types: {self.VALID_SUBMISSION_TYPES}",
                                ValidationLevel.MEDIUM
                            )

            # Check gradable if present
            gradable = doc.find('a:gradable', ns)
            if gradable is not None:
                points = gradable.get('points_possible')
                if points:
                    try:
                        float(points)
                    except ValueError:
                        result.add_error(
                            f"points_possible must be a number, got '{points}'",
                            ValidationLevel.HIGH
                        )

            # Check identifier attribute - must exist and not be empty
            identifier = doc.get('identifier')
            if not identifier:
                result.add_error("Assignment missing 'identifier' attribute", ValidationLevel.HIGH)
            elif not identifier.strip():
                result.add_error("Assignment has empty 'identifier' attribute", ValidationLevel.HIGH)
            elif not identifier.startswith('i'):
                result.add_warning(
                    "Brightspace identifiers typically start with 'i' prefix"
                )

        except Exception as e:
            result.add_error(f"Failed to validate structure: {str(e)}", ValidationLevel.HIGH)

        return result

    def get_resource_type(self) -> str:
        """Return the correct manifest resource type for assignments."""
        return "assignment_xmlv1p0"


def validate_assignment(xml_content: str) -> ValidationResult:
    """
    Convenience function to validate assignment XML.

    Args:
        xml_content: Assignment XML string

    Returns:
        ValidationResult with all validation findings
    """
    validator = AssignmentValidator()
    return validator.validate(xml_content)
