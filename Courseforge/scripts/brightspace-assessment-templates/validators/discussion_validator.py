"""
Discussion Topic XML Validator

Validates IMSCC discussion topic XML files for correct format and Brightspace compatibility.
"""

from typing import List, Optional
from .xml_validator import IMSCCValidator, ValidationResult, ValidationLevel

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    import xml.etree.ElementTree as etree


class DiscussionValidator(IMSCCValidator):
    """
    Validator for IMSCC discussion topic XML files.

    Validates against the correct namespace:
    http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3

    IMPORTANT: Root element must be <topic>, NOT <discussion>
    """

    # Correct namespace for discussions
    DISCUSSION_NAMESPACE = 'http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3'

    # Deprecated/incorrect namespace (should NOT be used)
    DEPRECATED_NAMESPACE = 'http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0'

    # Required elements for a valid discussion
    REQUIRED_ELEMENTS = ['title', 'text']

    # Correct root element
    CORRECT_ROOT = 'topic'

    # Incorrect root element that is commonly used
    INCORRECT_ROOT = 'discussion'

    def validate(self, xml_content: str) -> ValidationResult:
        """
        Perform comprehensive validation of discussion topic XML.

        Args:
            xml_content: Discussion XML string

        Returns:
            ValidationResult with all validation findings
        """
        result = ValidationResult(is_valid=True)

        # Check for incorrect root element
        root_check = self._check_incorrect_root(xml_content)
        result.merge(root_check)

        # Check for deprecated namespace
        deprecated_result = self._check_deprecated_namespace(xml_content)
        result.merge(deprecated_result)

        # Validate namespace
        ns_result = self.validate_namespace(xml_content, 'discussion')
        result.merge(ns_result)

        # Validate root element
        root_result = self.validate_root_element(xml_content, self.CORRECT_ROOT)
        result.merge(root_result)

        # Validate required elements
        req_result = self.validate_required_elements(
            xml_content,
            self.REQUIRED_ELEMENTS,
            self.DISCUSSION_NAMESPACE
        )
        result.merge(req_result)

        # Validate structure
        struct_result = self._validate_structure(xml_content)
        result.merge(struct_result)

        # Schema validation
        schema_result = self.validate_xml_string(xml_content, 'discussion')
        result.merge(schema_result)

        return result

    def _check_incorrect_root(self, xml_content: str) -> ValidationResult:
        """Check if XML uses incorrect <discussion> root element."""
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
                local_name = etree.QName(doc.tag).localname
            else:
                doc = etree.fromstring(xml_content)
                local_name = doc.tag.split('}')[-1] if '}' in doc.tag else doc.tag

            if local_name == self.INCORRECT_ROOT:
                result.add_error(
                    f"Discussion uses INCORRECT root element '<{self.INCORRECT_ROOT}>'. "
                    f"Must use '<{self.CORRECT_ROOT}>' instead.",
                    ValidationLevel.CRITICAL
                )
        except Exception as e:
            result.add_error(f"Failed to parse XML: {str(e)}", ValidationLevel.CRITICAL)

        return result

    def _check_deprecated_namespace(self, xml_content: str) -> ValidationResult:
        """Check if XML uses deprecated d2l_2p0 namespace."""
        result = ValidationResult(is_valid=True)

        if self.DEPRECATED_NAMESPACE in xml_content:
            result.add_error(
                f"Discussion uses DEPRECATED namespace '{self.DEPRECATED_NAMESPACE}'. "
                f"Use '{self.DISCUSSION_NAMESPACE}' instead.",
                ValidationLevel.CRITICAL
            )

        return result

    def _validate_structure(self, xml_content: str) -> ValidationResult:
        """Validate discussion-specific structure requirements."""
        result = ValidationResult(is_valid=True)

        if not LXML_AVAILABLE:
            result.add_warning("lxml not available - detailed structure validation skipped")
            return result

        try:
            doc = etree.fromstring(xml_content.encode('utf-8'))
            ns = {'d': self.DISCUSSION_NAMESPACE}

            # Check title is not empty
            title = doc.find('d:title', ns)
            if title is not None and not title.text:
                result.add_error("Discussion title is empty", ValidationLevel.HIGH)

            # Check text element has texttype attribute
            text = doc.find('d:text', ns)
            if text is not None:
                texttype = text.get('texttype')
                if texttype not in ['text/plain', 'text/html']:
                    result.add_warning(
                        f"text texttype should be 'text/plain' or 'text/html', got '{texttype}'"
                    )

                # Check text is not empty
                if not text.text:
                    result.add_warning("Discussion text/prompt is empty")

        except Exception as e:
            result.add_error(f"Failed to validate structure: {str(e)}", ValidationLevel.HIGH)

        return result

    def get_resource_type(self) -> str:
        """Return the correct manifest resource type for discussions."""
        return "imsdt_xmlv1p3"


def validate_discussion(xml_content: str) -> ValidationResult:
    """
    Convenience function to validate discussion topic XML.

    Args:
        xml_content: Discussion XML string

    Returns:
        ValidationResult with all validation findings
    """
    validator = DiscussionValidator()
    return validator.validate(xml_content)
