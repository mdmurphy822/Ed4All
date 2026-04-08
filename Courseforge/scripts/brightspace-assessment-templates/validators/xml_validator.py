"""
Base XML Validator for IMSCC Components

Provides XSD schema validation and namespace verification for IMSCC XML files.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    import xml.etree.ElementTree as etree


class ValidationLevel(Enum):
    """Severity levels for validation issues."""
    CRITICAL = "CRITICAL"  # Will cause import failure
    HIGH = "HIGH"          # Functional failure
    MEDIUM = "MEDIUM"      # Quality issue
    LOW = "LOW"            # Minor formatting


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    level: ValidationLevel = ValidationLevel.HIGH

    def add_error(self, message: str, level: ValidationLevel = ValidationLevel.HIGH):
        """Add an error message."""
        self.errors.append(f"[{level.value}] {message}")
        self.is_valid = False
        if level.value in ['CRITICAL', 'HIGH']:
            self.level = level

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


class IMSCCValidator:
    """
    Base validator for IMSCC XML content.

    Provides schema validation and namespace verification for all
    IMSCC component types (assignments, discussions, quizzes, manifests).
    """

    # Schema directory relative to this file
    SCHEMA_DIR = Path(__file__).parent.parent.parent.parent / "schemas" / "imscc"

    # Known IMSCC namespaces
    NAMESPACES = {
        'assignment': 'http://www.imsglobal.org/xsd/imscc_extensions/assignment',
        'discussion': 'http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3',
        'qti': 'http://www.imsglobal.org/xsd/ims_qtiasiv1p2',
        'manifest_v1p1': 'http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1',
        'manifest_v1p2': 'http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1',
        'manifest_v1p3': 'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1',
        'lom_resource': 'http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource',
        'lom_manifest': 'http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest',
    }

    # Schema file mapping
    SCHEMA_FILES = {
        'assignment': 'cc_extresource_assignmentv1p0.xsd',
        'discussion': 'ccv1p3_imsdt_v1p3.xsd',
        'qti': 'ccv1p3_qtiasiv1p2p1.xsd',
    }

    def __init__(self):
        """Initialize validator and load schemas if lxml is available."""
        self.schemas = {}
        if LXML_AVAILABLE:
            self._load_schemas()

    def _load_schemas(self):
        """Load XSD schemas from schema directory."""
        for schema_type, filename in self.SCHEMA_FILES.items():
            schema_path = self.SCHEMA_DIR / filename
            if schema_path.exists():
                try:
                    with open(schema_path, 'rb') as f:
                        schema_doc = etree.parse(f)
                        self.schemas[schema_type] = etree.XMLSchema(schema_doc)
                except Exception as e:
                    print(f"Warning: Could not load schema {filename}: {e}")

    def validate_xml_string(self, xml_content: str, schema_type: str) -> ValidationResult:
        """
        Validate XML content against specified schema.

        Args:
            xml_content: XML string to validate
            schema_type: Type of schema ('assignment', 'discussion', 'qti')

        Returns:
            ValidationResult with validation status and any errors
        """
        result = ValidationResult(is_valid=True)

        # First check well-formedness
        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
            else:
                doc = etree.fromstring(xml_content)
        except Exception as e:
            result.add_error(f"XML syntax error: {str(e)}", ValidationLevel.CRITICAL)
            return result

        # Schema validation (only with lxml)
        if LXML_AVAILABLE and schema_type in self.schemas:
            schema = self.schemas[schema_type]
            if not schema.validate(doc):
                for error in schema.error_log:
                    result.add_error(f"Schema violation: {error.message}", ValidationLevel.HIGH)
        elif not LXML_AVAILABLE:
            result.add_warning("lxml not installed - schema validation skipped")

        return result

    def validate_namespace(self, xml_content: str, expected_type: str) -> ValidationResult:
        """
        Validate that XML uses the correct namespace for its type.

        Args:
            xml_content: XML string to validate
            expected_type: Expected content type ('assignment', 'discussion', 'qti')

        Returns:
            ValidationResult with namespace validation status
        """
        result = ValidationResult(is_valid=True)
        expected_ns = self.NAMESPACES.get(expected_type)

        if not expected_ns:
            result.add_error(f"Unknown content type: {expected_type}", ValidationLevel.HIGH)
            return result

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
                actual_ns = doc.nsmap.get(None, '')
            else:
                doc = etree.fromstring(xml_content)
                # Standard library doesn't preserve namespace map nicely
                # Extract from tag
                if doc.tag.startswith('{'):
                    actual_ns = doc.tag[1:doc.tag.index('}')]
                else:
                    actual_ns = ''

            if actual_ns != expected_ns:
                result.add_error(
                    f"Namespace mismatch: expected '{expected_ns}', got '{actual_ns}'",
                    ValidationLevel.CRITICAL
                )
        except Exception as e:
            result.add_error(f"Failed to parse XML: {str(e)}", ValidationLevel.CRITICAL)

        return result

    def validate_root_element(self, xml_content: str, expected_root: str) -> ValidationResult:
        """
        Validate that XML has the expected root element.

        Args:
            xml_content: XML string to validate
            expected_root: Expected root element name (without namespace)

        Returns:
            ValidationResult with root element validation status
        """
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
                # lxml includes namespace in tag
                local_name = etree.QName(doc.tag).localname
            else:
                doc = etree.fromstring(xml_content)
                # Standard library tag might include namespace prefix
                local_name = doc.tag.split('}')[-1] if '}' in doc.tag else doc.tag

            if local_name != expected_root:
                result.add_error(
                    f"Root element mismatch: expected '{expected_root}', got '{local_name}'",
                    ValidationLevel.CRITICAL
                )
        except Exception as e:
            result.add_error(f"Failed to parse XML: {str(e)}", ValidationLevel.CRITICAL)

        return result

    def validate_required_elements(self, xml_content: str, required: List[str],
                                   namespace: str = None) -> ValidationResult:
        """
        Validate that XML contains all required child elements.

        Args:
            xml_content: XML string to validate
            required: List of required element names
            namespace: Optional namespace URI

        Returns:
            ValidationResult with required elements validation status
        """
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
            else:
                doc = etree.fromstring(xml_content)

            for elem_name in required:
                if namespace and LXML_AVAILABLE:
                    found = doc.find(f'{{{namespace}}}{elem_name}')
                else:
                    # Try with and without namespace
                    found = doc.find(elem_name)
                    if found is None:
                        found = doc.find(f'.//{elem_name}')

                if found is None:
                    result.add_error(
                        f"Missing required element: {elem_name}",
                        ValidationLevel.HIGH
                    )
        except Exception as e:
            result.add_error(f"Failed to parse XML: {str(e)}", ValidationLevel.CRITICAL)

        return result

    def check_xml_escaping(self, xml_content: str) -> ValidationResult:
        """
        Check for common XML escaping issues in content and attributes.

        Args:
            xml_content: XML string to check

        Returns:
            ValidationResult with escaping issues found
        """
        result = ValidationResult(is_valid=True)

        import re

        # Common unescaped characters that cause issues in text content
        problematic_patterns = [
            ('&(?!(amp|lt|gt|quot|apos|#))', 'Unescaped ampersand (&) found'),
            ('<(?!/|[a-zA-Z!?])', 'Potentially unescaped less-than (<) found'),
        ]

        for pattern, message in problematic_patterns:
            if re.search(pattern, xml_content):
                result.add_warning(message)

        # Validate attribute values for unescaped quotes
        # Find attribute values and check for issues
        if LXML_AVAILABLE:
            try:
                doc = etree.fromstring(xml_content.encode('utf-8'))
                for elem in doc.iter():
                    for attr_name, attr_value in elem.attrib.items():
                        if attr_value:
                            # Note: If lxml parsed successfully, special chars were properly escaped.
                            # Check for empty or whitespace-only critical attributes
                            if attr_name in ('identifier', 'identifierref', 'ident', 'href'):
                                if not attr_value.strip():
                                    result.add_error(
                                        f"Element '{elem.tag}' has empty or whitespace-only '{attr_name}' attribute",
                                        ValidationLevel.HIGH
                                    )
            except Exception:
                pass  # Parsing errors handled elsewhere

        return result
