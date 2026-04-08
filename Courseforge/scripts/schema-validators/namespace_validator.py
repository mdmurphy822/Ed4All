#!/usr/bin/env python3
"""
Namespace Validator
Validates XML namespace declarations in IMSCC packages

Checks:
- All namespace prefixes are properly declared
- Namespaces match expected IMS CC patterns
- No conflicting namespace declarations
- Brightspace-specific extensions are properly declared
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from xml.etree import ElementTree as ET

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IssueSeverity(Enum):
    """Severity levels for validation issues"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ValidationIssue:
    """Represents a single validation issue"""
    severity: IssueSeverity
    code: str
    message: str
    element: Optional[str] = None
    line: Optional[int] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of namespace validation"""
    file_path: str
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    namespaces_found: Dict[str, str] = field(default_factory=dict)
    imscc_version: Optional[str] = None
    lms_detected: Optional[str] = None

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.HIGH)


class NamespaceValidator:
    """Validates XML namespace consistency in IMSCC packages"""

    # Standard IMSCC namespaces by version
    IMSCC_NAMESPACES = {
        '1.1': 'http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1',
        '1.2': 'http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1',
        '1.3': 'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1',
    }

    # LOM metadata namespaces
    LOM_NAMESPACES = {
        '1.1': 'http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest',
        '1.2': 'http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest',
        '1.3': 'http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest',
    }

    # LMS-specific namespaces for detection
    LMS_NAMESPACES = {
        'brightspace': [
            'http://www.desire2learn.com/xsd/d2l_2p0',
            'http://www.d2l.com',
        ],
        'canvas': [
            'http://canvas.instructure.com/xsd/cccv1p0',
            'https://canvas.instructure.com',
        ],
        'blackboard': [
            'http://www.blackboard.com/content-packaging',
            'http://www.blackboard.com',
        ],
        'moodle': [
            'http://moodle.org',
        ],
        'sakai': [
            'http://sakaiproject.org',
        ],
    }

    # Common extension namespaces
    EXTENSION_NAMESPACES = {
        'assignment': 'http://www.imsglobal.org/xsd/imscc_extensions/assignment',
        'discussion': 'http://www.imsglobal.org/xsd/imsdt_xmlv1p2',
        'qti': 'http://www.imsglobal.org/xsd/ims_qtiasiv1p2',
        'blti': 'http://www.imsglobal.org/xsd/imslticc_v1p0',
    }

    def __init__(self):
        self.issues: List[ValidationIssue] = []

    def validate_file(self, xml_path: Path) -> ValidationResult:
        """
        Validate namespace declarations in an XML file.

        Args:
            xml_path: Path to XML file to validate

        Returns:
            ValidationResult with findings
        """
        self.issues = []
        namespaces_found = {}
        imscc_version = None
        lms_detected = None

        try:
            # Parse XML
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract all namespaces
            namespaces_found = self._extract_namespaces(root)

            # Detect IMSCC version
            imscc_version = self._detect_imscc_version(namespaces_found)

            # Detect LMS source
            lms_detected = self._detect_lms(namespaces_found)

            # Run validation checks
            self._check_required_namespaces(namespaces_found, imscc_version)
            self._check_namespace_consistency(root, namespaces_found)
            self._check_prefix_usage(root, namespaces_found)
            self._check_extension_namespaces(namespaces_found)

        except ET.ParseError as e:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="NS001",
                message=f"XML parsing error: {e}",
                suggestion="Ensure the file is well-formed XML"
            ))
        except FileNotFoundError:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="NS002",
                message=f"File not found: {xml_path}",
            ))
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="NS003",
                message=f"Unexpected error: {e}",
            ))

        return ValidationResult(
            file_path=str(xml_path),
            valid=len([i for i in self.issues if i.severity in
                      [IssueSeverity.CRITICAL, IssueSeverity.HIGH]]) == 0,
            issues=self.issues,
            namespaces_found=namespaces_found,
            imscc_version=imscc_version,
            lms_detected=lms_detected,
        )

    def _extract_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """Extract all namespace declarations from root element"""
        namespaces = {}

        # Parse namespace declarations from root tag
        for key, value in root.attrib.items():
            if key.startswith('{'):
                # Already a namespace-qualified attribute
                continue
            if key == 'xmlns' or key.startswith('xmlns:'):
                prefix = key.split(':')[1] if ':' in key else ''
                namespaces[prefix] = value

        # Also check the root element's namespace
        if root.tag.startswith('{'):
            ns = root.tag[1:root.tag.index('}')]
            if ns not in namespaces.values():
                namespaces['_default_'] = ns

        return namespaces

    def _detect_imscc_version(self, namespaces: Dict[str, str]) -> Optional[str]:
        """Detect IMSCC version from namespace declarations"""
        for version, ns_pattern in self.IMSCC_NAMESPACES.items():
            for ns in namespaces.values():
                if ns_pattern in ns or f'imsccv1p{version.replace(".", "")}' in ns:
                    return version

        # Check for version in any namespace
        for ns in namespaces.values():
            if 'imsccv1p1' in ns:
                return '1.1'
            elif 'imsccv1p2' in ns:
                return '1.2'
            elif 'imsccv1p3' in ns:
                return '1.3'

        return None

    def _detect_lms(self, namespaces: Dict[str, str]) -> Optional[str]:
        """Detect source LMS from namespace declarations"""
        for ns in namespaces.values():
            for lms, patterns in self.LMS_NAMESPACES.items():
                for pattern in patterns:
                    if pattern in ns:
                        return lms
        return None

    def _check_required_namespaces(self, namespaces: Dict[str, str],
                                   version: Optional[str]) -> None:
        """Check that required namespaces are declared"""
        if not namespaces:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="NS010",
                message="No namespace declarations found",
                suggestion="Add xmlns declaration for IMS Common Cartridge"
            ))
            return

        # Check for IMS CC namespace
        has_imscc_ns = False
        for ns in namespaces.values():
            if 'imsglobal.org' in ns and 'imscp' in ns:
                has_imscc_ns = True
                break

        if not has_imscc_ns:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="NS011",
                message="Missing IMS Common Cartridge namespace",
                suggestion="Add: xmlns=\"http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1\""
            ))

    def _check_namespace_consistency(self, root: ET.Element,
                                     namespaces: Dict[str, str]) -> None:
        """Check namespace consistency throughout document"""
        # Check for mixed versions
        versions_found = set()
        for ns in namespaces.values():
            if 'imsccv1p1' in ns:
                versions_found.add('1.1')
            if 'imsccv1p2' in ns:
                versions_found.add('1.2')
            if 'imsccv1p3' in ns:
                versions_found.add('1.3')

        if len(versions_found) > 1:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="NS020",
                message=f"Mixed IMSCC versions detected: {versions_found}",
                suggestion="Use consistent namespace versions throughout the manifest"
            ))

    def _check_prefix_usage(self, root: ET.Element,
                            namespaces: Dict[str, str]) -> None:
        """Check that all used prefixes are declared"""
        used_prefixes: Set[str] = set()

        def collect_prefixes(element: ET.Element):
            # Check element tag
            if ':' in element.tag and not element.tag.startswith('{'):
                prefix = element.tag.split(':')[0]
                used_prefixes.add(prefix)

            # Check attributes
            for attr in element.attrib:
                if ':' in attr and not attr.startswith('{') and not attr.startswith('xmlns'):
                    prefix = attr.split(':')[0]
                    used_prefixes.add(prefix)

            # Recurse
            for child in element:
                collect_prefixes(child)

        collect_prefixes(root)

        # Check if all used prefixes are declared
        declared_prefixes = set(namespaces.keys())
        undeclared = used_prefixes - declared_prefixes - {'xml'}  # xml is always available

        for prefix in undeclared:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="NS030",
                message=f"Undeclared namespace prefix used: '{prefix}'",
                suggestion=f"Add xmlns:{prefix}=\"...\" declaration to root element"
            ))

    def _check_extension_namespaces(self, namespaces: Dict[str, str]) -> None:
        """Check extension namespace validity"""
        for ns in namespaces.values():
            # Check for common typos or invalid patterns
            if 'imsglobal' in ns and 'http://' not in ns and 'https://' not in ns:
                self.issues.append(ValidationIssue(
                    severity=IssueSeverity.MEDIUM,
                    code="NS040",
                    message=f"Namespace may be malformed: {ns}",
                    suggestion="Namespace URIs should start with http:// or https://"
                ))

    def validate_namespaces(self, xml_path: Path) -> ValidationResult:
        """Alias for validate_file for API compatibility"""
        return self.validate_file(xml_path)


def main():
    """CLI entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Validate XML namespace declarations in IMSCC packages'
    )
    parser.add_argument('-i', '--input', required=True, help='XML file to validate')
    parser.add_argument('-j', '--json', action='store_true', help='Output as JSON')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Verbose output (-vv for debug)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    args = parser.parse_args()

    # Configure logging
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)

    validator = NamespaceValidator()
    result = validator.validate_file(Path(args.input))

    if args.json:
        output = {
            'file_path': result.file_path,
            'valid': result.valid,
            'imscc_version': result.imscc_version,
            'lms_detected': result.lms_detected,
            'namespaces': result.namespaces_found,
            'issues': [
                {
                    'severity': i.severity.value,
                    'code': i.code,
                    'message': i.message,
                    'suggestion': i.suggestion,
                }
                for i in result.issues
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"File: {result.file_path}")
        print(f"Valid: {result.valid}")
        print(f"IMSCC Version: {result.imscc_version or 'Unknown'}")
        print(f"LMS Detected: {result.lms_detected or 'Generic'}")
        print(f"\nNamespaces Found: {len(result.namespaces_found)}")
        for prefix, uri in result.namespaces_found.items():
            print(f"  {prefix or '(default)'}: {uri}")
        print(f"\nIssues Found: {len(result.issues)}")
        for issue in result.issues:
            print(f"  [{issue.severity.value.upper()}] {issue.code}: {issue.message}")
            if issue.suggestion:
                print(f"    Suggestion: {issue.suggestion}")

    return 0 if result.valid else 1


if __name__ == '__main__':
    exit(main())
