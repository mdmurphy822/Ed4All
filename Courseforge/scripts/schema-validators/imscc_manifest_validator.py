#!/usr/bin/env python3
"""
IMSCC Manifest Validator
Validates manifest XML against IMS Common Cartridge specifications

Checks:
- XML well-formedness
- Required namespace declarations
- Required elements (manifest, organizations, resources)
- Resource identifier uniqueness
- Organization hierarchy structure
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set
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
    wcag_criterion: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of manifest validation"""
    file_path: str
    valid: bool
    imscc_version: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    resource_count: int = 0
    organization_count: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def compliant(self) -> bool:
        return self.critical_count == 0


class IMSCCManifestValidator:
    """Validates IMSCC manifest against IMS CC specifications"""

    SUPPORTED_VERSIONS = ['1.1.0', '1.2.0', '1.3.0']

    REQUIRED_NAMESPACES = {
        '1.1.0': 'http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1',
        '1.2.0': 'http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1',
        '1.3.0': 'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1',
    }

    VALID_RESOURCE_TYPES = [
        'webcontent',
        'associatedcontent/imscc_xmlv1p1/learning-application-resource',
        'associatedcontent/imscc_xmlv1p2/learning-application-resource',
        'associatedcontent/imscc_xmlv1p3/learning-application-resource',
        'imsqti_xmlv1p2/imscc_xmlv1p1/assessment',
        'imsqti_xmlv1p2/imscc_xmlv1p2/assessment',
        'imsqti_xmlv1p2/imscc_xmlv1p3/assessment',
        'imswl_xmlv1p2',
        'imsdt_xmlv1p2',
        'imsbasiclti_xmlv1p0',
    ]

    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.resource_ids: Set[str] = set()

    def validate_manifest(self, manifest_path: Path) -> ValidationResult:
        """
        Validate an IMSCC manifest file.

        Args:
            manifest_path: Path to imsmanifest.xml

        Returns:
            ValidationResult with findings
        """
        self.issues = []
        self.resource_ids = set()
        resource_count = 0
        organization_count = 0
        imscc_version = None

        if not manifest_path.exists():
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="MF001",
                message=f"Manifest file not found: {manifest_path}",
            ))
            return ValidationResult(
                file_path=str(manifest_path),
                valid=False,
                issues=self.issues,
            )

        try:
            # Parse XML
            tree = ET.parse(manifest_path)
            root = tree.getroot()

            # Validate root element
            self._validate_root_element(root)

            # Detect and validate version
            imscc_version = self._detect_version(root)

            # Validate required sections
            self._validate_metadata(root)
            organization_count = self._validate_organizations(root)
            resource_count = self._validate_resources(root)

            # Validate resource types
            self._validate_resource_types(root)

            # Validate identifier uniqueness
            self._validate_identifier_uniqueness(root)

            # Validate organization-resource references
            self._validate_references(root)

        except ET.ParseError as e:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="MF002",
                message=f"XML parsing error: {e}",
                suggestion="Ensure the manifest is well-formed XML"
            ))
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="MF003",
                message=f"Unexpected error: {e}",
            ))

        return ValidationResult(
            file_path=str(manifest_path),
            valid=self._calculate_validity(),
            imscc_version=imscc_version,
            issues=self.issues,
            resource_count=resource_count,
            organization_count=organization_count,
        )

    def _calculate_validity(self) -> bool:
        """Determine if manifest is valid based on issues"""
        critical_high = sum(1 for i in self.issues
                          if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH])
        return critical_high == 0

    def _validate_root_element(self, root: ET.Element) -> None:
        """Validate the root manifest element"""
        # Check root tag
        if not (root.tag.endswith('manifest') or root.tag == 'manifest'):
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="MF010",
                message=f"Root element must be 'manifest', found: {root.tag}",
            ))

        # Check for identifier attribute
        if not root.get('identifier'):
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="MF011",
                message="Manifest element missing 'identifier' attribute",
                suggestion="Add a unique identifier to the manifest element"
            ))

    def _detect_version(self, root: ET.Element) -> Optional[str]:
        """Detect IMSCC version from namespace or schemaversion"""
        # Check namespace
        tag = root.tag
        if '{' in tag:
            ns = tag[1:tag.index('}')]
            for version, expected_ns in self.REQUIRED_NAMESPACES.items():
                if expected_ns in ns:
                    return version

        # Check schemaversion in metadata
        for elem in root.iter():
            if elem.tag.endswith('schemaversion') or elem.tag == 'schemaversion':
                if elem.text:
                    version = elem.text.strip()
                    if version in self.SUPPORTED_VERSIONS:
                        return version
                    # Try to extract version
                    if '1.1' in version:
                        return '1.1.0'
                    elif '1.2' in version:
                        return '1.2.0'
                    elif '1.3' in version:
                        return '1.3.0'

        self.issues.append(ValidationIssue(
            severity=IssueSeverity.MEDIUM,
            code="MF020",
            message="Could not determine IMSCC version",
            suggestion="Ensure schemaversion is specified in metadata"
        ))
        return None

    def _validate_metadata(self, root: ET.Element) -> None:
        """Validate metadata section"""
        metadata = None
        for elem in root:
            if elem.tag.endswith('metadata') or elem.tag == 'metadata':
                metadata = elem
                break

        if metadata is None:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="MF030",
                message="Missing metadata section",
                suggestion="Add metadata section with schema and schemaversion"
            ))
            return

        # Check for schema element
        has_schema = False
        for elem in metadata.iter():
            if elem.tag.endswith('schema') or elem.tag == 'schema':
                has_schema = True
                if elem.text and 'IMS Common Cartridge' not in elem.text:
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.MEDIUM,
                        code="MF031",
                        message=f"Unexpected schema value: {elem.text}",
                        suggestion="Schema should be 'IMS Common Cartridge'"
                    ))
                break

        if not has_schema:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.MEDIUM,
                code="MF032",
                message="Missing schema element in metadata",
            ))

    def _validate_organizations(self, root: ET.Element) -> int:
        """Validate organizations section"""
        organizations = None
        for elem in root:
            if elem.tag.endswith('organizations') or elem.tag == 'organizations':
                organizations = elem
                break

        if organizations is None:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="MF040",
                message="Missing organizations section",
                suggestion="Add organizations section with at least one organization"
            ))
            return 0

        # Count organizations
        org_count = 0
        for elem in organizations:
            if elem.tag.endswith('organization') or elem.tag == 'organization':
                org_count += 1

                # Validate organization has identifier
                if not elem.get('identifier'):
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.HIGH,
                        code="MF041",
                        message="Organization element missing identifier attribute",
                    ))

                # Check for items
                item_count = sum(1 for child in elem.iter()
                               if child.tag.endswith('item') or child.tag == 'item')
                if item_count == 0:
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.MEDIUM,
                        code="MF042",
                        message="Organization has no items",
                        suggestion="Add item elements to define course structure"
                    ))

        if org_count == 0:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="MF043",
                message="No organization elements found",
            ))

        return org_count

    def _validate_resources(self, root: ET.Element) -> int:
        """Validate resources section"""
        resources = None
        for elem in root:
            if elem.tag.endswith('resources') or elem.tag == 'resources':
                resources = elem
                break

        if resources is None:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="MF050",
                message="Missing resources section",
            ))
            return 0

        # Count and validate resources
        res_count = 0
        for elem in resources:
            if elem.tag.endswith('resource') or elem.tag == 'resource':
                res_count += 1
                res_id = elem.get('identifier')
                res_type = elem.get('type')
                res_href = elem.get('href')

                if res_id:
                    self.resource_ids.add(res_id)
                else:
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.HIGH,
                        code="MF051",
                        message="Resource element missing identifier attribute",
                    ))

                if not res_type:
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.HIGH,
                        code="MF052",
                        message=f"Resource '{res_id}' missing type attribute",
                    ))

        if res_count == 0:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="MF053",
                message="No resource elements found",
            ))

        return res_count

    def _validate_resource_types(self, root: ET.Element) -> None:
        """Validate resource type values"""
        for elem in root.iter():
            if elem.tag.endswith('resource') or elem.tag == 'resource':
                res_type = elem.get('type')
                res_id = elem.get('identifier', 'unknown')

                if res_type and not self._is_valid_resource_type(res_type):
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.MEDIUM,
                        code="MF060",
                        message=f"Non-standard resource type for '{res_id}': {res_type}",
                        element=res_id,
                        suggestion="Use standard IMS CC resource types"
                    ))

    def _is_valid_resource_type(self, res_type: str) -> bool:
        """Check if resource type is valid"""
        # Exact match
        if res_type in self.VALID_RESOURCE_TYPES:
            return True

        # Partial match for common types
        valid_prefixes = ['webcontent', 'imsqti', 'imswl', 'imsdt', 'imsbasiclti',
                         'associatedcontent']
        for prefix in valid_prefixes:
            if res_type.startswith(prefix):
                return True

        return False

    def _validate_identifier_uniqueness(self, root: ET.Element) -> None:
        """Validate that all identifiers are unique"""
        all_ids: Dict[str, int] = {}

        for elem in root.iter():
            identifier = elem.get('identifier')
            if identifier:
                all_ids[identifier] = all_ids.get(identifier, 0) + 1

        for id_val, count in all_ids.items():
            if count > 1:
                self.issues.append(ValidationIssue(
                    severity=IssueSeverity.HIGH,
                    code="MF070",
                    message=f"Duplicate identifier found: '{id_val}' (appears {count} times)",
                    element=id_val,
                    suggestion="Ensure all identifiers are unique"
                ))

    def _validate_references(self, root: ET.Element) -> None:
        """Validate organization item references to resources"""
        for elem in root.iter():
            if elem.tag.endswith('item') or elem.tag == 'item':
                ref = elem.get('identifierref')
                if ref and ref not in self.resource_ids:
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.HIGH,
                        code="MF080",
                        message=f"Item references non-existent resource: '{ref}'",
                        element=ref,
                        suggestion="Ensure identifierref matches a resource identifier"
                    ))


def main():
    """CLI entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Validate IMSCC manifest against IMS CC specifications'
    )
    parser.add_argument('-i', '--input', required=True, help='Path to imsmanifest.xml')
    parser.add_argument('-j', '--json', action='store_true', help='Output as JSON')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Verbose output (-vv for debug)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    args = parser.parse_args()

    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)

    validator = IMSCCManifestValidator()
    result = validator.validate_manifest(Path(args.input))

    if args.json:
        output = {
            'file_path': result.file_path,
            'valid': result.valid,
            'imscc_version': result.imscc_version,
            'resource_count': result.resource_count,
            'organization_count': result.organization_count,
            'issues': [
                {
                    'severity': i.severity.value,
                    'code': i.code,
                    'message': i.message,
                    'element': i.element,
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
        print(f"Resources: {result.resource_count}")
        print(f"Organizations: {result.organization_count}")
        print(f"\nIssues Found: {len(result.issues)}")
        for issue in result.issues:
            print(f"  [{issue.severity.value.upper()}] {issue.code}: {issue.message}")
            if issue.suggestion:
                print(f"    Suggestion: {issue.suggestion}")

    return 0 if result.valid else 1


if __name__ == '__main__':
    exit(main())
