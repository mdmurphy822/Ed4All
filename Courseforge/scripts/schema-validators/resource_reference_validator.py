#!/usr/bin/env python3
"""
Resource Reference Validator
Validates that all resource references in IMSCC packages resolve correctly

Checks:
- All resource href attributes point to existing files
- All organization identifierref values exist in resources
- All file references use relative paths
- No broken internal links in HTML content
"""

import logging
import re
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
    resource_id: Optional[str] = None
    file_path: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of resource reference validation"""
    package_path: str
    valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    resources_checked: int = 0
    files_checked: int = 0
    broken_references: int = 0

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.CRITICAL)

    @property
    def high_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.HIGH)


class ResourceReferenceValidator:
    """Validates all resource references resolve in IMSCC packages"""

    # Common IMSCC namespaces
    NAMESPACES = {
        'imscp': 'http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1',
        'imscp11': 'http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1',
        'imscp13': 'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1',
    }

    def __init__(self):
        self.issues: List[ValidationIssue] = []
        self.resource_ids: Set[str] = set()
        self.resource_hrefs: Dict[str, str] = {}  # id -> href

    def validate_references(self, package_dir: Path) -> ValidationResult:
        """
        Validate all resource references in an IMSCC package.

        Args:
            package_dir: Path to extracted IMSCC package directory

        Returns:
            ValidationResult with findings
        """
        self.issues = []
        self.resource_ids = set()
        self.resource_hrefs = {}
        resources_checked = 0
        files_checked = 0

        manifest_path = package_dir / 'imsmanifest.xml'

        if not manifest_path.exists():
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="RR001",
                message="Manifest file not found: imsmanifest.xml",
                suggestion="Ensure the package contains imsmanifest.xml at root level"
            ))
            return ValidationResult(
                package_path=str(package_dir),
                valid=False,
                issues=self.issues,
            )

        try:
            tree = ET.parse(manifest_path)
            root = tree.getroot()

            # Detect namespace
            ns = self._detect_namespace(root)

            # Collect all resource identifiers and hrefs
            resources_checked = self._collect_resources(root, ns)

            # Validate organization references
            self._validate_organization_refs(root, ns)

            # Validate file references exist
            files_checked = self._validate_file_references(package_dir, root, ns)

            # Check for absolute paths
            self._check_path_format()

            # Validate internal HTML links
            self._validate_html_links(package_dir)

        except ET.ParseError as e:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="RR002",
                message=f"XML parsing error in manifest: {e}",
            ))
        except Exception as e:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.CRITICAL,
                code="RR003",
                message=f"Unexpected error: {e}",
            ))

        broken_count = sum(1 for i in self.issues
                         if i.severity in [IssueSeverity.CRITICAL, IssueSeverity.HIGH])

        return ValidationResult(
            package_path=str(package_dir),
            valid=broken_count == 0,
            issues=self.issues,
            resources_checked=resources_checked,
            files_checked=files_checked,
            broken_references=broken_count,
        )

    def _detect_namespace(self, root: ET.Element) -> str:
        """Detect the namespace used in the manifest"""
        tag = root.tag
        if tag.startswith('{'):
            return tag[1:tag.index('}')]
        return ''

    def _collect_resources(self, root: ET.Element, ns: str) -> int:
        """Collect all resource identifiers and their hrefs"""
        count = 0

        # Find resources element
        ns_prefix = f'{{{ns}}}' if ns else ''
        resources = root.find(f'.//{ns_prefix}resources')

        if resources is None:
            # Try without namespace
            resources = root.find('.//resources')

        if resources is None:
            self.issues.append(ValidationIssue(
                severity=IssueSeverity.HIGH,
                code="RR010",
                message="No resources section found in manifest",
            ))
            return 0

        for resource in resources.iter():
            if resource.tag.endswith('resource') or resource.tag == 'resource':
                res_id = resource.get('identifier')
                res_href = resource.get('href')

                if res_id:
                    self.resource_ids.add(res_id)
                    if res_href:
                        self.resource_hrefs[res_id] = res_href
                    count += 1

        logger.debug(f"Collected {count} resources")
        return count

    def _validate_organization_refs(self, root: ET.Element, ns: str) -> None:
        """Validate that organization item identifierrefs point to valid resources"""
        ns_prefix = f'{{{ns}}}' if ns else ''

        # Find all items with identifierref
        for elem in root.iter():
            if elem.tag.endswith('item') or elem.tag == 'item':
                ref = elem.get('identifierref')
                if ref and ref not in self.resource_ids:
                    self.issues.append(ValidationIssue(
                        severity=IssueSeverity.HIGH,
                        code="RR020",
                        message=f"Organization item references non-existent resource: {ref}",
                        resource_id=ref,
                        suggestion="Ensure the identifierref matches a resource identifier"
                    ))

    def _validate_file_references(self, package_dir: Path, root: ET.Element,
                                  ns: str) -> int:
        """Validate that all file hrefs point to existing files"""
        count = 0

        for elem in root.iter():
            # Check resource href
            if elem.tag.endswith('resource') or elem.tag == 'resource':
                href = elem.get('href')
                if href:
                    count += 1
                    file_path = package_dir / href
                    if not file_path.exists():
                        self.issues.append(ValidationIssue(
                            severity=IssueSeverity.HIGH,
                            code="RR030",
                            message=f"Resource href points to missing file: {href}",
                            file_path=href,
                            resource_id=elem.get('identifier'),
                            suggestion="Ensure the file exists in the package"
                        ))

            # Check file elements
            if elem.tag.endswith('file') or elem.tag == 'file':
                href = elem.get('href')
                if href:
                    count += 1
                    file_path = package_dir / href
                    if not file_path.exists():
                        self.issues.append(ValidationIssue(
                            severity=IssueSeverity.CRITICAL,
                            code="RR031",
                            message=f"File element references missing file: {href}",
                            file_path=href,
                            suggestion="Add the missing file to the package"
                        ))

        return count

    def _check_path_format(self) -> None:
        """Check that all paths are relative, not absolute"""
        for res_id, href in self.resource_hrefs.items():
            if href.startswith('/') or (len(href) > 1 and href[1] == ':'):
                self.issues.append(ValidationIssue(
                    severity=IssueSeverity.MEDIUM,
                    code="RR040",
                    message=f"Absolute path found in resource href: {href}",
                    resource_id=res_id,
                    file_path=href,
                    suggestion="Use relative paths in IMSCC packages"
                ))

            if '\\' in href:
                self.issues.append(ValidationIssue(
                    severity=IssueSeverity.MEDIUM,
                    code="RR041",
                    message=f"Windows-style path separator found: {href}",
                    resource_id=res_id,
                    file_path=href,
                    suggestion="Use forward slashes (/) for path separators"
                ))

    def _validate_html_links(self, package_dir: Path) -> None:
        """Validate internal links in HTML files"""
        html_files = list(package_dir.rglob('*.html')) + list(package_dir.rglob('*.htm'))

        for html_file in html_files:
            try:
                content = html_file.read_text(encoding='utf-8', errors='ignore')

                # Find href and src attributes
                link_pattern = r'(?:href|src)=["\']([^"\'#]+?)(?:#[^"\']*)?["\']'
                links = re.findall(link_pattern, content, re.IGNORECASE)

                for link in links:
                    # Skip external links and data URIs
                    if (link.startswith('http://') or link.startswith('https://') or
                        link.startswith('mailto:') or link.startswith('data:') or
                        link.startswith('javascript:')):
                        continue

                    # Resolve relative to HTML file
                    if link.startswith('/'):
                        target = package_dir / link[1:]
                    else:
                        target = html_file.parent / link

                    # Normalize path
                    try:
                        target = target.resolve()
                        if not target.exists() and not str(target).endswith(('.css', '.js')):
                            rel_path = str(html_file.relative_to(package_dir))
                            self.issues.append(ValidationIssue(
                                severity=IssueSeverity.MEDIUM,
                                code="RR050",
                                message=f"Broken internal link in {rel_path}: {link}",
                                file_path=rel_path,
                                suggestion="Fix the link or add the missing file"
                            ))
                    except (ValueError, OSError):
                        pass  # Path resolution failed, skip

            except Exception as e:
                logger.debug(f"Error checking HTML links in {html_file}: {e}")


def main():
    """CLI entry point"""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Validate resource references in IMSCC packages'
    )
    parser.add_argument('-i', '--input', required=True,
                       help='Path to extracted IMSCC package directory')
    parser.add_argument('-j', '--json', action='store_true', help='Output as JSON')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Verbose output (-vv for debug)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    args = parser.parse_args()

    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)

    validator = ResourceReferenceValidator()
    result = validator.validate_references(Path(args.input))

    if args.json:
        output = {
            'package_path': result.package_path,
            'valid': result.valid,
            'resources_checked': result.resources_checked,
            'files_checked': result.files_checked,
            'broken_references': result.broken_references,
            'issues': [
                {
                    'severity': i.severity.value,
                    'code': i.code,
                    'message': i.message,
                    'resource_id': i.resource_id,
                    'file_path': i.file_path,
                    'suggestion': i.suggestion,
                }
                for i in result.issues
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Package: {result.package_path}")
        print(f"Valid: {result.valid}")
        print(f"Resources Checked: {result.resources_checked}")
        print(f"Files Checked: {result.files_checked}")
        print(f"Broken References: {result.broken_references}")
        print(f"\nIssues Found: {len(result.issues)}")
        for issue in result.issues:
            print(f"  [{issue.severity.value.upper()}] {issue.code}: {issue.message}")
            if issue.file_path:
                print(f"    File: {issue.file_path}")
            if issue.suggestion:
                print(f"    Suggestion: {issue.suggestion}")

    return 0 if result.valid else 1


if __name__ == '__main__':
    exit(main())
