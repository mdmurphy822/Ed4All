"""
IMSCC Manifest XML Validator

Validates imsmanifest.xml files for correct format and Brightspace compatibility.
"""

from typing import List, Dict, Set, Optional
from pathlib import Path
from .xml_validator import IMSCCValidator, ValidationResult, ValidationLevel

try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    import xml.etree.ElementTree as etree


class ManifestValidator(IMSCCValidator):
    """
    Validator for IMSCC manifest (imsmanifest.xml) files.

    Validates against IMSCC 1.1, 1.2, or 1.3 namespaces.
    """

    # Supported manifest namespaces by version
    MANIFEST_NAMESPACES = {
        '1.1.0': 'http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1',
        '1.2.0': 'http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1',
        '1.3.0': 'http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1',
    }

    # Valid resource types
    VALID_RESOURCE_TYPES = {
        'webcontent': 'Web content (HTML, PDF, etc.)',
        'assignment_xmlv1p0': 'Assignment with dropbox',
        'imsdt_xmlv1p3': 'Discussion topic',
        'imsdt_xmlv1p1': 'Discussion topic (v1.1)',
        'imsqti_xmlv1p2/imscc_xmlv1p3/assessment': 'QTI 1.2 assessment (v1.3)',
        'imsqti_xmlv1p2/imscc_xmlv1p2/assessment': 'QTI 1.2 assessment (v1.2)',
        'imsqti_xmlv1p2/imscc_xmlv1p1/assessment': 'QTI 1.2 assessment (v1.1)',
        'imsbasiclti_xmlv1p0': 'Basic LTI link',
        'imsbasiclti_xmlv1p3': 'Basic LTI link (v1.3)',
        'associatedcontent/imscc_xmlv1p1/learning-application-resource': 'Learning app resource',
        'associatedcontent/imscc_xmlv1p3/learning-application-resource': 'Learning app resource (v1.3)',
    }

    # Deprecated resource types to warn about
    DEPRECATED_RESOURCE_TYPES = {
        'imsccv1p1/d2l_2p0/assignment': 'Use assignment_xmlv1p0 instead',
        'imsccv1p1/d2l_2p0/discussion': 'Use imsdt_xmlv1p3 instead',
        'discussion_xmlv1p0': 'Use imsdt_xmlv1p3 instead',
    }

    def validate(self, xml_content: str, package_path: Path = None) -> ValidationResult:
        """
        Perform comprehensive validation of manifest XML.

        Args:
            xml_content: Manifest XML string
            package_path: Optional path to package root for file validation

        Returns:
            ValidationResult with all validation findings
        """
        result = ValidationResult(is_valid=True)

        # Validate root element
        root_result = self.validate_root_element(xml_content, 'manifest')
        result.merge(root_result)

        # Validate namespace and version consistency
        version_result = self._validate_version_consistency(xml_content)
        result.merge(version_result)

        # Validate required elements
        req_result = self._validate_required_elements(xml_content)
        result.merge(req_result)

        # Validate resources
        resource_result = self._validate_resources(xml_content)
        result.merge(resource_result)

        # Validate organization structure
        org_result = self._validate_organization(xml_content)
        result.merge(org_result)

        # Validate identifier references
        ref_result = self._validate_identifier_references(xml_content)
        result.merge(ref_result)

        # Validate file references if package path provided
        if package_path:
            file_result = self._validate_file_references(xml_content, package_path)
            result.merge(file_result)

        return result

    def _validate_version_consistency(self, xml_content: str) -> ValidationResult:
        """Validate that schema version matches namespace."""
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
                actual_ns = doc.nsmap.get(None, '')
            else:
                doc = etree.fromstring(xml_content)
                if doc.tag.startswith('{'):
                    actual_ns = doc.tag[1:doc.tag.index('}')]
                else:
                    actual_ns = ''

            # Find schemaversion
            schema_version = None
            for elem in doc.iter():
                local_name = etree.QName(elem.tag).localname if LXML_AVAILABLE else elem.tag.split('}')[-1]
                if local_name == 'schemaversion':
                    schema_version = elem.text
                    break

            if schema_version:
                expected_ns = self.MANIFEST_NAMESPACES.get(schema_version)
                if expected_ns and expected_ns != actual_ns:
                    result.add_error(
                        f"Version/namespace mismatch: schemaversion '{schema_version}' "
                        f"requires namespace '{expected_ns}', but got '{actual_ns}'",
                        ValidationLevel.CRITICAL
                    )
            else:
                result.add_warning("No schemaversion element found in manifest")

        except Exception as e:
            result.add_error(f"Failed to validate version: {str(e)}", ValidationLevel.HIGH)

        return result

    def _validate_required_elements(self, xml_content: str) -> ValidationResult:
        """Validate required manifest elements."""
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
            else:
                doc = etree.fromstring(xml_content)

            # Check for metadata
            metadata_found = False
            organizations_found = False
            resources_found = False

            for child in doc:
                local_name = etree.QName(child.tag).localname if LXML_AVAILABLE else child.tag.split('}')[-1]
                if local_name == 'metadata':
                    metadata_found = True
                elif local_name == 'organizations':
                    organizations_found = True
                elif local_name == 'resources':
                    resources_found = True

            if not metadata_found:
                result.add_warning("Manifest missing <metadata> element")
            if not organizations_found:
                result.add_warning("Manifest missing <organizations> element")
            if not resources_found:
                result.add_error("Manifest missing <resources> element", ValidationLevel.CRITICAL)

        except Exception as e:
            result.add_error(f"Failed to validate elements: {str(e)}", ValidationLevel.HIGH)

        return result

    # Resource types that require href attribute
    RESOURCE_TYPES_REQUIRING_HREF = {
        'assignment_xmlv1p0',
        'imsdt_xmlv1p3',
        'imsdt_xmlv1p1',
        'imsqti_xmlv1p2/imscc_xmlv1p3/assessment',
        'imsqti_xmlv1p2/imscc_xmlv1p2/assessment',
        'imsqti_xmlv1p2/imscc_xmlv1p1/assessment',
        'webcontent',
    }

    def _validate_resources(self, xml_content: str) -> ValidationResult:
        """Validate resource declarations."""
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
            else:
                doc = etree.fromstring(xml_content)

            resource_count = 0
            seen_ids: Set[str] = set()

            for elem in doc.iter():
                local_name = etree.QName(elem.tag).localname if LXML_AVAILABLE else elem.tag.split('}')[-1]
                if local_name == 'resource':
                    resource_count += 1
                    res_type = elem.get('type')
                    res_id = elem.get('identifier')
                    href = elem.get('href')

                    if not res_id or not res_id.strip():
                        result.add_error("Resource missing or empty 'identifier' attribute", ValidationLevel.HIGH)
                    else:
                        # Check for duplicate identifiers
                        if res_id in seen_ids:
                            result.add_error(
                                f"Duplicate resource identifier: '{res_id}'",
                                ValidationLevel.HIGH
                            )
                        seen_ids.add(res_id)

                    if not res_type:
                        result.add_error(f"Resource '{res_id}' missing 'type' attribute", ValidationLevel.HIGH)
                    elif res_type in self.DEPRECATED_RESOURCE_TYPES:
                        result.add_error(
                            f"Resource '{res_id}' uses deprecated type '{res_type}'. "
                            f"{self.DEPRECATED_RESOURCE_TYPES[res_type]}",
                            ValidationLevel.HIGH
                        )
                    elif res_type not in self.VALID_RESOURCE_TYPES:
                        result.add_warning(f"Resource '{res_id}' has unrecognized type '{res_type}'")

                    # Check href for resource types that require it
                    if res_type in self.RESOURCE_TYPES_REQUIRING_HREF:
                        if not href:
                            result.add_error(
                                f"Resource '{res_id}' with type '{res_type}' missing required 'href' attribute",
                                ValidationLevel.HIGH
                            )
                        elif not href.strip():
                            result.add_error(
                                f"Resource '{res_id}' has empty 'href' attribute",
                                ValidationLevel.HIGH
                            )

            if resource_count == 0:
                result.add_error("No resources defined in manifest", ValidationLevel.HIGH)

        except Exception as e:
            result.add_error(f"Failed to validate resources: {str(e)}", ValidationLevel.HIGH)

        return result

    def _validate_organization(self, xml_content: str) -> ValidationResult:
        """Validate organization structure."""
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
            else:
                doc = etree.fromstring(xml_content)

            org_found = False
            for elem in doc.iter():
                local_name = etree.QName(elem.tag).localname if LXML_AVAILABLE else elem.tag.split('}')[-1]
                if local_name == 'organization':
                    org_found = True
                    structure = elem.get('structure')
                    if structure and structure != 'rooted-hierarchy':
                        result.add_warning(f"Organization structure '{structure}' is non-standard")

                    # Check for items
                    items = list(elem.iter())
                    item_count = sum(1 for e in items if
                                    (etree.QName(e.tag).localname if LXML_AVAILABLE else e.tag.split('}')[-1]) == 'item')
                    if item_count == 0:
                        result.add_warning("Organization has no items (empty course structure)")

            if not org_found:
                result.add_warning("No organization element found")

        except Exception as e:
            result.add_error(f"Failed to validate organization: {str(e)}", ValidationLevel.MEDIUM)

        return result

    def _validate_identifier_references(self, xml_content: str) -> ValidationResult:
        """Validate that identifierref values point to existing resources."""
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
            else:
                doc = etree.fromstring(xml_content)

            # Collect all resource identifiers
            resource_ids: Set[str] = set()
            for elem in doc.iter():
                local_name = etree.QName(elem.tag).localname if LXML_AVAILABLE else elem.tag.split('}')[-1]
                if local_name == 'resource':
                    res_id = elem.get('identifier')
                    if res_id and res_id.strip():
                        resource_ids.add(res_id)

            # Check all identifierref values
            for elem in doc.iter():
                identifierref = elem.get('identifierref')
                if identifierref is not None:
                    if not identifierref.strip():
                        result.add_error(
                            "Empty or whitespace-only identifierref attribute found",
                            ValidationLevel.HIGH
                        )
                    elif identifierref not in resource_ids:
                        result.add_error(
                            f"identifierref '{identifierref}' does not match any resource identifier",
                            ValidationLevel.HIGH
                        )

        except Exception as e:
            result.add_error(f"Failed to validate references: {str(e)}", ValidationLevel.MEDIUM)

        return result

    def _validate_file_references(self, xml_content: str, package_path: Path) -> ValidationResult:
        """Validate that all file href values point to existing files."""
        result = ValidationResult(is_valid=True)

        try:
            if LXML_AVAILABLE:
                doc = etree.fromstring(xml_content.encode('utf-8'))
            else:
                doc = etree.fromstring(xml_content)

            for elem in doc.iter():
                local_name = etree.QName(elem.tag).localname if LXML_AVAILABLE else elem.tag.split('}')[-1]
                if local_name in ['file', 'resource']:
                    href = elem.get('href')
                    if href:
                        file_path = package_path / href
                        if not file_path.exists():
                            result.add_error(
                                f"File reference '{href}' does not exist",
                                ValidationLevel.HIGH
                            )

        except Exception as e:
            result.add_error(f"Failed to validate files: {str(e)}", ValidationLevel.MEDIUM)

        return result


def validate_manifest(xml_content: str, package_path: Path = None) -> ValidationResult:
    """
    Convenience function to validate manifest XML.

    Args:
        xml_content: Manifest XML string
        package_path: Optional path to package root for file validation

    Returns:
        ValidationResult with all validation findings
    """
    validator = ManifestValidator()
    return validator.validate(xml_content, package_path)
