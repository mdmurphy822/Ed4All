"""
IMSCC Manifest XML Generator

Generates imsmanifest.xml files with correct IMSCC 1.3 format and resource types.
"""

import os
from typing import List, Optional, Dict, Set
from dataclasses import dataclass, field
from pathlib import Path
from .base_generator import BaseGenerator, escape_for_cdata, escape_xml_attribute, generate_brightspace_id
from .constants import (
    NAMESPACES,
    RESOURCE_TYPES,
    MAX_TITLE_LENGTH,
)


@dataclass
class ResourceEntry:
    """Represents a resource in the manifest."""
    identifier: str
    resource_type: str
    href: str
    files: List[str] = field(default_factory=list)
    title: str = ""
    dependencies: List[str] = field(default_factory=list)


@dataclass
class OrganizationItem:
    """Represents an item in the organization structure."""
    identifier: str
    title: str
    resource_ref: str = ""  # identifierref
    children: List['OrganizationItem'] = field(default_factory=list)


class ManifestGenerator(BaseGenerator):
    """
    Generator for IMSCC manifest (imsmanifest.xml) files.

    Uses IMSCC 1.3 format:
    - Namespace: http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1
    - Schema version: 1.3.0

    Resource Types:
    - webcontent: HTML and other web content
    - assignment_xmlv1p0: Assignments
    - imsdt_xmlv1p3: Discussion topics
    - imsqti_xmlv1p2/imscc_xmlv1p3/assessment: QTI quizzes
    """

    # IMSCC 1.3 namespace - sourced from constants for single source of truth
    NAMESPACE = NAMESPACES['manifest']
    LOM_RESOURCE_NS = NAMESPACES.get('lom', "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource")
    LOM_MANIFEST_NS = NAMESPACES.get('lomimscc', "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest")
    SCHEMA_VERSION = "1.3.0"

    # Resource types - sourced from constants
    MANIFEST_RESOURCE_TYPES = {
        'webcontent': RESOURCE_TYPES.get('webcontent', 'webcontent'),
        'assignment': RESOURCE_TYPES.get('assignment', 'assignment_xmlv1p0'),
        'discussion': RESOURCE_TYPES.get('discussion', 'imsdt_xmlv1p3'),
        'quiz': RESOURCE_TYPES.get('quiz', 'imsqti_xmlv1p2/imscc_xmlv1p3/assessment'),
        'lti': 'imsbasiclti_xmlv1p3',
    }

    def validate_file_references(self,
                                  resources: List[ResourceEntry],
                                  base_path: Path = None) -> List[str]:
        """
        Validate that all file references in resources exist.

        Args:
            resources: List of ResourceEntry objects
            base_path: Base directory to resolve relative paths

        Returns:
            List of error messages for missing files (empty if all valid)
        """
        errors = []
        if base_path is None:
            return errors  # Skip validation if no base path

        base = Path(base_path)
        for res in resources:
            # Check main href
            if res.href:
                file_path = base / res.href
                if not file_path.exists():
                    errors.append(f"Missing file: {res.href} (resource: {res.identifier})")

            # Check additional files
            for f in res.files:
                file_path = base / f
                if not file_path.exists():
                    errors.append(f"Missing file: {f} (resource: {res.identifier})")

        return errors

    def _detect_circular_dependencies(self, resources: List[ResourceEntry]) -> List[str]:
        """
        Detect circular dependencies between resources.

        Args:
            resources: List of ResourceEntry objects

        Returns:
            List of error messages for circular dependencies (empty if none)
        """
        # Build dependency graph
        graph: Dict[str, Set[str]] = {}
        for res in resources:
            graph[res.identifier] = set(res.dependencies)

        # Check for cycles using DFS
        errors = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                cycle = path[path.index(node):] + [node]
                errors.append(f"Circular dependency detected: {' -> '.join(cycle)}")
                return True
            if node in visited:
                return False

            visited.add(node)
            rec_stack.add(node)

            for dep in graph.get(node, []):
                if dfs(dep, path + [node]):
                    return True

            rec_stack.remove(node)
            return False

        for res_id in graph:
            if res_id not in visited:
                dfs(res_id, [])

        return errors

    def generate(self,
                 course_title: str,
                 resources: List[ResourceEntry],
                 organization: List[OrganizationItem] = None,
                 identifier: str = None,
                 description: str = "",
                 validate_files: bool = False,
                 base_path: Path = None) -> str:
        """
        Generate complete manifest XML.

        Args:
            course_title: Title of the course
            resources: List of ResourceEntry objects
            organization: Optional hierarchical organization structure
            identifier: Manifest identifier (auto-generated if not provided)
            description: Optional course description
            validate_files: If True, validate that referenced files exist
            base_path: Base directory for file validation

        Returns:
            Valid manifest XML string

        Raises:
            ValueError: If validation fails
        """
        # Validate course title
        if not course_title or not course_title.strip():
            raise ValueError("Course title is required")
        if len(course_title) > MAX_TITLE_LENGTH:
            raise ValueError(f"Course title exceeds maximum length ({MAX_TITLE_LENGTH} chars)")

        # Check for circular dependencies
        cycle_errors = self._detect_circular_dependencies(resources)
        if cycle_errors:
            raise ValueError(f"Circular dependencies found: {'; '.join(cycle_errors)}")

        # Validate file references if requested
        if validate_files and base_path:
            file_errors = self.validate_file_references(resources, base_path)
            if file_errors:
                raise ValueError(f"Missing files: {'; '.join(file_errors)}")

        if identifier is None:
            identifier = self.generate_id()

        org_id = self.generate_id()

        # Generate resources XML
        resources_xml = self._generate_resources(resources)

        # Generate organization XML
        if organization:
            org_items_xml = self._generate_organization_items(organization)
        else:
            # Create flat organization from resources
            org_items_xml = self._generate_flat_organization(resources)

        # Generate root item identifier to ensure it's escaped
        root_item_id = self.generate_id()

        xml = f'''<?xml version="1.0" encoding="utf-8"?>
<manifest identifier="{escape_xml_attribute(identifier)}"
          xmlns="{self.NAMESPACE}"
          xmlns:lomr="{self.LOM_RESOURCE_NS}"
          xmlns:lomm="{self.LOM_MANIFEST_NS}"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="{self.LOM_RESOURCE_NS} http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lomresource_v1p0.xsd {self.NAMESPACE} http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscp_v1p2_v1p0.xsd {self.LOM_MANIFEST_NS} http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lommanifest_v1p0.xsd">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>{self.SCHEMA_VERSION}</schemaversion>
    <lomm:lom>
      <lomm:general>
        <lomm:title>
          <lomm:string language="en-US">{escape_for_cdata(course_title)}</lomm:string>
        </lomm:title>
      </lomm:general>
    </lomm:lom>
  </metadata>
  <organizations>
    <organization identifier="{escape_xml_attribute(org_id)}" structure="rooted-hierarchy">
      <item identifier="{escape_xml_attribute(root_item_id)}">
{org_items_xml}
      </item>
      <metadata>
        <lomm:lom />
      </metadata>
    </organization>
  </organizations>
  <resources>
{resources_xml}
  </resources>
</manifest>'''

        return xml

    def _generate_resources(self, resources: List[ResourceEntry]) -> str:
        """Generate resources section XML."""
        lines = []
        for res in resources:
            # Build file elements - use escape_xml_attribute for href attribute values
            files_xml = ''
            if res.files:
                files_xml = '\n'.join(
                    f'      <file href="{escape_xml_attribute(f)}" />'
                    for f in res.files
                )
            elif res.href:
                files_xml = f'      <file href="{escape_xml_attribute(res.href)}" />'

            # Build dependencies - escape identifierref attribute values
            deps_xml = ''
            if res.dependencies:
                deps_xml = '\n'.join(
                    f'      <dependency identifierref="{escape_xml_attribute(d)}" />'
                    for d in res.dependencies
                )

            # Escape identifier and type attribute values
            resource_xml = f'''    <resource identifier="{escape_xml_attribute(res.identifier)}" type="{escape_xml_attribute(res.resource_type)}">
{files_xml}
{deps_xml}
    </resource>'''
            lines.append(resource_xml)

        return '\n'.join(lines)

    def _generate_organization_items(self, items: List[OrganizationItem], indent: int = 8) -> str:
        """Generate organization items recursively."""
        lines = []
        pad = ' ' * indent

        for item in items:
            # Escape all attribute values
            if item.resource_ref:
                item_xml = f'{pad}<item identifier="{escape_xml_attribute(item.identifier)}" identifierref="{escape_xml_attribute(item.resource_ref)}">'
            else:
                item_xml = f'{pad}<item identifier="{escape_xml_attribute(item.identifier)}">'

            lines.append(item_xml)
            lines.append(f'{pad}  <title>{escape_for_cdata(item.title)}</title>')

            if item.children:
                children_xml = self._generate_organization_items(item.children, indent + 2)
                lines.append(children_xml)

            lines.append(f'{pad}</item>')

        return '\n'.join(lines)

    def _generate_flat_organization(self, resources: List[ResourceEntry]) -> str:
        """Generate flat organization structure from resources."""
        lines = []
        pad = ' ' * 8

        for res in resources:
            title = res.title or res.identifier
            # Escape identifier and identifierref attribute values
            lines.append(f'{pad}<item identifier="{escape_xml_attribute(self.generate_id())}" identifierref="{escape_xml_attribute(res.identifier)}">')
            lines.append(f'{pad}  <title>{escape_for_cdata(title)}</title>')
            lines.append(f'{pad}</item>')

        return '\n'.join(lines)

    def create_resource(self,
                        resource_type: str,
                        href: str,
                        title: str = "",
                        identifier: str = None,
                        files: List[str] = None) -> ResourceEntry:
        """
        Create a ResourceEntry with proper type.

        Args:
            resource_type: Type key ('webcontent', 'assignment', 'discussion', 'quiz')
            href: Path to main resource file
            title: Resource title
            identifier: Resource identifier (auto-generated if not provided)
            files: Additional files for this resource

        Returns:
            ResourceEntry object
        """
        if identifier is None:
            identifier = self.generate_id() + "_R"

        actual_type = self.MANIFEST_RESOURCE_TYPES.get(resource_type, resource_type)

        return ResourceEntry(
            identifier=identifier,
            resource_type=actual_type,
            href=href,
            title=title,
            files=files or [href]
        )

    def create_webcontent_resource(self, href: str, title: str = "", identifier: str = None) -> ResourceEntry:
        """Create a webcontent resource entry."""
        return self.create_resource('webcontent', href, title, identifier)

    def create_assignment_resource(self, href: str, title: str = "", identifier: str = None) -> ResourceEntry:
        """Create an assignment resource entry."""
        return self.create_resource('assignment', href, title, identifier)

    def create_discussion_resource(self, href: str, title: str = "", identifier: str = None) -> ResourceEntry:
        """Create a discussion resource entry."""
        return self.create_resource('discussion', href, title, identifier)

    def create_quiz_resource(self, href: str, title: str = "", identifier: str = None) -> ResourceEntry:
        """Create a quiz/assessment resource entry."""
        return self.create_resource('quiz', href, title, identifier)

    def get_namespace(self) -> str:
        """Return the IMSCC manifest namespace."""
        return self.NAMESPACE

    def get_schema_version(self) -> str:
        """Return the schema version."""
        return self.SCHEMA_VERSION


# Convenience functions

def generate_manifest(course_title: str, resources: List[ResourceEntry]) -> str:
    """
    Convenience function to generate manifest XML.

    Args:
        course_title: Course title
        resources: List of ResourceEntry objects

    Returns:
        Manifest XML string
    """
    generator = ManifestGenerator()
    return generator.generate(course_title, resources)
