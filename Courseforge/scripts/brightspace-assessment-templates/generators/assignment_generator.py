"""
Assignment XML Generator

Generates IMSCC assignment XML files using the correct namespace.
"""

import os
from typing import List, Optional
from .base_generator import BaseGenerator, escape_for_cdata, escape_xml_attribute
from .constants import (
    NAMESPACES,
    SCHEMA_LOCATIONS,
    RESOURCE_TYPES,
    MAX_POINTS,
    MIN_POINTS,
    MAX_TITLE_LENGTH,
    MAX_CONTENT_LENGTH,
    VALID_SUBMISSION_TYPES,
)


class AssignmentGenerator(BaseGenerator):
    """
    Generator for IMSCC assignment XML files.

    Uses the correct namespace: http://www.imsglobal.org/xsd/imscc_extensions/assignment
    NOT the deprecated d2l_2p0 namespace.

    Manifest resource type: assignment_xmlv1p0
    """

    # Correct namespace - sourced from constants for single source of truth
    NAMESPACE = NAMESPACES['assignment']
    SCHEMA_LOCATION = SCHEMA_LOCATIONS['assignment']

    # Manifest resource type - sourced from constants
    RESOURCE_TYPE = RESOURCE_TYPES['assignment']

    # Valid submission format types
    SUBMISSION_TYPES = {
        'file': '<format type="file" />',
        'text': '<format type="text" />',
        'html': '<format type="html" />',
        'url': '<format type="url" />',
    }

    def _validate_attachment_path(self, path: str) -> None:
        """
        Validate an attachment path is safe (no path traversal).

        Args:
            path: Attachment file path

        Raises:
            ValueError: If path contains traversal or is absolute
        """
        # Reject absolute paths
        if os.path.isabs(path):
            raise ValueError(f"Attachment path must be relative, not absolute: {path}")

        # Reject path traversal
        if '..' in path:
            raise ValueError(f"Attachment path cannot contain '..': {path}")

        # Normalize and check
        normalized = os.path.normpath(path)
        if normalized.startswith('..') or normalized.startswith('/'):
            raise ValueError(f"Invalid attachment path: {path}")

    def generate(self,
                 title: str,
                 instructions: str,
                 points: float = 100.0,
                 submission_types: List[str] = None,
                 identifier: str = None,
                 include_text_submission: bool = False) -> str:
        """
        Generate assignment XML.

        Args:
            title: Assignment title
            instructions: HTML-formatted instructions
            points: Points possible (default 100)
            submission_types: List of submission types ['file', 'text', 'url']
                             If None, defaults to ['file']
            identifier: Unique identifier (auto-generated if not provided)
            include_text_submission: If True and submission_types not specified,
                                    include both 'file' and 'text'

        Returns:
            Valid assignment XML string

        Raises:
            ValueError: If validation fails
        """
        # Validate title
        if not title or not title.strip():
            raise ValueError("Assignment title is required")
        if len(title) > MAX_TITLE_LENGTH:
            raise ValueError(f"Assignment title exceeds maximum length ({MAX_TITLE_LENGTH} chars)")

        # Validate instructions
        if not instructions or not instructions.strip():
            raise ValueError("Assignment instructions are required")
        if len(instructions) > MAX_CONTENT_LENGTH:
            raise ValueError(f"Assignment instructions exceed maximum length ({MAX_CONTENT_LENGTH} chars)")

        # Validate points
        if points < MIN_POINTS:
            raise ValueError(f"Points must be non-negative (got {points})")
        if points > MAX_POINTS:
            raise ValueError(f"Points exceed maximum ({MAX_POINTS})")

        if identifier is None:
            identifier = self.generate_id()

        if submission_types is None:
            if include_text_submission:
                submission_types = ['file', 'text']
            else:
                submission_types = ['file']

        # Build submission formats XML
        formats = '\n    '.join(
            self.SUBMISSION_TYPES[t]
            for t in submission_types
            if t in self.SUBMISSION_TYPES
        )

        # Escape HTML content for XML
        escaped_instructions = escape_for_cdata(instructions)
        escaped_title = escape_for_cdata(title)

        # Format points
        formatted_points = self._format_points(points)

        xml = f'''<?xml version="1.0" encoding="utf-8"?>
<assignment xmlns="{self.NAMESPACE}"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="{self.NAMESPACE} {self.SCHEMA_LOCATION}"
            identifier="{escape_xml_attribute(identifier)}">
  <title>{escaped_title}</title>
  <instructor_text texttype="text/html">{escaped_instructions}</instructor_text>
  <submission_formats>
    {formats}
  </submission_formats>
  <gradable points_possible="{formatted_points}">true</gradable>
</assignment>'''

        return xml

    def generate_with_attachments(self,
                                   title: str,
                                   instructions: str,
                                   points: float = 100.0,
                                   attachments: List[str] = None,
                                   submission_types: List[str] = None,
                                   identifier: str = None) -> str:
        """
        Generate assignment XML with file attachments.

        Args:
            title: Assignment title
            instructions: HTML-formatted instructions
            points: Points possible
            attachments: List of attachment file paths (relative to package)
            submission_types: List of submission types
            identifier: Unique identifier

        Returns:
            Valid assignment XML string with attachments

        Raises:
            ValueError: If validation fails (empty title, invalid points, path traversal)
        """
        # Validate title
        if not title or not title.strip():
            raise ValueError("Assignment title is required")
        if len(title) > MAX_TITLE_LENGTH:
            raise ValueError(f"Assignment title exceeds maximum length ({MAX_TITLE_LENGTH} chars)")

        # Validate instructions
        if not instructions or not instructions.strip():
            raise ValueError("Assignment instructions are required")
        if len(instructions) > MAX_CONTENT_LENGTH:
            raise ValueError(f"Assignment instructions exceed maximum length ({MAX_CONTENT_LENGTH} chars)")

        # Validate points
        if points < MIN_POINTS:
            raise ValueError(f"Points must be non-negative (got {points})")
        if points > MAX_POINTS:
            raise ValueError(f"Points exceed maximum ({MAX_POINTS})")

        if identifier is None:
            identifier = self.generate_id()

        if submission_types is None:
            submission_types = ['file']

        if attachments is None:
            attachments = []

        # Validate attachment paths for security
        for path in attachments:
            self._validate_attachment_path(path)

        # Build submission formats XML
        formats = '\n    '.join(
            self.SUBMISSION_TYPES[t]
            for t in submission_types
            if t in self.SUBMISSION_TYPES
        )

        # Build attachments XML
        if attachments:
            attachments_xml = '  <attachments>\n'
            for href in attachments:
                attachments_xml += f'    <attachment href="{escape_xml_attribute(href)}" />\n'
            attachments_xml += '  </attachments>\n'
        else:
            attachments_xml = ''

        # Escape content
        escaped_instructions = escape_for_cdata(instructions)
        escaped_title = escape_for_cdata(title)
        formatted_points = self._format_points(points)

        xml = f'''<?xml version="1.0" encoding="utf-8"?>
<assignment xmlns="{self.NAMESPACE}"
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
            xsi:schemaLocation="{self.NAMESPACE} {self.SCHEMA_LOCATION}"
            identifier="{escape_xml_attribute(identifier)}">
  <title>{escaped_title}</title>
  <instructor_text texttype="text/html">{escaped_instructions}</instructor_text>
{attachments_xml}  <submission_formats>
    {formats}
  </submission_formats>
  <gradable points_possible="{formatted_points}">true</gradable>
</assignment>'''

        return xml

    def get_resource_type(self) -> str:
        """Return the manifest resource type for assignments."""
        return self.RESOURCE_TYPE

    def get_namespace(self) -> str:
        """Return the XML namespace for assignments."""
        return self.NAMESPACE


def generate_assignment(title: str,
                        instructions: str,
                        points: float = 100.0,
                        submission_types: List[str] = None) -> str:
    """
    Convenience function to generate assignment XML.

    Args:
        title: Assignment title
        instructions: HTML instructions
        points: Points possible
        submission_types: List of submission format types

    Returns:
        Assignment XML string
    """
    generator = AssignmentGenerator()
    return generator.generate(title, instructions, points, submission_types)
