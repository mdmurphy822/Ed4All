"""
Discussion Topic XML Generator

Generates IMSCC discussion topic XML files using the correct namespace and element.
"""

import os
from typing import List, Optional
from .base_generator import BaseGenerator, escape_for_cdata, escape_xml_attribute
from .constants import (
    NAMESPACES,
    SCHEMA_LOCATIONS,
    RESOURCE_TYPES,
    MAX_TITLE_LENGTH,
    MAX_CONTENT_LENGTH,
)


class DiscussionGenerator(BaseGenerator):
    """
    Generator for IMSCC discussion topic XML files.

    IMPORTANT: Uses <topic> root element, NOT <discussion>
    Uses namespace: http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3

    Manifest resource type: imsdt_xmlv1p3
    """

    # Correct namespace - sourced from constants for single source of truth
    NAMESPACE = NAMESPACES['discussion']
    SCHEMA_LOCATION = SCHEMA_LOCATIONS['discussion']

    # Manifest resource type - sourced from constants
    RESOURCE_TYPE = RESOURCE_TYPES['discussion']

    def _validate_attachment_path(self, path: str) -> None:
        """
        Validate an attachment path is safe (no path traversal).

        Args:
            path: Attachment file path

        Raises:
            ValueError: If path contains traversal or is absolute
        """
        if os.path.isabs(path):
            raise ValueError(f"Attachment path must be relative, not absolute: {path}")
        if '..' in path:
            raise ValueError(f"Attachment path cannot contain '..': {path}")
        normalized = os.path.normpath(path)
        if normalized.startswith('..') or normalized.startswith('/'):
            raise ValueError(f"Invalid attachment path: {path}")

    def generate(self,
                 title: str,
                 prompt: str,
                 attachments: List[str] = None) -> str:
        """
        Generate discussion topic XML.

        Args:
            title: Discussion title
            prompt: HTML-formatted discussion prompt/description
            attachments: Optional list of attachment file paths

        Returns:
            Valid discussion topic XML string

        Raises:
            ValueError: If validation fails

        Note:
            Root element is <topic>, NOT <discussion>
        """
        # Validate title
        if not title or not title.strip():
            raise ValueError("Discussion title is required")
        if len(title) > MAX_TITLE_LENGTH:
            raise ValueError(f"Discussion title exceeds maximum length ({MAX_TITLE_LENGTH} chars)")

        # Validate prompt
        if not prompt or not prompt.strip():
            raise ValueError("Discussion prompt is required")
        if len(prompt) > MAX_CONTENT_LENGTH:
            raise ValueError(f"Discussion prompt exceeds maximum length ({MAX_CONTENT_LENGTH} chars)")

        # Validate attachment paths
        if attachments:
            for path in attachments:
                self._validate_attachment_path(path)

        # Escape content
        escaped_title = escape_for_cdata(title)
        escaped_prompt = escape_for_cdata(prompt)

        # Build attachments XML if provided
        attachments_xml = ''
        if attachments:
            attachments_xml = '\n  <attachments>'
            for href in attachments:
                attachments_xml += f'\n    <attachment href="{escape_xml_attribute(href)}" />'
            attachments_xml += '\n  </attachments>'

        xml = f'''<?xml version="1.0" encoding="utf-8"?>
<topic xmlns="{self.NAMESPACE}"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="{self.NAMESPACE} {self.SCHEMA_LOCATION}">
  <title>{escaped_title}</title>
  <text texttype="text/html">{escaped_prompt}</text>{attachments_xml}
</topic>'''

        return xml

    def generate_graded(self,
                        title: str,
                        prompt: str,
                        points: float = 25.0,
                        attachments: List[str] = None) -> str:
        """
        Generate discussion topic XML for a graded discussion.

        Note: Grading configuration is typically handled at the LMS level
        during import, not in the discussion XML itself. This method
        generates the same XML as generate() - the grading points are
        configured in Brightspace after import or via gradebook integration.

        Args:
            title: Discussion title
            prompt: HTML-formatted discussion prompt
            points: Points value (for reference, not embedded in XML)
            attachments: Optional list of attachment file paths

        Returns:
            Valid discussion topic XML string
        """
        # Standard IMSCC discussion topics don't embed grading info
        # That's handled by the LMS during/after import
        return self.generate(title, prompt, attachments)

    def get_resource_type(self) -> str:
        """Return the manifest resource type for discussions."""
        return self.RESOURCE_TYPE

    def get_namespace(self) -> str:
        """Return the XML namespace for discussions."""
        return self.NAMESPACE


def generate_discussion(title: str, prompt: str) -> str:
    """
    Convenience function to generate discussion topic XML.

    Args:
        title: Discussion title
        prompt: Discussion prompt (HTML)

    Returns:
        Discussion topic XML string
    """
    generator = DiscussionGenerator()
    return generator.generate(title, prompt)
