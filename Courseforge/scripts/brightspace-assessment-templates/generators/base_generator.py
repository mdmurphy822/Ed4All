"""
Base Generator for IMSCC XML Components

Provides common utilities for all IMSCC XML generators:
- UUID generation matching Brightspace format
- XML escaping
- Template loading
"""

import uuid
import html
from pathlib import Path
from typing import Optional, Dict, Any
from xml.sax.saxutils import escape as xml_escape


def generate_brightspace_id() -> str:
    """
    Generate a UUID in Brightspace format.

    Brightspace uses 'i' prefix followed by 32-character hex UUID.

    Returns:
        String like 'i9c92b88bf2b64efa9cc8e6943b6028fb'
    """
    return f"i{uuid.uuid4().hex}"


def generate_short_id() -> str:
    """
    Generate a shorter ID for internal references.

    Returns:
        String like 'i9c92b88b-f2b6-4efa'
    """
    u = uuid.uuid4()
    return f"i{str(u)[:18]}"


def escape_xml_content(content: str) -> str:
    """
    Escape content for safe inclusion in XML.

    DEPRECATED: Use escape_for_cdata() instead for HTML content,
    or escape_xml_attribute() for attribute values.

    This handles HTML content that needs to be embedded in XML,
    converting HTML entities and escaping special XML characters.

    Args:
        content: Raw content string (may contain HTML)

    Returns:
        XML-safe escaped string
    """
    # Use html.escape which handles &, <, >, ", '
    return html.escape(content, quote=True)


def escape_for_cdata(content: str) -> str:
    """
    Escape content for inclusion in XML where it will be displayed as HTML.

    For mattext elements with texttype="text/html", we need to escape
    the HTML so it's preserved as-is in the XML.

    Args:
        content: HTML content string

    Returns:
        Escaped string suitable for XML text content
    """
    # Escape XML special characters so HTML is preserved
    return html.escape(content, quote=False)


def escape_xml_attribute(value: str) -> str:
    """
    Escape a value for safe use in XML attribute values.

    This properly escapes all characters that could break XML attribute parsing:
    - & becomes &amp;
    - < becomes &lt;
    - > becomes &gt;
    - " becomes &quot;
    - ' becomes &apos;

    IMPORTANT: Use this for ALL XML attribute values including:
    - identifier attributes
    - href attributes
    - type attributes
    - Any other attribute that could contain user-provided data

    Args:
        value: Raw string to be used as attribute value

    Returns:
        Escaped string safe for use in XML attributes

    Example:
        >>> escape_xml_attribute('test"value')
        'test&quot;value'
        >>> f'<element attr="{escape_xml_attribute(user_input)}" />'
    """
    if value is None:
        return ''
    # Use xml_escape with custom entities dict to include quotes
    return xml_escape(str(value), {'"': '&quot;', "'": '&apos;'})


class BaseGenerator:
    """
    Base class for IMSCC XML generators.

    Provides common functionality for loading templates and
    generating XML content.
    """

    # Template directory relative to this file
    TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

    def __init__(self):
        """Initialize generator."""
        self._template_cache: Dict[str, str] = {}

    def _load_template(self, template_name: str) -> str:
        """
        Load a template file from the templates directory.

        Args:
            template_name: Name of template file (e.g., 'assignment_template.xml')

        Returns:
            Template content as string

        Raises:
            FileNotFoundError: If template doesn't exist
        """
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        template_path = self.TEMPLATE_DIR / template_name
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, 'r', encoding='utf-8') as f:
            content = f.read()

        self._template_cache[template_name] = content
        return content

    def _fill_template(self, template: str, values: Dict[str, Any]) -> str:
        """
        Fill template placeholders with values.

        Args:
            template: Template string with {PLACEHOLDER} markers
            values: Dictionary of placeholder -> value mappings

        Returns:
            Filled template string
        """
        result = template
        for key, value in values.items():
            placeholder = f"{{{key}}}"
            if isinstance(value, (int, float)):
                result = result.replace(placeholder, str(value))
            elif value is None:
                result = result.replace(placeholder, '')
            else:
                result = result.replace(placeholder, str(value))
        return result

    def _format_points(self, points: float) -> str:
        """
        Format points value in Brightspace format.

        Brightspace uses 9 decimal places for points.

        Args:
            points: Numeric points value

        Returns:
            String like '100.000000000'
        """
        return f"{float(points):.9f}"

    def generate_id(self) -> str:
        """Generate a new Brightspace-format ID."""
        return generate_brightspace_id()

    def validate_output(self, xml_content: str) -> bool:
        """
        Basic validation of generated XML.

        Args:
            xml_content: Generated XML string

        Returns:
            True if XML is well-formed
        """
        try:
            from xml.etree import ElementTree as ET
            ET.fromstring(xml_content)
            return True
        except Exception:
            return False
