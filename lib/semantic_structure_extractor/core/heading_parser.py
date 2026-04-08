"""
Heading Parser Module

Extracts heading hierarchy from DART-processed HTML documents.
DART outputs semantic HTML with:
- h1-h6 heading hierarchy
- <section aria-labelledby="heading-id"> wrappers
- Roman numeral detection (I., II., III.) -> h2
- Numbered section detection (1., 1.1, 1.2) -> h2/h3
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup, Tag


@dataclass
class HeadingNode:
    """Represents a single heading in the document hierarchy."""
    id: str
    level: int
    text: str
    element_id: Optional[str] = None
    section_element: Optional[Tag] = None
    parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "level": self.level,
            "text": self.text,
            "elementId": self.element_id,
            "parentId": self.parent_id,
            "children": self.children
        }


@dataclass
class HeadingHierarchy:
    """Complete heading hierarchy for a document."""
    document_title: Optional[str] = None
    root_nodes: List[HeadingNode] = field(default_factory=list)
    all_nodes: Dict[str, HeadingNode] = field(default_factory=dict)

    def get_node(self, node_id: str) -> Optional[HeadingNode]:
        """Get a node by its ID."""
        return self.all_nodes.get(node_id)

    def get_children(self, node_id: str) -> List[HeadingNode]:
        """Get all child nodes of a given node."""
        node = self.all_nodes.get(node_id)
        if not node:
            return []
        return [self.all_nodes[child_id] for child_id in node.children if child_id in self.all_nodes]

    def to_toc(self) -> List[Dict[str, Any]]:
        """Convert to table of contents structure."""
        def build_toc_entry(node: HeadingNode) -> Dict[str, Any]:
            entry = {
                "level": node.level,
                "text": node.text,
                "id": node.element_id
            }
            if node.children:
                entry["children"] = [
                    build_toc_entry(self.all_nodes[child_id])
                    for child_id in node.children
                    if child_id in self.all_nodes
                ]
            return entry

        return [build_toc_entry(node) for node in self.root_nodes]


class HeadingParser:
    """
    Parses heading hierarchy from DART-processed HTML.

    DART creates semantic HTML with:
    - Proper h1-h6 heading levels
    - Section wrappers with aria-labelledby attributes
    - Consistent heading detection from various patterns
    """

    # Patterns that DART uses for heading detection
    ROMAN_NUMERAL_PATTERN = re.compile(r'^([IVXLCDM]+)\.\s*(.+)$', re.IGNORECASE)
    NUMBERED_SECTION_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\.\s*(.+)$')
    LETTERED_SECTION_PATTERN = re.compile(r'^([A-Z])\.\s*(.+)$')

    def __init__(self, min_heading_level: int = 1, max_heading_level: int = 6):
        """
        Initialize the heading parser.

        Args:
            min_heading_level: Minimum heading level to extract (default h1)
            max_heading_level: Maximum heading level to extract (default h6)
        """
        self.min_heading_level = min_heading_level
        self.max_heading_level = max_heading_level
        self._node_counter = 0

    def _generate_node_id(self, heading_text: str) -> str:
        """Generate a unique node ID."""
        self._node_counter += 1
        # Create a slug from the heading text
        slug = re.sub(r'[^\w\s-]', '', heading_text.lower())
        slug = re.sub(r'[-\s]+', '-', slug).strip('-')[:50]
        return f"node_{self._node_counter}_{slug}"

    def parse(self, html_content: str) -> HeadingHierarchy:
        """
        Parse HTML content and extract heading hierarchy.

        Args:
            html_content: The HTML string to parse

        Returns:
            HeadingHierarchy object containing all headings
        """
        self._node_counter = 0
        soup = BeautifulSoup(html_content, 'html.parser')

        hierarchy = HeadingHierarchy()

        # Extract document title from h1 or title element
        h1 = soup.find('h1')
        if h1:
            hierarchy.document_title = h1.get_text(strip=True)
        else:
            title_elem = soup.find('title')
            if title_elem:
                hierarchy.document_title = title_elem.get_text(strip=True)

        # Find all headings
        heading_elements = self._find_all_headings(soup)

        # Build hierarchy
        self._build_hierarchy(heading_elements, hierarchy)

        return hierarchy

    def parse_file(self, file_path: str) -> HeadingHierarchy:
        """
        Parse HTML file and extract heading hierarchy.

        Args:
            file_path: Path to the HTML file

        Returns:
            HeadingHierarchy object containing all headings
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        return self.parse(html_content)

    def _find_all_headings(self, soup: BeautifulSoup) -> List[tuple]:
        """
        Find all heading elements in the document.

        Returns:
            List of tuples: (heading_element, level, section_element)
        """
        headings = []

        # Find headings by tag name
        for level in range(self.min_heading_level, self.max_heading_level + 1):
            for heading in soup.find_all(f'h{level}'):
                # Find the parent section if it exists
                section = self._find_parent_section(heading)
                headings.append((heading, level, section))

        # Sort by document order
        headings.sort(key=lambda x: self._get_element_position(x[0], soup))

        return headings

    def _find_parent_section(self, heading: Tag) -> Optional[Tag]:
        """
        Find the parent section element for a heading.

        DART wraps sections with: <section aria-labelledby="heading-id">
        """
        # Check if the heading has an ID
        heading_id = heading.get('id')
        if heading_id:
            # Look for a section with aria-labelledby pointing to this heading
            parent = heading.parent
            while parent:
                if parent.name == 'section':
                    aria_labelledby = parent.get('aria-labelledby')
                    if aria_labelledby == heading_id:
                        return parent
                parent = parent.parent

        # Fall back to finding the nearest parent section
        parent = heading.parent
        while parent:
            if parent.name == 'section':
                return parent
            parent = parent.parent

        return None

    def _get_element_position(self, element: Tag, soup: BeautifulSoup) -> int:
        """Get the position of an element in document order."""
        all_elements = list(soup.descendants)
        try:
            return all_elements.index(element)
        except ValueError:
            return 0

    def _build_hierarchy(self, heading_elements: List[tuple], hierarchy: HeadingHierarchy) -> None:
        """
        Build the heading hierarchy from a flat list of headings.

        Uses a stack-based approach to properly nest headings based on their levels.
        """
        if not heading_elements:
            return

        # Stack to track parent headings at each level
        level_stack: List[Optional[HeadingNode]] = [None] * (self.max_heading_level + 1)

        for heading_elem, level, section_elem in heading_elements:
            text = heading_elem.get_text(strip=True)
            element_id = heading_elem.get('id')

            # Create the node
            node = HeadingNode(
                id=self._generate_node_id(text),
                level=level,
                text=text,
                element_id=element_id,
                section_element=section_elem
            )

            # Find the parent (nearest heading with lower level number)
            parent_node = None
            for parent_level in range(level - 1, 0, -1):
                if level_stack[parent_level] is not None:
                    parent_node = level_stack[parent_level]
                    break

            if parent_node:
                node.parent_id = parent_node.id
                parent_node.children.append(node.id)
            else:
                # This is a root node
                hierarchy.root_nodes.append(node)

            # Update the stack
            level_stack[level] = node
            # Clear deeper levels (they can no longer be parents)
            for deeper_level in range(level + 1, self.max_heading_level + 1):
                level_stack[deeper_level] = None

            # Add to the all_nodes dictionary
            hierarchy.all_nodes[node.id] = node

    def extract_section_content(self, heading_node: HeadingNode) -> Optional[str]:
        """
        Extract the content associated with a heading's section.

        Args:
            heading_node: The HeadingNode to extract content for

        Returns:
            HTML content of the section, or None if no section element
        """
        if heading_node.section_element:
            return str(heading_node.section_element)
        return None

    def detect_heading_pattern(self, text: str) -> Dict[str, Any]:
        """
        Detect the pattern used in a heading.

        DART normalizes various patterns, but we can still detect the original format.

        Returns:
            Dictionary with pattern information
        """
        result = {
            "pattern": "plain",
            "number": None,
            "title": text
        }

        # Check for Roman numerals
        match = self.ROMAN_NUMERAL_PATTERN.match(text)
        if match:
            result["pattern"] = "roman_numeral"
            result["number"] = match.group(1)
            result["title"] = match.group(2).strip()
            return result

        # Check for numbered sections
        match = self.NUMBERED_SECTION_PATTERN.match(text)
        if match:
            result["pattern"] = "numbered"
            result["number"] = match.group(1)
            result["title"] = match.group(2).strip()
            return result

        # Check for lettered sections
        match = self.LETTERED_SECTION_PATTERN.match(text)
        if match:
            result["pattern"] = "lettered"
            result["number"] = match.group(1)
            result["title"] = match.group(2).strip()
            return result

        return result


def extract_heading_hierarchy(html_path: str) -> Dict[str, Any]:
    """
    Convenience function to extract heading hierarchy from an HTML file.

    Args:
        html_path: Path to the HTML file

    Returns:
        Dictionary containing the hierarchy data
    """
    parser = HeadingParser()
    hierarchy = parser.parse_file(html_path)

    return {
        "documentTitle": hierarchy.document_title,
        "tableOfContents": hierarchy.to_toc(),
        "totalHeadings": len(hierarchy.all_nodes),
        "rootNodes": [node.to_dict() for node in hierarchy.root_nodes]
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python heading_parser.py <html_file>")
        sys.exit(1)

    result = extract_heading_hierarchy(sys.argv[1])
    print(json.dumps(result, indent=2))
