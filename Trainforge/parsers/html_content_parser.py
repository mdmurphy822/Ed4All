"""
HTML Content Parser

Extracts structured content from Courseforge-generated HTML modules.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from html.parser import HTMLParser


@dataclass
class ContentSection:
    """A section of content from an HTML module."""
    heading: str
    level: int  # h1=1, h2=2, etc.
    content: str
    word_count: int
    components: List[str] = field(default_factory=list)  # flip-card, accordion, etc.


@dataclass
class LearningObjective:
    """A learning objective extracted from HTML content."""
    id: Optional[str]
    text: str
    bloom_level: Optional[str] = None
    bloom_verb: Optional[str] = None


@dataclass
class ParsedHTMLModule:
    """Parsed HTML module structure."""
    title: str
    word_count: int
    sections: List[ContentSection] = field(default_factory=list)
    learning_objectives: List[LearningObjective] = field(default_factory=list)
    key_concepts: List[str] = field(default_factory=list)
    interactive_components: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.current_tag = None
        self.in_script = False
        self.in_style = False

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag == 'script':
            self.in_script = True
        elif tag == 'style':
            self.in_style = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.in_script = False
        elif tag == 'style':
            self.in_style = False
        self.current_tag = None

    def handle_data(self, data):
        if not self.in_script and not self.in_style:
            text = data.strip()
            if text:
                self.text_parts.append(text)

    def get_text(self) -> str:
        return ' '.join(self.text_parts)


class HTMLContentParser:
    """
    Parser for Courseforge-generated HTML content.

    Usage:
        parser = HTMLContentParser()
        module = parser.parse(html_content)
        print(f"Word count: {module.word_count}")
        for obj in module.learning_objectives:
            print(f"LO: {obj.text}")
    """

    # Bloom's taxonomy verbs by level
    BLOOM_VERBS = {
        "remember": ["define", "list", "recall", "identify", "recognize", "name", "state"],
        "understand": ["explain", "describe", "summarize", "interpret", "classify", "compare"],
        "apply": ["apply", "demonstrate", "implement", "solve", "use", "execute"],
        "analyze": ["analyze", "differentiate", "examine", "compare", "contrast", "organize"],
        "evaluate": ["evaluate", "assess", "critique", "judge", "justify", "argue"],
        "create": ["create", "design", "develop", "construct", "produce", "formulate"]
    }

    # Interactive component patterns
    COMPONENT_PATTERNS = {
        "flip-card": r'class="[^"]*flip-card[^"]*"',
        "accordion": r'class="[^"]*accordion[^"]*"',
        "tabs": r'class="[^"]*nav-tabs[^"]*"',
        "callout": r'class="[^"]*(?:callout|alert)[^"]*"',
        "knowledge-check": r'class="[^"]*knowledge-check[^"]*"',
        "activity-card": r'class="[^"]*activity-card[^"]*"'
    }

    def parse(self, html_content: str) -> ParsedHTMLModule:
        """
        Parse HTML content into structured format.

        Args:
            html_content: HTML string to parse

        Returns:
            ParsedHTMLModule with extracted structure
        """
        # Extract text
        extractor = HTMLTextExtractor()
        extractor.feed(html_content)
        text = extractor.get_text()
        word_count = len(text.split())

        # Extract title
        title = self._extract_title(html_content)

        # Extract sections
        sections = self._extract_sections(html_content)

        # Extract learning objectives
        objectives = self._extract_objectives(html_content)

        # Extract key concepts
        concepts = self._extract_concepts(html_content)

        # Detect interactive components
        components = self._detect_components(html_content)

        return ParsedHTMLModule(
            title=title,
            word_count=word_count,
            sections=sections,
            learning_objectives=objectives,
            key_concepts=concepts,
            interactive_components=components
        )

    def _extract_title(self, html: str) -> str:
        """Extract page title."""
        # Try <title> tag
        title_match = re.search(r'<title>([^<]+)</title>', html, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()

        # Try <h1>
        h1_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.IGNORECASE)
        if h1_match:
            return h1_match.group(1).strip()

        return "Untitled Module"

    def _extract_sections(self, html: str) -> List[ContentSection]:
        """Extract content sections by heading."""
        sections = []

        # Find all headings
        heading_pattern = r'<h([1-6])[^>]*>([^<]+)</h\1>'
        headings = list(re.finditer(heading_pattern, html, re.IGNORECASE))

        for i, match in enumerate(headings):
            level = int(match.group(1))
            heading_text = match.group(2).strip()

            # Get content between this heading and the next
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(html)
            section_html = html[start:end]

            # Extract text
            extractor = HTMLTextExtractor()
            extractor.feed(section_html)
            content = extractor.get_text()

            # Detect components in section
            components = self._detect_components(section_html)

            sections.append(ContentSection(
                heading=heading_text,
                level=level,
                content=content,
                word_count=len(content.split()),
                components=components
            ))

        return sections

    def _extract_objectives(self, html: str) -> List[LearningObjective]:
        """Extract learning objectives from HTML."""
        objectives = []

        # Pattern 1: List items with "objective" context
        obj_section = re.search(
            r'(?:learning\s+)?objectives?.*?<ul[^>]*>(.*?)</ul>',
            html,
            re.IGNORECASE | re.DOTALL
        )

        if obj_section:
            list_items = re.findall(r'<li[^>]*>([^<]+)</li>', obj_section.group(1))
            for item in list_items:
                text = item.strip()
                bloom_level, bloom_verb = self._detect_bloom_level(text)
                objectives.append(LearningObjective(
                    id=None,
                    text=text,
                    bloom_level=bloom_level,
                    bloom_verb=bloom_verb
                ))

        # Pattern 2: Structured objective markers
        structured = re.findall(
            r'data-objective-id="([^"]*)"[^>]*>([^<]+)',
            html
        )
        for obj_id, text in structured:
            bloom_level, bloom_verb = self._detect_bloom_level(text)
            objectives.append(LearningObjective(
                id=obj_id,
                text=text.strip(),
                bloom_level=bloom_level,
                bloom_verb=bloom_verb
            ))

        return objectives

    def _detect_bloom_level(self, text: str) -> tuple:
        """Detect Bloom's taxonomy level from objective text."""
        text_lower = text.lower()

        for level, verbs in self.BLOOM_VERBS.items():
            for verb in verbs:
                if text_lower.startswith(verb) or f" {verb} " in text_lower:
                    return level, verb

        return None, None

    def _extract_concepts(self, html: str) -> List[str]:
        """Extract key concepts from HTML."""
        concepts = []

        # Look for bold/strong terms
        bold_terms = re.findall(r'<(?:strong|b)[^>]*>([^<]+)</(?:strong|b)>', html)
        concepts.extend([t.strip() for t in bold_terms if len(t.strip()) > 2])

        # Look for definition terms
        dt_terms = re.findall(r'<dt[^>]*>([^<]+)</dt>', html)
        concepts.extend([t.strip() for t in dt_terms])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in concepts:
            if c.lower() not in seen:
                seen.add(c.lower())
                unique.append(c)

        return unique[:20]  # Limit to top 20

    def _detect_components(self, html: str) -> List[str]:
        """Detect interactive components in HTML."""
        components = []

        for component, pattern in self.COMPONENT_PATTERNS.items():
            if re.search(pattern, html, re.IGNORECASE):
                components.append(component)

        return components
