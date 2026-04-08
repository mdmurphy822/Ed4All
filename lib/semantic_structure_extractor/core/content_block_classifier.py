"""
Content Block Classifier Module

Classifies content blocks within DART-processed HTML documents.
Identifies:
- Paragraphs with definitions (term: definition pattern)
- Key terms (emphasized with <strong> or <em>)
- Lists (ordered and unordered)
- Definition lists (<dl> elements)
- Tables with their data
- Figures with captions
- Callout boxes (info, warning, note)
- Examples and case studies
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from bs4 import BeautifulSoup, Tag, NavigableString
from enum import Enum


class BlockType(Enum):
    """Types of content blocks."""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ORDERED = "list_ordered"
    LIST_UNORDERED = "list_unordered"
    DEFINITION_LIST = "definition_list"
    TABLE = "table"
    FIGURE = "figure"
    CALLOUT_INFO = "callout_info"
    CALLOUT_WARNING = "callout_warning"
    CALLOUT_NOTE = "callout_note"
    CODE_BLOCK = "code_block"
    BLOCKQUOTE = "blockquote"
    EXAMPLE = "example"
    SUMMARY = "summary"
    OBJECTIVES = "objectives"
    REVIEW_QUESTIONS = "review_questions"


@dataclass
class Definition:
    """A term-definition pair."""
    term: str
    definition: str
    source_type: str  # 'dl_element', 'strong_colon', 'inline_definition'


@dataclass
class KeyTerm:
    """An emphasized key term."""
    term: str
    context: str
    emphasis_type: str  # 'strong', 'em', 'heading', 'callout'


@dataclass
class TableData:
    """Structured table data."""
    caption: Optional[str]
    headers: List[str]
    rows: List[List[str]]


@dataclass
class FigureData:
    """Figure/image data."""
    src: str
    alt: str
    caption: Optional[str]


@dataclass
class ContentBlock:
    """A classified content block."""
    id: str
    block_type: BlockType
    content: str = ""
    list_items: List[str] = field(default_factory=list)
    table_data: Optional[TableData] = None
    figure_data: Optional[FigureData] = None
    definitions: List[Definition] = field(default_factory=list)
    key_terms: List[KeyTerm] = field(default_factory=list)
    word_count: int = 0
    element: Optional[Tag] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "blockType": self.block_type.value,
            "content": self.content,
            "wordCount": self.word_count,
            "containsDefinitions": len(self.definitions) > 0,
            "containsKeyTerms": len(self.key_terms) > 0
        }

        if self.list_items:
            result["listItems"] = self.list_items

        if self.table_data:
            result["tableData"] = {
                "caption": self.table_data.caption,
                "headers": self.table_data.headers,
                "rows": self.table_data.rows
            }

        if self.figure_data:
            result["figureData"] = {
                "src": self.figure_data.src,
                "alt": self.figure_data.alt,
                "caption": self.figure_data.caption
            }

        return result


class ContentBlockClassifier:
    """
    Classifies content blocks within HTML sections.

    Designed to work with DART-processed HTML which has:
    - Clean semantic structure
    - Proper heading hierarchy
    - Accessible tables and figures
    - Callout boxes with role="note"
    """

    # Patterns for detecting definitions
    DEFINITION_PATTERNS = [
        re.compile(r'<strong>([^<]+)</strong>\s*[:—-]\s*(.+)', re.IGNORECASE),
        re.compile(r'\*\*([^*]+)\*\*\s*[:—-]\s*(.+)'),
        re.compile(r'([A-Z][^.]+?)\s+(?:is|are|refers to|means)\s+(.+)', re.IGNORECASE),
    ]

    # Patterns for detecting examples
    EXAMPLE_INDICATORS = [
        'for example',
        'for instance',
        'such as',
        'consider the following',
        'here is an example',
        'as an illustration',
        'e.g.',
        'i.e.',
    ]

    # Patterns for detecting learning objectives
    OBJECTIVE_INDICATORS = [
        'learning objective',
        'after completing',
        'by the end of',
        'you will be able to',
        'students will',
        'learners will',
        'upon completion',
    ]

    # Patterns for summary sections
    SUMMARY_INDICATORS = [
        'summary',
        'key takeaways',
        'key points',
        'in conclusion',
        'chapter summary',
        'section summary',
        'recap',
    ]

    # Review question indicators
    REVIEW_INDICATORS = [
        'review question',
        'practice question',
        'self-assessment',
        'check your understanding',
        'knowledge check',
        'quiz',
    ]

    def __init__(self):
        self._block_counter = 0

    def _generate_block_id(self) -> str:
        """Generate a unique block ID."""
        self._block_counter += 1
        return f"block_{self._block_counter}"

    def classify_section(self, section_element: Tag) -> List[ContentBlock]:
        """
        Classify all content blocks within a section element.

        Args:
            section_element: BeautifulSoup Tag representing a section

        Returns:
            List of ContentBlock objects
        """
        blocks = []

        # Process direct children of the section
        for child in section_element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    blocks.append(self._create_paragraph_block(text))
            elif isinstance(child, Tag):
                block = self._classify_element(child)
                if block:
                    blocks.append(block)

        return blocks

    def classify_html(self, html_content: str) -> List[ContentBlock]:
        """
        Classify all content blocks in an HTML document.

        Args:
            html_content: The HTML string

        Returns:
            List of ContentBlock objects
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        main = soup.find('main') or soup.find('body') or soup

        blocks = []
        self._block_counter = 0

        for element in main.descendants:
            if isinstance(element, Tag) and element.parent == main:
                block = self._classify_element(element)
                if block:
                    blocks.append(block)

        return blocks

    def _classify_element(self, element: Tag) -> Optional[ContentBlock]:
        """Classify a single HTML element."""
        tag_name = element.name.lower()

        # Headings
        if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            return self._create_heading_block(element)

        # Paragraphs
        if tag_name == 'p':
            return self._create_paragraph_block_from_element(element)

        # Ordered lists
        if tag_name == 'ol':
            return self._create_ordered_list_block(element)

        # Unordered lists
        if tag_name == 'ul':
            return self._create_unordered_list_block(element)

        # Definition lists
        if tag_name == 'dl':
            return self._create_definition_list_block(element)

        # Tables
        if tag_name == 'table':
            return self._create_table_block(element)

        # Figures
        if tag_name == 'figure':
            return self._create_figure_block(element)

        # Callout/note divs
        if tag_name == 'div':
            return self._classify_div_element(element)

        # Code blocks
        if tag_name == 'pre':
            return self._create_code_block(element)

        # Blockquotes
        if tag_name == 'blockquote':
            return self._create_blockquote_block(element)

        # Sections (process recursively)
        if tag_name == 'section':
            # Return None - sections are containers, not content blocks
            return None

        return None

    def _classify_div_element(self, element: Tag) -> Optional[ContentBlock]:
        """Classify a div element based on its classes and role."""
        classes = element.get('class', [])
        role = element.get('role', '')

        # Check for callout boxes (DART uses role="note")
        if role == 'note' or 'callout' in ' '.join(classes):
            return self._create_callout_block(element, classes)

        # Check for example boxes
        if 'example' in ' '.join(classes):
            return self._create_example_block(element)

        return None

    def _create_heading_block(self, element: Tag) -> ContentBlock:
        """Create a heading content block."""
        text = element.get_text(strip=True)
        return ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.HEADING,
            content=text,
            word_count=len(text.split()),
            element=element
        )

    def _create_paragraph_block(self, text: str) -> ContentBlock:
        """Create a paragraph block from text."""
        return ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.PARAGRAPH,
            content=text,
            word_count=len(text.split())
        )

    def _create_paragraph_block_from_element(self, element: Tag) -> ContentBlock:
        """Create a paragraph block from a <p> element."""
        text = element.get_text(strip=True)
        html = str(element)

        block = ContentBlock(
            id=self._generate_block_id(),
            block_type=self._detect_paragraph_subtype(element, text),
            content=text,
            word_count=len(text.split()),
            element=element
        )

        # Extract definitions from the paragraph
        block.definitions = self._extract_definitions(element)

        # Extract key terms
        block.key_terms = self._extract_key_terms(element)

        return block

    def _detect_paragraph_subtype(self, element: Tag, text: str) -> BlockType:
        """Detect if a paragraph has a special subtype."""
        text_lower = text.lower()

        # Check for objectives section
        for indicator in self.OBJECTIVE_INDICATORS:
            if indicator in text_lower:
                return BlockType.OBJECTIVES

        # Check for summary
        for indicator in self.SUMMARY_INDICATORS:
            if indicator in text_lower:
                return BlockType.SUMMARY

        # Check for examples
        for indicator in self.EXAMPLE_INDICATORS:
            if indicator in text_lower:
                return BlockType.EXAMPLE

        return BlockType.PARAGRAPH

    def _create_ordered_list_block(self, element: Tag) -> ContentBlock:
        """Create an ordered list block."""
        items = [li.get_text(strip=True) for li in element.find_all('li', recursive=False)]
        content = element.get_text(strip=True)

        block = ContentBlock(
            id=self._generate_block_id(),
            block_type=self._detect_list_subtype(element, content),
            content=content,
            list_items=items,
            word_count=len(content.split()),
            element=element
        )

        # Extract key terms from list items
        for li in element.find_all('li'):
            block.key_terms.extend(self._extract_key_terms(li))

        return block

    def _create_unordered_list_block(self, element: Tag) -> ContentBlock:
        """Create an unordered list block."""
        items = [li.get_text(strip=True) for li in element.find_all('li', recursive=False)]
        content = element.get_text(strip=True)

        block = ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.LIST_UNORDERED,
            content=content,
            list_items=items,
            word_count=len(content.split()),
            element=element
        )

        # Extract key terms from list items
        for li in element.find_all('li'):
            block.key_terms.extend(self._extract_key_terms(li))

        return block

    def _detect_list_subtype(self, element: Tag, content: str) -> BlockType:
        """Detect if an ordered list is a special type (objectives, review questions)."""
        content_lower = content.lower()

        # Check parent for context
        parent = element.parent
        if parent:
            parent_text = parent.get_text(strip=True).lower() if parent.name != 'body' else ""

            for indicator in self.OBJECTIVE_INDICATORS:
                if indicator in parent_text:
                    return BlockType.OBJECTIVES

            for indicator in self.REVIEW_INDICATORS:
                if indicator in parent_text:
                    return BlockType.REVIEW_QUESTIONS

        return BlockType.LIST_ORDERED

    def _create_definition_list_block(self, element: Tag) -> ContentBlock:
        """Create a definition list block."""
        definitions = []

        dt_elements = element.find_all('dt')
        dd_elements = element.find_all('dd')

        for dt, dd in zip(dt_elements, dd_elements):
            definitions.append(Definition(
                term=dt.get_text(strip=True),
                definition=dd.get_text(strip=True),
                source_type='dl_element'
            ))

        content = element.get_text(strip=True)

        return ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.DEFINITION_LIST,
            content=content,
            definitions=definitions,
            word_count=len(content.split()),
            element=element
        )

    def _create_table_block(self, element: Tag) -> ContentBlock:
        """Create a table block with structured data."""
        # Extract caption
        caption_elem = element.find('caption')
        caption = caption_elem.get_text(strip=True) if caption_elem else None

        # Extract headers
        headers = []
        thead = element.find('thead')
        if thead:
            header_row = thead.find('tr')
            if header_row:
                headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

        # Extract rows
        rows = []
        tbody = element.find('tbody') or element
        for tr in tbody.find_all('tr'):
            if tr.parent.name != 'thead':
                row = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
                rows.append(row)

        table_data = TableData(
            caption=caption,
            headers=headers,
            rows=rows
        )

        content = element.get_text(strip=True)

        return ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.TABLE,
            content=content,
            table_data=table_data,
            word_count=len(content.split()),
            element=element
        )

    def _create_figure_block(self, element: Tag) -> ContentBlock:
        """Create a figure block."""
        img = element.find('img')
        figcaption = element.find('figcaption')

        figure_data = FigureData(
            src=img.get('src', '') if img else '',
            alt=img.get('alt', '') if img else '',
            caption=figcaption.get_text(strip=True) if figcaption else None
        )

        content = element.get_text(strip=True)

        return ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.FIGURE,
            content=content,
            figure_data=figure_data,
            word_count=len(content.split()),
            element=element
        )

    def _create_callout_block(self, element: Tag, classes: List[str]) -> ContentBlock:
        """Create a callout/note block."""
        # Determine callout type
        classes_str = ' '.join(classes).lower()

        if 'warning' in classes_str:
            block_type = BlockType.CALLOUT_WARNING
        elif 'info' in classes_str:
            block_type = BlockType.CALLOUT_INFO
        else:
            block_type = BlockType.CALLOUT_NOTE

        content = element.get_text(strip=True)

        block = ContentBlock(
            id=self._generate_block_id(),
            block_type=block_type,
            content=content,
            word_count=len(content.split()),
            element=element
        )

        # Extract key terms from callout
        block.key_terms = self._extract_key_terms(element)

        return block

    def _create_example_block(self, element: Tag) -> ContentBlock:
        """Create an example block."""
        content = element.get_text(strip=True)

        return ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.EXAMPLE,
            content=content,
            word_count=len(content.split()),
            element=element
        )

    def _create_code_block(self, element: Tag) -> ContentBlock:
        """Create a code block."""
        code = element.find('code')
        content = code.get_text() if code else element.get_text()

        return ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.CODE_BLOCK,
            content=content,
            word_count=len(content.split()),
            element=element
        )

    def _create_blockquote_block(self, element: Tag) -> ContentBlock:
        """Create a blockquote block."""
        content = element.get_text(strip=True)

        return ContentBlock(
            id=self._generate_block_id(),
            block_type=BlockType.BLOCKQUOTE,
            content=content,
            word_count=len(content.split()),
            element=element
        )

    def _extract_definitions(self, element: Tag) -> List[Definition]:
        """Extract definitions from an element's content."""
        definitions = []
        html = str(element)

        # Check for <strong>Term</strong>: Definition pattern
        for strong in element.find_all('strong'):
            next_text = strong.next_sibling
            if next_text:
                text = str(next_text).strip()
                if text.startswith(':') or text.startswith('-') or text.startswith('—'):
                    definitions.append(Definition(
                        term=strong.get_text(strip=True),
                        definition=text.lstrip(':- —').strip(),
                        source_type='strong_colon'
                    ))

        # Check for "Term is/means/refers to" patterns
        text = element.get_text()
        for pattern in self.DEFINITION_PATTERNS[2:]:  # Skip HTML patterns
            match = pattern.search(text)
            if match:
                definitions.append(Definition(
                    term=match.group(1).strip(),
                    definition=match.group(2).strip(),
                    source_type='inline_definition'
                ))

        return definitions

    def _extract_key_terms(self, element: Tag) -> List[KeyTerm]:
        """Extract emphasized key terms from an element."""
        key_terms = []

        # Find <strong> terms
        for strong in element.find_all('strong'):
            term = strong.get_text(strip=True)
            if term and len(term) < 100:  # Reasonable term length
                # Get surrounding context
                parent_text = strong.parent.get_text(strip=True) if strong.parent else ""
                context = parent_text[:200] if len(parent_text) > 200 else parent_text

                key_terms.append(KeyTerm(
                    term=term,
                    context=context,
                    emphasis_type='strong'
                ))

        # Find <em> terms
        for em in element.find_all('em'):
            term = em.get_text(strip=True)
            if term and len(term) < 100:
                parent_text = em.parent.get_text(strip=True) if em.parent else ""
                context = parent_text[:200] if len(parent_text) > 200 else parent_text

                key_terms.append(KeyTerm(
                    term=term,
                    context=context,
                    emphasis_type='em'
                ))

        return key_terms


def classify_html_content(html_path: str) -> Dict[str, Any]:
    """
    Convenience function to classify content blocks in an HTML file.

    Args:
        html_path: Path to the HTML file

    Returns:
        Dictionary containing classified blocks
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    classifier = ContentBlockClassifier()
    blocks = classifier.classify_html(html_content)

    # Summarize by block type
    type_counts = {}
    for block in blocks:
        type_name = block.block_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    # Collect all definitions
    all_definitions = []
    for block in blocks:
        for defn in block.definitions:
            all_definitions.append({
                "term": defn.term,
                "definition": defn.definition,
                "sourceType": defn.source_type
            })

    # Collect all key terms
    all_key_terms = []
    for block in blocks:
        for term in block.key_terms:
            all_key_terms.append({
                "term": term.term,
                "context": term.context,
                "emphasisType": term.emphasis_type
            })

    return {
        "totalBlocks": len(blocks),
        "blockTypeCounts": type_counts,
        "blocks": [block.to_dict() for block in blocks],
        "allDefinitions": all_definitions,
        "allKeyTerms": all_key_terms
    }


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python content_block_classifier.py <html_file>")
        sys.exit(1)

    result = classify_html_content(sys.argv[1])
    print(json.dumps(result, indent=2))
