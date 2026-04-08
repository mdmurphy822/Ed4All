"""
Markdown Parser for Semantic Structure Extraction

Parses Markdown content into a hierarchical structure compatible with the
HTML-based heading parser and content block classifier. Supports:
- Standard Markdown headings (#-######)
- Unordered and ordered lists
- Code blocks (fenced and indented)
- Tables (GFM style)
- Blockquotes
- YAML front matter for metadata

Output structure mirrors HeadingHierarchy from heading_parser.py for
seamless integration with the semantic structure extractor.
"""

import re
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum, auto


class MarkdownBlockType(Enum):
    """Types of Markdown blocks."""
    HEADING = auto()
    PARAGRAPH = auto()
    UNORDERED_LIST = auto()
    ORDERED_LIST = auto()
    CODE_BLOCK = auto()
    TABLE = auto()
    BLOCKQUOTE = auto()
    HORIZONTAL_RULE = auto()
    BLANK = auto()


@dataclass
class MarkdownBlock:
    """A parsed Markdown content block."""
    block_type: MarkdownBlockType
    content: str
    level: int = 0  # For headings (1-6), list nesting, blockquote depth
    language: Optional[str] = None  # For code blocks
    items: List[str] = field(default_factory=list)  # For lists
    rows: List[List[str]] = field(default_factory=list)  # For tables
    headers: List[str] = field(default_factory=list)  # For tables
    raw_text: str = ""  # Original text before parsing

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "blockType": self.block_type.name.lower(),
            "content": self.content,
        }
        if self.level > 0:
            result["level"] = self.level
        if self.language:
            result["language"] = self.language
        if self.items:
            result["items"] = self.items
        if self.rows:
            result["headers"] = self.headers
            result["rows"] = self.rows
        return result


@dataclass
class MarkdownSection:
    """A section of Markdown content under a heading."""
    id: str
    level: int
    title: str
    content_blocks: List[MarkdownBlock] = field(default_factory=list)
    subsections: List['MarkdownSection'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "headingLevel": self.level,
            "headingText": self.title,
            "headingId": self.id,
            "contentBlocks": [b.to_dict() for b in self.content_blocks],
            "subsections": [s.to_dict() for s in self.subsections]
        }


@dataclass
class MarkdownDocument:
    """Complete parsed Markdown document."""
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    sections: List[MarkdownSection] = field(default_factory=list)
    content_blocks: List[MarkdownBlock] = field(default_factory=list)  # Before first heading

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching semantic extractor output format."""
        return {
            "documentInfo": {
                "title": self.title or "",
                "sourceFormat": "markdown",
                "metadata": self.metadata
            },
            "chapters": [s.to_dict() for s in self.sections],
            "preambleBlocks": [b.to_dict() for b in self.content_blocks]
        }


class MarkdownParser:
    """
    Parses Markdown content into structured hierarchy.

    Designed to produce output compatible with the HTML-based
    SemanticStructureExtractor for unified processing.
    """

    # Regex patterns for Markdown elements
    HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+?)(?:\s+\{#([^}]+)\})?\s*$')
    UNORDERED_LIST_PATTERN = re.compile(r'^(\s*)[-*+]\s+(.+)$')
    ORDERED_LIST_PATTERN = re.compile(r'^(\s*)\d+[.)]\s+(.+)$')
    CODE_FENCE_PATTERN = re.compile(r'^```(\w*)?\s*$')
    TABLE_ROW_PATTERN = re.compile(r'^\|(.+)\|$')
    TABLE_SEPARATOR_PATTERN = re.compile(r'^\|[\s\-:|]+\|$')
    BLOCKQUOTE_PATTERN = re.compile(r'^(>+)\s*(.*)$')
    HORIZONTAL_RULE_PATTERN = re.compile(r'^[-*_]{3,}\s*$')
    YAML_FRONT_MATTER_START = re.compile(r'^---\s*$')

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parser.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self._section_counter = 0

    def parse(self, content: str, source_path: str = "") -> MarkdownDocument:
        """
        Parse Markdown content into structured document.

        Args:
            content: Markdown text content
            source_path: Optional source file path for metadata

        Returns:
            MarkdownDocument with parsed structure
        """
        self._section_counter = 0
        lines = content.split('\n')

        # Extract YAML front matter if present
        metadata, start_line = self._extract_front_matter(lines)

        # Parse remaining content
        lines = lines[start_line:]
        blocks = self._parse_blocks(lines)

        # Build hierarchical structure from flat blocks
        doc = self._build_hierarchy(blocks, metadata)

        # Set title from first H1 or metadata
        if not doc.title and doc.sections:
            for section in doc.sections:
                if section.level == 1:
                    doc.title = section.title
                    break
        if not doc.title:
            doc.title = metadata.get('title', '')

        return doc

    def parse_file(self, file_path: str) -> MarkdownDocument:
        """
        Parse a Markdown file.

        Args:
            file_path: Path to Markdown file

        Returns:
            MarkdownDocument with parsed structure
        """
        path = Path(file_path)
        content = path.read_text(encoding='utf-8')
        doc = self.parse(content, str(path))
        doc.metadata['sourcePath'] = str(path)
        return doc

    def _extract_front_matter(self, lines: List[str]) -> Tuple[Dict[str, Any], int]:
        """Extract YAML front matter from beginning of document."""
        if not lines or not self.YAML_FRONT_MATTER_START.match(lines[0]):
            return {}, 0

        end_idx = -1
        for i, line in enumerate(lines[1:], 1):
            if self.YAML_FRONT_MATTER_START.match(line):
                end_idx = i
                break

        if end_idx == -1:
            return {}, 0

        yaml_content = '\n'.join(lines[1:end_idx])
        try:
            metadata = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            metadata = {}

        return metadata, end_idx + 1

    def _parse_blocks(self, lines: List[str]) -> List[MarkdownBlock]:
        """Parse lines into a flat list of blocks."""
        blocks = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Blank line
            if not line.strip():
                i += 1
                continue

            # Heading
            heading_match = self.HEADING_PATTERN.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                blocks.append(MarkdownBlock(
                    block_type=MarkdownBlockType.HEADING,
                    content=title,
                    level=level,
                    raw_text=line
                ))
                i += 1
                continue

            # Horizontal rule
            if self.HORIZONTAL_RULE_PATTERN.match(line):
                blocks.append(MarkdownBlock(
                    block_type=MarkdownBlockType.HORIZONTAL_RULE,
                    content="",
                    raw_text=line
                ))
                i += 1
                continue

            # Fenced code block
            fence_match = self.CODE_FENCE_PATTERN.match(line)
            if fence_match:
                language = fence_match.group(1) or ""
                code_lines = []
                i += 1
                while i < len(lines) and not self.CODE_FENCE_PATTERN.match(lines[i]):
                    code_lines.append(lines[i])
                    i += 1
                i += 1  # Skip closing fence
                blocks.append(MarkdownBlock(
                    block_type=MarkdownBlockType.CODE_BLOCK,
                    content='\n'.join(code_lines),
                    language=language,
                    raw_text='\n'.join([line] + code_lines + ['```'])
                ))
                continue

            # Table
            if self.TABLE_ROW_PATTERN.match(line):
                table_lines = [line]
                i += 1
                while i < len(lines) and (
                    self.TABLE_ROW_PATTERN.match(lines[i]) or
                    self.TABLE_SEPARATOR_PATTERN.match(lines[i])
                ):
                    table_lines.append(lines[i])
                    i += 1

                block = self._parse_table(table_lines)
                if block:
                    blocks.append(block)
                continue

            # Blockquote
            bq_match = self.BLOCKQUOTE_PATTERN.match(line)
            if bq_match:
                quote_lines = []
                depth = len(bq_match.group(1))
                while i < len(lines):
                    bq_m = self.BLOCKQUOTE_PATTERN.match(lines[i])
                    if bq_m:
                        quote_lines.append(bq_m.group(2))
                        i += 1
                    elif lines[i].strip() == '':
                        # Check if next non-empty line is still a blockquote
                        j = i + 1
                        while j < len(lines) and lines[j].strip() == '':
                            j += 1
                        if j < len(lines) and self.BLOCKQUOTE_PATTERN.match(lines[j]):
                            quote_lines.append('')
                            i += 1
                        else:
                            break
                    else:
                        break

                blocks.append(MarkdownBlock(
                    block_type=MarkdownBlockType.BLOCKQUOTE,
                    content='\n'.join(quote_lines).strip(),
                    level=depth,
                    raw_text='\n'.join(lines[i-len(quote_lines):i])
                ))
                continue

            # Unordered list
            ul_match = self.UNORDERED_LIST_PATTERN.match(line)
            if ul_match:
                list_items = []
                start_i = i
                while i < len(lines):
                    ul_m = self.UNORDERED_LIST_PATTERN.match(lines[i])
                    if ul_m:
                        list_items.append(ul_m.group(2).strip())
                        i += 1
                    elif lines[i].strip() == '':
                        # Check if list continues after blank line
                        j = i + 1
                        while j < len(lines) and lines[j].strip() == '':
                            j += 1
                        if j < len(lines) and self.UNORDERED_LIST_PATTERN.match(lines[j]):
                            i += 1
                        else:
                            break
                    elif lines[i].startswith('  ') or lines[i].startswith('\t'):
                        # Continuation of previous item
                        if list_items:
                            list_items[-1] += ' ' + lines[i].strip()
                        i += 1
                    else:
                        break

                blocks.append(MarkdownBlock(
                    block_type=MarkdownBlockType.UNORDERED_LIST,
                    content='\n'.join(list_items),
                    items=list_items,
                    raw_text='\n'.join(lines[start_i:i])
                ))
                continue

            # Ordered list
            ol_match = self.ORDERED_LIST_PATTERN.match(line)
            if ol_match:
                list_items = []
                start_i = i
                while i < len(lines):
                    ol_m = self.ORDERED_LIST_PATTERN.match(lines[i])
                    if ol_m:
                        list_items.append(ol_m.group(2).strip())
                        i += 1
                    elif lines[i].strip() == '':
                        j = i + 1
                        while j < len(lines) and lines[j].strip() == '':
                            j += 1
                        if j < len(lines) and self.ORDERED_LIST_PATTERN.match(lines[j]):
                            i += 1
                        else:
                            break
                    elif lines[i].startswith('  ') or lines[i].startswith('\t'):
                        if list_items:
                            list_items[-1] += ' ' + lines[i].strip()
                        i += 1
                    else:
                        break

                blocks.append(MarkdownBlock(
                    block_type=MarkdownBlockType.ORDERED_LIST,
                    content='\n'.join(list_items),
                    items=list_items,
                    raw_text='\n'.join(lines[start_i:i])
                ))
                continue

            # Paragraph (default)
            para_lines = [line]
            i += 1
            while i < len(lines):
                if (lines[i].strip() == '' or
                    self.HEADING_PATTERN.match(lines[i]) or
                    self.CODE_FENCE_PATTERN.match(lines[i]) or
                    self.UNORDERED_LIST_PATTERN.match(lines[i]) or
                    self.ORDERED_LIST_PATTERN.match(lines[i]) or
                    self.TABLE_ROW_PATTERN.match(lines[i]) or
                    self.BLOCKQUOTE_PATTERN.match(lines[i]) or
                    self.HORIZONTAL_RULE_PATTERN.match(lines[i])):
                    break
                para_lines.append(lines[i])
                i += 1

            blocks.append(MarkdownBlock(
                block_type=MarkdownBlockType.PARAGRAPH,
                content=' '.join(para_lines).strip(),
                raw_text='\n'.join(para_lines)
            ))

        return blocks

    def _parse_table(self, lines: List[str]) -> Optional[MarkdownBlock]:
        """Parse table lines into a table block."""
        if len(lines) < 2:
            return None

        # First row is headers
        header_match = self.TABLE_ROW_PATTERN.match(lines[0])
        if not header_match:
            return None

        headers = [cell.strip() for cell in header_match.group(1).split('|')]

        # Second row should be separator
        if len(lines) > 1 and self.TABLE_SEPARATOR_PATTERN.match(lines[1]):
            data_start = 2
        else:
            data_start = 1

        # Parse data rows
        rows = []
        for line in lines[data_start:]:
            row_match = self.TABLE_ROW_PATTERN.match(line)
            if row_match:
                cells = [cell.strip() for cell in row_match.group(1).split('|')]
                rows.append(cells)

        return MarkdownBlock(
            block_type=MarkdownBlockType.TABLE,
            content='\n'.join(lines),
            headers=headers,
            rows=rows,
            raw_text='\n'.join(lines)
        )

    def _build_hierarchy(self, blocks: List[MarkdownBlock], metadata: Dict[str, Any]) -> MarkdownDocument:
        """Build hierarchical document structure from flat blocks."""
        doc = MarkdownDocument(metadata=metadata)

        # Separate preamble (content before first heading) from sectioned content
        section_stack: List[MarkdownSection] = []
        current_blocks: List[MarkdownBlock] = []

        for block in blocks:
            if block.block_type == MarkdownBlockType.HEADING:
                # Create new section
                self._section_counter += 1
                section_id = f"section_{self._section_counter}"
                new_section = MarkdownSection(
                    id=section_id,
                    level=block.level,
                    title=block.content
                )

                # Find parent section
                while section_stack and section_stack[-1].level >= block.level:
                    section_stack.pop()

                if section_stack:
                    section_stack[-1].subsections.append(new_section)
                else:
                    # Add any accumulated blocks to preamble
                    if current_blocks and not doc.sections:
                        doc.content_blocks.extend(current_blocks)
                        current_blocks = []
                    doc.sections.append(new_section)

                section_stack.append(new_section)
                current_blocks = []
            else:
                current_blocks.append(block)
                # Add to current section or preamble
                if section_stack:
                    section_stack[-1].content_blocks.append(block)
                elif not doc.sections:
                    doc.content_blocks.append(block)

        return doc

    def extract_inline_formatting(self, text: str) -> Dict[str, Any]:
        """
        Extract inline formatting from text.

        Args:
            text: Text with Markdown inline formatting

        Returns:
            Dictionary with plain text and formatting information
        """
        # Bold: **text** or __text__
        bold_pattern = re.compile(r'\*\*(.+?)\*\*|__(.+?)__')
        # Italic: *text* or _text_
        italic_pattern = re.compile(r'\*(.+?)\*|_(.+?)_')
        # Code: `text`
        code_pattern = re.compile(r'`(.+?)`')
        # Links: [text](url)
        link_pattern = re.compile(r'\[(.+?)\]\((.+?)\)')
        # Images: ![alt](url)
        image_pattern = re.compile(r'!\[(.+?)\]\((.+?)\)')

        formatting = {
            "bold": [],
            "italic": [],
            "code": [],
            "links": [],
            "images": []
        }

        for match in bold_pattern.finditer(text):
            formatting["bold"].append(match.group(1) or match.group(2))

        for match in italic_pattern.finditer(text):
            content = match.group(1) or match.group(2)
            # Avoid matching inside bold markers
            if content and not content.startswith('*'):
                formatting["italic"].append(content)

        for match in code_pattern.finditer(text):
            formatting["code"].append(match.group(1))

        for match in link_pattern.finditer(text):
            formatting["links"].append({
                "text": match.group(1),
                "url": match.group(2)
            })

        for match in image_pattern.finditer(text):
            formatting["images"].append({
                "alt": match.group(1),
                "url": match.group(2)
            })

        # Strip formatting for plain text
        plain = text
        plain = bold_pattern.sub(r'\1\2', plain)
        plain = italic_pattern.sub(r'\1\2', plain)
        plain = code_pattern.sub(r'\1', plain)
        plain = link_pattern.sub(r'\1', plain)
        plain = image_pattern.sub(r'\1', plain)

        return {
            "plain_text": plain,
            "formatting": formatting
        }


def detect_format(content: str) -> str:
    """
    Detect whether content is Markdown or HTML.

    Args:
        content: Text content to analyze

    Returns:
        "markdown" or "html"
    """
    # Check for HTML indicators
    html_indicators = [
        re.search(r'<html', content, re.IGNORECASE),
        re.search(r'<head', content, re.IGNORECASE),
        re.search(r'<body', content, re.IGNORECASE),
        re.search(r'<!DOCTYPE', content, re.IGNORECASE),
        re.search(r'<div\s', content, re.IGNORECASE),
        re.search(r'<p\s', content, re.IGNORECASE),
    ]

    # Check for Markdown indicators
    markdown_indicators = [
        re.search(r'^#{1,6}\s+', content, re.MULTILINE),  # Headings
        re.search(r'^```', content, re.MULTILINE),  # Code fences
        re.search(r'^\s*[-*+]\s+', content, re.MULTILINE),  # Lists
        re.search(r'^\s*\d+\.\s+', content, re.MULTILINE),  # Numbered lists
        re.search(r'^---\s*$', content, re.MULTILINE),  # YAML front matter or HR
        re.search(r'\[.+?\]\(.+?\)', content),  # Links
    ]

    html_score = sum(1 for i in html_indicators if i)
    markdown_score = sum(1 for i in markdown_indicators if i)

    return "html" if html_score > markdown_score else "markdown"


# Convenience function for direct use
def parse_markdown(content: str, source_path: str = "") -> Dict[str, Any]:
    """
    Parse Markdown content and return dictionary representation.

    Args:
        content: Markdown text content
        source_path: Optional source file path

    Returns:
        Dictionary matching semantic extractor output format
    """
    parser = MarkdownParser()
    doc = parser.parse(content, source_path)
    return doc.to_dict()
