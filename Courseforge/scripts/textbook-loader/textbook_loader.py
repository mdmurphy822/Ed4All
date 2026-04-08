#!/usr/bin/env python3
"""
Textbook Loader - Load DART-processed HTML textbooks for Courseforge

This module provides the bridge between DART output and Courseforge input,
loading accessible HTML files from DART and extracting structured content
for course generation.

Pipeline Position:
    DART (PDF→HTML) → [textbook_loader.py] → Courseforge (content generation)
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# Add project paths
ED4ALL_ROOT = Path(__file__).resolve().parents[3]  # → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

# Add Trainforge for HTML parser
TRAINFORGE_PATH = ED4ALL_ROOT / "Trainforge"
if str(TRAINFORGE_PATH) not in sys.path:
    sys.path.insert(0, str(TRAINFORGE_PATH))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

logger = logging.getLogger(__name__)


@dataclass
class TextbookSection:
    """A section extracted from a DART-processed textbook."""
    section_id: str
    title: str
    content: str
    level: int  # Heading level (1-6)
    word_count: int
    has_images: bool = False
    has_math: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "word_count": self.word_count,
            "has_images": self.has_images,
            "has_math": self.has_math,
        }


@dataclass
class TextbookContent:
    """Structured content from a DART-processed textbook."""
    path: Path
    title: str
    sections: List[TextbookSection] = field(default_factory=list)
    learning_objectives: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "learning_objectives": self.learning_objectives,
            "concepts": self.concepts,
            "metadata": self.metadata,
        }

    @property
    def total_word_count(self) -> int:
        return sum(s.word_count for s in self.sections)

    @property
    def section_count(self) -> int:
        return len(self.sections)


class TextbookLoader:
    """
    Load DART-processed HTML textbooks for Courseforge content generation.

    This loader:
    1. Scans the textbooks directory for DART HTML output
    2. Parses each HTML file to extract sections, objectives, concepts
    3. Returns structured content for course generation

    Usage:
        loader = TextbookLoader()
        textbooks = loader.load_all(textbooks_dir)
        for tb in textbooks:
            print(f"{tb.title}: {tb.section_count} sections")
    """

    def __init__(
        self,
        capture: Optional["DecisionCapture"] = None,
    ):
        """
        Initialize the textbook loader.

        Args:
            capture: Optional DecisionCapture for logging loading decisions
        """
        self.capture = capture
        self._parser = None

    def _get_parser(self):
        """Lazy-load the HTML content parser."""
        if self._parser is None:
            try:
                from parsers.html_content_parser import HTMLContentParser
                self._parser = HTMLContentParser()
            except ImportError:
                logger.warning("HTMLContentParser not available, using basic parsing")
                self._parser = None
        return self._parser

    def load_all(
        self,
        textbooks_dir: Path,
        recursive: bool = True,
    ) -> List[TextbookContent]:
        """
        Load all DART-processed textbooks from a directory.

        Args:
            textbooks_dir: Path to textbooks directory
            recursive: If True, search subdirectories

        Returns:
            List of TextbookContent objects
        """
        textbooks_dir = Path(textbooks_dir)
        if not textbooks_dir.exists():
            logger.warning(f"Textbooks directory not found: {textbooks_dir}")
            return []

        pattern = "**/*.html" if recursive else "*.html"
        html_files = list(textbooks_dir.glob(pattern))

        if not html_files:
            logger.info(f"No HTML files found in {textbooks_dir}")
            return []

        textbooks = []
        for html_file in html_files:
            try:
                content = self.load_file(html_file)
                if content:
                    textbooks.append(content)
            except Exception as e:
                logger.error(f"Failed to load {html_file}: {e}")

        # Log decision capture
        if self.capture:
            self.capture.log_decision(
                decision_type="textbook_integration",
                decision=f"Loaded {len(textbooks)} textbooks from {textbooks_dir}",
                rationale=(
                    f"Files scanned: {len(html_files)}, "
                    f"Successfully loaded: {len(textbooks)}"
                ),
            )

        return textbooks

    def load_file(self, html_file: Path) -> Optional[TextbookContent]:
        """
        Load a single DART-processed HTML file.

        If a .quality.json sidecar file exists (produced by DART's
        multi_source_interpreter), its metadata is attached to the
        TextbookContent so downstream consumers can assess source
        reliability.

        Args:
            html_file: Path to HTML file

        Returns:
            TextbookContent or None if parsing fails
        """
        html_file = Path(html_file)
        if not html_file.exists():
            logger.error(f"File not found: {html_file}")
            return None

        parser = self._get_parser()

        content = None
        if parser:
            # Use Trainforge HTML parser
            try:
                parsed = parser.parse_file(str(html_file))
                content = self._convert_parsed_content(html_file, parsed)
            except Exception as e:
                logger.warning(f"Parser failed for {html_file}: {e}, using basic parsing")

        if content is None:
            # Fallback to basic parsing
            content = self._basic_parse(html_file)

        # Load DART quality report if available
        if content is not None:
            quality_path = html_file.with_suffix('.quality.json')
            if quality_path.exists():
                try:
                    quality_data = json.loads(
                        quality_path.read_text(encoding='utf-8')
                    )
                    content.metadata["dart_quality"] = quality_data
                    content.metadata["dart_confidence"] = quality_data.get(
                        "confidence_score", 0.0
                    )
                    logger.info(
                        "Loaded DART quality report for %s (confidence: %.2f)",
                        html_file.name,
                        quality_data.get("confidence_score", 0.0),
                    )
                except (json.JSONDecodeError, OSError) as e:
                    logger.warning(
                        "Failed to load quality report for %s: %s",
                        html_file.name, e,
                    )

        return content

    def _convert_parsed_content(
        self,
        html_file: Path,
        parsed: Any,
    ) -> TextbookContent:
        """Convert parser output to TextbookContent."""
        sections = []
        for i, section in enumerate(getattr(parsed, 'sections', [])):
            sections.append(TextbookSection(
                section_id=f"sec_{i+1}",
                title=getattr(section, 'title', f'Section {i+1}'),
                content=getattr(section, 'content', ''),
                level=getattr(section, 'level', 2),
                word_count=len(getattr(section, 'content', '').split()),
                has_images=getattr(section, 'has_images', False),
                has_math=getattr(section, 'has_math', False),
            ))

        return TextbookContent(
            path=html_file,
            title=getattr(parsed, 'title', html_file.stem),
            sections=sections,
            learning_objectives=getattr(parsed, 'learning_objectives', []),
            concepts=getattr(parsed, 'concepts', []),
            metadata=getattr(parsed, 'metadata', {}),
        )

    def _basic_parse(self, html_file: Path) -> Optional[TextbookContent]:
        """Basic HTML parsing fallback."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.error("BeautifulSoup not available for basic parsing")
            return None

        with open(html_file, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')

        # Extract title
        title_tag = soup.find('title')
        h1_tag = soup.find('h1')
        title = title_tag.text if title_tag else (h1_tag.text if h1_tag else html_file.stem)

        # Extract sections from headings
        sections = []
        for i, heading in enumerate(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])):
            level = int(heading.name[1])
            content = self._extract_section_content(heading)
            sections.append(TextbookSection(
                section_id=f"sec_{i+1}",
                title=heading.get_text(strip=True),
                content=content,
                level=level,
                word_count=len(content.split()),
                has_images=bool(heading.find_next('img')),
                has_math='math' in content.lower() or 'mathjax' in content.lower(),
            ))

        # Extract learning objectives (common patterns)
        objectives = []
        for pattern in ['learning objective', 'by the end of', 'you will be able to']:
            for elem in soup.find_all(string=lambda s: s and pattern.lower() in s.lower()):
                parent = elem.parent
                if parent:
                    for li in parent.find_all('li'):
                        objectives.append(li.get_text(strip=True))

        return TextbookContent(
            path=html_file,
            title=title,
            sections=sections,
            learning_objectives=objectives,
            concepts=[],  # Would need NLP for concept extraction
            metadata={"source": "basic_parse"},
        )

    def _extract_section_content(self, heading) -> str:
        """Extract content between this heading and the next."""
        content_parts = []
        sibling = heading.next_sibling

        while sibling:
            if sibling.name and sibling.name.startswith('h') and sibling.name[1:].isdigit():
                # Stop at next heading
                break
            if hasattr(sibling, 'get_text'):
                text = sibling.get_text(strip=True)
                if text:
                    content_parts.append(text)
            sibling = sibling.next_sibling

        return ' '.join(content_parts)


def load_textbooks(
    textbooks_dir: Path,
    capture: Optional["DecisionCapture"] = None,
) -> List[TextbookContent]:
    """
    Convenience function to load textbooks.

    Args:
        textbooks_dir: Path to textbooks directory
        capture: Optional DecisionCapture

    Returns:
        List of TextbookContent objects
    """
    loader = TextbookLoader(capture=capture)
    return loader.load_all(textbooks_dir)


# Default textbooks directory
DEFAULT_TEXTBOOKS_DIR = ED4ALL_ROOT / "Courseforge" / "inputs" / "textbooks"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load DART-processed textbooks")
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_TEXTBOOKS_DIR,
        help="Textbooks directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON file",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    textbooks = load_textbooks(args.dir)
    print(f"Loaded {len(textbooks)} textbooks")

    for tb in textbooks:
        print(f"  - {tb.title}: {tb.section_count} sections, {tb.total_word_count} words")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump([tb.to_dict() for tb in textbooks], f, indent=2)
        print(f"Saved to {args.output}")
