"""
Semantic Structure Extractor

Main module that extracts complete semantic structure from HTML or Markdown content.
Combines heading hierarchy parsing and content block classification to produce
a structured representation of content suitable for presentation generation.

Supports:
- HTML input (DART-processed or generic)
- Markdown input with YAML front matter
- Content profiling (difficulty, concepts)
- Concept graph building
- Presentation schema transformation

Output conforms to schemas/presentation/presentation_schema.json or
textbook_structure.schema.json based on extraction method used.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

# TOC-like heading texts that should NOT be promoted to chapter titles
# on their own. DART's converter emits many "Contents" h2s when page
# chrome wraps every printed page; if we hand one of those to the
# course planner we end up with a chapter named "Contents" and every
# real chapter demoted to a section. Case-insensitive exact match.
_TOC_HEADING_TEXTS = frozenset({
    "contents",
    "table of contents",
    "toc",
    "index",
})


def _is_toc_heading(text: Optional[str]) -> bool:
    """Whether a heading text is a table-of-contents artifact."""
    if not text:
        return False
    return text.strip().lower() in _TOC_HEADING_TEXTS

from .analysis.concept_graph import ConceptGraphBuilder
from .analysis.content_profiler import ContentProfiler
from .core.content_block_classifier import (
    BlockType,
    ContentBlock,
    ContentBlockClassifier,
)
from .core.heading_parser import HeadingHierarchy, HeadingNode, HeadingParser

# Import extended modules
from .formats.markdown_parser import MarkdownParser, detect_format
from .transformers.presentation_transformer import PresentationTransformer


@dataclass
class ExtractedProcedure:
    """A step-by-step procedure extracted from content."""
    name: str
    steps: List[str]
    context: str
    chapter_id: str
    section_id: Optional[str] = None


@dataclass
class ExtractedExample:
    """An example or case study extracted from content."""
    title: Optional[str]
    content: str
    related_concept: Optional[str]
    chapter_id: str
    section_id: Optional[str] = None


@dataclass
class ReviewQuestion:
    """A review question extracted from content."""
    question: str
    chapter_id: str
    section_id: Optional[str] = None
    bloom_level: Optional[str] = None


@dataclass
class SectionStructure:
    """Structured representation of a section."""
    id: str
    heading_level: int
    heading_text: str
    heading_id: Optional[str]
    content_blocks: List[ContentBlock]
    subsections: List['SectionStructure'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "headingLevel": self.heading_level,
            "headingText": self.heading_text,
            "headingId": self.heading_id,
            "contentBlocks": [b.to_dict() for b in self.content_blocks],
            "subsections": [s.to_dict() for s in self.subsections]
        }


@dataclass
class ChapterStructure:
    """Structured representation of a chapter."""
    id: str
    heading_level: int
    heading_text: str
    heading_id: Optional[str]
    explicit_objectives: List[Dict[str, str]]
    content_blocks: List[ContentBlock]
    sections: List[SectionStructure]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "headingLevel": self.heading_level,
            "headingText": self.heading_text,
            "headingId": self.heading_id,
            "explicitObjectives": self.explicit_objectives,
            "contentBlocks": [b.to_dict() for b in self.content_blocks],
            "sections": [s.to_dict() for s in self.sections]
        }


class SemanticStructureExtractor:
    """
    Extracts complete semantic structure from DART-processed HTML.

    Uses HeadingParser and ContentBlockClassifier to build a hierarchical
    representation of textbook content suitable for learning objective extraction.
    """

    # Bloom's taxonomy verb patterns for question analysis. These are
    # regex alternations (not plain verb lists), so migrating to
    # schemas/taxonomies/bloom_verbs.json requires a pattern-schema
    # layer. See the canonical tracking TODO at
    # `lib/semantic_structure_extractor/analysis/content_profiler.py`
    # — Wave 28f deduped the TODO to a single site.
    BLOOM_PATTERNS = {
        'remember': [
            r'\b(define|list|recall|identify|name|state|label|match|recognize)\b',
        ],
        'understand': [
            r'\b(explain|describe|summarize|classify|compare|interpret|discuss)\b',
        ],
        'apply': [
            r'\b(demonstrate|implement|solve|use|execute|apply|compute|calculate)\b',
        ],
        'analyze': [
            r'\b(analyze|differentiate|examine|distinguish|organize|compare.*contrast)\b',
        ],
        'evaluate': [
            r'\b(evaluate|assess|critique|justify|judge|argue|defend)\b',
        ],
        'create': [
            r'\b(create|design|construct|develop|formulate|compose|plan)\b',
        ],
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the extractor.

        Args:
            config_path: Optional path to configuration file
        """
        self.heading_parser = HeadingParser()
        self.block_classifier = ContentBlockClassifier()
        self.config = self._load_config(config_path)

        # Initialize new modules
        self.markdown_parser = MarkdownParser(self.config.get('markdown_parsing', {}))
        self.content_profiler = ContentProfiler(self.config)
        self.concept_builder = ConceptGraphBuilder(self.config)
        self.presentation_transformer = PresentationTransformer(self.config)

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "chapter_heading_levels": [1, 2],
            "section_heading_levels": [2, 3, 4],
            "subsection_heading_levels": [4, 5, 6],
            "min_procedure_steps": 2,
            "min_example_words": 20
        }

        if config_path:
            path = Path(config_path)
            if path.exists():
                with open(path) as f:
                    loaded = json.load(f)
                    default_config.update(loaded)

        return default_config

    def extract(self, html_content: str, source_path: str = "") -> Dict[str, Any]:
        """
        Extract semantic structure from HTML content.

        Args:
            html_content: The HTML string to process
            source_path: Path to the source file (for metadata)

        Returns:
            Dictionary conforming to textbook_structure.schema.json
        """
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract heading hierarchy
        hierarchy = self.heading_parser.parse(html_content)

        # Extract document info
        document_info = self._extract_document_info(soup, source_path)

        # Build chapter structure
        chapters = self._build_chapter_structure(soup, hierarchy)

        # Extract concepts
        extracted_concepts = self._extract_all_concepts(chapters)

        # Extract review questions
        review_questions = self._extract_review_questions(soup, chapters)

        return {
            "documentInfo": document_info,
            "tableOfContents": hierarchy.to_toc(),
            "chapters": [ch.to_dict() for ch in chapters],
            "extractedConcepts": extracted_concepts,
            "reviewQuestions": [
                {
                    "question": q.question,
                    "chapterId": q.chapter_id,
                    "sectionId": q.section_id,
                    "bloomLevel": q.bloom_level
                }
                for q in review_questions
            ]
        }

    def extract_file(self, file_path: str, format: str = "auto") -> Dict[str, Any]:
        """
        Extract semantic structure from a file (HTML or Markdown).

        Args:
            file_path: Path to the file
            format: Format hint ("auto", "html", "markdown")

        Returns:
            Dictionary conforming to textbook_structure.schema.json
        """
        path = Path(file_path)
        with open(path, encoding='utf-8') as f:
            content = f.read()

        # Auto-detect format if needed
        if format == "auto":
            if path.suffix.lower() in ['.md', '.markdown']:
                format = "markdown"
            elif path.suffix.lower() in ['.html', '.htm']:
                format = "html"
            else:
                format = detect_format(content)

        return self.extract(content, str(path), format=format)

    def extract(self, content: str, source_path: str = "", format: str = "auto") -> Dict[str, Any]:  # noqa: F811
        """
        Extract semantic structure from content (HTML or Markdown).

        Args:
            content: The content string to process
            source_path: Path to the source file (for metadata)
            format: Format hint ("auto", "html", "markdown")

        Returns:
            Dictionary conforming to textbook_structure.schema.json
        """
        # Auto-detect format
        if format == "auto":
            format = detect_format(content)

        if format == "markdown":
            return self._extract_from_markdown(content, source_path)
        else:
            return self._extract_from_html(content, source_path)

    def _extract_from_markdown(self, content: str, source_path: str = "") -> Dict[str, Any]:
        """Extract semantic structure from Markdown content."""
        doc = self.markdown_parser.parse(content, source_path)
        result = doc.to_dict()

        # Add extraction metadata
        result['documentInfo']['extractionTimestamp'] = datetime.now().isoformat()
        result['documentInfo']['sourcePath'] = source_path
        result['documentInfo']['sourceFormat'] = 'markdown'

        return result

    def _extract_from_html(self, html_content: str, source_path: str = "") -> Dict[str, Any]:
        """Extract semantic structure from HTML content (original method)."""
        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract heading hierarchy
        hierarchy = self.heading_parser.parse(html_content)

        # Extract document info
        document_info = self._extract_document_info(soup, source_path)

        # Build chapter structure
        chapters = self._build_chapter_structure(soup, hierarchy)

        # Extract concepts
        extracted_concepts = self._extract_all_concepts(chapters)

        # Extract review questions
        review_questions = self._extract_review_questions(soup, chapters)

        return {
            "documentInfo": document_info,
            "tableOfContents": hierarchy.to_toc(),
            "chapters": [ch.to_dict() for ch in chapters],
            "extractedConcepts": extracted_concepts,
            "reviewQuestions": [
                {
                    "question": q.question,
                    "chapterId": q.chapter_id,
                    "sectionId": q.section_id,
                    "bloomLevel": q.bloom_level
                }
                for q in review_questions
            ]
        }

    def extract_with_profiling(
        self,
        content: str,
        source_path: str = "",
        format: str = "auto"
    ) -> Dict[str, Any]:
        """
        Extract semantic structure with content profiling.

        Adds difficulty assessment, concept extraction, and concept graph.

        Args:
            content: Content to extract from
            source_path: Source file path
            format: Format hint

        Returns:
            Dictionary with semantic structure plus profiling data
        """
        # Get base extraction
        structure = self.extract(content, source_path, format)

        # Profile content
        profiles = self._profile_all_content(structure)
        structure['contentProfiles'] = profiles

        # Build concept graph
        concept_graph = self.concept_builder.build_graph(structure)
        structure['conceptGraph'] = concept_graph.to_dict()

        # Detect pedagogical pattern
        pattern = self.content_profiler.detect_pedagogical_pattern(
            structure.get('chapters', [])
        )
        structure['pedagogicalPattern'] = pattern.value

        return structure

    def extract_for_presentation(
        self,
        content: str,
        source_path: str = "",
        format: str = "auto"
    ) -> Dict[str, Any]:
        """
        Extract and transform content directly to presentation schema format.

        This is the primary method for the presentation generation pipeline.

        Args:
            content: Content to extract from
            source_path: Source file path
            format: Format hint

        Returns:
            Dictionary conforming to schemas/presentation/presentation_schema.json
        """
        # Get profiled extraction
        structure = self.extract_with_profiling(content, source_path, format)

        # Transform to presentation format
        concept_graph = structure.get('conceptGraph', {})
        presentation = self.presentation_transformer.transform(
            structure,
            concept_graph
        )

        return presentation

    def _profile_all_content(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Profile all content in the structure."""
        profiles = {
            'sections': {},
            'aggregate': None,
            'difficultyDistribution': {
                'beginner': 0,
                'intermediate': 0,
                'advanced': 0
            }
        }

        all_profiles = []

        for chapter in structure.get('chapters', []):
            section_profile = self.content_profiler.profile_section(chapter)
            profiles['sections'][chapter.get('id', '')] = section_profile.to_dict()

            if section_profile.aggregate_profile:
                all_profiles.append(section_profile.aggregate_profile)

                # Track difficulty distribution
                level = section_profile.aggregate_profile.difficulty_level.value
                profiles['difficultyDistribution'][level] = (
                    profiles['difficultyDistribution'].get(level, 0) + 1
                )

        # Create overall aggregate
        if all_profiles:
            profiles['aggregate'] = self.content_profiler._aggregate_profiles(
                all_profiles, 'document'
            ).to_dict()

        return profiles

    def _extract_document_info(self, soup: BeautifulSoup, source_path: str) -> Dict[str, Any]:
        """Extract document metadata."""
        # Get title
        title = ""
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)
        else:
            title_elem = soup.find('title')
            if title_elem:
                title = title_elem.get_text(strip=True)

        # Get metadata from meta tags
        authors = []
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            authors = [author_meta.get('content', '')]

        description = ""
        desc_meta = soup.find('meta', attrs={'name': 'description'})
        if desc_meta:
            description = desc_meta.get('content', '')

        keywords = []
        keywords_meta = soup.find('meta', attrs={'name': 'keywords'})
        if keywords_meta:
            keywords = [k.strip() for k in keywords_meta.get('content', '').split(',')]

        # Determine source format
        source_format = self._detect_source_format(soup)

        return {
            "title": title,
            "sourcePath": source_path,
            "sourceFormat": source_format,
            "extractionTimestamp": datetime.now().isoformat(),
            "metadata": {
                "authors": authors,
                "description": description,
                "keywords": keywords,
                "language": soup.find('html').get('lang', 'en') if soup.find('html') else 'en'
            }
        }

    def _detect_source_format(self, soup: BeautifulSoup) -> str:
        """Detect the source format of the HTML."""
        # Check for DART markers
        # DART adds skip-link, specific ARIA landmarks
        if soup.find('a', class_='skip-link'):
            main = soup.find('main', attrs={'role': 'main'})
            if main:
                return 'dart_html'

        return 'generic_html'

    def _build_chapter_structure(
        self,
        soup: BeautifulSoup,
        hierarchy: HeadingHierarchy
    ) -> List[ChapterStructure]:
        """Build chapter structure from heading hierarchy.

        Wave 19: first look for ``<article role="doc-chapter">`` wrappers
        emitted by the Wave 13+ DART converter. When present, each
        article becomes a chapter with its inner ``<h2>`` as the title
        and inner ``<section>`` wrappers as sections. Falls back to the
        legacy ``<h2>`` grouping heuristic when no doc-chapter articles
        exist (pre-Wave-13 DART HTML, generic third-party HTML).

        Wave 74 Session 3: when both primary paths produce trivial
        output (0 chapters, or chapters that are all TOC artifacts, or
        a single chapter with no sections but the DOM has many
        ``<h2>``/``<h3>`` headings), synthesize chapters from the raw
        heading hierarchy. This handles third-party DART HTML that
        lacks ``<section>`` wrappers and doc-chapter articles but still
        carries a rich heading structure (W3C specs are the canonical
        example).
        """
        # Wave 19 primary path: DPUB-ARIA doc-chapter articles.
        doc_chapter_articles = soup.find_all(
            'article', attrs={'role': 'doc-chapter'}
        )
        primary_chapters: List[ChapterStructure] = []
        if doc_chapter_articles:
            for idx, article in enumerate(doc_chapter_articles, start=1):
                chapter = self._build_chapter_from_article(
                    soup, article, idx
                )
                primary_chapters.append(chapter)

        if not primary_chapters:
            # Legacy heading-hierarchy path.
            chapter_counter = 0

            # Find h1 or h2 headings that represent chapters
            chapter_levels = self.config.get('chapter_heading_levels', [1, 2])

            for root_node in hierarchy.root_nodes:
                # Process h1 as document title, h2s as chapters
                if root_node.level == 1:
                    # Process children of h1 as chapters
                    for child_id in root_node.children:
                        child_node = hierarchy.get_node(child_id)
                        if child_node and child_node.level in chapter_levels:
                            chapter_counter += 1
                            chapter = self._build_chapter(
                                soup, hierarchy, child_node, chapter_counter
                            )
                            primary_chapters.append(chapter)
                elif root_node.level in chapter_levels:
                    chapter_counter += 1
                    chapter = self._build_chapter(
                        soup, hierarchy, root_node, chapter_counter
                    )
                    primary_chapters.append(chapter)

        # Wave 74 Session 3: heading-hierarchy fallback.
        # Fires when the primary paths degenerate to trivial output but
        # the DOM still carries meaningful h2/h3 structure.
        if self._primary_output_is_trivial(soup, primary_chapters):
            fallback = self._build_chapters_from_headings(soup)
            if fallback:
                source_path = self._document_source_hint(soup)
                logger.warning(
                    "SemanticStructureExtractor: primary extraction paths "
                    "produced trivial output (%d chapter(s)); falling back "
                    "to heading-hierarchy synthesis and emitted %d "
                    "chapter(s). source=%s",
                    len(primary_chapters),
                    len(fallback),
                    source_path or "<inline>",
                )
                return fallback

        return primary_chapters

    # ------------------------------------------------------------------
    # Wave 74 Session 3: heading-hierarchy fallback
    # ------------------------------------------------------------------

    def _primary_output_is_trivial(
        self,
        soup: BeautifulSoup,
        chapters: List[ChapterStructure],
    ) -> bool:
        """Whether the primary extraction paths produced trivial output.

        Trivial means one of:

        * Zero chapters.
        * All chapter titles are TOC artifacts (``Contents``, ``Index``,
          etc.) — the extractor caught the TOC h2 and missed the real
          chapter headings that follow it as siblings.
        * Zero chapters with non-empty sections AND the raw DOM carries
          at least three h2/h3 headings that aren't TOC artifacts.
          This covers specs like rdf11-primer (1 TOC h2, 13 real h3s)
          and the W3C family more broadly.
        """
        if not chapters:
            return True

        non_toc_chapters = [
            c for c in chapters if not _is_toc_heading(c.heading_text)
        ]
        if not non_toc_chapters:
            return True

        chapters_with_sections = [
            c for c in chapters if c.sections
        ]
        if chapters_with_sections:
            return False

        # Count real (non-TOC) h2/h3 headings in the DOM. If there's
        # a genuine hierarchy lurking, the primary paths missed it.
        real_heading_count = 0
        for tag in soup.find_all(['h2', 'h3']):
            text = tag.get_text(strip=True)
            if text and not _is_toc_heading(text):
                real_heading_count += 1
                if real_heading_count >= 3:
                    return True
        return False

    def _document_source_hint(self, soup: BeautifulSoup) -> Optional[str]:
        """Best-effort source identifier for log messages."""
        title = soup.find('title')
        if title:
            text = title.get_text(strip=True)
            if text:
                return text
        h1 = soup.find('h1')
        if h1:
            text = h1.get_text(strip=True)
            if text:
                return text
        return None

    def _build_chapters_from_headings(
        self,
        soup: BeautifulSoup,
    ) -> List[ChapterStructure]:
        """Synthesize chapter/section hierarchy from raw heading levels.

        Strategy:

        1. Walk every ``h1``..``h6`` in document order inside ``<main>``
           (falling back to ``<body>`` then the whole soup).
        2. Drop TOC artifacts (``Contents``, ``Table of Contents``).
        3. Pick the "chapter level" as the shallowest heading level
           that has at least two non-TOC entries. If the only real
           heading level is h3 (e.g., rdf11-primer), h3s become
           chapters; if h2 and h3 both exist with real content, h2s
           become chapters and h3s become sections.
        4. Content blocks between two consecutive headings attach to
           the most recent open heading's chapter/section.
        5. ``data-dart-*`` attributes on individual content elements
           carry through via ``ContentBlockClassifier._classify_element``.
        """
        container = soup.find('main') or soup.find('body') or soup
        if container is None:
            return []

        # Collect every heading in document order, filtering TOC noise.
        all_headings: List[Tag] = []
        for tag in container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            text = tag.get_text(strip=True)
            if not text:
                continue
            if _is_toc_heading(text):
                continue
            all_headings.append(tag)

        if not all_headings:
            return []

        # Figure out which level acts as chapter vs section.
        level_counts: Dict[int, int] = {}
        for tag in all_headings:
            try:
                lv = int(tag.name.lstrip('h'))
            except ValueError:
                continue
            level_counts[lv] = level_counts.get(lv, 0) + 1

        # Chapter level: shallowest heading level with >= 1 entry,
        # preferring levels with multiple entries when present. h1 is
        # skipped as chapter-level when it appears exactly once (it's
        # the document title).
        sorted_levels = sorted(level_counts.keys())
        chapter_level: Optional[int] = None
        skip_solo_h1 = False
        for lv in sorted_levels:
            if lv == 1 and level_counts[lv] < 2:
                # Treat a solo h1 as the document title, not a chapter.
                skip_solo_h1 = True
                continue
            chapter_level = lv
            break
        if chapter_level is None:
            # Only a single h1 exists — promote it to a chapter anyway
            # so we at least emit one meaningful entry.
            chapter_level = sorted_levels[0]
            skip_solo_h1 = False

        section_level = chapter_level + 1
        subsection_level = chapter_level + 2

        # Walk the full descendants stream of the container. Maintain
        # a "current chapter / section / subsection" pointer and attach
        # any non-heading ContentBlock-yielding element to the deepest
        # open target.
        chapters: List[ChapterStructure] = []
        current_chapter: Optional[ChapterStructure] = None
        current_section: Optional[SectionStructure] = None
        current_subsection: Optional[SectionStructure] = None
        chapter_counter = 0
        section_counter = 0
        subsection_counter = 0
        classifier = ContentBlockClassifier()
        heading_set = set(id(h) for h in all_headings)

        # Track elements we've already processed to avoid double-counting
        # when a parent tag emits both itself and its children through
        # the descendant iterator.
        consumed: set = set()

        def _walk(node: Tag) -> None:
            nonlocal current_chapter, current_section, current_subsection
            nonlocal chapter_counter, section_counter, subsection_counter

            for child in node.children:
                if not isinstance(child, Tag):
                    continue
                if id(child) in consumed:
                    continue
                name = child.name.lower() if child.name else ''

                # Heading — open a new chapter/section/subsection.
                if name in ('h1', 'h2', 'h3', 'h4', 'h5', 'h6'):
                    if id(child) not in heading_set:
                        # Filtered (TOC artifact) — ignore.
                        continue
                    try:
                        lv = int(name.lstrip('h'))
                    except ValueError:
                        continue
                    heading_text = child.get_text(strip=True)
                    heading_id = child.get('id')

                    # Skip a solo h1 that's serving as the document title
                    # when chapter_level is deeper (e.g., chapter_level==3
                    # because h2 only had TOC entries).
                    if skip_solo_h1 and lv == 1:
                        consumed.add(id(child))
                        continue

                    if lv <= chapter_level:
                        chapter_counter += 1
                        current_chapter = ChapterStructure(
                            id=f"ch{chapter_counter}",
                            heading_level=lv,
                            heading_text=heading_text,
                            heading_id=heading_id,
                            explicit_objectives=[],
                            content_blocks=[],
                            sections=[],
                        )
                        chapters.append(current_chapter)
                        current_section = None
                        current_subsection = None
                        section_counter = 0
                        subsection_counter = 0
                    elif lv == section_level:
                        # Ensure there's a chapter to attach to.
                        if current_chapter is None:
                            chapter_counter += 1
                            current_chapter = ChapterStructure(
                                id=f"ch{chapter_counter}",
                                heading_level=chapter_level,
                                heading_text=heading_text,
                                heading_id=heading_id,
                                explicit_objectives=[],
                                content_blocks=[],
                                sections=[],
                            )
                            chapters.append(current_chapter)
                        section_counter += 1
                        current_section = SectionStructure(
                            id=f"{current_chapter.id}_s{section_counter}",
                            heading_level=lv,
                            heading_text=heading_text,
                            heading_id=heading_id,
                            content_blocks=[],
                            subsections=[],
                        )
                        current_chapter.sections.append(current_section)
                        current_subsection = None
                        subsection_counter = 0
                    elif lv >= subsection_level:
                        # Ensure a section exists; synthesize if needed.
                        if current_chapter is None:
                            chapter_counter += 1
                            current_chapter = ChapterStructure(
                                id=f"ch{chapter_counter}",
                                heading_level=chapter_level,
                                heading_text=heading_text,
                                heading_id=heading_id,
                                explicit_objectives=[],
                                content_blocks=[],
                                sections=[],
                            )
                            chapters.append(current_chapter)
                        if current_section is None:
                            section_counter += 1
                            current_section = SectionStructure(
                                id=f"{current_chapter.id}_s{section_counter}",
                                heading_level=section_level,
                                heading_text=heading_text,
                                heading_id=None,
                                content_blocks=[],
                                subsections=[],
                            )
                            current_chapter.sections.append(current_section)
                        subsection_counter += 1
                        current_subsection = SectionStructure(
                            id=(
                                f"{current_section.id}_sub{subsection_counter}"
                            ),
                            heading_level=lv,
                            heading_text=heading_text,
                            heading_id=heading_id,
                            content_blocks=[],
                            subsections=[],
                        )
                        current_section.subsections.append(current_subsection)
                    # Mark the heading as consumed — we don't want to
                    # reclassify it as a ContentBlock.
                    consumed.add(id(child))
                    continue

                # Non-heading leaf-like element — try to classify as a
                # content block and attach to the deepest open target.
                if name in (
                    'p', 'ul', 'ol', 'dl', 'pre', 'code', 'blockquote',
                    'table', 'figure', 'img', 'aside', 'div',
                ):
                    # Skip elements that contain nested headings — we
                    # want to recurse into them so the headings land in
                    # the right chapter/section.
                    has_nested_heading = any(
                        id(h) in heading_set
                        for h in child.find_all(
                            ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
                        )
                    )
                    if has_nested_heading:
                        _walk(child)
                        continue

                    block = classifier._classify_element(child)
                    if block is None:
                        continue
                    if current_subsection is not None:
                        current_subsection.content_blocks.append(block)
                    elif current_section is not None:
                        current_section.content_blocks.append(block)
                    elif current_chapter is not None:
                        current_chapter.content_blocks.append(block)
                    consumed.add(id(child))
                    continue

                # Structural wrappers (section/article/header/nav/main/
                # body) — recurse so we find nested headings.
                if name in (
                    'section', 'article', 'header', 'footer', 'nav',
                    'main', 'body', 'html', 'div',
                ):
                    _walk(child)

        _walk(container)

        # Drop any chapter whose only heading_text is a TOC artifact
        # AND that has no sections / content — belt and braces.
        chapters = [
            c for c in chapters
            if not (
                _is_toc_heading(c.heading_text)
                and not c.sections
                and not c.content_blocks
            )
        ]

        return chapters

    def _build_chapter_from_article(
        self,
        soup: BeautifulSoup,
        article: Tag,
        chapter_num: int,
    ) -> ChapterStructure:
        """Build a chapter from a ``<article role="doc-chapter">`` wrapper.

        Wave 19: DART's Wave 13+ converter emits every chapter as a
        standalone article with the chapter heading inside a ``<header>``
        block. We prefer the ``id`` attribute on the article itself
        (``chap-{N}``) for the chapter id; falling back to a synthesized
        ``ch{N}`` identifier when the article lacks an explicit id.
        """
        chapter_id = str(article.get('id') or f'ch{chapter_num}').strip()

        # Title: the first <h2> or <h1> inside the article (Wave 13 uses h2).
        heading_tag = article.find(['h1', 'h2'])
        heading_text = None
        heading_id = None
        heading_level = 2
        if heading_tag:
            heading_text = heading_tag.get_text(strip=True) or None
            heading_id = heading_tag.get('id')
            try:
                heading_level = int(heading_tag.name.lstrip('h'))
            except ValueError:
                heading_level = 2
        if not heading_text:
            heading_text = article.get('aria-label') or f'Chapter {chapter_num}'

        # Explicit objectives: reuse the existing helper on the article.
        explicit_objectives = self._extract_explicit_objectives(article)

        # Content blocks that appear directly in the article, before any
        # nested <section>. Treat the article like a chapter's own
        # section_elem for _extract_chapter_content.
        class _ArticleLike:
            """Duck-typed shim so ``_extract_chapter_content`` walks the
            article exactly like a ``<section>`` root.
            """
            def __init__(self, elem):
                self._elem = elem

            @property
            def children(self):
                return self._elem.children

        content_blocks = self._extract_chapter_content(
            _ArticleLike(article), None
        )

        # Build sections from every top-level <section> child inside the
        # article. The heading hierarchy isn't consulted here — Wave 13's
        # chapter article wraps its own section tree, so we walk the DOM
        # directly.
        sections: List[SectionStructure] = []
        sec_counter = 0
        for child in article.find_all('section', recursive=False):
            sec_counter += 1
            sections.append(
                self._build_section_from_element(
                    soup, child, chapter_id, sec_counter,
                )
            )
        # When sections don't live as direct children (common — Wave 13
        # emits the chapter article and lets the assembler sibling the
        # section blocks), also pull any <section> following the article
        # until the next <article role="doc-chapter"> or document end.
        if not sections:
            sibling = article.next_sibling
            while sibling is not None:
                if isinstance(sibling, Tag):
                    if (
                        sibling.name == 'article'
                        and sibling.get('role') == 'doc-chapter'
                    ):
                        break
                    if sibling.name == 'section':
                        sec_counter += 1
                        sections.append(
                            self._build_section_from_element(
                                soup, sibling, chapter_id, sec_counter,
                            )
                        )
                sibling = sibling.next_sibling

        return ChapterStructure(
            id=chapter_id,
            heading_level=heading_level,
            heading_text=heading_text,
            heading_id=heading_id,
            explicit_objectives=explicit_objectives,
            content_blocks=content_blocks,
            sections=sections,
        )

    def _build_section_from_element(
        self,
        soup: BeautifulSoup,
        section_elem: Tag,
        parent_id: str,
        section_num: int,
    ) -> SectionStructure:
        """Build a ``SectionStructure`` directly from a DOM ``<section>``.

        Wave 19 DART output emits flat ``<section>`` wrappers rather
        than nesting them under article children, so we read heading
        info off the section itself.
        """
        section_id = f"{parent_id}_s{section_num}"
        heading_tag = section_elem.find(
            ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
        )
        heading_text = (
            heading_tag.get_text(strip=True) if heading_tag else ''
        ) or ''
        heading_id = heading_tag.get('id') if heading_tag else None
        try:
            heading_level = int(heading_tag.name.lstrip('h')) if heading_tag else 3
        except ValueError:
            heading_level = 3

        content_blocks = self.block_classifier.classify_section(section_elem)
        # Nested <section> children become subsections.
        subsections: List[SectionStructure] = []
        sub_counter = 0
        for nested in section_elem.find_all('section', recursive=False):
            sub_counter += 1
            subsections.append(
                self._build_section_from_element(
                    soup, nested, section_id, sub_counter,
                )
            )

        return SectionStructure(
            id=section_id,
            heading_level=heading_level,
            heading_text=heading_text,
            heading_id=heading_id,
            content_blocks=content_blocks,
            subsections=subsections,
        )

    def _build_chapter(
        self,
        soup: BeautifulSoup,
        hierarchy: HeadingHierarchy,
        node: HeadingNode,
        chapter_num: int
    ) -> ChapterStructure:
        """Build a single chapter structure."""
        chapter_id = f"ch{chapter_num}"

        # Get the section element for this heading
        section_elem = node.section_element
        if not section_elem and node.element_id:
            heading = soup.find(id=node.element_id)
            if heading:
                section_elem = heading.find_parent('section')

        # Extract explicit objectives if present
        explicit_objectives = self._extract_explicit_objectives(section_elem)

        # Extract content blocks for this chapter (before subsections)
        content_blocks = self._extract_chapter_content(section_elem, node)

        # Build section structure for children
        sections = []
        section_counter = 0
        for child_id in node.children:
            child_node = hierarchy.get_node(child_id)
            if child_node:
                section_counter += 1
                section = self._build_section(
                    soup, hierarchy, child_node,
                    chapter_id, section_counter
                )
                sections.append(section)

        return ChapterStructure(
            id=chapter_id,
            heading_level=node.level,
            heading_text=node.text,
            heading_id=node.element_id,
            explicit_objectives=explicit_objectives,
            content_blocks=content_blocks,
            sections=sections
        )

    def _build_section(
        self,
        soup: BeautifulSoup,
        hierarchy: HeadingHierarchy,
        node: HeadingNode,
        parent_id: str,
        section_num: int
    ) -> SectionStructure:
        """Build a section structure."""
        section_id = f"{parent_id}_s{section_num}"

        # Get section element
        section_elem = node.section_element
        if not section_elem and node.element_id:
            heading = soup.find(id=node.element_id)
            if heading:
                section_elem = heading.find_parent('section')

        # Extract content blocks
        content_blocks = []
        if section_elem:
            content_blocks = self.block_classifier.classify_section(section_elem)

        # Build subsections
        subsections = []
        subsection_counter = 0
        for child_id in node.children:
            child_node = hierarchy.get_node(child_id)
            if child_node:
                subsection_counter += 1
                subsection = self._build_section(
                    soup, hierarchy, child_node,
                    section_id, subsection_counter
                )
                subsections.append(subsection)

        return SectionStructure(
            id=section_id,
            heading_level=node.level,
            heading_text=node.text,
            heading_id=node.element_id,
            content_blocks=content_blocks,
            subsections=subsections
        )

    def _extract_chapter_content(
        self,
        section_elem: Optional[Tag],
        node: HeadingNode
    ) -> List[ContentBlock]:
        """Extract content blocks that belong directly to a chapter (not in subsections)."""
        if not section_elem:
            return []

        # Find content that appears before the first subsection
        content_blocks = []
        classifier = ContentBlockClassifier()

        for child in section_elem.children:
            if isinstance(child, Tag):
                # Stop at subsections
                if child.name == 'section':
                    break

                # Skip the heading itself
                if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    continue

                block = classifier._classify_element(child)
                if block:
                    content_blocks.append(block)

        return content_blocks

    def _extract_explicit_objectives(self, section_elem: Optional[Tag]) -> List[Dict[str, str]]:
        """Extract explicitly stated learning objectives from a section."""
        if not section_elem:
            return []

        objectives = []

        # Look for objectives section
        objectives_section = section_elem.find(
            'section',
            attrs={'aria-labelledby': lambda x: x and 'objective' in x.lower()}
        )

        if not objectives_section:
            # Look for heading with "objectives" or "learning objectives"
            for heading in section_elem.find_all(['h2', 'h3', 'h4']):
                if 'objective' in heading.get_text().lower():
                    objectives_section = heading.find_parent('section') or heading.parent
                    break

        if objectives_section:
            # Find the list of objectives
            obj_list = objectives_section.find(['ul', 'ol'])
            if obj_list:
                for li in obj_list.find_all('li'):
                    objectives.append({
                        "text": li.get_text(strip=True),
                        "source": "objectives_section"
                    })
        else:
            # Look for patterns like "After completing this chapter, you will be able to:"
            text = section_elem.get_text()
            patterns = [
                r'(?:After|Upon|By the end)[^:]+:\s*([^.]+\.(?:\s*[^.]+\.)*)',
                r'(?:you will be able to|students will|learners will)[^:]*:\s*([^.]+\.(?:\s*[^.]+\.)*)',
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    # Split by common delimiters
                    obj_text = match.group(1)
                    for obj in re.split(r'[;•\n]', obj_text):
                        obj = obj.strip()
                        if obj and len(obj) > 10:
                            objectives.append({
                                "text": obj,
                                "source": "inline"
                            })

        return objectives

    def _extract_all_concepts(self, chapters: List[ChapterStructure]) -> Dict[str, Any]:
        """Extract all concepts from chapters."""
        all_definitions = []
        all_key_terms = []
        all_procedures = []
        all_examples = []

        for chapter in chapters:
            # Extract from chapter content blocks
            self._extract_concepts_from_blocks(
                chapter.content_blocks,
                chapter.id,
                None,
                all_definitions,
                all_key_terms,
                all_procedures,
                all_examples
            )

            # Extract from sections
            for section in chapter.sections:
                self._extract_concepts_from_section(
                    section,
                    chapter.id,
                    all_definitions,
                    all_key_terms,
                    all_procedures,
                    all_examples
                )

        return {
            "definitions": all_definitions,
            "keyTerms": all_key_terms,
            "procedures": all_procedures,
            "examples": all_examples
        }

    def _extract_concepts_from_section(
        self,
        section: SectionStructure,
        chapter_id: str,
        all_definitions: List,
        all_key_terms: List,
        all_procedures: List,
        all_examples: List
    ) -> None:
        """Recursively extract concepts from a section."""
        self._extract_concepts_from_blocks(
            section.content_blocks,
            chapter_id,
            section.id,
            all_definitions,
            all_key_terms,
            all_procedures,
            all_examples
        )

        for subsection in section.subsections:
            self._extract_concepts_from_section(
                subsection,
                chapter_id,
                all_definitions,
                all_key_terms,
                all_procedures,
                all_examples
            )

    def _extract_concepts_from_blocks(
        self,
        blocks: List[ContentBlock],
        chapter_id: str,
        section_id: Optional[str],
        all_definitions: List,
        all_key_terms: List,
        all_procedures: List,
        all_examples: List
    ) -> None:
        """Extract concepts from a list of content blocks."""
        for block in blocks:
            # Add definitions
            for defn in block.definitions:
                all_definitions.append({
                    "term": defn.term,
                    "definition": defn.definition,
                    "sourceType": defn.source_type,
                    "chapterId": chapter_id,
                    "sectionId": section_id
                })

            # Add key terms
            for term in block.key_terms:
                all_key_terms.append({
                    "term": term.term,
                    "context": term.context,
                    "emphasisType": term.emphasis_type,
                    "chapterId": chapter_id,
                    "sectionId": section_id
                })

            # Check for procedures (ordered lists with multiple steps)
            if block.block_type == BlockType.LIST_ORDERED:
                min_steps = self.config.get('min_procedure_steps', 2)
                if len(block.list_items) >= min_steps:
                    # Check if it looks like a procedure
                    if self._looks_like_procedure(block.list_items):
                        all_procedures.append({
                            "name": self._infer_procedure_name(block),
                            "steps": block.list_items,
                            "context": "",
                            "chapterId": chapter_id,
                            "sectionId": section_id
                        })

            # Check for examples
            if block.block_type == BlockType.EXAMPLE:
                min_words = self.config.get('min_example_words', 20)
                if block.word_count >= min_words:
                    all_examples.append({
                        "title": None,
                        "content": block.content,
                        "relatedConcept": None,
                        "chapterId": chapter_id,
                        "sectionId": section_id
                    })

    def _looks_like_procedure(self, items: List[str]) -> bool:
        """Determine if a list looks like a procedure."""
        # Check for action verbs at start of items
        action_patterns = [
            r'^(click|select|enter|type|open|close|save|create|delete|configure|set|add|remove)',
            r'^(first|next|then|finally|after|before)',
            r'^\d+[.)]\s*',
        ]

        action_count = 0
        for item in items:
            for pattern in action_patterns:
                if re.match(pattern, item.lower()):
                    action_count += 1
                    break

        return action_count >= len(items) / 2

    def _infer_procedure_name(self, block: ContentBlock) -> str:
        """Infer a name for a procedure from its context."""
        # Try to find a preceding heading or strong text
        return "Procedure"

    def _extract_review_questions(
        self,
        soup: BeautifulSoup,
        chapters: List[ChapterStructure]
    ) -> List[ReviewQuestion]:
        """Extract review questions from the document."""
        questions = []

        # Look for review sections
        review_sections = soup.find_all(
            'section',
            attrs={'aria-labelledby': lambda x: x and any(
                term in x.lower() for term in ['review', 'question', 'quiz', 'assessment']
            )}
        )

        for review_section in review_sections:
            # Find the parent chapter
            chapter_id = self._find_parent_chapter_id(review_section, chapters)

            # Extract questions from ordered list
            for ol in review_section.find_all('ol'):
                for li in ol.find_all('li'):
                    question_text = li.get_text(strip=True)
                    bloom_level = self._infer_bloom_level(question_text)

                    questions.append(ReviewQuestion(
                        question=question_text,
                        chapter_id=chapter_id,
                        section_id=None,
                        bloom_level=bloom_level
                    ))

        return questions

    def _find_parent_chapter_id(
        self,
        element: Tag,
        chapters: List[ChapterStructure]
    ) -> str:
        """Find the chapter ID that contains an element."""
        # Simple heuristic: find the nearest h2 ancestor
        parent = element
        while parent:
            h2 = parent.find_previous('h2')
            if h2:
                h2_text = h2.get_text(strip=True).lower()
                for chapter in chapters:
                    if chapter.heading_text.lower() in h2_text or h2_text in chapter.heading_text.lower():
                        return chapter.id
            parent = parent.parent

        return chapters[0].id if chapters else "ch1"

    def _infer_bloom_level(self, question_text: str) -> Optional[str]:
        """Infer Bloom's taxonomy level from question text."""
        question_lower = question_text.lower()

        for level, patterns in self.BLOOM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    return level

        return None


def extract_textbook_structure(file_path: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract textbook structure from a file.

    Args:
        file_path: Path to the HTML or Markdown file
        config_path: Optional path to configuration file

    Returns:
        Dictionary conforming to textbook_structure.schema.json
    """
    extractor = SemanticStructureExtractor(config_path)
    return extractor.extract_file(file_path)


def extract_for_presentation(file_path: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to extract and transform to presentation format.

    Args:
        file_path: Path to the HTML or Markdown file
        config_path: Optional path to configuration file

    Returns:
        Dictionary conforming to presentation_schema.json
    """
    extractor = SemanticStructureExtractor(config_path)
    path = Path(file_path)
    with open(path, encoding='utf-8') as f:
        content = f.read()
    return extractor.extract_for_presentation(content, str(path))


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract semantic structure from HTML or Markdown content'
    )
    parser.add_argument('input_file', help='Path to the HTML or Markdown file')
    parser.add_argument(
        '-c', '--config',
        help='Path to configuration file',
        default=None
    )
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: stdout)',
        default=None
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output'
    )
    parser.add_argument(
        '-f', '--format',
        choices=['auto', 'html', 'markdown'],
        default='auto',
        help='Input format (default: auto-detect)'
    )
    parser.add_argument(
        '-m', '--mode',
        choices=['basic', 'profiled', 'presentation'],
        default='basic',
        help='Extraction mode: basic, profiled (with concept graph), or presentation (full transform)'
    )

    args = parser.parse_args()

    extractor = SemanticStructureExtractor(args.config)

    # Read input file
    path = Path(args.input_file)
    with open(path, encoding='utf-8') as f:
        content = f.read()

    # Extract based on mode
    if args.mode == 'presentation':
        result = extractor.extract_for_presentation(content, str(path), args.format)
    elif args.mode == 'profiled':
        result = extractor.extract_with_profiling(content, str(path), args.format)
    else:
        result = extractor.extract(content, str(path), args.format)

    # Output
    indent = 2 if args.pretty else None
    output = json.dumps(result, indent=indent, ensure_ascii=False)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Output written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
