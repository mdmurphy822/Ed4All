"""
Presentation Transformer for Semantic Structure Extraction

Transforms semantic extraction output into the Slideforge presentation schema
format, ready for PPTX generation. Implements:
- Block-to-slide type mapping
- 6x6 rule enforcement with content splitting
- Speaker notes generation
- Concept-based importance scoring
"""

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum


class SlideType(Enum):
    """Available slide types in presentation schema."""
    TITLE = "title"
    SECTION_HEADER = "section_header"
    CONTENT = "content"
    TWO_CONTENT = "two_content"
    COMPARISON = "comparison"
    IMAGE = "image"
    QUOTE = "quote"
    BLANK = "blank"
    TABLE = "table"
    PROCESS_FLOW = "process_flow"
    KEY_VALUE = "key_value"


@dataclass
class SlideCandidate:
    """A candidate slide generated from content."""
    source_section_id: str
    source_block_ids: List[str]
    slide_type: SlideType
    title: str
    content: Dict[str, Any]
    notes: str = ""
    importance_score: float = 0.5
    concepts: List[str] = field(default_factory=list)
    difficulty: str = "intermediate"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to presentation schema slide format."""
        result = {
            "type": self.slide_type.value,
            "title": self.title,
            "content": self.content,
        }
        if self.notes:
            result["notes"] = self.notes
        return result


@dataclass
class ProvenanceEntry:
    """Tracks source-to-output lineage."""
    source_id: str
    source_type: str  # chapter, section, block
    source_path: str  # JSONPath-like reference
    target_slide_index: int
    transformation: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sourceId": self.source_id,
            "sourceType": self.source_type,
            "sourcePath": self.source_path,
            "targetSlideIndex": self.target_slide_index,
            "transformation": self.transformation,
            "timestamp": self.timestamp
        }


class PresentationTransformer:
    """
    Transforms semantic structure into presentation schema format.

    Handles the conversion of chapters/sections/blocks into slides,
    applying 6x6 rule, generating speaker notes, and tracking provenance.
    """

    # Block type to slide type mapping
    BLOCK_TO_SLIDE_MAPPING: Dict[str, SlideType] = {
        'paragraph': SlideType.CONTENT,
        'list_ordered': SlideType.CONTENT,
        'list_unordered': SlideType.CONTENT,
        'definition_list': SlideType.TWO_CONTENT,
        'blockquote': SlideType.QUOTE,
        'table': SlideType.TABLE,
        'figure': SlideType.IMAGE,
        'code_block': SlideType.CONTENT,
        'callout_info': SlideType.CONTENT,
        'callout_warning': SlideType.CONTENT,
        'callout_tip': SlideType.CONTENT,
        'example': SlideType.CONTENT,
        'objectives': SlideType.CONTENT,
        'summary': SlideType.CONTENT,
        'heading': SlideType.SECTION_HEADER,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the transformer.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._load_default_config()
        self.provenance: List[ProvenanceEntry] = []
        self._slide_counter = 0

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        config_path = Path(__file__).parent / "config" / "extractor_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def transform(
        self,
        semantic_structure: Dict[str, Any],
        concept_graph: Optional[Dict[str, Any]] = None,
        content_profiles: Optional[Dict[str, Any]] = None,
        rag_enhancer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Transform semantic structure to presentation schema format.

        Args:
            semantic_structure: Output from SemanticStructureExtractor
            concept_graph: Optional concept graph for importance scoring
            content_profiles: Optional content profiles for difficulty info
            rag_enhancer: Optional RAGNotesEnhancer for speaker notes enhancement

        Returns:
            Presentation JSON matching schemas/presentation/presentation_schema.json
        """
        self.provenance = []
        self._slide_counter = 0

        # Extract importance rankings from concept graph
        importance_map = {}
        if concept_graph:
            for concept_data in concept_graph.get('topConcepts', []):
                term = concept_data.get('term', '').lower()
                importance_map[term] = concept_data.get('score', 0.5)

        # Build presentation structure
        presentation = {
            "metadata": self._transform_metadata(semantic_structure),
            "sections": []
        }

        # Transform chapters to sections
        for chapter_idx, chapter in enumerate(semantic_structure.get('chapters', [])):
            section = self._transform_chapter(chapter, importance_map, chapter_idx)
            if section['slides']:  # Only add non-empty sections
                presentation['sections'].append(section)

        # RAG Enhancement: Enhance speaker notes with UDL/accessibility guidance
        if rag_enhancer and hasattr(rag_enhancer, 'enhance_slide'):
            presentation = self._apply_rag_enhancement(presentation, rag_enhancer)

        # Add provenance to presentation (for debugging/tracking)
        presentation['_provenance'] = [p.to_dict() for p in self.provenance]

        return presentation

    def _apply_rag_enhancement(
        self,
        presentation: Dict[str, Any],
        rag_enhancer: Any
    ) -> Dict[str, Any]:
        """
        Apply RAG enhancement to all slides in the presentation.

        Args:
            presentation: The presentation dictionary
            rag_enhancer: RAGNotesEnhancer instance

        Returns:
            Enhanced presentation with RAG-enhanced speaker notes
        """
        for section in presentation.get('sections', []):
            for slide in section.get('slides', []):
                slide_type = slide.get('type', 'content')
                title = slide.get('title', '')
                content = slide.get('content', {})
                original_notes = slide.get('notes', '')

                # Extract content items for context
                content_items = []
                if isinstance(content, dict):
                    if 'bullets' in content:
                        content_items = content['bullets']
                    elif 'items' in content:
                        content_items = content['items']
                    elif 'text' in content:
                        content_items = [content['text']]
                elif isinstance(content, list):
                    content_items = content

                # Check for images and tables
                has_images = slide_type == 'image' or 'image' in str(content).lower()
                has_tables = slide_type == 'table' or 'table' in str(content).lower()

                try:
                    enhanced = rag_enhancer.enhance_slide(
                        slide_type=slide_type,
                        title=title,
                        content=content_items,
                        original_notes=original_notes,
                        has_images=has_images,
                        has_tables=has_tables,
                    )

                    # Update slide with enhanced notes
                    slide['notes'] = enhanced.to_formatted_string()
                    slide['_rag_sources'] = enhanced.sources

                except Exception as e:
                    # Log error but don't fail the transformation
                    import logging
                    logging.getLogger(__name__).warning(
                        f"RAG enhancement failed for slide '{title}': {e}"
                    )

        return presentation

    def _transform_metadata(self, semantic_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Transform document info to presentation metadata."""
        doc_info = semantic_structure.get('documentInfo', {})
        metadata = doc_info.get('metadata', {})

        return {
            "title": doc_info.get('title', '') or metadata.get('title', 'Untitled Presentation'),
            "subtitle": metadata.get('subtitle', ''),
            "author": self._get_first_author(metadata.get('authors', [])),
            "date": doc_info.get('extractionTimestamp', '')[:10] if doc_info.get('extractionTimestamp') else '',
            "subject": metadata.get('description', ''),
            "keywords": metadata.get('keywords', [])
        }

    def _get_first_author(self, authors: Any) -> str:
        """Extract first author from various formats."""
        if isinstance(authors, list) and authors:
            return str(authors[0])
        elif isinstance(authors, str):
            return authors
        return ''

    def _transform_chapter(
        self,
        chapter: Dict[str, Any],
        importance_map: Dict[str, float],
        chapter_idx: int
    ) -> Dict[str, Any]:
        """
        Transform a chapter to a presentation section.

        Args:
            chapter: Chapter dictionary from semantic structure
            importance_map: Concept importance scores
            chapter_idx: Chapter index for provenance

        Returns:
            Section dictionary for presentation
        """
        section = {
            "title": chapter.get('headingText', ''),
            "slides": []
        }

        chapter_id = chapter.get('id', f'chapter_{chapter_idx}')

        # Add section header slide
        header_slide = self._create_section_header(chapter, chapter_id, chapter_idx)
        section['slides'].append(header_slide)

        # Transform content blocks
        for block_idx, block in enumerate(chapter.get('contentBlocks', [])):
            slides = self._transform_block(
                block, chapter_id, chapter_idx, block_idx, importance_map
            )
            section['slides'].extend(slides)

        # Transform nested sections
        for section_idx, subsection in enumerate(chapter.get('sections', [])):
            subsection_slides = self._transform_section(
                subsection, chapter_id, importance_map,
                f"{chapter_idx}.{section_idx}"
            )
            section['slides'].extend(subsection_slides)

        return section

    def _transform_section(
        self,
        section: Dict[str, Any],
        parent_id: str,
        importance_map: Dict[str, float],
        path_prefix: str
    ) -> List[Dict[str, Any]]:
        """Transform a section to slides."""
        slides = []
        section_id = section.get('id', f'section_{path_prefix}')

        # Add section header if it's a significant section (level 2-3)
        level = section.get('headingLevel', 3)
        if level <= 3:
            header_slide = {
                "type": "section_header",
                "title": section.get('headingText', ''),
                "content": {"subtitle": ""},
                "notes": f"Section: {section.get('headingText', '')}"
            }
            slides.append(header_slide)
            self._slide_counter += 1

        # Transform content blocks
        for block_idx, block in enumerate(section.get('contentBlocks', [])):
            block_slides = self._transform_block(
                block, section_id, path_prefix, block_idx, importance_map
            )
            slides.extend(block_slides)

        # Transform subsections recursively
        for sub_idx, subsection in enumerate(section.get('subsections', [])):
            sub_slides = self._transform_section(
                subsection, section_id, importance_map,
                f"{path_prefix}.{sub_idx}"
            )
            slides.extend(sub_slides)

        return slides

    def _transform_block(
        self,
        block: Dict[str, Any],
        parent_id: str,
        chapter_idx: Any,
        block_idx: int,
        importance_map: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """
        Transform a content block to one or more slides.

        Args:
            block: Content block dictionary
            parent_id: Parent section ID
            chapter_idx: Chapter index
            block_idx: Block index
            importance_map: Concept importance scores

        Returns:
            List of slide dictionaries
        """
        slides = []

        # Get block type and content
        block_type = block.get('blockType', 'paragraph').lower()
        content = block.get('content', '') or block.get('text', '')
        items = block.get('items', [])
        block_id = block.get('id', f'block_{chapter_idx}_{block_idx}')

        # Determine slide type
        slide_type = self.BLOCK_TO_SLIDE_MAPPING.get(block_type, SlideType.CONTENT)

        # Transform based on block type
        if block_type in ('list_ordered', 'list_unordered', 'unordered_list', 'ordered_list'):
            slides = self._transform_list_to_slides(items, parent_id, block_id, importance_map)
        elif block_type == 'definition_list':
            slides = self._transform_definitions_to_slides(block, parent_id, block_id)
        elif block_type == 'blockquote':
            slides = [self._transform_quote_to_slide(content, parent_id, block_id)]
        elif block_type == 'table':
            slides = [self._transform_table_to_slide(block, parent_id, block_id)]
        elif block_type in ('figure', 'image'):
            slides = [self._transform_figure_to_slide(block, parent_id, block_id)]
        elif block_type == 'paragraph':
            slides = self._transform_paragraph_to_slides(content, parent_id, block_id, importance_map)
        else:
            # Default: content slide
            slides = self._transform_generic_to_slides(block, parent_id, block_id)

        # Record provenance for each slide
        for slide in slides:
            self.provenance.append(ProvenanceEntry(
                source_id=block_id,
                source_type='block',
                source_path=f"chapters[{chapter_idx}].contentBlocks[{block_idx}]",
                target_slide_index=self._slide_counter,
                transformation=f"{block_type}_to_{slide.get('type', 'content')}"
            ))
            self._slide_counter += 1

        return slides

    def _transform_list_to_slides(
        self,
        items: List[str],
        parent_id: str,
        block_id: str,
        importance_map: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Transform a list to one or more content slides."""
        config = self.config.get('slide_transformation', {})
        max_bullets = config.get('max_bullets_per_slide', 6)
        max_words = config.get('max_words_per_bullet', 12)

        slides = []

        # Apply 6x6 rule: split if more than 6 items
        if len(items) <= max_bullets:
            # Single slide
            bullets, notes = self._apply_word_limit(items, max_words)
            slides.append({
                "type": "content",
                "title": self._generate_slide_title(items, importance_map),
                "content": {"bullets": bullets},
                "notes": notes if notes else self._generate_notes_from_list(items)
            })
        else:
            # Split into multiple slides
            chunks = [items[i:i + max_bullets] for i in range(0, len(items), max_bullets)]
            for idx, chunk in enumerate(chunks):
                bullets, notes = self._apply_word_limit(chunk, max_words)
                title_suffix = f" ({idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
                slides.append({
                    "type": "content",
                    "title": self._generate_slide_title(chunk, importance_map) + title_suffix,
                    "content": {"bullets": bullets},
                    "notes": notes if notes else self._generate_notes_from_list(chunk)
                })

        return slides

    def _transform_paragraph_to_slides(
        self,
        content: str,
        parent_id: str,
        block_id: str,
        importance_map: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Transform a paragraph to bullet-point slides."""
        if not content.strip():
            return []

        # Split paragraph into sentences for bullets
        sentences = self._split_into_bullets(content)

        if not sentences:
            return []

        return self._transform_list_to_slides(sentences, parent_id, block_id, importance_map)

    def _transform_definitions_to_slides(
        self,
        block: Dict[str, Any],
        parent_id: str,
        block_id: str
    ) -> List[Dict[str, Any]]:
        """Transform definitions to two-content slides."""
        definitions = block.get('definitions', [])
        if not definitions:
            # Try to parse from content
            content = block.get('content', '')
            return [{
                "type": "content",
                "title": "Definitions",
                "content": {"bullets": [content[:100]]},
                "notes": content
            }]

        slides = []
        config = self.config.get('slide_transformation', {})
        max_items = config.get('max_bullets_per_slide', 6)

        # Group definitions (3 per slide for readability)
        chunk_size = min(3, max_items // 2)
        chunks = [definitions[i:i + chunk_size] for i in range(0, len(definitions), chunk_size)]

        for idx, chunk in enumerate(chunks):
            terms = [d.get('term', '') for d in chunk]
            defs = [d.get('definition', '')[:80] for d in chunk]  # Truncate

            title_suffix = f" ({idx + 1}/{len(chunks)})" if len(chunks) > 1 else ""
            slides.append({
                "type": "two_content",
                "title": "Key Terms" + title_suffix,
                "content": {
                    "left": terms,
                    "right": defs
                },
                "notes": "\n".join([f"**{d.get('term', '')}**: {d.get('definition', '')}" for d in chunk])
            })

        return slides if slides else [{
            "type": "content",
            "title": "Definitions",
            "content": {"bullets": ["See speaker notes for definitions"]},
            "notes": str(definitions)
        }]

    def _transform_quote_to_slide(
        self,
        content: str,
        parent_id: str,
        block_id: str
    ) -> Dict[str, Any]:
        """Transform a blockquote to a quote slide."""
        # Try to extract attribution
        attribution = ""
        quote_text = content

        # Check for common attribution patterns
        if " - " in content:
            parts = content.rsplit(" - ", 1)
            if len(parts) == 2 and len(parts[1]) < 50:
                quote_text = parts[0]
                attribution = parts[1]

        return {
            "type": "quote",
            "title": "",
            "content": {
                "text": quote_text[:300],  # Limit quote length
                "attribution": attribution
            },
            "notes": content if len(content) > 300 else ""
        }

    def _transform_table_to_slide(
        self,
        block: Dict[str, Any],
        parent_id: str,
        block_id: str
    ) -> Dict[str, Any]:
        """Transform a table to a table/content slide."""
        headers = block.get('headers', [])
        rows = block.get('rows', [])

        # For now, use content slide with description
        # (Full table support requires extended schema)
        if headers and rows:
            # Create a text representation
            bullet_summary = [f"Columns: {', '.join(headers[:5])}"]
            bullet_summary.append(f"Rows: {len(rows)}")

            return {
                "type": "content",
                "title": "Data Table",
                "content": {"bullets": bullet_summary},
                "notes": f"Table with headers: {headers}\nRows: {len(rows)}"
            }

        return {
            "type": "content",
            "title": "Table",
            "content": {"bullets": ["See accompanying materials for full table"]},
            "notes": str(block.get('content', ''))
        }

    def _transform_figure_to_slide(
        self,
        block: Dict[str, Any],
        parent_id: str,
        block_id: str
    ) -> Dict[str, Any]:
        """Transform a figure/image to an image slide."""
        return {
            "type": "image",
            "title": block.get('caption', 'Figure'),
            "content": {
                "image_path": block.get('src', block.get('path', '')),
                "alt_text": block.get('alt', block.get('caption', 'Image'))
            },
            "notes": block.get('description', '')
        }

    def _transform_generic_to_slides(
        self,
        block: Dict[str, Any],
        parent_id: str,
        block_id: str
    ) -> List[Dict[str, Any]]:
        """Transform any block type to content slides."""
        content = block.get('content', '') or block.get('text', '')
        items = block.get('items', [])

        if items:
            bullets = items[:6]  # Apply 6x6 rule
            notes = '\n'.join(items)
        elif content:
            sentences = self._split_into_bullets(content)
            bullets = sentences[:6]
            notes = content
        else:
            return []

        return [{
            "type": "content",
            "title": block.get('title', 'Content'),
            "content": {"bullets": bullets},
            "notes": notes
        }]

    def _create_section_header(
        self,
        chapter: Dict[str, Any],
        chapter_id: str,
        chapter_idx: int
    ) -> Dict[str, Any]:
        """Create a section header slide."""
        objectives = chapter.get('explicitObjectives', [])
        subtitle = ""
        if objectives:
            subtitle = f"Objectives: {len(objectives)}"

        return {
            "type": "section_header",
            "title": chapter.get('headingText', ''),
            "content": {"subtitle": subtitle},
            "notes": self._generate_section_notes(chapter)
        }

    def _generate_section_notes(self, chapter: Dict[str, Any]) -> str:
        """Generate speaker notes for section header."""
        notes_parts = []
        notes_parts.append(f"Section: {chapter.get('headingText', '')}")

        objectives = chapter.get('explicitObjectives', [])
        if objectives:
            notes_parts.append("\nLearning Objectives:")
            for obj in objectives[:5]:
                if isinstance(obj, dict):
                    notes_parts.append(f"- {obj.get('text', obj.get('objective', ''))}")
                else:
                    notes_parts.append(f"- {obj}")

        return '\n'.join(notes_parts)

    def _apply_word_limit(
        self,
        items: List[str],
        max_words: int
    ) -> Tuple[List[str], str]:
        """
        Apply word limit to bullets, moving excess to notes.

        Returns:
            Tuple of (shortened_bullets, notes_text)
        """
        shortened = []
        notes_parts = []

        for item in items:
            words = item.split()
            if len(words) <= max_words:
                shortened.append(item)
            else:
                # Truncate and add ellipsis
                truncated = ' '.join(words[:max_words - 1]) + '...'
                shortened.append(truncated)
                notes_parts.append(f"Full: {item}")

        notes = '\n'.join(notes_parts) if notes_parts else ''
        return shortened, notes

    def _split_into_bullets(self, text: str) -> List[str]:
        """Split text into bullet-worthy sentences."""
        import re

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        bullets = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Skip very short fragments
                bullets.append(sentence)

        return bullets

    def _generate_slide_title(
        self,
        items: List[str],
        importance_map: Dict[str, float]
    ) -> str:
        """Generate a slide title based on content and importance."""
        # Extract key terms from items
        all_text = ' '.join(items).lower()
        words = all_text.split()

        # Find most important concept
        best_term = None
        best_score = 0.0

        for term, score in importance_map.items():
            if term in all_text and score > best_score:
                best_term = term
                best_score = score

        if best_term:
            return best_term.title()

        # Fallback: use first significant word
        for word in words:
            if len(word) > 4 and word.isalpha():
                return word.title()

        return "Key Points"

    def _generate_notes_from_list(self, items: List[str]) -> str:
        """Generate speaker notes from list items."""
        if not items:
            return ""

        notes_parts = ["Key points to cover:"]
        for item in items:
            notes_parts.append(f"- {item}")

        return '\n'.join(notes_parts)


# Convenience function
def transform_to_presentation(
    semantic_structure: Dict[str, Any],
    concept_graph: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Transform semantic structure to presentation format.

    Args:
        semantic_structure: Output from SemanticStructureExtractor
        concept_graph: Optional concept graph for importance scoring
        config: Optional configuration

    Returns:
        Presentation JSON matching schema
    """
    transformer = PresentationTransformer(config)
    return transformer.transform(semantic_structure, concept_graph)
