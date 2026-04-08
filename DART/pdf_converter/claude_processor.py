"""
Claude-based text processor for PDF structure detection and ordering.

Uses Claude API to review extracted PDF text, fix ordering issues,
and detect document structure for WCAG-compliant HTML generation.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class BlockType(str, Enum):
    """Types of content blocks in a document."""
    TITLE = "title"
    AUTHOR = "author"
    ABSTRACT = "abstract"
    METADATA = "metadata"
    TOC_ITEM = "toc_item"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    REFERENCE = "reference"
    FIGURE_CAPTION = "figure_caption"
    TABLE = "table"
    DEFINITION = "definition"
    ALGORITHM = "algorithm"
    FOOTER = "footer"


@dataclass
class StructuredBlock:
    """A block of structured content from Claude."""
    block_type: str
    content: str
    heading_level: Optional[int] = None
    section_number: Optional[str] = None
    reference_number: Optional[int] = None
    id: Optional[str] = None


@dataclass
class DocumentStructure:
    """Complete structured document from Claude."""
    title: str
    authors: List[str]
    abstract: Optional[str]
    blocks: List[StructuredBlock]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Exceptions
class ClaudeProcessingError(Exception):
    """Base exception for Claude processing failures."""
    pass


class ClaudeAPIError(ClaudeProcessingError):
    """Claude API call failed."""
    pass


class ClaudeRateLimitError(ClaudeProcessingError):
    """Rate limit exceeded."""
    pass


class ClaudeInvalidResponseError(ClaudeProcessingError):
    """Claude returned invalid/unparseable JSON."""
    pass


class ResponseCache:
    """File-based cache for Claude responses."""

    def __init__(self, cache_dir: str = None):
        if cache_dir is None:
            cache_dir = Path.home() / '.cache' / 'pdf_converter'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, text: str, prompt_version: str) -> str:
        """Generate cache key from text content hash."""
        content = f"{prompt_version}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        return self.cache_dir / f"{key}.json"

    def get(self, text: str, prompt_version: str) -> Optional[dict]:
        """Retrieve cached response if exists."""
        key = self._get_cache_key(text, prompt_version)
        cache_path = self._get_cache_path(key)

        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    logger.debug(f"Cache hit for key {key[:8]}...")
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.warning(f"Failed to read cache file {cache_path}")
                return None
        return None

    def set(self, text: str, prompt_version: str, response: dict) -> None:
        """Cache response."""
        key = self._get_cache_key(text, prompt_version)
        cache_path = self._get_cache_path(key)

        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(response, f, indent=2)
            logger.debug(f"Cached response with key {key[:8]}...")
        except IOError as e:
            logger.warning(f"Failed to write cache: {e}")


class ClaudeProcessor:
    """
    Claude-based text processor for PDF structure detection.

    Uses Claude to:
    1. Fix text ordering issues from PDF extraction
    2. Identify document structure (headings, sections, abstract, etc.)
    3. Return structured JSON for HTML generation
    """

    PROMPT_VERSION = "v1.0"

    SYSTEM_PROMPT = '''You are a document structure analyzer for PDF accessibility conversion. Your task is to:

1. ANALYZE the raw text extracted from a PDF document
2. FIX any text ordering issues (PDF extraction sometimes produces out-of-order text, especially from multi-column layouts)
3. IDENTIFY the document structure (title, authors, abstract, sections, references, etc.)
4. RETURN structured JSON that maps to accessible HTML

IMPORTANT RULES:
- Preserve ALL original text content - do not summarize or omit anything
- Fix ordering but maintain the author's intended structure
- Identify heading levels based on numbering patterns:
  - Roman numerals (I., II., III.) = level 2 headings
  - Capital letters (A., B., C.) = level 3 headings
  - Numbered (1., 2.) without decimals = level 2 headings
  - Numbered with decimals (1.1, 2.1) = level 3 headings
  - Sub-sub sections (1.1.1) = level 4 headings
- Detect the abstract section and mark it appropriately
- Identify references/bibliography and number them sequentially
- Mark figure captions, table content, definitions, and algorithms
- Use the gold standard template structure as reference

The gold standard document structure follows this order:
1. Title (centered h1)
2. Authors (paragraph with author names)
3. Metadata (arXiv ID, keywords, etc.)
4. Abstract (special section)
5. Table of Contents (navigation)
6. Numbered sections (1. Introduction, 2. Background, etc.)
7. References (ordered list of citations)
8. Footer (author affiliations)

OUTPUT FORMAT: Return ONLY valid JSON, no markdown code fences, no explanation.'''

    USER_PROMPT_TEMPLATE = '''## Raw Extracted Text to Process:
{raw_text}

## Instructions:
Analyze this text and return a JSON object with the following structure:

{{
  "title": "Document title",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Abstract text if present, null otherwise",
  "blocks": [
    {{
      "block_type": "heading",
      "content": "Section heading text",
      "heading_level": 2,
      "section_number": "1."
    }},
    {{
      "block_type": "paragraph",
      "content": "Paragraph text content"
    }},
    {{
      "block_type": "reference",
      "content": "Reference citation text",
      "reference_number": 1
    }}
  ],
  "metadata": {{
    "keywords": [],
    "arxiv_id": null,
    "date": null
  }}
}}

Valid block_type values: title, author, abstract, metadata, toc_item, heading, paragraph, list_item, reference, figure_caption, table, definition, algorithm, footer

For headings, always include heading_level (2, 3, or 4) and section_number if present.
For references, include reference_number (1, 2, 3, etc.).

Return ONLY the JSON object, no other text.'''

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 16384,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
    ):
        """
        Initialize Claude processor.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            max_tokens: Maximum tokens in response
            cache_dir: Directory for caching responses
            enable_cache: Whether to use caching
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.model = model
        self.max_tokens = max_tokens
        self.cache = ResponseCache(cache_dir) if enable_cache else None
        self._client = None

    @property
    def client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ClaudeProcessingError(
                    "anthropic package not installed. Run: pip install anthropic"
                )
        return self._client

    def process_text(
        self,
        raw_text: str,
        gold_standard_template: str = None,
    ) -> DocumentStructure:
        """
        Process raw extracted text into structured document.

        Args:
            raw_text: Raw text from pdftotext/OCR
            gold_standard_template: HTML template showing target format (optional)

        Returns:
            DocumentStructure with ordered, classified blocks

        Raises:
            ClaudeProcessingError: On any processing failure
        """
        if not self.api_key:
            raise ClaudeProcessingError(
                "No API key provided. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key to ClaudeProcessor."
            )

        # Check cache first
        if self.cache:
            cached = self.cache.get(raw_text, self.PROMPT_VERSION)
            if cached:
                logger.info("Using cached Claude response")
                return self._parse_response(cached)

        # Check if document needs chunking (very large documents)
        estimated_tokens = len(raw_text) // 4
        if estimated_tokens > 150000:
            logger.info(f"Large document ({estimated_tokens} est. tokens), processing in chunks")
            return self._process_chunked(raw_text)

        # Build prompt
        user_prompt = self.USER_PROMPT_TEMPLATE.format(raw_text=raw_text)

        # Call Claude API
        try:
            import anthropic

            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
            )
        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str and 'limit' in error_str:
                raise ClaudeRateLimitError(f"Rate limit exceeded: {e}")
            raise ClaudeAPIError(f"API error: {e}")

        # Parse response
        response_text = response.content[0].text
        response_json = self._extract_json(response_text)

        # Cache successful response
        if self.cache:
            self.cache.set(raw_text, self.PROMPT_VERSION, response_json)

        return self._parse_response(response_json)

    def _process_chunked(
        self,
        raw_text: str,
        pages_per_chunk: int = 15,
    ) -> DocumentStructure:
        """
        Process large documents by chunking on page boundaries.

        Args:
            raw_text: Raw text from pdftotext/OCR
            pages_per_chunk: Number of pages per chunk

        Returns:
            Merged DocumentStructure
        """
        # Split on form feed characters (page breaks from pdftotext)
        pages = raw_text.split('\f')

        if len(pages) <= pages_per_chunk:
            return self.process_text(raw_text)

        logger.info(f"Document has {len(pages)} pages, processing in chunks of {pages_per_chunk}")

        all_blocks = []
        title = None
        authors = []
        abstract = None
        metadata = {}

        for i in range(0, len(pages), pages_per_chunk):
            chunk_pages = pages[i:i + pages_per_chunk]
            chunk_text = '\f'.join(chunk_pages)

            chunk_num = i // pages_per_chunk + 1
            total_chunks = (len(pages) + pages_per_chunk - 1) // pages_per_chunk
            logger.info(f"Processing chunk {chunk_num}/{total_chunks}")

            # Process chunk (without caching individual chunks)
            old_cache = self.cache
            self.cache = None
            try:
                chunk_result = self.process_text(chunk_text)
            finally:
                self.cache = old_cache

            # First chunk contains title, authors, abstract
            if i == 0:
                title = chunk_result.title
                authors = chunk_result.authors
                abstract = chunk_result.abstract
                metadata = chunk_result.metadata

            all_blocks.extend(chunk_result.blocks)

        return DocumentStructure(
            title=title or "Untitled Document",
            authors=authors,
            abstract=abstract,
            blocks=all_blocks,
            metadata=metadata,
        )

    def _extract_json(self, response_text: str) -> dict:
        """Extract and parse JSON from Claude's response."""
        # First try direct parsing
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code block
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find JSON object in response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        raise ClaudeInvalidResponseError(
            f"Could not parse JSON from response: {response_text[:500]}..."
        )

    def _parse_response(self, response: dict) -> DocumentStructure:
        """Parse and validate Claude's JSON response."""
        # Validate required fields
        if 'title' not in response:
            response['title'] = "Untitled Document"

        if 'blocks' not in response:
            raise ClaudeInvalidResponseError("Response missing 'blocks' field")

        # Parse blocks
        blocks = []
        valid_types = {e.value for e in BlockType}

        for block_data in response.get('blocks', []):
            block_type = block_data.get('block_type', 'paragraph')

            # Validate or default block type
            if block_type not in valid_types:
                logger.warning(f"Unknown block type '{block_type}', defaulting to 'paragraph'")
                block_type = 'paragraph'

            content = block_data.get('content', '')
            if not content:
                continue

            blocks.append(StructuredBlock(
                block_type=block_type,
                content=content,
                heading_level=block_data.get('heading_level'),
                section_number=block_data.get('section_number'),
                reference_number=block_data.get('reference_number'),
                id=block_data.get('id'),
            ))

        return DocumentStructure(
            title=response.get('title', 'Untitled Document'),
            authors=response.get('authors', []),
            abstract=response.get('abstract'),
            blocks=blocks,
            metadata=response.get('metadata', {}),
        )
