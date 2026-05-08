"""
Claude-based text processor for PDF structure detection and ordering.

Uses Claude API to review extracted PDF text, fix ordering issues,
and detect document structure for WCAG-compliant HTML generation.

Source-provenance note (Wave 8): this is the legacy DART path. Per P5
decision, it emits only a minimal ``data-dart-source="claude_llm"`` stamp
on top-level ``<section>`` wrappers (applied in
``DART/pdf_converter/converter.py::_generate_html_from_structure``). For
full per-block source attribution, page refs, and confidence envelopes,
the multi-source interpreter (``DART/multi_source_interpreter.py``) is
the primary source-provenance path. See
``plans/source-provenance/design.md`` §"DART changes".
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from MCP.orchestrator.llm_backend import LLMBackend


# Phase 6 Subtask 22 (Phase 3c env-vars): env-var-first model resolution
# for the legacy Claude-driven structure-detection path. Default preserves
# the previous hardcoded ``claude-sonnet-4-20250514`` behavior; operators
# retraining DART against a different teacher model set
# ``DART_CLAUDE_MODEL``. This module owns the canonical resolver helper —
# ``DART/pdf_converter/converter.py`` and
# ``DART/pdf_converter/alt_text_generator.py`` import it from here so a
# single env var pin propagates to every DART call site.
DART_CLAUDE_MODEL_ENV = "DART_CLAUDE_MODEL"
DART_CLAUDE_MODEL_DEFAULT = "claude-sonnet-4-20250514"

# Wave W-D13: DART provider routing. Operators running a license-clean
# pipeline route DART through a local OSS server (Ollama / vLLM / llama.cpp
# / LM Studio) or Together AI's OSS endpoint instead of Anthropic. Default
# preserves the legacy ``anthropic`` behaviour. The split between
# ``DART_PROVIDER`` (text-mode block classification, structure detection)
# and ``DART_VISION_PROVIDER`` (alt-text generation) lets an operator pin
# a small text model for cheap classification AND a 90B+ vision model for
# alt-text without flipping every call to the larger model. Mirrors the
# Courseforge ``OUTLINE_PROVIDER`` / ``REWRITE_PROVIDER`` split.
DART_PROVIDER_ENV = "DART_PROVIDER"
DART_VISION_PROVIDER_ENV = "DART_VISION_PROVIDER"
DART_PROVIDER_DEFAULT = "anthropic"


def _resolve_dart_claude_model(explicit: Optional[str] = None) -> str:
    """Pick the effective Claude model for DART call sites.

    Priority order:
      1. ``explicit`` argument (constructor kwarg) when truthy.
      2. ``DART_CLAUDE_MODEL`` env var when set (and non-empty).
      3. ``DART_CLAUDE_MODEL_DEFAULT`` (preserves legacy behavior).

    Mirrors the precedent in
    ``Trainforge/align_chunks.py::_resolve_align_model`` (Phase 4 Subtask 35).
    """
    if explicit:
        return explicit
    return os.environ.get(DART_CLAUDE_MODEL_ENV) or DART_CLAUDE_MODEL_DEFAULT


def _resolve_dart_model_for_provider(
    explicit: Optional[str], provider: str
) -> Optional[str]:
    """Wave W-D13. Resolve a model identifier appropriate for ``provider``.

    For ``provider="anthropic"`` (legacy default), always returns the
    Anthropic model from the canonical resolver — preserving the
    pre-W-D13 contract verbatim.

    For any non-anthropic provider, returns:
      1. ``explicit`` when truthy (operator pinned a specific model).
      2. ``DART_CLAUDE_MODEL`` env var when set AND not equal to the
         Anthropic legacy default (operator opted into a model name
         appropriate for the new backend).
      3. ``None`` — letting the registry resolver pick the
         provider-appropriate default (e.g. ``LOCAL_SYNTHESIS_MODEL``
         for ``provider="local"``, the registry's ``model_default`` for
         ``provider="together-vision"``, etc.).

    Without this helper a DART call routed to ``provider="local"``
    would stamp ``claude-sonnet-4-20250514`` as the model identifier on
    the local server, which does not understand it. Returning ``None``
    on the non-anthropic happy path lets the registry handle model
    resolution per its own env-var chain.
    """
    if provider == "anthropic":
        return _resolve_dart_claude_model(explicit)
    if explicit:
        return explicit
    env_value = os.environ.get(DART_CLAUDE_MODEL_ENV)
    if env_value and env_value != DART_CLAUDE_MODEL_DEFAULT:
        return env_value
    return None


def _resolve_dart_provider(explicit: Optional[str] = None) -> str:
    """Pick the effective DART text-mode provider.

    Resolution chain:
      1. ``explicit`` constructor kwarg when truthy.
      2. ``DART_PROVIDER`` env var when set (and non-empty).
      3. ``DART_PROVIDER_DEFAULT`` (``"anthropic"``) — legacy default.

    Returns a provider name from the universe of registered backends:
    ``"anthropic"`` (legacy default; routes through ``AnthropicBackend``),
    or any entry registered in
    ``MCP/orchestrator/llm_backend.py::_OPENAI_COMPATIBLE_PROVIDERS``
    (``"local"``, ``"together"``, ``"together-vision"``, ...). Adding a
    new provider is a registry-entry change in that file — DART picks it
    up automatically.
    """

    if explicit:
        return explicit
    return os.environ.get(DART_PROVIDER_ENV) or DART_PROVIDER_DEFAULT


def _resolve_dart_vision_provider(explicit: Optional[str] = None) -> str:
    """Pick the effective DART vision-mode provider (alt-text generation).

    Resolution chain:
      1. ``explicit`` constructor kwarg when truthy.
      2. ``DART_VISION_PROVIDER`` env var when set (and non-empty).
      3. Falls THROUGH to ``DART_PROVIDER`` (when set) — operators who
         want one provider for text + vision set ``DART_PROVIDER`` once
         and don't need to set both.
      4. ``DART_PROVIDER_DEFAULT`` (``"anthropic"``) — legacy default.

    The split exists so an operator can pin text-mode DART on a small
    7B/14B model AND vision-mode DART on a 90B Llama-3.2-Vision model
    without flipping every text call to the larger model. Mirrors the
    Courseforge ``OUTLINE_PROVIDER`` / ``REWRITE_PROVIDER`` split.
    """

    if explicit:
        return explicit
    return (
        os.environ.get(DART_VISION_PROVIDER_ENV)
        or os.environ.get(DART_PROVIDER_ENV)
        or DART_PROVIDER_DEFAULT
    )


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
                with open(cache_path, encoding='utf-8') as f:
                    logger.debug(f"Cache hit for key {key[:8]}...")
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
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
        except OSError as e:
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
        model: Optional[str] = None,
        max_tokens: int = 16384,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        llm: Optional["LLMBackend"] = None,
        provider: Optional[str] = None,
    ):
        """
        Initialize Claude processor.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
                Ignored when ``llm`` is provided OR ``provider`` resolves to
                a non-anthropic backend.
            model: Claude model to use. When ``None``, resolves via
                env-var-first chain: ``DART_CLAUDE_MODEL`` env var, then the
                legacy default ``claude-sonnet-4-20250514`` (Phase 6 Subtask
                22 / Phase 3c env-vars).
            max_tokens: Maximum tokens in response
            cache_dir: Directory for caching responses
            enable_cache: Whether to use caching
            llm: Optional pre-built LLM backend (e.g., an
                :class:`MCP.orchestrator.LLMBackend` instance). When provided,
                the processor routes completions through it instead of
                constructing a backend from ``provider``. Keeps existing
                callers that pass ``api_key`` working unchanged.
            provider: Wave W-D13. Provider name to route the structure-
                detection LLM call through. Resolution chain
                (`_resolve_dart_provider`): explicit kwarg > ``DART_PROVIDER``
                env var > ``"anthropic"`` (legacy default). Values:
                ``"anthropic"`` (legacy AnthropicBackend), or any entry
                registered in
                ``MCP/orchestrator/llm_backend.py::_OPENAI_COMPATIBLE_PROVIDERS``
                (``"local"``, ``"together"``, etc). Adding a new provider is
                a registry-entry change in that file. Ignored when ``llm``
                is supplied (the injected backend wins).
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        # Wave W-D13: resolve effective text-mode provider FIRST so the
        # model resolver knows which backend it's resolving for. Stored
        # on the instance so the lazy ``client`` property knows which
        # backend class to build.
        self.provider = _resolve_dart_provider(provider)
        # Phase 6 Subtask 22 + W-D13: provider-aware model resolution.
        # Anthropic provider keeps the legacy Claude resolver; non-
        # anthropic providers fall through to the registry's
        # provider-specific model env (LOCAL_SYNTHESIS_MODEL etc.) when
        # the operator hasn't pinned a non-default DART_CLAUDE_MODEL.
        self.model = _resolve_dart_model_for_provider(model, self.provider)
        self.max_tokens = max_tokens
        self.cache = ResponseCache(cache_dir) if enable_cache else None
        self._client = None
        self._llm = llm

    @property
    def client(self):
        """Lazy initialization of an LLM backend per the resolved provider.

        Only used when no injected ``llm`` backend was supplied. New callers
        should prefer ``llm=...`` in the constructor so that mode switching
        (local / api / mock) works uniformly. This path is retained for
        backward compatibility with direct-SDK callers.

        W-D13: when ``self.provider == "anthropic"`` the legacy
        ``AnthropicBackend`` path stays unchanged. For any other registered
        OpenAI-compatible provider, build the generic
        ``OpenAICompatibleBackend`` via the registry resolver. The model
        override (``self.model``) flows through so an operator pinning
        ``DART_CLAUDE_MODEL`` to a non-Anthropic identifier still gets
        their model honoured (the resolver short-circuits on the
        ``model_override`` argument).
        """
        if self._client is None:
            try:
                if self.provider == "anthropic":
                    from MCP.orchestrator.llm_backend import AnthropicBackend

                    self._client = AnthropicBackend(
                        api_key=self.api_key,
                        default_model=self.model or DART_CLAUDE_MODEL_DEFAULT,
                    )
                else:
                    from MCP.orchestrator.llm_backend import (
                        resolve_openai_compatible_backend,
                    )

                    # ``model`` override applies only when the operator
                    # explicitly pinned a non-default DART_CLAUDE_MODEL
                    # (per ``_resolve_dart_model_for_provider``). When
                    # ``self.model`` is ``None`` the registry resolver
                    # picks the provider-appropriate default — so
                    # ``DART_PROVIDER=local`` without a custom
                    # ``DART_CLAUDE_MODEL`` lands on
                    # ``LOCAL_SYNTHESIS_MODEL`` (or the registry's
                    # ``qwen2.5:14b-instruct-q4_K_M`` default).
                    self._client = resolve_openai_compatible_backend(
                        self.provider,
                        model_override=self.model,
                    )
            except Exception as exc:  # noqa: BLE001
                raise ClaudeProcessingError(
                    f"Could not initialize LLM backend for provider "
                    f"{self.provider!r}: {exc}"
                ) from exc
        return self._client

    def _complete(self, system: str, user: str) -> str:
        """Route a completion through the injected backend or the legacy path."""
        backend = self._llm if self._llm is not None else self.client
        return backend.complete_sync(
            system=system,
            user=user,
            model=self.model,
            max_tokens=self.max_tokens,
        )

    def process_text(
        self,
        raw_text: str,
        gold_standard_template: str = None,
    ) -> DocumentStructure:
        """Process raw extracted text into structured document.

        Args:
            raw_text: Raw text from pdftotext/OCR.
            gold_standard_template: Optional HTML template showing the
                target accessibility scaffolding (landmarks, heading
                hierarchy, skip-link shape). Wave 31 smoke-wires this
                into the WCAG-validator reference pattern — when
                supplied, ``DART.pdf_converter.wcag_validator.WCAGValidator``
                can compare the emitted HTML's landmark structure
                against the template's as a baseline check (full
                semantic comparison is deferred — see Wave 31 notes).
                Today the parameter is accepted + stored but not yet
                consumed in the prompt body; a future WCAGValidator
                patch is the planned consumer.

        Returns:
            DocumentStructure with ordered, classified blocks.

        Raises:
            ClaudeProcessingError: On any processing failure.
        """
        # Wave 31: stash the template on the instance so downstream
        # accessibility validators can pick it up without a signature
        # change. Never mutates behaviour — purely additive metadata.
        if gold_standard_template:
            self._gold_standard_template = gold_standard_template
        if self._llm is None and not self.api_key:
            raise ClaudeProcessingError(
                "No API key provided and no LLM backend injected. Set "
                "ANTHROPIC_API_KEY, pass api_key=..., or inject an "
                "LLMBackend via llm=... to ClaudeProcessor."
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

        # Call Claude via the LLM backend abstraction
        try:
            response_text = self._complete(
                system=self.SYSTEM_PROMPT,
                user=user_prompt,
            )
        except Exception as e:
            error_str = str(e).lower()
            if 'rate' in error_str and 'limit' in error_str:
                raise ClaudeRateLimitError(f"Rate limit exceeded: {e}") from e
            raise ClaudeAPIError(f"API error: {e}") from e

        # Parse response
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
