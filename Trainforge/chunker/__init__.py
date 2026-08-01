"""Trainforge canonical chunker.

Owns the deterministic structural chunking surface used by:

- DART chunkset emit (``MCP/tools/pipeline_tools.py::_run_dart_chunking``)
- IMSCC chunkset emit (``MCP/tools/pipeline_tools.py::_run_imscc_chunking``)
- Trainforge's in-process ``CourseProcessor._chunk_content`` /
  ``_chunk_text_block`` wrappers
- ``Trainforge/rag/boilerplate_detector.py`` (re-exports
  ``strip_boilerplate`` from ``boilerplate.py``)

Originally a standalone workspace package (``ed4all-chunker/``) lifted
out of ``Trainforge/`` in Phase 7a; folded back into ``Trainforge/`` in
the post-Phase-8 review (see ``plans/post-phase8-review-2026-05.md``)
because (a) every caller was inside this repo so the workspace-member
machinery added no value, (b) the package already had lazy imports
back into ``Trainforge.parsers`` to dodge module-load cycles — those
become direct imports now that the chunker lives inside Trainforge.

Submodules:

- :mod:`Trainforge.chunker.boilerplate` — ``strip_boilerplate`` and
  friends; lifted from ``Trainforge/rag/boilerplate_detector.py``.
- :mod:`Trainforge.chunker.helpers` — text + section helpers
  (``extract_plain_text``, ``extract_section_html``,
  ``strip_assessment_feedback``, ``strip_feedback_from_text``,
  ``type_from_resource``).
- :mod:`Trainforge.chunker.chunker` — ``chunk_content`` /
  ``chunk_text_block`` / ``merge_small_sections`` /
  ``merge_section_source_ids`` + ``MIN_CHUNK_SIZE`` /
  ``MAX_CHUNK_SIZE`` / ``TARGET_CHUNK_SIZE`` constants +
  ``ChunkerContext`` callback shape.

Re-exports below let callers ``from Trainforge.chunker import
strip_boilerplate`` without reaching into the submodule.
"""

# Pipeline-contract version. Consumed by ``MCP/tools/pipeline_tools.py``
# when stamping ``chunker_version`` on ``course_manifest.json`` and
# the ``dart_chunks/`` / ``imscc_chunks/`` sidecar manifests.
# Decoupled from any Python-package version: the chunker_schema is the
# emit contract, not the package release. Bump when the emit SHAPE or
# semantics change.
CHUNKER_SCHEMA_VERSION = "v4"
__version__ = CHUNKER_SCHEMA_VERSION

# Chunk-TEXT extraction-contract version. ORTHOGONAL to
# ``CHUNKER_SCHEMA_VERSION`` (the emit SHAPE): this integer versions the
# semantics of WHAT TEXT lands in a chunk's ``text`` field — i.e. the
# ``Trainforge/parsers/html_content_parser.py::HTMLTextExtractor`` contract
# (see ``Trainforge/CLAUDE.md`` § "Extraction-text change class (no opt-out
# by design)").
#
# Version ``1`` was: screen-reader-scaffolding suppression (``sr-only`` /
# ``visually-hidden`` / ``aria-hidden="true"`` subtree drop) + structural
# delimiters (per-``<li>`` / table-row newlines, ``|`` between table cells,
# ``: `` after each ``<dt>``) — assembled by ``.strip()``-ing every HTML text
# node and re-joining the pieces with a single space.
#
# Version ``2`` (current) keeps both of those and fixes the joining model
# itself: the strip-and-space-join FABRICATED a separator at every inline tag
# boundary, so ``<p>The result is <strong>79</strong>.</p>`` extracted as
# ``The result is 79 .``. Text is now assembled whitespace-correctly
# (``html_content_parser._TextAssembler``) — whitespace actually present in
# the source is preserved (collapsed to one space), an INLINE element boundary
# contributes no separator, and a BLOCK element boundary does — so the same
# markup extracts as ``The result is 79.``. Corpora chunked under contract 1
# carry that fabricated whitespace in their ``text`` field; it also tripped
# the ``degenerate_source_stem`` arm of the training-synthesis content gate
# (``Trainforge/synthesis_eligibility.py::_STEM_SLOT_GAP_RE``), so re-chunking
# under contract 2 makes previously-excluded chunks eligible again.
#
# ``CHUNKER_SCHEMA_VERSION`` deliberately stays ``"v4"`` across extraction-text
# changes (the emit shape is unchanged), so a coarse ``chunker_version`` check
# can't distinguish a corpus chunked before vs. after these semantics shifted.
# This finer marker is stamped as ``extraction_contract`` on the chunkset
# sidecar + course manifests so a provenance consumer (e.g. the tier-2
# citation-anchoring floor gate) can tell whether a corpus was chunked under
# the current extraction-text contract. BUMP THIS whenever chunk-text
# extraction semantics change again (a new always-on suppression / delimiter
# rule), so downstream provenance gates re-classify pre-change corpora as
# legacy.
EXTRACTION_TEXT_CONTRACT_VERSION = 2

from Trainforge.chunker.boilerplate import (
    DEFAULT_MIN_DOC_FRAC,
    DEFAULT_NGRAM_TOKENS,
    BoilerplateConfig,
    contamination_rate,
    detect_repeated_ngrams,
    strip_boilerplate,
)
from Trainforge.chunker.cross_course_dedup import (
    DEFAULT_CHUNK_DEDUP_MIN_TOKENS,
    ENV_CHUNK_DEDUP,
    ENV_CHUNK_DEDUP_MIN_TOKENS,
    exact_content_hash,
    exact_token_count,
    normalize_exact,
    resolve_chunk_dedup_enabled,
    resolve_chunk_dedup_min_tokens,
)
from Trainforge.chunker.chunker import (
    CANONICAL_CHUNK_TYPES,
    CHUNK_OVERLAP_WORDS_ENV,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    TARGET_CHUNK_SIZE,
    ChunkContentResult,
    ChunkerContext,
    ChunkerContextRequired,
    apply_chunk_overlap,
    chunk_content,
    chunk_text_block,
    merge_section_source_ids,
    merge_small_sections,
    resolve_chunk_overlap_words,
    split_by_sentences,
    type_from_heading,
)
from Trainforge.chunker.stranded_heading_tails import (
    RELOCATE_STRANDED_HEADINGS_ENV,
    relocate_stranded_heading_tails,
    resolve_relocate_stranded_headings,
)
from Trainforge.chunker.helpers import (
    extract_learning_outcome_refs,
    extract_plain_text,
    extract_section_html,
    harvest_semantik_source_refs,
    parse_dart_pages_attr,
    strip_assessment_feedback,
    strip_feedback_from_text,
    type_from_resource,
    union_source_pages,
)

__all__ = [
    "__version__",
    "CHUNKER_SCHEMA_VERSION",
    "EXTRACTION_TEXT_CONTRACT_VERSION",
    "BoilerplateConfig",
    "CANONICAL_CHUNK_TYPES",
    "CHUNK_OVERLAP_WORDS_ENV",
    "ChunkContentResult",
    "ChunkerContext",
    "ChunkerContextRequired",
    "DEFAULT_CHUNK_DEDUP_MIN_TOKENS",
    "DEFAULT_MIN_DOC_FRAC",
    "DEFAULT_NGRAM_TOKENS",
    "ENV_CHUNK_DEDUP",
    "ENV_CHUNK_DEDUP_MIN_TOKENS",
    "MAX_CHUNK_SIZE",
    "MIN_CHUNK_SIZE",
    "RELOCATE_STRANDED_HEADINGS_ENV",
    "TARGET_CHUNK_SIZE",
    "apply_chunk_overlap",
    "chunk_content",
    "chunk_text_block",
    "contamination_rate",
    "detect_repeated_ngrams",
    "exact_content_hash",
    "exact_token_count",
    "extract_learning_outcome_refs",
    "extract_plain_text",
    "extract_section_html",
    "harvest_semantik_source_refs",
    "merge_section_source_ids",
    "normalize_exact",
    "parse_dart_pages_attr",
    "merge_small_sections",
    "relocate_stranded_heading_tails",
    "resolve_chunk_dedup_enabled",
    "resolve_chunk_dedup_min_tokens",
    "resolve_chunk_overlap_words",
    "resolve_relocate_stranded_headings",
    "split_by_sentences",
    "strip_assessment_feedback",
    "strip_boilerplate",
    "strip_feedback_from_text",
    "type_from_heading",
    "type_from_resource",
    "union_source_pages",
]
