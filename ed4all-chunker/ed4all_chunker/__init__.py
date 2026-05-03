"""ed4all-chunker: canonical chunker for the Ed4All pipeline.

Phase 7a is lifting the chunking surface out of
``Trainforge/process_course.py`` and ``Trainforge/rag/boilerplate_detector.py``
so DART, Courseforge, and Trainforge can share one chunker:

- Subtask 1 — package skeleton (this layout).
- Subtask 2 — :mod:`ed4all_chunker.boilerplate` (``strip_boilerplate``
  and friends; lifted from ``Trainforge/rag/boilerplate_detector.py``).
- Subtask 3 — :mod:`ed4all_chunker.helpers` (text + section helpers).
- Subtask 4 — :mod:`ed4all_chunker.chunker` (``chunk_content`` /
  ``chunk_text_block`` / ``merge_small_sections`` /
  ``merge_section_source_ids`` + ``MIN_CHUNK_SIZE`` / ``MAX_CHUNK_SIZE`` /
  ``TARGET_CHUNK_SIZE`` constants + ``ChunkerContext`` callback shape).

Re-exports below let callers ``from ed4all_chunker import strip_boilerplate``
without reaching into the submodule.
"""

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version

    try:
        __version__ = _pkg_version("ed4all-chunker")
    except PackageNotFoundError:  # pragma: no cover — uninstalled source-tree fallback
        __version__ = "0.1.0"
except ImportError:  # pragma: no cover — Python <3.8 not supported (requires-python >=3.11)
    __version__ = "0.1.0"

from ed4all_chunker.boilerplate import (
    DEFAULT_MIN_DOC_FRAC,
    DEFAULT_NGRAM_TOKENS,
    BoilerplateConfig,
    contamination_rate,
    detect_repeated_ngrams,
    strip_boilerplate,
)
from ed4all_chunker.chunker import (
    CANONICAL_CHUNK_TYPES,
    MAX_CHUNK_SIZE,
    MIN_CHUNK_SIZE,
    TARGET_CHUNK_SIZE,
    ChunkContentResult,
    ChunkerContext,
    ChunkerContextRequired,
    chunk_content,
    chunk_text_block,
    merge_section_source_ids,
    merge_small_sections,
    split_by_sentences,
    type_from_heading,
)
from ed4all_chunker.helpers import (
    extract_plain_text,
    extract_section_html,
    strip_assessment_feedback,
    strip_feedback_from_text,
    type_from_resource,
)

__all__ = [
    "__version__",
    "BoilerplateConfig",
    "CANONICAL_CHUNK_TYPES",
    "ChunkContentResult",
    "ChunkerContext",
    "ChunkerContextRequired",
    "DEFAULT_MIN_DOC_FRAC",
    "DEFAULT_NGRAM_TOKENS",
    "MAX_CHUNK_SIZE",
    "MIN_CHUNK_SIZE",
    "TARGET_CHUNK_SIZE",
    "chunk_content",
    "chunk_text_block",
    "contamination_rate",
    "detect_repeated_ngrams",
    "extract_plain_text",
    "extract_section_html",
    "merge_section_source_ids",
    "merge_small_sections",
    "split_by_sentences",
    "strip_assessment_feedback",
    "strip_boilerplate",
    "strip_feedback_from_text",
    "type_from_heading",
    "type_from_resource",
]
