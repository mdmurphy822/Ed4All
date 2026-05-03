"""ed4all-chunker: canonical chunker for the Ed4All pipeline.

Phase 7a is lifting the chunking surface out of
``Trainforge/process_course.py`` and ``Trainforge/rag/boilerplate_detector.py``
so DART, Courseforge, and Trainforge can share one chunker:

- Subtask 1 — package skeleton (this layout).
- Subtask 2 — :mod:`ed4all_chunker.boilerplate` (``strip_boilerplate``
  and friends; lifted from ``Trainforge/rag/boilerplate_detector.py``).
- Subtask 3 — :mod:`ed4all_chunker.helpers` (text + section helpers).
- Subtask 4 — :mod:`ed4all_chunker.chunker` (``chunk_content`` /
  ``chunk_text_block``).

Re-exports below let callers ``from ed4all_chunker import strip_boilerplate``
without reaching into the submodule.
"""

from ed4all_chunker.boilerplate import (
    DEFAULT_MIN_DOC_FRAC,
    DEFAULT_NGRAM_TOKENS,
    BoilerplateConfig,
    contamination_rate,
    detect_repeated_ngrams,
    strip_boilerplate,
)

__all__ = [
    "BoilerplateConfig",
    "DEFAULT_MIN_DOC_FRAC",
    "DEFAULT_NGRAM_TOKENS",
    "contamination_rate",
    "detect_repeated_ngrams",
    "strip_boilerplate",
]
