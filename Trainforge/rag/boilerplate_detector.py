"""Backwards-compatible shim for `Trainforge.rag.boilerplate_detector`.

Phase 7a Subtask 2 lifted the canonical implementation into the
`ed4all-chunker` package so DART, Courseforge, and Trainforge can share
one boilerplate detector. This module re-exports the same public
surface (``BoilerplateConfig``, ``detect_repeated_ngrams``,
``strip_boilerplate``, ``contamination_rate`` and the public constants)
from `ed4all_chunker.boilerplate` so every existing call site
(``Trainforge.process_course``, ``lib.leak_checker``, the
``test_generator_defects`` regression suite) keeps importing from this
path without modification.

The shim is intentionally thin: do **not** add new functionality here.
All edits land in `ed4all_chunker/boilerplate.py`; this file only
re-exports.
"""

from __future__ import annotations

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
