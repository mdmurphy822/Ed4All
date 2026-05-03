"""Backwards-compatible shim for `Trainforge.rag.boilerplate_detector`.

The canonical implementation lives at ``Trainforge/chunker/boilerplate.py``
(re-merged per ``plans/post-phase8-review-2026-05.md``; previously the
short-lived ``ed4all-chunker`` workspace package). This module re-exports
the same public surface (``BoilerplateConfig``, ``detect_repeated_ngrams``,
``strip_boilerplate``, ``contamination_rate`` and the public constants)
from ``Trainforge.chunker.boilerplate`` so every existing call site
(``Trainforge.process_course``, ``lib.leak_checker``, the
``test_generator_defects`` regression suite) keeps importing from this
path without modification.

The shim is intentionally thin: do **not** add new functionality here.
All edits land in ``Trainforge/chunker/boilerplate.py``; this file only
re-exports.
"""

from __future__ import annotations

from Trainforge.chunker.boilerplate import (
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
