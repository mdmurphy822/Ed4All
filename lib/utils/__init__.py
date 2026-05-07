"""Cross-cutting utility helpers shared across DART, Courseforge, Trainforge, LibV2.

These are pure-stdlib (or near-stdlib) helpers that previously lived inline
in 20+ call sites under various spellings (``_sha256_file``, ``_write_jsonl``,
``_TextExtractor``, ``_bootstrap_percentile_ci``, etc.). W-D6 consolidates
them so a future audit can grep one location for the canonical shape.

Public surface (top-level re-export, mirrors :mod:`lib.embedding`):

- :func:`sha256_file` / :func:`sha256_text` -- content-addressed file/string
  hashing (manifest gates, promotion chain, replay diff).
- :func:`read_jsonl` / :func:`write_jsonl` / :func:`append_jsonl` --
  JSONL I/O with canonical defaults (``ensure_ascii=False, sort_keys=True``,
  atomic tmp+rename for non-append writes).
- :func:`build_validator` -- Draft 2020-12 ``jsonschema`` validator with a
  ``referencing.Registry`` resolver; replaces 2 hand-rolled paths in
  Courseforge + Trainforge.
- :func:`bootstrap_percentile_ci` / :func:`percentile` -- pure-stdlib
  bootstrap CI estimator.
- :class:`TextExtractor` / :func:`strip_html_to_text` -- stdlib HTMLParser
  subclass that accumulates text from data events.

W-D3 T3.5 may have introduced a partial ``jsonl_io`` / ``json_io`` split;
W-D6 supersedes that boundary -- everything lands in :mod:`lib.utils.jsonl`.
See ``plans/wave-D6-lib-utils-package-2026-05-07.md`` Section 5 for the
merge plan.
"""
from __future__ import annotations

from lib.utils.hashing import sha256_file, sha256_text
from lib.utils.html_text import TextExtractor, strip_html_to_text
from lib.utils.jsonl import append_jsonl, read_jsonl, write_jsonl
from lib.utils.jsonschema import build_validator
from lib.utils.stats import bootstrap_percentile_ci, percentile

__all__ = [
    "TextExtractor",
    "append_jsonl",
    "bootstrap_percentile_ci",
    "build_validator",
    "percentile",
    "read_jsonl",
    "sha256_file",
    "sha256_text",
    "strip_html_to_text",
    "write_jsonl",
]
