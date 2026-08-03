"""Internal LibV2 loader for the canonical Bloom verb taxonomy.

LibV2 is sandboxed from importing Ed4All's ``lib/`` package (cross-package
caveat documented in ``LibV2/CLAUDE.md``). It therefore reads the taxonomy
artifact directly from the Ed4All repository contract at
``schemas/taxonomies/bloom_verbs.json`` without importing the canonical
Python package or maintaining a second byte copy.

This module exposes ``get_verbs_list()`` with the same signature as
``lib.ontology.bloom.get_verbs_list`` so that call sites inside LibV2 can
obtain the same data without crossing the package boundary.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_BLOOM_LEVELS = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)

_CANONICAL_PATH = (
    Path(__file__).resolve().parents[3]
    / "schemas"
    / "taxonomies"
    / "bloom_verbs.json"
)


@lru_cache(maxsize=1)
def _load_raw() -> Dict[str, List[str]]:
    if not _CANONICAL_PATH.is_file():
        raise FileNotFoundError(
            f"Canonical Bloom taxonomy missing at {_CANONICAL_PATH}. "
            "LibV2 must run inside the Ed4All repository layout."
        )
    with open(_CANONICAL_PATH, encoding="utf-8") as f:
        schema = json.load(f)
    properties = schema.get("properties", {})
    return {
        level: [
            entry["verb"]
            for entry in properties[level]["default"]
        ]
        for level in _BLOOM_LEVELS
    }


def get_verbs_list() -> Dict[str, List[str]]:
    """Return ``Dict[str, List[str]]`` — ordered verb-string lists per level.

    Mirrors ``lib.ontology.bloom.get_verbs_list`` to keep the two call paths
    API-compatible. Returns a fresh defensive copy each call.
    """
    cached = _load_raw()
    return {level: list(cached[level]) for level in _BLOOM_LEVELS}


# ---------------------------------------------------------------------------
# Canonical-data detector
# ---------------------------------------------------------------------------
#
# ``lib.ontology.bloom.detect_bloom_level`` is the authoritative matcher.
# LibV2 cannot import it directly (cross-package boundary), so this small
# package-local matcher reads the authoritative taxonomy artifact directly.
# Any divergence in detection logic between this implementation and the
# canonical matcher is caught by
# ``lib/tests/test_bloom_detector_unification.py``.

_LEVEL_PRIORITY = {level: idx for idx, level in enumerate(_BLOOM_LEVELS)}


@lru_cache(maxsize=1)
def _detection_order() -> Tuple[Tuple[str, str], ...]:
    """Build the (verb, level) iteration order — longest-first, higher-
    level-tie-wins, alphabetical for stability."""
    raw = _load_raw()
    pairs: List[Tuple[str, str]] = []
    for level in _BLOOM_LEVELS:
        for verb in raw[level]:
            pairs.append((verb, level))
    pairs.sort(key=lambda p: (-len(p[0]), -_LEVEL_PRIORITY[p[1]], p[0]))
    return tuple(pairs)


def detect_bloom_level(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Detect the Bloom's level and verb from free text.

    Lowercases + strips the input, searches for each canonical verb as a
    whole word (``\\b{verb}\\b``). Returns ``(level, verb)`` on first match
    or ``(None, None)`` if no verb is found. Iteration order is longest-
    verb-first with higher-level ties winning — identical to
    ``lib.ontology.bloom.detect_bloom_level``.
    """
    if not text:
        return (None, None)
    lowered = text.lower().strip()
    for verb, level in _detection_order():
        if re.search(rf"\b{re.escape(verb)}\b", lowered):
            return (level, verb)
    return (None, None)
