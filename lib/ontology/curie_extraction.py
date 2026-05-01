"""CURIE extraction helpers (Wave 135b hoist from
``lib/validators/curie_preservation.py``).

Wave 135b moved these helpers out of the validator module so the
synthesis-side force-injection paths in ``Trainforge/generators/`` can
import them without inverting the layering (generators MUST NOT depend
on validators). The validator now re-exports the symbols from this
module, so existing tests that import ``CURIE_REGEX`` /
``EXCLUDED_PREFIXES`` from ``lib.validators.curie_preservation``
continue to resolve.

Behavior is unchanged from the Wave 131 implementation:

* Open-prefix CURIE detection — any ``prefix:LocalName`` pair where
  ``prefix`` is NOT in ``EXCLUDED_PREFIXES`` (URL schemes).
* Local-name's first character must be a letter — mathematically
  rejects ``localhost:8080`` / ``10:30`` / ``8:00 AM`` without an
  explicit exclusion list for digit-led local names.
"""
from __future__ import annotations

import re
from typing import Set


# URL schemes to exclude from CURIE detection. Anything else that fits
# the ``prefix:LocalName`` shape is treated as a CURIE.
EXCLUDED_PREFIXES = frozenset({
    "http", "https", "ftp", "file", "ws", "wss", "mailto", "tel",
    "urn", "data", "blob", "about", "localhost", "javascript",
})


CURIE_REGEX = re.compile(
    r"\b([A-Za-z][A-Za-z0-9_-]*):([A-Za-z][A-Za-z0-9_]*)\b"
)


def extract_curies(text: str) -> Set[str]:
    """Return the set of CURIEs found in ``text``.

    Open-prefix detection — any ``prefix:LocalName`` pair where
    ``prefix`` is NOT a URL scheme (filtered via ``EXCLUDED_PREFIXES``).
    Forward-compatible with new W3C vocabularies that land in chunks.

    Empty / falsy input returns an empty set rather than raising.
    """
    if not text:
        return set()
    return {
        f"{p}:{n}"
        for p, n in CURIE_REGEX.findall(text)
        if p.lower() not in EXCLUDED_PREFIXES
    }


__all__ = ["EXCLUDED_PREFIXES", "CURIE_REGEX", "extract_curies"]
