"""Internal LibV2 loader for vendored Bloom verb data.

LibV2 is sandboxed from importing Ed4All's ``lib/`` package (cross-package
caveat documented in ``LibV2/CLAUDE.md``). Instead of reaching across the
package boundary, LibV2 reads a byte-identical vendored copy of
``schemas/taxonomies/bloom_verbs.json`` at ``LibV2/vendor/bloom_verbs.json``.

The vendored copy is kept in sync with the authoritative source via:
  * CI hash check in ``ci/integrity_check.py``
  * Regression test ``lib/tests/test_bloom_ontology.py::test_libv2_vendor_hash_sync``

This module exposes ``get_verbs_list()`` with the same signature as
``lib.ontology.bloom.get_verbs_list`` so that call sites inside LibV2 can
obtain the same data without crossing the package boundary.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_BLOOM_LEVELS = (
    "remember",
    "understand",
    "apply",
    "analyze",
    "evaluate",
    "create",
)

_VENDOR_PATH = (
    Path(__file__).resolve().parents[2] / "vendor" / "bloom_verbs.json"
)


@lru_cache(maxsize=1)
def _load_raw() -> Dict[str, List[str]]:
    if not _VENDOR_PATH.exists():
        raise FileNotFoundError(
            f"Vendored bloom verbs missing at {_VENDOR_PATH}. "
            "Expected byte-copy of schemas/taxonomies/bloom_verbs.json."
        )
    with open(_VENDOR_PATH, encoding="utf-8") as f:
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
