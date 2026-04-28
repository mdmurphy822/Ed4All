"""Helpers for comparing corpus chunk identifiers in eval code.

RDF/SHACL course archives use full chunk IDs such as
``rdf_shacl_551_chunk_00270`` while older fixtures use short IDs such as
``chunk_00270``. Eval code should treat both forms as the same source
anchor when the suffix after the rightmost ``chunk_`` matches.
"""
from __future__ import annotations

from typing import Any, Optional


def is_chunk_id(value: Any) -> bool:
    """Return True when ``value`` carries a chunk-id token."""
    return isinstance(value, str) and "chunk_" in value


def normalize_chunk_id(value: Any) -> Optional[str]:
    """Normalize a full corpus chunk ID to its ``chunk_*`` suffix.

    Returns ``None`` for non-string values and strings that do not contain
    a chunk token. The comparison suffix is enough because this eval run is
    scoped to one course archive at a time.
    """
    if not isinstance(value, str):
        return None
    idx = value.rfind("chunk_")
    if idx == -1:
        return None
    return value[idx:]


def chunk_ids_match(left: Any, right: Any) -> bool:
    """Compare two chunk IDs after canonical suffix normalization."""
    left_norm = normalize_chunk_id(left)
    right_norm = normalize_chunk_id(right)
    return left_norm is not None and left_norm == right_norm


__all__ = ["chunk_ids_match", "is_chunk_id", "normalize_chunk_id"]
