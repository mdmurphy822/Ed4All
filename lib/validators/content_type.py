"""Opt-in content_type enum validator (REC-VOC-03 Phase 2).

Wires Worker F's Wave 1 taxonomy (schemas/taxonomies/content_type.json) into
the two free-string content_type consumers:

- Trainforge/synthesize_training.py (instruction_pair emission)
- LibV2/tools/libv2/retriever.py (ChunkFilter.content_type_label)

Gated by TRAINFORGE_ENFORCE_CONTENT_TYPE=true. Default behavior: accept any
string (backward-compat with existing free-string consumers — no legacy data
migration per the Wave 4 opt-in policy).

The env var is read on each call (not at import) so tests can toggle it via
monkeypatch.setenv without importlib.reload gymnastics.

Design decisions (see plans/kg-quality-review-2026-04/worker-t-subplan.md § 2):
- ChunkType-only enforcement for Trainforge + LibV2 — SectionContentType is
  exposed via a helper but not wired into any enforcement path here.
- Strict-schema variant approach (sibling instruction_pair.strict.schema.json)
  rather than conditional allOf, because JSON Schema can't branch on env vars.
- Flag off = silent passthrough. Flag on = fail-closed (raise). No warn-log
  middle tier.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import FrozenSet

from lib.paths import SCHEMAS_PATH

_ENFORCE_ENV_VAR = "TRAINFORGE_ENFORCE_CONTENT_TYPE"


@lru_cache(maxsize=1)
def _load_content_type_schema() -> dict:
    """Load content_type.json once per process (lru_cache) and memoize."""
    path = SCHEMAS_PATH / "taxonomies" / "content_type.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_valid_chunk_types() -> FrozenSet[str]:
    """Return the ChunkType enum as a frozenset.

    These are the labels Trainforge emits on chunk.chunk_type (and that flow
    into instruction_pair.content_type via _normalize_content_type).
    """
    schema = _load_content_type_schema()
    return frozenset(schema["$defs"]["ChunkType"]["enum"])


@lru_cache(maxsize=1)
def get_valid_section_content_types() -> FrozenSet[str]:
    """Return the SectionContentType enum as a frozenset.

    These are the labels Courseforge emits on section data-cf-content-type.
    Exposed for completeness; not wired into any enforcement path in this PR.
    """
    schema = _load_content_type_schema()
    return frozenset(schema["$defs"]["SectionContentType"]["enum"])


def _is_enforcement_enabled() -> bool:
    """Read the env var each call so monkeypatch.setenv works in tests.

    A module-level constant would require importlib.reload in every test that
    toggles the flag. Per-call reads are ~150 ns and acceptable for the hot
    path (instruction_pair emission is already O(n_chunks) file I/O bound).
    """
    return os.getenv(_ENFORCE_ENV_VAR, "").strip().lower() == "true"


def validate_chunk_type(value: str) -> bool:
    """Return True if `value` is acceptable as a chunk content_type.

    With flag off: always True (backward-compat).
    With flag on: True iff value is a member of ChunkType enum.
    """
    if not _is_enforcement_enabled():
        return True
    return value in get_valid_chunk_types()


def validate_section_content_type(value: str) -> bool:
    """Return True if `value` is acceptable as a section content_type.

    With flag off: always True.
    With flag on: True iff value is a member of SectionContentType enum.
    """
    if not _is_enforcement_enabled():
        return True
    return value in get_valid_section_content_types()


def assert_chunk_type(value: str, context: str = "") -> None:
    """Raise ValueError when flag on and `value` is not a valid ChunkType.

    No-op when flag off or value is valid. Convenience wrapper for call-sites
    that want fail-closed semantics matching Worker I's chunk validation
    pattern (Trainforge/process_course.py:1987-2009).

    Args:
        value: the content_type label to validate.
        context: optional hint (e.g. "ChunkFilter.content_type_label" or
                 "chunk_id=foo_42") included in the error message.

    Raises:
        ValueError: if enforcement is on and value is not in ChunkType enum.
    """
    if validate_chunk_type(value):
        return
    valid = sorted(get_valid_chunk_types())
    ctx = f" ({context})" if context else ""
    raise ValueError(
        f"content_type {value!r}{ctx} is not a valid ChunkType. "
        f"Valid values: {valid}. "
        f"Set {_ENFORCE_ENV_VAR}=false (or unset) to disable enforcement."
    )


__all__ = [
    "get_valid_chunk_types",
    "get_valid_section_content_types",
    "validate_chunk_type",
    "validate_section_content_type",
    "assert_chunk_type",
]
