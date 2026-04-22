"""Teaching Role taxonomy loader + deterministic (component, purpose) mapper.

Loads `schemas/taxonomies/teaching_role.json` (published by Worker F in
Wave 1.1) and exposes:

    TEACHING_ROLES      -- immutable tuple of the six canonical role names
                           in schema-declared order.
    load_teaching_roles -- raw schema dict (for advanced callers / tests).
    get_valid_roles     -- Set[str] of the six canonical role names.
    map_role            -- (component, purpose) -> Optional[str]
                           Deterministic lookup into the schema's
                           `x-component-mapping` array. Returns None for
                           any unmapped or partial input, so callers can
                           fall back to heuristic / LLM classification.

Mirrors the pattern in `lib/ontology/bloom.py`. Schema is read once at
import time and cached.

See `plans/kg-quality-review-2026-04/worker-k-subplan.md` (Wave 2,
REC-VOC-02) for the rollout design and the three emit sites in
`Courseforge/scripts/generate_course.py` that depend on this module.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

__all__ = [
    "TEACHING_ROLES",
    "load_teaching_roles",
    "get_valid_roles",
    "map_role",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical six teaching-role values, in schema-declared order (low → high
# pedagogical complexity, matching the Trainforge align_chunks.py:33
# VALID_ROLES tuple).
TEACHING_ROLES: Tuple[str, ...] = (
    "introduce",
    "elaborate",
    "reinforce",
    "assess",
    "transfer",
    "synthesize",
)


# ---------------------------------------------------------------------------
# Schema path + cache
# ---------------------------------------------------------------------------

_TEACHING_ROLE_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "teaching_role.json"
)


@lru_cache(maxsize=1)
def load_teaching_roles() -> Dict[str, Any]:
    """Read the teaching_role schema and return the raw dict.

    Cached for the process lifetime. Callers that mutate the dict will
    only affect their local reference because `_build_mapping` deep-copies
    into a frozen lookup before returning anything to mapping callers.
    """
    if not _TEACHING_ROLE_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"teaching_role schema not found at {_TEACHING_ROLE_SCHEMA_PATH}. "
            "Expected canonical copy from Worker F (Wave 1.1)."
        )

    with open(_TEACHING_ROLE_SCHEMA_PATH, encoding="utf-8") as f:
        schema = json.load(f)

    if not isinstance(schema, dict):
        raise ValueError(
            f"teaching_role schema at {_TEACHING_ROLE_SCHEMA_PATH} is not a JSON object"
        )

    return schema


@lru_cache(maxsize=1)
def _canonical_role_enum() -> Tuple[str, ...]:
    """Read the schema's `$defs.TeachingRole.enum` list.

    Verifies at load time that the schema's enum matches our frozen
    `TEACHING_ROLES` tuple — any drift fails fast.
    """
    schema = load_teaching_roles()
    defs = schema.get("$defs", {})
    tr_def = defs.get("TeachingRole", {})
    enum_values = tr_def.get("enum", [])
    if not enum_values:
        raise ValueError(
            "teaching_role schema is missing $defs.TeachingRole.enum"
        )
    if tuple(enum_values) != TEACHING_ROLES:
        raise ValueError(
            "teaching_role schema drift: "
            f"schema enum={enum_values} vs TEACHING_ROLES={TEACHING_ROLES}"
        )
    return tuple(enum_values)


@lru_cache(maxsize=1)
def _build_mapping() -> Dict[Tuple[str, str], str]:
    """Compile `x-component-mapping` into a {(component, purpose): role} dict.

    Built once at first lookup. Malformed entries raise at build time —
    this surfaces schema bugs before the first emit.
    """
    schema = load_teaching_roles()
    mapping_entries = schema.get("x-component-mapping", [])
    if not isinstance(mapping_entries, list):
        raise ValueError(
            "teaching_role schema: x-component-mapping must be a JSON array"
        )

    valid_roles = set(_canonical_role_enum())
    compiled: Dict[Tuple[str, str], str] = {}
    for entry in mapping_entries:
        if not isinstance(entry, dict):
            raise ValueError(
                "teaching_role schema: x-component-mapping entries must be objects"
            )
        component = entry.get("component")
        purpose = entry.get("purpose")
        role = entry.get("teaching_role")
        if not (isinstance(component, str) and component):
            raise ValueError(
                f"teaching_role schema: mapping entry missing component: {entry!r}"
            )
        if not (isinstance(purpose, str) and purpose):
            raise ValueError(
                f"teaching_role schema: mapping entry missing purpose: {entry!r}"
            )
        if role not in valid_roles:
            raise ValueError(
                f"teaching_role schema: mapping role {role!r} not in canonical enum "
                f"{sorted(valid_roles)} (entry: {entry!r})"
            )
        compiled[(component, purpose)] = role
    return compiled


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_valid_roles() -> Set[str]:
    """Return the six canonical teaching-role values as a fresh Set[str].

    Suitable for validation:

        if role in get_valid_roles(): ...

    Verified identical to `Trainforge.align_chunks.VALID_ROLES`.
    """
    return set(_canonical_role_enum())


def map_role(component: Optional[str], purpose: Optional[str]) -> Optional[str]:
    """Deterministic (component, purpose) -> teaching_role lookup.

    Returns the mapped role string for any declared `(component, purpose)`
    pair from the schema's `x-component-mapping` array. Returns `None`
    for any unmapped, partial, or empty input — the caller falls back to
    the next strategy in its precedence chain (JSON-LD, heuristic, LLM).

    Examples:
        >>> map_role("flip-card", "term-definition")
        'introduce'
        >>> map_role("self-check", "formative-assessment")
        'assess'
        >>> map_role("activity", "practice")
        'transfer'
        >>> map_role("accordion", "progressive-disclosure") is None
        True
        >>> map_role(None, "term-definition") is None
        True
        >>> map_role("flip-card", None) is None
        True
    """
    if not component or not purpose:
        return None
    return _build_mapping().get((component, purpose))
