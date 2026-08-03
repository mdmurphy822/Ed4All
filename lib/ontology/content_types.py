"""Canonical block_type -> content_type single-source-of-truth.

Every non-scaffolding ``Courseforge/scripts/blocks.py::BLOCK_TYPES`` member maps
to exactly one canonical ChunkType label (``schemas/taxonomies/content_type.json``
``$defs.ChunkType`` / ``lib/validators/content_type.py::get_valid_chunk_types``).
This is the ONE place the content-type backstops ask "what content_type should a
``scenario`` block carry when the rewrite LLM omitted (or mangled) it?" instead
of each call site maintaining its own partial map.

Before this module two parallel partial maps existed and drifted:

* ``MCP/tools/pipeline_tools.py::_REWRITE_BLOCK_TYPE_CONTENT_TYPE`` — the str-
  rewrite backstop. It carried only the component / new-block-type tokens, so it
  could NOT fill a content-type for the rest of the palette (``scenario`` /
  ``problem`` / ``concept`` / ``example`` / …) when the LLM omitted it →
  ``BlockContentTypeValidator`` fired ``OUTLINE_BLOCK_MISSING_CONTENT_TYPE``.
* ``Courseforge/scripts/rendering/generate_course.py::_IB5_DEFAULT_CONTENT_TYPE`` — the
  IB5 renderer-root default. It carried only the four IB5 types.

Both now delegate here. The map is the authoritative one; the call sites keep a
thin wrapper for their fill / repair plumbing.

The gate's scaffolding skip set — ``objective`` / ``chrome`` / ``recap``
(``Courseforge/router/inter_tier_gates.py::_CONTENT_TYPE_SCAFFOLDING_TYPES``) —
is intentionally NOT keyed: ``BlockContentTypeValidator`` skips those block
types, so a backstop default for them would never be consulted, and
``resolve_block_content_type`` returns ``None`` for them so a renderer that asks
gets the no-op (fall through to its own legacy default) rather than a stamped
content-type.

Surface:

    BLOCK_TYPE_CONTENT_TYPE: Dict[str, str]            # the map
    resolve_block_content_type(block_type) -> Optional[str]
    is_valid_content_type(value) -> bool               # ChunkType enum membership
"""

from __future__ import annotations

from typing import Dict, Optional

__all__ = [
    "BLOCK_TYPE_CONTENT_TYPE",
    "resolve_block_content_type",
    "is_valid_content_type",
    "get_valid_content_types",
]


# block_type -> canonical ChunkType label. EVERY value is a member of
# ``get_valid_chunk_types()`` (asserted by the drift-guard test
# ``MCP/tools/tests/test_content_type_backstop.py``). The scaffolding skip set
# (objective / chrome / recap) is deliberately absent — the gate skips those.
BLOCK_TYPE_CONTENT_TYPE: Dict[str, str] = {
    # Exposition (B03).
    "concept": "explanation",
    "explanation": "explanation",
    "multimedia": "explanation",
    # Worked / illustrative instances (B05 / B03).
    "example": "example",
    "worked_example": "example",
    "flip_card_grid": "example",
    # Highlighted ideas (B12).
    "callout": "key_idea",
    "key_idea": "key_idea",
    # Vocabulary (B12/B03).
    "vocab_card": "vocabulary",
    # Structural exposition (B06 / B03).
    "formula": "formula",
    "table": "table",
    "acronym": "acronym",
    "diagram": "diagram",
    # Practice / exercise (B08).
    "activity": "exercise",
    "guided_practice": "exercise",
    "checklist": "checklist",
    "problem": "problem",
    # Case / scenario (B09).
    "scenario": "scenario",
    # Knowledge-check / assessment (B07 / B14).
    "self_check_question": "assessment_item",
    "assessment_item": "assessment_item",
    # Misconceptions (B03/B12).
    "misconception": "common_pitfall",
    # Metacognitive / collaborative.
    "reflection_prompt": "reflection",
    "discussion_prompt": "discussion",
    # Summary / framing (B13 / B02 / B15).
    "summary_takeaway": "summary",
    "prereq_set": "overview",
    "hook": "overview",
    "resources": "overview",
}


def resolve_block_content_type(block_type: Optional[str]) -> Optional[str]:
    """Return the canonical ChunkType label for ``block_type`` (or ``None``).

    ``None`` when the block_type has no backstop default — the scaffolding types
    the content-type gate skips (objective / chrome / recap) or an unknown
    token. A ``None`` return is the caller's signal to leave the block alone
    (no fill, no repair).
    """
    if not isinstance(block_type, str):
        return None
    return BLOCK_TYPE_CONTENT_TYPE.get(block_type)


def get_valid_content_types() -> frozenset:
    """Return the canonical ChunkType enum as a frozenset (lazy import).

    Imported lazily so this module stays import-light for callers that only
    need the static :data:`BLOCK_TYPE_CONTENT_TYPE` map. Fail-soft to an empty
    set on a schema-load hiccup so a backstop degrades to "no repair" rather
    than crashing the run.
    """
    try:
        from lib.validators.content_type import get_valid_chunk_types

        return frozenset(get_valid_chunk_types())
    except Exception:  # noqa: BLE001 — never break a backstop on schema load
        return frozenset()


def is_valid_content_type(value: object) -> bool:
    """True iff ``value`` is a member of the canonical ChunkType enum."""
    if not isinstance(value, str) or not value:
        return False
    return value in get_valid_content_types()
