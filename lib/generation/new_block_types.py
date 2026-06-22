"""IB5 — ``ED4ALL_NEW_BLOCK_TYPES`` flag resolver.

Single source of truth for the IB5 gate that makes the four framework-aligned
pedagogical block types selectable + renderable + gated:

* ``hook`` (B02 activation)
* ``multimedia`` (B04, the mandatory time-based-media a11y stack)
* ``worked_example`` (B05, subgoal labels + per-step Why + fade-state)
* ``diagram`` (B06, structured long-description + data-table equivalent)

Default OFF — when unset (or falsey / garbage), the four tokens are never
selected by the planner (``lib/generation/block_planner.py``), never rendered
(``Courseforge/scripts/generate_course.py``), their type-specific Block fields
(``fade_state`` / ``long_description`` / ``media_a11y``) are never projected to
HTML / JSON-LD, and the IB5 a11y arms in ``rewrite_html_shape`` are a no-op, so
every existing snapshot / ``contentHash`` stays byte-identical (mirrors
``ED4ALL_BLOCK_A11Y`` / ``ED4ALL_DYNAMIC_BLOCK_PLAN`` / ``ED4ALL_KEY_TERMS_PAGE``).

The four tokens are unconditionally valid ``BLOCK_TYPES`` members (so a Block
constructs without raising) — exactly the ``table`` / ``acronym`` / ``key_idea``
posture; only the planner SELECTION + renderer DISPATCH + field EMIT + validator
arm are flag-gated. No-op in practice when ``COURSEFORGE_TWO_PASS`` is unset
because the four types are only selectable on the dynamic-planner / two-pass
outline path.

Parse-with-fallback: truthy ``1`` / ``true`` / ``yes`` / ``on`` enables;
everything else (falsey / garbage / unset) → off. Read each call so tests can
toggle the env var inline.
"""

from __future__ import annotations

import os

__all__ = ["ENV_NEW_BLOCK_TYPES", "NEW_BLOCK_TYPES", "resolve_new_block_types"]

ENV_NEW_BLOCK_TYPES = "ED4ALL_NEW_BLOCK_TYPES"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# The four IB5 framework-aligned pedagogical block types -> framework B-code.
# ORDERED so callers iterate deterministically (planner nudges, doc prose).
NEW_BLOCK_TYPES: "tuple[str, ...]" = (
    "hook",            # B02
    "multimedia",      # B04
    "worked_example",  # B05
    "diagram",         # B06
)


def resolve_new_block_types(value: object = None) -> bool:
    """Return True iff IB5 new-block-type selection / emit / gating is enabled.

    ``value`` (optional) overrides the env var when not ``None`` — accepts the
    same truthy tokens (case-insensitive) so a caller can thread an explicit
    decision through. Falsey / garbage / unset → False.
    """
    if value is None:
        raw = os.environ.get(ENV_NEW_BLOCK_TYPES, "")
    else:
        raw = value
    if isinstance(raw, bool):
        return raw
    try:
        token = str(raw).strip().lower()
    except Exception:  # noqa: BLE001 — never crash a resolve on a weird value
        return False
    return token in _TRUTHY
