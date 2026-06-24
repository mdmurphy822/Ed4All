"""FR-INT-03 — ``ED4ALL_REFLECTION_CALIBRATION`` flag resolver.

Single source of truth for the FR-INT-03 gate that turns a bare B11 reflection
("just ask") into a **predict-then-reveal calibration** interaction: capture the
learner's PREDICTION / judgment, REVEAL the benchmark / expert answer, then give
CALIBRATION feedback comparing the two.

Default OFF — when unset (or falsey / garbage), the three B11 calibration Block
fields (``prediction_prompt`` / ``reveal_content`` / ``calibration_feedback``)
are never projected to HTML (the ``<details>`` predict-then-reveal scaffold
returns ``""``), the ``interaction_feedback`` ``REFLECTION_NO_CAPTURE`` arm and
the ``anatomy_slot_presence`` benchmark-in-feedback check both no-op, so every
existing snapshot / ``contentHash`` stays byte-identical (mirrors
``ED4ALL_NEW_BLOCK_TYPES`` / ``ED4ALL_BLOCK_A11Y``).

The three Block fields are unconditionally valid (additive Optional/None
defaults, hash-excluded) — only the renderer EMIT + the two validator arms are
flag-gated.

Parse-with-fallback: truthy ``1`` / ``true`` / ``yes`` / ``on`` enables;
everything else (falsey / garbage / unset) → off. Read each call so tests can
toggle the env var inline.
"""

from __future__ import annotations

import os

__all__ = [
    "ENV_REFLECTION_CALIBRATION",
    "resolve_reflection_calibration",
]

ENV_REFLECTION_CALIBRATION = "ED4ALL_REFLECTION_CALIBRATION"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_reflection_calibration(value: object = None) -> bool:
    """Return True iff FR-INT-03 reflection-calibration emit / gating is enabled.

    ``value`` (optional) overrides the env var when not ``None`` — accepts the
    same truthy tokens (case-insensitive). Falsey / garbage / unset → False.
    """
    if value is None:
        raw = os.environ.get(ENV_REFLECTION_CALIBRATION, "")
    else:
        raw = value
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in _TRUTHY
