"""Bloom-ladder initiative — ``ED4ALL_BLOOM_LADDER`` flag resolver (IB5 precedent).

Single source of truth for the gate that makes the Bloom-ladder — every
Bloom rung at or below an objective's OWN ``bloom_level``, each carrying its
permitted ``block_types`` + misconception-probe strategy
(``lib.ontology.bloom_ladder``) — selectable / renderable / emittable /
validated:

* planner SELECTION (``lib/generation/block_planner.py``) — draws candidate
  block types from the ladder instead of only the objective's own rung.
* renderer DISPATCH (``Courseforge/scripts/generate_course.py``) — renders a
  block at the rung it was planned for.
* field EMIT (``Courseforge/scripts/blocks.py::Block.to_html_attrs`` /
  ``to_jsonld_entry``) — projects the rung onto the block (``ladder_rung`` /
  ``mc_bloom_rung``) as HTML / JSON-LD.
* validator arm (misconception-probe / ladder-conformance checks) — asserts a
  block's rung is on the ladder for its objective's ceiling.

Default OFF — when unset (or falsey / garbage), no gated surface consults the
ladder: the planner only ever selects at an objective's own rung (today's
behavior), the renderer dispatch table is untouched, ``ladder_rung`` /
``mc_bloom_rung`` are never emitted, and the validator arm is a no-op — so
every existing snapshot / ``contentHash`` stays byte-identical (mirrors
``ED4ALL_NEW_BLOCK_TYPES`` / ``ED4ALL_BLOCK_A11Y`` / ``ED4ALL_DYNAMIC_BLOCK_PLAN``).

This module is the single choke point every gated surface imports —
``ladder_plan_for_objective`` delegates to
``lib.ontology.bloom_ladder.rungs_up_to`` (the WI-01 loader) rather than each
call site importing the ontology loader directly, so a future change to how
the ladder is capped only touches one function.

Parse-with-fallback: truthy ``1`` / ``true`` / ``yes`` / ``on`` enables;
everything else (falsey / garbage / unset) → off. Read each call so tests can
toggle the env var inline. Simple boolean — no mode selector.
"""

from __future__ import annotations

import os

from lib.ontology.bloom_ladder import RungEntry, rungs_up_to

__all__ = ["ENV_BLOOM_LADDER", "resolve_bloom_ladder", "ladder_plan_for_objective"]

ENV_BLOOM_LADDER = "ED4ALL_BLOOM_LADDER"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_bloom_ladder(value: object = None) -> bool:
    """Return True iff Bloom-ladder selection / emit / gating is enabled.

    ``value`` (optional) overrides the env var when not ``None`` — accepts the
    same truthy tokens (case-insensitive) so a caller can thread an explicit
    decision through. Falsey / garbage / unset → False.
    """
    if value is None:
        raw = os.environ.get(ENV_BLOOM_LADDER, "")
    else:
        raw = value
    if isinstance(raw, bool):
        return raw
    try:
        token = str(raw).strip().lower()
    except Exception:  # noqa: BLE001 — never crash a resolve on a weird value
        return False
    return token in _TRUTHY


def ladder_plan_for_objective(bloom_level: str) -> "tuple[RungEntry, ...]":
    """Return the ladder rungs available to an objective ceilinged at `bloom_level`.

    Thin delegation to ``lib.ontology.bloom_ladder.rungs_up_to`` — every rung
    whose ordinal is <= `bloom_level`'s ordinal, ordered low->high, so a
    caller never draws a block type / misconception probe from a rung above
    the objective's own synthesized ``bloom_level``. Raises ``ValueError`` if
    `bloom_level` is not one of the six canonical Bloom levels (same
    fail-loud contract as ``rungs_up_to``).

    Unconditional — this function does not itself consult
    ``resolve_bloom_ladder``. Every gated surface MUST check
    ``resolve_bloom_ladder()`` before calling this (or consuming its result)
    so the flag-off path never touches the ladder at all.
    """
    return rungs_up_to(bloom_level)
