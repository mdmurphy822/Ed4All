"""Neutral single-source-of-truth for pedagogical-unit boundary detection.

Hoisted here (Phase 1 of the pedagogical-unit REGROUP) so that BOTH the
assembler-side absorb stopgap (``assembler/gold_shell_markup.py``) and the
Stage-5e cross-kind detector (``qwen_specialists/block_resegment.py``) — and
the assembler ``shell.py`` (Phase 7) — import the SAME boundary contract and
the SAME ``SEMANTIK_UNIT_REGROUP`` resolver, so the absorb and the regroup can
never disagree on what a unit boundary is and the cross-package resolver import
is cycle-free.

This module is deliberately **dependency-light**: it imports neither the heavy
``reviewer`` / ``structure_graph`` modules nor anything from the ``assembler``
or ``qwen_specialists`` packages, so it can be imported from either side
without a module-load cycle.

The four boundary symbols are lifted VERBATIM from
``assembler/gold_shell_markup.py`` (their former private names re-export from
that module for byte-identical back-compat). ``is_unit_boundary`` is the former
``_is_absorption_boundary`` with one addition: the html-emptiness arm becomes
OPTIONAL so the Stage-5e detector (which has no rendered HTML yet) can call it
with ``html=None`` and fall back to ``region.payload['text']`` emptiness.
"""

from __future__ import annotations

import os
from typing import Any

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_unit_regroup_mode() -> bool:
    """Return True when the Stage-5e pedagogical-unit REGROUP pass is enabled.

    Reads ``SEMANTIK_UNIT_REGROUP``. Default OFF (unset / blank / falsey /
    garbage) -> byte-identical to today (mirrors
    ``block_resegment.resolve_block_resegment_mode``). A truthy value
    (``1``/``true``/``yes``/``on``, case-insensitive) enables it. The merge
    additionally REQUIRES the reading-order fix on (the Phase-6 driver guard);
    the user-visible gold box additionally requires ``SEMANTIK_GOLD_SHELL`` —
    but the merge itself FIRES off ``payload['css_class']`` alone.
    """
    raw = (os.environ.get("SEMANTIK_UNIT_REGROUP") or "").strip().lower()
    return raw in _TRUTHY


# Component classes whose label region absorbs the following body regions.
BODY_BEARING_COMPONENT_CLASSES = frozenset(
    {"worked_example", "definition_region", "exercise"}
)

# A following region carrying one of these ``payload['css_class']`` hints STARTS
# A NEW UNIT, so it is a boundary (never absorbed). ``pedagogy-solution`` /
# ``pedagogy-step`` are DELIBERATELY ABSENT — they are the example's BODY and DO
# get absorbed (mirrors ``deterministic_structure._PEDAGOGICAL_LABEL_CLASSES``).
UNIT_START_PEDAGOGY_CLASSES = frozenset(
    {
        "pedagogy-example",
        "pedagogy-try-it",
        "pedagogy-how-to",
        "pedagogy-be-prepared",
        "pedagogy-practice",
    }
)

# Conservative cap on a single absorption run — at most this many following
# regions are pulled into one box, so a mis-detected boundary can never swallow
# a whole section (a runaway absorb is bounded to a small bite).
ABSORB_MAX_RUN = 8


def _region_payload_get(region: Any, key: str) -> Any:
    return (getattr(region, "payload", None) or {}).get(key)


def is_unit_boundary(region: Any, html: Any = None) -> bool:
    """True when ``region`` STOPS a pedagogical-unit run (it must NOT be pulled
    into the preceding unit's box / merged region).

    Boundaries (the FIRST one ends the run):
      * a heading region (``region.kind == "heading"``);
      * an empty / whitespace fragment (nothing to absorb — stop
        conservatively). When ``html`` is supplied (the assembler, which has
        the rendered fragment) the byte-emptiness of ``html`` is checked; when
        ``html is None`` (Stage-5e, pre-Stage-6 — no rendered HTML yet) the
        emptiness of ``region.payload['text']`` (the storage SoT) is checked
        instead, keeping this module decoupled from the heavy renderer;
      * a unit-opening pedagogical label (``payload['css_class']`` in
        :data:`UNIT_START_PEDAGOGY_CLASSES`);
      * a region that itself starts a body-bearing component
        (``payload['semantic_class']`` in :data:`BODY_BEARING_COMPONENT_CLASSES`
        — the NEXT example/definition/exercise opens its own unit).

    Everything else — ``pedagogy-solution`` / ``pedagogy-step`` regions, plain
    paragraphs, tables — is absorbable BODY.

    ``html`` is a positional-or-keyword parameter (NO leading ``*``) so the
    existing positional call ``_is_absorption_boundary(regions[end],
    region_html[end])`` in ``compute_absorption_runs`` stays byte-identical.
    """
    if getattr(region, "kind", None) == "heading":
        return True
    if html is None:
        text = _region_payload_get(region, "text")
        if not text or not str(text).strip():
            return True
    elif not html or not html.strip():
        return True
    if _region_payload_get(region, "css_class") in UNIT_START_PEDAGOGY_CLASSES:
        return True
    if _region_payload_get(region, "semantic_class") in BODY_BEARING_COMPONENT_CLASSES:
        return True
    return False


__all__ = [
    "ABSORB_MAX_RUN",
    "BODY_BEARING_COMPONENT_CLASSES",
    "UNIT_START_PEDAGOGY_CLASSES",
    "is_unit_boundary",
    "resolve_unit_regroup_mode",
]
