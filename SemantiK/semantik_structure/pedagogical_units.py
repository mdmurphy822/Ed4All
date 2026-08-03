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
_FALSEY = frozenset({"0", "false", "no", "off"})


def resolve_unit_regroup_mode() -> bool:
    """Return True when the Stage-5e pedagogical-unit REGROUP pass is enabled.

    Reads ``SEMANTIK_UNIT_REGROUP``. **Default ON** (unset / blank / truthy /
    garbage -> on): only an EXPLICIT falsey value (``0``/``false``/``no``/``off``,
    case-insensitive) reverts to the byte-identical legacy split-label/body
    partition — the one-env-var revert lever. Mirrors the default-ON falsey-set
    pattern of ``structure_graph.resolve_reading_order_fix`` (house pattern for
    ``SEMANTIK_STRUCTURE_CLEAN`` / ``SEMANTIK_READING_ORDER_FIX`` /
    ``SEMANTIK_SPECIALIST_BATCH``), NOT the truthy-set opt-in it used pre-ITEM1.

    Calibrated 2026-07 (ITEM1 Phase A, ``scripts/calibration/regroup_calibration_ab.py``):
    0 dual-oracle over-merges, full elimination of the residual label-only-box
    population on an anchored corpus, and a byte-identical no-op on an
    anchor-free control. The merge additionally REQUIRES the reading-order fix
    on (the Phase-6 driver guard at ``block_resegment.resegment_blocks`` —
    ``resolve_unit_regroup_mode() AND resolve_reading_order_fix()``, both now
    default-ON so the default composite is ON); the user-visible gold box
    additionally requires ``SEMANTIK_GOLD_SHELL`` — but the merge itself FIRES
    off ``payload['css_class']`` alone.
    """
    raw = (os.environ.get("SEMANTIK_UNIT_REGROUP") or "").strip().lower()
    return raw not in _FALSEY


def resolve_unit_regroup_table_mode() -> bool:
    """Return True when the table-v2 narrow render-time passthrough absorb is
    enabled (``SEMANTIK_UNIT_REGROUP_TABLE``).

    Default OFF (unset / blank / falsey / garbage) -> byte-identical to the
    committed v1 regroup (the trailing Solution table renders ADJACENT to the
    worked-example box). A truthy value (``1``/``true``/``yes``/``on``,
    case-insensitive) opts in. This resolver returns the RAW bool; the
    effective no-op-unless-the-regroup-is-firing behaviour is enforced at the
    Phase-2 call site by the composite
    ``assembler.shell.resolve_narrow_table_absorb_mode`` (``SEMANTIK_UNIT_REGROUP``
    AND ``SEMANTIK_READING_ORDER_FIX`` both on).

    NOTE: this is a **truthy-set opt-in** (only ``1``/``true``/``yes``/``on``
    enable it) — deliberately NOT flipped to the default-ON falsey-set posture
    that ``resolve_unit_regroup_mode`` adopted in ITEM1. The table-v2 absorb is
    render-inert while ``SEMANTIK_GOLD_SHELL`` is off, so its default stays off;
    the two resolver bodies therefore no longer match.
    """
    raw = (os.environ.get("SEMANTIK_UNIT_REGROUP_TABLE") or "").strip().lower()
    return raw in _TRUTHY


def is_passthrough_region(region: Any) -> bool:
    """Whether ``region`` is a Stage-4 passthrough (table / math / figure) —
    i.e. it carries a non-null ``source_region_id`` (set at
    ``structure_graph.py`` for the table/math/figure passthrough emit).

    Single source of truth for the passthrough predicate: the Stage-5e
    ``block_resegment._is_passthrough`` aliases this so a future
    passthrough-kind change updates ONE definition. The ``source_region_id``
    test (not a ``kind`` check) covers table AND math AND figure passthrough
    identically.
    """
    return getattr(region, "source_region_id", None) is not None


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


def _unit_start_text_boundary(text: Any) -> bool:
    r"""True when ``text`` (the storage-SoT ``payload['text']``) BEGINS with a
    UNIT-START pedagogical label — the text-pattern fallback for a following
    unit-start region whose ``css_class`` was NEVER stamped.

    The over-merge this closes: the unit-boundary detector keyed off
    ``css_class`` alone, but the clean pass only stamps ``css_class`` on a
    re-tagged (demoted) HEADING — a council-typed *paragraph* "EXAMPLE N.M" /
    "TRY IT" label slips through unstamped, so the forward walk does NOT stop
    and swallows the next unit (~22% of ``worked_example`` boxes fused two
    examples).

    Mirrors the css path by reusing the SAME single source of truth
    (``deterministic_structure._pedagogical_class_for`` — the anchored
    ``_PEDAGOGICAL_LABEL_CLASSES`` regex the clean pass uses to STAMP the
    ``css_class``) and filtering its result to :data:`UNIT_START_PEDAGOGY_CLASSES`,
    so the text fallback and the css arm can NEVER disagree on what opens a
    unit. The lazy import keeps this module dependency-light (no module-load
    cycle: ``deterministic_structure`` never imports this module) and mirrors
    the ``block_resegment._pedagogical_label_at`` precedent.

    SPECIFIC by construction (avoids UNDER-merge / false boundaries): the SoT
    patterns are anchored at text start (``^\s*``), so a body sentence
    mentioning "for example" mid-text never trips it; ``pedagogy-solution`` /
    ``pedagogy-step`` / ``pedagogy-objectives`` map to BODY/other classes that
    are NOT in :data:`UNIT_START_PEDAGOGY_CLASSES`, so "Solution" / "Step 1" /
    plain prose are NOT boundaries (they remain absorbable body).
    """
    if not text:
        return False
    # Lazy import: the neutral module stays dependency-light and cycle-free
    # (deterministic_structure does not import pedagogical_units).
    from .qwen_specialists.deterministic_structure import _pedagogical_class_for

    return _pedagogical_class_for(str(text)) in UNIT_START_PEDAGOGY_CLASSES


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
      * (``html is None`` Stage-5e path ONLY) a region whose ``payload['text']``
        text-STARTS a unit-opening label (``_unit_start_text_boundary`` —
        "EXAMPLE N.M" / "TRY IT" / "HOW TO" / "BE PREPARED" / "PRACTICE MAKES
        PERFECT") even when its ``css_class`` was never stamped — the
        over-merge fix. Reuses the SAME SoT regex the clean pass stamps the
        ``css_class`` with, filtered to :data:`UNIT_START_PEDAGOGY_CLASSES`, so
        the text fallback and the css arm agree. Deliberately NOT consulted on
        the assembler-absorb (``html`` supplied) path, which stays byte-identical;
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
        # Text-pattern unit-start fallback — Stage-5e path ONLY (html is None).
        # A following region that text-STARTS a new unit is a boundary even
        # when its css_class was never stamped (the over-merge fix). Gated to
        # html is None so the assembler absorb (which always passes real html)
        # is byte-identical (it keeps relying on css_class / semantic_class).
        if _unit_start_text_boundary(text):
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
    "is_passthrough_region",
    "is_unit_boundary",
    "resolve_unit_regroup_mode",
    "resolve_unit_regroup_table_mode",
]
