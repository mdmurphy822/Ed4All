r"""Tier-2 composite-unit planner (adapter-seam, model-free, lexicon-role-driven).

Wave #22 (owner directive 1 — higher-level composite blocks coagulating smaller
blocks). The SemantiK adapter promotes pedagogical openers to headings and
delivers list/table/definition_list bodies, but the emitted blocks stay a FLAT
run of sibling ``<section>`` / callout-group elements: a worked example, its
solution, and its practice exercises are three unrelated siblings; a glossary
``<dl>`` and the example that illustrates it never bind. This module plans the
higher-level COMPOSITE UNIT — a run of adjacent sibling items that together form
one pedagogical whole — over the ABSTRACT lexicon ROLES (never subject-matter
vocabulary, never corpus-specific strings), so a new corpus is served by the
lexicon, not code.

The planner is pure: it consumes an ordered list of :class:`UnitItem`
(``role`` + ``boundary`` + ``members`` + ``has_heading``) and returns
:class:`UnitSpan` groupings. The adapter (:mod:`lib.semantik.adapter`) derives
the items from its rendered top-level element stream and wraps each planned span
in a ``<section class="dart-unit dart-unit-<type>" role="group">``.

Association rules (each expressed over abstract roles; the concrete opener slug
-> abstract role map lives in the lexicon as ``association_role``):

* ``worked_example``  : example  (+ solution)? (+ practice)*     [>= 2 members]
* ``section_opener``  : objectives (+ readiness)+                [>= 2 members]
* ``procedure``       : procedure (+ step)+                      [>= 2 members]
* ``exercise_set``    : directive? exercise_list (+ exercise_list)*  [>= 2 members]
* ``definition_group``: definition (+ example)                   [conservative]
* ``figure_group``    : figure (+ figure|caption)                [conservative]

GUARDS (anti-fabrication — when in doubt, don't group): adjacency only (a span
is a contiguous slice), a genuine section/chapter heading (``boundary``) never
enters a unit and always closes an open one, no gap-jumping over a non-member
item, a hard cap of :data:`MAX_UNIT_BLOCKS` source blocks per unit, and a unit
needs >= :data:`MIN_UNIT_ITEMS` member items (a lone example box stays
ungrouped).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "UnitItem",
    "UnitSpan",
    "plan_units",
    "MAX_UNIT_BLOCKS",
    "MIN_UNIT_ITEMS",
    "UNIT_TYPES",
    # Abstract roles (opener association roles + structural roles).
    "ROLE_EXAMPLE",
    "ROLE_SOLUTION",
    "ROLE_PRACTICE",
    "ROLE_PROCEDURE",
    "ROLE_OBJECTIVES",
    "ROLE_READINESS",
    "ROLE_DEFINITION",
    "ROLE_FIGURE",
    "ROLE_EXERCISE_LIST",
    "ROLE_DIRECTIVE",
    "ROLE_STEP",
    "ROLE_CAPTION",
]

# Abstract composite-unit roles.
ROLE_EXAMPLE = "example"
ROLE_SOLUTION = "solution"
ROLE_PRACTICE = "practice"
ROLE_PROCEDURE = "procedure"
ROLE_OBJECTIVES = "objectives"
ROLE_READINESS = "readiness"
ROLE_DEFINITION = "definition"
ROLE_FIGURE = "figure"
ROLE_EXERCISE_LIST = "exercise_list"
ROLE_DIRECTIVE = "directive"
ROLE_STEP = "step"
ROLE_CAPTION = "caption"

#: Hard cap on source blocks aggregated into one unit (guard).
MAX_UNIT_BLOCKS = 12
#: A unit needs at least this many member items.
MIN_UNIT_ITEMS = 2

UNIT_TYPES = (
    "worked_example",
    "section_opener",
    "procedure",
    "exercise_set",
    "definition_group",
    "figure_group",
)


@dataclass
class UnitItem:
    """One top-level rendered item (a callout-group or a standalone section).

    ``role`` is the abstract composite-unit role (or ``None`` for inert prose).
    ``boundary`` marks a genuine section/chapter heading that closes units and
    never enters one. ``members`` is the count of source blocks the item carries
    (a callout-group folds several; a bare section is 1) for the block cap.
    ``has_heading`` records whether the item leads with a visible heading whose
    ``sid`` can be the unit's ``aria-labelledby`` target.
    """

    role: Optional[str]
    boundary: bool = False
    members: int = 1
    has_heading: bool = False


@dataclass
class UnitSpan:
    """A planned grouping over ``items[start:end]`` (``end`` exclusive).

    ``unit_type`` is ``None`` for a pass-through single item (emitted as-is);
    otherwise one of :data:`UNIT_TYPES`. ``lead_index`` points at the member
    whose heading names the group (``aria-labelledby``); ``None`` when no member
    has a heading (the adapter falls back to a generated ``aria-label``).
    """

    start: int
    end: int
    unit_type: Optional[str] = None
    lead_index: Optional[int] = None


def _extend_while(
    items: List[UnitItem], start: int, allowed: tuple, budget: int
) -> int:
    """Greedily consume adjacent items whose role is in ``allowed``.

    Stops at a boundary item, a role outside ``allowed``, or when adding the
    next item would exceed the ``budget`` block cap. Returns the exclusive end.
    """
    j = start
    used = 0
    while j < len(items):
        it = items[j]
        if it.boundary or it.role not in allowed:
            break
        if used + it.members > budget:
            break
        used += it.members
        j += 1
    return j


def _match_from(items: List[UnitItem], i: int) -> Optional[UnitSpan]:
    """Try to match a composite unit STARTING at ``items[i]`` (priority order)."""
    head = items[i]
    if head.boundary:
        return None
    budget = MAX_UNIT_BLOCKS - head.members
    if budget < 0:
        return None

    # section_opener: objectives (+ readiness)+
    if head.role == ROLE_OBJECTIVES:
        end = _extend_while(items, i + 1, (ROLE_READINESS,), budget)
        if end - i >= MIN_UNIT_ITEMS:
            return UnitSpan(i, end, "section_opener", i)

    # worked_example: example (+ solution)? (+ practice)*
    if head.role == ROLE_EXAMPLE:
        end = _extend_while(
            items, i + 1, (ROLE_SOLUTION, ROLE_PRACTICE), budget
        )
        if end - i >= MIN_UNIT_ITEMS:
            return UnitSpan(i, end, "worked_example", i)

    # procedure: procedure (+ step|exercise_list)+
    if head.role == ROLE_PROCEDURE:
        end = _extend_while(
            items, i + 1, (ROLE_STEP, ROLE_EXERCISE_LIST), budget
        )
        if end - i >= MIN_UNIT_ITEMS:
            return UnitSpan(i, end, "procedure", i)

    # definition_group: definition (+ example)  — conservative (exactly the dl
    # plus ONE illustrating example, and only when that example does NOT itself
    # lead a worked-example chain — never split a worked example to bind a dl).
    if head.role == ROLE_DEFINITION:
        nxt = items[i + 1] if i + 1 < len(items) else None
        after = items[i + 2] if i + 2 < len(items) else None
        if (
            nxt is not None
            and not nxt.boundary
            and nxt.role == ROLE_EXAMPLE
            and head.members + nxt.members <= MAX_UNIT_BLOCKS
            and (after is None or after.role not in (ROLE_SOLUTION, ROLE_PRACTICE))
        ):
            lead = i if head.has_heading else (i + 1 if nxt.has_heading else None)
            return UnitSpan(i, i + 2, "definition_group", lead)

    # figure_group: figure (+ figure|caption)  — conservative.
    if head.role == ROLE_FIGURE:
        end = _extend_while(items, i + 1, (ROLE_FIGURE, ROLE_CAPTION), budget)
        if end - i >= MIN_UNIT_ITEMS:
            lead = i if head.has_heading else None
            return UnitSpan(i, end, "figure_group", lead)

    # exercise_set: directive? exercise_list (+ exercise_list)*
    if head.role in (ROLE_DIRECTIVE, ROLE_EXERCISE_LIST):
        if head.role == ROLE_DIRECTIVE:
            # A directive must be immediately followed by an exercise list.
            if (
                i + 1 < len(items)
                and not items[i + 1].boundary
                and items[i + 1].role == ROLE_EXERCISE_LIST
                and head.members + items[i + 1].members <= MAX_UNIT_BLOCKS
            ):
                end = _extend_while(
                    items, i + 1, (ROLE_EXERCISE_LIST,), budget
                )
                lead = i if head.has_heading else None
                return UnitSpan(i, end, "exercise_set", lead)
        else:  # exercise_list-led — needs a second exercise_list to be a unit.
            end = _extend_while(items, i + 1, (ROLE_EXERCISE_LIST,), budget)
            if end - i >= MIN_UNIT_ITEMS:
                lead = i if head.has_heading else None
                return UnitSpan(i, end, "exercise_set", lead)

    return None


def plan_units(items: List[UnitItem]) -> List[UnitSpan]:
    """Plan composite-unit spans over ``items`` (greedy, left-to-right).

    Returns a partition of ``items`` into :class:`UnitSpan`\\ s. A matched unit
    is a span with a ``unit_type``; every other item is a pass-through span
    (``unit_type=None``). Anti-fabrication guards are enforced by
    :func:`_match_from` (adjacency, boundary-closes, block cap, >= 2 members).
    """
    out: List[UnitSpan] = []
    i = 0
    n = len(items)
    while i < n:
        span = _match_from(items, i)
        if span is not None and span.end - span.start >= MIN_UNIT_ITEMS:
            out.append(span)
            i = span.end
        else:
            out.append(UnitSpan(i, i + 1, None, None))
            i += 1
    return out
