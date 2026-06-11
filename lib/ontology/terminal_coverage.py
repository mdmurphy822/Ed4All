"""Terminal-objective coverage helpers — single source of truth.

A *terminal* learning objective (``TO-NN``) is "covered" when downstream
content generation will produce teaching + assessment chunks that resolve
to it. The LibV2 archival gate
(``lib/validators/libv2/packet_integrity.py::_rule_to_has_teaching_and_assessment``)
enforces that contract terminally, emitting ``UNCOVERED_TERMINAL_OUTCOME``
when a terminal has no teaching/assessment chunk.

That archival failure is a *late* symptom of an objectives set that was
internally inconsistent at *planning* time: a terminal objective was
minted (or reused) with nothing rolling up to it, so no content / chunk /
assessment ever covered it. This module centralises the two-shape
roll-up model so the synthesis-side prune (Layer A) and the fail-fast
``course_planning`` gate (Layer B) agree byte-for-byte on what "orphan
terminal" means.

Two real objectives shapes ship in the corpus:

* **CO-bearing courses** (RDF/SHACL reference corpus): a non-empty
  ``chapter_objectives`` where every chapter objective (``CO-NN``) carries
  a back-pointer to its parent terminal under one of ``parent_to`` /
  ``parent_terminal`` / ``parent_terminal_id`` / ``terminal_id``. A terminal
  with **zero** rolling-up COs is an orphan-by-construction. This is the
  canonical, fully-checkable case.

  The roll-up model is only "in use" when ≥1 chapter objective actually
  carries one of those back-pointer keys. A course can be CO-bearing
  (non-empty ``chapter_objectives``) yet have **zero** COs carrying any
  parent-terminal key — this is what the deterministic synthesizer, the
  Stage-2 provider, and ``_normalize_objective_entry`` all produce. In that
  case the roll-up set is empty, so *every* terminal would look orphaned,
  even though orphanhood is genuinely **unprovable from the objectives set
  alone**. Layer A prunes nothing and Layer B emits a
  ``TERMINAL_COVERAGE_UNVERIFIED`` warning + passes; the archival gate
  remains the terminal backstop.

* **CO-less courses** (e.g. textbook-derived or single-unit courses): an empty
  ``chapter_objectives`` where each terminal is its own teaching unit and
  carries a ``chapter`` back-pointer into the textbook structure. Here the
  roll-up model does not apply — coverage is content-driven — so the
  terminal is reachable iff its ``chapter`` resolves to a real chapter in
  the textbook structure.

Cross-references:

* ``lib/validators/terminal_objective_coverage.py`` — the Layer-B gate.
* ``MCP/tools/pipeline_tools.py::_plan_course_structure`` — the Layer-A
  prune call-site.
* ``lib/validators/libv2/packet_integrity.py`` — the terminal archival
  gate this module exists to keep from ever firing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

# Keys a chapter objective (CO) may carry its parent-terminal back-pointer
# under. ``parent_to`` is what the RDF reference corpus emits; the other
# two mirror the archival gate's ``parent_terminal`` and the generic
# ``terminal_id`` alias so this helper resolves every shape on disk.
_PARENT_TERMINAL_KEYS: Tuple[str, ...] = (
    "parent_to",
    "parent_terminal",
    "parent_terminal_id",
    "terminal_id",
)

# Keys a terminal objective may carry its chapter back-pointer under.
_CHAPTER_KEYS: Tuple[str, ...] = (
    "chapter",
    "chapter_id",
    "chapterId",
    "parent_chapter",
)


#: Leading chapter-label prefixes stripped before comparing a terminal's
#: ``chapter`` back-pointer against a textbook_structure chapter ID. The
#: same logical chapter is written several ways across the corpus
#: (``"6"`` on a reused objectives set vs ``"ch6"`` in the structure vs
#: ``"Chapter 6"`` / ``"Week 6"`` in prose), so coverage must compare on a
#: normalised key rather than the literal string — otherwise the CO-less
#: branch over-fires on every course whose chapter labels differ in
#: surface form from the structure IDs.
_CHAPTER_LABEL_PREFIX = re.compile(r"^(?:ch(?:apter)?|week|unit|module)[\s_-]*", re.I)


def _normalise_chapter_key(raw: Optional[str]) -> Optional[str]:
    """Normalise a chapter label to a comparable key.

    Strips a leading ``ch`` / ``chapter`` / ``week`` / ``unit`` / ``module``
    prefix and surrounding whitespace, lower-cases the remainder. Returns
    ``None`` for an empty / unusable input.
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    s = _CHAPTER_LABEL_PREFIX.sub("", s).strip().lower()
    return s or None


def _terminal_id(entry: Mapping[str, Any]) -> Optional[str]:
    """Return a terminal objective's canonical ID (lower-cased), or None."""
    tid = entry.get("id")
    if isinstance(tid, str) and tid.strip():
        return tid.strip().lower()
    return None


def co_parent_terminal(co: Mapping[str, Any]) -> Optional[str]:
    """Return the parent-terminal ID a chapter objective rolls up to.

    Lower-cased for case-insensitive matching against terminal IDs
    (downstream Trainforge consumers already match LO refs
    case-insensitively).
    """
    for key in _PARENT_TERMINAL_KEYS:
        val = co.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return None


def terminal_chapter_ref(to: Mapping[str, Any]) -> Optional[str]:
    """Return the chapter ID a terminal objective is attributed to."""
    for key in _CHAPTER_KEYS:
        val = to.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        # ``chapter`` is frequently an int ("chapter": 1).
        if isinstance(val, int):
            return str(val)
    return None


def flatten_chapter_objectives(chapter_objectives: Any) -> List[Dict[str, Any]]:
    """Flatten ``chapter_objectives`` (any of three shapes) to a CO list.

    Tolerates:

    * the canonical group-of-groups shape
      (``[{"chapter": str, "objectives": [...]}, ...]``),
    * a flat list of CO dicts, and
    * an empty / missing value (returns ``[]``).
    """
    out: List[Dict[str, Any]] = []
    if not isinstance(chapter_objectives, list):
        return out
    for entry in chapter_objectives:
        if not isinstance(entry, Mapping):
            continue
        inner = entry.get("objectives")
        if isinstance(inner, list):
            out.extend(o for o in inner if isinstance(o, Mapping))
        else:
            out.append(dict(entry))
    return out


def rolled_up_terminal_ids(chapter_objectives: Any) -> Set[str]:
    """Return the set of terminal IDs that ≥1 chapter objective rolls up to."""
    rolled: Set[str] = set()
    for co in flatten_chapter_objectives(chapter_objectives):
        parent = co_parent_terminal(co)
        if parent:
            rolled.add(parent)
    return rolled


def is_co_bearing(chapter_objectives: Any) -> bool:
    """True when the course uses the CO roll-up model (≥1 chapter objective)."""
    return bool(flatten_chapter_objectives(chapter_objectives))


def _chapter_ids_in_structure(
    textbook_structure: Optional[Mapping[str, Any]],
    *,
    require_text: bool = False,
) -> Set[str]:
    """Collect the chapter IDs present in a textbook_structure dict.

    When ``require_text`` is set, only chapters whose ``chapter_text`` is
    non-empty are counted (the content-producing set).
    """
    ids: Set[str] = set()
    if not isinstance(textbook_structure, Mapping):
        return ids
    chapters = textbook_structure.get("chapters")
    if not isinstance(chapters, list):
        return ids
    for ch in chapters:
        if not isinstance(ch, Mapping):
            continue
        cid = ch.get("id")
        if not (isinstance(cid, str) and cid.strip()):
            continue
        if require_text:
            text = ch.get("chapter_text")
            if not (isinstance(text, str) and text.strip()):
                continue
        norm = _normalise_chapter_key(cid)
        if norm:
            ids.add(norm)
    return ids


def find_orphan_terminals(
    terminal_objectives: Sequence[Mapping[str, Any]],
    chapter_objectives: Any,
    *,
    textbook_structure: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return the orphan terminals in an objectives set.

    An orphan terminal is one that downstream content generation cannot
    cover, surfaced per the two-shape model:

    * **CO-bearing course** — a terminal with zero chapter objectives
      rolling up to it (``reason="no_chapter_objective"``).
    * **CO-less course** — only checked when ``textbook_structure`` is
      supplied: a terminal whose ``chapter`` back-pointer does not resolve
      to any chapter ID in the structure
      (``reason="chapter_not_in_structure"``). When the structure is
      absent the CO-less case cannot be adjudicated and returns no
      orphans (the caller decides whether absence is a graceful-degrade
      pass).

    Each returned dict carries ``{"id", "reason", "chapter"?}``. IDs are
    returned in the original (un-lowercased) form for human-facing
    messages.
    """
    terminals = [t for t in terminal_objectives if isinstance(t, Mapping)]
    orphans: List[Dict[str, Any]] = []

    if is_co_bearing(chapter_objectives):
        rolled = rolled_up_terminal_ids(chapter_objectives)
        if not rolled:
            # CO-bearing but zero COs carry any parent-terminal key ⟹ the
            # roll-up model is not in use; orphanhood is unprovable from the
            # objectives set alone. Return no orphans so Layer A (prune) and
            # Layer B (gate) stay byte-agreed — the Layer-B gate emits a
            # TERMINAL_COVERAGE_UNVERIFIED warning + passes loudly instead.
            return orphans
        for to in terminals:
            tid = _terminal_id(to)
            if tid is None:
                continue
            if tid not in rolled:
                orphans.append(
                    {
                        "id": str(to.get("id")),
                        "reason": "no_chapter_objective",
                    }
                )
        return orphans

    # CO-less course: only adjudicate when we have the textbook structure.
    if textbook_structure is None:
        return orphans
    chapter_ids = _chapter_ids_in_structure(textbook_structure)
    if not chapter_ids:
        return orphans
    for to in terminals:
        tid = _terminal_id(to)
        if tid is None:
            continue
        chap = terminal_chapter_ref(to)
        if chap is None:
            # No chapter back-pointer and no CO roll-up — unreachable.
            orphans.append(
                {"id": str(to.get("id")), "reason": "no_chapter_ref"}
            )
            continue
        if _normalise_chapter_key(chap) not in chapter_ids:
            orphans.append(
                {
                    "id": str(to.get("id")),
                    "reason": "chapter_not_in_structure",
                    "chapter": chap,
                }
            )
    return orphans


def _structure_chapter_ids_ordered(
    textbook_structure: Optional[Mapping[str, Any]],
) -> List[str]:
    """Return the chapter IDs in a textbook_structure, in document order.

    Unlike :func:`_chapter_ids_in_structure` (which returns a *normalised*
    set for membership tests), this preserves the raw chapter ID strings in
    their original order so a back-pointer assignment can write a value that
    round-trips through :func:`terminal_chapter_ref` +
    :func:`_normalise_chapter_key` and resolves against the structure.
    """
    out: List[str] = []
    if not isinstance(textbook_structure, Mapping):
        return out
    chapters = textbook_structure.get("chapters")
    if not isinstance(chapters, list):
        return out
    for ch in chapters:
        if not isinstance(ch, Mapping):
            continue
        cid = ch.get("id")
        if isinstance(cid, str) and cid.strip():
            out.append(cid.strip())
    return out


def attach_terminal_chapter_refs(
    terminal_objectives: Sequence[Mapping[str, Any]],
    chapter_objectives: Any,
    *,
    textbook_structure: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Stamp a resolvable ``chapter`` back-pointer on CO-less terminals.

    Layer-A coverage guarantee for the **CO-less** branch (the companion to
    :func:`prune_orphan_terminals`, which only handles the CO-bearing case).
    A CO-less course (empty ``chapter_objectives``) is reachable iff every
    terminal carries a ``chapter`` back-pointer that resolves to a real
    chapter in ``textbook_structure`` (see :func:`find_orphan_terminals`,
    ``reason="no_chapter_ref"`` / ``"chapter_not_in_structure"``).

    The deterministic synthesizer mints terminals with no chapter
    back-pointer (the topic-side parser and the structure-side parser are
    independent, so a terminal's originating topic often has
    ``chapter_id=None`` even though the structure synthesized real
    chapters). Rather than let that internal inconsistency flow to the
    fail-fast ``course_planning`` gate, this helper makes the objectives set
    consistent BEFORE it is written to disk:

    * No-op for a CO-bearing course (the roll-up model owns coverage there).
    * No-op when ``textbook_structure`` has no resolvable chapter IDs (the
      gate's graceful-degrade ``TERMINAL_COVERAGE_UNVERIFIED`` path owns
      that case — we have nothing to anchor to).
    * Otherwise, for each terminal whose existing ``chapter`` back-pointer
      does NOT resolve against the structure (or is absent), assign one of
      the structure's real chapter IDs round-robin (in document order) so
      terminals spread across the available chapters deterministically.

    Returns ``(terminals, assigned_terminal_ids)`` — a fresh terminal list
    (input never mutated) plus the IDs that received a new/repaired
    back-pointer (for logging).
    """
    terminals = [dict(t) for t in terminal_objectives if isinstance(t, Mapping)]
    if is_co_bearing(chapter_objectives):
        return terminals, []
    ordered_ids = _structure_chapter_ids_ordered(textbook_structure)
    if not ordered_ids:
        return terminals, []
    resolvable = {_normalise_chapter_key(cid) for cid in ordered_ids}
    resolvable.discard(None)

    assigned: List[str] = []
    rr = 0
    for to in terminals:
        existing = terminal_chapter_ref(to)
        if existing is not None and _normalise_chapter_key(existing) in resolvable:
            continue  # already resolvable — leave verbatim
        to["chapter"] = ordered_ids[rr % len(ordered_ids)]
        rr += 1
        tid = to.get("id")
        assigned.append(str(tid) if tid is not None else "?")
    return terminals, assigned


def prune_orphan_terminals(
    terminal_objectives: Sequence[Mapping[str, Any]],
    chapter_objectives: Any,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Prune orphan terminals from a **CO-bearing** objectives set.

    Returns ``(kept_terminals, pruned_terminal_ids)``. Only the CO-bearing
    roll-up case is pruned by construction — that is the case where the
    objectives set alone is sufficient to prove a terminal is uncoverable
    (no chapter objective rolls up to it). A CO-less course is returned
    untouched: its terminals are self-contained teaching units whose
    coverage is content-driven, so pruning them at synthesis time would
    silently delete valid course material. The Layer-B gate
    (with the textbook structure in hand) adjudicates the CO-less case.

    This keeps CO-bearing + CO-less runs byte-identical (a consistent
    CO-bearing course prunes nothing; a CO-less course is untouched).
    """
    terminals = [dict(t) for t in terminal_objectives if isinstance(t, Mapping)]
    if not is_co_bearing(chapter_objectives):
        return terminals, []
    rolled = rolled_up_terminal_ids(chapter_objectives)
    if not rolled:
        # CO-bearing but zero COs carry any parent-terminal key ⟹ the
        # roll-up model is not in use; orphanhood is unprovable from the
        # objectives set alone; prune nothing (Layer-B gate adjudicates
        # loudly). Without this guard EVERY terminal is pruned on the
        # deterministic-synthesizer / Stage-2 / _normalize_objective_entry
        # paths (none of which emit parent-terminal keys), silently
        # shipping zero terminal objectives.
        return terminals, []
    kept: List[Dict[str, Any]] = []
    pruned: List[str] = []
    for to in terminals:
        tid = _terminal_id(to)
        if tid is not None and tid not in rolled:
            pruned.append(str(to.get("id")))
            continue
        kept.append(to)
    return kept, pruned


__all__ = [
    "co_parent_terminal",
    "terminal_chapter_ref",
    "flatten_chapter_objectives",
    "rolled_up_terminal_ids",
    "is_co_bearing",
    "find_orphan_terminals",
    "attach_terminal_chapter_refs",
    "prune_orphan_terminals",
]
