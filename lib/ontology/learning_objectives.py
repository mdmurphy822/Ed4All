"""Canonical Learning-Objective ID helper (Wave 24).

Single source of truth for minting + validating learning-objective IDs
across the Ed4All pipeline. Before this module landed, two disjoint
schemes existed:

  * ``TO-NN`` / ``CO-NN`` (terminal / chapter) minted by
    ``_content_gen_helpers.synthesize_objectives_from_topics`` → emitted
    to Courseforge JSON-LD → harvested by Trainforge as
    ``chunks[*].learning_outcome_refs``.
  * ``{COURSE}_OBJ_N`` minted by ``create_course_project`` → threaded
    into ``phase_outputs.objective_extraction.objective_ids`` → routed
    into assessment generation → every resulting
    ``assessments.json.questions[].objective_id`` was a phantom never
    referenced by any HTML page → 896 broken refs downstream.

This helper eliminates the second scheme entirely. All mint sites route
through ``mint_lo_id`` + ``split_terminal_chapter`` so a single canonical
pattern (``^[A-Z]{2,}-\\d{2,}$``) owns LO identity end-to-end.

The canonical pattern originates in
``schemas/knowledge/courseforge_jsonld_v1.schema.json::learningObjectives[].id``;
this module is the Python mirror.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Tuple

# ---------------------------------------------------------------------------
# Canonical regex + constants
# ---------------------------------------------------------------------------

#: The LO ID pattern enforced by ``courseforge_jsonld_v1.schema.json``.
#: Must stay byte-identical with the schema pattern.
LO_ID_PATTERN = re.compile(r"^[A-Z]{2,}-\d{2,}$")

#: Path to the externalized prefix-to-hierarchy taxonomy (Wave 133b).
_LO_HIERARCHY_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "taxonomies"
    / "lo_hierarchy.json"
)


@lru_cache(maxsize=1)
def _load_prefix_map() -> Dict[str, str]:
    """Load the canonical prefix-to-hierarchy mapping (Wave 133b).

    Sources from ``schemas/taxonomies/lo_hierarchy.json``. Cached so the
    JSON read is one-shot per process. Wave 133b moved this off the
    hardcoded ``{"TO": "terminal", "CO": "chapter"}`` literal so future
    course families using ``MO/PO/UO/LO/SO`` do NOT need a code edit —
    the JSON is the single source of truth.

    Raises:
        FileNotFoundError: if the taxonomy file is missing.
        ValueError: if the JSON is malformed (missing or non-dict
            ``prefixes`` root).
    """
    if not _LO_HIERARCHY_TAXONOMY_PATH.exists():
        raise FileNotFoundError(
            f"LO hierarchy taxonomy not found at "
            f"{_LO_HIERARCHY_TAXONOMY_PATH}. Wave 133b expects this "
            f"file to exist; check your worktree."
        )
    with open(_LO_HIERARCHY_TAXONOMY_PATH, encoding="utf-8") as f:
        data = json.load(f)
    prefixes = data.get("prefixes")
    if not isinstance(prefixes, dict) or not prefixes:
        raise ValueError(
            f"Malformed LO hierarchy taxonomy at "
            f"{_LO_HIERARCHY_TAXONOMY_PATH}: missing or non-dict "
            f"'prefixes' root."
        )
    return dict(prefixes)


Hierarchy = Literal[
    "terminal",
    "chapter",
    "module",
    "program",
    "unit",
    "lesson",
    "session",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mint_lo_id(hierarchy: Hierarchy, counter: int) -> str:
    """Mint a canonical LO ID.

    Args:
        hierarchy: ``"terminal"`` → ``TO-NN``, ``"chapter"`` → ``CO-NN``.
        counter: 1-indexed counter. Values < 1 raise ``ValueError``.

    Returns:
        A zero-padded ID that matches ``LO_ID_PATTERN``.

    Raises:
        ValueError: On invalid hierarchy or non-positive counter.
    """
    if counter < 1:
        raise ValueError(f"LO counter must be >= 1, got {counter}")
    if hierarchy == "terminal":
        prefix = "TO"
    elif hierarchy == "chapter":
        prefix = "CO"
    else:
        raise ValueError(
            f"hierarchy must be 'terminal' or 'chapter', got {hierarchy!r}"
        )
    # Always at least 2 digits, expands if needed (e.g., counter=100 → '100').
    width = max(2, len(str(counter)))
    return f"{prefix}-{str(counter).zfill(width)}"


def validate_lo_id(s: str) -> bool:
    """Return True iff ``s`` matches the canonical LO ID pattern.

    Callers that need to fail closed on invalid IDs should pair this
    with an explicit ``raise``; this helper is a boolean check only.
    """
    if not isinstance(s, str):
        return False
    return bool(LO_ID_PATTERN.match(s))


def hierarchy_from_id(lo_id: str) -> Hierarchy:
    """Return the hierarchy level embedded in a canonical LO ID.

    Args:
        lo_id: Must match ``LO_ID_PATTERN``. Recognized prefixes are sourced
               from ``schemas/taxonomies/lo_hierarchy.json`` (Wave 133b);
               anything else raises.

    Raises:
        ValueError: If ``lo_id`` is not canonical or carries an unknown prefix.
    """
    if not validate_lo_id(lo_id):
        raise ValueError(
            f"LO ID does not match canonical pattern [A-Z]{{2,}}-\\d{{2,}}: "
            f"{lo_id!r}"
        )
    prefix = lo_id.split("-", 1)[0]
    prefix_map = _load_prefix_map()
    hierarchy = prefix_map.get(prefix)
    if hierarchy is None:
        raise ValueError(
            f"LO prefix {prefix!r} is not a recognized hierarchy "
            f"(expected one of {sorted(prefix_map.keys())})"
        )
    return hierarchy


def split_terminal_chapter(
    total: int,
    *,
    terminal_ratio: float = 0.25,
    min_terminal: int = 2,
    max_terminal: int = 6,
) -> Tuple[int, int]:
    """Split ``total`` objectives into (terminal_count, chapter_count).

    Replaces the hard-coded ``if to_counter <= 2`` magic in
    ``_content_gen_helpers.synthesize_objectives_from_topics`` with a
    configurable ratio and explicit bounds.

    Args:
        total: Total LO count to split. Negative values return (0, 0).
        terminal_ratio: Fraction that should become terminal outcomes.
                        Default 0.25 — roughly the "one terminal per
                        four chapters" convention from the existing
                        Courseforge assets.
        min_terminal: Floor on terminal count. Default 2 — matches the
                      historical 2-TO minimum so v0.1.x fixtures stay
                      stable. Clamped to ``total`` when ``total`` is
                      smaller.
        max_terminal: Ceiling on terminal count. Default 6 — avoids
                      drowning the COs when the corpus is rich.

    Returns:
        ``(terminal_count, chapter_count)`` where
        ``terminal_count + chapter_count == total``.
    """
    if total <= 0:
        return (0, 0)
    raw = int(round(total * terminal_ratio))
    terminal = max(min_terminal, min(max_terminal, raw))
    terminal = min(terminal, total)  # never exceed total
    chapter = total - terminal
    return (terminal, chapter)


def assign_lo_ids(
    total: int,
    *,
    terminal_ratio: float = 0.25,
    min_terminal: int = 2,
    max_terminal: int = 6,
) -> List[Tuple[str, Hierarchy]]:
    """Return ``total`` LO IDs in emit order with their hierarchy labels.

    Convenience wrapper that composes :func:`split_terminal_chapter` with
    :func:`mint_lo_id`. The emit order is terminal-first (all ``TO-*``
    in one block), chapter-next (all ``CO-*``) — matches the historical
    order observed in Courseforge JSON-LD payloads.

    Returns:
        List of ``(lo_id, hierarchy_level)`` tuples. Empty when
        ``total <= 0``.
    """
    terminal_count, chapter_count = split_terminal_chapter(
        total,
        terminal_ratio=terminal_ratio,
        min_terminal=min_terminal,
        max_terminal=max_terminal,
    )
    out: List[Tuple[str, Hierarchy]] = []
    for i in range(1, terminal_count + 1):
        out.append((mint_lo_id("terminal", i), "terminal"))
    for i in range(1, chapter_count + 1):
        out.append((mint_lo_id("chapter", i), "chapter"))
    return out


__all__ = [
    "LO_ID_PATTERN",
    "Hierarchy",
    "assign_lo_ids",
    "hierarchy_from_id",
    "mint_lo_id",
    "split_terminal_chapter",
    "validate_lo_id",
]
