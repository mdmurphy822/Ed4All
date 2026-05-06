"""GPT Feedback v2 Wave 1 (W1.C) — course-level final-status helpers.

Two pure helpers, no production callers in Wave 1:

  * :func:`load_course_status_schema` — loads
    ``schemas/governance/course_status.schema.json`` and returns the JSON-Schema
    dict. Symmetric with :func:`lib.ontology.taxonomy.load_taxonomy` (same
    cached single-shot loader pattern).

  * :func:`compose_course_status` — placeholder decision-logic helper that takes
    a mapping of ``{arrow_name: promotion_decision}`` and returns one of the
    five GPT 5-value enum members. Wave 1 implements only the ``failed`` and
    ``non_certified_archive`` branches; the three ``certified_*`` branches
    raise :class:`NotImplementedError` until Wave 3 wires the gate-by-gate
    compositional decision logic against the live promotion-chain aggregator.

The enum values are the five members of
``schemas/governance/course_status.schema.json``:

    certified_accessible
    certified_instructional
    certified_trainable
    non_certified_archive
    failed

See ``plans/gpt-feedback-2-wave1-schemas-2026-05.md`` § "Worker W1.C" for the
authoring rationale and ``schemas/ONTOLOGY.md`` for the broader governance
posture.
"""

from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping

__all__ = [
    "load_course_status_schema",
    "compose_course_status",
]


# ---------------------------------------------------------------------------
# Schema path + cache
# ---------------------------------------------------------------------------

_COURSE_STATUS_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "schemas"
    / "governance"
    / "course_status.schema.json"
)


@lru_cache(maxsize=1)
def _load_course_status_schema_cached() -> Dict[str, Any]:
    """Cached single-shot loader for the course-status schema."""
    if not _COURSE_STATUS_SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Course-status schema not found at {_COURSE_STATUS_SCHEMA_PATH}. "
            "Expected canonical copy from Wave 1 (Worker W1.C)."
        )
    with open(_COURSE_STATUS_SCHEMA_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "enum" not in data:
        raise ValueError(
            f"Malformed course-status schema at {_COURSE_STATUS_SCHEMA_PATH}: "
            "missing top-level 'enum' member."
        )
    return data


def load_course_status_schema() -> Dict[str, Any]:
    """Load and return the course-status JSON-Schema dict.

    Returns a defensive deep copy of the cached schema so callers may
    mutate the returned dict (including its nested ``enum`` list) without
    polluting the cache. Mirrors the pattern in
    :func:`lib.ontology.taxonomy.load_taxonomy`.

    Raises:
        FileNotFoundError: when the schema file is missing.
        ValueError: when the schema shape is invalid (missing ``enum``).
    """
    return copy.deepcopy(_load_course_status_schema_cached())


# ---------------------------------------------------------------------------
# Decision-logic stub
# ---------------------------------------------------------------------------

# Sentinel boundary between the "accessible+instructional" arrows (1-5) and the
# "trainable" arrows (6+). Wave 1 only knows the 5-arrow boundary; the canonical
# 9-arrow chain identifies arrows by name in Wave 3 when the promotion-chain
# aggregator lands. For Wave 1 the helper accepts an arbitrary mapping and
# treats arrows up to this index (inclusive) as the "accessible+instructional"
# slice.
_ACCESSIBLE_INSTRUCTIONAL_ARROW_BOUNDARY = 5

# Promotion-decision values that count as a passing arrow. Wave 1 treats every
# other value (including ``"missing"`` / absent / ``"fail"``) as not-passing.
_PASSING_DECISIONS = frozenset({"pass", "warn"})

# Promotion-decision values that count as a definite failure for the
# course-level rollup. Any arrow with one of these values short-circuits the
# course to ``"failed"``.
_FAILING_DECISIONS = frozenset({"fail"})


def compose_course_status(per_arrow_decisions: Mapping[str, str]) -> str:
    """Compose a course-level final-status enum from per-arrow decisions.

    Wave 1 implements only the two simplest branches:

      * Returns ``"failed"`` when ANY arrow has
        ``promotion_decision == "fail"``.
      * Returns ``"non_certified_archive"`` when every arrow up to a
        5-arrow boundary passes (``"pass"`` or ``"warn"``) and arrows
        beyond that boundary are missing or non-passing.

    The three ``certified_*`` branches require gate-by-gate compositional
    logic against the live promotion-chain aggregator output (Wave 3 G1)
    and are not implemented in Wave 1; calls that would route into those
    branches raise :class:`NotImplementedError` with a message pointing at
    Wave 3.

    Arrow naming convention: keys are the per-arrow names (today informally
    ``"arrow1"`` … ``"arrow9"``); values are the per-arrow promotion
    decisions, drawn from the same enum the PhaseOutput
    ``promotion_decision`` field carries (``"pass" | "warn" | "fail" |
    "escalate"``) plus a Wave-1 sentinel ``"missing"`` for arrows that have
    not yet emitted a decision. The mapping order does NOT matter; the
    helper sorts keys by their numeric suffix when present and falls back
    to lexicographic sort otherwise.

    Args:
        per_arrow_decisions: mapping of ``{arrow_name: promotion_decision}``.

    Returns:
        One of the five enum values from
        ``schemas/governance/course_status.schema.json``.

    Raises:
        NotImplementedError: for the three ``certified_*`` branches.
            Message points at Wave 3 (where the gate-by-gate decision
            logic lands alongside the promotion-chain aggregator).
    """
    # Defensive copy so callers can mutate the input afterward.
    decisions: Dict[str, str] = dict(per_arrow_decisions)

    # --- Branch 1: any explicit fail short-circuits to "failed". -----------
    if any(value in _FAILING_DECISIONS for value in decisions.values()):
        return "failed"

    # --- Branch 2: arrows 1-5 pass, arrows 6+ missing/failed ---------------
    # Partition the arrow keys by their numeric suffix when discoverable.
    # Keys without a numeric suffix are treated as "outside the 1-5 slice"
    # so the helper degrades gracefully on non-canonical inputs.
    def _arrow_index(name: str) -> int:
        # Strip the canonical "arrow" prefix when present and try to parse
        # the trailing digits. Anything non-parseable sorts beyond the
        # 1-5 boundary so it won't accidentally satisfy the
        # "non_certified_archive" precondition.
        suffix = name[len("arrow"):] if name.startswith("arrow") else name
        try:
            return int(suffix)
        except (TypeError, ValueError):
            return _ACCESSIBLE_INSTRUCTIONAL_ARROW_BOUNDARY + 1

    accessible_slice = {
        name: value
        for name, value in decisions.items()
        if _arrow_index(name) <= _ACCESSIBLE_INSTRUCTIONAL_ARROW_BOUNDARY
    }
    trainable_slice = {
        name: value
        for name, value in decisions.items()
        if _arrow_index(name) > _ACCESSIBLE_INSTRUCTIONAL_ARROW_BOUNDARY
    }

    accessible_all_pass = bool(accessible_slice) and all(
        value in _PASSING_DECISIONS for value in accessible_slice.values()
    )
    trainable_any_pass = any(
        value in _PASSING_DECISIONS for value in trainable_slice.values()
    )

    if accessible_all_pass and not trainable_any_pass:
        return "non_certified_archive"

    # --- Branches 3-5: certified_* (Wave 3) -------------------------------
    # When arrows 1-5 pass AND any arrow 6+ also passes, the course is a
    # candidate for one of the three certified_* tiers. The exact tier
    # selection requires gate-by-gate compositional logic that depends on
    # the Wave-3 promotion-chain aggregator output (e.g. arrow-6 alone =>
    # certified_accessible vs arrows 6+7 => certified_instructional vs
    # full chain => certified_trainable). Wave 1 deliberately stops here.
    raise NotImplementedError(
        "compose_course_status: certified_* branch routing is deferred to "
        "Wave 3 (governance G1 — promotion-chain aggregator). Wave 1 only "
        "implements the 'failed' and 'non_certified_archive' branches. See "
        "plans/gpt-feedback-2-wave1-schemas-2026-05.md § 'Worker W1.C' for "
        "the deferral rationale."
    )
