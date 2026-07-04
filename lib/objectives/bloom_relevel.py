"""Feature 1 — deterministic Bloom-level RELEVEL (label honesty).

The stage-2 window synthesis and reconciliation emit each CO/TO with a declared
``bloom_level`` field AND a main action verb (in the statement and, when present,
in ``abcd.behavior.verb``). The 7B frequently drifts these apart: it authors an
``apply``-verb statement ("Apply the order of operations …") but stamps
``bloom_level: understand`` — a label that lies about the objective's cognitive
demand. The ``abcd_verb_alignment`` gate CATCHES this disagreement against the
canonical verb table (``lib/ontology/bloom.py``), but the synthesis path never
CORRECTS it.

This pass is the deterministic cure. For every objective whose declared
``bloom_level`` disagrees with the level the objective's MAIN VERB belongs to in
the canonical verb table, it re-derives ``bloom_level`` from that verb — the same
table ``abcd_verb_alignment`` validates against. The verb is read from
``abcd.behavior.verb`` when present, else detected from the statement via
:func:`lib.ontology.bloom.detect_bloom_level`.

HARD CONTRACT:
  * STATEMENTS NEVER CHANGE — only the ``bloom_level`` field is rewritten.
  * A relevel fires ONLY when the declared level is a valid Bloom level AND the
    verb resolves to a DIFFERENT valid level. A verb absent from the canonical
    table (no derivable level) → no change. A missing / invalid declared level →
    no change (there is nothing to "disagree").
  * DETERMINISTIC — no LLM, no embeddings.

Default OFF (opt-in via ``ED4ALL_OBJECTIVE_BLOOM_RELEVEL``) — default-off is
byte-identical to today (no objective's ``bloom_level`` is touched, no capture
fires).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: Feature 1 — deterministic Bloom-relevel toggle. Default OFF (opt-in).
_DEFAULT_BLOOM_RELEVEL = False
ENV_BLOOM_RELEVEL = "ED4ALL_OBJECTIVE_BLOOM_RELEVEL"


@dataclass
class RelevelResult:
    """Outcome of the deterministic Bloom-relevel pass."""

    objectives: List[Dict[str, Any]] = field(default_factory=list)
    scanned_count: int = 0
    releveled_count: int = 0
    #: Per-releveled-objective audit rows: {id, verb, old_level, new_level}.
    changes: List[Dict[str, Any]] = field(default_factory=list)
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scanned_count": self.scanned_count,
            "releveled_count": self.releveled_count,
            "changes": [dict(c) for c in self.changes],
            "available": self.available,
        }


def resolve_bloom_relevel(enabled: Optional[bool] = None) -> bool:
    """Resolve the Feature-1 relevel toggle: arg → env → default (OFF).

    Truthy env values (``1``/``true``/``yes``/``on``) enable; anything else
    (incl. garbage / unset) → the default. Mirrors the parse-with-fallback
    posture of the sibling prong resolvers.
    """
    if enabled is not None:
        return bool(enabled)
    raw = str(os.environ.get(ENV_BLOOM_RELEVEL, "")).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    return _DEFAULT_BLOOM_RELEVEL


@lru_cache(maxsize=1)
def _verb_to_level() -> Dict[str, str]:
    """Reverse the canonical ``level → {verbs}`` table into ``verb → level``.

    Sourced from :func:`lib.ontology.bloom.get_verbs` (the SAME table
    ``abcd_verb_alignment`` validates against). The canonical list has no verb
    at two levels (verified in ``lib/ontology/bloom.py``), but if one ever were,
    the HIGHER-order level wins — matching ``detect_bloom_level``'s tie rule.
    Cached; a taxonomy-load failure returns an empty map (relevel then no-ops).
    """
    try:
        from lib.ontology.bloom import BLOOM_LEVELS, get_verbs

        per_level = get_verbs()
    except Exception:  # noqa: BLE001 — taxonomy load must never break the pass
        return {}
    out: Dict[str, str] = {}
    # Iterate low → high so the higher level wins on the (defensive) tie.
    for level in BLOOM_LEVELS:
        for verb in per_level.get(level, set()):
            out[str(verb).strip().lower()] = level
    return out


def _valid_levels() -> frozenset:
    try:
        from lib.ontology.bloom import BLOOM_LEVELS

        return frozenset(BLOOM_LEVELS)
    except Exception:  # noqa: BLE001
        return frozenset(
            {"remember", "understand", "apply", "analyze", "evaluate", "create"}
        )


def _abcd_verb(objective: Dict[str, Any]) -> str:
    """Read ``abcd.behavior.verb`` defensively (lowercased), else ``""``."""
    abcd = objective.get("abcd")
    if not isinstance(abcd, dict):
        return ""
    behavior = abcd.get("behavior")
    if not isinstance(behavior, dict):
        return ""
    return str(behavior.get("verb") or "").strip().lower()


def _statement(objective: Dict[str, Any]) -> str:
    return str(objective.get("statement") or objective.get("text") or "").strip()


def derive_level(objective: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(canonical_level, verb)`` for an objective's MAIN verb, or
    ``(None, None)`` when no verb resolves to a canonical level.

    Precedence: ``abcd.behavior.verb`` (when present AND in the canonical verb
    table) wins; otherwise the statement's main verb is detected via
    :func:`lib.ontology.bloom.detect_bloom_level`.
    """
    verb_map = _verb_to_level()
    abcd_verb = _abcd_verb(objective)
    if abcd_verb and abcd_verb in verb_map:
        return verb_map[abcd_verb], abcd_verb
    try:
        from lib.ontology.bloom import detect_bloom_level

        level, verb = detect_bloom_level(_statement(objective))
    except Exception:  # noqa: BLE001
        return (None, None)
    if level:
        return level, verb
    return (None, None)


def relevel_objectives(
    objectives: List[Dict[str, Any]],
    *,
    enabled: Optional[bool] = None,
    capture: Optional[Any] = None,
) -> RelevelResult:
    """Re-derive each objective's ``bloom_level`` from its main verb (opt-in).

    Mutates the objective dicts IN PLACE (``bloom_level`` only — statements are
    never touched). Emits one ``bloom_level_assignment`` decision-capture event
    per releveled objective (dynamic rationale: id, verb, old→new level).

    Default-off / empty input → no-op (``available=False``, objectives
    untouched).
    """
    if not resolve_bloom_relevel(enabled):
        return RelevelResult(objectives=objectives, available=False)
    if not objectives:
        return RelevelResult(objectives=objectives, available=True)

    valid = _valid_levels()
    scanned = 0
    releveled = 0
    changes: List[Dict[str, Any]] = []
    for obj in objectives:
        if not isinstance(obj, dict):
            continue
        declared = str(obj.get("bloom_level") or "").strip().lower()
        if declared not in valid:
            # No valid declared level → nothing to "disagree" with; skip.
            continue
        scanned += 1
        derived_level, verb = derive_level(obj)
        if not derived_level or derived_level not in valid:
            continue
        if derived_level == declared:
            continue
        # Disagreement — relevel the field only (statement untouched).
        obj["bloom_level"] = derived_level
        releveled += 1
        row = {
            "id": str(obj.get("id") or obj.get("co_id") or ""),
            "verb": verb or "",
            "old_level": declared,
            "new_level": derived_level,
        }
        changes.append(row)
        _emit_relevel_capture(capture, row=row)

    return RelevelResult(
        objectives=objectives,
        scanned_count=scanned,
        releveled_count=releveled,
        changes=changes,
        available=True,
    )


def _emit_relevel_capture(capture: Any, *, row: Dict[str, Any]) -> None:
    """Best-effort ``bloom_level_assignment`` capture (never raises).

    Reuses the existing ``bloom_level_assignment`` decision_type (no schema
    edit). Rationale interpolates the objective id, the resolving verb, and the
    old→new level so the relabel is replayable post-hoc.
    """
    if capture is None:
        return
    try:
        capture.log_decision(
            decision_type="bloom_level_assignment",
            decision=(
                f"releveled objective {row['id'] or '(unminted)'} "
                f"{row['old_level']} -> {row['new_level']}"
            ),
            rationale=(
                f"Feature 1 (deterministic Bloom relevel): objective "
                f"{row['id'] or '(unminted)'} declared bloom_level "
                f"'{row['old_level']}' but its main verb '{row['verb']}' "
                f"belongs to level '{row['new_level']}' in the canonical verb "
                f"table (lib/ontology/bloom.py — the table abcd_verb_alignment "
                f"validates against). Re-derived bloom_level from the verb for "
                f"label honesty; the STATEMENT is unchanged (only the "
                f"bloom_level field). Deterministic, no LLM."
            ),
            alternatives_considered=[
                "keep the mislabelled bloom_level (abcd_verb_alignment then "
                "flags the disagreement but the label stays wrong)",
            ],
        )
    except Exception as exc:  # noqa: BLE001 — capture is best-effort
        logger.debug(
            "bloom_level_assignment (relevel) capture failed (%s); continuing",
            exc,
        )


__all__ = [
    "RelevelResult",
    "relevel_objectives",
    "derive_level",
    "resolve_bloom_relevel",
    "_DEFAULT_BLOOM_RELEVEL",
    "ENV_BLOOM_RELEVEL",
]
