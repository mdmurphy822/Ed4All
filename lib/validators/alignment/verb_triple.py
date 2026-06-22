"""IB3.1 — Constructive-alignment verb-triple resolver helper.

The shared, pure-function verb-band reasoning that every IB3 alignment gate
calls (IB3.2 triple equality, IB3.3 evidence form, IB3.4 anchored rubric,
IB3.5 triangle completeness). The framework's single most load-bearing
validity rule is ``objective-verb = activity-verb = assessment-verb`` checked
against the Bloom band — and that band reasoning MUST be computed ONE way, so
this module is the single source of truth.

NO new dependencies: string ops + :mod:`lib.ontology.bloom` (``BLOOM_LEVELS``,
``detect_bloom_level``, ``detect_bloom_verbs``). No Bloom tables are
re-implemented here — the verb→level mapping is reused wholesale.

Verb-band semantics: two verbs "share a band" when they resolve to the same
canonical Bloom level (``BLOOM_LEVELS.index`` equality). ``bloom_below`` is the
Alignment-cap-at-1 trigger (assessment Bloom strictly below objective Bloom).
``is_apply_plus`` is the Apply-and-above gate for the evidence-form check.

All functions are tolerant of missing / unresolvable inputs — they return
``(None, None)`` / ``False`` rather than raising, so a gate that wires this in
behind a flag degrades cleanly on legacy / thin block shapes.
"""
from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Tuple

from lib.ontology.bloom import (
    BLOOM_LEVELS,
    detect_bloom_level,
    detect_bloom_verbs,
)

__all__ = [
    "ALIGNMENT_VERB_TRIPLE_ENV",
    "alignment_verb_triple_enabled",
    "resolve_objective_verb_level",
    "resolve_block_verb_level",
    "verbs_share_band",
    "bloom_below",
    "is_apply_plus",
]


# IB3 master gate flag. Default OFF → every IB3 alignment check (the
# verb-triple equality axis, the evidence-form check, the anchored-rubric
# validator, the triangle-completeness validator) no-ops and the
# ``Block.anchored_rubric`` field stays None + hash-excluded so snapshots are
# byte-identical. Falsey / garbage → off (parse-with-fallback). Read each call
# so tests can toggle it.
ALIGNMENT_VERB_TRIPLE_ENV = "ED4ALL_ALIGNMENT_VERB_TRIPLE"
_ALIGNMENT_TRUTHY = frozenset({"1", "true", "yes", "on"})


def alignment_verb_triple_enabled() -> bool:
    """True iff ``ED4ALL_ALIGNMENT_VERB_TRIPLE`` is truthy (read each call)."""
    return (
        os.environ.get(ALIGNMENT_VERB_TRIPLE_ENV, "").strip().lower()
        in _ALIGNMENT_TRUTHY
    )


# Canonical Bloom-level → index map (low→high). Mirrors the
# ``_BLOOM_INDEX`` re-export in ``block_objective_delivery`` but built
# locally off the canonical ``BLOOM_LEVELS`` tuple so this module has no
# import coupling to the validator it is consumed by.
_BLOOM_INDEX = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}

#: Apply-and-above threshold index (the evidence-form / performance bar).
_APPLY_INDEX = _BLOOM_INDEX["apply"]


def _block_attr(block: Any, key: str) -> Any:
    """Read ``block.<key>`` (dataclass) OR ``block[<key>]`` (dict).

    Mirrors the shape-tolerant accessor in
    :mod:`lib.validators.block_objective_delivery` so this helper works
    on both the frozen ``Block`` dataclass and the outline-tier block dict.
    """
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, Mapping):
        return block.get(key)
    return None


def _norm_level(value: Any) -> Optional[str]:
    """Lowercase + strip a Bloom level to a canonical member or None."""
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    return s if s in _BLOOM_INDEX else None


def _norm_verb(value: Any) -> Optional[str]:
    """Lowercase + strip a verb to a non-empty string or None."""
    if not isinstance(value, str):
        return None
    s = value.strip().lower()
    return s or None


def resolve_objective_verb_level(
    lo: Mapping[str, Any],
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ``(verb, level)`` for a learning objective.

    Canonical resolution (mirrors ``abcd_objective._bloom_level``):

    1. Prefer the structured ABCD surface — ``abcd.behavior.verb`` for the
       verb and the objective's ``bloom_level`` / ``bloomLevel`` for the
       level. This is the authored truth and what
       :class:`lib.validators.abcd_objective.AbcdObjectiveValidator` audits.
    2. Fall back to ``bloom_verb`` / ``bloom_level`` flat keys.
    3. Fall back to :func:`lib.ontology.bloom.detect_bloom_level` over the
       objective ``statement`` when neither structured surface resolves.

    Both elements lowercased. Returns ``(None, None)`` when unresolvable.
    The verb may be present while the level is not (and vice-versa); each is
    resolved independently and a missing element stays None.
    """
    if not isinstance(lo, Mapping):
        return (None, None)

    verb: Optional[str] = None
    level: Optional[str] = None

    # (1) ABCD surface — the canonical authored truth.
    abcd = lo.get("abcd")
    if isinstance(abcd, Mapping):
        behavior = abcd.get("behavior")
        if isinstance(behavior, Mapping):
            verb = _norm_verb(behavior.get("verb"))

    # Level: bloom_level / bloomLevel (mirror abcd_objective._bloom_level).
    level = _norm_level(lo.get("bloom_level"))
    if level is None:
        level = _norm_level(lo.get("bloomLevel"))

    # (2) Flat bloom_verb fallback for the verb.
    if verb is None:
        verb = _norm_verb(lo.get("bloom_verb"))
    if verb is None:
        verb = _norm_verb(lo.get("bloomVerb"))

    # (3) Statement detection fallback — fills whichever element is still
    # missing without overriding an authored one.
    if verb is None or level is None:
        statement = lo.get("statement")
        if isinstance(statement, str) and statement.strip():
            det_level, det_verb = detect_bloom_level(statement)
            if level is None:
                level = _norm_level(det_level)
            if verb is None:
                verb = _norm_verb(det_verb)

    return (verb, level)


def resolve_block_verb_level(
    block: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ``(verb, level)`` — the verb the learner actually *does*.

    For an activity / interaction or assessment block, the cognitive demand
    the block exercises is what the framework's verb-triple compares against
    the objective. Resolution order:

    1. Prefer the IB1 ``interaction`` slot text (the framework's
       slot-addressable interaction verb); fall back to the ``content`` body
       text when the interaction slot is empty (degraded prose-heuristic
       mode — IB3 lands degraded-against-IB1 until the slot is populated, per
       the plan §4 dependency note).
    2. Over that text, run :func:`lib.ontology.bloom.detect_bloom_verbs` and
       pick the HIGHEST-Bloom verb present (the strongest cognitive demand
       the block actually exercises — a block that both "lists" and
       "evaluates" exercises evaluate).
    3. Fall back to the block's declared ``bloom_verb`` / ``bloom_level`` when
       no verb is detectable in the text.

    Returns ``(None, None)`` when no text and no declared bloom data resolve.
    """
    text_parts = []
    interaction = _block_attr(block, "interaction")
    if isinstance(interaction, str) and interaction.strip():
        text_parts.append(interaction)
    else:
        content = _block_attr(block, "content")
        if isinstance(content, str) and content.strip():
            text_parts.append(content)
        elif isinstance(content, Mapping):
            # Pull stem / prompt / question text from a structured payload.
            for key in ("stem", "prompt", "question", "text", "body"):
                val = content.get(key)
                if isinstance(val, str) and val.strip():
                    text_parts.append(val)

    text = "\n".join(text_parts)

    detected_verb: Optional[str] = None
    detected_level: Optional[str] = None
    if text.strip():
        # detect_bloom_verbs returns longest-first, higher-level ties first.
        # We want the HIGHEST-Bloom verb present → pick by level index.
        matches = detect_bloom_verbs(text)
        best_idx = -1
        for level, verb in matches:
            idx = _BLOOM_INDEX.get(level, -1)
            if idx > best_idx:
                best_idx = idx
                detected_level = level
                detected_verb = verb

    if detected_verb is not None and detected_level is not None:
        return (detected_verb, detected_level)

    # Fall back to the block's declared bloom metadata.
    declared_verb = _norm_verb(_block_attr(block, "bloom_verb"))
    declared_level = _norm_level(_block_attr(block, "bloom_level"))
    # Prefer any text-detected element over the declared one, but never
    # return a half-resolved pair when the text gave us nothing.
    verb = detected_verb if detected_verb is not None else declared_verb
    level = detected_level if detected_level is not None else declared_level
    return (verb, level)


def verbs_share_band(
    level_a: Optional[str],
    level_b: Optional[str],
) -> bool:
    """True iff both levels resolve to the SAME canonical Bloom level.

    The framework's verb-by-verb equality is enforced at the BAND (level)
    granularity — "critique" and "judge" are both ``evaluate`` and so share a
    band, but "list" (``remember``) and "critique" (``evaluate``) do not.
    Returns ``False`` when either level is None / non-canonical (a level we
    can't resolve cannot be asserted to share a band — fail safe to mismatch
    is wrong here; an unresolvable pair is "unverifiable" upstream, so the
    caller must guard on None before treating a False as a mismatch).
    """
    a = _norm_level(level_a)
    b = _norm_level(level_b)
    if a is None or b is None:
        return False
    return _BLOOM_INDEX[a] == _BLOOM_INDEX[b]


def bloom_below(
    assessment_level: Optional[str],
    objective_level: Optional[str],
) -> bool:
    """True iff ``assessment_level`` is strictly below ``objective_level``.

    The Alignment-cap-at-1 trigger: an assessment certifying a LOWER
    cognitive demand than the objective declares cannot validly certify it
    (a recall MCQ for a Create objective). Returns ``False`` when either
    level is unresolvable (can't assert a cap on a level we can't place).
    """
    a = _norm_level(assessment_level)
    o = _norm_level(objective_level)
    if a is None or o is None:
        return False
    return _BLOOM_INDEX[a] < _BLOOM_INDEX[o]


def is_apply_plus(level: Optional[str]) -> bool:
    """True iff ``level`` is Apply-or-above (the performance-evidence bar).

    Apply / Analyze / Evaluate / Create require performance/scenario/project
    evidence; Remember / Understand admit recall items. Returns ``False`` for
    an unresolvable level.
    """
    norm = _norm_level(level)
    if norm is None:
        return False
    return _BLOOM_INDEX[norm] >= _APPLY_INDEX
