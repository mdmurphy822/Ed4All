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
    "verb_triple_misaligned",
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


def _content_attr(block: Any, key: str) -> Any:
    """Read ``key`` off the block's structured ``content`` payload.

    Outline-tier blocks carry their declared ``bloom_level`` / ``bloom_verb``
    INSIDE the ``content`` dict (e.g. ``content["bloom_level"]``), not as a
    top-level attr. This pulls from there so the declared-bloom anchor below
    resolves on the real outline-tier shape.
    """
    content = _block_attr(block, "content")
    if isinstance(content, Mapping):
        return content.get(key)
    return None


def resolve_block_verb_level(
    block: Any,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ``(verb, level)`` — the verb the learner actually *does*.

    For an activity / interaction or assessment block, the cognitive demand
    the block exercises is what the framework's verb-triple compares against
    the objective.

    IB3 recalibration (2-corpus calibration gap-map). The original resolver
    led with a raw prose scan that picked the HIGHEST-Bloom verb anywhere in
    the text. On math/STEM corpora that systematically INFLATES the resolved
    Bloom: the arithmetic idiom "Evaluate the expression to find …" (= Apply)
    string-matches as Bloom EVALUATE, and "Create a number line" (= construct)
    matches as Bloom CREATE. On the alg Chapter-1 corpus the raw prose scan
    yielded 8 evaluate + 2 create across 16 interaction/assessment blocks —
    implausible for an integers unit — and every one of those over-resolved
    blocks then tripped the strict band-equality mismatch as a false positive.

    A DECLARED Bloom level is a far more reliable signal than "highest verb
    anywhere in prose" (the author/planner stated the intended demand; it is
    not an incidental idiom). Resolution order is therefore now:

    1. Prefer the IB1 ``interaction`` slot text (the framework's
       slot-addressable interaction verb), when populated — the most direct
       signal of what the learner *does*. (The slot is empty on legacy /
       current corpora until IB1 populates it; this stays the top preference
       for when it lands.)
    2. Else prefer the block's DECLARED ``bloom_level`` (from the structured
       ``content`` payload or a top-level attr), pairing it with the declared
       ``bloom_verb`` when present. This is the planner/author's stated demand
       and is NOT vulnerable to incidental high-Bloom idioms in body prose.
    3. Else FALL BACK to the prose scan over the ``content`` body (highest-
       Bloom verb present). This path is the LAST resort — it is the one
       vulnerable to the math-idiom inflation, so the asymmetric
       over-shoot tolerance in :func:`verb_triple_misaligned` is what
       neutralizes the resulting FPs (an inflated block resolves ABOVE the
       objective band, which is now treated as ALIGNED, not a mismatch). The
       declared-bloom anchor in step 2 is the primary defense; this fallback
       only runs when no declared bloom exists at all.

    Returns ``(None, None)`` when no text and no declared bloom data resolve.
    """
    # (1) IB1 interaction slot text — the most direct demand signal.
    interaction = _block_attr(block, "interaction")
    interaction_text: Optional[str] = None
    if isinstance(interaction, str) and interaction.strip():
        interaction_text = interaction

    if interaction_text is not None:
        verb, level = _scan_prose_for_demand(interaction_text)
        if level is not None:
            return (verb, level)

    # (2) DECLARED bloom anchor — reliable, idiom-proof. Read from the
    # structured content payload first (outline-tier shape), then top-level.
    declared_level = _norm_level(_content_attr(block, "bloom_level"))
    if declared_level is None:
        declared_level = _norm_level(_block_attr(block, "bloom_level"))
    declared_verb = _norm_verb(_content_attr(block, "bloom_verb"))
    if declared_verb is None:
        declared_verb = _norm_verb(_block_attr(block, "bloom_verb"))
    if declared_level is not None:
        # Pair with a declared verb when present, else with the declared
        # level's name as a stand-in verb token.
        return (declared_verb or declared_level, declared_level)

    # (3) Prose fallback over the content body — corroboration-guarded so a
    # lone incidental idiom can't dominate the resolved band.
    content = _block_attr(block, "content")
    body_text: Optional[str] = None
    if isinstance(content, str) and content.strip():
        body_text = content
    elif isinstance(content, Mapping):
        parts = []
        for key in ("stem", "prompt", "question", "text", "body"):
            val = content.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val)
        if parts:
            body_text = "\n".join(parts)

    if body_text is not None:
        verb, level = _scan_prose_for_demand(body_text)
        if level is not None:
            return (verb, level)

    # No interaction text, no declared bloom, no usable body prose → fall
    # back to the bare declared verb (level may still be None).
    return (declared_verb, declared_level)


def _scan_prose_for_demand(
    text: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve ``(verb, level)`` from free prose — highest-Bloom verb present.

    detect_bloom_verbs returns longest-first, higher-level ties first; we pick
    the HIGHEST-Bloom verb present (the strongest cognitive demand the prose
    exercises — a block that both "lists" and "evaluates" exercises evaluate).

    This is the prose path's historical behavior, preserved deliberately: it
    is the LAST resort (steps 1-2 of :func:`resolve_block_verb_level` —
    interaction slot then declared bloom — take precedence), and the
    math-idiom inflation it can produce is neutralized downstream by the
    asymmetric over-shoot tolerance in :func:`verb_triple_misaligned` rather
    than by a brittle lexical demotion heuristic here.
    """
    if not text.strip():
        return (None, None)
    matches = detect_bloom_verbs(text)
    detected_verb: Optional[str] = None
    detected_level: Optional[str] = None
    best_idx = -1
    for level, verb in matches:
        idx = _BLOOM_INDEX.get(level, -1)
        if idx > best_idx:
            best_idx = idx
            detected_level = level
            detected_verb = verb
    return (detected_verb, detected_level)


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


def verb_triple_misaligned(
    objective_level: Optional[str],
    block_level: Optional[str],
) -> bool:
    """True iff the block UNDER-delivers the objective's Bloom band.

    IB3 recalibration (2-corpus calibration gap-map). The framework's
    verb-triple equality rule, enforced as strict band-equality via
    :func:`verbs_share_band`, fired ~32% mostly as FALSE POSITIVES — blocks
    whose RESOLVED Bloom landed ABOVE the objective's band. The over-shoot is
    an ARTIFACT of two compounding effects:

    * the prose-scan resolver (:func:`resolve_block_verb_level`) inflating a
      block's Bloom on math idioms ("Evaluate the expression to find …" is
      arithmetic = Apply, not Bloom Evaluate; "Create a number line" is
      construction, not Bloom Create), and
    * even a genuinely higher-order block still SERVING a lower-order
      objective — a block that teaches at Analyze for an Apply objective has
      not failed the objective; it exceeds it.

    So this gate is now ASYMMETRIC and tolerant of over-shoot:

    * block band == objective band  → ALIGNED (not misaligned).
    * block band  > objective band  → ALIGNED (over-shoot is pedagogically
      acceptable AND the dominant FP class; a higher-order block still serves
      the objective). NOT a mismatch fire.
    * block band  < objective band  → MISALIGNED (genuine UNDER-delivery —
      the real constructive-alignment defect the keystone must catch: an
      objective demanding Evaluate "checked" by an Understand block does not
      certify the objective). This is the ONLY case that fires.

    UNDER-delivery, not over-shoot, is what invalidates constructive
    alignment: you cannot certify a cognitive demand with evidence collected
    at a LOWER demand. Over-shoot at worst wastes effort; it never
    fraudulently certifies. Reserving the fire for strict under-delivery
    stops the over-shoot FP storm WITHOUT weakening the gate's ability to
    catch the genuine under/cross-level misalignment it exists for.

    Returns ``False`` when either level is unresolvable (an unverifiable pair
    is not asserted misaligned — mirrors :func:`verbs_share_band` /
    :func:`bloom_below` fail-safe-to-unverifiable semantics; the caller maps
    the None pair to ``unverifiable`` before reaching this).
    """
    o = _norm_level(objective_level)
    b = _norm_level(block_level)
    if o is None or b is None:
        return False
    # Strictly below the objective band → genuine under-delivery → fire.
    # At-or-above (equal OR over-shoot) → aligned → do not fire.
    return _BLOOM_INDEX[b] < _BLOOM_INDEX[o]


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
