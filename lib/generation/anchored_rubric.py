"""FR-11 — Anchored-rubric PRODUCER (the author for ``Block.anchored_rubric``).

The IB3.4 :class:`lib.validators.alignment.anchored_rubric.AnchoredRubricValidator`
is a CONSUMER with no producer: nothing authors ``Block.anchored_rubric``, so
the IB3 flag (``ED4ALL_ALIGNMENT_VERB_TRIPLE``) on its own would flag every
Evaluate/Create scored block as ``ANCHORED_RUBRIC_MISSING``. This module is the
missing producer — a DETERMINISTIC, GROUNDED rubric author.

What the validator requires (consumer contract, mirrored here)
--------------------------------------------------------------

For every block whose ``block_type`` is in ``_RUBRIC_BLOCK_TYPES``
(``assessment_item`` / ``scenario`` / ``problem`` / ``reflection_prompt``) AND
whose RESOLVED objective Bloom is ``evaluate`` / ``create``, the block MUST ship
``anchored_rubric`` of shape::

    {
      "criteria": [
        {"name": str,
         "bands": [{"level": str, "descriptor": str, "exemplar": str}, ...]},
        ...
      ],
      "published_before_task": True,
    }

with ≥1 criterion, ≥2 bands per criterion, and a non-empty ``exemplar`` on every
band. The framework (pp.26-33, 5.2, B14, B11) requires an EXEMPLAR-anchored
rubric published BEFORE the task — ambiguous descriptor words alone are far too
ambiguous to score the highest-Bloom work.

How this producer GROUNDS criteria + anchors (anti-fabrication)
---------------------------------------------------------------

Criteria are NOT invented wholesale. Each criterion is derived from the block's
own grounded signals:

* The objective's Bloom VERB (resolved via the IB3.1 verb-triple resolver) names
  the cognitive performance the criterion certifies — a Create objective's
  ``design`` verb becomes the "Quality of the <verb>d work" criterion; the
  framework's Evaluate/Create dimension set (justification / use of criteria /
  originality / accuracy) is selected by the resolved band.
* The assessed CONCEPT — the noun the work is about — is pulled, in order of
  preference, from the objective's ``abcd.behavior.action_object``, then the
  objective statement's content phrase, then the block's ``key_terms`` / body
  text. The concept is woven into each criterion's descriptors + exemplars so
  the rubric is about THIS block's content, not a generic template.

Each band's ``exemplar`` is a concrete "what a response at this level looks
like" anchor phrased against the resolved concept + verb (e.g. for an
``evaluate`` objective about "linear equations": a top-band exemplar reads
"Weighs at least two solution methods for the linear equation against stated
criteria and justifies the chosen method with evidence."). The exemplar is an
illustrative anchor, not a fabricated source claim — it describes the FORM of an
adequate response, which is exactly what a published-first rubric exemplar is.

ANTI-FABRICATION hard contract: if NEITHER a Bloom verb NOR a concept can be
resolved (a genuinely empty objective + empty block), the producer returns
``None`` — and the validator then honestly flags ``ANCHORED_RUBRIC_MISSING``
rather than us shipping a fake rubric.

Gating + byte-stability
-----------------------

The producer is gated by ``ED4ALL_ALIGNMENT_VERB_TRIPLE`` (the IB3 master flag,
read via :func:`lib.validators.alignment.verb_triple.alignment_verb_triple_enabled`).
When OFF (default), :func:`attach_anchored_rubrics` is a strict no-op — no block
gets an ``anchored_rubric``, the field stays ``None`` (hash-excluded), and every
snapshot is byte-identical. The flag is the SAME one the validator gates on, so
producer + consumer turn on together.

No LLM / no model load — this is a deterministic grounded producer (the plan's
preferred reliable seam). No new dependency beyond :mod:`lib.ontology.bloom`
(verb bands) + the IB3.1 resolver.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lib.validators.alignment.verb_triple import (
    alignment_verb_triple_enabled,
    resolve_objective_verb_level,
)

logger = logging.getLogger(__name__)

__all__ = [
    "build_anchored_rubric",
    "attach_anchored_rubrics",
    "RUBRIC_BLOCK_TYPES",
    "RUBRIC_BLOOM_LEVELS",
]

#: Block types that produce/score work whose objective, at Evaluate/Create,
#: requires an anchored rubric. MIRRORS the consumer's ``_RUBRIC_BLOCK_TYPES``
#: (kept local so the producer module has no import coupling to the validator).
RUBRIC_BLOCK_TYPES: frozenset = frozenset(
    {"assessment_item", "scenario", "problem", "reflection_prompt"}
)
#: Objective Bloom levels that require an anchored rubric.
RUBRIC_BLOOM_LEVELS: frozenset = frozenset({"evaluate", "create"})

# Three canonical band levels (high → low). ≥2 satisfies the validator; three
# gives a usable scoring scale (the framework's exemplar-anchored bands).
_BAND_LEVELS: Sequence[str] = ("exemplary", "proficient", "developing")

# Per-Bloom dimension menu (framework Evaluate/Create criteria, pp.26-33). Each
# entry: (criterion_name_template, per-band descriptor templates keyed by level).
# ``{verb}`` / ``{concept}`` are interpolated from the block's grounded signals.
# A criterion is only EMITTED when its grounding (concept / verb) resolved, so
# the rubric never ships a template with an unfilled placeholder.
_EVALUATE_DIMENSIONS: Sequence[Dict[str, Any]] = (
    {
        "name": "Quality of judgement about {concept}",
        "bands": {
            "exemplary": (
                "Reaches a clear, defensible judgement about {concept} and "
                "{verb}s it against explicit criteria."
            ),
            "proficient": (
                "Reaches a judgement about {concept} with some supporting "
                "criteria, though one or two are implicit."
            ),
            "developing": (
                "States an opinion about {concept} without applying clear "
                "criteria to {verb} it."
            ),
        },
    },
    {
        "name": "Use of evidence and criteria",
        "bands": {
            "exemplary": (
                "Cites specific, relevant evidence about {concept} and weighs "
                "it against each stated criterion."
            ),
            "proficient": (
                "Cites evidence about {concept} but applies the criteria "
                "unevenly."
            ),
            "developing": (
                "Offers little evidence about {concept}; criteria are missing "
                "or misapplied."
            ),
        },
    },
)
_CREATE_DIMENSIONS: Sequence[Dict[str, Any]] = (
    {
        "name": "Quality of the {verb}d work on {concept}",
        "bands": {
            "exemplary": (
                "Produces an original, coherent {concept} artifact that fully "
                "meets the brief and is internally consistent."
            ),
            "proficient": (
                "Produces a workable {concept} artifact that meets most of the "
                "brief with minor gaps."
            ),
            "developing": (
                "Produces a {concept} artifact that is incomplete or copies an "
                "existing model rather than {verb}ing a new one."
            ),
        },
    },
    {
        "name": "Justification of design choices",
        "bands": {
            "exemplary": (
                "Justifies every major {concept} design choice against the "
                "goal and the constraints."
            ),
            "proficient": (
                "Justifies the main {concept} design choices, leaving some "
                "decisions unexplained."
            ),
            "developing": (
                "Makes {concept} design choices without explaining why."
            ),
        },
    },
)

# Per-band exemplar anchor templates (the concrete "what a response looks like"
# anchor the framework demands). Keyed by Bloom level then band level.
_EXEMPLAR_TEMPLATES: Dict[str, Dict[str, str]] = {
    "evaluate": {
        "exemplary": (
            "Weighs at least two options for {concept} against the stated "
            "criteria and justifies the chosen one with specific evidence."
        ),
        "proficient": (
            "Compares options for {concept} and reaches a judgement, citing "
            "some but not all of the criteria."
        ),
        "developing": (
            "Picks an option for {concept} but does not weigh it against the "
            "criteria or cite evidence."
        ),
    },
    "create": {
        "exemplary": (
            "Designs a complete, original {concept} that satisfies every "
            "requirement and explains each design decision."
        ),
        "proficient": (
            "Designs a functional {concept} that meets most requirements with "
            "a partial rationale."
        ),
        "developing": (
            "Assembles a {concept} that is incomplete or reuses an existing "
            "design without adapting it."
        ),
    },
}

# Stopwords stripped when extracting a concept phrase from an objective
# statement (so "students will evaluate the linear equation" → "linear
# equation"). Includes Bloom verbs implicitly via the verb already being
# resolved separately.
_CONCEPT_STOPWORDS: frozenset = frozenset(
    {
        "the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "with",
        "by", "from", "their", "its", "this", "that", "these", "those",
        "student", "students", "learner", "learners", "will", "be", "able",
        "given", "using", "use", "about", "into", "as", "at", "is", "are",
        "your", "you", "they", "them", "it", "each", "all", "any", "such",
    }
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")


def _block_attr(block: Any, key: str) -> Any:
    if hasattr(block, key):
        return getattr(block, key)
    if isinstance(block, Mapping):
        return block.get(key)
    return None


def _resolve_objective_ids(block: Any) -> List[str]:
    """Mirror the consumer's objective-id resolution (structural → content)."""
    structural = list(_block_attr(block, "objective_ids") or ())
    if structural:
        return [o for o in structural if isinstance(o, str) and o]
    content = _block_attr(block, "content")
    if isinstance(content, Mapping):
        raw = content.get("objective_ids") or content.get("objective_refs") or []
        return [o for o in raw if isinstance(o, str) and o]
    return []


def _resolve_objective_and_band(
    block: Any, objectives_map: Mapping[str, Mapping[str, Any]],
) -> Optional[tuple]:
    """Return ``(objective_dict, verb, bloom_level)`` for the highest-band
    Evaluate/Create objective the block serves, or ``None``.

    Mirrors the consumer's resolution exactly so producer + validator agree on
    WHICH blocks need a rubric: a block whose served objective is below Evaluate
    yields ``None`` (no rubric authored)."""
    for obj_id in _resolve_objective_ids(block):
        obj = objectives_map.get(obj_id)
        if not isinstance(obj, Mapping):
            continue
        verb, level = resolve_objective_verb_level(obj)
        if level in RUBRIC_BLOOM_LEVELS:
            return (obj, verb, level)
    return None


def _extract_concept(
    block: Any, objective: Mapping[str, Any],
) -> Optional[str]:
    """Resolve the assessed CONCEPT (the noun the work is about) — grounded.

    Preference order (anti-fabrication: every source is the block's own data):
      1. ``objective.abcd.behavior.action_object`` (the authored object of the
         performance verb — the framework's canonical "behavior object").
      2. The objective ``statement`` minus stopwords + the leading Bloom verb —
         the content phrase the objective is about.
      3. The block's ``key_terms`` (first term) — the page's own vocabulary.
    Returns a short concept phrase, or ``None`` when nothing grounds.
    """
    # (1) ABCD behavior action_object — the authored truth.
    abcd = objective.get("abcd") if isinstance(objective, Mapping) else None
    if isinstance(abcd, Mapping):
        behavior = abcd.get("behavior")
        if isinstance(behavior, Mapping):
            obj_noun = behavior.get("action_object")
            if isinstance(obj_noun, str) and obj_noun.strip():
                return _trim_concept(obj_noun)

    # (2) Objective statement content phrase.
    statement = objective.get("statement") if isinstance(objective, Mapping) else None
    if isinstance(statement, str) and statement.strip():
        phrase = _concept_from_statement(statement)
        if phrase:
            return phrase

    # (3) Block key_terms.
    key_terms = _block_attr(block, "key_terms") or ()
    for term in key_terms:
        if isinstance(term, str) and term.strip():
            return _trim_concept(term.replace("_", " ").replace("-", " "))

    return None


def _concept_from_statement(statement: str) -> Optional[str]:
    """Pull the content noun-phrase from an objective statement (deterministic).

    Tokenises, drops stopwords, and keeps the last contiguous run of
    content words (the object of the objective is overwhelmingly the tail of an
    "[audience] will [verb] [object] [condition]" statement). Returns up to the
    final 4 content tokens joined, or ``None`` when nothing survives."""
    tokens = [t.lower() for t in _WORD_RE.findall(statement)]
    content_tokens = [t for t in tokens if t not in _CONCEPT_STOPWORDS]
    if not content_tokens:
        return None
    # Keep the trailing content phrase (the objective's object), capped at 4
    # words so the criterion text stays readable.
    tail = content_tokens[-4:]
    return _trim_concept(" ".join(tail))


def _trim_concept(value: str) -> str:
    """Normalise a concept phrase (collapse whitespace, cap length)."""
    cleaned = " ".join(str(value).split())
    # Cap at ~60 chars so a criterion name stays readable.
    return cleaned[:60].strip()


def _verb_for_text(verb: Optional[str], level: str) -> str:
    """Resolve the display verb woven into criteria/exemplars.

    Falls back to the canonical Bloom-level verb when the objective gave no
    verb (so the criterion always reads naturally)."""
    if isinstance(verb, str) and verb.strip():
        return verb.strip().lower()
    return "evaluate" if level == "evaluate" else "create"


def build_anchored_rubric(
    block: Any,
    objectives_map: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Author a grounded, exemplar-anchored, published-first rubric for ``block``.

    Returns the rubric dict (satisfying ``AnchoredRubricValidator``) when:
      * the block is a scored/produced-work type (``RUBRIC_BLOCK_TYPES``),
      * its served objective resolves to an Evaluate/Create Bloom band, AND
      * a concept (and verb) can be GROUNDED from the objective/block.
    Returns ``None`` otherwise (NOT a scored type / below Evaluate / nothing to
    anchor) so the validator honestly flags a genuine miss rather than us
    shipping a fabricated rubric.

    The rubric is deterministic — no model call. Criteria + exemplars are
    interpolated from the resolved Bloom verb + assessed concept.
    """
    block_type = _block_attr(block, "block_type")
    if block_type not in RUBRIC_BLOCK_TYPES:
        return None

    resolved = _resolve_objective_and_band(block, objectives_map)
    if resolved is None:
        return None
    objective, verb, level = resolved

    concept = _extract_concept(block, objective)
    if not concept:
        # ANTI-FABRICATION: nothing to anchor → leave None; the validator flags
        # it honestly rather than shipping an invented rubric.
        return None

    display_verb = _verb_for_text(verb, level)
    dimensions = (
        _EVALUATE_DIMENSIONS if level == "evaluate" else _CREATE_DIMENSIONS
    )
    exemplar_templates = _EXEMPLAR_TEMPLATES[level]

    criteria: List[Dict[str, Any]] = []
    for dim in dimensions:
        name = dim["name"].format(verb=display_verb, concept=concept)
        bands: List[Dict[str, str]] = []
        for band_level in _BAND_LEVELS:
            descriptor = dim["bands"][band_level].format(
                verb=display_verb, concept=concept,
            )
            exemplar = exemplar_templates[band_level].format(
                verb=display_verb, concept=concept,
            )
            bands.append({
                "level": band_level,
                "descriptor": descriptor,
                "exemplar": exemplar,
            })
        criteria.append({"name": name, "bands": bands})

    if not criteria:
        return None

    return {
        "criteria": criteria,
        "published_before_task": True,
    }


def attach_anchored_rubrics(
    *,
    blocks: Sequence[Any],
    objectives_payload: Sequence[Mapping[str, Any]],
    capture: Any = None,
    course_code: str = "",
) -> List[Any]:
    """Stamp ``anchored_rubric`` onto every Evaluate/Create scored block.

    GATED on ``ED4ALL_ALIGNMENT_VERB_TRIPLE`` (the IB3 master flag). When OFF
    (default) this is a STRICT NO-OP — returns ``blocks`` unchanged so the field
    stays ``None`` (hash-excluded) and every snapshot is byte-identical, and
    producer + consumer turn on together.

    ``objectives_payload`` is the list of objective dicts (the outline phase's
    ``terminal_objectives + chapter_objectives`` shape) carrying ``id`` /
    ``statement`` / ``bloom_level`` / ``abcd``. Returns a new list of blocks;
    each scored Evaluate/Create block whose rubric grounds is replaced via
    :func:`dataclasses.replace` (or has its ``anchored_rubric`` key set, for a
    dict-shaped block); every other block is returned UNCHANGED (identity).

    NEVER raises — any per-block error is logged and that block passes through
    unchanged (mirrors the other best-effort outline post-passes).
    """
    if not alignment_verb_triple_enabled():
        return list(blocks)
    if not blocks:
        return list(blocks)

    objectives_map: Dict[str, Mapping[str, Any]] = {}
    for obj in objectives_payload or ():
        if isinstance(obj, Mapping):
            oid = obj.get("id")
            if isinstance(oid, str) and oid:
                objectives_map[oid] = obj

    out: List[Any] = []
    authored = 0
    skipped_unanchored = 0
    for block in blocks:
        try:
            rubric = build_anchored_rubric(block, objectives_map)
        except Exception as exc:  # noqa: BLE001 — best-effort, never abort
            logger.warning(
                "anchored_rubric producer: failed for block %r (%s); "
                "passing through unchanged.",
                _block_attr(block, "block_id"), exc,
            )
            out.append(block)
            continue
        if rubric is None:
            # Track scored Evaluate/Create blocks we COULD NOT anchor (genuine
            # misses the validator will flag) for the capture rationale.
            if (
                _block_attr(block, "block_type") in RUBRIC_BLOCK_TYPES
                and _resolve_objective_and_band(block, objectives_map) is not None
            ):
                skipped_unanchored += 1
            out.append(block)
            continue
        out.append(_set_anchored_rubric(block, rubric))
        authored += 1

    _emit_decision(
        capture,
        course_code=course_code,
        authored=authored,
        skipped_unanchored=skipped_unanchored,
        total=len(out),
    )
    return out


def _set_anchored_rubric(block: Any, rubric: Dict[str, Any]) -> Any:
    """Return a copy of ``block`` carrying ``anchored_rubric=rubric``.

    Frozen ``Block`` dataclass → :func:`dataclasses.replace`; a dict-shaped
    block (legacy / test caller) → a shallow copy with the key set."""
    import dataclasses  # noqa: PLC0415

    if dataclasses.is_dataclass(block) and not isinstance(block, type):
        return dataclasses.replace(block, anchored_rubric=rubric)
    if isinstance(block, dict):
        new = dict(block)
        new["anchored_rubric"] = rubric
        return new
    # Last-resort: attempt attribute set (non-frozen object).
    try:
        setattr(block, "anchored_rubric", rubric)
    except Exception:  # noqa: BLE001
        logger.warning(
            "anchored_rubric producer: could not set anchored_rubric on a "
            "block of type %s; passing through unchanged.", type(block).__name__,
        )
    return block


def _emit_decision(
    capture: Any,
    *,
    course_code: str,
    authored: int,
    skipped_unanchored: int,
    total: int,
) -> None:
    """Emit one ``anchored_rubric_authored`` decision event (replayable)."""
    if capture is None:
        return
    rationale = (
        f"FR-11 anchored-rubric producer (course={course_code or 'n/a'!r}): "
        f"authored {authored} grounded exemplar-anchored rubric(s) across "
        f"{total} block(s) for Evaluate/Create scored blocks; "
        f"{skipped_unanchored} scored Evaluate/Create block(s) had no "
        f"groundable concept and were left without a rubric (the validator "
        f"flags those honestly rather than shipping a fabricated rubric). "
        f"Each rubric carries >=2 exemplar-anchored bands per criterion and is "
        f"published_before_task. Gated on ED4ALL_ALIGNMENT_VERB_TRIPLE."
    )
    try:
        capture.log_decision(
            decision_type="content_selection",
            decision=f"anchored_rubric: authored {authored}/{total}",
            rationale=rationale,
        )
    except Exception as exc:  # noqa: BLE001 — capture must never break a run
        logger.debug("anchored_rubric producer: decision capture failed: %s", exc)
