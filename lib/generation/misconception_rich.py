"""``ED4ALL_MISCONCEPTION_RICH`` flag resolver + deterministic named-misconception
grounder.

Single source of truth for the named subject-specific misconception +
productive-failure (predict → reveal → reconcile) gate on the B03/B12
``misconception`` block. Today a misconception block is a generic
claim → correction card with no named-concept link and no productive-failure
structure. This feature stamps a NAMED faulty-model target grounded ONLY in the
block's own ``concept_tags`` / ``domain_concept_vocabulary`` surface forms
(anti-fabrication: ``None`` when nothing resolves — never invented) and routes a
predict-then-reveal authoring contract to the existing rewrite tier.

Default OFF — when unset (or falsey / garbage) the three new ``Block`` fields
(``mc_named_concept`` / ``mc_predict_prompt`` / ``mc_reconcile``, all
Optional/None, hash-excluded) are never projected, the planner pass no-ops, the
rewrite contract keeps its claim → correction bytes, the render scaffold returns
``""``, and the ``misconception_productive_failure`` gate no-ops. Mirrors
``lib/generation/reflection_calibration.py`` /
``lib/generation/new_block_types.py`` exactly.

Parse-with-fallback: truthy ``1`` / ``true`` / ``yes`` / ``on`` enables;
everything else → off. Read each call so tests can toggle the env inline.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence

__all__ = [
    "ENV_MISCONCEPTION_RICH",
    "resolve_misconception_rich",
    "ground_named_misconception",
    "compose_predict_prompt",
    "compose_reconcile",
]

ENV_MISCONCEPTION_RICH = "ED4ALL_MISCONCEPTION_RICH"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

_WORD_RE = re.compile(r"[a-z0-9]+")


def resolve_misconception_rich(value: object = None) -> bool:
    """Return True iff misconception-rich emit / gating is enabled.

    ``value`` (optional) overrides the env var when not ``None``. Falsey /
    garbage / unset → False.
    """
    if value is None:
        raw = os.environ.get(ENV_MISCONCEPTION_RICH, "")
    else:
        raw = value
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in _TRUTHY


def _descriptor_text(misconception_descriptor: Any) -> str:
    """Flatten a misconception descriptor (dict or str) to lower-cased text."""
    if isinstance(misconception_descriptor, str):
        return misconception_descriptor.lower()
    if isinstance(misconception_descriptor, dict):
        parts: List[str] = []
        for key in ("misconception", "correction", "content_focus", "statement"):
            v = misconception_descriptor.get(key)
            if isinstance(v, str):
                parts.append(v)
        # also any nested content dict
        content = misconception_descriptor.get("content")
        if isinstance(content, dict):
            for v in content.values():
                if isinstance(v, str):
                    parts.append(v)
        return " ".join(parts).lower()
    return ""


def _surface_form_tokens(surface: str) -> List[str]:
    """Lower-cased word tokens of a concept slug / surface form."""
    if not isinstance(surface, str):
        return []
    # slugs use hyphen/underscore separators; split on non-alphanumerics.
    return _WORD_RE.findall(surface.lower())


def ground_named_misconception(
    misconception_descriptor: Any,
    co_chunk_concept_tags: Sequence[str],
    domain_vocab_surface_forms: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Return the most-salient grounded concept slug for a misconception, or None.

    Anti-fabrication contract: the returned slug is ALWAYS a member of
    ``co_chunk_concept_tags`` (the block's OWN grounded concept tags) — never a
    value absent from the input. When ``domain_vocab_surface_forms`` is supplied,
    a candidate tag is additionally REQUIRED to match a domain surface form
    (the per-course vocabulary) so an off-domain tag is rejected.

    Salience ranking (deterministic):
      1. A tag whose surface form (its word tokens) ALL appear in the
         misconception descriptor text — the concept is literally discussed.
      2. Otherwise the first eligible tag in input order.

    Returns ``None`` when no tag resolves (the block stays a generic
    misconception — always-keep, never invent).
    """
    tags = [str(t).strip() for t in (co_chunk_concept_tags or []) if str(t).strip()]
    if not tags:
        return None

    # Build the domain-vocab token sets (when supplied) for the grounding check.
    vocab_token_sets: Optional[List[set]] = None
    if domain_vocab_surface_forms is not None:
        vocab_token_sets = [
            set(_surface_form_tokens(s))
            for s in domain_vocab_surface_forms
            if isinstance(s, str) and s.strip()
        ]

    desc = _descriptor_text(misconception_descriptor)
    desc_tokens = set(_WORD_RE.findall(desc))

    eligible: List[str] = []
    described: List[str] = []
    for tag in tags:
        tag_tokens = set(_surface_form_tokens(tag))
        if not tag_tokens:
            continue
        # Domain-vocab grounding gate (when a vocabulary was supplied).
        if vocab_token_sets is not None:
            if not any(
                tag_tokens and tag_tokens.issubset(vts) or tag_tokens == vts
                for vts in vocab_token_sets
            ):
                # Allow a looser overlap match (shared tokens) so a multiword
                # surface form still grounds a single-token tag.
                if not any(tag_tokens & vts for vts in vocab_token_sets):
                    continue
        eligible.append(tag)
        if tag_tokens and tag_tokens.issubset(desc_tokens):
            described.append(tag)

    if described:
        return described[0]
    if eligible:
        return eligible[0]
    return None


def compose_predict_prompt(named_concept: Optional[str]) -> Optional[str]:
    """Compose the deterministic predict-then-reveal PROMPT seam text.

    The actual prose is authored by the rewrite tier; this is the structural
    seam stamped onto the descriptor so the renderer + validator can detect the
    predict phase. Returns ``None`` when no concept is named (anti-fabrication —
    no productive-failure scaffold without a named target).
    """
    if not (isinstance(named_concept, str) and named_concept.strip()):
        return None
    label = named_concept.replace("_", " ").replace("-", " ").strip()
    return (
        f"Before reading the correction, predict: what does a learner who "
        f"misunderstands {label} expect to happen?"
    )


def compose_reconcile(named_concept: Optional[str]) -> Optional[str]:
    """Compose the deterministic RECONCILE seam text (why the named model fails).

    Returns ``None`` when no concept is named.
    """
    if not (isinstance(named_concept, str) and named_concept.strip()):
        return None
    label = named_concept.replace("_", " ").replace("-", " ").strip()
    return (
        f"Reconcile: compare your prediction with the correct account of "
        f"{label} and explain why the faulty model fails."
    )


def apply_misconception_grounding(
    selected: List[Dict[str, Any]],
    *,
    concept_tags_by_co: Optional[Dict[str, Sequence[str]]] = None,
    domain_vocab_surface_forms: Optional[Sequence[str]] = None,
    enabled: Optional[bool] = None,
) -> Dict[str, int]:
    """Stamp ``mc_named_concept`` onto every grounded ``misconception`` descriptor.

    STAMP-ONLY (adds no block, invents no CO id). For each ``misconception``
    descriptor in ``selected``, gathers the concept tags of its ``target_co_ids``
    (via ``concept_tags_by_co``) and grounds a named faulty-model concept via
    :func:`ground_named_misconception`. ``None`` when unresolved (stays generic).

    Identity no-op when ``enabled`` resolves False. Returns a counts dict
    ``{seen, named, generic}`` for the decision capture.
    """
    if enabled is None:
        enabled = resolve_misconception_rich()
    counts = {"seen": 0, "named": 0, "generic": 0}
    if not enabled:
        return counts

    tags_by_co = concept_tags_by_co or {}
    for block in selected:
        bt = block.get("block_type") if isinstance(block, dict) else None
        if not (isinstance(bt, str) and bt.strip().lower() == "misconception"):
            continue
        counts["seen"] += 1
        co_ids = block.get("target_co_ids") or []
        block_tags: List[str] = []
        for cid in co_ids:
            block_tags.extend(str(t) for t in (tags_by_co.get(str(cid)) or []))
        named = ground_named_misconception(
            block, block_tags, domain_vocab_surface_forms
        )
        if named:
            block["mc_named_concept"] = named
            block["mc_predict_prompt"] = compose_predict_prompt(named)
            block["mc_reconcile"] = compose_reconcile(named)
            counts["named"] += 1
        else:
            counts["generic"] += 1
    return counts
