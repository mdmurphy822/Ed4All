"""``ED4ALL_RECALL_SELF_CHECK`` flag resolver + deterministic planner pass.

Single source of truth for the free-recall / cloze self-check variant gate. The
legacy structured self-check renderer is strictly MCQ-recognition (radio inputs
over ``q["options"]``) and the UDL response-format map hardcodes
``self_check_question → "select"`` (recognition). Recall-format closes that gap:
a recognition → recall retrieval-practice variant (produce-the-answer free recall
OR a cloze ``____`` blank) with the answer behind a ``<details>`` reveal.

Default OFF — when unset (or falsey / garbage) the new ``Block.recall_format``
field stays None (hash-excluded), the planner pass :func:`apply_recall_self_check`
is skipped, the structured renderer emits its legacy MCQ radio markup, the
rewrite-tier ``self_check_question`` contract is unchanged, the
``_UDL_RESPONSE_BY_BLOCK_TYPE`` map still returns ``select``, and the
``recall_self_check_format`` gate no-ops. Mirrors
``lib/generation/reflection_calibration.py`` / ``lib/generation/new_block_types.py``.

Parse-with-fallback: truthy ``1`` / ``true`` / ``yes`` / ``on`` enables;
everything else (falsey / garbage / unset) → off. ``recall_format`` itself is
parse-with-fallback too: an unknown / garbage value resolves to ``None`` (no
recall variant — treated as legacy recognition).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "ENV_RECALL_SELF_CHECK",
    "RECALL_FORMATS",
    "resolve_recall_self_check",
    "resolve_recall_format",
    "apply_recall_self_check",
]

ENV_RECALL_SELF_CHECK = "ED4ALL_RECALL_SELF_CHECK"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: The two recall variants. ``free_recall`` = produce-the-answer prompt with a
#: ``<details>`` reveal reusing the question's EXISTING correct option (no
#: fabricated answer). ``cloze`` = a ``____`` fill-in-the-blank prompt; the blank
#: term is authored by the LLM rewrite tier from grounded source (never guessed
#: deterministically).
RECALL_FORMATS = frozenset({"free_recall", "cloze"})

#: Bloom levels for which a CLOZE (single-term retrieval) is the natural recall
#: variant; higher-order objectives get a free-recall (produce-the-explanation)
#: prompt. Deterministic so two runs of the same plan mark the same blocks.
_CLOZE_BLOOM_LEVELS = frozenset({"remember", "understand"})


def resolve_recall_self_check(value: object = None) -> bool:
    """Return True iff the recall-self-check emit / gating is enabled.

    ``value`` (optional) overrides the env var when not ``None`` — accepts the
    same truthy tokens (case-insensitive). Falsey / garbage / unset → False.
    """
    if value is None:
        raw = os.environ.get(ENV_RECALL_SELF_CHECK, "")
    else:
        raw = value
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in _TRUTHY


def resolve_recall_format(value: object) -> Optional[str]:
    """Normalize a recall-format token to a canonical member or ``None``.

    Parse-with-fallback: an unknown / garbage / empty value → ``None`` (no
    recall variant — the block stays legacy recognition).
    """
    if not isinstance(value, str):
        return None
    token = value.strip().lower()
    return token if token in RECALL_FORMATS else None


def _block_get(block: Any, key: str) -> Any:
    if isinstance(block, dict):
        return block.get(key)
    return getattr(block, key, None)


def _block_type_of(block: Any) -> str:
    bt = _block_get(block, "block_type")
    return bt.strip().lower() if isinstance(bt, str) else ""


def _co_ids_of(block: Any) -> List[str]:
    cos = _block_get(block, "target_co_ids")
    if isinstance(cos, (list, tuple)):
        return [str(c) for c in cos if c]
    return []


#: Exposition / content block types that count as "teaching" a CO earlier in the
#: week so a later self_check is genuine SPACED retrieval (not a pre-quiz). An
#: EXACT mirror of the retrieval-floor's exposition set is not required — any
#: non-check block that targets the same CO earlier suffices.
_NON_CHECK_TYPES = frozenset(
    {
        "concept", "explanation", "example", "worked_example", "formula",
        "diagram", "vocab_card", "table", "key_idea", "hook", "scenario",
        "guided_practice", "problem", "activity", "callout", "multimedia",
    }
)


def apply_recall_self_check(
    selected: List[Dict[str, Any]],
    *,
    enabled: Optional[bool] = None,
) -> Tuple[List[Dict[str, Any]], int, Dict[str, int]]:
    """Stamp ``recall_format`` onto already-taught-CO ``self_check_question`` blocks.

    Deterministic, bounded, anti-fabrication: STAMP-ONLY. Adds no block, invents
    no CO id, re-orders nothing. A ``self_check_question`` block is marked iff an
    EARLIER block in ``selected`` (a non-check exposition block) targets an
    intersecting CO — i.e. the check is genuine SPACED retrieval of material the
    week already taught. The format is chosen deterministically from the block's
    ``target_bloom`` (remember/understand → ``cloze`` single-term retrieval;
    higher-order → ``free_recall`` produce-the-explanation).

    Identity no-op when ``enabled`` resolves False (returns ``selected``
    untouched, 0 marked).

    Returns ``(selected, marked_count, format_counts)``.
    """
    if enabled is None:
        enabled = resolve_recall_self_check()
    if not enabled:
        return selected, 0, {}

    # Accumulate the CO ids taught by every non-check block seen SO FAR, so a
    # self_check at index i is marked only when an earlier block taught its CO.
    taught_cos: set = set()
    marked = 0
    format_counts: Dict[str, int] = {}
    for block in selected:
        btype = _block_type_of(block)
        co_ids = _co_ids_of(block)
        if btype == "self_check_question":
            # Already-carried recall_format wins (idempotent / operator override).
            existing = resolve_recall_format(_block_get(block, "recall_format"))
            if existing is not None:
                continue
            if co_ids and any(c in taught_cos for c in co_ids):
                bloom = _block_get(block, "target_bloom")
                bloom_l = bloom.strip().lower() if isinstance(bloom, str) else ""
                fmt = "cloze" if bloom_l in _CLOZE_BLOOM_LEVELS else "free_recall"
                if isinstance(block, dict):
                    block["recall_format"] = fmt
                    marked += 1
                    format_counts[fmt] = format_counts.get(fmt, 0) + 1
            # A self_check still teaches nothing toward `taught_cos`.
            continue
        if btype in _NON_CHECK_TYPES and co_ids:
            taught_cos.update(co_ids)

    return selected, marked, format_counts
