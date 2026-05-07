"""Shared action-verb cooccurrence axis for objective-delivery validators.

Pass when at least one whole-word match of the declared bloom_level's
canonical verb set (∪ ``{bloom_verb}``) appears in the surface text.
Skip when no synonym set can be assembled.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.3.2.
"""
from __future__ import annotations

import re
from typing import Optional, Set, Tuple

from lib.ontology.bloom import get_verbs

__all__ = ["score_verb_cooccurrence_axis"]


def score_verb_cooccurrence_axis(
    *,
    text: str,
    declared_bloom: Optional[str],
    bloom_verb: Optional[str],
) -> Tuple[Optional[bool], Optional[int], Set[str]]:
    """Return ``(passed, match_count, synonym_set_used)``.

    * ``passed=None`` + empty set when no synonym set can be built
      (declared_bloom missing AND bloom_verb missing) — caller emits
      VERB_AXIS_UNAVAILABLE.
    * ``passed=False`` when synonyms exist but text matches zero.
    * ``passed=True`` otherwise.
    """
    verbs_by_level = get_verbs()
    synonym_set: Set[str] = set()
    if declared_bloom is not None and declared_bloom in verbs_by_level:
        synonym_set = set(verbs_by_level[declared_bloom])
        if bloom_verb:
            synonym_set.add(bloom_verb)
    elif bloom_verb:
        synonym_set = {bloom_verb}

    if not synonym_set or not text:
        return None, None, synonym_set

    lowered = text.lower()
    matches = 0
    for v in synonym_set:
        if not v:
            continue
        if re.search(rf"\b{re.escape(v)}\b", lowered):
            matches += 1
    return matches > 0, matches, synonym_set
