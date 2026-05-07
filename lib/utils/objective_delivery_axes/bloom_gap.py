"""Shared Bloom-gap axis for objective-delivery validators.

Pass when ``BLOOM_LEVELS.index(declared) - BLOOM_LEVELS.index(observed) < 2``
(scaffolding tolerance). Skip when either side is None or unrecognised.

See plan ``plans/wave-D7-validator-splits-2026-05-07.md`` §3.3.2.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from lib.ontology.bloom import BLOOM_LEVELS

__all__ = ["BLOOM_INDEX", "score_bloom_gap_axis"]


BLOOM_INDEX: Dict[str, int] = {level: idx for idx, level in enumerate(BLOOM_LEVELS)}


def score_bloom_gap_axis(
    *,
    declared_bloom: Optional[str],
    observed_bloom: Optional[str],
    gap_threshold: int = 2,
) -> Tuple[Optional[bool], Optional[int]]:
    """Return ``(passed, gap)`` for the Bloom-gap axis.

    * ``passed=None`` when either bloom level is missing or not in the
      canonical :data:`BLOOM_INDEX` (axis skipped).
    * ``passed=False`` when ``gap >= gap_threshold``.
    * ``passed=True`` otherwise.
    """
    if declared_bloom is None or observed_bloom is None:
        return None, None
    if declared_bloom not in BLOOM_INDEX or observed_bloom not in BLOOM_INDEX:
        return None, None
    gap = BLOOM_INDEX[declared_bloom] - BLOOM_INDEX[observed_bloom]
    if gap >= gap_threshold:
        return False, gap
    return True, gap
