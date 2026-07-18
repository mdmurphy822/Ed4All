"""Deterministic per-claim candidate ORDERING for the frontier NLI stage.

The frontier stage-1 (``ED4ALL_GROUNDEDNESS_FRONTIER`` in
:mod:`lib.retrieval.groundedness`) scores each claim against its passage pool in
priority order and retires the claim the moment a premise entails it — so the
order in which premises are visited decides how early a truly-supported claim
stops (the early-exit is sound because the verdict fold is a pure max over the
FULL pool for any claim that does *not* retire). This module produces that order.

Contract (measured constraint, see the frontier design):

  * It **only ORDERS** — every passage index appears exactly once in the output;
    nothing is ever excluded. Cosine similarity may only *reorder*, never drop a
    candidate (the baseline entailing chunk's cosine rank is ~uniform-random, so
    any cosine cut would silently flip verdicts).
  * Ordering is a stable, fully-deterministic sort over four tiers:
      (a) passages whose ``chunk_id`` is in the block's DIRECT source-ref set;
      (b) descending lexical anchor score (shared uncommon tokens, 2× per shared
          numeric / equation-ish token);
      (c) descending MiniLM cosine (FINAL tiebreaker; skipped when the caller
          passes no vectors — embedder-unavailable degrades to lexical-only);
      (d) ascending original index (stable — makes the order reproducible).

Pure CPU: no model loads, no embedding here — the caller supplies any vectors.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Sequence, Set

#: An "uncommon" content token: a lowercased alphabetic run of length ≥ 5. Short
#: / numeric tokens are handled by the numeric-anchor tier so common stop-words
#: (``the``, ``and``) never inflate the anchor score.
_UNCOMMON_TOKEN_RE = re.compile(r"[a-z]{5,}")

#: Whitespace-delimited token (for the numeric / equation-ish anchor set).
_WORD_RE = re.compile(r"\S+")

#: A token counts as numeric / equation-ish when it carries any of these — a
#: digit or an arithmetic operator glyph embedded in the word (``3``, ``25*4``,
#: ``x^2``, ``a=b``). Shared numeric anchors are worth 2× an uncommon token
#: because a matching literal / formula is a far stronger grounding signal.
_NUMERICISH_CHARS = frozenset("0123456789=^*/+-")


def _uncommon_tokens(text: Optional[str]) -> Set[str]:
    """Set of lowercased alphabetic tokens (len ≥ 5) in ``text``."""
    return set(_UNCOMMON_TOKEN_RE.findall((text or "").lower()))


def _numericish_tokens(text: Optional[str]) -> Set[str]:
    """Set of lowercased whitespace tokens carrying a digit / operator glyph."""
    out: Set[str] = set()
    for tok in _WORD_RE.findall((text or "").lower()):
        if any(c in _NUMERICISH_CHARS for c in tok):
            out.add(tok)
    return out


def anchor_score(claim_text: str, passage_text: str) -> int:
    """Lexical grounding-affinity score between a claim and a passage.

    ``|shared uncommon tokens| + 2 × |shared numeric/equation-ish tokens|``.
    Deterministic, symmetric in text content, and independent of any model.
    """
    shared_uncommon = len(_uncommon_tokens(claim_text) & _uncommon_tokens(passage_text))
    shared_numeric = len(_numericish_tokens(claim_text) & _numericish_tokens(passage_text))
    return shared_uncommon + 2 * shared_numeric


def order_passages_for_claim(
    claim_text: str,
    passage_texts: Sequence[str],
    passage_chunk_ids: Sequence[Optional[str]],
    *,
    direct_cited_ids: Optional[Set[str]] = None,
    claim_vec: Optional[Any] = None,
    passage_vecs: Optional[Sequence[Any]] = None,
) -> List[int]:
    """Return passage indices ``0..N-1`` in frontier scoring order.

    The output is a permutation of ``range(len(passage_texts))`` — a pure
    reordering, never a filter. ``claim_vec`` / ``passage_vecs`` (unit-normalized
    MiniLM vectors from the caller) drive tier (c); pass ``None`` for either to
    fall back to lexical-only ordering (embedder unavailable). Vectors are used
    as a tiebreaker only — cosine NEVER excludes a candidate.
    """
    n = len(passage_texts)
    if n == 0:
        return []
    direct = direct_cited_ids or set()

    scores = [anchor_score(claim_text, passage_texts[i]) for i in range(n)]

    cos: List[float] = [0.0] * n
    if claim_vec is not None and passage_vecs is not None:
        try:
            import numpy as np

            pv = np.asarray(passage_vecs, dtype=np.float32)
            cvv = np.asarray(claim_vec, dtype=np.float32)
            if pv.ndim == 2 and pv.shape[0] == n and pv.shape[1] == cvv.shape[-1]:
                cos = [float(x) for x in (pv @ cvv)]
        except Exception:  # noqa: BLE001 — cosine is a best-effort tiebreaker
            cos = [0.0] * n

    def _key(i: int):
        cid = passage_chunk_ids[i] if i < len(passage_chunk_ids) else None
        tier_a = 0 if (cid is not None and cid in direct) else 1
        return (tier_a, -scores[i], -cos[i], i)

    return sorted(range(n), key=_key)


__all__ = [
    "anchor_score",
    "order_passages_for_claim",
]
