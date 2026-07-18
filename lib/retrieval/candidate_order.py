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
      (b) descending POOL-IDF lexical anchor score (see below) + a small
          sequential-locality prior for chunks adjacent to a direct-cited chunk;
      (c) descending MiniLM cosine (FINAL tiebreaker; skipped when the caller
          passes no vectors — embedder-unavailable degrades to lexical-only,
          measured near-random so it must never outrank lexical signal);
      (d) ascending original index (stable — makes the order reproducible).

Lexical anchor (tier b) — pool-IDF weighted, all computed over THE BLOCK'S OWN
candidate pool (measured: cosine rank of the true entailing chunk is
uniform-random here, so the lexical tier carries the ordering and a flat
shared-token count was too weak — a token shared with 300/379 pool chunks is
worthless, one shared with 3 is gold):

  * shared uncommon tokens (len ≥ 5), each weighted by ``log(N / df)`` where
    ``N`` = pool size and ``df`` = number of pool chunks containing the token;
  * shared numeric / equation-ish tokens, IDF-weighted and ×2 (a matching
    literal / formula is a far stronger grounding signal);
  * shared rare word-bigrams (both words len ≥ 4), weighted by bigram IDF over
    the pool — textbook claims echo distinctive phrases ("distributive
    property", "least common denominator").

Ordering NEVER affects verdicts — only *speed* (how early a supported claim
retires). The frontier fold is a pure max over the identical pool with the
identical entailment floor regardless of visit order, so ``ORDERING_VERSION``
is deliberately NOT part of any groundedness / prose-entailment fingerprint
(same pool, same fold → bitwise-identical verdict; only the retire depth moves).

Pure CPU: no model loads, no embedding here — the caller supplies any vectors.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

#: Ordering algorithm identity. Bumped when the tier-b lexical scoring changes.
#: v1 = flat |shared uncommon| + 2×|shared numeric| count. v2 = pool-IDF weighted
#: uncommon + numeric + rare-bigram anchors, plus the sequential-locality prior.
#: This is a SPEED knob only — see the module docstring on why it is excluded
#: from every verdict fingerprint.
ORDERING_VERSION = 2

#: An "uncommon" content token: a lowercased alphabetic run of length ≥ 5. Short
#: / numeric tokens are handled by the numeric-anchor tier so common stop-words
#: (``the``, ``and``) never inflate the anchor score.
_UNCOMMON_TOKEN_RE = re.compile(r"[a-z]{5,}")

#: Ordered lowercased alphabetic word run (len ≥ 4 filtered later) — used to form
#: consecutive word-bigrams for the rare-bigram anchor tier.
_WORD_SEQ_RE = re.compile(r"[a-z]{4,}")

#: Whitespace-delimited token (for the numeric / equation-ish anchor set).
_WORD_RE = re.compile(r"\S+")

#: A token counts as numeric / equation-ish when it carries any of these — a
#: digit or an arithmetic operator glyph embedded in the word (``3``, ``25*4``,
#: ``x^2``, ``a=b``). Shared numeric anchors are worth 2× an uncommon token
#: because a matching literal / formula is a far stronger grounding signal.
_NUMERICISH_CHARS = frozenset("0123456789=^*/+-")

#: Small additive prior added to a NON-cited passage that sits next to a
#: direct-cited chunk (cited neighborhoods carry the support). Deliberately small
#: — roughly one token shared with ~1/e of the pool — so it breaks ties among
#: weak-lexical passages without ever dominating a real multi-token IDF match.
_LOCALITY_BONUS = 1.0

#: Trailing-integer split of a ``chunk_id`` → ``(prefix, ordinal)`` for the
#: sequential-locality adjacency test (``..._03`` neighbors ``..._02`` / ``_04``).
_CHUNK_ORD_RE = re.compile(r"^(.*?)(\d+)$")


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


def _bigrams(text: Optional[str]) -> Set[Tuple[str, str]]:
    """Set of consecutive lowercased word-bigrams (both words len ≥ 4)."""
    words = _WORD_SEQ_RE.findall((text or "").lower())
    return set(zip(words, words[1:]))


def _chunk_ordinal(chunk_id: Optional[str]) -> Optional[Tuple[str, int]]:
    """Split a ``chunk_id`` into ``(non-numeric prefix, trailing ordinal)``.

    ``None`` when the id has no trailing integer (unorderable). Only same-prefix
    ordinals are treated as adjacent, so ``sec_1_03`` never neighbors ``sec_2_04``.
    """
    m = _CHUNK_ORD_RE.match(chunk_id or "")
    if m is None:
        return None
    return (m.group(1), int(m.group(2)))


def _pool_idf(doc_token_sets: Sequence[Set[Any]], n: int) -> Dict[Any, float]:
    """``token → log(N / df)`` over the pool (df = # docs containing the token).

    A token in every pool chunk gets IDF 0 (worthless); a token in one chunk
    gets ``log(N)`` (gold). ``df`` is document-frequency (set membership per
    passage), so a token repeated within one passage still counts once.
    """
    df: Counter = Counter()
    for s in doc_token_sets:
        df.update(s)
    return {t: math.log(n / c) for t, c in df.items() if c > 0}


def anchor_score(claim_text: str, passage_text: str) -> int:
    """LEGACY flat lexical grounding-affinity score (pool-free).

    ``|shared uncommon tokens| + 2 × |shared numeric/equation-ish tokens|``.
    Deterministic, symmetric in text content, and independent of any model.

    Retained as the ``ORDERING_VERSION == 1`` scorer for the offline A/B harness
    (``scratchpad/ordering_depth_eval.py``) and as a simple public helper. The
    production ordering (:func:`order_passages_for_claim`) uses the pool-IDF
    weighting instead — a flat count cannot tell a token shared with 300/379
    pool chunks from one shared with 3.
    """
    shared_uncommon = len(_uncommon_tokens(claim_text) & _uncommon_tokens(passage_text))
    shared_numeric = len(_numericish_tokens(claim_text) & _numericish_tokens(passage_text))
    return shared_uncommon + 2 * shared_numeric


def _idf_lexical_scores(
    claim_text: str,
    passage_texts: Sequence[str],
) -> List[float]:
    """Pool-IDF weighted lexical anchor score per passage (tier b, no locality).

    Uncommon-token IDF sum + 2 × numeric-token IDF sum + rare-bigram IDF sum,
    every IDF computed over ``passage_texts`` itself (the block's own pool).
    """
    n = len(passage_texts)
    unc_sets = [_uncommon_tokens(t) for t in passage_texts]
    num_sets = [_numericish_tokens(t) for t in passage_texts]
    big_sets = [_bigrams(t) for t in passage_texts]

    idf_unc = _pool_idf(unc_sets, n)
    idf_num = _pool_idf(num_sets, n)
    idf_big = _pool_idf(big_sets, n)

    c_unc = _uncommon_tokens(claim_text)
    c_num = _numericish_tokens(claim_text)
    c_big = _bigrams(claim_text)

    scores: List[float] = []
    for i in range(n):
        s = sum(idf_unc.get(t, 0.0) for t in (c_unc & unc_sets[i]))
        s += 2.0 * sum(idf_num.get(t, 0.0) for t in (c_num & num_sets[i]))
        s += sum(idf_big.get(b, 0.0) for b in (c_big & big_sets[i]))
        scores.append(s)
    return scores


def _locality_bonuses(
    passage_chunk_ids: Sequence[Optional[str]],
    direct: Set[str],
    n: int,
) -> List[float]:
    """``_LOCALITY_BONUS`` for each NON-cited passage adjacent to a cited chunk.

    Adjacency by pool index (``i±1`` is a direct-cited position) OR by parseable
    same-prefix chunk-id ordinal (``±1``). Cited passages are already floated by
    tier (a), so they never receive the neighbor bonus.
    """
    bonus = [0.0] * n
    if not direct:
        return bonus

    cited_idx = {
        i
        for i in range(n)
        if (passage_chunk_ids[i] if i < len(passage_chunk_ids) else None) in direct
    }
    if not cited_idx:
        return bonus

    cited_ords: Set[Tuple[str, int]] = set()
    for i in cited_idx:
        o = _chunk_ordinal(passage_chunk_ids[i] if i < len(passage_chunk_ids) else None)
        if o is not None:
            cited_ords.add(o)

    for i in range(n):
        if i in cited_idx:
            continue
        adjacent = (i - 1) in cited_idx or (i + 1) in cited_idx
        if not adjacent and cited_ords:
            o = _chunk_ordinal(
                passage_chunk_ids[i] if i < len(passage_chunk_ids) else None
            )
            if o is not None:
                prefix, num = o
                if (prefix, num - 1) in cited_ords or (prefix, num + 1) in cited_ords:
                    adjacent = True
        if adjacent:
            bonus[i] = _LOCALITY_BONUS
    return bonus


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
    reordering, never a filter. Tier (b) is the pool-IDF lexical anchor (see the
    module docstring) plus a small sequential-locality prior for passages next to
    a direct-cited chunk. ``claim_vec`` / ``passage_vecs`` (unit-normalized
    MiniLM vectors from the caller) drive tier (c); pass ``None`` for either to
    fall back to lexical-only ordering (embedder unavailable). Vectors are used
    as a FINAL tiebreaker only — cosine NEVER excludes or outranks a lexical
    signal.
    """
    n = len(passage_texts)
    if n == 0:
        return []
    direct = direct_cited_ids or set()

    lex = _idf_lexical_scores(claim_text, passage_texts)
    loc = _locality_bonuses(passage_chunk_ids, direct, n)
    scores = [lex[i] + loc[i] for i in range(n)]

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
    "ORDERING_VERSION",
    "anchor_score",
    "order_passages_for_claim",
]
