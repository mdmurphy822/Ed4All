"""Hermetic fakes for the W2 objective-synthesis tests.

NO numpy / torch / sentence-transformers — these stubs let the four W2
invariants run in a slim CI checkout. The embedding stub returns plain
``List[List[float]]`` unit vectors (the ``lib.embedding._math.cosine_similarity``
helper has a pure-Python fallback when numpy is absent), and the NLI stub mimics
the ``score_batch(pairs=...)`` surface ``score_groundedness`` consumes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Fake NLI classifier
# ---------------------------------------------------------------------------
@dataclass
class _FakeScore:
    entailment: float
    contradiction: float = 0.0
    neutral: float = 0.0


class FakeNli:
    """Deterministic NLI stub.

    ``entail_when`` maps a (premise-substring, hypothesis-substring) signal to a
    high entailment. The default rule: entailment is high (0.9) when the premise
    text shares a configured keyword with the hypothesis; else low (0.1). A
    callable ``rule(premise, hypothesis) -> float`` overrides the keyword rule.
    """

    def __init__(
        self,
        *,
        rule: Optional[Any] = None,
        high: float = 0.9,
        low: float = 0.1,
    ) -> None:
        self._rule = rule
        self._high = high
        self._low = low
        # Provenance attrs score_groundedness probes (best-effort).
        self._revision = "fake-nli-v1"
        self.device = "cpu"

    def score_batch(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[_FakeScore]:
        out: List[_FakeScore] = []
        for premise, hypothesis in pairs:
            if self._rule is not None:
                ent = float(self._rule(premise, hypothesis))
            else:
                ent = (
                    self._high
                    if self._keyword_overlap(premise, hypothesis)
                    else self._low
                )
            out.append(_FakeScore(entailment=ent, contradiction=0.0))
        return out

    @staticmethod
    def _keyword_overlap(premise: str, hypothesis: str) -> bool:
        p = set(w.lower() for w in premise.split() if len(w) > 3)
        h = set(w.lower() for w in hypothesis.split() if len(w) > 3)
        return bool(p & h)


# ---------------------------------------------------------------------------
# Fake embedding client (no numpy)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _FakeResolved:
    kind: str = "fake-test"  # NOT "fake" so the anti-poisoning gate passes


class FakeEmbed:
    """Deterministic embedding stub returning unit vectors over a FIXED basis.

    Each text → an L2-normalized feature-hashed bag-of-words vector in a fixed
    ``dim``-dimensional space (token → ``sha256`` bucket). The fixed basis makes
    vectors COMPARABLE ACROSS ``encode_batch`` calls (objectives encoded in one
    call, blocks in another) — the property real sentence-transformers have and a
    per-call vocabulary would NOT. Two texts sharing most tokens get cosine near
    1.0; disjoint texts get cosine near 0.0 — enough to exercise the clustering /
    alignment machinery.
    """

    _DIM = 256

    def __init__(self) -> None:
        self.resolved = _FakeResolved()

    @staticmethod
    def _bucket(token: str, dim: int) -> int:
        import hashlib

        h = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], "big") % dim

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        dim = self._DIM
        vecs: List[List[float]] = []
        for t in texts:
            toks = [w.lower().strip(".,;:!?()'\"") for w in str(t).split()]
            toks = [w for w in toks if len(w) > 2]
            vec = [0.0] * dim
            for w in toks:
                vec[self._bucket(w, dim)] += 1.0
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            vecs.append(vec)
        return vecs


__all__ = ["FakeNli", "FakeEmbed", "_FakeScore"]
