"""Answer-key / TOC "apparatus dump" chunk filter for the canonical chunker.

Motivation
==========

A scanned textbook produces two families of NON-instructional chunk that the
front-matter filter (:mod:`Trainforge.chunker.frontmatter`) does not catch,
because they sit MID-corpus (past the cover region) and carry no marketing /
copyright signatures:

* **Answer-key dumps** — an end-of-chapter answer key OCR'd as a bare
  ``"problem#. numeric-answer"`` run::

      "46. 550 47. 22,335 48. 39,075 37. 84 38. 9,696 39. 75 40. 78 ..."

  These get cited by synthesized objectives and pollute grounding with pure
  number soup.

* **Chapter-outline / Table-of-Contents chunks** — a chapter's navigation
  outline OCR'd as an ``overview`` content chunk::

      "Chapter Outline 1.2 Use the Language of Algebra 1.3 Add and Subtract ..."

  This is a menu of section titles, not instructional prose, yet it lands as an
  ``overview`` and gets treated as content.

Both are captured here as a single mid-corpus drop pass, sibling to the
front-matter filter and wired at the same emit site behind
``TRAINFORGE_DROP_APPARATUS_DUMPS``.

Contract
========

:func:`classify_apparatus_dump` is a PURE function over one chunk dict. It is
HIGH-PRECISION by design — it drops only unambiguous answer-key runs and
explicit chapter-outline / ToC chunks, so real math-dense exercise/example
chunks (which carry LaTeX and legitimate numbers) are never dropped. Calibrated
against a real 3-chapter algebra scan: drops exactly the 1 pure answer-key
chunk + the 3 chapter-outline chunks, drops ZERO instructional/exercise prose.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "FLAG_ENV",
    "ApparatusVerdict",
    "classify_apparatus_dump",
    "partition_apparatus_dumps",
]

FLAG_ENV = "TRAINFORGE_DROP_APPARATUS_DUMPS"

# ---------------------------------------------------------------------------
# Answer-key detection.
#
# A TRUE answer-key pair is "problem#. <bare numeric answer>" where the answer
# is a pure number (commas / decimals allowed) NOT followed by a word / LaTeX /
# math markup. Real exercise prose ("... simplify. 554. $\frac{3}{4}$ ...")
# carries an item number followed by LaTeX, so it never matches this pattern.
# ---------------------------------------------------------------------------

_ANSWER_PAIR_RE = re.compile(
    r"(?<![\d.])\d{1,3}\.\s+\d[\d,]*(?:\.\d+)?(?![\w$\\])"
)

#: Minimum answer-key pairs before a chunk is even a candidate.
_ANSWER_PAIR_MIN = 8

#: The answer-key run must DOMINATE the chunk head: this fraction of the first
#: ``_LEAD_TOKENS`` word tokens must be consumed by answer-key pairs. A chunk
#: with a real instructional lead-in (a question stem, an "In the following
#: exercises" banner) fails this floor and is kept.
_LEAD_TOKENS = 30
_LEAD_COVERAGE_MIN = 0.40


# ---------------------------------------------------------------------------
# Chapter-outline / Table-of-Contents detection.
#
# A high-precision leading-header match. "Chapter Outline" / "Table of
# Contents" as the chunk's first words, corroborated by >= _OUTLINE_ENUM_MIN
# "N.M <Title>" section enumerations (so a prose chunk that merely mentions the
# phrase mid-sentence is never dropped).
# ---------------------------------------------------------------------------

_OUTLINE_HEADER_RE = re.compile(
    r"^\s*(?:chapter\s+outline|table\s+of\s+contents)\b", re.IGNORECASE
)
_OUTLINE_ENUM_RE = re.compile(r"\b\d+\.\d+\s+[A-Z]")
_OUTLINE_ENUM_MIN = 3


@dataclass
class ApparatusVerdict:
    """Result of classifying one chunk as an answer-key / ToC apparatus dump."""

    is_apparatus: bool
    reason: str = ""
    chunk_id: str = ""

    def rationale(self) -> str:
        """Dynamic, replayable rationale for the decision-capture event."""
        if self.is_apparatus:
            return (
                f"chunk {self.chunk_id!r} dropped as non-instructional "
                f"apparatus dump ({self.reason}): answer-key/ToC layout, not "
                f"prose — excluded from objective/grounding use before emit."
            )
        return f"chunk {self.chunk_id!r} retained (not an apparatus dump)."


def _lead_coverage(text: str) -> float:
    toks = text.split()[:_LEAD_TOKENS]
    if not toks:
        return 0.0
    joined = " ".join(toks)
    covered = sum(len(m.group(0).split()) for m in _ANSWER_PAIR_RE.finditer(joined))
    return covered / len(toks)


def classify_apparatus_dump(chunk: Dict[str, object]) -> ApparatusVerdict:
    """Classify one chunk as an answer-key / ToC apparatus dump (or not).

    PURE + deterministic. Reads ``text`` (falls back to ``html``) and ``id``.
    """
    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    text = str(chunk.get("text") or chunk.get("html") or "")
    if not text.strip():
        return ApparatusVerdict(False, chunk_id=chunk_id)

    # Answer-key dump: many "N. <number>" pairs that dominate the chunk head.
    pair_count = len(_ANSWER_PAIR_RE.findall(text))
    if pair_count >= _ANSWER_PAIR_MIN and _lead_coverage(text) >= _LEAD_COVERAGE_MIN:
        return ApparatusVerdict(
            True,
            reason=(
                f"answer_key_dump: {pair_count} 'N. <number>' pairs, "
                f"lead_coverage={_lead_coverage(text):.2f}"
            ),
            chunk_id=chunk_id,
        )

    # Chapter-outline / ToC: leading header + several N.M section enumerations.
    if _OUTLINE_HEADER_RE.search(text):
        enum_count = len(_OUTLINE_ENUM_RE.findall(text))
        if enum_count >= _OUTLINE_ENUM_MIN:
            return ApparatusVerdict(
                True,
                reason=f"chapter_outline_toc: {enum_count} 'N.M Title' enumerations",
                chunk_id=chunk_id,
            )

    return ApparatusVerdict(False, chunk_id=chunk_id)


def partition_apparatus_dumps(
    chunks: List[Dict[str, object]],
) -> "Tuple[List[Dict[str, object]], List[ApparatusVerdict]]":
    """Classify a chunk list; return (kept_chunks, dropped_verdicts).

    Each dropped chunk yields an :class:`ApparatusVerdict` so the caller can
    emit one decision-capture event per drop. Kept chunks keep their order.
    """
    kept: List[Dict[str, object]] = []
    dropped: List[ApparatusVerdict] = []
    for chunk in chunks:
        verdict = classify_apparatus_dump(chunk)
        if verdict.is_apparatus:
            dropped.append(verdict)
        else:
            kept.append(chunk)
    return kept, dropped
