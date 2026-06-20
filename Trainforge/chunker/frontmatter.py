"""Front-matter / donor / marketing CONTENT classifier for the canonical chunker.

Motivation
==========

A full-textbook PDF ingested through DART produces, as its FIRST chunks, the
book's *front matter*: the cover/title page, the contributing-author list, the
copyright + ISBN + Creative-Commons attribution block, the donor /
foundation acknowledgements, the marketing "web view" page, the Table of
Contents, and the "Key Features / Additional Resources" preface. None of that
is course content — yet the OpenStax 7B build's chunkset carried ~8 such chunks
(``..._chunk_00001``/``_00002``/...), which then got CITED by synthesized
objectives and polluted grounding. The curated baseline (``sample-course-a``)
escaped this only because its source HTML was a hand-trimmed chapter excerpt
that never contained the cover pages — there is no reusable pruning script to
formalise, so this module is the formalisation.

Contract
========

:func:`classify_frontmatter` is a PURE function over one chunk dict + its
ordinal position. It returns a :class:`FrontmatterVerdict` carrying
``is_frontmatter`` plus the matched signature names + math-veto state, so the
caller can (a) drop / flag the chunk and (b) emit a decision-capture event with
a dynamic rationale citing the matched signatures.

Design goals (high precision, low false-positive):

* **Multi-signal scoring.** A chunk is front-matter only if it hits ≥2 distinct
  signature families (or ≥1 strong signature while still inside the cover
  region) AND carries no real math content. Single-keyword matching is
  deliberately avoided.

* **Hard math veto.** A chunk that carries real math (worked-example markers
  like ``EXAMPLE 1.1`` / ``TRY IT``, arithmetic operators, equations, OR a
  dense cluster of distinct math terms in running prose) is NEVER classified
  front-matter, regardless of how many marketing signatures it also hits.
  This is the guard that protects real algebra content.

* **The "Foundations" false-positive guard.** 68/72 chunks of the OpenStax
  Elementary Algebra corpus contain the word "foundation" — almost entirely
  because of the legitimate running header / chapter title "Chapter 1
  Foundations". The donor signal therefore matches the *acknowledgement
  context* ("Foundation, Inc.", named-donor patterns) and NEVER bare
  "Foundation" / "Foundations". A chunk whose only "foundation" hit is the
  chapter title is not even a candidate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Cover region — the leading window where front-matter lives. A chunk past
# this ordinal needs the full ≥2-signature evidence; a chunk inside it can be
# flagged on a single STRONG signature (cover/copyright/donor/marketing/TOC)
# because that's where the colophon physically sits.
# ---------------------------------------------------------------------------

DEFAULT_COVER_REGION_CHUNKS: int = 8


# ---------------------------------------------------------------------------
# Math-content veto — a chunk that hits ANY of these is real course content.
#
# The veto requires GENUINE COMPUTATIONAL signal — worked-example markers
# numbered to a section ("EXAMPLE 1.1"), the "TRY IT" exercise lead-in,
# arithmetic operators between numbers, circled-letter sub-question markers,
# or multi-line equation glyphs. It deliberately does NOT veto on bare math
# *vocabulary* (the word "factor", "fraction", "polynomial", "Solution"
# standing alone), because a Table of Contents and the "Key Features" preface
# enumerate math TOPICS as menu entries ("1.5 Visualize Fractions", "Chapter 7
# Factoring") without containing a single worked step. Vocabulary-only matching
# let every TOC / preface chunk falsely veto out of the front-matter set.
# Measured separation on the real OpenStax corpus: front-matter chunks (cover /
# authors / copyright / donor / marketing / TOC / preface) carry ZERO of these
# markers; every real instructional chunk carries ≥1.
# ---------------------------------------------------------------------------

_MATH_CONTENT_RE = re.compile(
    r"\bEXAMPLE\s+\d+\.\d+\b"           # numbered worked example "EXAMPLE 1.1"
    r"|\bTRY\s+IT\s*:"                  # "TRY IT : : 1.9" exercise lead-in
    r"|[=÷×±≤≥≠]"                       # math operators / relations
    r"|\d+\s*[÷×]\s*\d+"               # explicit multiply/divide glyphs between numbers
    r"|\d+\s+[+\-*/·]\s+\d+"           # SPACED arithmetic "5 + 3", "5 · 3" (ISBN-safe:
                                        #   a hyphenated "947172-26" has no spaces, so
                                        #   the ISBN's internal hyphens never match)
    r"|⎛|⎞|⎝|⎠"                         # multi-line bracket glyphs
    r"|ⓐ|ⓑ|ⓒ|ⓓ|ⓔ"                      # circled-letter sub-question markers
)


def _has_math_content(text: str) -> bool:
    """True when ``text`` carries real (worked / computational) math content.

    This is the HARD VETO: a chunk that matches is never classified
    front-matter, no matter how many marketing signatures it also carries.
    See the module-level note above for why this is computational-signal-only
    rather than vocabulary-based.
    """

    return bool(_MATH_CONTENT_RE.search(text))


# ---------------------------------------------------------------------------
# Front-matter signature detectors. Each returns True/False; the verdict is a
# multi-signal vote. "STRONG" signatures (the colophon/marketing/TOC family)
# can flag a chunk inside the cover region on their own.
# ---------------------------------------------------------------------------

# --- copyright / license / publisher attribution -------------------------
_COPYRIGHT_RE = re.compile(
    r"access for free at openstax"
    r"|creative commons"
    r"|\bISBN[-\s]?(?:10|13)?\b|\bISBN\b"
    r"|all rights reserved"
    r"|©|\(c\)\s*\d{4}|copyright\s+\d{4}|copyright\s+©"
    r"|rice university"
    r"|\btrademark(?:s)?\b"
    r"|openstax logo|openstax cnx|connexions",
    re.IGNORECASE,
)


def _hit_copyright(text: str) -> bool:
    return bool(_COPYRIGHT_RE.search(text))


# --- author / contributor list -------------------------------------------
_AUTHOR_RE = re.compile(
    r"senior contributing authors?"
    r"|contributing authors?"
    r"|\bcontributing author\b"
    r"|^\s*reviewers?\s*$"
    r"|faculty reviewers?",
    re.IGNORECASE | re.MULTILINE,
)
# An author block reads "<NAME>, <COLLEGE/UNIVERSITY>" repeated. Detect ≥2 such
# "Name, Institution" lines as a corroborating author-list shape.
_AUTHOR_AFFIL_RE = re.compile(
    r"[A-Z][A-Za-z.\-]+(?:\s+[A-Z][A-Za-z.\-]+){1,3},\s*"
    r"[A-Z][A-Za-z.\-]+(?:\s+[A-Za-z.\-]+){0,4}\s+"
    r"(?:College|University|Community College|Institute)\b",
)


def _hit_author_list(text: str) -> bool:
    if _AUTHOR_RE.search(text):
        return True
    return len(_AUTHOR_AFFIL_RE.findall(text)) >= 2


# --- donor / foundation acknowledgement ----------------------------------
# CRITICAL false-positive guard: this MUST NOT match the chapter title
# "Chapter 1 Foundations" / a bare "foundation". It matches the donor
# *acknowledgement* surface only: an incorporated foundation ("Foundation,
# Inc." / "Foundation Trust"), named individual-donor patterns ("Tammy and
# Guillermo Treviño"), or explicit philanthropic giving language.
_DONOR_RE = re.compile(
    r"\bfoundation,?\s+inc\.?"                 # "Lowenstein Foundation, Inc."
    r"|\bfoundation\s+trust\b"
    r"|\bcharitable\s+(?:foundation|trust|fund)\b"
    r"|\bphilanthrop"
    r"|our\s+(?:generous\s+)?(?:donors?|supporters?|sponsors?)"
    r"|made\s+possible\s+by\s+(?:the\s+)?(?:generous\s+)?support"
    r"|thank(?:s)?\s+(?:our|the)\s+(?:generous\s+)?(?:donors?|sponsors?|supporters?|partners?)",
    re.IGNORECASE,
)
# Named-individual donor couplet ("Tammy and Guillermo Treviño") — two
# capitalised GIVEN names joined by "and", followed by a capitalised surname.
# This pattern is dangerously close to ordinary title-case section words
# ("Add and Subtract Integers", "Expressions and Equations In"), so it is
# guarded by an English-function-word stoplist: a "couplet" whose tokens are
# any of these common words is a section title, not a donor name. Only fires
# inside an explicit acknowledgement context (gated by the caller below) so it
# never trips on running prose.
_DONOR_COUPLE_RE = re.compile(
    r"\b[A-Z][a-z]{2,}\s+and\s+[A-Z][a-z]{2,}\s+[A-Z][a-zñáéíóú]{2,}\b",
)
# Tokens that mark a "X and Y Z" match as a section title rather than a name.
_NON_NAME_TOKENS = frozenset({
    "add", "subtract", "multiply", "divide", "solve", "graph", "use", "find",
    "expressions", "equations", "integers", "fractions", "decimals", "real",
    "numbers", "coverage", "scope", "student", "instructor", "resources",
    "systems", "linear", "inequalities", "properties", "polynomials",
    "factoring", "rational", "roots", "radicals", "models", "the", "and",
    "in", "of", "to", "for", "with", "review", "practice", "test", "chapter",
})
# Donor acknowledgements physically sit alongside one of these context cues.
_DONOR_CONTEXT_RE = re.compile(
    r"\bfoundation\b|\bdonor|\bsupporter|\bsponsor|\bgift\b|\bthank|\btrust\b"
    r"|generous|philanthrop|web view|access for free",
    re.IGNORECASE,
)


def _hit_donor(text: str) -> bool:
    if _DONOR_RE.search(text):
        return True
    # The named-couplet arm only fires inside an acknowledgement context AND
    # when the matched couplet doesn't read like a math section title.
    if not _DONOR_CONTEXT_RE.search(text):
        return False
    for match in _DONOR_COUPLE_RE.finditer(text):
        tokens = {tok.lower() for tok in re.findall(r"[A-Za-z]+", match.group(0))}
        if not (tokens & _NON_NAME_TOKENS):
            return True
    return False


# --- marketing / web-view / "access for free" ----------------------------
_MARKETING_RE = re.compile(
    r"study where you want"
    r"|web view"
    r"|our books are free"
    r"|access\.\s*the future of education"
    r"|note-taking features"
    r"|online highlighting"
    r"|get started at openstax",
    re.IGNORECASE,
)


def _hit_marketing(text: str) -> bool:
    return bool(_MARKETING_RE.search(text))


# --- "Key Features" / "Additional Resources" preface ---------------------
_FEATURES_RE = re.compile(
    r"key features"
    r"|additional resources"
    r"|student and instructor resources"
    r"|key features and boxes",
    re.IGNORECASE,
)


def _hit_features(text: str) -> bool:
    return bool(_FEATURES_RE.search(text))


# --- Table of Contents pattern -------------------------------------------
# A TOC line is "<N.N> <Title> <pagenum>" e.g. "1.1 Introduction to Whole
# Numbers 5". Require >= _TOC_LINE_FLOOR distinct such patterns so a single
# "1.1" cross-reference inside running prose doesn't trip it.
_TOC_LINE_RE = re.compile(r"\b\d+\.\d+\s+[A-Z][A-Za-z][^.\n]{4,60}?\s+\d{1,3}\b")
_TOC_HEADER_RE = re.compile(r"table of contents", re.IGNORECASE)
_TOC_LINE_FLOOR: int = 4


def _hit_toc(text: str) -> bool:
    if _TOC_HEADER_RE.search(text):
        return True
    return len(_TOC_LINE_RE.findall(text)) >= _TOC_LINE_FLOOR


# Strong signatures: any ONE of these inside the cover region is sufficient.
_STRONG_SIGNATURES: Dict[str, "object"] = {
    "copyright_attribution": _hit_copyright,
    "donor_acknowledgement": _hit_donor,
    "marketing_web_view": _hit_marketing,
    "table_of_contents": _hit_toc,
}
# Supporting signatures: corroborate a strong hit toward the ≥2 vote outside
# the cover region.
_SUPPORTING_SIGNATURES: Dict[str, "object"] = {
    "author_contributor_list": _hit_author_list,
    "book_features": _hit_features,
}


@dataclass
class FrontmatterVerdict:
    """Result of classifying one chunk for front-matter / donor contamination."""

    is_frontmatter: bool
    matched_signatures: List[str] = field(default_factory=list)
    has_math_content: bool = False
    in_cover_region: bool = False
    chunk_id: str = ""

    @property
    def signature_count(self) -> int:
        return len(self.matched_signatures)

    def rationale(self) -> str:
        """Dynamic, replayable rationale for the decision-capture event."""

        if self.is_frontmatter:
            return (
                f"chunk {self.chunk_id!r} classified front-matter / donor / "
                f"marketing contamination: matched {self.signature_count} "
                f"signature(s) {sorted(self.matched_signatures)} "
                f"(cover_region={self.in_cover_region}, "
                f"math_content={self.has_math_content}); excluded from "
                f"objective/grounding use before dedup/emit."
            )
        if self.has_math_content and self.matched_signatures:
            return (
                f"chunk {self.chunk_id!r} retained despite "
                f"{self.signature_count} front-matter signature(s) "
                f"{sorted(self.matched_signatures)}: hard math-content veto "
                f"(real equations/worked examples present) — never dropped."
            )
        return (
            f"chunk {self.chunk_id!r} retained: "
            f"{self.signature_count} front-matter signature(s) "
            f"{sorted(self.matched_signatures)} (cover_region="
            f"{self.in_cover_region}) below the drop threshold."
        )


def classify_frontmatter(
    chunk: Dict[str, object],
    position: int,
    *,
    cover_region_chunks: int = DEFAULT_COVER_REGION_CHUNKS,
) -> FrontmatterVerdict:
    """Classify one chunk as front-matter contamination (or not).

    Args:
        chunk: a chunk dict; reads ``text`` (falls back to ``html``) and
            ``id``.
        position: the chunk's 0-based ordinal in the emitted chunk list (used
            for the cover-region threshold relaxation).
        cover_region_chunks: chunks at ``position < this`` may be flagged on a
            single STRONG signature; later chunks need ≥2 signatures.

    Returns:
        A :class:`FrontmatterVerdict`.

    Decision rule (PURE, deterministic):

    1. Compute the hard math veto. If the chunk has real math content, it is
       ALWAYS retained — return immediately with ``is_frontmatter=False``
       (matched signatures still recorded for audit).
    2. Otherwise tally the strong + supporting signature hits.
    3. ``is_frontmatter`` iff:
         * (≥1 strong signature) AND (in cover region), OR
         * (≥2 distinct signatures total, strong + supporting).
    """

    chunk_id = str(chunk.get("id") or chunk.get("chunk_id") or "")
    text = str(chunk.get("text") or chunk.get("html") or "")
    in_cover = position < cover_region_chunks

    has_math = _has_math_content(text)

    matched: List[str] = []
    strong_hits = 0
    for name, detector in _STRONG_SIGNATURES.items():
        if detector(text):  # type: ignore[operator]
            matched.append(name)
            strong_hits += 1
    for name, detector in _SUPPORTING_SIGNATURES.items():
        if detector(text):  # type: ignore[operator]
            matched.append(name)

    # Hard math veto wins unconditionally — real content is never front-matter.
    if has_math:
        return FrontmatterVerdict(
            is_frontmatter=False,
            matched_signatures=matched,
            has_math_content=True,
            in_cover_region=in_cover,
            chunk_id=chunk_id,
        )

    total_hits = len(matched)
    is_frontmatter = bool(
        (strong_hits >= 1 and in_cover) or total_hits >= 2
    )

    return FrontmatterVerdict(
        is_frontmatter=is_frontmatter,
        matched_signatures=matched,
        has_math_content=False,
        in_cover_region=in_cover,
        chunk_id=chunk_id,
    )


def partition_frontmatter(
    chunks: List[Dict[str, object]],
    *,
    cover_region_chunks: int = DEFAULT_COVER_REGION_CHUNKS,
) -> "tuple[List[Dict[str, object]], List[FrontmatterVerdict]]":
    """Classify an ordered chunk list; return (kept_chunks, dropped_verdicts).

    ``chunks`` order is significant — ``position`` drives the cover-region
    threshold. Each dropped chunk yields a :class:`FrontmatterVerdict` so the
    caller can emit one decision-capture event per drop. Kept chunks are
    returned in their original order.
    """

    kept: List[Dict[str, object]] = []
    dropped: List[FrontmatterVerdict] = []
    for position, chunk in enumerate(chunks):
        verdict = classify_frontmatter(
            chunk, position, cover_region_chunks=cover_region_chunks
        )
        if verdict.is_frontmatter:
            dropped.append(verdict)
        else:
            kept.append(chunk)
    return kept, dropped


__all__ = [
    "DEFAULT_COVER_REGION_CHUNKS",
    "FrontmatterVerdict",
    "classify_frontmatter",
    "partition_frontmatter",
]
