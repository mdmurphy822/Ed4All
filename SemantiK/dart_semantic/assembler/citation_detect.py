"""Citation / cross-reference resolution helpers (Plans/04 §1.4 + §2).

Implements the ``citation_unresolved`` detection trigger:

  1. Build a reference index from the document's headings (section
     numbers parsed from heading text) — Pass 1 in Plans/04 §1.4.
  2. Regex-find candidate inline references in region prose text —
     ``Section X.Y`` / ``Figure N`` / ``Table N`` / ``[N]`` (Pass 2).
  3. For each match, classify it as RESOLVED / AMBIGUOUS / DANGLING /
     NEAR_MISS (Pass 3):

       * RESOLVED  — target exists in the index → not a gap (the
         assembler would have wrapped it in an ``<a>``; v1 doesn't do
         reference resolution, so we simply don't flag it).
       * AMBIGUOUS — ``see above`` / ``the foregoing`` / etc. → never
         flagged (Plans/04 §1.4 + the do-NOT list).
       * DANGLING  — target missing AND no near-miss in the index →
         not flagged (genuinely external; gap-fill can't help).
       * NEAR_MISS — target missing but a near-miss target exists
         (Levenshtein ≤ 2 on the section number, or |ΔN| ≤ 1 for a
         numeric label) → flag ``citation_unresolved``.

The near-miss gate is the whole point: a ``Section 4.2`` ref with no
``4.2`` heading but a ``Section 4.1`` heading is a probable parse slip
the gap-fill specialist can repair; a ``[99]`` ref in a 3-entry
bibliography is genuinely dangling and is left as plain text.

CPU-only, pure-python, no ML.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Reference patterns (Plans/04 §1.4 Pass 2)
# --------------------------------------------------------------------------- #

# Section / Appendix references carry a dotted alphanumeric number.
_SECTION_RE = re.compile(
    r"\b(?:Section|Sec\.?|Appendix)\s+([0-9A-Z]+(?:\.[0-9A-Za-z]+)*)\b",
)
# Figure / Table references carry a bare integer.
_FIGURE_RE = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+)\b")
_TABLE_RE = re.compile(r"\b(?:Table|Tab\.?)\s+(\d+)\b")
# Numbered citation: a bracketed integer (single — multi-cite like
# "[1, 2]" is out of scope for the v1 near-miss gate; we only flag the
# clean single-number form so the restored anchor is unambiguous).
_NUMBERED_CITE_RE = re.compile(r"\[(\d+)\]")

# Leading section-number prefix on a heading text (the index builder).
# Matches "3.2 Methods", "A.1 Appendix", "4 Results".
_HEADING_SECNUM_RE = re.compile(r"^\s*([0-9A-Z]+(?:\.[0-9A-Za-z]+)*)\b")

# Ambiguous references that are genuinely under-specified in the source —
# gap-fill cannot beat "leave plain text". Never flagged (Plans/04 §1.4).
_AMBIGUOUS_RE = re.compile(
    r"\b(?:see\s+above|see\s+below|the\s+foregoing|the\s+previous\s+section|"
    r"the\s+following\s+section|the\s+preceding\s+section|"
    r"as\s+(?:mentioned|noted|discussed)\s+(?:above|below|earlier))\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RefMatch:
    """One inline reference matched in a region's prose text."""

    reference_kind: str  # "section" | "figure" | "table" | "numbered_citation"
    match_text: str       # the literal span, e.g. "Section 4.2"
    key: str              # the parsed number/id, e.g. "4.2" or "5"
    start: int            # char offset in the region text
    end: int


# Caption-defining patterns — these mark a figure/table number as a real
# target in the document (e.g. "Figure 3: ..." / "Table 2." at the start
# of a caption region). Used to populate the present-labels sets so a
# numeric near-miss can distinguish a parse-slip ref from a dangling one.
_FIGURE_DEF_RE = re.compile(r"^\s*(?:Figure|Fig\.?)\s+(\d+)\b")
_TABLE_DEF_RE = re.compile(r"^\s*(?:Table|Tab\.?)\s+(\d+)\b")
# A numbered-citation TARGET appears as a leading "[N]" in a bibliography
# entry (the REFERENCE block convention).
_BIB_DEF_RE = re.compile(r"^\s*\[(\d+)\]")


@dataclass(frozen=True)
class ReferenceIndex:
    """Document reference index built from headings + captions + bib entries.

    Pass 1 of Plans/04 §1.4:

      * ``section_numbers`` maps a section-number string (e.g. "3.2") to
        the anchor id of the heading carrying it.
      * ``figure_numbers`` / ``table_numbers`` / ``citation_numbers`` are
        the sets of integer labels DEFINED by caption / bibliography
        regions — the targets a ``Figure N`` / ``Table N`` / ``[N]`` ref
        could resolve to.
    """

    section_numbers: dict[str, str]  # "3.2" -> "methods" (heading anchor id)
    figure_numbers: set[str]
    table_numbers: set[str]
    citation_numbers: set[str]

    def has_section(self, num: str) -> bool:
        return num in self.section_numbers

    def anchor_for_section(self, num: str) -> str | None:
        return self.section_numbers.get(num)

    def present_for_kind(self, reference_kind: str) -> set[str]:
        if reference_kind == "figure":
            return self.figure_numbers
        if reference_kind == "table":
            return self.table_numbers
        if reference_kind == "numbered_citation":
            return self.citation_numbers
        return set()


def build_reference_index(
    heading_texts: list[str],
    heading_ids: list[str],
    *,
    region_texts: list[str] | None = None,
) -> ReferenceIndex:
    """Build the document reference index from headings (+ caption/bib regions).

    Pass 1 of Plans/04 §1.4:

      * Headings contribute their leading section-number prefix → anchor id.
      * ``region_texts`` (when given) are scanned for caption-defining
        ``Figure N`` / ``Table N`` leads and bibliography ``[N]`` leads,
        which populate the present-label sets the numeric near-miss gate
        consults. Headings without a leading number contribute nothing.
    """
    section_numbers: dict[str, str] = {}
    for text, ident in zip(heading_texts, heading_ids):
        m = _HEADING_SECNUM_RE.match(text or "")
        if m is None:
            continue
        num = m.group(1)
        # A bare letter ("A") with no dot is too noisy (matches the word
        # "A" starting a heading) — require either a dot or a digit.
        if not any(c.isdigit() for c in num) and "." not in num:
            continue
        section_numbers.setdefault(num, ident or "")

    figure_numbers: set[str] = set()
    table_numbers: set[str] = set()
    citation_numbers: set[str] = set()
    for text in region_texts or []:
        fm = _FIGURE_DEF_RE.match(text or "")
        if fm is not None:
            figure_numbers.add(fm.group(1))
        tm = _TABLE_DEF_RE.match(text or "")
        if tm is not None:
            table_numbers.add(tm.group(1))
        bm = _BIB_DEF_RE.match(text or "")
        if bm is not None:
            citation_numbers.add(bm.group(1))

    return ReferenceIndex(
        section_numbers=section_numbers,
        figure_numbers=figure_numbers,
        table_numbers=table_numbers,
        citation_numbers=citation_numbers,
    )


def find_references(text: str) -> list[RefMatch]:
    """Run the Plans/04 §1.4 Pass-2 regex set over one region's text."""
    out: list[RefMatch] = []
    for m in _SECTION_RE.finditer(text):
        out.append(RefMatch(
            reference_kind="section",
            match_text=m.group(0),
            key=m.group(1),
            start=m.start(),
            end=m.end(),
        ))
    for m in _FIGURE_RE.finditer(text):
        out.append(RefMatch(
            reference_kind="figure",
            match_text=m.group(0),
            key=m.group(1),
            start=m.start(),
            end=m.end(),
        ))
    for m in _TABLE_RE.finditer(text):
        out.append(RefMatch(
            reference_kind="table",
            match_text=m.group(0),
            key=m.group(1),
            start=m.start(),
            end=m.end(),
        ))
    for m in _NUMBERED_CITE_RE.finditer(text):
        out.append(RefMatch(
            reference_kind="numbered_citation",
            match_text=m.group(0),
            key=m.group(1),
            start=m.start(),
            end=m.end(),
        ))
    out.sort(key=lambda r: r.start)
    return out


def is_ambiguous(surrounding_text: str) -> bool:
    """True if the surrounding text contains an ambiguous reference cue."""
    return _AMBIGUOUS_RE.search(surrounding_text or "") is not None


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein edit distance (small strings; no deps)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


def near_miss_section_targets(
    num: str,
    index: ReferenceIndex,
) -> list[tuple[str, str]]:
    """Return (section_number, anchor_id) near-misses for a missing section.

    Near-miss = Levenshtein distance ≤ 2 on the section-number string
    (Plans/04 §2). The exact target must be ABSENT (callers check that
    before calling, but we exclude the exact key defensively).
    """
    out: list[tuple[str, str]] = []
    for cand_num, anchor in index.section_numbers.items():
        if cand_num == num:
            continue
        if _levenshtein(num, cand_num) <= 2:
            out.append((cand_num, anchor))
    return out


def _numeric_near_miss(key: str, present: set[str]) -> list[str]:
    """Return numeric labels within |ΔN| ≤ 1 of ``key`` present in ``present``."""
    try:
        n = int(key)
    except ValueError:
        return []
    return [str(c) for c in (n - 1, n + 1) if str(c) in present]


@dataclass(frozen=True)
class CitationGap:
    """A NEAR_MISS classification ready to become a ``GapSlot.context``."""

    ref: RefMatch
    surrounding_50_chars: str
    candidate_targets: list[dict[str, str]]  # [{key, anchor_id, snippet}, ...]


def classify_reference(
    ref: RefMatch,
    text: str,
    index: ReferenceIndex,
) -> CitationGap | None:
    """Classify one matched reference; return a CitationGap iff NEAR_MISS.

    Decision ladder (Plans/04 §1.4 Pass 3):

      RESOLVED  — target exists in the index → return None.
      AMBIGUOUS — the surrounding text is an ambiguous cue → None.
      DANGLING  — missing AND no near-miss → None.
      NEAR_MISS — missing AND a near-miss target exists → CitationGap.
    """
    lo = max(0, ref.start - 50)
    hi = min(len(text), ref.end + 50)
    surrounding = text[lo:hi]

    # AMBIGUOUS — never flag (do-NOT list). Checked against the local
    # window so an ambiguous cue elsewhere in a long paragraph doesn't
    # suppress a genuine broken ref.
    if is_ambiguous(surrounding):
        return None

    candidate_targets: list[dict[str, str]] = []
    if ref.reference_kind == "section":
        if index.has_section(ref.key):
            return None  # RESOLVED
        for cand_num, anchor in near_miss_section_targets(ref.key, index):
            candidate_targets.append({
                "key": cand_num,
                "anchor_id": anchor or f"sec-{cand_num.replace('.', '-')}",
                "snippet": cand_num,
            })
    else:
        present = index.present_for_kind(ref.reference_kind)
        if ref.key in present:
            return None  # RESOLVED
        prefix = {
            "figure": "fig",
            "table": "tab",
            "numbered_citation": "ref",
        }[ref.reference_kind]
        for cand_num in _numeric_near_miss(ref.key, present):
            candidate_targets.append({
                "key": cand_num,
                "anchor_id": f"{prefix}-{cand_num}",
                "snippet": cand_num,
            })

    if not candidate_targets:
        return None  # DANGLING — no near-miss; leave plain text

    return CitationGap(
        ref=ref,
        surrounding_50_chars=surrounding,
        candidate_targets=candidate_targets,
    )


__all__ = [
    "CitationGap",
    "RefMatch",
    "ReferenceIndex",
    "build_reference_index",
    "classify_reference",
    "find_references",
    "is_ambiguous",
    "near_miss_section_targets",
]
