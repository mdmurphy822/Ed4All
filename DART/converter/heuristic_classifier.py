"""Phase 2 (Wave 12 shim): heuristic block classification.

Ports the regex logic that lives in
``MCP/tools/pipeline_tools.py::_raw_text_to_accessible_html`` into the
new ``BlockRole``-returning interface, plus adds the supplementary
patterns the plan calls for (paper-section keywords, TOC detection,
front-matter hints, author / affiliation metadata). No LLM dependency,
no file-system access.

Wave 14 will add ``LLMBackend``-backed classification; this heuristic
classifier remains the offline fallback. Every classifier (heuristic,
LLM, mock) returns ``List[ClassifiedBlock]`` in the same shape, so
callers never need to know which one produced the decisions.
"""

from __future__ import annotations

import logging
import re
from typing import List

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex definitions (ported from pipeline_tools.py + plan spec)
# ---------------------------------------------------------------------------

# "Chapter 3: Foo" / "1. Foo" / "I. Foo" / "Part 2: Bar"
_CHAPTER_HEADING = re.compile(
    r"^(?:"
    r"(?:Chapter|Part|Section|Unit)\s+\d+[.:]\s*|"
    r"(?:I{1,3}V?|VI{0,3}|IX|X{1,3})\.\s+|"
    r"\d{1,2}\.\s+"
    r")(.+)$"
)

# Title-Case short line without trailing punctuation (candidate sub heading)
_SUB_HEADING = re.compile(r"^[A-Z][A-Za-z\s,&:'\-]{5,80}$")

# Numbered paper section ("1 Introduction" / "2.1 Related Work")
_PAPER_SECTION_NUMBERED = re.compile(
    r"^\s*\d+(?:\.\d+){0,2}\s+([A-Z][A-Za-z\s,&:'\-]{2,80})\s*$"
)

# Canonical paper / report section keywords. When a standalone block
# matches one of these (case-insensitive, with optional colon), the
# classifier promotes it to the corresponding structural role.
_PAPER_SECTION_KEYWORDS = {
    "abstract": BlockRole.ABSTRACT,
    "introduction": BlockRole.SECTION_HEADING,
    "background": BlockRole.SECTION_HEADING,
    "related work": BlockRole.SECTION_HEADING,
    "methodology": BlockRole.SECTION_HEADING,
    "methods": BlockRole.SECTION_HEADING,
    "approach": BlockRole.SECTION_HEADING,
    "results": BlockRole.SECTION_HEADING,
    "discussion": BlockRole.SECTION_HEADING,
    "conclusion": BlockRole.SECTION_HEADING,
    "conclusions": BlockRole.SECTION_HEADING,
    "references": BlockRole.SECTION_HEADING,
    "bibliography": BlockRole.SECTION_HEADING,
    "acknowledgements": BlockRole.SECTION_HEADING,
    "acknowledgments": BlockRole.SECTION_HEADING,
    "appendix": BlockRole.SECTION_HEADING,
    "keywords": BlockRole.KEYWORDS,
}

# "Title ........... 42" / "Title . . . . . 42" / "Title    42" TOC-style
# lines. Whitespace-collapsed pdftotext output often shows the second
# form once soft-hyphens are rejoined, so we accept either leader.
_TOC_DOT_LEADER = re.compile(
    r"^.{5,80}(?:\.{3,}|(?:\.\s){3,})\s*\d{1,4}\s*$"
)
_TOC_ENTRY = re.compile(r"^.{5,60}\s{3,}\d{1,4}\s*$")

# Roman-numeral page footers ("iii", "xiv", "MMXX").
_TOC_ROMAN_PAGE = re.compile(r"^\s*(?:[ivxlcdm]{1,8}|[IVXLCDM]{1,8})\s*$")

# Copyright / ISBN / licensing front-matter lines.
_FRONT_MATTER_HINT = re.compile(
    r"(?:^|\s)(?:©|\(c\)|copyright|isbn[-:\s]|all rights reserved|"
    r"licen[cs]ed under|creative commons|public domain)",
    re.IGNORECASE,
)

# arxiv-style metadata header ("arXiv:1234.56789v2 [cs.LG] 3 Mar 2024").
_ARXIV_META_HINT = re.compile(r"arxiv\s*:\s*\d{4}\.\d{4,5}", re.IGNORECASE)

# Email-address lines used to mark author affiliations in papers.
_EMAIL_HINT = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# Column-layout heuristic: three-or-more wide-whitespace runs per line
# across many short lines indicates multi-column text that pdftotext
# interleaved. The classifier demotes such blocks to low-confidence
# paragraphs so the assembler can optionally skip them later.
_COLUMN_LAYOUT = re.compile(r"\S+\s{3,}\S+\s{3,}\S+")

# Bibliography entry: starts with "[N]" citation or "Lastname, F."
_BIBLIOGRAPHY_ENTRY = re.compile(
    r"^(?:\[\d{1,3}\]|\(\d{1,3}\))\s+|"
    r"^[A-Z][a-zA-Z\-']+,\s+[A-Z]\.(?:\s*[A-Z]\.)*(?:\s*,|\s+&|\s+and)"
)

# Footnote marker: "[1] ..." / "¹ ..." / "* ..." style prefix.
_FOOTNOTE = re.compile(r"^(?:\*+|\[\d{1,3}\]|\(\d{1,3}\)|\d{1,3}\.?)\s+\S")


# ---------------------------------------------------------------------------
# Support helpers
# ---------------------------------------------------------------------------

def _is_valid_subheading(text: str) -> bool:
    """Guardrail for ``_SUB_HEADING`` matches.

    Subheadings must be title-case, short, standalone (no trailing
    punctuation), and carry no column-layout artefacts. The intent is
    to avoid promoting a runaway prose line starting with a capital
    letter into an ``<h3>``.
    """
    if not text or len(text) > 80:
        return False
    words = text.split()
    if len(words) == 0 or len(words) > 10:
        return False
    if text.endswith((".", ",", ":", ";", "!", "?")):
        return False
    if not text[0].isupper():
        return False
    if _COLUMN_LAYOUT.search(text):
        return False
    return True


def _is_low_signal_heading(text: str) -> bool:
    """Reject noisy heading candidates.

    A heading is "low signal" when it looks like a place name, a date
    stamp, a phone number, a URL, or boilerplate front-matter. These
    were previously passing through the ``sub_heading`` regex and
    landing in the output as spurious ``<h3>`` nodes.
    """
    if not text:
        return True

    upper_only = text.isupper() and len(text.split()) <= 4
    looks_like_place = bool(
        re.match(r"^[A-Z]{2,}(?:\s+[A-Z]{2,})*$", text)
        or re.match(r"^[A-Z][a-z]+,\s*[A-Z]{2}$", text)  # "Vancouver, BC"
    )
    looks_like_date = bool(
        re.match(r"^\d{1,2}\s+\w+\s+\d{4}$", text)
        or re.match(r"^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", text)
    )
    looks_like_phone = bool(re.match(r"^[\d\s\-\+\(\)]+$", text))
    looks_like_url = "http://" in text.lower() or "www." in text.lower()

    return (
        upper_only
        or looks_like_place
        or looks_like_date
        or looks_like_phone
        or looks_like_url
    )


def _keyword_role(text: str) -> BlockRole | None:
    """Return the canonical paper-section role for a keyword line, else None."""
    cleaned = text.strip().rstrip(":").strip().lower()
    return _PAPER_SECTION_KEYWORDS.get(cleaned)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class HeuristicClassifier:
    """Regex / keyword based classifier used as the Wave 12 default.

    Stateless across ``classify`` calls. Every emitted ``ClassifiedBlock``
    carries ``classifier_source="heuristic"``. Confidence values are
    rough qualitative markers:

        0.95 - canonical keyword or anchored regex match
        0.80 - numbered heading / structured match with strong guardrails
        0.70 - subheading candidate passing guardrails
        0.60 - metadata line (arxiv, copyright, email) via keyword hint
        0.50 - fallback paragraph
        0.30 - column-layout or low-signal residue
    """

    async def classify(self, blocks: List[RawBlock]) -> List[ClassifiedBlock]:
        """Classify ``blocks`` into ``ClassifiedBlock`` instances.

        The method is ``async`` for symmetry with ``LLMClassifier`` (Wave 14);
        the heuristic logic itself is pure-Python and does not await. Callers
        that cannot run an event loop may call the sync helper
        :meth:`classify_sync` instead.
        """
        return self.classify_sync(blocks)

    def classify_sync(self, blocks: List[RawBlock]) -> List[ClassifiedBlock]:
        """Synchronous convenience wrapper around :meth:`classify`."""
        results: List[ClassifiedBlock] = []
        for block in blocks:
            results.append(self._classify_one(block))
        logger.debug("Heuristic classifier produced %d decisions", len(results))
        return results

    # -- Internals --------------------------------------------------------

    def _classify_one(self, block: RawBlock) -> ClassifiedBlock:
        # Wave 16: when the segmenter attached an ``extractor_hint``
        # (structured extraction like pdfplumber tables / PyMuPDF
        # figures), trust the hint at full confidence and forward the
        # ``extra`` payload straight through to the template layer —
        # text classification would only mislabel tables as prose.
        if block.extractor_hint is not None:
            return self._make(
                block,
                block.extractor_hint,
                1.0,
                attributes=dict(block.extra or {}),
            )

        text = block.text.strip()

        if not text:
            return self._make(block, BlockRole.PARAGRAPH, 0.30)

        # TOC / page-break style lines come first: these must not
        # bleed into headings even though they can look title-cased.
        if _TOC_DOT_LEADER.match(text) or _TOC_ENTRY.match(text):
            return self._make(block, BlockRole.TOC_NAV, 0.85)

        if _TOC_ROMAN_PAGE.match(text) and len(text) <= 8:
            return self._make(block, BlockRole.PAGE_BREAK, 0.80)

        # Canonical keyword headings (Abstract, Introduction, Keywords...).
        kw_role = _keyword_role(text)
        if kw_role is not None:
            return self._make(block, kw_role, 0.95)

        # Front-matter / licensing hints before structural heading
        # matches so a "Copyright 2024 Foo" line doesn't promote via
        # chapter regex.
        if _FRONT_MATTER_HINT.search(text):
            return self._make(block, BlockRole.COPYRIGHT_LICENSE, 0.80)

        # arxiv meta lines + standalone email affiliations.
        if _ARXIV_META_HINT.search(text):
            return self._make(block, BlockRole.AUTHOR_AFFILIATION, 0.85)

        if _EMAIL_HINT.search(text) and len(text) < 200:
            return self._make(block, BlockRole.AUTHOR_AFFILIATION, 0.60)

        # "Chapter 3: Foo", "II. Bar", "11. Behaviorism..."
        chapter_match = _CHAPTER_HEADING.match(text)
        if chapter_match and len(text) < 160:
            heading_text = chapter_match.group(1).strip()
            return self._make(
                block,
                BlockRole.CHAPTER_OPENER,
                0.90,
                attributes={"heading_text": heading_text},
            )

        # Numbered paper section ("1 Introduction", "2.1 Related Work").
        paper_match = _PAPER_SECTION_NUMBERED.match(text)
        if paper_match:
            heading_text = paper_match.group(1).strip()
            level = text.split()[0].count(".") + 1
            role = (
                BlockRole.SECTION_HEADING
                if level == 1
                else BlockRole.SUBSECTION_HEADING
            )
            return self._make(
                block,
                role,
                0.85,
                attributes={"heading_text": heading_text, "level": level},
            )

        # Bibliography entry (author-year or bracket-numbered).
        if _BIBLIOGRAPHY_ENTRY.match(text) and len(text) < 500:
            return self._make(block, BlockRole.BIBLIOGRAPHY_ENTRY, 0.80)

        # Footnote marker (small prefix + short body).
        if _FOOTNOTE.match(text) and len(text) < 400:
            return self._make(block, BlockRole.FOOTNOTE, 0.70)

        # Title-case short line subheading fallback, after low-signal filter.
        if (
            _SUB_HEADING.match(text)
            and _is_valid_subheading(text)
            and not _is_low_signal_heading(text)
        ):
            return self._make(block, BlockRole.SUBSECTION_HEADING, 0.70)

        # Column-layout residue: downgrade to low-confidence paragraph so
        # the assembler can skip it in a future wave if desired.
        if _COLUMN_LAYOUT.search(text) and len(text.split()) < 25:
            return self._make(block, BlockRole.PARAGRAPH, 0.30)

        # Default: paragraph.
        return self._make(block, BlockRole.PARAGRAPH, 0.50)

    @staticmethod
    def _make(
        block: RawBlock,
        role: BlockRole,
        confidence: float,
        attributes: dict | None = None,
    ) -> ClassifiedBlock:
        return ClassifiedBlock(
            raw=block,
            role=role,
            confidence=confidence,
            attributes=attributes or {},
            classifier_source="heuristic",
        )


__all__ = ["HeuristicClassifier"]
