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
from typing import List, Optional

from DART.converter.block_roles import BlockRole, ClassifiedBlock, RawBlock

logger = logging.getLogger(__name__)

# Wave 18: threshold multiplier for font-size-based heading promotion.
# A block whose dominant span renders at ``_FONT_SIZE_HEADING_RATIO * median``
# or more is a heading candidate. The value is deliberately generous — we
# only *promote* paragraph-classified blocks, never demote explicit roles.
_FONT_SIZE_HEADING_RATIO = 1.5

# Bolded runs at ratios between ``_BOLD_HEADING_RATIO`` and
# ``_FONT_SIZE_HEADING_RATIO`` get demoted to the weaker ``SUBSECTION_HEADING``
# role — bold is a secondary tiebreaker and only fires when the font size
# is already somewhat larger than body text.
_BOLD_HEADING_RATIO = 1.15

# Larger-still ratios ("display size") promote to the stronger
# ``SECTION_HEADING`` role.
_SECTION_HEADING_RATIO = 1.9


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

def _match_block_to_spans(block, spans):
    """Return the dominant :class:`ExtractedTextSpan` overlapping ``block``.

    Wave 18 helper used by the heuristic classifier's font-size-based
    heading promoter. The "dominant" span is the one whose ``.text``
    shares the most characters with ``block.text`` AND sits on the same
    page (when the block carries a page number).

    Returns ``None`` when no span meets a minimum overlap threshold so
    the promoter can no-op and leave the regex-based classification
    untouched. This is deliberately conservative — we'd rather miss a
    heading than promote a paragraph.
    """
    if not spans or not block.text:
        return None

    block_text = block.text.strip()
    if not block_text:
        return None

    block_page = block.page

    best_span = None
    best_overlap = 0
    for span in spans:
        span_text = (span.text or "").strip()
        if not span_text:
            continue
        # Page match when we know the block's page — avoids accidentally
        # matching a same-wording heading from a different page.
        if block_page is not None and span.page != block_page:
            continue
        # Cheap overlap: is the span's text a prefix of the block text?
        # For single-line headings the whole span IS the block; we want
        # short span-length / overlap tolerance so a 40-char heading
        # that appears verbatim as the first run wins.
        if block_text.startswith(span_text) or span_text.startswith(block_text):
            overlap = min(len(span_text), len(block_text))
        elif span_text in block_text:
            overlap = len(span_text)
        else:
            continue

        if overlap > best_overlap:
            best_overlap = overlap
            best_span = span

    # Require the matched span to cover at least half the block text so
    # a stray one-word span embedded mid-paragraph doesn't hijack the
    # font-size signal.
    if best_span is None or best_overlap < max(4, len(block_text) // 2):
        return None
    return best_span


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

    Wave 18 additions:

    * ``text_spans`` — optional list of
      :class:`DART.converter.extractor.ExtractedTextSpan` carrying
      font-size + bbox metadata. When provided alongside
      ``median_body_font_size``, the classifier promotes
      regex-classified ``PARAGRAPH`` blocks whose dominant span
      renders at >= 1.5x median body font to ``SECTION_HEADING`` /
      ``SUBSECTION_HEADING`` (display ratios vs. merely-larger ratios).
      Promotions never override explicit regex matches — bold /
      large-font signals only affect fallback paragraphs.
    * ``median_body_font_size`` — optional float. Callers can compute
      it via :func:`DART.converter.extractor.median_body_font_size`
      and pass it through so the classifier doesn't need to re-scan.
      When ``None`` and ``text_spans`` is populated, the classifier
      computes the median lazily on its own.
    """

    def __init__(
        self,
        *,
        text_spans: Optional[list] = None,
        median_body_font_size: Optional[float] = None,
    ) -> None:
        self._text_spans = list(text_spans) if text_spans else []
        if median_body_font_size is not None:
            self._median_body_font_size = float(median_body_font_size)
        elif self._text_spans:
            # Lazy median fallback — import here to avoid a cycle at
            # module load (extractor imports other converter modules).
            from DART.converter.extractor import median_body_font_size as _median

            self._median_body_font_size = _median(self._text_spans)
        else:
            self._median_body_font_size = None

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
        results = [self._maybe_promote_by_font_size(r) for r in results]
        logger.debug("Heuristic classifier produced %d decisions", len(results))
        return results

    # ------------------------------------------------------------------
    # Wave 18 font-size heading promoter
    # ------------------------------------------------------------------

    def _maybe_promote_by_font_size(self, classified: ClassifiedBlock) -> ClassifiedBlock:
        """Promote paragraph-classified blocks that render at heading-sized font.

        Rules:

        * No spans or no median available → no-op.
        * Block already has an explicit heading role → no-op (never demote).
        * Dominant-span font ratio >= ``_SECTION_HEADING_RATIO`` →
          promote to ``SECTION_HEADING``.
        * Dominant-span font ratio >= ``_FONT_SIZE_HEADING_RATIO`` →
          promote to ``SUBSECTION_HEADING``.
        * Dominant-span font ratio >= ``_BOLD_HEADING_RATIO`` AND the
          span is bold → promote to ``SUBSECTION_HEADING``.
        """
        if not self._text_spans or not self._median_body_font_size:
            return classified
        # Only act on the fallback paragraph role. Anything else is the
        # regex classifier asserting a stronger role we shouldn't undo.
        if classified.role != BlockRole.PARAGRAPH:
            return classified

        span = _match_block_to_spans(classified.raw, self._text_spans)
        if span is None or span.font_size <= 0:
            return classified

        ratio = span.font_size / self._median_body_font_size
        attrs = dict(classified.attributes or {})
        attrs["heading_text"] = classified.raw.text
        attrs["font_size"] = span.font_size
        attrs["font_size_ratio"] = round(ratio, 2)

        if ratio >= _SECTION_HEADING_RATIO:
            return ClassifiedBlock(
                raw=classified.raw,
                role=BlockRole.SECTION_HEADING,
                confidence=0.75,
                attributes=attrs,
                classifier_source="heuristic",
            )
        if ratio >= _FONT_SIZE_HEADING_RATIO:
            return ClassifiedBlock(
                raw=classified.raw,
                role=BlockRole.SUBSECTION_HEADING,
                confidence=0.70,
                attributes=attrs,
                classifier_source="heuristic",
            )
        if span.is_bold and ratio >= _BOLD_HEADING_RATIO:
            return ClassifiedBlock(
                raw=classified.raw,
                role=BlockRole.SUBSECTION_HEADING,
                confidence=0.65,
                attributes=attrs,
                classifier_source="heuristic",
            )
        return classified

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
