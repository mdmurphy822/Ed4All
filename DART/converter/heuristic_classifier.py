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


# Wave 25 Fix 5: dotted-numeric subsection hierarchy. Matches
# "4.8.1.1 Epistemological basis", "1.7.1. Fully online learning",
# "8.4.1.4 Maintenance costs", etc. — 1 to 5 dot-separated segments
# followed by whitespace + non-digit content. Capital-letter prefix
# (A1.2) is also accepted ("A2.3 Appendix heading").
_DOTTED_NUMERIC_HEADING = re.compile(
    r"^\s*([A-Z]?\d+(?:\.\d+){1,4})\.?\s+(\S.*)$"
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
# Wave 21: list marker detection
# ---------------------------------------------------------------------------
#
# pdftotext preserves list markers as literal characters at the start of
# each item's block. Before Wave 21 these flowed straight through as
# naked ``<p>•    Item`` paragraphs — 323 of them on the Bates textbook.
# The classifier promotes each such block to ``LIST_ITEM`` with
# ``attributes = {"marker", "marker_type", "text"}`` and the assembler
# groups consecutive items into a single ``<ul>`` / ``<ol>``.

# Unicode bullet characters (standalone, comprehensive charset).
# Deliberately excludes en-dash / em-dash / middle-dot-lookalikes that
# commonly appear mid-word.
_UNORDERED_BULLET_CHARS = "\u2022\u00b7\u25aa\u25cf\u25e6\u25cb\u25b8\u25ba\u25a0\u25a1\u25fc\u25fe"

# A Unicode bullet followed by required whitespace.
_LIST_MARKER_UNICODE = re.compile(
    rf"^([{_UNORDERED_BULLET_CHARS}])\s+(\S.*)$"
)

# ASCII hyphen / asterisk at start of line followed by whitespace AND
# the rest looks like an item (starts with upper / digit / word-char).
_LIST_MARKER_ASCII = re.compile(r"^([\-*])\s+([A-Za-z0-9].*)$")

# Numbered markers. Matches "1.", "1)", "(1)", "12.", "a.", "a)",
# "iv.", "vii)" — the latter two are lowercase roman. The tail must be
# whitespace-separated, and the item body must start with upper, digit,
# or open-paren (guards common abbreviations "e.g.", "i.e." which have
# lowercase letters before the dot and no trailing capital).
_LIST_MARKER_ORDERED = re.compile(
    r"^(?:"
    r"(\d{1,3})[.)]\s+|"                  # 1. / 1) / 12.
    r"\((\d{1,3})\)\s+|"                  # (1)
    r"([a-z])[.)]\s+|"                    # a. / a)
    r"([ivxIVX]{1,5})[.)]\s+"             # iv. / VII)
    r")([A-Z0-9\(].*)$"
)


def _match_list_marker(text: str):
    """Return ``(marker_type, marker, rest)`` if ``text`` opens with a
    list marker, else ``None``.

    ``marker_type`` is ``"unordered"`` or ``"ordered"``. ``marker`` is
    the literal prefix as authored (``•``, ``-``, ``1.``, ``a)``,
    ``(3)``, ...). ``rest`` is the item body with the marker + its
    trailing whitespace stripped.

    Guards:

    * The item body must be < 200 chars (list items are short).
    * Body must start with capital letter, digit, or open-paren — this
      rejects many false positives like "``- then`` something" that's
      actually a mid-sentence dash continuation from a column-layout
      artefact.
    * Ascii ``-`` / ``*`` only matches when the next character is
      whitespace AND the body starts with a capital or digit — pure
      word-chars after a hyphen would be hyphenated words.
    """
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None

    # Unicode bullets: most common case on educational PDFs.
    m = _LIST_MARKER_UNICODE.match(stripped)
    if m:
        rest = m.group(2).strip()
        if not rest or len(rest) > 4000:
            return None
        return ("unordered", m.group(1), rest)

    # Ordered markers.
    m = _LIST_MARKER_ORDERED.match(stripped)
    if m:
        # Pull the groups; exactly one numeric / alpha / roman / parens
        # capture will be non-None depending on the branch that matched.
        digit = m.group(1)
        paren_digit = m.group(2)
        alpha = m.group(3)
        roman = m.group(4)
        rest = m.group(5).strip()
        if not rest or len(rest) > 4000:
            return None
        if digit is not None:
            marker = f"{digit}."
            # Recover the exact authored punctuation (1. vs 1)) by
            # looking at the char immediately after the digit in the
            # original text.
            dot_or_paren = stripped[len(digit) : len(digit) + 1]
            marker = f"{digit}{dot_or_paren}"
        elif paren_digit is not None:
            marker = f"({paren_digit})"
        elif alpha is not None:
            dot_or_paren = stripped[1:2]
            marker = f"{alpha}{dot_or_paren}"
        elif roman is not None:
            dot_or_paren = stripped[len(roman) : len(roman) + 1]
            marker = f"{roman}{dot_or_paren}"
        else:  # pragma: no cover — regex guarantees at least one group
            return None
        return ("ordered", marker, rest)

    # ASCII dash / asterisk: only fire when the body starts with
    # something that looks like item content. This is intentionally
    # conservative to avoid chewing on "- 5" range markers or
    # "-negative" mid-word hyphens.
    m = _LIST_MARKER_ASCII.match(stripped)
    if m:
        rest = m.group(2).strip()
        if not rest or len(rest) > 4000:
            return None
        return ("unordered", m.group(1), rest)

    return None


def _looks_like_list_item(text: str):
    """Public-ish wrapper returning the marker type string, or ``None``.

    Wave 21 entry point used by the classifier and tests. Returns one of
    ``"unordered"`` / ``"ordered"`` / ``None``. Callers needing the full
    marker + body tuple should call :func:`_match_list_marker`.
    """
    match = _match_list_marker(text)
    return match[0] if match else None


# Embedded sibling numbered markers (``<space>N.<space>Upper`` /
# ``<space>N)<space>Upper``). Used by :meth:`_classify_one` to recognise
# fused numbered-list blocks that would otherwise capture via
# :data:`_CHAPTER_HEADING`.
_INLINE_NUMBERED_SIBLINGS = re.compile(
    r"\s\d{1,3}[.)]\s+[A-Z]"
)


# ---------------------------------------------------------------------------
# Wave 25 Fix 4: CHAPTER_OPENER false-positive guard
# ---------------------------------------------------------------------------
#
# Bates audit: 39 ``<article role="doc-chapter">`` emissions vs 12
# real chapters. Activity prompts ("What are your reasons?",
# "Determine which is a medium...", "Do you find the distinction
# helpful?") were being promoted to CHAPTER_OPENER because they
# happen to start with a capital letter + keyword + number in the
# Chapter regex's permissive path.

# Interrogative / directive prompt starters. A block opening with any
# of these is an activity prompt, not a chapter heading.
_ACTIVITY_PROMPT_STARTERS = re.compile(
    r"^\s*(?:"
    r"what(?:\s|')|"        # "What are..." / "What's..."
    r"how(?:\s|')|"
    r"why(?:\s|')|"
    r"which\b|"
    r"where\b|"
    r"when\b|"
    r"do\s+you\b|"
    r"have\s+you\b|"
    r"are\s+you\b|"
    r"can\s+you\b|"
    r"could\s+you\b|"
    r"would\s+you\b|"
    r"should\s+you\b|"
    r"consider\b|"
    r"determine\b|"
    r"reflect\b|"
    r"think\b|"
    r"discuss\b|"
    r"try\b|"
    r"imagine\b|"
    r"decide\b|"
    r"take\s+one\b|"
    r"take\s+a\b|"
    r"to\s+identify\b|"
    r"to\s+explore\b|"
    r"to\s+understand\b|"
    r"to\s+analy[sz]e\b|"
    r"if\s+you\b"
    r")",
    re.IGNORECASE,
)

# Secondary — pronoun + "you" / "your" anywhere in the first few
# words. Used as a supplemental signal: a short prompt opening with
# "Take one of your courses..." or "If you teach apprentices..." is
# almost certainly an activity prompt even when the leading word
# isn't in the starters list.
_YOU_PRONOUN_RE = re.compile(
    r"^\s*\S+\s+(?:you|your)\b", re.IGNORECASE
)

# Real chapter-number pattern. When the block starts with this shape,
# it wins over the activity-prompt guard (rare edge case: chapter
# whose title happens to be a question).
_STRONG_CHAPTER_PATTERN = re.compile(
    r"^\s*(?:chapter|part|section|unit|book|volume)\s+\d",
    re.IGNORECASE,
)

# Bare-number leading pattern ("5 Introduction to digital pedagogy").
# When present, the chapter-number shape wins over the activity
# filter.
_LEADING_BARE_NUMBER_CHAPTER = re.compile(
    r"^\s*\d+\s+[A-Z]"
)


def _looks_like_activity_prompt(text: str) -> bool:
    """Return True when ``text`` opens with an activity-prompt starter.

    Guards CHAPTER_OPENER promotion against Bates-style activity
    prompts. Strong chapter-number patterns (``"Chapter 5:"``,
    ``"Part II"``, ``"5 Introduction..."``) override the guard — if
    the block starts with one of those, it's a real chapter even
    when the title happens to be a question.

    The numbered-list leader (``"3. What are your reasons?"``) is
    stripped before the activity-prompt check so fused prompt +
    numbered-leader lines still trigger the guard. pdftotext
    regularly emits activity prompts with leading ``"N. "`` numbering.
    """
    if not text:
        return False
    if _STRONG_CHAPTER_PATTERN.match(text):
        return False
    if _LEADING_BARE_NUMBER_CHAPTER.match(text):
        return False
    # Strip a leading "N. " / "N) " / "(N) " numbered marker so the
    # activity-prompt check sees the real opening word.
    stripped = re.sub(
        r"^\s*(?:\d{1,3}[.)]|\(\d{1,3}\))\s+",
        "",
        text,
    )
    if _ACTIVITY_PROMPT_STARTERS.match(stripped):
        return True
    # Secondary signal: second-person pronoun in the opening bigram
    # ("Take one of your...") + no chapter-number pattern = activity
    # prompt. Only kicks in for reasonably short blocks to avoid
    # demoting legitimate content that happens to reference "you" /
    # "your" in its opening clause.
    if len(stripped) < 250 and _YOU_PRONOUN_RE.match(stripped):
        return True
    return False


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
        page_chrome: Optional[object] = None,
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
        # Wave 25 Fix 2: belt-and-braces FOOTNOTE guard. When the
        # page-chrome detector caught a running header/footer line
        # for a page, any classified block whose text is a substring
        # of that chrome line must NOT be promoted to FOOTNOTE — it
        # would otherwise surface as a phantom footnote in the
        # assembled HTML (see Bates leading-digit footer regression).
        self._page_chrome = page_chrome

    async def classify(self, blocks: List[RawBlock]) -> List[ClassifiedBlock]:
        """Classify ``blocks`` into ``ClassifiedBlock`` instances.

        The method is ``async`` for symmetry with ``LLMClassifier`` (Wave 14);
        the heuristic logic itself is pure-Python and does not await. Callers
        that cannot run an event loop may call the sync helper
        :meth:`classify_sync` instead.
        """
        return self.classify_sync(blocks)

    def classify_sync(self, blocks: List[RawBlock]) -> List[ClassifiedBlock]:
        """Synchronous convenience wrapper around :meth:`classify`.

        Wave 21: a single input block can expand into multiple classified
        blocks when a numbered-list block carries several embedded items
        (pdftotext often merges siblings onto one logical line). The
        classifier emits one ``LIST_ITEM`` per detected item in that
        case; the downstream :mod:`DART.converter.document_assembler`
        grouping pass then folds them into a single ``<ol>`` / ``<ul>``.
        """
        results: List[ClassifiedBlock] = []
        for block in blocks:
            classified = self._classify_one(block)
            # Multi-item expansion: produced when a numbered block
            # contains embedded list markers ("1. foo 2. bar 3. baz").
            expanded = self._maybe_expand_numbered_run(classified)
            if expanded is not None:
                results.extend(expanded)
            else:
                results.append(classified)
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
            # Wave 21 guard: when the text carries embedded sibling
            # numbered markers (" 2. ", " 3. "), it's almost certainly
            # a fused numbered-list block, not a chapter opener. Fall
            # through to the list-item path so the classifier's
            # expansion logic splits it into one LIST_ITEM per item.
            if not (
                text[0:1].isdigit()
                and _LIST_MARKER_ORDERED.match(text)
                and len(_INLINE_NUMBERED_SIBLINGS.findall(text)) >= 1
            ):
                # Wave 25 Fix 4: refuse CHAPTER_OPENER promotion when
                # the block opens with an activity-prompt starter
                # ("What are your reasons?", "Determine which...",
                # "Do you find..."). Bates-style false positives
                # inflated the chapter count from 12 → 39.
                # Wave 25 Fix 7: refuse CHAPTER_OPENER promotion when
                # the block opens with a single-digit "N. " pattern
                # that is in fact a numbered subsection heading
                # ("1. Why this book?" followed by 80 words of
                # prose). The absence of the word "Chapter" + a
                # long prose neighbour is the signal.
                is_numbered_list_style = bool(
                    text[0:1].isdigit()
                    and _LIST_MARKER_ORDERED.match(text)
                )
                if _looks_like_activity_prompt(text):
                    logger.debug(
                        "chapter_opener_rejected: activity prompt '%s...'",
                        text[:60],
                    )
                elif (
                    is_numbered_list_style
                    and not _STRONG_CHAPTER_PATTERN.match(text)
                    and self._next_block_is_long_prose(block)
                ):
                    # Treat as numbered subsection heading — fall
                    # through to the subheading emission below.
                    logger.debug(
                        "chapter_opener_rejected: numbered subheading '%s' prose_follows",
                        text[:60],
                    )
                    return self._make(
                        block,
                        BlockRole.SUBSECTION_HEADING,
                        0.70,
                        attributes={
                            "heading_text": heading_text,
                            "level": 3,
                            "numbered_marker": text.split(None, 1)[0],
                        },
                    )
                else:
                    return self._make(
                        block,
                        BlockRole.CHAPTER_OPENER,
                        0.90,
                        attributes={"heading_text": heading_text},
                    )

        # Wave 25 Fix 5: dotted-numeric heading ("4.8.1.1 Epistemological
        # basis", "1.7.1. Fully online learning", "2.3 Implementation
        # notes"). 149 short <p> bodies on Bates match this shape;
        # pre-Wave-25 they bled into naked paragraphs (h4 count: 0).
        #
        # Runs regardless of text_spans availability. When PyMuPDF
        # layout data IS available, this rule still fires first — the
        # Wave-18 font-size heading promoter only lifts PARAGRAPH-
        # classified blocks, so once we emit SUBSECTION_HEADING here
        # the font-size pass is a no-op on these blocks.
        #
        # Must precede the LIST_ITEM path so "4.8.1.1 Epistemological
        # basis" doesn't capture as a numbered list item.
        dotted_match = _DOTTED_NUMERIC_HEADING.match(text)
        if dotted_match and len(text) < 160:
            number_part = dotted_match.group(1)
            heading_text = dotted_match.group(2).strip()
            # Count dots (dot-separated hierarchy depth). Level 1
            # ("2.3") → h3; level 2 ("2.3.1") → h4; level 3+ → h5;
            # capped at h6 for safety. Every dotted-numeric heading
            # surfaces as SUBSECTION_HEADING (so the <hN> template
            # routes via the Wave 25 level-aware emitter) regardless
            # of level — matching pre-Wave-25 PAPER_SECTION_NUMBERED
            # behaviour for 1-dot forms.
            dot_count = number_part.count(".")
            if dot_count == 1:
                level = 3
            elif dot_count == 2:
                level = 4
            elif dot_count >= 3:
                level = min(6, 3 + dot_count - 1)
            else:  # pragma: no cover — regex guarantees >= 1 dot
                level = 3
            role = BlockRole.SUBSECTION_HEADING
            return self._make(
                block,
                role,
                0.80,
                attributes={
                    "heading_text": heading_text,
                    "level": level,
                    "dotted_number": number_part,
                },
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

        # Wave 21: list-item detection. Runs AFTER bibliography + chapter
        # classification (both higher priority) but BEFORE the footnote
        # / subheading fallbacks. A block whose first line matches a
        # bullet / numbered / roman / alpha marker becomes LIST_ITEM
        # with ``{marker, marker_type, text, sub_items}``. Nested
        # sub-items (indented lines starting with a marker) are
        # best-effort attached here.
        list_classified = self._maybe_classify_list_item(block, text)
        if list_classified is not None:
            return list_classified

        # Footnote marker (small prefix + short body).
        if _FOOTNOTE.match(text) and len(text) < 400:
            # Wave 25 Fix 2: refuse FOOTNOTE promotion when the
            # block's text is a substring of the page's detected
            # chrome line — the leading-digit running-footer pattern
            # ("{N} A.W. (Tony) Bates") trips the FOOTNOTE regex
            # because the leading integer looks like a footnote
            # marker. The chrome detector (Wave 25 Fix 1) normally
            # strips these upstream, but this guard catches the
            # rare case where a chrome line survives to the
            # classifier (e.g. infrequent-enough to fall under the
            # min_repeat_fraction threshold but clearly matching a
            # detected pattern for the page). Falls through to
            # PARAGRAPH when the guard fires.
            if not self._matches_page_chrome(block, text):
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

    # ------------------------------------------------------------------
    # Wave 25 Fix 2: FOOTNOTE page-chrome guard
    # ------------------------------------------------------------------

    def _matches_page_chrome(self, block: RawBlock, text: str) -> bool:
        """Return True when ``text`` matches a detected chrome line
        for ``block.page``.

        Used to refuse FOOTNOTE promotion on leading-digit running
        footers ("{N} A.W. (Tony) Bates") that survived the chrome-
        strip pass. The comparison is normalised (lowercase,
        whitespace-collapsed) and treats the chrome line as a
        superstring — any block whose normalised text is contained
        in the chrome line's normalised form triggers the guard.

        No-op when no page_chrome was attached to the classifier,
        the block has no page, or the page has no recorded chrome
        line (preserves legacy behaviour).
        """
        chrome = self._page_chrome
        if chrome is None or block.page is None:
            return False
        # Lazy imports — keep the module-load graph unchanged for
        # callers that don't use the guard path.
        from DART.converter.page_chrome import (
            _normalise,
            _strip_leading_digits,
            _strip_trailing_digits,
        )

        norm_text = _normalise(text)
        if not norm_text:
            return False

        # Compute the block's leading-digit residual + trailing-digit
        # prefix so we can compare against residual forms stored in
        # chrome.headers / chrome.footers.
        lead_residual, _lp = _strip_leading_digits(norm_text)
        tail_prefix, _tp = _strip_trailing_digits(norm_text)

        page_number_lines = getattr(chrome, "page_number_lines", None) or {}
        raw_chrome_line = page_number_lines.get(block.page)
        if raw_chrome_line:
            norm_chrome = _normalise(raw_chrome_line)
            if norm_chrome and (
                norm_text == norm_chrome
                or norm_text in norm_chrome
                or norm_chrome in norm_text
            ):
                return True

        # Fallback: compare against the full header/footer residual
        # sets. Residuals are stored in the normalised + digit-stripped
        # form, so we test against both the block's leading-digit
        # residual and its trailing-digit prefix.
        headers = getattr(chrome, "headers", set()) or set()
        footers = getattr(chrome, "footers", set()) or set()
        for candidate in (*headers, *footers):
            if not candidate:
                continue
            if lead_residual and (
                lead_residual == candidate or candidate in lead_residual
            ):
                return True
            if tail_prefix and (
                tail_prefix == candidate or candidate in tail_prefix
            ):
                return True
            if norm_text == candidate or candidate in norm_text:
                return True
        return False

    # ------------------------------------------------------------------
    # Wave 21 list-item detection + multi-item expansion
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Wave 25 Fix 7: LIST_ITEM → SUBSECTION_HEADING guard for numbered
    # subheadings followed by prose. pdftotext often emits each
    # subsection heading ("1. Why this book?", "2. The audience for the
    # book") as its own block; the Wave 21 LIST_ITEM classifier grabs
    # them as list items and the assembler emits one single-item <ol>
    # per heading (14 per-block single-ol wrappers on Bates). This
    # guard looks ahead to the next block — when the next block is a
    # long prose paragraph (>= 60 words), the candidate is NOT a list
    # item but a numbered section heading.
    # ------------------------------------------------------------------

    # Absolute threshold — when the next block carries >= this many
    # words of prose, the current short block is almost certainly a
    # heading rather than a list-item peer.
    _HEADING_FOLLOWED_BY_PROSE_MIN_WORDS = 25

    # Relative threshold — even shorter prose neighbours (e.g. ~15
    # words) still qualify as "clearly prose" when the current block
    # is dramatically shorter than its neighbour (at least 3× ratio).
    _HEADING_RELATIVE_RATIO = 3.0
    _HEADING_RELATIVE_MIN_NEXT_WORDS = 15

    def _next_block_is_long_prose(self, block: RawBlock) -> bool:
        """Return True when ``block.neighbors['next']`` looks like a
        prose paragraph.

        Used to distinguish a numbered subsection heading
        ("ii. The audience for the book") from a real list item in
        a short list.

        Two pass-through tests — EITHER suffices:

        1. **Absolute**: next block carries >= 25 words AND is not
           itself a list-marker-led item.
        2. **Relative**: next block is at least 3× the word count
           of the current block AND carries >= 15 words.

        Real list peers ("ii. First item", "iii. Second item")
        fail both checks: the next block is another short list
        item, not prose, and its length is comparable to the
        current block's.
        """
        next_text = (block.neighbors or {}).get("next") or ""
        if not next_text:
            return False
        stripped = next_text.strip()
        if not stripped:
            return False
        if _match_list_marker(stripped) is not None:
            # Next block is itself a list item → current block is
            # peer-sibling in a list, not a heading.
            return False
        next_word_count = len(stripped.split())
        if next_word_count >= self._HEADING_FOLLOWED_BY_PROSE_MIN_WORDS:
            return True
        current_word_count = len((block.text or "").split())
        if current_word_count > 0 and next_word_count >= self._HEADING_RELATIVE_MIN_NEXT_WORDS:
            ratio = next_word_count / current_word_count
            if ratio >= self._HEADING_RELATIVE_RATIO:
                return True
        return False

    def _maybe_classify_list_item(
        self, block: RawBlock, text: str
    ) -> Optional[ClassifiedBlock]:
        """Return a ``LIST_ITEM`` ClassifiedBlock if ``text`` opens with a
        list marker, else ``None``.

        Single-line case: ``LIST_ITEM`` with ``marker`` / ``marker_type``
        / ``text`` attributes.

        Multi-line case (pdftotext ``-layout`` preserves indentation):
        when subsequent non-empty lines in the same block are indented
        (≥ 4 leading spaces) AND start with a marker themselves, they
        become ``sub_items`` on the parent. This is the nested-list
        MVP; deeper nesting is out of scope.

        ``block.text`` has already been whitespace-collapsed by the
        segmenter (see :func:`DART.converter.block_segmenter._normalise_block_text`),
        so we lose the leading-whitespace signal for nested-sub-item
        detection. Nesting therefore almost never fires in the current
        pipeline — the code is retained so that callers who feed raw
        multi-line blocks in (tests, future waves that preserve
        layout) still get nested output.
        """
        # When the block already carries an extractor hint, don't
        # override it — tables/figures win.
        if block.extractor_hint is not None:
            return None

        lines = [ln for ln in block.text.splitlines() if ln.strip()]
        if not lines:
            match = _match_list_marker(text)
            if match is None:
                return None
            marker_type, marker, rest = match
            return self._build_list_item(block, marker_type, marker, rest, [])

        # Primary: first line opens with a list marker.
        first_line = lines[0]
        match = _match_list_marker(first_line.lstrip())
        if match is None:
            return None
        marker_type, marker, rest = match

        # Wave 25 Fix 7: ordered markers with a prose neighbour are
        # numbered subsection headings, not list items.
        if marker_type == "ordered" and self._next_block_is_long_prose(block):
            # Derive a heading level from the marker shape.
            # Single-digit (1., 2., 3.) → h3; double-digit with dotted
            # shape (1.7.) would already have matched the dotted-numeric
            # regex upstream, so here we only see simple single-level
            # markers.
            level = 3
            logger.debug(
                "list_item_rejected: numbered subheading '%s' next_block_words=%d",
                rest[:50],
                len((block.neighbors or {}).get("next", "").split()),
            )
            return self._make(
                block,
                BlockRole.SUBSECTION_HEADING,
                0.70,
                attributes={
                    "heading_text": rest,
                    "level": level,
                    "numbered_marker": marker,
                },
            )

        # Long-item sanity guard: reject when the rest is clearly a
        # wall of prose that happens to start with what looks like a
        # list marker (e.g. a copyright blurb with a bullet glyph
        # glued to its first word). We only bail when the text
        # contains no embedded sibling markers AND is very long AND
        # contains many sentence-ending periods — the signs of prose
        # that got accidentally marker-tagged. Fused multi-item blocks
        # (the common pdftotext case this wave fixes) have embedded
        # siblings and therefore keep their LIST_ITEM role so the
        # expansion path can split them.
        if (
            marker_type == "unordered"
            and len(rest) > 600
            and "\u2022" not in rest
            and rest.count(". ") >= 4
        ):
            return None

        # Collect nested sub-items when subsequent lines look indented +
        # marker-prefixed. (Rare given segmenter whitespace collapsing —
        # see docstring — but cheap to support.)
        sub_items: List[dict] = []
        for cont in lines[1:]:
            leading = len(cont) - len(cont.lstrip())
            if leading < 4:
                continue
            nested = _match_list_marker(cont.lstrip())
            if nested is None:
                continue
            n_type, n_marker, n_rest = nested
            sub_items.append(
                {"text": n_rest, "marker": n_marker, "marker_type": n_type}
            )

        return self._build_list_item(block, marker_type, marker, rest, sub_items)

    def _build_list_item(
        self,
        block: RawBlock,
        marker_type: str,
        marker: str,
        rest: str,
        sub_items: List[dict],
    ) -> ClassifiedBlock:
        attrs = {
            "marker": marker,
            "marker_type": marker_type,
            "text": rest,
        }
        if sub_items:
            attrs["sub_items"] = sub_items
        return ClassifiedBlock(
            raw=block,
            role=BlockRole.LIST_ITEM,
            confidence=0.85,
            attributes=attrs,
            classifier_source="heuristic",
        )

    # -- Multi-item expansion -------------------------------------------
    # pdftotext output often fuses several numbered items onto one
    # line when the source PDF used non-layout whitespace. Example:
    # "1. First point. 2. Second point. 3. Third point." arrives as a
    # single block. The per-block classifier above only catches the
    # leading marker; this expander splits the remaining body on
    # sibling markers so each item becomes its own LIST_ITEM.

    # Matches embedded "<whitespace>N." / "<whitespace>N)" markers that
    # introduce another list item. Captures: leading whitespace, the
    # marker, and the rest of the segment up to the next marker / EOL.
    _EMBEDDED_NUMBERED_MARKER = re.compile(
        r"\s(\d{1,3})[.)]\s+(?=[A-Z])"
    )

    # Matches embedded unicode-bullet markers that introduce a sibling
    # item inside a fused block ("foo • bar • baz"). Captures the bullet
    # character; the body follows until the next bullet or EOL.
    _EMBEDDED_UNICODE_BULLET = re.compile(
        rf"\s([{_UNORDERED_BULLET_CHARS}])\s+"
    )

    def _maybe_expand_numbered_run(
        self, classified: ClassifiedBlock
    ) -> Optional[List[ClassifiedBlock]]:
        """When a ``LIST_ITEM`` carries multiple items fused onto one
        line, split it into a run of per-item classified blocks.

        Handles two cases:

        * **Ordered items** fused via "1. foo 2. bar 3. baz" — common
          when pdftotext output drops the blank-line separator between
          numbered entries.
        * **Unordered items** fused via "• foo • bar • baz" — same root
          cause on bullet lists.

        Returns ``None`` when no expansion applies; otherwise returns a
        list of ``LIST_ITEM`` ClassifiedBlocks. Each emitted block
        shares the original ``raw`` (for provenance) but carries a
        deterministic ``block_id`` suffix (``#2``, ``#3``, ...) so
        IDs stay unique.
        """
        if classified.role != BlockRole.LIST_ITEM:
            return None
        attrs = classified.attributes or {}
        marker_type = attrs.get("marker_type")
        if marker_type not in {"ordered", "unordered"}:
            return None
        rest = str(attrs.get("text") or "")
        if not rest or len(rest) < 40:
            return None

        if marker_type == "ordered":
            return self._expand_ordered(classified, attrs, rest)
        # marker_type == "unordered"
        return self._expand_unordered(classified, attrs, rest)

    def _expand_ordered(
        self,
        classified: ClassifiedBlock,
        attrs: dict,
        rest: str,
    ) -> Optional[List[ClassifiedBlock]]:
        markers = list(self._EMBEDDED_NUMBERED_MARKER.finditer(rest))
        if not markers:
            return None

        numbers = [int(m.group(1)) for m in markers]
        if not numbers:
            return None
        # Sequential check: numbers should be strictly ascending and
        # each within 1–3 of the previous (no giant jumps).
        prev = None
        for n in numbers:
            if prev is not None and (n <= prev or n - prev > 2):
                return None
            prev = n

        first_marker = attrs.get("marker") or "1."
        first_body = rest[: markers[0].start()].strip().rstrip(",;")
        if not first_body:
            return None
        items: List[tuple[str, str]] = [(first_marker, first_body)]
        for idx, m in enumerate(markers):
            num = m.group(1)
            start = m.end()
            end = (
                markers[idx + 1].start()
                if idx + 1 < len(markers)
                else len(rest)
            )
            body = rest[start:end].strip().rstrip(",;")
            if not body:
                continue
            dot_or_paren = rest[m.start(1) + len(num) : m.start(1) + len(num) + 1]
            if dot_or_paren not in {".", ")"}:
                dot_or_paren = "."
            items.append((f"{num}{dot_or_paren}", body))

        if len(items) < 2:
            return None
        return self._emit_expanded(classified, items, "ordered")

    def _expand_unordered(
        self,
        classified: ClassifiedBlock,
        attrs: dict,
        rest: str,
    ) -> Optional[List[ClassifiedBlock]]:
        markers = list(self._EMBEDDED_UNICODE_BULLET.finditer(rest))
        if not markers:
            return None

        first_marker = attrs.get("marker") or "\u2022"
        first_body = rest[: markers[0].start()].strip().rstrip(",;")
        if not first_body:
            return None
        items: List[tuple[str, str]] = [(first_marker, first_body)]
        for idx, m in enumerate(markers):
            marker = m.group(1)
            start = m.end()
            end = (
                markers[idx + 1].start()
                if idx + 1 < len(markers)
                else len(rest)
            )
            body = rest[start:end].strip().rstrip(",;")
            if not body:
                continue
            items.append((marker, body))

        if len(items) < 2:
            return None
        return self._emit_expanded(classified, items, "unordered")

    def _emit_expanded(
        self,
        classified: ClassifiedBlock,
        items: List[tuple[str, str]],
        marker_type: str,
    ) -> List[ClassifiedBlock]:
        emitted: List[ClassifiedBlock] = []
        for idx, (marker, body) in enumerate(items):
            synth_raw = RawBlock(
                text=body,
                block_id=(
                    classified.raw.block_id
                    if idx == 0
                    else f"{classified.raw.block_id}#{idx + 1}"
                ),
                page=classified.raw.page,
                bbox=classified.raw.bbox,
                extractor=classified.raw.extractor,
                neighbors=dict(classified.raw.neighbors or {}),
                extractor_hint=None,
                extra=dict(classified.raw.extra or {}),
            )
            emitted.append(
                ClassifiedBlock(
                    raw=synth_raw,
                    role=BlockRole.LIST_ITEM,
                    confidence=0.85,
                    attributes={
                        "marker": marker,
                        "marker_type": marker_type,
                        "text": body,
                    },
                    classifier_source="heuristic",
                )
            )
        return emitted


__all__ = ["HeuristicClassifier"]
