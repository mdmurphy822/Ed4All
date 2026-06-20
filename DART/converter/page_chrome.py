"""Wave 20: running header / footer / page-number chrome detection.

pdftotext faithfully reproduces running headers and footers on every
page. For a long textbook (hundreds of pages), a two-line chrome
(title + page number) produces thousands of spurious lines in the
content stream, which then bulk-segment into standalone ``<p>`` blocks,
pollute block_templates output, and corrupt any downstream text index.

This module runs **before** block segmentation. It analyses the
form-feed-delimited per-page text from pdftotext, identifies lines that
repeat across pages with frequency above a configurable threshold, and
returns a :class:`PageChrome` record describing the detected chrome
lines plus a stripped variant of the raw text where the chrome lines
have been replaced with empty lines (form-feed boundaries preserved so
downstream per-page attribution still works).

Design notes
------------

* **Frequency-first.** The primary signal is "this normalised line
  appears at the top or bottom of at least ``min_repeat_fraction`` of
  pages". Bbox-based layout confirmation (when ``text_spans`` are
  provided from PyMuPDF) is a secondary tiebreak — we confirm a
  frequency-flagged candidate only when its bbox lives in the top 10%
  or bottom 10% of the page.
* **Page-number extraction.** When a chrome line ends in digits
  (``"<Book Title> 164"``, ``"164"``, ``"Chapter 3 — 47"``),
  we split the fixed prefix from the variable page tail and remember
  ``{page_number: original_line}`` in ``page_number_lines`` so
  downstream block attribution (``data-dart-pages="164"``) can still
  cite the right page.
* **False-positive guards.** Long lines (>= 80 chars), lines starting
  with common heading markers (``Chapter N``, ``Section N.M``), and
  cases where only the trailing digit varies with a fixed prefix
  shorter than three chars are excluded — these tend to be legitimate
  content, not chrome.
* **Idempotent.** Running ``strip_page_chrome`` twice yields the same
  output as running it once.

Only :func:`detect_page_chrome` and :func:`strip_page_chrome` are
public; everything else is implementation detail.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


_FORM_FEED = "\x0c"


# Number of leading / trailing non-blank lines per page to scan for
# chrome candidates. Three lines at top + three at bottom is sufficient
# to catch running headers + page numbers + section markers without
# leaking into body content.
_HEAD_SCAN_LINES = 3
_TAIL_SCAN_LINES = 3


# Regex — "text ending in digits" — captures the non-digit prefix and
# the trailing integer so we can split a chrome line like
# ``"<Book Title> 164"`` into ``("<Book Title>", 164)``.
#
# Boundary contract: the trailing page-number digits must be at the
# START of the string OR preceded by whitespace — they must NOT be
# glued to a non-space, non-digit character. Without this guard a
# constant footer URL like ``http://cnx.org/content/col31130/1.4``
# matches the ``4`` glued to ``1.`` and is mistaken for page number 4
# (the OpenStax page-stamping defect). The prefix group is optional
# (``(?:(.*?)\s+)?``) so a bare-number line (``"164"``) still extracts
# correctly with ``group(1) is None`` (the caller coerces that to the
# empty-prefix ``""`` semantics). The required ``\s+`` before the
# digit-run means ``col31130/1.4`` / ``col12116/1.7`` no longer yield
# a page number, while ``"<Book Title> 164"``, ``"Chapter 3 — 47"``
# (em-dash is space-surrounded after normalisation), and bare ``"164"``
# all still extract.
_TRAILING_DIGITS_RE = re.compile(r"^(?:(.*?)\s+)?(\d{1,4})\s*$")


# Wave 25 Fix 1: mirror of the trailing-digits regex for the
# even-page leading-digit footer pattern (``"{N} <author-name>"``).
# Captures the variable page-number head and the fixed residual. The
# residual must be at least one non-digit char after the required
# whitespace gap to avoid matching plain numbers.
_LEADING_DIGITS_RE = re.compile(r"^\s*(\d{1,4})\s+(\S.*?)\s*$")


# Regex — lines starting with a heading marker that should never be
# treated as chrome even when they happen to repeat.
_HEADING_MARKER_RE = re.compile(
    r"^\s*(chapter|section|appendix|part|book|volume|unit)\s+\d",
    re.IGNORECASE,
)


# Lines this long are presumed to be real content, never chrome —
# running headers are short by convention (book title, section
# reference, page number).
_MAX_CHROME_LINE_LEN = 80


@dataclass
class PageChrome:
    """Detected per-page chrome for a document.

    Attributes:
        headers: Normalised header-chrome text strings (without any
            trailing page number). Normalisation is lowercase, whitespace-
            collapsed, Unicode-normalised. Compare against a line's
            normalised form to test for membership.
        footers: Normalised footer-chrome text strings.
        page_number_lines: Mapping ``{page_number_1_indexed: original_line}``
            carrying the raw chrome line for each page where a numbered
            chrome was detected. Downstream block attribution reads this
            to populate ``data-dart-pages="N"``.
        stripped_pages: The per-page text (split on form-feed) after
            chrome lines have been replaced with empty lines. The
            caller (:func:`strip_page_chrome`) owns the form-feed
            reassembly; this field is kept here for callers that
            want per-page access without re-splitting.
    """

    headers: Set[str] = field(default_factory=set)
    footers: Set[str] = field(default_factory=set)
    page_number_lines: Dict[int, str] = field(default_factory=dict)
    stripped_pages: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def _normalise(line: str) -> str:
    """Return a canonical form for frequency comparison.

    Steps:

    1. Unicode normalise (NFKC) so non-breaking spaces / full-width
       digits collapse to their ASCII equivalents.
    2. Strip whitespace.
    3. Lower-case.
    4. Collapse internal whitespace runs to single spaces.
    """
    if not line:
        return ""
    nfkc = unicodedata.normalize("NFKC", line)
    stripped = nfkc.strip()
    if not stripped:
        return ""
    lowered = stripped.lower()
    return re.sub(r"\s+", " ", lowered)


def _strip_trailing_digits(normalised: str) -> Tuple[str, Optional[int]]:
    """Split ``normalised`` into ``(prefix, page_number)``.

    Returns ``(prefix, page_num)`` when the line ends in digits (with
    ``prefix`` lowercased + stripped); otherwise ``(normalised, None)``.
    """
    if not normalised:
        return "", None
    match = _TRAILING_DIGITS_RE.match(normalised)
    if not match:
        return normalised, None
    prefix = (match.group(1) or "").strip()
    try:
        page = int(match.group(2))
    except (TypeError, ValueError):
        return normalised, None
    return prefix, page


def _strip_leading_digits(normalised: str) -> Tuple[str, Optional[int]]:
    """Split ``normalised`` into ``(residual, page_number)`` — leading form.

    Wave 25 Fix 1: even-page running footers like
    ``"{N} <author-name>"`` put the page number BEFORE the fixed
    text. This is the mirror of :func:`_strip_trailing_digits` —
    returns ``(residual, page_num)`` when the line starts with digits
    followed by whitespace + residual text, otherwise
    ``(normalised, None)``.
    """
    if not normalised:
        return "", None
    match = _LEADING_DIGITS_RE.match(normalised)
    if not match:
        return normalised, None
    try:
        page = int(match.group(1))
    except (TypeError, ValueError):
        return normalised, None
    residual = (match.group(2) or "").strip()
    return residual, page


def _is_heading_marker(line: str) -> bool:
    """Return ``True`` when ``line`` looks like a chapter/section heading."""
    if not line:
        return False
    return bool(_HEADING_MARKER_RE.match(line))


def _page_non_blank_lines(page_text: str) -> List[Tuple[int, str]]:
    """Return ``[(line_index, raw_line)]`` for non-blank lines on a page.

    ``line_index`` is the index into ``page_text.splitlines()`` — the
    caller uses it to mutate the page text in place when stripping.
    """
    if not page_text:
        return []
    result: List[Tuple[int, str]] = []
    for idx, line in enumerate(page_text.splitlines()):
        if line.strip():
            result.append((idx, line))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_page_chrome(
    raw_pdftotext: str,
    *,
    text_spans: Optional[list] = None,
    min_repeat_fraction: float = 0.3,
    min_pages_to_analyze: int = 4,
) -> PageChrome:
    """Detect running headers, footers, and page-number chrome.

    Parameters
    ----------
    raw_pdftotext:
        The raw ``pdftotext -layout`` output for the full document.
        Form-feed (``\\x0c``) boundaries between pages are required —
        single-page / form-feed-less input yields an empty chrome
        record.
    text_spans:
        Optional list of PyMuPDF-sourced ``ExtractedTextSpan`` records
        (bbox + text). When provided, bbox-based layout confirmation
        upgrades a frequency candidate to confirmed chrome when its
        bbox lives in the top 10% or bottom 10% of the page. Absence
        of spans is fine — frequency alone is still a strong signal.
    min_repeat_fraction:
        Minimum fraction of analysable pages a line must appear on (at
        the top or bottom) to be considered chrome. Default 0.3 (30%)
        — lower than a naive 0.5 threshold to catch chapter-odd /
        chapter-even alternating headers that only show on half the
        pages.
    min_pages_to_analyze:
        Minimum number of pages in the document for frequency analysis
        to run. Short documents (< 4 pages) lack the repetition signal
        required for reliable chrome detection, so we return an empty
        :class:`PageChrome` and skip stripping.

    Returns
    -------
    PageChrome
        Populated with detected chrome lines + per-page page-number
        mapping. Never ``None``. When detection fails (short document,
        no form-feeds, no repetition), every field is empty and
        ``stripped_pages`` lists the input pages verbatim.
    """
    if not raw_pdftotext or _FORM_FEED not in raw_pdftotext:
        return PageChrome(stripped_pages=[raw_pdftotext] if raw_pdftotext else [])

    pages = raw_pdftotext.split(_FORM_FEED)
    if len(pages) < min_pages_to_analyze:
        return PageChrome(stripped_pages=pages)

    # Per-page candidate gathering. For every page, record the
    # normalised + prefix-stripped form of the top-N and bottom-N
    # non-blank lines together with the raw line and its positional
    # index so we can mutate the page text later.
    #
    # Wave 25 Fix 1: each candidate now records BOTH possible
    # partitions — trailing-digit (``"<Book Title> 42"``) and leading-
    # digit (``"42 <author-name>"``). The frequency counter then evaluates
    # each partition independently, so a document that uses odd-page
    # trailing-digit headers AND even-page leading-digit footers
    # detects both patterns simultaneously. The tuple stored is
    # ``(tail_key, head_key, raw_line, idx)`` where either key may be
    # ``None`` when that partition does not apply (line has no digits,
    # or the residual is too short). Later counting logic iterates
    # both keys per candidate.
    top_candidates: List[List[Tuple[Optional[str], Optional[str], str, int]]] = []
    bottom_candidates: List[List[Tuple[Optional[str], Optional[str], str, int]]] = []

    def _derive_keys(raw_line: str) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(trailing_key, leading_key)`` for ``raw_line``.

        Either element may be ``None`` when that partition yields no
        usable chrome candidate (e.g. the line has no digits; the
        residual is empty; the leading-digit residual is too short).

        The ``trailing_key`` mirrors pre-Wave-25 behaviour — the
        prefix (variable tail stripped) or the ``__page_number_only__``
        sentinel for bare-number lines.

        The ``leading_key`` is keyed by the residual text AFTER the
        leading integer (e.g. the author-name residual of
        ``"42 J. Smith"`` is ``"j. smith"``). Bare numbers
        have no leading-digit residual, so ``leading_key`` is None
        for them — the trailing-digit path already handles that case
        via the sentinel.
        """
        norm = _normalise(raw_line)
        if not norm:
            return None, None
        tail_prefix, _tail_page = _strip_trailing_digits(norm)
        tail_key: Optional[str] = tail_prefix if tail_prefix else "__page_number_only__"
        lead_residual, lead_page = _strip_leading_digits(norm)
        lead_key: Optional[str]
        if lead_page is None or not lead_residual:
            lead_key = None
        else:
            # Guard: residual must be non-trivial (>= 3 chars) to
            # avoid false positives like ``"5 X"`` (a lone letter).
            # This matches the trailing-digit short-prefix guard.
            if len(lead_residual) < 3:
                lead_key = None
            else:
                # Mark with a sentinel prefix so leading-keyed and
                # trailing-keyed detections never collide in the
                # shared counts dicts (the residual text could
                # coincidentally match a trailing-key prefix from
                # some other line).
                lead_key = f"__lead__:{lead_residual}"
        return tail_key, lead_key

    for page_text in pages:
        non_blank = _page_non_blank_lines(page_text)
        top = non_blank[:_HEAD_SCAN_LINES]
        bottom = non_blank[-_TAIL_SCAN_LINES:] if non_blank else []

        top_list: List[Tuple[Optional[str], Optional[str], str, int]] = []
        for idx, raw_line in top:
            tail_key, lead_key = _derive_keys(raw_line)
            if tail_key is None and lead_key is None:
                continue
            top_list.append((tail_key, lead_key, raw_line, idx))
        top_candidates.append(top_list)

        bottom_list: List[Tuple[Optional[str], Optional[str], str, int]] = []
        for idx, raw_line in bottom:
            tail_key, lead_key = _derive_keys(raw_line)
            if tail_key is None and lead_key is None:
                continue
            bottom_list.append((tail_key, lead_key, raw_line, idx))
        bottom_candidates.append(bottom_list)

    # Count per-position. Keep separate counts for top vs bottom so a
    # line that only appears as a footer isn't wrongly classified as a
    # header (and vice versa). Both partitions (trailing / leading) are
    # counted independently so even-page leading-digit footers land
    # alongside odd-page trailing-digit headers.
    def _accumulate(
        page_lists: List[List[Tuple[Optional[str], Optional[str], str, int]]],
    ) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for page_list in page_lists:
            seen_on_page: Set[str] = set()
            for tail_key, lead_key, _raw, _idx in page_list:
                for key in (tail_key, lead_key):
                    if not key or key in seen_on_page:
                        continue
                    seen_on_page.add(key)
                    counts[key] = counts.get(key, 0) + 1
        return counts

    top_counts = _accumulate(top_candidates)
    bottom_counts = _accumulate(bottom_candidates)

    threshold = max(2, int(round(len(pages) * float(min_repeat_fraction))))

    header_keys = {k for k, c in top_counts.items() if c >= threshold}
    footer_keys = {k for k, c in bottom_counts.items() if c >= threshold}

    # False-positive guards applied AFTER frequency thresholding:
    #
    #   * Drop any key whose prefix starts with a heading marker
    #     ("Chapter 1", "Section 2.3"). These look like real content.
    #   * Drop any key whose displayed representation is excessively
    #     long — running headers are short by convention.
    #   * Drop any key where only the digit varies AND the fixed prefix
    #     is shorter than three chars — a two-char prefix like "p." is
    #     chrome-ish, but a single-char/empty prefix often catches
    #     numbered list bleed. Empty-prefix ("__page_number_only__")
    #     stays allowed because bare page numbers are legitimate chrome.
    def _is_valid_chrome_key(key: str, counts: Dict[str, int]) -> bool:
        if key == "__page_number_only__":
            return True
        # Wave 25 Fix 1: leading-digit keys carry a ``__lead__:``
        # sentinel prefix — strip it for the guards so the residual
        # is evaluated against the same rules as trailing-digit keys.
        if key.startswith("__lead__:"):
            body = key[len("__lead__:") :]
            if _is_heading_marker(body):
                return False
            if len(body) >= _MAX_CHROME_LINE_LEN:
                return False
            if len(body) < 3:
                return False
            return True
        if _is_heading_marker(key):
            return False
        # Length guard on the normalised prefix.
        if len(key) >= _MAX_CHROME_LINE_LEN:
            return False
        # Short-prefix guard (skip very short prefixes — too ambiguous).
        if len(key) < 3:
            return False
        return True

    header_keys = {k for k in header_keys if _is_valid_chrome_key(k, top_counts)}
    footer_keys = {k for k in footer_keys if _is_valid_chrome_key(k, bottom_counts)}

    # Bbox-based confirmation: when text_spans are available, keep only
    # keys that appear in the top-10% or bottom-10% of at least one
    # page. Absent spans => frequency alone is the signal.
    if text_spans:
        confirmed_header_keys = _confirm_chrome_by_bbox(
            header_keys, text_spans, edge="top"
        )
        confirmed_footer_keys = _confirm_chrome_by_bbox(
            footer_keys, text_spans, edge="bottom"
        )
        # Don't drop a key just because bbox confirmation fails — many
        # PDFs won't have perfectly clean bbox data. Use bbox as a
        # UPGRADE signal (union-or-keep), not a filter.
        header_keys = header_keys | confirmed_header_keys
        footer_keys = footer_keys | confirmed_footer_keys

    # Build the displayed headers/footers set and collect per-page
    # page-number mappings.
    #
    # Wave 25 Fix 1: leading-digit keys carry a ``__lead__:`` sentinel
    # prefix; we store them stripped of the prefix in the public
    # ``headers`` / ``footers`` sets so downstream callers see the
    # real residual text (the author / fixed suffix after the page
    # number). Internal matching during stripping uses the
    # sentinel-prefixed form.
    def _display_form(key: str) -> Optional[str]:
        if key == "__page_number_only__":
            return None
        if key.startswith("__lead__:"):
            return key[len("__lead__:") :]
        return key

    headers: Set[str] = set()
    for key in header_keys:
        display = _display_form(key)
        if display is not None:
            headers.add(display)

    footers: Set[str] = set()
    for key in footer_keys:
        display = _display_form(key)
        if display is not None:
            footers.add(display)

    page_number_lines: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Constant-value guard (OpenStax page-stamping defect).
    #
    # A real running page number VARIES across the pages it appears on
    # (164, 165, 166, ...). A digit that is IDENTICAL on every page a
    # given chrome key fires is fixed chrome text — a version string
    # tail (``.../col31130/1.4``), a date, an edition number — never
    # pagination. The boundary fix above already rejects digits glued
    # to non-space characters, but a footer that legitimately ends in a
    # space-separated constant digit (``"... Edition 2"`` repeated on
    # every page) would still slip through, so we additionally suppress
    # any key whose extracted page-numbers never vary.
    #
    # We accumulate, per matched chrome key (trailing prefix / leading
    # ``__lead__:`` residual / ``__page_number_only__`` sentinel), the
    # set of page-numbers extracted across all pages where that key
    # fired. A key whose number-set has size <= 1 (constant or absent
    # variation) is dropped from the page-number contribution. Chrome
    # STRIPPING is unaffected — the line is still cleared from the
    # content stream below; only its hijacking of ``page_number_lines``
    # to a single fixed label is suppressed.
    def _matched_key_and_page(
        partition: str,
        tail_key: Optional[str],
        lead_key: Optional[str],
        norm: str,
    ) -> Tuple[Optional[str], Optional[int]]:
        """Return ``(chrome_key, page_number)`` for a matched line."""
        if partition == "tail":
            _prefix, maybe_page = _strip_trailing_digits(norm)
            return tail_key, maybe_page
        _residual, maybe_page = _strip_leading_digits(norm)
        return lead_key, maybe_page

    key_page_numbers: Dict[str, Set[int]] = {}
    for page_index in range(len(pages)):
        for tail_key, lead_key, raw_line, _idx in top_candidates[page_index]:
            if tail_key is not None and tail_key in header_keys:
                partition = "tail"
            elif lead_key is not None and lead_key in header_keys:
                partition = "lead"
            else:
                continue
            key, maybe_page = _matched_key_and_page(
                partition, tail_key, lead_key, _normalise(raw_line)
            )
            if key is not None and maybe_page is not None:
                key_page_numbers.setdefault(key, set()).add(maybe_page)
        for tail_key, lead_key, raw_line, _idx in bottom_candidates[page_index]:
            if tail_key is not None and tail_key in footer_keys:
                partition = "tail"
            elif lead_key is not None and lead_key in footer_keys:
                partition = "lead"
            else:
                continue
            key, maybe_page = _matched_key_and_page(
                partition, tail_key, lead_key, _normalise(raw_line)
            )
            if key is not None and maybe_page is not None:
                key_page_numbers.setdefault(key, set()).add(maybe_page)

    # A key whose extracted page-numbers never vary across the pages it
    # fired on is fixed chrome text, not pagination.
    constant_keys: Set[str] = {
        key for key, numbers in key_page_numbers.items() if len(numbers) <= 1
    }

    # Now strip: for every page, walk the stored (tail_key, lead_key,
    # raw, idx) lists and clear each chrome-flagged line. Also extract
    # the page number from a numbered chrome line — the partition
    # that matched determines whether we look at the head or tail.
    stripped_pages: List[str] = []
    for page_index, page_text in enumerate(pages):
        page_number_1based = page_index + 1
        lines = page_text.splitlines()
        to_clear: Set[int] = set()

        for tail_key, lead_key, raw_line, idx in top_candidates[page_index]:
            matched_partition: Optional[str] = None
            if tail_key is not None and tail_key in header_keys:
                matched_partition = "tail"
            elif lead_key is not None and lead_key in header_keys:
                matched_partition = "lead"
            if matched_partition is None:
                continue
            to_clear.add(idx)
            norm = _normalise(raw_line)
            matched_key = tail_key if matched_partition == "tail" else lead_key
            if matched_partition == "tail":
                _prefix, maybe_page = _strip_trailing_digits(norm)
            else:
                _residual, maybe_page = _strip_leading_digits(norm)
            if maybe_page is not None and matched_key not in constant_keys:
                page_number_lines.setdefault(page_number_1based, raw_line)

        for tail_key, lead_key, raw_line, idx in bottom_candidates[page_index]:
            matched_partition = None
            if tail_key is not None and tail_key in footer_keys:
                matched_partition = "tail"
            elif lead_key is not None and lead_key in footer_keys:
                matched_partition = "lead"
            if matched_partition is None:
                continue
            to_clear.add(idx)
            norm = _normalise(raw_line)
            matched_key = tail_key if matched_partition == "tail" else lead_key
            if matched_partition == "tail":
                _prefix, maybe_page = _strip_trailing_digits(norm)
            else:
                _residual, maybe_page = _strip_leading_digits(norm)
            if maybe_page is not None and matched_key not in constant_keys:
                page_number_lines.setdefault(page_number_1based, raw_line)

        if to_clear:
            new_lines = [
                "" if line_idx in to_clear else line
                for line_idx, line in enumerate(lines)
            ]
            # Collapse leading / trailing blank runs so the segmenter
            # doesn't produce phantom empty blocks from stripped chrome.
            while new_lines and not new_lines[0].strip():
                new_lines.pop(0)
            while new_lines and not new_lines[-1].strip():
                new_lines.pop()
            stripped_pages.append("\n".join(new_lines))
        else:
            stripped_pages.append(page_text)

    return PageChrome(
        headers=headers,
        footers=footers,
        page_number_lines=page_number_lines,
        stripped_pages=stripped_pages,
    )


# ---------------------------------------------------------------------------
# Phase 4: printed-label accuracy via physical→printed offset interpolation
# ---------------------------------------------------------------------------
#
# The page-chrome detector populates ``PageChrome.page_number_lines`` with the
# DIRECTLY-DETECTED printed labels (``{physical_page_1indexed: chrome_line}``).
# On corpora with sparse / no digit-tailed chrome, most pages have NO direct
# label and historically fall back to the raw PHYSICAL PDF page — wrong by the
# front-matter offset (printed p.1 may be physical p.15).
#
# ``derive_page_label_map`` closes that gap WITHOUT fabricating: from the
# confirmed ``(physical, printed)`` pairs it derives the physical→printed
# offset (``printed - physical``) and, for an unlabeled page whose run is
# covered by a CONFIDENT, consistent offset, emits the INTERPOLATED printed
# page. Pages with no confident offset stay physical. The result feeds a
# three-valued provenance kind so downstream consumers know whether a "p. N"
# is directly printed, interpolated, or a raw physical fallback.
#
# Anti-fabrication contract (RISK-A):
#   * Interpolate ONLY from confirmed ``(physical, printed)`` pairs.
#   * Zero confirmed pairs  -> every page is ``physical`` (never guess).
#   * A run with conflicting / inconsistent offsets -> that run is ``physical``.
#   * A directly-detected label is always ``printed``.

PAGE_KIND_PRINTED = "printed"
PAGE_KIND_INTERPOLATED = "interpolated"
PAGE_KIND_PHYSICAL = "physical"

# Default provenance kind for any page with no resolvable printed signal.
# Pinned-contract: an ABSENT ``data-dart-page-kind`` attribute means
# ``physical`` (back-compat), so the emitter omits the attribute for this
# value; this constant is the in-memory default the resolver returns.
DEFAULT_PAGE_KIND = PAGE_KIND_PHYSICAL


@dataclass(frozen=True)
class PageLabel:
    """Resolved printed-page label for a single physical page.

    Attributes:
        physical: 1-indexed physical PDF page.
        printed: The printed label to surface (``data-dart-pages``). Equals
            ``physical`` for a ``physical``-kind page.
        kind: One of ``printed`` / ``interpolated`` / ``physical``.
    """

    physical: int
    printed: int
    kind: str


def _confirmed_pairs(page_number_lines: Dict[int, str]) -> List[Tuple[int, int]]:
    """Extract confirmed ``(physical, printed)`` pairs from chrome lines.

    Each value in ``page_number_lines`` is the raw chrome line; the printed
    page number is its trailing-or-leading digit run. Returns the pairs sorted
    by physical page. A line that yields no parseable number is skipped (it
    never confirms an offset). Both the trailing-digit (``"Title 47"``) and
    leading-digit (``"47 Author"``) chrome forms are honoured so even-page
    footers contribute pairs too.
    """
    pairs: List[Tuple[int, int]] = []
    for physical in sorted(page_number_lines):
        raw_line = page_number_lines[physical]
        norm = _normalise(raw_line)
        _prefix, printed = _strip_trailing_digits(norm)
        if printed is None:
            _residual, printed = _strip_leading_digits(norm)
        if printed is None:
            continue
        pairs.append((int(physical), int(printed)))
    return pairs


def _segment_offsets(
    pairs: List[Tuple[int, int]],
) -> List[Tuple[int, int, int]]:
    """Group confirmed pairs into contiguous runs sharing a constant offset.

    Returns a list of ``(start_physical, end_physical, offset)`` segments
    where ``offset = printed - physical`` is constant across the segment and
    consistent with EVERY confirmed pair inside the ``[start, end]`` physical
    range. A run breaks when a later pair's offset differs from the run's
    offset (a front-matter→body restart is exactly such a break).

    The segment boundaries extend only as far as the OUTERMOST confirmed pairs
    that agree on the offset — interpolation never extends a run past the last
    pair that validates it. Pairs whose offset is internally contradictory
    (two pairs at the same physical page disagreeing, or a pair that breaks an
    otherwise-consistent run mid-stream) start a new segment, so a corrupt run
    cannot poison a clean one.
    """
    if not pairs:
        return []

    segments: List[Tuple[int, int, int]] = []
    run_start = pairs[0][0]
    run_end = pairs[0][0]
    run_offset = pairs[0][1] - pairs[0][0]

    for physical, printed in pairs[1:]:
        offset = printed - physical
        if offset == run_offset:
            run_end = physical
        else:
            segments.append((run_start, run_end, run_offset))
            run_start = physical
            run_end = physical
            run_offset = offset

    segments.append((run_start, run_end, run_offset))
    return segments


def derive_page_label_map(
    page_number_lines: Dict[int, str],
    *,
    total_pages: Optional[int] = None,
    interpolation_reach: Optional[int] = None,
) -> Dict[int, PageLabel]:
    """Resolve a printed label + provenance kind for every physical page.

    Parameters
    ----------
    page_number_lines:
        ``PageChrome.page_number_lines`` — the directly-detected printed
        labels keyed by physical page.
    total_pages:
        When known, the resolver returns a :class:`PageLabel` for every
        physical page in ``1..total_pages`` (unlabeled pages get either an
        interpolated or a physical entry). When ``None``, the map covers only
        directly-detected pages plus pages reachable by interpolation within
        ``interpolation_reach`` of a confirmed pair.
    interpolation_reach:
        Maximum physical-page distance from a confirmed pair within a segment
        that an unlabeled page may be interpolated across. ``None`` (default)
        means "no distance cap inside a segment" — every page inside a
        confident segment interpolates. A finite reach also lets a confident
        segment extend its offset to the immediately-adjacent unlabeled pages
        even past the outermost confirmed pair (bounded extrapolation); set it
        when front-matter pages before the first confirmed label should still
        resolve. Defaults to in-segment-only interpolation (the honest, no
        guessing-past-the-evidence posture).

    Returns
    -------
    Dict[int, PageLabel]
        ``{physical_page: PageLabel}``. A directly-detected page is always
        ``kind="printed"``. A page inside a confident offset segment is
        ``kind="interpolated"``. Every other page is ``kind="physical"``
        (``printed == physical``). With ZERO confirmed pairs the map is empty
        (caller falls back to physical for every page) — never a guess.
    """
    pairs = _confirmed_pairs(page_number_lines)

    # Detect contradictory pairs at the SAME physical page: a physical page
    # that yields two different printed numbers across detection is
    # low-confidence; drop it from the confirmed set entirely.
    seen: Dict[int, int] = {}
    contradictory: Set[int] = set()
    for physical, printed in pairs:
        if physical in seen and seen[physical] != printed:
            contradictory.add(physical)
        else:
            seen[physical] = printed
    if contradictory:
        pairs = [(p, n) for (p, n) in pairs if p not in contradictory]

    result: Dict[int, PageLabel] = {}
    direct: Dict[int, int] = {p: n for (p, n) in pairs}

    # Always stamp the directly-detected printed labels first.
    for physical, printed in direct.items():
        result[physical] = PageLabel(physical, printed, PAGE_KIND_PRINTED)

    if not pairs:
        # Anti-fabrication: no confirmed pair -> nothing to interpolate. The
        # caller treats absence as physical.
        if total_pages:
            for physical in range(1, int(total_pages) + 1):
                result.setdefault(
                    physical, PageLabel(physical, physical, PAGE_KIND_PHYSICAL)
                )
        return result

    segments = _segment_offsets(pairs)

    def _interpolate(physical: int) -> Optional[PageLabel]:
        for start, end, offset in segments:
            lo, hi = start, end
            if interpolation_reach is not None:
                lo = start - interpolation_reach
                hi = end + interpolation_reach
            if lo <= physical <= hi:
                printed = physical + offset
                if printed < 1:
                    # Never emit a non-positive printed page (front-matter
                    # below the printed-1 boundary): honest physical fallback.
                    return None
                return PageLabel(physical, printed, PAGE_KIND_INTERPOLATED)
        return None

    # Determine the physical pages to resolve.
    if total_pages:
        candidate_pages = range(1, int(total_pages) + 1)
    else:
        # Resolve only directly-detected pages + pages reachable by
        # interpolation inside a segment (and any reach extension).
        covered: Set[int] = set(direct)
        for start, end, _offset in segments:
            lo = start if interpolation_reach is None else start - interpolation_reach
            hi = end if interpolation_reach is None else end + interpolation_reach
            covered.update(range(max(1, lo), hi + 1))
        candidate_pages = sorted(covered)

    for physical in candidate_pages:
        if physical in result:
            continue  # already a directly-detected printed label
        interp = _interpolate(physical)
        if interp is not None:
            result[physical] = interp
        elif total_pages:
            result[physical] = PageLabel(physical, physical, PAGE_KIND_PHYSICAL)

    return result


def _confirm_chrome_by_bbox(
    candidate_keys: Set[str],
    text_spans: list,
    *,
    edge: str,
) -> Set[str]:
    """Confirm chrome candidates whose bbox sits at the top/bottom edge.

    Conservative: only confirms; never filters out. Absent bbox data
    degrades to an empty confirmation set.
    """
    if not candidate_keys or not text_spans:
        return set()

    per_page_heights: Dict[int, float] = {}
    for span in text_spans:
        bbox = getattr(span, "bbox", None) or ()
        if len(bbox) < 4:
            continue
        page = getattr(span, "page", None)
        if not isinstance(page, int):
            continue
        _, _, _, y1 = bbox
        try:
            y1f = float(y1)
        except (TypeError, ValueError):
            continue
        prev = per_page_heights.get(page, 0.0)
        if y1f > prev:
            per_page_heights[page] = y1f

    confirmed: Set[str] = set()
    for span in text_spans:
        text = getattr(span, "text", "") or ""
        if not text.strip():
            continue
        bbox = getattr(span, "bbox", None) or ()
        if len(bbox) < 4:
            continue
        page = getattr(span, "page", None)
        if not isinstance(page, int):
            continue
        try:
            _, y0, _, y1 = (float(x) for x in bbox)
        except (TypeError, ValueError):
            continue
        page_height = per_page_heights.get(page, 0.0)
        if page_height <= 0:
            continue
        if edge == "top":
            if y0 > page_height * 0.10:
                continue
        elif edge == "bottom":
            if y1 < page_height * 0.90:
                continue
        else:
            continue

        norm = _normalise(text)
        prefix, _ = _strip_trailing_digits(norm)
        trailing_key = prefix if prefix else "__page_number_only__"
        if trailing_key in candidate_keys:
            confirmed.add(trailing_key)
        # Wave 25 Fix 1: also produce the leading-digit key so
        # bbox-layer confirmation upgrades leading-digit footers
        # (``"{N} <author-name>"``) the same way it does trailing-digit
        # headers.
        residual, lead_page = _strip_leading_digits(norm)
        if lead_page is not None and residual and len(residual) >= 3:
            leading_key = f"__lead__:{residual}"
            if leading_key in candidate_keys:
                confirmed.add(leading_key)

    return confirmed


# ---------------------------------------------------------------------------
# OQ-3: bbox printed-label CROSS-CHECK (raises the printed-label hit-rate)
# ---------------------------------------------------------------------------
#
# The chrome detector confirms ``(physical, printed)`` pairs ONLY from
# digit-tail (or leading-digit) running-chrome lines in the pdftotext stream.
# On many corpora that signal is sparse — a clean PDF whose page number sits
# alone in the bottom margin produces a bare ``"47"`` line that the
# frequency/threshold path may never promote to chrome (a lone short digit run
# looks like list-bleed), so NO confirmed pair is recorded and the page falls
# back to its raw physical number.
#
# This cross-check ADDS confirmed pairs from PyMuPDF span POSITIONS. For each
# page it scans the spans whose bbox lives in the top-10% / bottom-10% margin
# band, finds a SHORT mostly-digit token (a bare page-number candidate), and —
# critically — only accepts it as a confirmed pair when it is CONSISTENT:
#
#   * It must form a MONOTONIC-increasing digit run across consecutive pages
#     at the SAME margin (top vs bottom), OR
#   * It must CORROBORATE an already-confirmed chrome pair (same printed number
#     on that physical page) / extend a chrome run by +1.
#
# A single arbitrary number floating in body text (outside the margin band) is
# never a candidate; an isolated, non-monotonic margin number is dropped. The
# bar is "agrees with or extends the evidence", never "invent a label".

# A printed-page-number candidate must be a SHORT token. A long header line
# that merely ends in digits is handled by the chrome digit-tail path, not
# here — the bbox cross-check targets the bare-number margin case.
_MAX_BBOX_LABEL_TOKEN_LEN = 12


def _span_printed_number(text: str) -> Optional[int]:
    """Extract a printed page number from a SHORT margin span ``text``.

    Reuses the same digit-extraction helpers chrome detection uses
    (``_strip_trailing_digits`` / ``_strip_leading_digits``) so the bbox
    rule matches the chrome rule. Accepts:

    * a bare digit run (``"47"``) — the common clean-PDF margin case, OR
    * a tiny header/footer where the page number is the trailing or leading
      digit run (``"47 Chapter Title"`` truncated, ``"p. 47"``).

    Rejects anything whose stripped form is longer than
    :data:`_MAX_BBOX_LABEL_TOKEN_LEN` after removing the number (a long
    running-header line is the chrome path's job, not this cross-check's), and
    anything with no parseable number.
    """
    norm = _normalise(text)
    if not norm:
        return None
    # Bare number — the high-confidence margin case.
    if re.fullmatch(r"\d{1,4}", norm):
        try:
            return int(norm)
        except ValueError:
            return None
    # Trailing-digit short token (e.g. "p. 47").
    prefix, page = _strip_trailing_digits(norm)
    if page is not None and len(prefix) <= _MAX_BBOX_LABEL_TOKEN_LEN:
        return page
    # Leading-digit short token (e.g. "47 ch3").
    residual, page = _strip_leading_digits(norm)
    if page is not None and len(residual) <= _MAX_BBOX_LABEL_TOKEN_LEN:
        return page
    return None


def _collect_bbox_label_candidates(
    text_spans: list,
) -> Dict[str, Dict[int, int]]:
    """Group bbox printed-number candidates by margin edge.

    Returns ``{"top": {physical_page: printed}, "bottom": {...}}`` — the
    raw, UN-validated candidates (one per page per edge; the first qualifying
    span on a page at a given edge wins). Anti-fabrication consistency is
    applied later by :func:`_confirm_monotonic_runs`.

    Graceful: spans without 4-tuple bbox / int page / positive page height are
    skipped. Page height is the max span ``y1`` on that page.
    """
    # Per-page height = max y1 across that page's spans.
    per_page_height: Dict[int, float] = {}
    for span in text_spans:
        bbox = getattr(span, "bbox", None) or ()
        if len(bbox) < 4:
            continue
        page = getattr(span, "page", None)
        if not isinstance(page, int):
            continue
        try:
            y1f = float(bbox[3])
        except (TypeError, ValueError):
            continue
        if y1f > per_page_height.get(page, 0.0):
            per_page_height[page] = y1f

    top: Dict[int, int] = {}
    bottom: Dict[int, int] = {}
    for span in text_spans:
        text = getattr(span, "text", "") or ""
        if not text.strip():
            continue
        bbox = getattr(span, "bbox", None) or ()
        if len(bbox) < 4:
            continue
        page = getattr(span, "page", None)
        if not isinstance(page, int):
            continue
        try:
            _, y0, _, y1 = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            continue
        height = per_page_height.get(page, 0.0)
        if height <= 0:
            continue
        printed = _span_printed_number(text)
        if printed is None:
            continue
        # Margin band: top 10% (y0 <= 0.10H) or bottom 10% (y1 >= 0.90H).
        if y0 <= height * 0.10:
            top.setdefault(page, printed)
        elif y1 >= height * 0.90:
            bottom.setdefault(page, printed)
        # Body-text numbers (neither band) are NEVER taken — anti-fabrication.

    return {"top": top, "bottom": bottom}


def _confirm_monotonic_runs(
    edge_candidates: Dict[int, int],
    *,
    existing_pairs: Dict[int, int],
) -> Dict[int, int]:
    """Filter raw margin candidates down to anti-fabrication-confirmed pairs.

    A candidate ``{physical: printed}`` at one margin edge is CONFIRMED when:

    * it belongs to a MONOTONIC-increasing run of length >= 2 across
      consecutive (or near-consecutive) physical pages whose printed numbers
      step in lock-step with the physical page (constant offset, +1 per page),
      OR
    * it CORROBORATES an existing chrome-confirmed pair on the same physical
      page (same printed number), OR
    * it extends an existing chrome pair by exactly the chrome offset on an
      adjacent physical page.

    An isolated, non-monotonic, offset-inconsistent margin number is DROPPED.
    Returns ``{physical: printed}`` of confirmed pairs only.

    The monotonic test keys on ``offset = printed - physical`` being CONSTANT
    across a contiguous physical run: a true running page number has a fixed
    front-matter offset, so consecutive pages share one offset. An arbitrary
    body number that slipped into a margin band almost never lines up with its
    neighbours' offsets, so it is rejected.
    """
    confirmed: Dict[int, int] = {}
    if not edge_candidates:
        return confirmed

    pages = sorted(edge_candidates)

    # 1) Monotonic constant-offset runs. Walk consecutive physical pages and
    #    grow a run while ``offset`` stays constant AND the physical step is +1
    #    (printed increments lock-step with physical). Runs of length >= 2 are
    #    confirmed in full.
    run: List[int] = [pages[0]]
    run_offset = edge_candidates[pages[0]] - pages[0]

    def _flush(r: List[int]) -> None:
        if len(r) >= 2:
            for p in r:
                confirmed[p] = edge_candidates[p]

    for phys in pages[1:]:
        offset = edge_candidates[phys] - phys
        prev = run[-1]
        if offset == run_offset and phys == prev + 1:
            run.append(phys)
        else:
            _flush(run)
            run = [phys]
            run_offset = offset
    _flush(run)

    # 2) Corroboration / single-page extension of the chrome-confirmed pairs.
    #    A lone margin candidate that AGREES with a chrome pair (same printed
    #    number, same physical page) is high-confidence even without a run.
    #    A candidate one physical page away from a chrome pair, stepping by the
    #    chrome offset, extends it.
    for phys, printed in edge_candidates.items():
        if phys in confirmed:
            continue
        if existing_pairs.get(phys) == printed:
            confirmed[phys] = printed
            continue
        for ex_phys, ex_printed in existing_pairs.items():
            ex_offset = ex_printed - ex_phys
            if printed - phys == ex_offset and abs(phys - ex_phys) == 1:
                confirmed[phys] = printed
                break

    return confirmed


def confirm_printed_labels_by_bbox(
    text_spans: list,
    *,
    existing_page_number_lines: Optional[Dict[int, str]] = None,
) -> Dict[int, str]:
    """Return ADDITIONAL ``{physical: chrome_line}`` pairs from span bboxes.

    The returned dict is in the SAME shape as
    :attr:`PageChrome.page_number_lines` (physical page -> a raw line whose
    digit run is the printed number) so it can be merged straight into a
    :class:`PageChrome` and consumed unchanged by
    :func:`derive_page_label_map`. The "line" we synthesise for a bbox-only
    confirmation is the bare printed-number string (e.g. ``"47"``), which
    re-parses to the same number through ``_confirmed_pairs``.

    Anti-fabrication: only consistency-confirmed candidates (monotonic margin
    runs, or corroboration/extension of the existing chrome pairs) are
    returned. Body-text numbers and isolated margin numbers are never
    returned. When ``text_spans`` is falsy or no candidate qualifies, the
    result is an EMPTY dict (caller's chrome-only behaviour is byte-identical).

    The result NEVER overrides an existing chrome pair — keys already present
    in ``existing_page_number_lines`` are omitted (the chrome line is the
    higher-provenance record). Both top and bottom margins contribute; on the
    rare page where both yield a confirmed-but-different number, the existing
    chrome pair (if any) wins, else the bottom-margin reading wins (page
    numbers live in the footer far more often than the header).
    """
    if not text_spans:
        return {}

    existing_pairs: Dict[int, int] = {}
    for physical, raw_line in (existing_page_number_lines or {}).items():
        norm = _normalise(raw_line)
        _prefix, printed = _strip_trailing_digits(norm)
        if printed is None:
            _residual, printed = _strip_leading_digits(norm)
        if printed is not None and isinstance(physical, int):
            existing_pairs[physical] = printed

    try:
        edges = _collect_bbox_label_candidates(text_spans)
    except Exception as exc:  # noqa: BLE001 — cross-check never blocks
        logger.debug("bbox printed-label candidate scan failed: %s", exc)
        return {}

    confirmed_top = _confirm_monotonic_runs(
        edges.get("top", {}), existing_pairs=existing_pairs
    )
    confirmed_bottom = _confirm_monotonic_runs(
        edges.get("bottom", {}), existing_pairs=existing_pairs
    )

    # Merge: bottom wins over top on conflict (footers carry page numbers
    # far more often); an existing chrome pair always wins over both.
    merged: Dict[int, int] = {}
    for phys, printed in confirmed_top.items():
        merged[phys] = printed
    for phys, printed in confirmed_bottom.items():
        merged[phys] = printed  # bottom overrides top

    additions: Dict[int, str] = {}
    for phys, printed in merged.items():
        if phys in existing_pairs:
            continue  # chrome pair is higher-provenance; never override
        additions[phys] = str(printed)
    return additions


def enrich_page_chrome_with_bbox(chrome: PageChrome, text_spans: list) -> int:
    """Merge bbox-confirmed printed-label pairs INTO ``chrome`` in place.

    Returns the number of pairs added. A no-op (returns 0) when
    ``text_spans`` is absent or no candidate qualifies — the chrome record is
    untouched and downstream behaviour stays byte-identical. Graceful: any
    failure is swallowed (returns 0) so the cross-check never blocks
    extraction.
    """
    if chrome is None or not text_spans:
        return 0
    try:
        additions = confirm_printed_labels_by_bbox(
            text_spans,
            existing_page_number_lines=chrome.page_number_lines,
        )
    except Exception as exc:  # noqa: BLE001 — cross-check never blocks
        logger.debug("bbox printed-label cross-check failed: %s", exc)
        return 0
    added = 0
    for physical, line in additions.items():
        if physical not in chrome.page_number_lines:
            chrome.page_number_lines[physical] = line
            added += 1
    return added


def strip_page_chrome(raw_pdftotext: str, chrome: PageChrome) -> str:
    """Return ``raw_pdftotext`` with every chrome line removed.

    Form-feed page boundaries are preserved so downstream per-page
    attribution still works. Leading / trailing blank runs per page are
    collapsed so the segmenter doesn't emit phantom empty blocks from
    stripped chrome.

    Idempotent: ``strip_page_chrome(strip_page_chrome(x, c), c)`` equals
    ``strip_page_chrome(x, c)`` — once the chrome is gone, a second
    pass finds nothing to strip.
    """
    if not raw_pdftotext:
        return raw_pdftotext

    # Short-circuit: detector already produced per-page stripped output
    # during analysis and this is the same raw_text it was called with.
    # When the count matches, return the cached per-page output joined
    # on form feeds.
    if chrome.stripped_pages and (
        _FORM_FEED not in raw_pdftotext
        or raw_pdftotext.split(_FORM_FEED) == raw_pdftotext.split(_FORM_FEED)  # always true
    ):
        # Re-derive from the chrome record when lengths match; otherwise
        # fall through and re-run the line-level strip against
        # ``raw_pdftotext`` (handles the idempotency case where
        # ``raw_pdftotext`` is itself already chrome-stripped).
        page_count_in = raw_pdftotext.count(_FORM_FEED) + 1 if _FORM_FEED in raw_pdftotext else 1
        if page_count_in == len(chrome.stripped_pages):
            return _FORM_FEED.join(chrome.stripped_pages)

    # General path: line-by-line strip against the current input. Used
    # for idempotency and for callers that didn't cache stripped_pages.
    if not (chrome.headers or chrome.footers or chrome.page_number_lines):
        return raw_pdftotext

    if _FORM_FEED in raw_pdftotext:
        pages = raw_pdftotext.split(_FORM_FEED)
    else:
        pages = [raw_pdftotext]

    stripped: List[str] = []
    for page_text in pages:
        lines = page_text.splitlines()
        new_lines: List[str] = []
        for line in lines:
            norm = _normalise(line)
            if not norm:
                new_lines.append(line)
                continue
            prefix, _page = _strip_trailing_digits(norm)
            key = prefix if prefix else "__page_number_only__"
            # Wave 25 Fix 1: leading-digit residual (mirror of prefix).
            residual, lead_page = _strip_leading_digits(norm)
            # Test against the same key forms stored in headers/footers
            # (prefixes, never the "__page_number_only__" sentinel —
            # that one we only match when explicitly a bare number).
            drop = False
            if prefix and (prefix in chrome.headers or prefix in chrome.footers):
                drop = True
            elif (
                lead_page is not None
                and residual
                and len(residual) >= 3
                and (
                    residual in chrome.headers or residual in chrome.footers
                )
            ):
                # Wave 25 Fix 1: leading-digit chrome line (``"{N}
                # <author-name>"``) — residual matches an even-page
                # footer recorded in the chrome record.
                drop = True
            elif key == "__page_number_only__" and chrome.page_number_lines:
                # Bare numbers were chrome: drop when the line is just
                # digits (we can't tell header vs footer from content
                # alone, but page_number_lines existence tells us we
                # detected page-number chrome).
                if re.fullmatch(r"\s*\d{1,4}\s*", line):
                    drop = True
            if drop:
                new_lines.append("")
            else:
                new_lines.append(line)
        while new_lines and not new_lines[0].strip():
            new_lines.pop(0)
        while new_lines and not new_lines[-1].strip():
            new_lines.pop()
        stripped.append("\n".join(new_lines))

    return _FORM_FEED.join(stripped) if _FORM_FEED in raw_pdftotext else stripped[0]


__all__ = [
    "PageChrome",
    "PageLabel",
    "PAGE_KIND_PRINTED",
    "PAGE_KIND_INTERPOLATED",
    "PAGE_KIND_PHYSICAL",
    "DEFAULT_PAGE_KIND",
    "confirm_printed_labels_by_bbox",
    "derive_page_label_map",
    "detect_page_chrome",
    "enrich_page_chrome_with_bbox",
    "strip_page_chrome",
]
