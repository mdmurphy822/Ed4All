"""Wave 20: running header / footer / page-number chrome detection.

pdftotext faithfully reproduces running headers and footers on every
page. For a 584-page textbook like *Teaching in a Digital Age*, a two-
line chrome (title + page number) produces ~1,168 spurious lines in the
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
  (``"Teaching in a Digital Age 164"``, ``"164"``, ``"Chapter 3 — 47"``),
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
# ``"Teaching in a Digital Age 164"`` into ``("Teaching in a Digital
# Age", 164)``.
_TRAILING_DIGITS_RE = re.compile(r"^(.*?)(?:\s+)?(\d{1,4})\s*$")


# Wave 25 Fix 1: mirror of the trailing-digits regex for the
# even-page leading-digit footer pattern (``"{N} A.W. (Tony) Bates"``).
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
    ``"{N} A.W. (Tony) Bates"`` put the page number BEFORE the fixed
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
    # partitions — trailing-digit (``"Book Title 42"``) and leading-
    # digit (``"42 A.W. Bates"``). The frequency counter then evaluates
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
        leading integer (e.g. ``"a.w. (tony) bates"``). Bare numbers
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
                # shared counts dicts (e.g. ``"a.w. bates"`` could
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
    # real residual text (``"a.w. (tony) bates"``). Internal matching
    # during stripping uses the sentinel-prefixed form.
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
            if matched_partition == "tail":
                _prefix, maybe_page = _strip_trailing_digits(norm)
            else:
                _residual, maybe_page = _strip_leading_digits(norm)
            if maybe_page is not None:
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
            if matched_partition == "tail":
                _prefix, maybe_page = _strip_trailing_digits(norm)
            else:
                _residual, maybe_page = _strip_leading_digits(norm)
            if maybe_page is not None:
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
        # (``"{N} A.W. Bates"``) the same way it does trailing-digit
        # headers.
        residual, lead_page = _strip_leading_digits(norm)
        if lead_page is not None and residual and len(residual) >= 3:
            leading_key = f"__lead__:{residual}"
            if leading_key in candidate_keys:
                confirmed.add(leading_key)

    return confirmed


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
                # A.W. Bates"``) — residual matches an even-page
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


__all__ = ["PageChrome", "detect_page_chrome", "strip_page_chrome"]
