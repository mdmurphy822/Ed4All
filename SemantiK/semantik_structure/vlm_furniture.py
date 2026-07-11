"""Document-level repeated page-furniture detection for the VLM fusion path.

Defect 1 (2026-07-03 forensic audit of the first full-feature VLM-fused scan
chapter): the per-page running FOOTER (a copyright / "available for free at
<url>" line) and the page-top running HEADER ("Chapter N <Title>" +/- a page
number) are emitted by the VLM as their OWN markdown lines at the page top /
bottom. Tesseract used to garble these so they never aligned in the P1 DP
fusion and fell out; the VLM reads them CLEANLY, so they now survive the fusion
and land glued into body prose (a paragraph ending "... This <publisher> book
is available for free at http://..."). ~52% of the run's repeated-12gram excess
was this furniture.

Fix (deterministic, NO hardcoded publisher strings): detect lines whose
number-masked, whitespace-normalized signature RECURS at the page-top or
page-bottom position on >= a tunable share of the SAME document's pages, then
drop every line matching such a signature from the VLM markdown lines (and any
surviving tesseract line) BEFORE fusion. Mirrors the
``deterministic_structure._detect_repeated_chapter_furniture`` precedent (a
running header recurs many times; real content never does), but operates on the
raw VLM lines cross-page rather than on formed heading Regions.

Number masking is load-bearing: a running header "Chapter 9 Roots and Radicals
803" on one page and "... 815" on the next differ only in the page number, so
the digit runs are masked to a placeholder before the signature is compared.

The strip is ON by default whenever VLM fusion is on (a correctness fix, not an
opt-in feature) — ``SEMANTIK_VLM_FUSION`` is itself off by default, so the
flag-off (no-fusion) path is byte-identical. ``SEMANTIK_VLM_STRIP_FURNITURE=0``
is the operator escape hatch; ``SEMANTIK_VLM_FURNITURE_MIN_SHARE`` tunes the
recurrence threshold. Pure module: stdlib only.
"""

from __future__ import annotations

import math
import os
import re

__all__ = [
    "resolve_strip_furniture_mode",
    "resolve_furniture_min_share",
    "normalize_furniture_sig",
    "detect_furniture_signatures",
    "strip_furniture_lines",
]

_STRIP_ENV = "SEMANTIK_VLM_STRIP_FURNITURE"
_SHARE_ENV = "SEMANTIK_VLM_FURNITURE_MIN_SHARE"
_FALSEY = {"0", "false", "no", "off"}

# Page-top / page-bottom position bands a furniture line must appear within to
# vote (a running header is at the very top; a footer at the very bottom).
_TOP_N = 2
_BOTTOM_N = 2
# Default recurrence threshold: a signature seen at a margin on >= this SHARE of
# the document's pages is furniture. 0.35 catches running headers that alternate
# recto/verso (so appear on ~half the pages) while a varied body line never
# recurs verbatim at a margin across a third of the pages.
_DEFAULT_MIN_SHARE = 0.35
# Need at least this many pages before a "recurring" line is trustworthy.
_MIN_PAGES = 3
# Ignore trivially-short signatures (a bare "9.1" / "803") so masking a page
# number never collapses a real short section title into furniture.
_MIN_SIG_LEN = 8
# Minimum whitespace tokens for a furniture signature. A 1-2-token recurring
# margin line is far likelier a legitimate short apparatus / pedagogical label
# ("Solution", "PRACTICE TEST", "TRY IT") than page furniture — a real running
# header / footer carries a title or URL (>= 3 tokens). Anti-FP guard:
# number-masking would otherwise make a bare recurring label at a margin
# strippable ("exercise # b" == "exercise # b" across pages).
_MIN_SIG_TOKENS = 3

_DIGIT_RUN_RE = re.compile(r"\d+")
_WS_RE = re.compile(r"\s+")
# Leading markdown structural markers (heading #, blockquote >, list markers)
# stripped before comparison so "# Chapter 9 ... 803" == "Chapter 9 ... 815".
_LEADING_MARKER_RE = re.compile(r"^[#>\-*+\s]+")


def resolve_strip_furniture_mode() -> bool:
    """Whether the repeated-furniture strip runs (default ON).

    Reads ``SEMANTIK_VLM_STRIP_FURNITURE``. Only an explicit falsey value
    (``0`` / ``false`` / ``no`` / ``off``, case-insensitive) disables it; unset
    / truthy / garbage keeps it on. It is a NO-OP without VLM fusion (the caller
    only invokes it on the fusion path), so the flag-off pipeline is unchanged.
    Read at CALL time (mirrors ``resolve_vlm_fusion_mode``).
    """
    raw = os.environ.get(_STRIP_ENV)
    if raw is None:
        return True
    return str(raw).strip().lower() not in _FALSEY


def resolve_furniture_min_share() -> float:
    """Parse-with-fallback ``SEMANTIK_VLM_FURNITURE_MIN_SHARE`` (default 0.35).

    The minimum share of a document's pages a line signature must recur at a
    margin on to be furniture. Garbage / non-float / out-of-``(0, 1]`` → 0.35.
    """
    raw = os.environ.get(_SHARE_ENV)
    if not raw:
        return _DEFAULT_MIN_SHARE
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_MIN_SHARE
    if v != v or v <= 0.0 or v > 1.0:
        return _DEFAULT_MIN_SHARE
    return v


def normalize_furniture_sig(line: str) -> str:
    """Number-masked, whitespace-normalized signature for a furniture line.

    Collapses whitespace, lowercases, strips leading markdown structural
    markers, and masks every digit run to ``#`` so "Chapter 9 ... 803" and
    "Chapter 9 ... 815" share one signature.
    """
    s = _WS_RE.sub(" ", (line or "").strip()).lower()
    s = _LEADING_MARKER_RE.sub("", s)
    s = _DIGIT_RUN_RE.sub("#", s)
    return s.strip()


def detect_furniture_signatures(
    pages_lines: list[list[str]],
    *,
    min_share: float | None = None,
    top_n: int = _TOP_N,
    bottom_n: int = _BOTTOM_N,
) -> set[str]:
    """Signatures that recur at a page margin on >= ``min_share`` of pages.

    ``pages_lines`` is one list of raw VLM markdown lines per page (document
    order). A signature votes at most ONCE per page, and only when it appears
    within the first ``top_n`` OR last ``bottom_n`` line positions of that page.
    Returns the set of number-masked signatures crossing the threshold. Empty
    for a doc under ``_MIN_PAGES`` pages (a "recurrence" is untrustworthy).
    """
    if min_share is None:
        min_share = resolve_furniture_min_share()
    pages = [list(p or []) for p in pages_lines]
    n_pages = len(pages)
    if n_pages < _MIN_PAGES:
        return set()

    counts: dict[str, int] = {}
    for lines in pages:
        m = len(lines)
        if m == 0:
            continue
        positions = set(range(0, min(top_n, m))) | set(range(max(0, m - bottom_n), m))
        seen_this_page: set[str] = set()
        for i in positions:
            sig = normalize_furniture_sig(lines[i])
            if len(sig) < _MIN_SIG_LEN or len(sig.split()) < _MIN_SIG_TOKENS:
                continue
            seen_this_page.add(sig)
        for sig in seen_this_page:
            counts[sig] = counts.get(sig, 0) + 1

    threshold = max(2, math.ceil(min_share * n_pages))
    return {sig for sig, c in counts.items() if c >= threshold}


def strip_furniture_lines(lines: list[str], signatures: set[str]) -> list[str]:
    """Drop every whole line whose signature is in ``signatures`` (anywhere).

    A confirmed furniture signature is a specific whole-line text repeated at
    the margins across the document, so a whole-line match anywhere on the page
    is itself furniture — safe to drop regardless of position.
    """
    if not signatures:
        return list(lines)
    out: list[str] = []
    for ln in lines:
        sig = normalize_furniture_sig(ln)
        if sig and sig in signatures:
            continue
        out.append(ln)
    return out
