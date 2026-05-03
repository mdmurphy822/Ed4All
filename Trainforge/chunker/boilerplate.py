"""
Boilerplate detector for Trainforge corpus chunks.

Detects corpus-wide repeated text spans (footers, copyright notices, template
chrome) using N-gram frequency analysis, strips them from chunk text/html,
and computes a contamination metric for quality reports.

Used by:
- process_course.py chunking stage (defensive strip before chunk write)
- quality_report.json (footer_contamination_rate metric)
- lib/leak_checker.py (corpus-wide boilerplate leak check)
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

DEFAULT_NGRAM_TOKENS = 15
DEFAULT_MIN_DOC_FRAC = 0.30
_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class BoilerplateConfig:
    """Tuning parameters for repeated-span detection."""
    min_ngram_tokens: int = DEFAULT_NGRAM_TOKENS
    min_doc_frac: float = DEFAULT_MIN_DOC_FRAC


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def _iter_ngrams(tokens: Sequence[str], n: int) -> Iterable[Tuple[str, ...]]:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def detect_repeated_ngrams(
    documents: Sequence[str],
    n: int = DEFAULT_NGRAM_TOKENS,
    min_doc_frac: float = DEFAULT_MIN_DOC_FRAC,
) -> List[str]:
    """Return verbatim n-gram spans appearing in at least min_doc_frac of documents.

    Each span is reconstructed as space-joined tokens. Only the longest maximal
    spans are returned (shorter spans contained inside a longer span are dropped)
    so callers don't waste work stripping overlapping sub-strings.

    Args:
        documents: source document texts (one per page, not per chunk).
        n: window size in tokens.
        min_doc_frac: minimum fraction of documents a span must appear in.

    Returns:
        List of verbatim span strings, longest first.
    """
    if not documents:
        return []

    threshold = max(2, int(round(len(documents) * min_doc_frac)))
    doc_sets: List[set] = []
    for doc in documents:
        tokens = _tokenize(doc)
        doc_sets.append(set(_iter_ngrams(tokens, n)))

    counts: Counter = Counter()
    for ngrams in doc_sets:
        counts.update(ngrams)

    frequent = {ng for ng, c in counts.items() if c >= threshold}
    if not frequent:
        return []

    # Merge overlapping frequent n-grams per document into maximal runs.
    maximal_spans: set = set()
    for doc in documents:
        tokens = _tokenize(doc)
        i = 0
        while i < len(tokens) - n + 1:
            if tuple(tokens[i : i + n]) in frequent:
                # Extend as far as every overlapping n-gram stays frequent.
                end = i + n
                while end < len(tokens) and tuple(tokens[end - n + 1 : end + 1]) in frequent:
                    end += 1
                maximal_spans.add(" ".join(tokens[i:end]))
                i = end
            else:
                i += 1

    return sorted(maximal_spans, key=len, reverse=True)


def strip_boilerplate(text: str, spans: Sequence[str]) -> Tuple[str, int]:
    """Remove every span from text. Tolerates whitespace drift between tokens.

    Returns (cleaned_text, number_of_span_occurrences_removed).
    """
    if not text or not spans:
        return text, 0

    removed = 0
    cleaned = text
    for span in spans:
        tokens = _tokenize(span)
        if not tokens:
            continue
        pattern = r"\s+".join(re.escape(t) for t in tokens)
        cleaned, n = re.subn(pattern, " ", cleaned)
        removed += n
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, removed


def contamination_rate(chunks: Sequence[dict], spans: Sequence[str]) -> float:
    """Fraction of chunks whose text still contains any of the given spans."""
    if not chunks or not spans:
        return 0.0
    patterns = []
    for span in spans:
        tokens = _tokenize(span)
        if tokens:
            patterns.append(re.compile(r"\s+".join(re.escape(t) for t in tokens)))
    if not patterns:
        return 0.0
    contaminated = 0
    for chunk in chunks:
        body = chunk.get("text", "") or ""
        if any(p.search(body) for p in patterns):
            contaminated += 1
    return contaminated / len(chunks)
