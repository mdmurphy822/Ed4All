"""Text-preservation check for Stage 7 hard gate.

Strips HTML to visible text and compares against the source FeatureBlock
text via stdlib SequenceMatcher.ratio. Catches truncation and
hallucinated content; doesn't try to detect paraphrasing (that's the
soft reranker's job at Stage 8).

Public API
----------
* :func:`extract_visible_text` — HTMLParser-based stripper.
* :func:`normalize` — collapse whitespace + lowercase.
* :func:`levenshtein_ratio` — wrapper around stdlib SequenceMatcher.
* :func:`text_preservation_ratio` — high-level helper used by the gate.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from html.parser import HTMLParser

_WS_RE = re.compile(r"\s+")


class _TextExtractor(HTMLParser):
    """Concat all visible text data from an HTML fragment.

    ``convert_charrefs=True`` (the default since 3.5) means &amp; et al
    arrive in :meth:`handle_data` as their resolved characters, so the
    stripped text matches what a human would read.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._chunks.append(data)

    @property
    def text(self) -> str:
        return " ".join(self._chunks)


def extract_visible_text(html: str) -> str:
    """Return the visible text content of an HTML fragment.

    Errors during parsing return ``""`` rather than raising — a
    parse-failure HTML fragment will fail the html5-wellformedness
    check anyway, so we don't need to double-fail it here.
    """
    p = _TextExtractor()
    try:
        p.feed(html)
        p.close()
    except Exception:
        return ""
    return p.text


def normalize(s: str) -> str:
    """Collapse runs of whitespace, strip, and lowercase."""
    return _WS_RE.sub(" ", s.strip().lower())


def levenshtein_ratio(a: str, b: str) -> float:
    """Return :class:`SequenceMatcher`'s ratio (0..1).

    Despite the name (kept for the spec contract), this is *not*
    classical Levenshtein distance — it's Ratcliff/Obershelp via
    stdlib ``difflib``. Behaves the same way for the gate's purposes:
    higher = more shared content.
    """
    return SequenceMatcher(None, a, b).ratio()


def text_preservation_ratio(source_text: str, candidate_html: str) -> float:
    """Return ratio in ``[0, 1]``. Higher = more text preserved."""
    return levenshtein_ratio(
        normalize(source_text),
        normalize(extract_visible_text(candidate_html)),
    )


__all__ = [
    "extract_visible_text",
    "normalize",
    "levenshtein_ratio",
    "text_preservation_ratio",
]
