"""HTML to plain-text extractor based on the stdlib HTMLParser.

Replaces 2 byte-identical ``_TextExtractor`` + ``_strip_html_to_text``
copies at:

- :mod:`lib.validators.assessment_retrieval_grounding` lines 236-262.
- :mod:`lib.validators.rewrite_source_grounding` lines 134-167.

Pure stdlib. ``convert_charrefs=True`` (default in py3.5+) decodes
HTML entities (``&amp;``, ``&lt;``, etc.) so the extracted text matches
the rendered prose, not the source markup.

See plan ``plans/wave-D6-lib-utils-package-2026-05-07.md`` Section 3.5.
"""
from __future__ import annotations

import logging
from html.parser import HTMLParser
from typing import List

logger = logging.getLogger(__name__)

__all__ = ["TextExtractor", "strip_html_to_text"]


class TextExtractor(HTMLParser):
    """Stdlib HTML parser that accumulates text from data events.

    Whitespace-stripped data events are joined with a single space when
    :meth:`text` is called. Entity decoding is on by default
    (``convert_charrefs=True``).
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._fragments: List[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self._fragments.append(data)

    def text(self) -> str:
        return " ".join(self._fragments).strip()


def strip_html_to_text(html: str) -> str:
    """Strip HTML to plain text via :class:`TextExtractor`.

    Defensive: returns ``""`` on parser exceptions (debug-logged) so a
    malformed HTML fragment in a validator path doesn't crash the gate
    runner. Empty / falsy input also returns ``""``.
    """
    if not html:
        return ""
    extractor = TextExtractor()
    try:
        extractor.feed(html)
        extractor.close()
    except Exception as exc:  # noqa: BLE001 -- defensive; shape gate is upstream
        logger.debug("HTML strip raised: %s", exc)
        return ""
    return extractor.text()
