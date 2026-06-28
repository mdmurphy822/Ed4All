"""Document shell template (DOCTYPE, <html lang>, <head>, <main>).

Plans/04 §1.5 — fixed shell:
  * <!DOCTYPE html>
  * <meta charset="utf-8">
  * <title>...</title>
  * <a class="skip-link" href="#main-content">Skip to main content</a>
  * <main id="main-content">

The ``<!-- DART_TITLE_SLOT -->`` comment immediately inside ``<main>`` is
the splice target Stage 9c uses when a ``missing_title`` GapSlot is
filled (the gap-fill output may carry a fresh ``<h1>`` that needs to
land before any other body content).
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from .gold_shell_markup import (
    GOLD_ACCESSIBILITY_JSONLD,
    GOLD_DOC_CLOSE,
    GOLD_DOC_OPEN,
    GOLD_JSONLD_SLOT,
    GOLD_STYLE_SLOT,
)
from .wcag22_css import WCAG22_CSS


# Truthy tokens for the parse-with-fallback gate below. Replicated inline
# (the assembler has no shared frozenset) to match the reviewer's
# ``reviewer._TRUTHY`` set exactly — keep the two in sync.
_TRUTHY = frozenset({"1", "true", "yes", "on"})


def resolve_gold_shell_mode() -> bool:
    """Return True when the assembler emits the gold ARIA markup + shell.

    Reads ``SEMANTIK_GOLD_SHELL``. Default OFF (unset / blank / falsey /
    garbage) -> byte-identical to today (mirrors the reviewer's
    ``resolve_block_review_mode`` parse-with-fallback posture). A truthy
    value (``1``/``true``/``yes``/``on``, case-insensitive) gates the gold
    per-region ARIA wrap + gold document shell/CSS. A pure read until Phase
    7+ consumes it.
    """
    raw = (os.environ.get("SEMANTIK_GOLD_SHELL") or "").strip().lower()
    return raw in _TRUTHY


DOC_OPEN = (
    '<!DOCTYPE html>\n'
    '<html lang="{lang}">\n'
    '<head>\n'
    '<meta charset="utf-8">\n'
    '<title>{title}</title>\n'
    '</head>\n'
    '<body>\n'
    '<a class="skip-link" href="#main-content">Skip to main content</a>\n'
    '<main id="main-content">\n'
    '<!-- DART_TITLE_SLOT -->\n'
)
DOC_CLOSE = '</main>\n</body>\n</html>\n'

TITLE_SLOT_SENTINEL = '<!-- DART_TITLE_SLOT -->\n'


def build_shell(
    *,
    lang: str,
    title: str,
    fabricated_title: bool = False,
) -> tuple[str, str]:
    """Return ``(doc_open, doc_close)`` for the document shell.

    When ``fabricated_title=True``, inject a
    ``<meta name="dart-title-source" content="fallback">`` immediately
    after the ``<title>`` tag so any consumer of ``assembled.html`` can
    see at a glance that the title was synthesised by the assembler
    (no doc_role:title region, no input h1, no Qwen gap-fill survivor)
    rather than extracted from the source document. Default ``False``
    keeps backward compatibility with existing callers.
    """
    if resolve_gold_shell_mode():
        # Gold accessible shell branch — returns the NEW gold constants
        # (authored in ``gold_shell_markup``), NEVER mutating the existing
        # minimal ``DOC_OPEN``/``DOC_CLOSE`` constants in place. ``GOLD_DOC_OPEN``
        # keeps the EXACT ``{lang}``/``{title}`` ``.format()`` contract, so the
        # call below is byte-for-byte the same shape as the minimal branch.
        doc_open = GOLD_DOC_OPEN.format(
            lang=_escape_attr(lang), title=_escape(title),
        )
        # Splice the WCAG CSS + schema.org JSON-LD at the brace-free sentinels
        # via ``str.replace`` (NOT ``.format`` — both payloads carry literal
        # ``{}`` that would break ``str.format``).
        doc_open = doc_open.replace(
            GOLD_STYLE_SLOT, f"<style>{WCAG22_CSS}</style>", 1,
        )
        doc_open = doc_open.replace(
            GOLD_JSONLD_SLOT, GOLD_ACCESSIBILITY_JSONLD, 1,
        )
        doc_close = GOLD_DOC_CLOSE
    else:
        doc_open = DOC_OPEN.format(
            lang=_escape_attr(lang), title=_escape(title),
        )
        doc_close = DOC_CLOSE

    if fabricated_title:
        doc_open = doc_open.replace(
            f"<title>{_escape(title)}</title>\n",
            (
                f"<title>{_escape(title)}</title>\n"
                '<meta name="dart-title-source" content="fallback">\n'
            ),
            1,
        )
    return (doc_open, doc_close)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _escape_attr(s: str) -> str:
    return _escape(s)


_LANG_ALLOWLIST = frozenset({"en", "es", "fr", "de", "zh", "ja"})


def detect_language(feature_blocks: Sequence[Any]) -> str:
    """Use lingua to detect doc language; return BCP-47 tag. Default 'en'.

    Mirrors the cascade in ``dart_semantic/enrich.py:69-94`` (a single
    cached detector, ``with_low_accuracy_mode`` so no model download
    blocks the smoke harness). Returns ``"en"`` on any failure mode —
    language wrong is a low-cost mistake (Plans/04 §1.5 rationale).
    """
    try:
        from lingua import LanguageDetectorBuilder
    except ImportError:
        return "en"
    text_parts: list[str] = []
    for fb in feature_blocks[:200]:
        raw = getattr(fb, "raw", None)
        text = getattr(raw, "text", None) if raw is not None else None
        if text:
            text_parts.append(text)
    text = " ".join(text_parts)[:8000]
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 200:
        return "en"
    try:
        detector = (
            LanguageDetectorBuilder.from_all_languages()
            .with_low_accuracy_mode()
            .build()
        )
        lang = detector.detect_language_of(text)
    except Exception:
        return "en"
    if lang is None:
        return "en"
    try:
        code = lang.iso_code_639_1.name.lower()
    except Exception:
        return "en"
    return code if code in _LANG_ALLOWLIST else "en"


__all__ = [
    "DOC_CLOSE",
    "DOC_OPEN",
    "TITLE_SLOT_SENTINEL",
    "build_shell",
    "detect_language",
    "resolve_gold_shell_mode",
]
