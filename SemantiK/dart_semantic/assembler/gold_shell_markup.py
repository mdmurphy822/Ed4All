"""Gold-standard accessible document-shell markup constants (data only).

Self-contained, SemantiK-owned (Apache-2.0) vendored copy of the gold
reference document's ``<head>`` meta, schema.org accessibility JSON-LD, and
``<body>`` landmark skeleton. Phase 1 vendors the *data* only — NOTHING
imports this module yet; the assembler splices these constants in a later
phase gated behind ``SEMANTIK_GOLD_SHELL``, so the unconsumed path is
byte-stable.

Constants
---------
``GOLD_HEAD_META``
    The viewport meta tag (charset + title are emitted by the shell's
    ``.format()`` slots; this is the extra meta the gold ``<head>`` carries).
``GOLD_ACCESSIBILITY_JSONLD``
    The ``<script type="application/ld+json">`` schema.org accessibility
    block, mirroring the gold reference template's advertised feature set:
    ``accessibilityFeature: [alternativeText, readingOrder,
    structuralNavigation, tableOfContents, ARIA]``, ``accessibilityHazard:
    none``, ``accessMode``, ``accessibilitySummary``. Generic ``CreativeWork``
    typing (the gold template's sample-specific author/name fields are
    dropped — those are placeholder artifacts, not part of the accessibility
    contract).
``GOLD_DOC_OPEN`` / ``GOLD_DOC_CLOSE``
    The document skeleton. ``GOLD_DOC_OPEN`` keeps the EXACT ``.format()``
    placeholder contract of ``shell.DOC_OPEN`` — ``{lang}`` + ``{title}`` and
    NO other unescaped braces — so Phase 8 can swap it in via the same
    ``DOC_OPEN.format(lang=..., title=...)`` call. The ``<style>`` bundle and
    the JSON-LD block are spliced in Phase 8 at the brace-free
    ``GOLD_STYLE_SLOT`` / ``GOLD_JSONLD_SLOT`` sentinels (kept brace-free so
    they never collide with ``str.format``). The ``DART_TITLE_SLOT`` sentinel
    is preserved (mirrors ``shell.TITLE_SLOT_SENTINEL``) so Stage-9c title
    gap-fill lands inside ``<header>``.
"""

from __future__ import annotations


# Brace-free splice sentinels (Phase 8 replaces these — kept free of ``{}`` so
# ``GOLD_DOC_OPEN.format(...)`` never trips over them).
GOLD_STYLE_SLOT = "<!-- GOLD_STYLE_SLOT -->"
GOLD_JSONLD_SLOT = "<!-- GOLD_JSONLD_SLOT -->"
GOLD_TITLE_SLOT = "<!-- DART_TITLE_SLOT -->"


GOLD_HEAD_META = '<meta name="viewport" content="width=device-width, initial-scale=1.0">'


GOLD_ACCESSIBILITY_JSONLD = """<script type="application/ld+json">
{
    "@context": "https://schema.org",
    "@type": "CreativeWork",
    "inLanguage": "en",
    "accessibilityFeature": [
        "alternativeText", "readingOrder", "structuralNavigation",
        "tableOfContents", "ARIA"
    ],
    "accessibilityHazard": "none",
    "accessMode": ["textual", "visual"],
    "accessibilitySummary": "Semantic HTML5 with ARIA landmarks, skip link, heading hierarchy, scoped tables, and reduced-motion/dark-mode support."
}
</script>"""


# Same ``.format(lang=..., title=...)`` contract as ``shell.DOC_OPEN``.
# ONLY ``{lang}`` and ``{title}`` are brace-bearing; every other slot is a
# brace-free HTML comment sentinel spliced in Phase 8.
GOLD_DOC_OPEN = (
    '<!DOCTYPE html>\n'
    '<html lang="{lang}">\n'
    '<head>\n'
    '<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
    '<title>{title}</title>\n'
    '<!-- GOLD_STYLE_SLOT -->\n'
    '<!-- GOLD_JSONLD_SLOT -->\n'
    '</head>\n'
    '<body>\n'
    '<a class="skip-link" href="#main-content">Skip to main content</a>\n'
    '<main id="main-content" role="main">\n'
    '<article>\n'
    '<header>\n'
    '<!-- DART_TITLE_SLOT -->\n'
    '</header>\n'
)


GOLD_DOC_CLOSE = (
    '<footer role="contentinfo">\n'
    '</footer>\n'
    '</article>\n'
    '</main>\n'
    '</body>\n'
    '</html>\n'
)


__all__ = (
    "GOLD_ACCESSIBILITY_JSONLD",
    "GOLD_DOC_CLOSE",
    "GOLD_DOC_OPEN",
    "GOLD_HEAD_META",
    "GOLD_JSONLD_SLOT",
    "GOLD_STYLE_SLOT",
    "GOLD_TITLE_SLOT",
)
