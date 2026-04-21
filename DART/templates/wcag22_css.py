"""WCAG 2.2 AA CSS bundle injected by the DART converter.

Single source of truth for the base styles emitted into the ``<style>``
block of every converter-produced document. Kept in its own module so
the assembler can import it without bloating ``document_assembler.py``
and so template tests can snapshot individual rules.

The bundle covers:

    * WCAG 2.4.7 — visible focus ring (3px outline + 2px offset)
    * WCAG 2.4.11/12 — ``scroll-margin-top`` so focused targets aren't
      obscured by sticky headers
    * WCAG 2.5.8 — 24px minimum target size on interactive elements
    * WCAG 1.4.13 — sr-only helper for screen-reader-only labels used
      by callouts and page-breaks
    * ``prefers-color-scheme: dark`` palette override
    * ``prefers-reduced-motion: reduce`` animation cutout
    * Callout variant styling for info / warning / tip / danger so the
      DPUB-ARIA roles map to visually distinct panels

Wave 15 may promote this to a data-theme / CSS-custom-properties
approach; for Wave 13 the inline string is intentionally simple so the
template tests can assert substring presence.
"""

from __future__ import annotations

WCAG22_CSS: str = """
  /* --- Base typography --- */
  body { font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; max-width: 50em; margin: 0 auto; padding: 1em; color: #1a1a1a; background: #ffffff; }
  h1 { font-size: 2em; border-bottom: 2px solid #333; padding-bottom: 0.3em; }
  h2 { font-size: 1.5em; margin-top: 2em; border-bottom: 1px solid #ccc; padding-bottom: 0.2em; }
  h3 { font-size: 1.25em; margin-top: 1.5em; }
  section { margin-bottom: 1.5em; }
  p { margin: 0.8em 0; }

  /* --- WCAG 1.4.13 screen-reader-only label helper --- */
  .sr-only { position: absolute; left: -9999px; width: 1px; height: 1px; overflow: hidden; }

  /* --- WCAG 2.4.11/12 scroll-margin-top so focused targets aren't obscured --- */
  :target, :focus { scroll-margin-top: 80px; }

  /* --- WCAG 2.4.7 visible focus indicator --- */
  :focus { outline: 3px solid #0066cc; outline-offset: 2px; }
  :focus:not(:focus-visible) { outline: none; }
  :focus-visible { outline: 3px solid #0066cc; outline-offset: 2px; }

  /* --- WCAG 2.5.8 minimum target size (24x24 CSS px) --- */
  :is(a, button, [role="button"]) { min-height: 24px; min-width: 24px; }

  /* --- Skip link --- */
  .skip-link { position: absolute; left: -9999px; top: auto; width: 1px; height: 1px; overflow: hidden; }
  .skip-link:focus { position: static; width: auto; height: auto; }

  /* --- Metadata aside --- */
  aside[role="complementary"] { border-left: 3px solid #888; padding-left: 1em; margin-top: 2em; font-size: 0.95em; }

  /* --- Callout variants (base shared + color per severity) --- */
  .callout { padding: 1rem; border-left: 4px solid; margin: 1em 0; }
  .callout h4 { margin-top: 0; }
  .callout-info { border-color: #0066cc; background: #e8f0fe; }
  .callout-warning { border-color: #b58900; background: #fff8e1; }
  .callout-tip { border-color: #2e7d32; background: #e8f5e9; }
  .callout-danger { border-color: #c62828; background: #ffebee; }

  /* --- Pullquote + page-break marker --- */
  aside.pullquote { border: none; font-size: 1.2em; font-style: italic; max-width: 30em; margin: 1.5em auto; }
  .page-break { display: block; height: 0; border-top: 1px dashed #bbb; margin: 1.5em 0; }

  /* --- Figure + table + code --- */
  figure { margin: 1em 0; }
  figcaption { font-size: 0.9em; color: #555; margin-top: 0.3em; }
  table[role="grid"] { border-collapse: collapse; width: 100%; }
  table[role="grid"] caption { font-weight: bold; text-align: left; margin-bottom: 0.3em; }
  table[role="grid"] th, table[role="grid"] td { border: 1px solid #bbb; padding: 0.4em 0.6em; text-align: left; }
  pre[role="region"] { background: #f4f4f4; padding: 0.8em; overflow-x: auto; }

  /* --- Motion preferences --- */
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; animation-iteration-count: 1 !important; transition-duration: 0.01ms !important; scroll-behavior: auto !important; }
  }

  /* --- Color-scheme preferences --- */
  @media (prefers-color-scheme: dark) {
    body { background: #1a1a1a; color: #e0e0e0; }
    h1, h2 { border-color: #555; }
    aside[role="complementary"] { border-color: #666; }
    .callout-info { background: #0e2a4a; }
    .callout-warning { background: #3a2f00; }
    .callout-tip { background: #16331a; }
    .callout-danger { background: #3a0f12; }
    pre[role="region"] { background: #262626; color: #e0e0e0; }
    :focus, :focus-visible { outline-color: #66b3ff; }
  }
"""


__all__ = ("WCAG22_CSS",)
