"""WCAG 2.2 AA CSS bundle for the SemantiK gold-standard accessible shell.

Self-contained, SemantiK-owned (Apache-2.0) vendored copy of the gold
accessible-component stylesheet. Phase 1 of the gold-standard
semantic-class plan vendors the *data* only — nothing imports this module
yet (the assembler reads it in a later phase gated behind
``SEMANTIK_GOLD_SHELL``), so the flag-off / unconsumed path is byte-stable.

``WCAG22_CSS`` is the UNION of two gold-standard CSS vocabularies that were
historically split across two template files:

  * the inline ``<style>`` block of the gold reference document — the
    component-class vocabulary (``.definition`` / ``.algorithm`` /
    ``.abstract`` / ``.toc`` / ``.references`` / ``.figure`` /
    ``.figure-caption`` / ``.pdf-download`` / ``.authors`` / ``.metadata`` /
    ``.skip-link`` + the ``<table>``/``th``/``td``/``caption`` rules + the
    ``:root`` design-token block), and
  * the WCAG-22 base bundle — the accessibility-primitive vocabulary
    (``.sr-only`` / typed ``.callout``-``info``/``-warning``/``-tip``/
    ``-danger`` / ``aside.pullquote`` / ``.page-break`` /
    ``pre[role=region]`` / ``aside[role=complementary]``).

The dark-mode, reduced-motion, ``max-width: 600px``, and ``@media print``
queries from both sources are preserved (merged into one block per
``@media`` to avoid duplicate rule blocks).

Net-new rules (NOT present in either source — the ONLY additions this
phase): ``.exercise`` and ``.objectives`` region boxes, styled
consistently with ``.definition`` / ``.abstract`` so the Phase-2 catalog's
``css_class`` ↔ CSS coverage test passes and the Phase-7 exercise /
objectives wrappers render.

Two reconciliation decisions, documented inline below:

  * **Table-role divergence.** The two sources disagreed on the table
    selector (``table[role="grid"]`` vs a bare ``<table>``). We pick the
    BARE ``table`` convention — a single ``table`` / ``th`` / ``td`` /
    ``caption`` rule set that styles both a bare ``<table>`` (what the
    existing ``fallbacks.fallback_table`` emits) and any ``role="grid"``
    table. We deliberately ship ONE table convention, not two conflicting
    rules.
  * **Design tokens.** We KEEP the gold ``--bg-color`` / ``--text-color`` /
    ``--heading-color`` / … token names VERBATIM rather than forking them
    toward the Ed4All ``--cf-*`` convention. The fork to ``--cf-*`` is a
    deliberate non-goal this phase (lowest-risk verbatim copy); this comment
    documents the choice so the divergence from the Ed4All token vocabulary
    is intentional and visible, not silent.
"""

from __future__ import annotations

WCAG22_CSS: str = """
  /* === Design tokens (gold :root — kept VERBATIM, not forked to --cf-*) === */
  :root {
    --bg-color: #ffffff;
    --text-color: #1a1a1a;
    --heading-color: #0d0d0d;
    --link-color: #0066cc;
    --link-hover: #004499;
    --border-color: #e0e0e0;
    --code-bg: #f5f5f5;
    --table-header-bg: #f0f0f0;
    --blockquote-bg: #f9f9f9;
    --blockquote-border: #0066cc;
  }

  * { box-sizing: border-box; }

  /* === Base typography === */
  body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.8;
    color: var(--text-color);
    background-color: var(--bg-color);
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem;
    font-size: 16px;
  }

  h1, h2, h3, h4, h5, h6 {
    color: var(--heading-color);
    margin-top: 2rem;
    margin-bottom: 1rem;
    line-height: 1.3;
    scroll-margin-top: 80px; /* WCAG 2.4.11 Focus Not Obscured for anchor links */
  }

  h1 {
    font-size: 2.2rem;
    text-align: center;
    border-bottom: 3px solid var(--border-color);
    padding-bottom: 1rem;
  }

  h2 {
    font-size: 1.8rem;
    border-bottom: 2px solid var(--border-color);
    padding-bottom: 0.5rem;
  }

  h3 { font-size: 1.4rem; }
  h4 { font-size: 1.2rem; }

  section { margin-bottom: 1.5em; }

  p {
    margin-bottom: 1rem;
    text-align: justify;
  }

  a {
    color: var(--link-color);
    text-decoration: none;
  }

  a:hover, a:focus {
    color: var(--link-hover);
    text-decoration: underline;
  }

  /* === WCAG 1.4.13 screen-reader-only label helper === */
  .sr-only { position: absolute; left: -9999px; width: 1px; height: 1px; overflow: hidden; }

  /* === WCAG 2.4.7 / 2.4.13 visible focus indicator (3px outline) === */
  :focus { outline: 3px solid var(--link-color); outline-offset: 2px; }
  :focus:not(:focus-visible) { outline: none; }
  :focus-visible { outline: 3px solid var(--link-color); outline-offset: 2px; }

  /* === WCAG 2.4.11/2.4.12 Focus Not Obscured === */
  :target, :focus { scroll-margin-top: 80px; }
  *:focus { scroll-margin-top: 80px; scroll-margin-bottom: 80px; }

  /* === WCAG 2.5.8 Target Size (Minimum) 24x24 CSS px === */
  a:not(p a):not(li a):not(td a):not(figcaption a),
  button,
  [role="button"],
  summary {
    min-height: 24px;
    min-width: 24px;
  }

  /* Inline links exempt per WCAG 2.5.8 exception */
  p a, li a, td a, figcaption a, span a {
    min-height: auto;
    min-width: auto;
  }

  /* === Skip link === */
  .skip-link {
    position: absolute;
    top: -40px;
    left: 0;
    background: var(--link-color);
    color: white;
    padding: 8px;
    z-index: 100;
  }

  .skip-link:focus { top: 0; }

  /* === PDF-download chrome === */
  .pdf-download {
    background-color: var(--code-bg);
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    text-align: center;
    border: 1px solid var(--border-color);
  }

  .pdf-download a { font-weight: bold; font-size: 1.1rem; }

  /* === Author / metadata chrome === */
  .authors { text-align: center; font-size: 1.1rem; margin-bottom: 2rem; }

  .metadata {
    font-size: 0.9rem;
    color: var(--text-color);
    opacity: 0.8;
    text-align: center;
    margin-bottom: 2rem;
  }

  /* === Metadata / complementary aside === */
  aside[role="complementary"] { border-left: 3px solid #888; padding-left: 1em; margin-top: 2em; font-size: 0.95em; }

  /* === Abstract / overview region === */
  .abstract {
    background-color: var(--blockquote-bg);
    padding: 1.5rem;
    border-left: 4px solid var(--blockquote-border);
    margin: 2rem 0;
    border-radius: 0 8px 8px 0;
  }

  .abstract h2 { margin-top: 0; border-bottom: none; }

  /* === Definition region === */
  .definition {
    background-color: var(--blockquote-bg);
    padding: 1.5rem;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    margin: 1.5rem 0;
  }

  /* === Algorithm / worked-example monospace region === */
  .algorithm {
    background-color: var(--code-bg);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 1.5rem 0;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    overflow-x: auto;
  }

  .algorithm h4 { margin-top: 0; }

  /* === Net-new (Phase 1): exercise region — boxed like .definition === */
  .exercise {
    background-color: var(--blockquote-bg);
    padding: 1.5rem;
    border: 2px solid var(--border-color);
    border-radius: 8px;
    margin: 1.5rem 0;
  }

  .exercise h4 { margin-top: 0; }

  /* === Net-new (Phase 1): objectives region — left-bar like .abstract === */
  .objectives {
    background-color: var(--blockquote-bg);
    padding: 1.5rem;
    border-left: 4px solid var(--blockquote-border);
    margin: 1.5rem 0;
    border-radius: 0 8px 8px 0;
  }

  .objectives h3 { margin-top: 0; border-bottom: none; }

  /* === Typed callout variants (base shared + color per severity) === */
  .callout { padding: 1rem; border-left: 4px solid; margin: 1em 0; }
  .callout h4 { margin-top: 0; }
  .callout-info { border-color: #0066cc; background: #e8f0fe; }
  .callout-warning { border-color: #b58900; background: #fff8e1; }
  .callout-tip { border-color: #2e7d32; background: #e8f5e9; }
  .callout-danger { border-color: #c62828; background: #ffebee; }

  /* === Pullquote + page-break marker === */
  aside.pullquote { border: none; font-size: 1.2em; font-style: italic; max-width: 30em; margin: 1.5em auto; }
  .page-break { display: block; height: 0; border-top: 1px dashed #bbb; margin: 1.5em 0; }

  /* === Figure === */
  figure { margin: 1em 0; }
  .figure { margin: 2rem 0; text-align: center; }
  figcaption { font-size: 0.9em; color: #555; margin-top: 0.3em; }
  .figure-caption { font-style: italic; margin-top: 0.5rem; font-size: 0.95rem; }

  /* === Table (reconciled: ONE bare-`table` convention covering both a bare
     <table> and any role=grid table — the role-scoped selector is dropped so
     a single rule set styles every table the assembler emits) === */
  table {
    width: 100%;
    border-collapse: collapse;
    margin: 1.5rem 0;
    font-size: 0.95rem;
  }

  th, td {
    border: 1px solid var(--border-color);
    padding: 0.75rem;
    text-align: left;
    vertical-align: top;
  }

  th { background-color: var(--table-header-bg); font-weight: bold; }

  caption {
    caption-side: bottom;
    padding: 0.75rem;
    font-style: italic;
    text-align: left;
  }

  /* === Code listing region === */
  pre[role="region"] { background: #f4f4f4; padding: 0.8em; overflow-x: auto; }

  /* === Lists === */
  ul, ol { margin-bottom: 1rem; padding-left: 2rem; }
  li { margin-bottom: 0.5rem; }

  /* === References === */
  .references { font-size: 0.9rem; }
  .references ol { padding-left: 2rem; }
  .references li { margin-bottom: 0.75rem; }

  /* === Table of contents nav === */
  .toc {
    background-color: var(--blockquote-bg);
    padding: 1.5rem;
    border-radius: 8px;
    margin: 2rem 0;
  }

  .toc h2 { margin-top: 0; border-bottom: none; }
  .toc ul { list-style-type: none; padding-left: 0; }
  .toc li { margin-bottom: 0.3rem; }
  .toc a { text-decoration: none; }

  /* === Motion preferences (merged from both sources) === */
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.01ms !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.01ms !important;
      scroll-behavior: auto !important;
    }
    html { scroll-behavior: auto !important; }
  }

  /* === Color-scheme preferences (merged: gold :root tokens + callout/pre) === */
  @media (prefers-color-scheme: dark) {
    :root {
      --bg-color: #1a1a1a;
      --text-color: #e0e0e0;
      --heading-color: #ffffff;
      --link-color: #66b3ff;
      --link-hover: #99ccff;
      --border-color: #404040;
      --code-bg: #2d2d2d;
      --table-header-bg: #2d2d2d;
      --blockquote-bg: #252525;
      --blockquote-border: #66b3ff;
    }
    aside[role="complementary"] { border-color: #666; }
    .callout-info { background: #0e2a4a; }
    .callout-warning { background: #3a2f00; }
    .callout-tip { background: #16331a; }
    .callout-danger { background: #3a0f12; }
    pre[role="region"] { background: #262626; color: #e0e0e0; }
    :focus, :focus-visible { outline-color: #66b3ff; }
  }

  /* === Small-screen layout === */
  @media (max-width: 600px) {
    body { padding: 1rem; font-size: 15px; }
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.5rem; }
    table { font-size: 0.85rem; }
    th, td { padding: 0.5rem; }
  }

  /* === Print === */
  @media print {
    body { max-width: none; padding: 0; }
    .pdf-download, .skip-link { display: none; }
  }
"""


__all__ = ("WCAG22_CSS",)
