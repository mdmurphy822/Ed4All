#!/usr/bin/env python3
r"""Assemble a learner-facing end-user HTML page from a SemantiK-converted
document + the Courseforge accessible template.

Motivation
----------
The SemantiK adapter (``lib/semantik/adapter.py``) emits a COMPLETE standalone
accessible document — its own ``<!DOCTYPE>`` / ``<html>`` / ``<body>`` /
skip-link / ``<main id="main-content">`` / ``<h1>``. A learner page wraps that
content in the Courseforge WCAG template (breadcrumb, header/footer landmarks,
navigation) + a MathJax runtime.

The ad-hoc assembler that predated this script INJECTED THE WHOLE STANDALONE
DOCUMENT into the template's ``<main>``, yielding the end-user-HTML audit's
BLOCKER defect (``plans/enduser-html-improvements-2026-07.md`` §B1): two
``<main>`` landmarks, duplicate ``id="main-content"``, two "Skip to main
content" links, two ``<h1>``s. This script is the canonical fix — it extracts
only the adapter's INNER content (the ``<article>`` chapters, dropping the
adapter's document chrome + its ``<h1>``) and injects that into the template's
single ``<main>``, so the assembled page has exactly one ``<main>`` / one
skip-link / one ``<h1>`` / one breadcrumb.

It also lands two more §Top-5 fixes at the packaging seam:
  * B2 — MathJax v3 configured with the ``a11y/assistive-mml`` extension so
    screen readers get real MathML for every ``$…$`` run (not "dollar backslash
    sqrt").
  * B5/item-6 — an optional ``--css-hooks`` flag inlines the new
    ``Courseforge/templates/accessibility/semantik_content.css`` so the textbook
    semantics (examples, solutions, TRY-IT / BE-PREPARED / HOW-TO openers,
    key-terms glossaries, responsive tables) render with visual affordance.

Arg-driven + generic: NO course paths are hardcoded. ``--output`` is REQUIRED
so the script never writes next to a live conversion output by default.

Example
-------
    python scripts/assemble_enduser_page.py \
        --content /tmp/rerender/foo-ch09_accessible.html \
        --template Courseforge/templates/accessibility/accessible_content_template.html \
        --title "Roots and Radicals" --course-name "Elementary Algebra" \
        --module-name "Chapter 9" --css-hooks \
        --output /tmp/enduser/foo-ch09_enduser.html
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Default template + stylesheet locations (still overridable via CLI flags).
_DEFAULT_TEMPLATE = (
    _REPO_ROOT
    / "Courseforge/templates/accessibility/accessible_content_template.html"
)
_DEFAULT_DART_CSS = (
    _REPO_ROOT / "Courseforge/templates/accessibility/semantik_content.css"
)

# MathJax v3 config WITH assistive-MML (B2): injects visually-hidden MathML
# alongside CHTML so a screen reader reads real math. Self-contained script tag
# pair; the CDN load matches the template's existing Bootstrap CDN posture.
_MATHJAX_HEAD = """<!-- MathJax v3 with assistive-MML (a11y) -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    // Round-7b — honor the adapter's HTML-only currency escape: a preserved
    // currency ``$`` is emitted as ``\\$`` so two amounts in one paragraph
    // ("costs $5 ... and $3") never FALSE-PAIR into an italic inline-math span.
    // processEscapes renders the ``\\$`` as a literal dollar sign.
    processEscapes: true
  },
  loader: { load: ['a11y/assistive-mml'] },
  options: {
    enableMenu: true,
    menuOptions: { settings: { assistiveMml: true } },
    // Never typeset the page chrome (breadcrumb / header / TOC nav). The
    // course crumb "Course Title (course-slug-01)" was rendering as math
    // italics; ignoring these regions keeps them plain text regardless of any
    // stray/unbalanced $ elsewhere in the (thousands of) content math runs.
    ignoreHtmlClass: 'tex2jax_ignore|mathjax_ignore',
    processHtmlClass: 'tex2jax_process'
  }
};
</script>
<script id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
"""

# Dark-scheme overrides for the Bootstrap 4.3 utility classes that clothe the
# template chrome (header/footer/breadcrumb/nav buttons) but read HARD-CODED
# hues, not the template's --cf-* tokens. Injected AFTER the inlined
# dart_content.css so it wins, and uses ``!important`` because Bootstrap's own
# ``.bg-light`` / ``.border-*`` utilities are ``!important``. Without this the
# body/content flip dark (via the token layers) while the breadcrumb bar stays
# white — the "dark mode shows white background" defect.
_DARK_CHROME_OVERRIDE = """<!-- dark-scheme Bootstrap-utility chrome overrides -->
<style>
@media (prefers-color-scheme: dark) {
  body { background-color: #1a1a1a; color: #e0e0e0; }
  .bg-light { background-color: #1f2733 !important; }
  .border-bottom, .border-top { border-color: #404040 !important; }
  header .breadcrumb, footer { color: #cfd6e0; }
  .breadcrumb-item a { color: #cfe3ff; }
  .breadcrumb-item.active { color: #b6bcc6; }
  .breadcrumb-item + .breadcrumb-item::before { color: #8a94a3; }
  .btn-outline-primary { color: #8cc0ff; border-color: #8cc0ff; }
  .btn-primary { background-color: #2c5aa0; border-color: #2c5aa0; color: #fff; }
}
</style>
"""

# Regexes that stamp the ``mathjax_ignore`` class onto the page chrome so the
# MathJax ignoreHtmlClass above catches them. Header + breadcrumb nav live in
# the template; the ``.toc`` nav comes from the adapter content fragment.
_HEADER_CLASS_RE = re.compile(r'(<header\b[^>]*\bclass=")', re.IGNORECASE)
_BREADCRUMB_NAV_RE = re.compile(
    r'(<nav\b)(\s+aria-label="Breadcrumb")', re.IGNORECASE
)
_TOC_NAV_RE = re.compile(r'(<nav\b[^>]*\bclass="toc)\b', re.IGNORECASE)


def _mark_chrome_ignored(page: str) -> str:
    """Stamp ``mathjax_ignore`` on the header, breadcrumb nav, and TOC nav."""
    page = _HEADER_CLASS_RE.sub(r"\1mathjax_ignore ", page, count=1)
    page = _BREADCRUMB_NAV_RE.sub(r'\1 class="mathjax_ignore"\2', page, count=1)
    page = _TOC_NAV_RE.sub(r"\1 mathjax_ignore", page, count=1)
    return page

_MAIN_RE = re.compile(r"(<main\b[^>]*>)(.*?)(</main>)", re.DOTALL | re.IGNORECASE)
_H1_RE = re.compile(r"<h1\b[^>]*>.*?</h1>", re.DOTALL | re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\{\{\s*[A-Z0-9_]+\s*\}\}")


def _extract_content_fragment(content_html: str) -> str:
    """Return the adapter's INNER content (the ``<article>`` chapters).

    Prefers the inner HTML of the adapter's ``<main>`` (dropping the document
    chrome + the adapter ``<h1>`` document title, so the template's single
    ``<h1>`` is authoritative). Falls back to ``<body>`` inner, then the whole
    string — always with the leading ``<h1>`` and any ``skip-link`` removed so
    no duplicate landmark survives.
    """
    m = _MAIN_RE.search(content_html)
    if m is not None:
        inner = m.group(2)
    else:
        body = re.search(
            r"<body\b[^>]*>(.*?)</body>", content_html, re.DOTALL | re.IGNORECASE
        )
        inner = body.group(1) if body else content_html
    # Drop the adapter's document-title <h1> (the template owns the page <h1>).
    inner = _H1_RE.sub("", inner, count=1)
    # Drop any skip-link the adapter emitted (the template owns the skip-link).
    inner = re.sub(
        r'<a\b[^>]*class="[^"]*skip-link[^"]*"[^>]*>.*?</a>',
        "",
        inner,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return inner.strip()


def _fill_placeholders(
    template: str,
    *,
    title: str,
    course_name: str,
    module_name: str,
    module_link: str,
) -> str:
    """Fill the named template placeholders; blank any that remain.

    Fixes the broken breadcrumb (§B1): the course crumb links to the course
    index and the module crumb to ``module_link`` — no ``href="#"`` dead link.
    Every un-supplied ``{{PLACEHOLDER}}`` is blanked so no raw token ships.
    """
    subs = {
        "{{PAGE_TITLE}}": title,
        "{{COURSE_NAME}}": course_name,
        "{{MODULE_NAME}}": module_name,
        "{{MODULE_LINK}}": module_link,
    }
    for token, value in subs.items():
        template = template.replace(token, value)
    # Any leftover placeholder (OBJECTIVE_1, IMAGE_SRC, YEAR, …) → blank.
    return _PLACEHOLDER_RE.sub("", template)


def _esc_attr(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def assemble(
    *,
    content_html: str,
    template_html: str,
    title: str,
    course_name: str,
    module_name: str,
    module_link: str,
    mathjax: bool,
    css_hooks: bool,
    dart_css_path: Optional[Path],
    description: str = "",
    author: str = "",
) -> str:
    """Assemble the end-user page string (pure; no I/O)."""
    fragment = _extract_content_fragment(content_html)

    # Replace the template's <main> INNER with a single <h1> + the fragment.
    # The template's placeholder demo sections (objectives/content/concepts) are
    # dropped; the <main> open/close tags + landmark stay (single <main>).
    def _inject(m: "re.Match[str]") -> str:
        open_tag, _old_inner, close_tag = m.group(1), m.group(2), m.group(3)
        h1 = f"<h1>{title}</h1>"
        return f"{open_tag}\n{h1}\n{fragment}\n{close_tag}"

    page = _MAIN_RE.sub(_inject, template_html, count=1)
    page = _fill_placeholders(
        page,
        title=title,
        course_name=course_name,
        module_name=module_name,
        module_link=module_link,
    )

    # Head injections (meta + MathJax + dart-content CSS), before </head>.
    head_extra = ""
    # Document metadata (exemplar convention — description + author).
    if description.strip():
        head_extra += (
            f'<meta name="description" content="{_esc_attr(description.strip())}">\n'
        )
    if author.strip():
        head_extra += (
            f'<meta name="author" content="{_esc_attr(author.strip())}">\n'
        )
    if css_hooks:
        css_path = dart_css_path or _DEFAULT_DART_CSS
        try:
            css_text = Path(css_path).read_text(encoding="utf-8")
            head_extra += (
                "<!-- dart-content textbook-semantics styling (B5) -->\n"
                f"<style>\n{css_text}\n</style>\n"
            )
        except OSError as exc:  # pragma: no cover - defensive
            print(f"warning: could not read dart CSS {css_path}: {exc}",
                  file=sys.stderr)
    # Dark-scheme Bootstrap-utility chrome overrides — injected AFTER the dart
    # CSS so it wins the cascade, so the breadcrumb/header/footer flip dark in
    # lockstep with the token-driven body + content.
    head_extra += _DARK_CHROME_OVERRIDE
    if mathjax:
        head_extra += _MATHJAX_HEAD
    if head_extra:
        if re.search(r"</head>", page, re.IGNORECASE):
            # Function replacement: a string replacement would re-process
            # backslashes (\\( -> \() and corrupt the MathJax delimiter
            # config — the round-8 "every paren pair became inline math" bug.
            page = re.sub(
                r"</head>", lambda m: head_extra + "</head>", page, count=1,
                flags=re.IGNORECASE,
            )
        else:  # pragma: no cover - templates always have a head
            page = head_extra + page
    if mathjax:
        # Stamp mathjax_ignore on the chrome so MathJax never typesets the
        # breadcrumb course name / header / TOC nav.
        page = _mark_chrome_ignored(page)
    return page


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--content", required=True,
        help="Converted accessible HTML (the SemantiK adapter output).",
    )
    ap.add_argument(
        "--template", default=str(_DEFAULT_TEMPLATE),
        help="Courseforge accessible content template (with {{PLACEHOLDER}}s).",
    )
    ap.add_argument("--output", required=True, help="Destination HTML (required).")
    ap.add_argument("--title", default="", help="Page <h1>/<title>/breadcrumb.")
    ap.add_argument("--course-name", default="Course", help="Breadcrumb course.")
    ap.add_argument("--module-name", default="Module", help="Breadcrumb module.")
    ap.add_argument(
        "--module-link", default="index.html", help="Breadcrumb module href."
    )
    ap.add_argument(
        "--no-mathjax", action="store_true",
        help="Skip the MathJax (assistive-MML) head injection.",
    )
    ap.add_argument(
        "--css-hooks", action="store_true",
        help="Inline the dart-content textbook-semantics stylesheet.",
    )
    ap.add_argument(
        "--dart-css", default=None,
        help="Path to semantik_content.css (default: templates/accessibility/).",
    )
    ap.add_argument(
        "--description", default="",
        help='Sets <meta name="description"> (exemplar convention).',
    )
    ap.add_argument(
        "--author", default="",
        help='Sets <meta name="author"> (exemplar convention).',
    )
    args = ap.parse_args(argv)

    content_html = Path(args.content).read_text(encoding="utf-8")
    template_html = Path(args.template).read_text(encoding="utf-8")
    title = args.title.strip() or _derive_title(content_html) or "Course Page"

    page = assemble(
        content_html=content_html,
        template_html=template_html,
        title=title,
        course_name=args.course_name,
        module_name=args.module_name,
        module_link=args.module_link,
        mathjax=not args.no_mathjax,
        css_hooks=args.css_hooks,
        dart_css_path=Path(args.dart_css) if args.dart_css else None,
        description=args.description,
        author=args.author,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    print(f"wrote {out_path} ({len(page.encode('utf-8'))} bytes)")
    return 0


def _derive_title(content_html: str) -> str:
    """Best-effort <title>/<h1> from the converted document (never fabricates)."""
    m = re.search(r"<title>(.*?)</title>", content_html, re.DOTALL | re.IGNORECASE)
    if m and m.group(1).strip():
        return m.group(1).strip()
    m = _H1_RE.search(content_html)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(0)).strip()
    return ""


if __name__ == "__main__":
    raise SystemExit(main())
