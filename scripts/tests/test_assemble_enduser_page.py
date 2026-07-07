"""B1/B2 — end-user page assembler: single <main>/skip-link/<h1>, MathJax
assistive-MML, dart-content CSS hooks. Synthetic content + the real template.
"""
from __future__ import annotations

import importlib.util
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE = (
    _REPO_ROOT
    / "Courseforge/templates/accessibility/accessible_content_template.html"
)

# Import the arg-driven script module by path (scripts/ is not a package).
_spec = importlib.util.spec_from_file_location(
    "assemble_enduser_page",
    _REPO_ROOT / "scripts/assemble_enduser_page.py",
)
assemble_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(assemble_mod)

# A SemantiK-adapter-shaped standalone document (its OWN chrome: skip-link,
# <main id="main-content">, <h1>) — exactly what the ad-hoc assembler wrongly
# injected whole (the §B1 double-<main> defect).
_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Roots and Radicals</title></head>
<body>
<a class="skip-link" href="#main-content">Skip to main content</a>
<main id="main-content" role="main" class="dart-document">
<h1>Roots and Radicals</h1>
<article role="doc-chapter" id="chap-1">
<h2>Chapter 9 Roots and Radicals</h2>
<section class="dart-section" aria-labelledby="try-1" data-dart-block-id="try-1"
         data-dart-source="synthesized" data-dart-opener="try_it">
<h4 id="try-1">Try It 9.1</h4>
</section>
<section class="dart-section" id="p1" data-dart-block-id="p1"
         data-dart-source="synthesized">
<p>A number whose square is $m$ is a square root of $m$.</p>
</section>
</article>
</main>
</body>
</html>
"""


def _assemble(**kw):
    template = _TEMPLATE.read_text(encoding="utf-8")
    defaults = dict(
        content_html=_CONTENT,
        template_html=template,
        title="Roots and Radicals",
        course_name="Elementary Algebra",
        module_name="Chapter 9",
        module_link="index.html",
        mathjax=True,
        css_hooks=False,
        dart_css_path=None,
    )
    defaults.update(kw)
    return assemble_mod.assemble(**defaults)


def test_single_main_landmark():
    page = _assemble()
    assert page.count("<main") == 1
    assert page.count("</main>") == 1


def test_single_main_content_id_and_skip_link():
    page = _assemble()
    assert page.count('id="main-content"') == 1
    assert page.lower().count("skip to main content") == 1
    assert 'class="skip-link"' in page  # the template's, not the adapter's


def test_single_h1_no_duplicate_title():
    page = _assemble()
    # The adapter's document <h1> was dropped; the template's single <h1> wins.
    assert page.count("<h1>") == 1
    assert "<h1>Roots and Radicals</h1>" in page


def test_injected_content_and_openers_present():
    page = _assemble()
    assert 'data-dart-opener="try_it"' in page
    assert ">Try It 9.1</h4>" in page
    assert "square root of $m$" in page  # math left intact for MathJax


def test_breadcrumb_not_broken():
    page = _assemble()
    # Placeholders filled; no dead "#" breadcrumb link; no raw {{...}} tokens.
    assert "Elementary Algebra" in page
    assert "Chapter 9" in page
    assert "{{" not in page and "}}" not in page


def test_mathjax_assistive_mml_config():
    page = _assemble(mathjax=True)
    assert "assistive-mml" in page
    assert "assistiveMml" in page
    assert "tex-mml-chtml.js" in page


def test_no_mathjax_when_disabled():
    page = _assemble(mathjax=False)
    assert "tex-mml-chtml.js" not in page


def test_mathjax_config_processes_escapes():
    # Round-7b — the tex config MUST enable processEscapes so the adapter's
    # HTML-only currency escape (``\$5``) renders as a literal dollar instead of
    # opening a false inline-math span.
    page = _assemble(mathjax=True)
    assert "processEscapes: true" in page


def test_css_hooks_inlines_dart_content_css():
    page = _assemble(css_hooks=True)
    assert ".dart-example" in page
    assert 'data-dart-opener="try_it"' in page
    assert ".dart-key-terms dt" in page


def test_css_hooks_absent_by_default():
    page = _assemble(css_hooks=False)
    assert ".dart-key-terms dt" not in page


def test_description_and_author_meta():
    page = _assemble(
        description="A chapter on roots & radicals.",
        author="Elementary Algebra 2e",
    )
    assert '<meta name="description" content="A chapter on roots &amp; radicals.">' in page
    assert '<meta name="author" content="Elementary Algebra 2e">' in page


def test_meta_absent_when_not_supplied():
    page = _assemble()
    assert '<meta name="description"' not in page
    assert '<meta name="author"' not in page


def test_css_hooks_ship_dark_mode_and_reading_column():
    page = _assemble(css_hooks=True)
    # Exemplar conventions: dark-scheme block + centered reading column.
    assert "prefers-color-scheme: dark" in page
    assert 'main[role="main"]' in page
    assert "max-width: 900px" in page
    assert ".toc" in page


# ---------------------------------------------------------------------------
# Dark-mode chrome + MathJax breadcrumb fix (end-user-HTML audit, 2026-07-04).
# ---------------------------------------------------------------------------
def test_mathjax_ignores_page_chrome():
    # DEFECT 2 — the breadcrumb course name "(course-slug-01)" was typeset as math.
    page = _assemble(mathjax=True)
    assert "ignoreHtmlClass" in page
    # header, breadcrumb nav, and TOC nav all carry the ignore class.
    assert re.search(r'<header\b[^>]*\bclass="[^"]*mathjax_ignore', page)
    assert '<nav class="mathjax_ignore" aria-label="Breadcrumb"' in page


def test_no_chrome_marking_without_mathjax():
    page = _assemble(mathjax=False)
    assert "ignoreHtmlClass" not in page
    assert "mathjax_ignore" not in page


def test_toc_nav_marked_ignored_when_present():
    # The adapter TOC nav (class="toc") in the content fragment is also ignored.
    content = _CONTENT.replace(
        '<article role="doc-chapter" id="chap-1">',
        '<nav class="toc" aria-label="Contents"><ol>'
        '<li><a href="#chap-1">X</a></li></ol></nav>'
        '<article role="doc-chapter" id="chap-1">',
    )
    page = _assemble(content_html=content, mathjax=True)
    assert '<nav class="toc mathjax_ignore"' in page


def test_dark_chrome_override_injected():
    # DEFECT 1 — the Bootstrap chrome utilities flip dark in lockstep.
    page = _assemble(css_hooks=True)
    assert "dark-scheme Bootstrap-utility chrome overrides" in page
    assert ".bg-light { background-color: #1f2733 !important; }" in page
    assert ".border-bottom, .border-top { border-color: #404040 !important; }" in page


def test_template_carries_dark_cf_token_block():
    # The template's own --cf-* layer now has a dark override so body/heading
    # flip dark (fixing the invisible white-on-white h1).
    page = _assemble(css_hooks=False)
    assert "@media (prefers-color-scheme: dark)" in page
    assert "--cf-background: #1a1a1a;" in page
    assert "--cf-text: #e0e0e0;" in page
