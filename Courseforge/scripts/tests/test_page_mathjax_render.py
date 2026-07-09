"""COURSEFORGE_PAGE_MATHJAX — page-level MathJax v3 include (default ON).

Rendered course pages carry inline ``\\( … \\)`` / display ``\\[ … \\]`` LaTeX
math but ship no math renderer, so standalone / preview rendering shows raw
LaTeX source (LMSes vary in whether they inject MathJax). ``_wrap_page`` injects
a MathJax v3 loader (``tex-chtml`` from the jsDelivr CDN) + inline config
enabling the two delimiter pairs + assistive-MML accessibility output.

The gate is DEFAULT ON — a documented deviation from the default-off convention
(presentation-only, no LLM/content change beyond the script tags). Covers:

* default (unset) → include present exactly once, config present exactly once.
* explicit truthy → include present.
* falsey tokens (0/false/no/off) → byte-identical absence.
* the ON page equals the OFF page with only the include block added.
* the include carries the two delimiter pairs + assistive-MML + the CDN loader.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import generate_course as gc  # noqa: E402

_BODY = "<h2>Section</h2><p>Compute \\(x^2 + y^2\\) and \\[E = mc^2\\].</p>"

_LOADER = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"


def _page(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("COURSEFORGE_PAGE_MATHJAX", raising=False)
    else:
        monkeypatch.setenv("COURSEFORGE_PAGE_MATHJAX", value)
    return gc._wrap_page("Chapter 1", "PHYS_101", 1, _BODY)


def test_default_on_include_present_exactly_once(monkeypatch):
    html = _page(monkeypatch, None)
    assert gc._page_mathjax_enabled() is True
    # Loader script present exactly once.
    assert html.count(_LOADER) == 1
    # Inline config present exactly once.
    assert html.count("window.MathJax") == 1
    # Injected inside <head>, before </head>.
    head = html[html.index("<head>") : html.index("</head>")]
    assert _LOADER in head
    assert "window.MathJax" in head


def test_truthy_value_include_present(monkeypatch):
    html = _page(monkeypatch, "on")
    assert html.count(_LOADER) == 1
    assert html.count("window.MathJax") == 1


def test_include_carries_delimiters_and_a11y(monkeypatch):
    html = _page(monkeypatch, "1")
    # Both delimiter pairs configured.
    assert "inlineMath: [['\\\\(', '\\\\)']]" in html
    assert "displayMath: [['\\\\[', '\\\\]']]" in html
    # Assistive-MML accessibility output enabled.
    assert "enableAssistiveMml: true" in html
    # Combined tex-chtml component from the CDN, async, with the MathJax id.
    assert 'id="MathJax-script"' in html
    assert "async" in html


def test_falsey_tokens_byte_identical_absence(monkeypatch):
    off_pages = {tok: _page(monkeypatch, tok) for tok in ("0", "false", "no", "off", "OFF")}
    for tok, html in off_pages.items():
        assert gc._page_mathjax_enabled() is False, tok
        assert "MathJax" not in html, tok
        assert _LOADER not in html, tok
    # All falsey tokens produce the identical page (the include is the only diff).
    variants = set(off_pages.values())
    assert len(variants) == 1


def test_on_equals_off_plus_include(monkeypatch):
    off = _page(monkeypatch, "off")
    on = _page(monkeypatch, "on")
    assert on != off
    # ON is exactly OFF with the include block (prefixed by one newline) added.
    assert on.replace("\n" + gc._PAGE_MATHJAX_INCLUDE, "") == off
