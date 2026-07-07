"""Document HARD-gate check: no external network references (scan-lane fix).

Guards the shipped page against the VLM fabricated-image defect leaking
through: an external ``http(s)://`` resource ``src=`` (``<img>`` / ``<script>``)
or stylesheet ``<link href>`` is a broken off-origin fetch (WCAG 1.1.1 + CSP),
and a residual ``![alt](url)`` is an un-sanitized fabricated image. Prose
``<a href>`` attribution links (publisher / cnx) are legitimate content and are
NOT gated.

Synthetic HTML fragments only — no course-data path, no cascade, no Chromium
(the check is a pure regex scan; axe is a separate, expensive doc check).
"""
from __future__ import annotations

from dart_semantic.gates.hard_document import _check_no_external_refs
from dart_semantic.gates.types import GateCheck


def test_external_img_src_fails():
    html = '<main><img src="https://i.imgur.com/abc.png" alt="x"></main>'
    out = _check_no_external_refs(html)
    assert out.check is GateCheck.NO_EXTERNAL_REFS
    assert out.passed is False


def test_external_script_src_fails():
    html = '<main><script src="http://evil.example/x.js"></script></main>'
    assert _check_no_external_refs(html).passed is False


def test_external_stylesheet_link_fails():
    html = '<link rel="stylesheet" href="https://cdn.example/x.css"><main></main>'
    assert _check_no_external_refs(html).passed is False


def test_residual_markdown_image_fails():
    html = "<main><p>See ![Figure 1.6](https://i.imgur.com/3Q5z5.png)</p></main>"
    out = _check_no_external_refs(html)
    assert out.check is GateCheck.NO_EXTERNAL_REFS
    assert out.passed is False


def test_escaped_external_img_tag_fails():
    # The VLM emits a raw <img> tag as literal text in a table cell; the HTML
    # path ESCAPES it, so the unescaped _EXTERNAL_SRC_RE never sees it. The
    # escaped-shape check must still fail the document closed.
    html = (
        "<main><table><tr><td>"
        "&lt;img src=&quot;https://i.imgur.com/1.png&quot;&gt;"
        "</td></tr></table></main>"
    )
    out = _check_no_external_refs(html)
    assert out.check is GateCheck.NO_EXTERNAL_REFS
    assert out.passed is False


def test_escaped_external_img_tag_single_quote_entity_fails():
    html = "<main>&lt;img src=&#39;http://evil.example/x.png&#39; /&gt;</main>"
    assert _check_no_external_refs(html).passed is False


def test_self_contained_doc_passes():
    html = (
        "<main><h1>Title</h1><p>Body prose with inline math $x^2$.</p>"
        '<img src="doc-ch01_figures/fig-3.png" alt="A chart">'
        '<span class="dart-figure-notation" role="img" '
        'aria-label="Figure 1.6 (image not recoverable)">[figure]</span>'
        "</main>"
    )
    out = _check_no_external_refs(html)
    assert out.passed is True


def test_prose_anchor_href_passes():
    # A prose attribution link is legitimate content, NOT a resource fetch.
    html = (
        '<main><p>This book is available at '
        '<a href="http://cnx.org/content/abc">cnx.org</a>.</p></main>'
    )
    out = _check_no_external_refs(html)
    assert out.passed is True
