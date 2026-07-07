"""URL linkification at the adapter seam (ITEM 1, round-2 end-user-HTML audit).

Scanned textbooks carry vendor MEDIA links as bare / angle-wrapped URLs in
block prose; unlinkified they render as MathJax-italic soup in the assembled
learner page. ``math_fold.linkify_urls`` + the adapter ``_linkify_block_urls``
pass turn every URL into a ``mathjax_ignore`` ``<a href>`` anchor (HTML body) or
a bare normalized URL (chunk/sidecar text), dropping the ``<…>`` wrapper + any
glued ``$``.

All synthetic IR — no course-data path, no model, no cascade run.
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)
from lib.semantik.math_fold import linkify_urls


# --- unit: linkify_urls -------------------------------------------------------
def test_linkify_bare_url_html_anchor():
    out = linkify_urls("See (https://vendor.example/l/25Foo) here", html=True)
    assert (
        '<a href="https://vendor.example/l/25Foo" rel="noopener" '
        'class="mathjax_ignore">https://vendor.example/l/25Foo</a>'
    ) in out


def test_linkify_angle_wrapper_dropped():
    out = linkify_urls("X (&lt;https://vendor.example/l/25Bar&gt;) Y", html=True)
    assert "&lt;" not in out and "&gt;" not in out
    assert 'href="https://vendor.example/l/25Bar"' in out
    assert ">https://vendor.example/l/25Bar</a>" in out


def test_linkify_strips_glued_math_delimiters():
    out = linkify_urls("$https://vendor.example/l/25Baz$", html=True)
    assert "$" not in out
    assert 'href="https://vendor.example/l/25Baz"' in out


def test_linkify_normalizes_ocr_spaced_scheme():
    out = linkify_urls("go https : //vendor.example/l/25Q end", html=False)
    assert out == "go https://vendor.example/l/25Q end"


def test_linkify_preserves_trailing_sentence_period():
    out = linkify_urls("end https://vendor.example/l/25Z. Next.", html=True)
    assert ">https://vendor.example/l/25Z</a>. Next." in out


def test_linkify_plain_mode_no_tag():
    out = linkify_urls("(https://vendor.example/l/25P)", html=False)
    assert out == "(https://vendor.example/l/25P)"
    assert "<a" not in out


def test_linkify_idempotent():
    once = linkify_urls("(https://vendor.example/l/25P) x", html=True)
    assert linkify_urls(once, html=True) == once


def test_linkify_noop_without_url():
    assert linkify_urls("no links in this prose at all", html=True) == (
        "no links in this prose at all"
    )


# --- over-capture guard: scheme-only / whitespace-crossing (href="https://Note") ---
def test_linkify_scheme_only_column_bleed_not_linkified():
    # A bare ``https://`` followed (after column-bleed) by whitespace + the next
    # token must NEVER become an anchor — the host char must follow ``//``.
    src = "...the contact page. permitted by https:// Note 1 to Section"
    for mode in (True, False):
        out = linkify_urls(src, html=mode)
        assert out == src
        assert "<a" not in out
        assert 'href="https://Note"' not in out


def test_linkify_bare_scheme_no_host_not_linkified():
    assert linkify_urls("trailing bare https:// end", html=True) == (
        "trailing bare https:// end"
    )
    assert "<a" not in linkify_urls("trailing bare https:// end", html=True)


def test_linkify_legit_host_still_anchored():
    # The conservative host-required pattern must not regress inline URLs.
    out = linkify_urls("see https://www.example.org/contact-page here", html=True)
    assert 'href="https://www.example.org/contact-page"' in out
    assert 'class="mathjax_ignore"' in out


# --- adapter integration ------------------------------------------------------
class _Result:
    def __init__(self, chapters):
        self.chapters = chapters
        self.exit_action = "ship_with_confidence"
        self.wcag_status = "passed"
        self.theta_score = 0.9
        self.flags = []
        self.lane_used = "x"
        self.lang = "en"


def test_adapter_linkifies_block_url_with_mathjax_ignore():
    body = (
        "Product Property (&lt;https://vendor.example/l/25ProductProp&gt;) "
        "Multiply Binomials with Square Roots."
    )
    block = _AdapterBlock(
        html=f"<p>{body}</p>",
        region_kind="paragraph",
        raw_block_index=0,
        raw_text=body,
        heading_text=None,
        pages=[1],
    )
    ch = _AdapterChapter(title="Chapter 1 Roots", blocks=[block])
    out = normalize_cascade_to_ed4all(_Result([ch]), pdf_stem="doc-ch01")
    html = out["html"]
    # HTML body carries a mathjax_ignore anchor; angle wrapper is gone.
    assert 'class="mathjax_ignore"' in html
    assert 'href="https://vendor.example/l/25ProductProp"' in html
    assert "&lt;https" not in html and "&gt;)" not in html
    # Sidecar/chunk text carries the bare URL (never an <a> tag).
    sec_text = out["synthesized_sidecar"]["sections"][0]["data"]["text"]
    assert "https://vendor.example/l/25ProductProp" in sec_text
    assert "<a" not in sec_text
