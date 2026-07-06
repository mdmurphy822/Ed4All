"""VLM fabricated-image sanitizer at the adapter seam (scan-lane fix).

The Qwen2.5-VL scan-lane transcriber sometimes INVENTS a Markdown image
``![alt](url)`` for a figure it cannot transcribe — the ``url`` is a
hallucinated external host / bare relative filename that resolves to nothing.
``math_fold.strip_markdown_images`` + the adapter ``_strip_markdown_images``
pass (which runs BEFORE ``_linkify_block_urls``) replace every image with the
accessible ``.dart-figure-notation`` placeholder, so the fabricated URL never
becomes a live ``<a href>`` anchor and never ships as a broken external
``<img>``. Legitimate prose URLs (cnx/openstax attribution) are still
linkified — only image syntax is stripped.

All synthetic IR — no course-data path, no model, no cascade run. Harness idiom
mirrors ``test_adapter_url_linkify.py``.
"""
from __future__ import annotations

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    normalize_cascade_to_ed4all,
)
from lib.semantik.math_fold import strip_markdown_images


# --- unit: strip_markdown_images ---------------------------------------------
def test_strip_external_host_html_placeholder():
    out = strip_markdown_images(
        "See Figure 1.6. ![Figure 1.6](https://i.imgur.com/3Q5z5.png)",
        html=True,
    )
    assert "imgur" not in out
    assert "![" not in out
    assert 'class="dart-figure-notation"' in out
    assert 'aria-label="Figure 1.6 (image not recoverable)"' in out


def test_strip_relative_target_url_agnostic():
    out = strip_markdown_images("![x](image.png)", html=True)
    assert "image.png" not in out
    assert "![" not in out
    assert 'role="img"' in out


def test_strip_example_com_target():
    out = strip_markdown_images("a ![alt](example.com/y) b", html=True)
    assert "example.com" not in out
    assert "![" not in out


def test_strip_plain_mode_no_span_no_url():
    out = strip_markdown_images("a ![Fig 2](https://h/x.png) b", html=False)
    assert "<span" not in out
    assert "https" not in out and "![" not in out
    assert "[figure: Fig 2]" in out


def test_strip_default_alt_is_figure():
    out = strip_markdown_images("![](https://h/x.png)", html=True)
    assert 'aria-label="Figure (image not recoverable)"' in out


def test_strip_idempotent():
    once = strip_markdown_images("![Fig](https://h/x.png)", html=True)
    assert strip_markdown_images(once, html=True) == once


def test_strip_noop_without_image():
    text = "Just prose, a http://cnx.org/x link, and $x^2$ math."
    assert strip_markdown_images(text, html=True) == text


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


def _one_block_out(body: str):
    block = _AdapterBlock(
        html=f"<p>{body}</p>",
        region_kind="paragraph",
        raw_block_index=0,
        raw_text=body,
        heading_text=None,
        pages=[1],
    )
    ch = _AdapterChapter(title="Chapter 1 Roots", blocks=[block])
    return normalize_cascade_to_ed4all(_Result([ch]), pdf_stem="doc-ch01")


def test_adapter_strips_fabricated_image_before_linkify():
    # The strip runs BEFORE _linkify_block_urls, so the fabricated imgur URL is
    # NEVER turned into an <a> anchor and the placeholder span is present.
    out = _one_block_out(
        "See Figure 1.6. ![Figure 1.6](https://i.imgur.com/3Q5z5.png)"
    )
    html = out["html"]
    assert "imgur" not in html
    assert "![" not in html
    # No anchor minted from the fabricated URL (the doc shell's skip-link <a>
    # is legitimate; only a fabricated-host href would be a defect).
    assert "href=" not in html or "i.imgur" not in html
    assert 'class="dart-figure-notation"' in html
    assert "Figure 1.6" in html  # aria-label carries the alt


def test_adapter_strips_relative_target():
    out = _one_block_out("Diagram here ![x](image.png) end")
    html = out["html"]
    assert "image.png" not in html
    assert "![" not in html


def test_adapter_strips_example_com_target():
    out = _one_block_out("Chart ![c](example.com/chart) done")
    html = out["html"]
    assert "example.com" not in html
    assert "![" not in html


def test_adapter_prose_url_still_linkified_not_stripped():
    # NEGATIVE control: a plain prose URL (no image syntax) is legitimate
    # attribution content and MUST still be linkified into an anchor.
    out = _one_block_out("This book is available at http://cnx.org/content/abc")
    html = out["html"]
    assert "<a" in html
    assert 'href="http://cnx.org/content/abc"' in html


def test_adapter_cleans_sidecar_text():
    # The fabricated URL must not reach the chunker/retrieval index either.
    out = _one_block_out(
        "See Figure 1.6. ![Figure 1.6](https://i.imgur.com/3Q5z5.png)"
    )
    sec_text = out["synthesized_sidecar"]["sections"][0]["data"]["text"]
    assert "imgur" not in sec_text
    assert "![" not in sec_text
