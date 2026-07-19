"""Part F — cascade_ir figure emitter ``<img src>`` wiring (no models/GPU).

Verifies the Ed4All adapter-IR emitter fills the previously-empty
``<img src="">`` from the ``image_src`` carried on the region_provenance,
keeps the anti-broken-``<img>`` contract (text-only ``<figure>``) when no
src resolved, and threads ``image_src`` onto the produced ``_AdapterBlock``.
"""

from __future__ import annotations

from lib.semantik.cascade_ir import (
    _block_from_provenance,
    _render_figure_html,
)


def test_render_figure_with_src():
    html = _render_figure_html("Figure 1: a widget", "A widget.", "./b_figures/fig-7.png")
    assert '<img src="./b_figures/fig-7.png"' in html
    assert 'alt="A widget."' in html
    # caption differs from alt → a <figcaption> rides along.
    assert "<figcaption>" in html


def test_render_figure_src_alt_falls_back_to_type_level():
    html = _render_figure_html("", "", "./b_figures/fig-1.png")
    assert '<img src="./b_figures/fig-1.png"' in html
    assert 'alt="Figure."' in html


def test_render_figure_no_src_is_text_only():
    # Byte-stable to the historic anti-broken-<img> contract.
    html = _render_figure_html("A widget.", "A widget.", None)
    assert "<img" not in html
    assert html == "<figure><figcaption>A widget.</figcaption></figure>"


def test_render_figure_empty_no_src_ships_type_level_caption(monkeypatch):
    # A figure with no caption / alt / src must NEVER degrade to a silent
    # empty <figure></figure> (the figure/table assembly drop this regression
    # closes). It ships the honest type-level "Figure." accessible name so the
    # figure stays labelled to a screen reader. D2 (default-on
    # SEMANTIK_GLMOCR_MATH_NORMALIZE): the placeholder rides an sr-only <span>
    # so it stays out of Trainforge chunk text; the accessible name is intact.
    monkeypatch.delenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", raising=False)
    html = _render_figure_html("", "", None)
    assert html != "<figure></figure>"
    assert html == (
        '<figure><figcaption><span class="sr-only">Figure.</span></figcaption></figure>'
    )
    assert "<img" not in html  # still no broken <img src="">
    # Explicit falsey flag → byte-identical legacy visible figcaption.
    monkeypatch.setenv("SEMANTIK_GLMOCR_MATH_NORMALIZE", "0")
    assert _render_figure_html("", "", None) == (
        "<figure><figcaption>Figure.</figcaption></figure>"
    )


def test_render_figure_caption_text_when_no_alt_or_raw():
    # The synthetic-image case: empty raw_text + no model alt, but the
    # structure graph resolved a "Figure N:" caption neighbor. The caption
    # text becomes the accessible name (and <figcaption>).
    html = _render_figure_html("", None, None, "Figure 3: a histogram.")
    assert "<figcaption>Figure 3: a histogram.</figcaption>" in html
    assert "<figure></figure>" not in html


def test_block_from_provenance_threads_image_src():
    prov = {
        "region_kind": "figure",
        "raw_text": "Figure 2: chart",
        "figure_alt": "A bar chart.",
        "image_src": "./doc_figures/fig-12.png",
        "first_raw_block_index": 12,
        "pages": [3],
    }
    block = _block_from_provenance(prov)
    assert block.image_src == "./doc_figures/fig-12.png"
    assert '<img src="./doc_figures/fig-12.png"' in block.html
    assert 'alt="A bar chart."' in block.html


def test_block_from_provenance_no_src_byte_stable():
    prov = {
        "region_kind": "figure",
        "raw_text": "A widget.",
        "figure_alt": "A widget.",
        "first_raw_block_index": 5,
        "pages": [1],
    }
    block = _block_from_provenance(prov)
    assert block.image_src is None
    assert "<img" not in block.html
