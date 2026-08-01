"""The not-recoverable-figure placeholder must surface ``alt`` VISIBLY.

Regression for the 2026-08-01 vendor-corpus finding: ``_figure_placeholder``
emitted the literal token ``[figure]`` as its visible body and put the ``alt``
only in ``aria-label``. The chunker was fine — ``Trainforge/parsers/
html_content_parser.py`` deliberately substitutes the ``aria-label`` — but the
LEARNER page was gutted on any corpus that renders maths as images. A vendor
algebra textbook converted to 15,790 opaque ``[figure]`` tokens across 16,876
images, so a reader saw "Evaluate [figure] when [figure]".

The invariant these tests pin: the ``aria-label`` contract is UNCHANGED (every
downstream consumer keeps working) while the visible body now carries the alt.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.semantik.math_fold import (  # noqa: E402
    _figure_placeholder,
    strip_literal_img_tags,
    strip_markdown_images,
)

TEX = r"{\left(x+y\right)}^{2}"
PROSE = "A horizontal number line with arrows on each end"


def _body(span: str) -> str:
    """Visible text between the placeholder span's tags."""
    m = re.search(r'aria-label="[^"]*">(.*)</span>$', span, re.S)
    assert m, f"not a placeholder span: {span!r}"
    return m.group(1)


def _label(span: str) -> str:
    m = re.search(r'aria-label="([^"]*)"', span)
    assert m, f"no aria-label: {span!r}"
    return m.group(1)


# --- the actual defect ----------------------------------------------------

def test_visible_body_is_never_the_bare_figure_token():
    for alt in (TEX, PROSE, "a", "."):
        assert _body(_figure_placeholder(alt, html=True)) != "[figure]"


def test_tex_alt_is_delimited_for_mathjax():
    body = _body(_figure_placeholder(TEX, html=True))
    assert body.startswith(r"\(") and body.endswith(r"\)")
    assert TEX in body, "the publisher's own TeX must pass through verbatim"


def test_prose_alt_is_plain_visible_text():
    body = _body(_figure_placeholder(PROSE, html=True))
    assert body == PROSE
    assert r"\(" not in body, "a figure description is not maths"


# --- the contract that must NOT move --------------------------------------

def test_aria_label_contract_is_unchanged():
    """Downstream reads this: chunker substitution + annotation stripper."""
    for alt in (TEX, PROSE):
        assert _label(_figure_placeholder(alt, html=True)) == (
            f"{alt} (image not recoverable)"
        )


def test_plain_mode_unchanged():
    assert _figure_placeholder(TEX, html=False) == f"[figure: {TEX}]"


def test_empty_alt_still_degrades_to_figure():
    span = _figure_placeholder("", html=True)
    assert _body(span) == "Figure"
    assert _label(span) == "Figure (image not recoverable)"


# --- sanitizer integration ------------------------------------------------

def test_remote_math_img_yields_visible_tex():
    tag = (
        '<img src="https://opentextbc.ca/ql-cache/quicklatex.com-abc_l3.svg" '
        'alt="a\\le b">'
    )
    out = strip_literal_img_tags(tag, html=True)
    assert r"\(a\le b\)" in out
    assert "[figure]" not in out


def test_local_figure_img_is_untouched():
    tag = '<img src="doc-ch01_figures/fig-3.png" alt="A chart">'
    assert strip_literal_img_tags(tag, html=True) == tag


def test_idempotent():
    """Re-running a sanitizer over its own output must not double-wrap."""
    tag = '<img src="https://x.test/q.svg" alt="a\\le b">'
    once = strip_literal_img_tags(tag, html=True)
    assert strip_literal_img_tags(once, html=True) == once
    assert strip_markdown_images(once, html=True) == once


def test_body_is_html_escaped():
    span = _figure_placeholder('<b>&"x"', html=True)
    assert "<b>" not in _body(span), "alt must not inject markup into the page"
    assert "&lt;b&gt;" in _body(span)
