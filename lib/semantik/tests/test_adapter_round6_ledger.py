r"""Round-6 visual-convergence ledger regression tests (2026-07-04).

Covers the two residual-defect items cleared in round 6 (final micro-polish) of
the SemantiK visual-convergence loop:

* ITEM 1 — HTML-mode leading-gutter scrub gap: a stray ``&gt;| `` gutter run
  glued to a masked ``<p>`` tag (``\x00N\x00&gt;| TRYIT``, ch09 [355]) sat
  between the tag sentinel and the text, so the old whitespace-only boundary
  could not reach it and it shipped as a visible ``&gt;| `` prefix. The gutter
  fold now folds a run at a mask-sentinel (masked-tag/math) boundary too, while a
  lone interior comparison operator (``x &gt; y``) and in-math ``&gt;`` survive.
* ITEM 2 — the ``:: MEDIA :: …`` colon-run residue inside the aria-hidden
  ``dart-continuation`` banners (ch04 / ch07) — the marker scrub runs on block
  bodies, never on the chapter-title/banner path — now folds at the banner emit
  site (user-invisible, but keep the source clean).

All fixtures are inline synthetic HTML/text — NO reference to ``inputs/`` or any
course-data path.
"""

from __future__ import annotations

import re

import pytest

from lib.semantik.adapter import (
    _AdapterBlock,
    _AdapterChapter,
    _scrub_marker_artifacts,
    normalize_cascade_to_ed4all,
)


def _result(chapters):
    r = type("R", (), {})()
    r.exit_action = "ship_with_confidence"
    r.wcag_status = "passed"
    r.lang = "en"
    r.chapters = chapters
    return r


# ---------------------------------------------------------------------------
# ITEM 1 — leading gutter after a masked tag (masked-tag boundary fold)
# ---------------------------------------------------------------------------
def test_item1_leading_gutter_after_masked_tag_scrubbed():
    """A ``&gt;| `` run glued to a masked ``<p>`` tag is folded (no visible
    leading gutter prefix), while the surrounding content survives."""
    out = _scrub_marker_artifacts(
        "<p>&gt;| TRYIT:: OR ... 9.175 Simplify ...</p>", html=True
    )
    assert "&gt;|" not in out
    assert "&gt;" not in out
    # The gutter run is gone; the block keeps its (now clean) marker + content.
    assert "TRYIT" in out and "9.175 Simplify" in out
    assert out.startswith("<p> TRYIT") or out.startswith("<p>TRYIT")


def test_item1_bare_gt_gutter_after_masked_tag_scrubbed():
    """A lone ``&gt;`` gutter glyph at a masked-tag boundary still folds (the
    edge case that is NOT a mid-prose comparison)."""
    out = _scrub_marker_artifacts("<p>&gt; leading gutter glyph here</p>", html=True)
    assert "&gt;" not in out
    assert "leading gutter glyph here" in out


@pytest.mark.parametrize(
    "text,html",
    [
        (r"a $x &gt; y$ b", True),          # &gt; inside masked math — untouched
        (r"a $x > y$ b", True),             # bare > inside masked math — untouched
        ("x &gt; y", True),                 # mid-prose comparison — untouched
        ("the value x > y always holds", False),  # mid-prose comparison — untouched
    ],
)
def test_item1_comparison_operators_preserved(text, html):
    """A legit ``>`` / ``&gt;`` comparison operator — inside ``$…$`` math OR a
    lone mid-prose operator flanked by single-char value operands — is NEVER
    folded."""
    assert _scrub_marker_artifacts(text, html=html) == text


@pytest.mark.parametrize(
    "text,html",
    [
        # Marker debris: a decimal exercise number ``>`` a capitalized instruction
        # word is NOT a value-operand comparison -> folds.
        ("TRY IT: 9.133 > Simplify: the radical", False),
        # Edge / punctuation-adjacent stray gutter glyph -> folds.
        ("<p>system. &gt; ![](img)</p>", True),
        # Trailing gutter glyph glued to a masked close tag -> folds.
        ("<p>Factor: $18u - 36$. &gt;</p>", True),
    ],
)
def test_item1_gutter_debris_still_folds(text, html):
    """The comparison guard must NOT leak OCR gutter/marker debris: a ``>`` that
    is edge-adjacent, punctuation-adjacent, or between non-value tokens still
    folds (anti-regression for the round-6 comparison-guard)."""
    out = _scrub_marker_artifacts(text, html=html)
    assert "&gt;" not in out and ">" not in _html_body_text(out)


def _html_body_text(html: str) -> str:
    """Visible text of a scrub result (tags stripped) for the debris assertion."""
    return re.sub(r"<[^>]+>", "", html)


def test_item1_leading_gutter_survives_end_to_end():
    """Through the full adapter: a content-bearing block whose HTML starts with a
    ``&gt;| `` gutter run emits with the gutter gone and the content intact."""
    chapters = [
        _AdapterChapter(
            title="Chapter 9 Roots and Radicals",
            blocks=[
                _AdapterBlock(
                    html="<p>&gt;| TRYIT:: 9.175 Simplify the radical expression here.</p>",
                    region_kind="paragraph",
                    raw_block_index=0,
                    raw_text=">| TRYIT:: 9.175 Simplify the radical expression here.",
                    pages=[1],
                    block_role="paragraph",
                )
            ],
        )
    ]
    out = normalize_cascade_to_ed4all(_result(chapters), pdf_stem="ea2e_ch9")
    html = out["html"]
    assert "&gt;|" not in html
    assert "Simplify the radical expression here." in html


# ---------------------------------------------------------------------------
# ITEM 2 — ``::`` fold inside the aria-hidden continuation banner
# ---------------------------------------------------------------------------
def _continuation_banner_text(title: str) -> str:
    ch = _AdapterChapter(
        title=title,
        blocks=[
            _AdapterBlock(
                html="<p>body content here</p>",
                region_kind="paragraph",
                raw_block_index=0,
                raw_text="body content here",
                pages=[1],
                block_role="paragraph",
            )
        ],
    )
    ch.continuation = True
    out = normalize_cascade_to_ed4all(_result([ch]), pdf_stem="ea2e_ch4")
    m = re.search(
        r'<div class="dart-continuation"[^>]*>(.*?)</div>', out["html"], re.DOTALL
    )
    assert m is not None, "continuation banner not emitted"
    return m.group(1)


def test_item2_banner_colon_run_folded():
    """A ``:: MEDIA :: …`` colon-run residue in a continuation banner folds to
    clean text (single-space fold, round-1 convention)."""
    assert (
        _continuation_banner_text(":: MEDIA :: Simplifying Higher Roots")
        == "MEDIA Simplifying Higher Roots"
    )


def test_item2_normal_banner_text_untouched():
    """A clean continuation-banner title is a no-op (no spurious fold)."""
    assert (
        _continuation_banner_text("Simplifying Higher Roots")
        == "Simplifying Higher Roots"
    )
