"""Render-time CID-glyph / double-encoded-UTF-8 mojibake repair (display-only).

Some vendor PDFs (CID-keyed fonts, Windows-1252-in-UTF-8 double-encoding) flow a
mis-decoded punctuation glyph into ``raw_text`` as an ``â€…`` mojibake sequence or
as an ``â`` trailed by unresolved ``(cid:N)`` glyphs (e.g. ``countryâ(cid:0)(cid:0)s``
for ``country's``). ``cascade_ir._repair_mojibake`` normalizes these at the render
seam only — ``raw_text`` / sourceIds stay verbatim — and is byte-identical on clean
text. All synthetic strings; no course-data path, no model, no cascade run.
"""
from __future__ import annotations

from lib.semantik.cascade_ir import _block_html_for_kind, _repair_mojibake


# --- CID-collapsed variant (â + (cid:N) glyphs) -------------------------------
def test_repair_cid_collapsed_possessive_apostrophe():
    assert _repair_mojibake("villageâ(cid:0)(cid:0)s harbor") == "village’s harbor"
    assert _repair_mojibake("the young townâ(cid:0)(cid:0)s charter") == (
        "the young town’s charter"
    )


def test_repair_cid_collapsed_single_glyph():
    # The 1-glyph ``â(cid:N)`` variant folds to the same right-single-quote.
    assert _repair_mojibake("Councilâ(cid:0)s request") == "Council’s request"


def test_repair_cid_digit_range_en_dash():
    # A digit-bounded collapsed span is deterministically a number-range en dash.
    assert _repair_mojibake("records from 1841â(cid:0)(cid:0)1843 show") == (
        "records from 1841–1843 show"
    )


def test_repair_strips_bare_cid_glyph():
    # An unmapped bare CID glyph is noise — dropped entirely.
    assert _repair_mojibake("alpha (cid:7) beta") == "alpha  beta"


# --- byte-recoverable cp1252-in-UTF-8 family ----------------------------------
def test_repair_double_encoded_family():
    assert _repair_mojibake("donâ€™t") == "don’t"
    assert _repair_mojibake("â€œquotedâ€\x9d") == "“quoted”"
    assert _repair_mojibake("3â€“5 and Aâ€”B") == "3–5 and A—B"
    assert _repair_mojibake("wellâ€¦ ok") == "well… ok"
    assert _repair_mojibake("nbspÂ here") == "nbsp here"


def test_repair_bare_curly_close_last():
    # The bare ``â€`` (curly-close) row must not clobber the longer rows above it.
    assert _repair_mojibake("â€™") == "’"
    assert _repair_mojibake("â€") == "”"


# --- byte-identical guard -----------------------------------------------------
def test_repair_clean_text_is_noop():
    for s in ["", "plain prose with no markers", "cost $5 and $3", "a < b & c"]:
        assert _repair_mojibake(s) == s


# --- render seam integration --------------------------------------------------
def test_block_render_repairs_mojibake_display_only():
    html = _block_html_for_kind("paragraph", "villageâ(cid:0)(cid:0)s harbor", None)
    assert html == "<p>village’s harbor</p>"
    assert "(cid:" not in html


def test_block_render_clean_text_byte_identical():
    assert _block_html_for_kind("paragraph", "Plain prose.", None) == "<p>Plain prose.</p>"
