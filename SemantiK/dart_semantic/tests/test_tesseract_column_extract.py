"""Column-aware Tesseract line assembly (SEMANTIK_COLUMN_EXTRACT, prong 1).

The pdfplumber column-band extraction (``test_column_extract.py``) shipped v1;
the OCR path (``_tesseract_page_blocks``) was left fused across the gutter
behind a ``TODO(column-extract v1 pdfplumber-only)``. This closes it:

  * ``_split_tesseract_lines_by_column`` re-keys each Tesseract
    ``(block, par, line)`` word group to ``(col, block, par, line)`` using the
    SHARED ``reading_order.column_edges_from_lines`` gutter engine, in the
    IMAGE-PIXEL space of the OCR bboxes (``page_w = image.width``).
  * a fused two-column OCR line splits into per-column line blocks; a genuine
    single-column scan collapses via the guard → byte-identical grouping.
  * ``_merge_page`` scales the column-major gutter by the OCR ``tesseract_width``
    (image pixels), not the PDF-point page width, on the OCR path.
  * the extract-cache salt bumps ``|col1`` → ``|col2`` (prong 1 changed the OCR
    shape under the same flag) and gains ``|ord1`` when SEMANTIK_COLUMN_ORDER is
    on with extraction off (closing a pre-existing cache hole).

All CPU-only, synthetic pytesseract dicts, no PDF, no model, no network.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest

from dart_semantic import extract_shared
from dart_semantic.reading_order import column_edges_from_lines


_EXTRACT = "SEMANTIK_COLUMN_EXTRACT"
_ORDER = "SEMANTIK_COLUMN_ORDER"
_COMPOSED = (
    _ORDER,
    "SEMANTIK_DEPLOY_PROFILE",
    "SEMANTIK_DETECT_FIGURES",
    "SEMANTIK_VLM_EXTRACT",
    "SEMANTIK_VLM_FUSION",
    "SEMANTIK_VLM_STRUCT_HINTS",
)


@pytest.fixture(autouse=True)
def _clear_flags():
    keys = (_EXTRACT, *_COMPOSED)
    prev = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Synthetic pytesseract boundary.
# ---------------------------------------------------------------------------


def _fake_pytesseract(words):
    """A stand-in pytesseract whose image_to_data returns ``words``.

    ``words`` is a list of dicts: {text, block, par, line, left, top, width,
    height, conf}. Parallel arrays are built in list order (OCR reading order).
    """
    mod = types.ModuleType("pytesseract")
    mod.Output = types.SimpleNamespace(DICT="dict")

    data = {
        "text": [w["text"] for w in words],
        "block_num": [w["block"] for w in words],
        "par_num": [w["par"] for w in words],
        "line_num": [w["line"] for w in words],
        "left": [w["left"] for w in words],
        "top": [w["top"] for w in words],
        "width": [w["width"] for w in words],
        "height": [w["height"] for w in words],
        "conf": [w.get("conf", 90) for w in words],
    }

    def image_to_data(image, output_type=None, config=None):
        return data

    mod.image_to_data = image_to_data
    return mod


def _run_tesseract(monkeypatch, words, img_w, img_h=2000):
    class _Img:
        width = img_w
        height = img_h

    monkeypatch.setattr(
        extract_shared,
        "_pypdfium2_render_page_to_image",
        lambda pdf_path, page_num, scale=2.0: _Img(),
    )
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract(words))
    out = extract_shared._tesseract_page_blocks(Path("/nonexistent.pdf"), 1)
    return out["text_blocks"]


def _w(text, block, par, line, left, top, width, height=30, conf=90):
    return {
        "text": text,
        "block": block,
        "par": par,
        "line": line,
        "left": left,
        "top": top,
        "width": width,
        "height": height,
        "conf": conf,
    }


# A synthetic 2-column page in IMAGE-PIXEL space (image width 1224 -> gutter
# 0.06*1224 = 73.44px). Two OCR "lines" each fusing a left-column and a
# right-column run into ONE (block, par, line) group.
def _two_col_fused_words():
    return [
        # line 0 (block0/par0/line0): left col "Alpha Beta" + right col "Gamma Delta"
        _w("Alpha", 0, 0, 0, 100, 100, 60),
        _w("Beta", 0, 0, 0, 180, 100, 60),   # right 240, gap to 700 > 73 -> col1 seed
        _w("Gamma", 0, 0, 0, 700, 100, 60),
        _w("Delta", 0, 0, 0, 780, 100, 60),
        # line 1 (block0/par0/line1)
        _w("One", 0, 0, 1, 100, 200, 60),
        _w("Two", 0, 0, 1, 180, 200, 60),
        _w("Three", 0, 0, 1, 700, 200, 60),
        _w("Four", 0, 0, 1, 780, 200, 60),
    ]


# ---------------------------------------------------------------------------
# 1. flag-off grouping byte-identical.
# ---------------------------------------------------------------------------


def test_flag_off_grouping_byte_identical(monkeypatch):
    words = _two_col_fused_words()
    off = _run_tesseract(monkeypatch, words, img_w=1224)
    # Flag off -> exactly two blocks (one per fused line), each a whole-line join.
    assert [b["text"] for b in off] == [
        "Alpha Beta Gamma Delta",
        "One Two Three Four",
    ]


# ---------------------------------------------------------------------------
# 2. two-column fused line splits per column.
# ---------------------------------------------------------------------------


def test_two_column_fused_line_splits_per_column(monkeypatch):
    os.environ[_EXTRACT] = "1"
    words = _two_col_fused_words()
    blocks = _run_tesseract(monkeypatch, words, img_w=1224)
    texts = [b["text"] for b in blocks]
    # 4 blocks: column 0 (both lines) then column 1 (both lines), top-sorted.
    assert texts == [
        "Alpha Beta",
        "One Two",
        "Gamma Delta",
        "Three Four",
    ]
    # No block mixes a col-0 and a col-1 token.
    for b in blocks:
        toks = set(b["text"].split())
        assert not (toks & {"Alpha", "Beta", "One", "Two"} and
                    toks & {"Gamma", "Delta", "Three", "Four"})
    # Per-column bboxes: col-0 blocks end before the gutter; col-1 start after.
    assert blocks[0]["bbox"][2] <= 700  # col0 right edge
    assert blocks[2]["bbox"][0] >= 700  # col1 left edge


# ---------------------------------------------------------------------------
# 3. single-column OCR byte-identical on vs off (the TODO's regression fear).
# ---------------------------------------------------------------------------


def _single_col_indented_words():
    # All words left-aligned ~100 with a mild indent; ONE right-margin page
    # number outlier (a minority column the guard must reject).
    return [
        _w("Body", 0, 0, 0, 100, 100, 200),
        _w("Indented", 0, 0, 1, 140, 200, 220),
        _w("More", 0, 0, 2, 100, 300, 200),
        _w("Text", 0, 0, 3, 130, 400, 200),
        _w("42", 0, 0, 4, 1100, 100, 40),  # lone right-margin outlier
    ]


def test_single_column_ocr_byte_identical_on_vs_off(monkeypatch):
    words = _single_col_indented_words()
    off = _run_tesseract(monkeypatch, words, img_w=1224)
    os.environ[_EXTRACT] = "1"
    on = _run_tesseract(monkeypatch, words, img_w=1224)
    assert [b["text"] for b in on] == [b["text"] for b in off]
    assert [b["bbox"] for b in on] == [b["bbox"] for b in off]


# ---------------------------------------------------------------------------
# 4. bridging full-width header stays one block, column 0.
# ---------------------------------------------------------------------------


def test_bridging_header_stays_one_block(monkeypatch):
    os.environ[_EXTRACT] = "1"
    words = _two_col_fused_words()
    # A header line whose single word STRADDLES the interior gutter (left 100,
    # width 800 -> right 900 spans the ~700 edge).
    words.append(_w("WIDEHEADER", 1, 0, 0, 100, 40, 800))
    blocks = _run_tesseract(monkeypatch, words, img_w=1224)
    texts = [b["text"] for b in blocks]
    # The header is ONE block, not scattered; it sorts to column 0 (top 40).
    assert "WIDEHEADER" in texts
    header = [b for b in blocks if b["text"] == "WIDEHEADER"][0]
    # It stays a single word block spanning the gutter.
    assert header["bbox"][0] <= 100 and header["bbox"][2] >= 900
    # Column 0 (which includes the top-most header) leads the emission.
    assert texts[0] == "WIDEHEADER"


# ---------------------------------------------------------------------------
# 5. word multiset conserved across re-key.
# ---------------------------------------------------------------------------


def test_word_multiset_conserved_across_rekey(monkeypatch):
    os.environ[_EXTRACT] = "1"
    words = _two_col_fused_words()
    blocks = _run_tesseract(monkeypatch, words, img_w=1224)
    got = sorted(tok for b in blocks for tok in b["text"].split())
    expected = sorted(w["text"] for w in words)
    assert got == expected


# ---------------------------------------------------------------------------
# 6. _merge_page uses tesseract_width for the gutter (the pixel-space bug fix).
# ---------------------------------------------------------------------------


def _px_block(text, x0, y0, x1, y1):
    return {"bbox": [float(x0), float(y0), float(x1), float(y1)], "text": text}


def test_merge_page_uses_tesseract_width_for_gutter():
    os.environ[_ORDER] = "1"
    # Two-column page in IMAGE-PIXEL space. Within-column left-edge variance
    # (~45px) is BELOW the correct 1224px gutter (73px) but ABOVE the buggy
    # 612pt gutter (37px) — so only the correct width yields clean 2 columns.
    page = {
        "width": 612,  # PDF points (the buggy value)
        "tesseract_width": 1224,  # image pixels (the correct space)
        "tesseract": {
            "text_blocks": [
                _px_block("c0-top", 100, 50, 300, 90),
                _px_block("c1-top", 700, 60, 900, 100),
                _px_block("c0-mid", 145, 200, 340, 240),
                _px_block("c1-mid", 745, 210, 940, 250),
            ]
        },
    }
    merged = extract_shared._merge_page(page, text_layer_ok=False)
    texts = [b["text"] for b in merged["text_blocks"]]
    # Column-major: all of column 0 (top-sorted) then all of column 1.
    assert texts == ["c0-top", "c0-mid", "c1-top", "c1-mid"]


def test_merge_page_without_tesseract_width_uses_point_width():
    # Regression guard: when tesseract_width is ABSENT the branch falls back to
    # the point width (no crash; the pre-fix behaviour on non-OCR pages).
    os.environ[_ORDER] = "1"
    page = {
        "width": 612,
        "tesseract": {
            "text_blocks": [
                _px_block("a", 100, 50, 300, 90),
                _px_block("b", 100, 200, 300, 240),
            ]
        },
    }
    merged = extract_shared._merge_page(page, text_layer_ok=False)
    assert [b["text"] for b in merged["text_blocks"]] == ["a", "b"]


# ---------------------------------------------------------------------------
# 7. cache-key ord salt.
# ---------------------------------------------------------------------------


class _FakePath:
    def stat(self):
        return types.SimpleNamespace(st_size=1234, st_mtime=1700000000)

    def resolve(self):
        return "/abs/fixture.pdf"


def _key():
    return extract_shared._compute_extract_cache_key(_FakePath())


def test_cache_key_ord_salt():
    import hashlib

    fig_key = "fig1" if extract_shared._detect_figures_enabled() else "fig0"

    # both off -> no col/ord salt (byte-identical historic key).
    base_raw = (
        f"v{extract_shared.EXTRACT_CACHE_VERSION}|{fig_key}|"
        f"/abs/fixture.pdf|1234|1700000000"
    )
    assert _key() == hashlib.sha256(base_raw.encode()).hexdigest()[:24]

    # order-on / extract-off -> |ord1 present, no |col2.
    os.environ[_ORDER] = "1"
    ord_raw = (
        f"v{extract_shared.EXTRACT_CACHE_VERSION}|{fig_key}|ord1|"
        f"/abs/fixture.pdf|1234|1700000000"
    )
    assert _key() == hashlib.sha256(ord_raw.encode()).hexdigest()[:24]
    os.environ.pop(_ORDER, None)

    # extract-on -> |col2 and NO |ord1 (extract already implies + salts order).
    os.environ[_EXTRACT] = "1"
    col_raw = (
        f"v{extract_shared.EXTRACT_CACHE_VERSION}|{fig_key}|col2|"
        f"/abs/fixture.pdf|1234|1700000000"
    )
    assert _key() == hashlib.sha256(col_raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# 8. column_edges_from_lines helper direct unit tests.
# ---------------------------------------------------------------------------


def test_column_edges_from_lines_two_column():
    lines = [
        [(100.0, 160.0), (700.0, 760.0)],
        [(100.0, 160.0), (700.0, 760.0)],
    ]
    edges = column_edges_from_lines(lines, page_w=1224.0)
    assert edges == [100.0, 700.0]


def test_column_edges_from_lines_single_column_returns_none():
    # One well-populated column -> guard collapses -> None.
    lines = [[(100.0, 300.0)], [(110.0, 290.0)], [(100.0, 305.0)]]
    assert column_edges_from_lines(lines, page_w=1224.0) is None


def test_column_edges_from_lines_empty_returns_none():
    assert column_edges_from_lines([], page_w=1224.0) is None
    assert column_edges_from_lines([[]], page_w=1224.0) is None


def test_column_edges_from_lines_seed_mask_excludes_table_words():
    # A wide "table" run in the middle would seed a spurious gutter; masking it
    # out (seed=False) collapses back to a single column -> None. The mask still
    # advances the prev-x1 chain, matching the inline pdfplumber seeding.
    lines = [[(100.0, 160.0), (500.0, 560.0), (900.0, 960.0)]]
    masks = [[True, False, False]]  # only the first word may seed
    assert column_edges_from_lines(lines, page_w=1224.0, seed_masks=masks) is None
