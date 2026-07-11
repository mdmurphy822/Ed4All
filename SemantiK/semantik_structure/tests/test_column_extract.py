"""Column-aware EXTRACTION pass (SEMANTIK_COLUMN_EXTRACT, default OFF).

Multi-column born-digital PDFs (Federal Register 3-col, CFR 2-col, IRS W-9
2-col) are extracted by ``extract_shared._pdfplumber_page``, which historically
bucketed each physical line across the FULL page width then x-sorted +
space-joined — fusing unrelated columns into one block string (the
"...OFFICE OF FOREIGN ASSETS Affairs..." salad). SEMANTIK_COLUMN_EXTRACT
segments the page into column bands BEFORE line assembly so each column's text
stays contiguous.

Invariants pinned here (all CPU-only, no model, no PDF):
  (a) resolver parse-with-fallback, default OFF,
  (b) a synthetic 2-column page emits col-1 block(s) before col-2, and NO
      emitted block mixes a col-1 and a col-2 token,
  (c) a synthetic single-column page is byte-identical flag ON vs OFF,
  (d) a full-width header line stays ONE block (not scattered across bands),
  (e) the cache key gains ``|col1`` only when the flag is on (flag-off key
      byte-identical to the historic key).
"""
from __future__ import annotations

import os

import pytest

from semantik_structure import extract_shared
from semantik_structure.extract_shared import (
    _assemble_text_blocks_columnaware,
    _assemble_text_blocks_singlewidth,
)
from semantik_structure.reading_order import (
    resolve_column_extract_mode,
    resolve_column_order_mode,
)

_FLAG = "SEMANTIK_COLUMN_EXTRACT"
# Flags that also feed the composed resolvers (COLUMN_ORDER / DEPLOY_PROFILE) or
# the sibling extract-cache salts (figures / VLM) — cleared so the environment
# cannot leak an ON state into a "default OFF" / historic-key assertion.
_COMPOSED = (
    "SEMANTIK_COLUMN_ORDER",
    "SEMANTIK_DEPLOY_PROFILE",
    "SEMANTIK_DETECT_FIGURES",
    "SEMANTIK_VLM_EXTRACT",
    "SEMANTIK_VLM_FUSION",
    "SEMANTIK_VLM_STRUCT_HINTS",
)


@pytest.fixture(autouse=True)
def _clear_flags():
    prev = {k: os.environ.get(k) for k in (_FLAG, *_COMPOSED)}
    for k in (_FLAG, *_COMPOSED):
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
# Resolver semantics (parse-with-fallback, default OFF).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("val", ["1", "true", "TRUE", "yes", "on", "On"])
def test_resolver_truthy_on(val):
    os.environ[_FLAG] = val
    assert resolve_column_extract_mode() is True
    # Extraction implies the column-major block sort — the two must not disagree.
    assert resolve_column_order_mode() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "garbage", "2x"])
def test_resolver_falsey_or_garbage_off(val):
    os.environ[_FLAG] = val
    assert resolve_column_extract_mode() is False


def test_resolver_unset_off():
    assert resolve_column_extract_mode() is False


def test_deploy_profile_enables_column_extract():
    os.environ["SEMANTIK_DEPLOY_PROFILE"] = "1"
    assert resolve_column_extract_mode() is True
    assert resolve_column_order_mode() is True


# ---------------------------------------------------------------------------
# Synthetic word fixtures (mimic pdfplumber extract_words dicts).
# ---------------------------------------------------------------------------


def _word(text, x0, top, *, w=40.0, h=10.0):
    return {
        "text": text,
        "x0": float(x0),
        "x1": float(x0 + w),
        "top": float(top),
        "bottom": float(top + h),
        "fontname": "Times",
        "size": 10.0,
    }


def _two_column_words():
    """Two columns sharing baselines on a LETTER page (page_w=612).

    Left col at x0=72, right col at x0=340 (gap 268 >> gutter 0.06*612=36.7).
    Each of 3 rows carries a left + a right word at the same ``top`` — the exact
    cross-column baseline that the legacy full-width join fuses into one string.
    """
    rows = [(100.0, "LEFT1", "RIGHT1"), (130.0, "LEFT2", "RIGHT2"), (160.0, "LEFT3", "RIGHT3")]
    words = []
    for top, l, r in rows:
        words.append(_word(l, 72.0, top))
        words.append(_word(r, 340.0, top))
    return words


def _sorted(words):
    # Mirror the (round(top), x0) pre-sort _pdfplumber_page applies before
    # dispatching to the assembly helpers.
    return sorted(words, key=lambda w: (round(float(w["top"]), 0), float(w["x0"])))


def _texts(blocks):
    return [b["text"] for b in blocks]


def test_two_column_columns_contiguous_and_ordered():
    words = _sorted(_two_column_words())
    blocks = _assemble_text_blocks_columnaware(words, [], 612.0)
    texts = _texts(blocks)
    # Left column block(s) come before right column block(s).
    left_tokens = {"LEFT1", "LEFT2", "LEFT3"}
    right_tokens = {"RIGHT1", "RIGHT2", "RIGHT3"}
    last_left = max(i for i, t in enumerate(texts) if any(tok in t for tok in left_tokens))
    first_right = min(i for i, t in enumerate(texts) if any(tok in t for tok in right_tokens))
    assert last_left < first_right, texts
    # No emitted block fuses a col-1 and a col-2 token (the anti-salad invariant).
    for t in texts:
        has_left = any(tok in t for tok in left_tokens)
        has_right = any(tok in t for tok in right_tokens)
        assert not (has_left and has_right), f"cross-column fusion: {t!r}"
    # Every source token survives (segmentation/order only, never a drop).
    joined = " ".join(texts)
    for tok in left_tokens | right_tokens:
        assert tok in joined


def test_single_column_byte_identical_on_vs_off():
    # A genuinely single-column page: all words left-aligned near x=72, one word
    # per line. The column guard must collapse to one column so ON == OFF.
    words = _sorted(
        [
            _word("Alpha", 72.0, 100.0),
            _word("Beta", 73.0, 120.0),
            _word("Gamma", 71.0, 140.0),
            _word("Delta", 72.5, 160.0),
        ]
    )
    off = _assemble_text_blocks_singlewidth(list(words), [])
    on = _assemble_text_blocks_columnaware(list(words), [], 612.0)
    assert on == off  # byte-identical block dicts (the load-bearing no-op proof)
    assert _texts(off) == ["Alpha", "Beta", "Gamma", "Delta"]


def test_full_width_header_stays_one_block():
    # A masthead line spanning the full page width (words evenly spaced, NO
    # gutter-sized internal gap) ABOVE a 2-column body. It must stay ONE block,
    # not scatter across the two bands.
    header = [
        _word("FEDERAL", 72.0, 40.0, w=90.0),
        _word("REGISTER", 172.0, 40.0, w=90.0),
        _word("VOL", 272.0, 40.0, w=90.0),
        _word("NO", 372.0, 40.0, w=90.0),
        _word("2026", 472.0, 40.0, w=60.0),
    ]
    words = _sorted(header + _two_column_words())
    blocks = _assemble_text_blocks_columnaware(words, [], 612.0)
    texts = _texts(blocks)
    # Exactly one block carries every header token, contiguous, unscattered.
    header_blocks = [t for t in texts if "FEDERAL" in t]
    assert len(header_blocks) == 1, texts
    assert header_blocks[0] == "FEDERAL REGISTER VOL NO 2026"
    # And it leads (a top masthead renders before the column bodies).
    assert texts[0] == "FEDERAL REGISTER VOL NO 2026"


def test_table_words_do_not_seed_a_spurious_gutter():
    # A single-column prose page with a wide 2-cell "table" row. The table words
    # are excluded from gutter SEEDING, so they must not spawn a column that
    # scatters the prose. The prose all shares one band.
    table_bbox = (60.0, 90.0, 560.0, 115.0)
    words = _sorted(
        [
            _word("PROSE1", 72.0, 60.0),
            _word("PROSE2", 72.0, 130.0),
            _word("PROSE3", 72.0, 160.0),
            # wide table row (would otherwise seed a left/right gutter):
            _word("CELLA", 72.0, 100.0),
            _word("CELLB", 500.0, 100.0),
        ]
    )
    blocks = _assemble_text_blocks_columnaware(words, [table_bbox], 612.0)
    texts = _texts(blocks)
    # Table row is skipped (inside the bbox); prose stays in document order.
    assert texts == ["PROSE1", "PROSE2", "PROSE3"], texts


# ---------------------------------------------------------------------------
# Cache key: flag-off byte-identical to historic; flag-on appends |col1.
# ---------------------------------------------------------------------------


class _FakeStat:
    st_size = 1234
    st_mtime = 1_700_000_000.0


class _FakePath:
    def stat(self):
        return _FakeStat()

    def resolve(self):
        return "/abs/fixture.pdf"


def _key():
    return extract_shared._compute_extract_cache_key(_FakePath())


def test_cache_key_flag_off_has_no_col_salt():
    # Flag off -> the key must NOT contain a col salt (byte-identical to historic).
    off = _key()
    # Sanity: recomputing the key raw ourselves for the flag-off shape.
    import hashlib

    fig_key = "fig1" if extract_shared._detect_figures_enabled() else "fig0"
    expected_raw = (
        f"v{extract_shared.EXTRACT_CACHE_VERSION}|{fig_key}|"
        f"/abs/fixture.pdf|1234|1700000000"
    )
    expected = hashlib.sha256(expected_raw.encode()).hexdigest()[:24]
    assert off == expected


def test_cache_key_flag_on_appends_col2():
    off = _key()
    os.environ[_FLAG] = "1"
    on = _key()
    assert on != off
    # The on-key is exactly the off shape with |col2 appended into the salt band
    # (bumped from |col1 when the Tesseract path became column-aware — prong 1).
    import hashlib

    fig_key = "fig1" if extract_shared._detect_figures_enabled() else "fig0"
    expected_raw = (
        f"v{extract_shared.EXTRACT_CACHE_VERSION}|{fig_key}|col2|"
        f"/abs/fixture.pdf|1234|1700000000"
    )
    expected = hashlib.sha256(expected_raw.encode()).hexdigest()[:24]
    assert on == expected
