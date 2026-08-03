"""Regression: SEMANTIK_VLM_FUSION deterministically fuses the P0 VLM markdown
source onto the tesseract line blocks (P1 lane).

The VLM endpoint is NEVER called here — P1 consumes the pinned ``page['vlm']``
contract (P0 = SEMANTIK_VLM_EXTRACT), so every test supplies synthetic VLM
lines / a synthetic ``page['vlm']`` dict. No pypdfium2 render, no real OCR, no
model load. CPU-only.

NB: these tests live under ``semantik_structure/tests/`` (the ACTUAL home of the
``test_ocr_render_scale.py`` / ``test_tesseract_config.py`` precedent the lane
map referenced — the map's "semantik_structure/ root" was a mislocation).
"""

from __future__ import annotations

import copy

import pytest

from semantik_structure import extract_shared
from semantik_structure.vlm_fusion import (
    SEMANTIK_VLM_COLLAPSE_REPETITION_ENV,
    SEMANTIK_VLM_FUSION_ENV,
    SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV,
    SEMANTIK_VLM_ORDER_DIVERGENCE_ENV,
    _REPEAT_KEEP_CYCLES,
    _collapse_degenerate_repetition,
    fuse_page,
    resolve_collapse_repetition_mode,
    resolve_vlm_fusion_mode,
    resolve_vlm_order_authoritative_mode,
    resolve_vlm_order_divergence_floor,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _tb(x0, y0, x1, y1, text, *, conf=0.9, fs=12.0):
    return {
        "bbox": [float(x0), float(y0), float(x1), float(y1)],
        "text": text,
        "font_size": fs,
        "font_name": None,
        "is_bold": False,
        "is_italic": False,
        "confidence": conf,
    }


# --------------------------------------------------------------------------
# resolver: parse-with-fallback, default OFF
# --------------------------------------------------------------------------


def test_resolver_default_off(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_FUSION_ENV, raising=False)
    assert resolve_vlm_fusion_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "ON", "  True "])
def test_resolver_truthy_on(monkeypatch, val):
    monkeypatch.setenv(SEMANTIK_VLM_FUSION_ENV, val)
    assert resolve_vlm_fusion_mode() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "garbage", "2x"])
def test_resolver_falsey_and_garbage_off(monkeypatch, val):
    monkeypatch.setenv(SEMANTIK_VLM_FUSION_ENV, val)
    assert resolve_vlm_fusion_mode() is False


# --------------------------------------------------------------------------
# aligner wrapper: MATCH / MERGE / SPLIT + lossless bucket accounting
# --------------------------------------------------------------------------


def test_one_to_one_match():
    tess = [_tb(10, 10, 100, 20, "the quick brown fox")]
    vlm = ["the quick brown fox"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["matched"] == 1
    assert stats["tesseract_only"] == 0 and stats["vlm_only"] == 0
    assert len(fused) == 1
    assert fused[0]["fusion"] == "vlm+tesseract"
    assert fused[0]["text"] == "the quick brown fox"
    assert fused[0]["vlm_coverage"] == "whole_block"


def test_k_tesseract_to_one_vlm_merge():
    # Two wrapped OCR print-lines fold into one VLM logical line (the case a
    # greedy monotonic matcher cannot express).
    tess = [
        _tb(10, 10, 100, 20, "the quick brown"),
        _tb(10, 22, 100, 32, "fox jumps over"),
    ]
    vlm = ["the quick brown fox jumps over"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["merge"] == 1
    assert stats["tesseract_only"] == 0 and stats["vlm_only"] == 0
    # One fused block spanning the union bbox of BOTH tesseract lines.
    aligned = [b for b in fused if b["fusion"] == "vlm+tesseract"]
    assert len(aligned) == 1
    assert aligned[0]["bbox"] == [10.0, 10.0, 100.0, 32.0]
    assert aligned[0]["text"] == "the quick brown fox jumps over"
    assert aligned[0]["vlm_coverage"] == "whole_block"


def test_one_tesseract_to_k_vlm_split_flag_off_joins(monkeypatch):
    # LEGACY (SEMANTIK_FUSION_LINE_UNITS=0): one OCR line unfolded into two VLM
    # lines (a fused title + body) joins into ONE block; the VLM head hint covers
    # only a PREFIX. This is the byte-identical revert leg of Fix B.
    monkeypatch.setenv("SEMANTIK_FUSION_LINE_UNITS", "0")
    tess = [_tb(10, 10, 200, 20, "Introduction Suppose a stone falls down")]
    vlm = ["# Introduction", "Suppose a stone falls down"]
    fused, stats = fuse_page(tess, vlm, (300.0, 300.0))
    assert stats["split"] == 1
    aligned = [b for b in fused if b["fusion"] == "vlm+tesseract"]
    assert len(aligned) == 1
    # markdown '#' stripped from TEXT; both VLM lines joined onto the one bbox.
    assert aligned[0]["text"] == "Introduction Suppose a stone falls down"
    assert aligned[0]["vlm_coverage"] == "prefix"
    # RAW markdown (markers preserved) rides vlm_md for the P2 hint channel.
    assert aligned[0]["vlm_md"].splitlines()[0] == "# Introduction"


def test_split_unit_emits_one_block_per_vlm_line(monkeypatch):
    # Fix B DEFAULT (SEMANTIK_FUSION_LINE_UNITS on): one tesseract mega-line
    # aligning to 4 synthetic VLM lines (label / stem / table row / note) emits
    # 4 fused blocks IN ORDER, each whole_block coverage, over monotone union
    # y-bands — instead of one space-joined mega-block that destroys the unit
    # boundaries (the p127 EXAMPLE-glued-to-Solution defect).
    monkeypatch.delenv("SEMANTIK_FUSION_LINE_UNITS", raising=False)
    tess = [_tb(10, 10, 200, 50, "LABEL stem row note")]
    vlm = ["## LABEL", "the stem sentence here", "row one two three", "a closing note"]
    fused, stats = fuse_page(tess, vlm, (300.0, 300.0))
    assert stats["split"] == 1  # ONE aligned unit (count is per-unit, not block)
    aligned = [b for b in fused if b["fusion"] == "vlm+tesseract"]
    assert len(aligned) == 4
    # In VLM order, markdown structure stripped, each one whole VLM line.
    assert [b["text"] for b in aligned] == [
        "LABEL",
        "the stem sentence here",
        "row one two three",
        "a closing note",
    ]
    assert all(b["vlm_coverage"] == "whole_block" for b in aligned)
    # Monotone, non-overlapping y-bands inside the tesseract union [10, 50].
    y0s = [b["bbox"][1] for b in aligned]
    y1s = [b["bbox"][3] for b in aligned]
    assert y0s == sorted(y0s)
    assert all(10.0 <= y0 < y1 <= 50.0 for y0, y1 in zip(y0s, y1s))
    for i in range(len(aligned) - 1):
        assert y1s[i] == pytest.approx(y0s[i + 1])

    # Flag OFF → exactly ONE joined block (byte-identical legacy).
    monkeypatch.setenv("SEMANTIK_FUSION_LINE_UNITS", "0")
    fused_off, _ = fuse_page(tess, vlm, (300.0, 300.0))
    aligned_off = [b for b in fused_off if b["fusion"] == "vlm+tesseract"]
    assert len(aligned_off) == 1
    assert aligned_off[0]["text"] == "LABEL the stem sentence here row one two three a closing note"


def test_lossless_accounting_every_line_once():
    # A MATCH, a tesseract-only, and a vlm-only — every input line lands in
    # exactly one bucket.
    tess = [
        _tb(10, 10, 100, 20, "aligned shared identical row"),
        _tb(10, 90, 100, 100, "ocronly kkk lll mmm"),
    ]
    vlm = ["aligned shared identical row", "vlmonly ppp qqq rrr"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    # 1 match consumes 1 tess + 1 vlm; 1 tess gap; 1 vlm gap.
    assert stats["matched"] == 1
    assert stats["tesseract_only"] == 1
    assert stats["vlm_only"] == 1
    kinds = sorted(b["fusion"] for b in fused)
    assert kinds == ["tesseract-only", "vlm+tesseract", "vlm-only-flagged"]


# --------------------------------------------------------------------------
# math-garble alignment: sim pre-normalization + positional rescue
# --------------------------------------------------------------------------


def test_math_garble_matches_via_sim_prenormalization():
    # Tesseract 'Vab = Va-Vb' (√ read as V) vs VLM LaTeX — near-zero raw token
    # sim, but the LaTeX-strip + confusable folds align them as a MATCH.
    tess = [_tb(10, 10, 100, 20, "Vab = Va-Vb", conf=0.6)]
    vlm = [r"$\sqrt{ab}=\sqrt{a}\cdot\sqrt{b}$"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["matched"] == 1
    assert stats["vlm_only"] == 0 and stats["tesseract_only"] == 0
    # The clean LaTeX is preserved VERBATIM as the fused block text.
    assert fused[0]["text"] == r"$\sqrt{ab}=\sqrt{a}\cdot\sqrt{b}$"
    assert fused[0]["fusion"] == "vlm+tesseract"


def test_gap_pair_positional_rescue_between_anchors():
    # A garbled OCR math line + a VLM-only LaTeX line sit between two clean
    # MATCH anchors → the (pdf-gap, gold-gap) pair is rescued positionally.
    tess = [
        _tb(10, 10, 100, 20, "header line here"),
        _tb(10, 30, 100, 40, "Gm zz garbled soup", conf=0.5),
        _tb(10, 60, 100, 70, "footer line there"),
    ]
    vlm = ["header line here", r"$x^2 + y^2 = z^2$", "footer line there"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["rescued"] == 1
    # The rescued block carries the VLM LaTeX text on the garbled line's bbox.
    rescued = [
        b for b in fused if b["fusion"] == "vlm+tesseract" and "x^2" in b["text"]
    ]
    assert len(rescued) == 1
    assert rescued[0]["bbox"] == [10.0, 30.0, 100.0, 40.0]
    # No verbatim garble survives — it was replaced, not duplicated.
    assert not any("garbled soup" in b["text"] for b in fused)


def _fake_align_result(*, rows, pdf_gap, gold_gap):
    """A minimal aligner-result stand-in for :func:`_reconstruct`.

    Deterministically controls which tess/vlm indices are aligned units vs.
    gaps, decoupling the rescue-emission test from the opaque DP cost model
    (which force-aligns short disjoint runs rather than leaving them as gaps)."""
    import types

    ledger = types.SimpleNamespace(
        pdf_bucket_indices={"pdf_gap": list(pdf_gap)},
        gold_bucket_indices={"gold_gap": list(gold_gap)},
    )
    row_objs = [
        types.SimpleNamespace(
            provenance={"pdf_block_indices": list(pdf)},
            align={"gold_indices": list(gold), "kind": kind, "confidence": conf},
        )
        for pdf, gold, kind, conf in rows
    ]
    return types.SimpleNamespace(ledger=ledger, rows=row_objs)


def test_rescue_run_emits_per_line_blocks(monkeypatch):
    """Fix B: an anchored (tess-run, vlm-run) gap pair with a 3-line VLM run
    emits 3 per-line blocks by default (the p127 unbounded-rescue mega-block
    source), and exactly 1 joined block on the ``=0`` revert leg.

    Driven through :func:`_reconstruct` with a hand-built aligner result so the
    3-line gap is deterministic (the DP aligner will not leave a multi-line gap
    on its own — it force-aligns short disjoint runs)."""
    from semantik_structure.vlm_fusion import _reconstruct

    # tess/vlm indices 0..4; 0 and 4 are MATCH anchors, 1..3 are gaps.
    tess = [
        _tb(10, 10, 100, 20, "header line here"),
        _tb(10, 30, 100, 40, "qq ww ee garble", conf=0.5),
        _tb(10, 42, 100, 52, "rr tt yy garble", conf=0.5),
        _tb(10, 54, 100, 64, "uu ii oo garble", conf=0.5),
        _tb(10, 80, 100, 90, "footer line here"),
    ]
    vlm = [
        "header line here",
        r"$a + b = 1$",
        r"$c + d = 2$",
        r"$e + f = 3$",
        "footer line here",
    ]
    result = _fake_align_result(
        rows=[([0], [0], "matched", 0.9), ([4], [4], "matched", 0.9)],
        pdf_gap=[1, 2, 3],
        gold_gap=[1, 2, 3],
    )

    # DEFAULT (line-units on): 3 rescued per-line blocks.
    monkeypatch.delenv("SEMANTIK_FUSION_LINE_UNITS", raising=False)
    fused, stats = _reconstruct(result, tess, vlm, 200.0, 300.0, 5, 5)
    assert stats["rescued"] == 1  # ONE rescue region (per-region count)
    rescued = [
        b for b in fused if b["fusion"] == "vlm+tesseract" and "=" in b["text"]
    ]
    assert len(rescued) == 3
    assert [b["text"] for b in rescued] == [
        r"$a + b = 1$",
        r"$c + d = 2$",
        r"$e + f = 3$",
    ]
    assert all(b["vlm_coverage"] == "whole_block" for b in rescued)
    # No verbatim garble survives — each garbled tess line was replaced.
    assert not any("garble" in b["text"] for b in fused)

    # Flag OFF → exactly 1 joined rescue block (byte-identical legacy).
    monkeypatch.setenv("SEMANTIK_FUSION_LINE_UNITS", "0")
    fused_off, stats_off = _reconstruct(result, tess, vlm, 200.0, 300.0, 5, 5)
    assert stats_off["rescued"] == 1
    rescued_off = [
        b for b in fused_off if b["fusion"] == "vlm+tesseract" and "=" in b["text"]
    ]
    assert len(rescued_off) == 1
    assert rescued_off[0]["text"] == r"$a + b = 1$ $c + d = 2$ $e + f = 3$"


def test_fuse_page_no_tesseract_emits_vlm_lines(monkeypatch):
    """Fix B (c): a page with VLM lines but NO tesseract blocks emits one
    vlm-only-flagged block per line by default (instead of silently dropping the
    whole page), and drops them all on the ``=0`` revert leg (legacy)."""
    vlm = ["first vlm line", "## second vlm line", "third vlm line"]
    # DEFAULT (line-units on): 3 vlm-only-flagged blocks.
    monkeypatch.delenv("SEMANTIK_FUSION_LINE_UNITS", raising=False)
    fused, stats = fuse_page([], vlm, (200.0, 300.0))
    inserts = [b for b in fused if b["fusion"] == "vlm-only-flagged"]
    assert len(inserts) == 3
    assert [b["text"] for b in inserts] == [
        "first vlm line",
        "second vlm line",  # '##' stripped
        "third vlm line",
    ]
    assert stats["vlm_only"] == 3 and stats.get("vlm_only_page") is True
    assert all(b["confidence"] == 0.30 for b in inserts)
    # Monotone y-bands over the synthetic page span.
    y0s = [b["bbox"][1] for b in inserts]
    assert y0s == sorted(y0s)

    # Flag OFF → legacy silent drop (empty tesseract list, no fused blocks).
    monkeypatch.setenv("SEMANTIK_FUSION_LINE_UNITS", "0")
    fused_off, stats_off = fuse_page([], vlm, (200.0, 300.0))
    assert fused_off == []
    assert stats_off["vlm_only"] == 3 and "vlm_only_page" not in stats_off


# --------------------------------------------------------------------------
# vlm-only insert positioning + tesseract-only verbatim
# --------------------------------------------------------------------------


def test_vlm_only_insert_sorts_between_neighbors():
    tess = [
        _tb(10, 10, 100, 20, "alpha beta gamma"),
        _tb(10, 50, 100, 60, "delta epsilon zeta"),
    ]
    vlm = ["alpha beta gamma", "inserted vlm only line", "delta epsilon zeta"]
    fused, _ = fuse_page(tess, vlm, (200.0, 300.0))
    insert = [b for b in fused if b["fusion"] == "vlm-only-flagged"]
    assert len(insert) == 1
    b = insert[0]
    assert b["confidence"] == 0.30
    # y strictly between prev.bottom (20) and next.top (50).
    assert 20.0 < b["bbox"][1] < b["bbox"][3] < 50.0
    # Under _merge_page's (y0, x0) key it lands at the intended position.
    merged = extract_shared._merge_page(
        {"tesseract": {"text_blocks": fused}}, text_layer_ok=False
    )
    texts = [tb["text"] for tb in merged["text_blocks"]]
    assert texts == [
        "alpha beta gamma",
        "inserted vlm only line",
        "delta epsilon zeta",
    ]


def test_tesseract_only_kept_byte_verbatim():
    tess = [
        _tb(10, 10, 100, 20, "matched line"),
        _tb(10, 90, 100, 100, "ocr only unique zzz", conf=0.71),
    ]
    vlm = ["matched line"]
    fused, _ = fuse_page(tess, vlm, (200.0, 300.0))
    only = [b for b in fused if b["fusion"] == "tesseract-only"]
    assert len(only) == 1
    # text / bbox / confidence untouched from the input block.
    assert only[0]["text"] == "ocr only unique zzz"
    assert only[0]["bbox"] == [10.0, 90.0, 100.0, 100.0]
    assert only[0]["confidence"] == 0.71


def test_multiple_consecutive_vlm_only_inserts_do_not_collide():
    # Distinctive multi-token anchors + long disjoint inserts so the aligner
    # leaves the two inserts as gold-gaps (it SPLIT-absorbs short gaps adjacent
    # to a match; long disjoint gaps stay gaps).
    tess = [
        _tb(10, 10, 100, 20, "alpha beta gamma"),
        _tb(10, 80, 100, 90, "delta epsilon zeta"),
    ]
    vlm = [
        "alpha beta gamma",
        "qqqqq wwwww eeeee rrrrr ttttt",
        "yyyyy uuuuu iiiii ooooo ppppp",
        "delta epsilon zeta",
    ]
    fused, _ = fuse_page(tess, vlm, (200.0, 300.0))
    inserts = sorted(
        (b for b in fused if b["fusion"] == "vlm-only-flagged"),
        key=lambda b: b["bbox"][1],
    )
    assert len(inserts) == 2
    # Strictly increasing, non-overlapping y bands inside (20, 80).
    assert 20.0 < inserts[0]["bbox"][1] < inserts[0]["bbox"][3]
    assert inserts[0]["bbox"][3] <= inserts[1]["bbox"][1]
    assert inserts[1]["bbox"][3] < 80.0


# --------------------------------------------------------------------------
# bbox sanitation — every fused bbox is valid in tesseract IMAGE-PIXEL space
# (x0<x1, y0<y1, clamped to page dims) so no downstream page-crop inverts
# --------------------------------------------------------------------------


def test_sanitize_bbox_corrects_inverted_and_out_of_range():
    from semantik_structure.vlm_fusion import _sanitize_bbox

    # Inverted on BOTH axes → order-swapped.
    assert _sanitize_bbox([515.0, 1971.0, 409.0, 792.0], 2448.0, 3168.0) == [
        409.0, 792.0, 515.0, 1971.0,
    ]
    # Out-of-page (y1 past page_h) → clamped to page_h, still ordered.
    x0, y0, x1, y1 = _sanitize_bbox([100.0, 3000.0, 200.0, 5000.0], 2448.0, 3168.0)
    assert (x0, y0, x1) == (100.0, 3000.0, 200.0)
    assert y1 == 3168.0
    assert y0 < y1


def test_sanitize_bbox_degenerate_and_missing_dims_stay_valid():
    from semantik_structure.vlm_fusion import _sanitize_bbox

    # A zero-extent box at a page edge is nudged inward to a >=1px band.
    x0, y0, x1, y1 = _sanitize_bbox([2448.0, 3168.0, 2448.0, 3168.0], 2448.0, 3168.0)
    assert x0 < x1 <= 2448.0 and y0 < y1 <= 3168.0
    # Missing page dims (0) must NOT collapse the box to the origin.
    assert _sanitize_bbox([10.0, 20.0, 40.0, 60.0], 0.0, 0.0) == [10.0, 20.0, 40.0, 60.0]
    # Non-finite input degrades safely (never propagates NaN/inf).
    out = _sanitize_bbox([float("nan"), 0.0, float("inf"), 10.0], 100.0, 100.0)
    assert all(v == v for v in out) and out[0] < out[2] and out[1] < out[3]


def test_trailing_vlm_only_insert_at_page_foot_is_clamped_in_range():
    # A trailing VLM-only line with NO bounding ``next`` neighbor takes the
    # fallback band ``y_lo + g*_FALLBACK_LINE_H`` — which OVERSHOOTS page_h when
    # the prev anchor's bottom sits at the page foot. This is the live-fire
    # fault path (an out-of-page insert bbox the Stage-5c figure renderer later
    # inverted after its own page-clamp). Sanitation must clamp it in-range.
    page_w, page_h = 200.0, 60.0
    tess = [_tb(10, 10, 100, 60, "alpha beta gamma delta")]  # bottom == page_h
    vlm = ["alpha beta gamma delta", "trailing insert only zzz"]
    fused, _ = fuse_page(tess, vlm, (page_w, page_h))
    insert = [b for b in fused if b["fusion"] == "vlm-only-flagged"]
    assert len(insert) == 1
    x0, y0, x1, y1 = insert[0]["bbox"]
    assert 0.0 <= x0 < x1 <= page_w
    assert 0.0 <= y0 < y1 <= page_h   # clamped despite the fallback overshoot


def test_all_fused_bboxes_valid_in_page_space():
    # Invariant sweep: whatever the alignment produces (aligned unions, rescue
    # unions, interpolated inserts), EVERY fused block ships a well-ordered,
    # in-range bbox.
    page_w, page_h = 200.0, 100.0
    tess = [
        _tb(10, 5, 190, 20, "alpha beta gamma"),
        _tb(10, 40, 190, 55, "delta epsilon zeta"),
        _tb(10, 85, 190, 100, "eta theta iota"),   # bottom == page_h
    ]
    vlm = [
        "alpha beta gamma",
        "inserted middle line one",
        "delta epsilon zeta",
        "eta theta iota",
        "trailing tail insert two",
    ]
    fused, _ = fuse_page(tess, vlm, (page_w, page_h))
    for b in fused:
        x0, y0, x1, y1 = b["bbox"]
        assert 0.0 <= x0 < x1 <= page_w, b
        assert 0.0 <= y0 < y1 <= page_h, b


# --------------------------------------------------------------------------
# divergence math + stats match ledger buckets
# --------------------------------------------------------------------------


def test_divergence_equals_gap_lines_over_total():
    # 2 matches bracket 2 VLM-only gaps → gaps=2, total = 2 tess + 4 vlm = 6.
    tess = [
        _tb(10, 10, 100, 20, "alpha beta gamma"),
        _tb(10, 80, 100, 90, "delta epsilon zeta"),
    ]
    vlm = [
        "alpha beta gamma",
        "qqqqq wwwww eeeee rrrrr ttttt",
        "yyyyy uuuuu iiiii ooooo ppppp",
        "delta epsilon zeta",
    ]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["tesseract_only"] == 0
    assert stats["vlm_only"] == 2
    assert stats["divergence"] == pytest.approx((0 + 2) / (2 + 4))


def test_stats_counts_match_ledger_bucket_sizes():
    # A pure-disjoint page: the raw tesseract_only / vlm_only stats ARE the
    # aligner's pdf_gap / gold_gap bucket sizes (no match, no rescue → exact).
    tess = [_tb(10, 10, 100, 20, "tessuniq kkk lll mmm")]
    vlm = ["vlmuniq ppp qqq rrr"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["tesseract_only"] == 1
    assert stats["vlm_only"] == 1
    assert stats["matched"] == 0
    assert stats["divergence"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# markdown-strip contract
# --------------------------------------------------------------------------


def test_markdown_structure_stripped_latex_preserved():
    tess = [
        _tb(10, 10, 100, 20, "heading text"),
        _tb(10, 30, 100, 40, "bullet item text"),
        _tb(10, 50, 100, 60, "the value is Vab"),
    ]
    vlm = ["## heading text", "- bullet item text", r"the value is $\sqrt{ab}$"]
    fused, _ = fuse_page(tess, vlm, (200.0, 300.0))
    by_y = sorted(fused, key=lambda b: b["bbox"][1])
    assert by_y[0]["text"] == "heading text"  # '##' stripped
    assert by_y[1]["text"] == "bullet item text"  # '- ' stripped
    assert by_y[2]["text"] == r"the value is $\sqrt{ab}$"  # LaTeX kept verbatim


def test_answer_grid_row_preserves_exercise_numbers(monkeypatch):
    """A VLM line with MULTIPLE 'N.' markers is an answer-grid / exercise row —
    each number is a CONTENT exercise label, not a list marker, so the leading
    number must be PRESERVED (regression: '37. 84' was stripped to '84').

    Multi-marker preservation holds in BOTH modes. The SINGLE-marker strip only
    applies on the legacy ``SEMANTIK_FUSION_KEEP_ORDERED_MARKERS=0`` leg (Fix A
    flips the default so a single-marker textbook exercise number survives)."""
    from SemantiK.semantik_structure.vlm_fusion import _strip_markdown_structure

    # Multi-marker grid rows: leading exercise number kept verbatim (both modes).
    monkeypatch.delenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", raising=False)
    assert _strip_markdown_structure("37. 84 38. 9,696 39. 75") == "37. 84 38. 9,696 39. 75"
    assert _strip_markdown_structure("46. 550 47. 22,335 48. 39,075") == (
        "46. 550 47. 22,335 48. 39,075"
    )

    # Legacy leg (=0): a genuine SINGLE-item ordered-list line still strips.
    monkeypatch.setenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", "0")
    assert _strip_markdown_structure("1. First point of the argument") == (
        "First point of the argument"
    )
    assert _strip_markdown_structure("3. See the theorem") == "See the theorem"
    # Multi-marker rows still preserved on the legacy leg.
    assert _strip_markdown_structure("37. 84 38. 9,696 39. 75") == "37. 84 38. 9,696 39. 75"


def test_single_marker_exercise_number_preserved(monkeypatch):
    """Fix A DEFAULT (SEMANTIK_FUSION_KEEP_ORDERED_MARKERS on): a single-marker
    line-leading 'N.' in a textbook is a CONTENT exercise/answer NUMBER, so it
    survives instead of being stripped as a markdown list marker (the p103
    '38 exercise numbers dropped' defect). '=0' restores the legacy strip."""
    from SemantiK.semantik_structure.vlm_fusion import _strip_markdown_structure

    # Synthetic per-line exercise lines (fabricated numbers/expressions).
    lines = [
        ("301. $\\frac{1}{2}+\\frac{1}{3}$", "301. $\\frac{1}{2}+\\frac{1}{3}$"),
        ("302. $-\\sqrt{49}$", "302. $-\\sqrt{49}$"),
        ("317) $x^2 - 4$", "317) $x^2 - 4$"),
    ]
    # DEFAULT (mode on): the leading number is KEPT.
    monkeypatch.delenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", raising=False)
    for src, want in lines:
        assert _strip_markdown_structure(src) == want

    # Legacy (=0): the single leading marker is stripped (exercise number lost).
    monkeypatch.setenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", "0")
    assert _strip_markdown_structure("301. $\\frac{1}{2}+\\frac{1}{3}$") == (
        "$\\frac{1}{2}+\\frac{1}{3}$"
    )
    assert _strip_markdown_structure("317) $x^2 - 4$") == "$x^2 - 4$"


def test_fuse_page_exercise_numbers_survive_in_fused_units(monkeypatch):
    """End-to-end (default flags): a per-line VLM exercise bank aligning to
    garbled tesseract keeps every exercise NUMBER in the fused block text —
    the p103 silent-loss regression, at the fuse_page level."""
    monkeypatch.delenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", raising=False)
    monkeypatch.delenv("SEMANTIK_FUSION_LINE_UNITS", raising=False)
    tess = [_tb(10, 10 + 12 * i, 200, 20 + 12 * i, f"{n} garbld") for i, n in enumerate((301, 302, 303))]
    vlm = ["301. first exercise body", "302. second exercise body", "303. third exercise body"]
    fused, _ = fuse_page(tess, vlm, (300.0, 300.0))
    joined = " ".join(b.get("text", "") for b in fused)
    for n in ("301.", "302.", "303."):
        assert n in joined


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_deterministic_repeated_calls():
    tess = [
        _tb(10, 10, 100, 20, "the quick brown"),
        _tb(10, 22, 100, 32, "fox jumps"),
        _tb(10, 50, 100, 60, "Vab = Va-Vb"),
    ]
    vlm = ["the quick brown fox jumps", r"$\sqrt{ab}=\sqrt{a}\cdot\sqrt{b}$", "extra vlm"]
    out1, s1 = fuse_page(copy.deepcopy(tess), list(vlm), (200.0, 300.0))
    out2, s2 = fuse_page(copy.deepcopy(tess), list(vlm), (200.0, 300.0))
    assert out1 == out2
    assert s1 == s2


# --------------------------------------------------------------------------
# pathological-page guard
# --------------------------------------------------------------------------


def test_pathological_page_skips_fusion_deterministically():
    # n*m over the 250x250 ceiling → fusion skipped, tesseract verbatim.
    tess = [_tb(0, i, 10, i + 1, f"line {i}") for i in range(300)]
    vlm = [f"vlm {i}" for i in range(300)]
    fused, stats = fuse_page(tess, vlm, (200.0, 4000.0))
    assert stats["skipped_too_large"] is True
    assert stats["divergence"] == 1.0
    # Blocks returned unchanged (byte-identical to the input tesseract blocks).
    assert fused == tess
    assert not any("fusion" in b for b in fused)


# --------------------------------------------------------------------------
# empty-source no-ops
# --------------------------------------------------------------------------


def test_empty_vlm_is_noop_verbatim_tesseract():
    tess = [_tb(10, 10, 100, 20, "line a"), _tb(10, 30, 100, 40, "line b")]
    fused, stats = fuse_page(tess, [], (200.0, 300.0))
    assert stats["tesseract_only"] == 2
    assert stats["matched"] == 0
    assert [b["text"] for b in fused] == ["line a", "line b"]
    assert all(b["fusion"] == "tesseract-only" for b in fused)


# --------------------------------------------------------------------------
# degenerate-repetition tripwire (SEMANTIK_VLM_COLLAPSE_REPETITION)
# --------------------------------------------------------------------------


def test_collapse_resolver_default_on(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, raising=False)
    assert resolve_collapse_repetition_mode() is True


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "garbage", ""])
def test_collapse_resolver_on_unless_explicit_falsey(monkeypatch, val):
    # Default-ON parse posture: unset / truthy / garbage / blank → on.
    monkeypatch.setenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, val)
    assert resolve_collapse_repetition_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF"])
def test_collapse_resolver_explicit_falsey_off(monkeypatch, val):
    monkeypatch.setenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, val)
    assert resolve_collapse_repetition_mode() is False


def test_collapse_period2_cycle_from_manipulative_figure():
    # The real defect: a "x + x + x + …" algebra-tiles transcription. Domain-
    # agnostic period-2 detection collapses it to a bounded representative.
    run = "x + " * 4047 + "x"
    text = f"[4x + 7x + x] [{run}] equations into English phrases."
    out = _collapse_degenerate_repetition(text)
    # The 8k-token poison is gone; a bounded marker + representative remain.
    assert "repeated" in out
    assert out.count("x +") <= 6  # only the kept representative cycles survive
    assert len(out.split()) < 30
    # Surrounding real content is preserved.
    assert "equations into English phrases." in out
    assert "4x" in out


def test_collapse_period1_single_glyph_run():
    text = "before " + ("0 " * 100).strip() + " after"
    out = _collapse_degenerate_repetition(text)
    assert "repeated 100 times" in out
    assert out.startswith("before ")
    assert out.endswith("after")


def test_collapse_count_marker_is_accurate():
    text = "lead " + ("- " * 50).strip()
    out = _collapse_degenerate_repetition(text)
    assert "[repeated 50 times]" in out


# --- false-positive guards: clean / legitimate input stays byte-identical ---


def test_collapse_noop_on_normal_prose():
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Addition combines like terms into a single expression."
    )
    out = _collapse_degenerate_repetition(text)
    assert out is text  # same object → byte-identical


def test_collapse_noop_on_legitimate_short_repeated_list():
    # A real short repeated list (well under the generous cycle floor) is kept.
    text = "Scores: 5 5 5 5 5 5 5 5 across the eight trials."
    out = _collapse_degenerate_repetition(text)
    assert out is text


def test_collapse_noop_on_long_word_repetition_below_floor():
    # Even a longish repeated token stays byte-identical below the cycle floor.
    text = " ".join(["really"] * 20)
    out = _collapse_degenerate_repetition(text)
    assert out is text


def test_collapse_noop_on_long_token_even_when_repeated_past_floor():
    # A LONG token repeated past the floor is left alone (only short tokens are
    # collapse-eligible — conservative, avoids collapsing a repeated real word).
    text = " ".join(["elephant"] * 60)
    out = _collapse_degenerate_repetition(text)
    assert out is text


def test_collapse_deterministic():
    text = "x + " * 100
    assert _collapse_degenerate_repetition(text) == _collapse_degenerate_repetition(text)


# --- fuse_page integration + flag gating ---


def test_fuse_page_collapses_degenerate_block_by_default(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, raising=False)
    run = "x + " * 200 + "x"
    tess = [_tb(10, 10, 100, 20, run)]
    vlm = [run]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["repetition_collapsed"] == 1
    assert "repeated" in fused[0]["text"]
    assert len(fused[0]["text"].split()) < 30


def test_fuse_page_flag_off_keeps_degenerate_block_verbatim(monkeypatch):
    monkeypatch.setenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, "0")
    run = "x + " * 200 + "x"
    tess = [_tb(10, 10, 100, 20, run)]
    vlm = [run]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert "repetition_collapsed" not in stats
    # The (poison) text ships verbatim — the =0 escape hatch.
    assert fused[0]["text"].count("x +") > 100


def test_fuse_page_clean_block_byte_identical_with_collapse_on(monkeypatch):
    # Collapse-on must be byte-identical to collapse-off on NON-pathological
    # fused text (the byte-identical-on-clean-input guarantee).
    tess = [
        _tb(10, 10, 100, 20, "the quick brown"),
        _tb(10, 22, 100, 32, "fox jumps over"),
    ]
    vlm = ["the quick brown fox jumps over"]
    monkeypatch.setenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, "1")
    on_fused, on_stats = fuse_page(copy.deepcopy(tess), list(vlm), (200.0, 300.0))
    monkeypatch.setenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, "0")
    off_fused, off_stats = fuse_page(copy.deepcopy(tess), list(vlm), (200.0, 300.0))
    assert [b["text"] for b in on_fused] == [b["text"] for b in off_fused]
    # The only stats delta is the additive zero-count tally.
    assert on_stats.get("repetition_collapsed") == 0
    off_stats["repetition_collapsed"] = 0
    assert on_stats == off_stats


# --------------------------------------------------------------------------
# SENTENCE / multi-word arm — the sentence-granularity blind spot
# --------------------------------------------------------------------------
#
# NB: no hardcoded corpus sentence — every unit below is a generic synthetic
# clause, keying on the repetition STRUCTURE, never a specific sentence.


def test_collapse_sentence_level_loop_collapses():
    # A whole SENTENCE looped ~33x (the token arm's blind spot — the repeating
    # unit is a long multi-word clause, not a <=4-char token). Collapses to a
    # bounded representative + an accurate count marker.
    unit = "There are two operations here and each one matters greatly. "
    blob = unit * 33
    out = _collapse_degenerate_repetition(blob)
    assert "[repeated 33 times]" in out
    # Only the kept representative copies of the clause survive.
    assert out.count("matters greatly.") <= _REPEAT_KEEP_CYCLES
    # The ~330-token poison blob is gone.
    assert len(out.split()) < 40


def test_collapse_sentence_count_marker_accurate():
    unit = "The measured value stays exactly the same here. "
    out = _collapse_degenerate_repetition(unit * 9)
    assert "[repeated 9 times]" in out


def test_collapse_sentence_deterministic():
    blob = "Everything here is quietly repeating once again now. " * 20
    assert (
        _collapse_degenerate_repetition(blob)
        == _collapse_degenerate_repetition(blob)
    )


def test_collapse_both_arms_compose():
    # A doc with BOTH a token loop AND a sentence loop collapses BOTH in one pass.
    token_run = "x + " * 200 + "x"
    sentence = "The result does not change between the trials. "
    blob = f"{token_run} lead {sentence * 10} tail here."
    out = _collapse_degenerate_repetition(blob)
    assert "[repeated 200 times]" in out  # token arm
    assert "[repeated 10 times]" in out  # sentence arm
    assert out.count("x +") <= 6
    assert out.count("between the trials.") <= _REPEAT_KEEP_CYCLES
    # Surrounding real content survives.
    assert "lead" in out.split()
    assert out.rstrip().endswith("tail here.")


# --- false-positive guards: legitimate repetition stays byte-identical ------


def test_collapse_noop_on_two_time_phrase_repeat():
    # An incidental 2x phrase restatement is far below the >=6 sentence floor;
    # padded past the fast-path space guard so the scan actually runs.
    phrase = "Please review the attached summary document very carefully today. "
    text = (
        phrase * 2
        + "Then something completely different follows in the remaining body text here."
    )
    out = _collapse_degenerate_repetition(text)
    assert out is text  # same object -> byte-identical


def test_collapse_noop_on_repeated_short_table_labels():
    # A table column of repeated SHORT labels: each unit is a 1-word label whose
    # interior/terminal tokens themselves end in '.', so the sentence arm's
    # interior-terminal-punct guard rejects it (and the token arm needs
    # <=4-char tokens). Byte-identical even at a high repeat count.
    text = "Column status values: " + ("Pending. " * 30).strip()
    out = _collapse_degenerate_repetition(text)
    assert out is text


def test_collapse_noop_on_numbered_list_repeat():
    # A numbered-list-ish run: the repeated unit's interior tokens end in '.'
    # (the 'N.' markers), so the sentence arm never fires. Byte-identical.
    text = "Items follow below here now: " + ("1. 2. 3. 4. " * 8).strip()
    out = _collapse_degenerate_repetition(text)
    assert out is text


def test_collapse_noop_on_normal_prose_past_guard():
    # Long, varied prose (past the fast-path space guard) is byte-identical —
    # no 4+-word clause repeats 6x consecutively.
    text = (
        "The quick brown fox jumps over the lazy dog while the sun sets slowly. "
        "Addition combines like terms into a single simplified expression today. "
        "Meanwhile the river flows gently past the old stone bridge downstream."
    )
    out = _collapse_degenerate_repetition(text)
    assert out is text


def test_fuse_page_collapses_sentence_loop_by_default(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, raising=False)
    unit = "The answer to the exercise is always the same value. "
    blob = unit * 33
    tess = [_tb(10, 10, 100, 20, blob)]
    vlm = [blob]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["repetition_collapsed"] == 1
    assert "[repeated 33 times]" in fused[0]["text"]
    assert len(fused[0]["text"].split()) < 40


def test_fuse_page_flag_off_keeps_sentence_loop_verbatim(monkeypatch):
    monkeypatch.setenv(SEMANTIK_VLM_COLLAPSE_REPETITION_ENV, "0")
    unit = "The answer to the exercise is always the same value. "
    blob = unit * 33
    tess = [_tb(10, 10, 100, 20, blob)]
    vlm = [blob]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert "repetition_collapsed" not in stats
    # The looped sentence ships verbatim — the =0 escape hatch.
    assert fused[0]["text"].count("the same value.") > 20


# --------------------------------------------------------------------------
# seam: _apply_vlm_fusion (mockable boundary, no extractors)
# --------------------------------------------------------------------------


def _synthetic_page():
    return {
        "page_num": 3,
        "tesseract": {
            "text_blocks": [
                _tb(10, 10, 100, 20, "the quick brown"),
                _tb(10, 22, 100, 32, "fox jumps"),
            ]
        },
        "tesseract_width": 200.0,
        "tesseract_height": 300.0,
        "vlm": {
            "markdown": "the quick brown fox jumps",
            "text_blocks": [
                {"bbox": None, "text": "the quick brown fox jumps"},
            ],
        },
    }


def test_seam_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_FUSION_ENV, raising=False)
    page = _synthetic_page()
    before = copy.deepcopy(page)
    extract_shared._apply_vlm_fusion(page)
    assert page == before  # byte-identical: no rewrite, no stats keys


def test_seam_noop_when_no_vlm_source(monkeypatch):
    monkeypatch.setenv(SEMANTIK_VLM_FUSION_ENV, "1")
    page = _synthetic_page()
    del page["vlm"]  # P0 not on for this page
    before = copy.deepcopy(page)
    extract_shared._apply_vlm_fusion(page)
    assert page == before


def test_seam_fuses_when_flag_on_and_vlm_present(monkeypatch):
    monkeypatch.setenv(SEMANTIK_VLM_FUSION_ENV, "1")
    page = _synthetic_page()
    extract_shared._apply_vlm_fusion(page)
    fused = page["tesseract"]["text_blocks"]
    assert len(fused) == 1  # the two wrapped lines MERGED
    assert fused[0]["fusion"] == "vlm+tesseract"
    assert fused[0]["text"] == "the quick brown fox jumps"
    assert page["vlm_divergence"] == 0.0
    assert page["vlm_fusion_stats"]["merge"] == 1


# --------------------------------------------------------------------------
# provenance invariance: fused blocks stay provenance='tesseract'; the fusion
# key never leaks into the council classifier input
# --------------------------------------------------------------------------


def test_provenance_stays_tesseract_and_fusion_key_not_in_source(monkeypatch):
    monkeypatch.setenv(SEMANTIK_VLM_FUSION_ENV, "1")
    page = _synthetic_page()
    extract_shared._apply_vlm_fusion(page)
    merged = extract_shared._merge_page(page, text_layer_ok=False)
    for b in merged["text_blocks"]:
        assert b["provenance"] == ["tesseract"]
    # blocks_from_shared maps provenance → RawBlock.source; must stay
    # 'tesseract' (the features.py council-input marker), never 'fusion'.
    from semantik_structure.extract import blocks_from_shared

    shared = {
        "pdf_path": "x.pdf",
        "metadata": {},
        "pages": [
            {
                "page_num": 3,
                "width": 200.0,
                "height": 300.0,
                "tesseract_width": 200.0,
                "tesseract_height": 300.0,
                "merged": merged,
            }
        ],
    }
    raws = blocks_from_shared(shared)
    assert raws, "expected at least one raw block"
    for rb in raws:
        assert rb.source == "tesseract"


def test_fusion_provenance_threads_onto_rawblock(monkeypatch):
    # Fix 3 plumbing — the merged block's ``fusion`` key rides onto
    # RawBlock.fusion (distinct from ``source``, forced 'tesseract'), so
    # structure typing can keep a ``vlm-only-flagged`` insert off the figure
    # track. A ``vlm-only-flagged`` insert + a ``vlm+tesseract`` aligned block
    # both appear on the same page.
    monkeypatch.setenv(SEMANTIK_VLM_FUSION_ENV, "1")
    page = {
        "page_num": 3,
        "tesseract": {
            "text_blocks": [
                _tb(10, 10, 100, 20, "alpha beta gamma"),
                _tb(10, 50, 100, 60, "delta epsilon zeta"),
            ]
        },
        "tesseract_width": 200.0,
        "tesseract_height": 300.0,
        "vlm": {
            "markdown": "",
            "text_blocks": [
                {"bbox": None, "text": "alpha beta gamma"},
                {"bbox": None, "text": "inserted vlm only line"},
                {"bbox": None, "text": "delta epsilon zeta"},
            ],
        },
    }
    extract_shared._apply_vlm_fusion(page)
    merged = extract_shared._merge_page(page, text_layer_ok=False)
    from semantik_structure.extract import blocks_from_shared

    shared = {
        "pages": [
            {
                "page_num": 3,
                "width": 200.0,
                "height": 300.0,
                "tesseract_width": 200.0,
                "tesseract_height": 300.0,
                "merged": merged,
            }
        ],
    }
    raws = blocks_from_shared(shared)
    fusions = {rb.text: rb.fusion for rb in raws}
    assert fusions.get("inserted vlm only line") == "vlm-only-flagged"
    # source stays 'tesseract' regardless (council-input marker unchanged).
    assert all(rb.source == "tesseract" for rb in raws)


# --------------------------------------------------------------------------
# cache-key salt: flipping the flag changes the key; OFF == current (P0) key
# --------------------------------------------------------------------------


def test_cache_key_salt_flips_and_off_is_byte_identical(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fake_extract(pdf_path):
        calls["n"] += 1
        return {"pdf_path": str(pdf_path), "metadata": {}, "pages": []}

    monkeypatch.setattr(extract_shared, "extract_shared", fake_extract)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake bytes")
    cache = tmp_path / "cache"

    # Flag OFF → one cache file.
    monkeypatch.delenv(SEMANTIK_VLM_FUSION_ENV, raising=False)
    extract_shared.extract_shared_cached(pdf, cache_dir=cache)
    off_files = set(p.name for p in cache.glob("*.json"))
    assert len(off_files) == 1

    # Flag ON → a DIFFERENT cache file (fresh extraction, no stale-serve).
    monkeypatch.setenv(SEMANTIK_VLM_FUSION_ENV, "1")
    extract_shared.extract_shared_cached(pdf, cache_dir=cache)
    all_files = set(p.name for p in cache.glob("*.json"))
    assert len(all_files) == 2
    assert calls["n"] == 2  # both were misses (distinct keys)


def test_cache_key_ordmark_fgran_salts_append_only_and_off_byte_identical(
    monkeypatch, tmp_path
):
    """Fix A/B cache salts (``|ordmark1`` / ``|fgran1``): fusion-on default
    salts the key; each mode's ``=0`` drops its salt; and NEITHER salt appears
    when fusion is OFF (append-only-when-fusion-on-AND-mode-on)."""
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake bytes")

    def key():
        return extract_shared._compute_extract_cache_key(pdf)

    # Fusion OFF → neither salt present regardless of the mode flags (byte-
    # identical historic no-fusion key).
    monkeypatch.delenv(SEMANTIK_VLM_FUSION_ENV, raising=False)
    monkeypatch.setenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", "1")
    monkeypatch.setenv("SEMANTIK_FUSION_LINE_UNITS", "1")
    off_key = key()
    monkeypatch.setenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", "0")
    monkeypatch.setenv("SEMANTIK_FUSION_LINE_UNITS", "0")
    assert key() == off_key  # fusion-off key is byte-identical either way

    # Fusion ON, both modes default (on) → both salts present.
    monkeypatch.setenv(SEMANTIK_VLM_FUSION_ENV, "1")
    monkeypatch.delenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", raising=False)
    monkeypatch.delenv("SEMANTIK_FUSION_LINE_UNITS", raising=False)
    default_key = key()

    # ordmark=0 → a DIFFERENT key (salt dropped).
    monkeypatch.setenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", "0")
    ordmark_off_key = key()
    assert ordmark_off_key != default_key
    monkeypatch.delenv("SEMANTIK_FUSION_KEEP_ORDERED_MARKERS", raising=False)

    # fgran=0 → a DIFFERENT key (salt dropped).
    monkeypatch.setenv("SEMANTIK_FUSION_LINE_UNITS", "0")
    fgran_off_key = key()
    assert fgran_off_key != default_key
    assert fgran_off_key != ordmark_off_key


# --------------------------------------------------------------------------
# Defect 2 — markdown code-fence strip from the fused block text.
# --------------------------------------------------------------------------


def test_strip_markdown_structure_drops_whole_line_fence():
    from semantik_structure.vlm_fusion import _strip_markdown_structure

    assert _strip_markdown_structure("```markdown") == ""
    assert _strip_markdown_structure("```") == ""
    assert _strip_markdown_structure("```math") == ""
    assert _strip_markdown_structure("  ```  ") == ""


def test_strip_markdown_structure_scrubs_stray_inline_fence():
    from semantik_structure.vlm_fusion import _strip_markdown_structure

    # Whole-page wrapper collapsed onto one line with the heading.
    assert (
        _strip_markdown_structure("```markdown Chapter 9 Roots and Radicals")
        == "Chapter 9 Roots and Radicals"
    )


def test_strip_markdown_structure_keeps_inline_latex():
    from semantik_structure.vlm_fusion import _strip_markdown_structure

    assert _strip_markdown_structure(r"$\sqrt{ab}$") == r"$\sqrt{ab}$"


def test_join_vlm_drops_fence_lines():
    from semantik_structure.vlm_fusion import _join_vlm

    lines = ["```markdown", "Chapter 9 Roots and Radicals", "```"]
    assert _join_vlm(lines, [0, 1, 2]) == "Chapter 9 Roots and Radicals"


# --------------------------------------------------------------------------
# Defect 6 — © -> (c) confusable fold (SCORING ONLY).
# --------------------------------------------------------------------------


def test_fold_confusables_copyright_to_paren_c():
    from semantik_structure.vlm_fusion import _fold_confusables, _score_norm

    assert "(c)" in _fold_confusables("©")
    # A VLM "©" option marker scores identically to the OCR "(c)".
    assert _score_norm("©") == _score_norm("(c)")


def test_fold_confusables_scoring_only_output_untouched():
    # The output-text transform (_strip_markdown_structure) must NOT rewrite ©.
    from semantik_structure.vlm_fusion import _strip_markdown_structure

    assert _strip_markdown_structure("© 2026 Example") == "© 2026 Example"


# --------------------------------------------------------------------------
# Defect B (coordinator follow-up) — LaTeX sectioning-wrapper normalization.
# --------------------------------------------------------------------------


def test_strip_markdown_structure_unwraps_latex_section():
    from semantik_structure.vlm_fusion import _strip_markdown_structure

    assert (
        _strip_markdown_structure(r"\section*{REVIEW EXERCISES}")
        == "REVIEW EXERCISES"
    )
    assert _strip_markdown_structure(r"\subsection*{Key Terms}") == "Key Terms"
    assert _strip_markdown_structure(r"\section{PRACTICE TEST}") == "PRACTICE TEST"


def test_latex_section_inline_mention_and_math_untouched():
    from semantik_structure.vlm_fusion import _strip_markdown_structure

    # An inline \section mention inside prose is never rewritten.
    assert (
        _strip_markdown_structure(r"see \section for details")
        == r"see \section for details"
    )
    # Inline $…$ math LaTeX is preserved verbatim.
    assert _strip_markdown_structure(r"$\sqrt{x}$ and $y^2$") == r"$\sqrt{x}$ and $y^2$"


# --------------------------------------------------------------------------
# Literal HTML-entity text is scrubbed from VLM lines before fusion.
# --------------------------------------------------------------------------


def test_strip_markdown_structure_scrubs_nbsp_entity_runs():
    from semantik_structure.vlm_fusion import _strip_markdown_structure

    # blank table spacing transcribed as entity runs
    assert (
        _strip_markdown_structure("Divide. &nbsp; &nbsp; &nbsp; 45 by 9")
        == "Divide. 45 by 9"
    )
    # double-escaped form unwraps in one pass
    assert (
        _strip_markdown_structure("a &amp;nbsp; &amp;nbsp; b") == "a b"
    )
    # a line that is ONLY entity spacing carries no content
    assert _strip_markdown_structure("&nbsp; &nbsp; &nbsp;") == ""


def test_strip_markdown_structure_decodes_common_entities():
    from semantik_structure.vlm_fusion import _strip_markdown_structure

    assert _strip_markdown_structure("4 &lt; 5 and 6 &gt; 3") == "4 < 5 and 6 > 3"
    assert _strip_markdown_structure("rise &amp; run") == "rise & run"
    # an UNKNOWN lowercase entity is dropped to a space (decode-or-drop)
    assert _strip_markdown_structure("a &copy; b") == "a b"
    # entity-free prose (incl. a bare '&' and the word nbsp) is byte-identical
    assert _strip_markdown_structure("rise & run with nbsp word") == (
        "rise & run with nbsp word"
    )


def test_fusion_carries_clean_text_not_entity_runs():
    # A VLM line with entity spacing fuses onto the tesseract bbox with CLEAN
    # text — the literal entity run never reaches the fused block text.
    tess = [_tb(10, 10, 100, 20, "Divide. 45 by 9 to simplify", conf=0.6)]
    vlm = ["Divide. &nbsp; &nbsp; 45 by 9 to simplify"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert stats["matched"] == 1
    assert fused[0]["text"] == "Divide. 45 by 9 to simplify"
    assert "nbsp" not in fused[0]["text"]


# --------------------------------------------------------------------------
# VLM-authoritative reading-order repair (dense multi-column pages).
# --------------------------------------------------------------------------


def test_order_authoritative_resolver_default_off(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV, raising=False)
    assert resolve_vlm_order_authoritative_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "ON", "  True "])
def test_order_authoritative_resolver_truthy_on(monkeypatch, val):
    monkeypatch.setenv(SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV, val)
    assert resolve_vlm_order_authoritative_mode() is True


@pytest.mark.parametrize("val", ["", "0", "false", "no", "off", "garbage", "2x"])
def test_order_authoritative_resolver_falsey_and_garbage_off(monkeypatch, val):
    # Opt-in posture (mirrors resolve_vlm_fusion_mode): only truthy enables.
    monkeypatch.setenv(SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV, val)
    assert resolve_vlm_order_authoritative_mode() is False


def test_order_divergence_floor_default(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_ORDER_DIVERGENCE_ENV, raising=False)
    assert resolve_vlm_order_divergence_floor() == pytest.approx(0.25)


@pytest.mark.parametrize("val,exp", [("0.4", 0.4), ("0.0", 0.0), ("1.0", 1.0)])
def test_order_divergence_floor_parsed(monkeypatch, val, exp):
    monkeypatch.setenv(SEMANTIK_VLM_ORDER_DIVERGENCE_ENV, val)
    assert resolve_vlm_order_divergence_floor() == pytest.approx(exp)


@pytest.mark.parametrize("val", ["", "abc", "-0.1", "1.5", "nan", "inf"])
def test_order_divergence_floor_falls_back(monkeypatch, val):
    monkeypatch.setenv(SEMANTIK_VLM_ORDER_DIVERGENCE_ENV, val)
    assert resolve_vlm_order_divergence_floor() == pytest.approx(0.25)


def test_apply_vlm_order_reorders_and_drops_garbage():
    # Direct, deterministic unit test of the reorder helper (no aligner): a
    # scrambled fused list (blocks at scrambled bboxes, one garbage-only
    # tesseract line carrying sort_key None) is re-emitted in VLM (gold)
    # order over synthetic monotonic single-column bands, garbage dropped.
    from semantik_structure.vlm_fusion import _apply_vlm_order

    fused = [
        {"text": "B second col2", "bbox": [300.0, 200.0, 560.0, 212.0], "fusion": "vlm+tesseract"},
        {"text": "D fourth col2", "bbox": [300.0, 400.0, 560.0, 412.0], "fusion": "vlm-only-flagged"},
        {"text": "A first col1", "bbox": [10.0, 100.0, 270.0, 112.0], "fusion": "vlm+tesseract"},
        {"text": "garbage ocr tail fragment", "bbox": [10.0, 900.0, 400.0, 912.0], "fusion": "tesseract-only"},
        {"text": "C third col1", "bbox": [10.0, 300.0, 270.0, 312.0], "fusion": "vlm-only-flagged"},
    ]
    # VLM (gold) indices for each block; the garbage line carries None.
    sort_keys = [1, 3, 0, None, 2]
    out = _apply_vlm_order(fused, sort_keys, 612.0, 792.0)
    # Emitted in VLM order; the tesseract-only garbage line is dropped.
    assert [b["text"] for b in out] == [
        "A first col1",
        "B second col2",
        "C third col1",
        "D fourth col2",
    ]
    assert not any("garbage" in b["text"] for b in out)
    # Synthetic bboxes: single column (x0 == 0), strictly monotonic y0 in-range.
    ys = [b["bbox"][1] for b in out]
    assert ys == sorted(ys) and len(set(ys)) == len(ys)
    assert all(b["bbox"][0] == 0.0 and 0.0 <= b["bbox"][1] < b["bbox"][3] <= 792.0 for b in out)
    # End-to-end: the downstream _merge_page (y0, x0) sort — which previously
    # re-scrambled the columns — now reconstructs VLM order.
    merged = extract_shared._merge_page(
        {"tesseract": {"text_blocks": out}}, text_layer_ok=False
    )
    assert [b["text"] for b in merged["text_blocks"]] == [
        "A first col1",
        "B second col2",
        "C third col1",
        "D fourth col2",
    ]


def test_apply_vlm_order_empty_vlm_set_is_noop():
    # No VLM-bearing block (all tesseract-only) → nothing to reorder, returned
    # unchanged (fail-open, never an empty page).
    from semantik_structure.vlm_fusion import _apply_vlm_order

    fused = [{"text": "only ocr line", "bbox": [1.0, 2.0, 3.0, 4.0], "fusion": "tesseract-only"}]
    out = _apply_vlm_order(fused, [None], 100.0, 100.0)
    assert out == fused


def _garbled_multicol_case():
    """One clean anchor line (col1) + two garbled-OCR lines that cannot align
    (col2 top + a bottom fragment) against a CLEAN 4-line VLM transcription.
    The garble → gaps → HIGH divergence; the garbled tesseract lines are the
    'garbage tails' the reorder must drop."""
    tess = [
        _tb(300, 100, 560, 112, "garbage xxxx yyyy zzzz top"),
        _tb(10, 100, 270, 112, "clean anchor line zero here"),
        _tb(10, 200, 270, 212, "garbage qwer asdf bottom fragment"),
    ]
    vlm = [
        "clean anchor line zero here",
        "clean prose line one alpha",
        "clean prose line two bravo",
        "clean prose line three charlie",
    ]
    return tess, vlm


def test_high_divergence_page_fuses_vlm_order_and_drops_garbage(monkeypatch):
    # Integration through fuse_page: a genuinely high-divergence page fires the
    # authoritative reorder — all VLM content survives in order, the garbled
    # tesseract 'tails' are dropped, and bboxes are synthetic monotonic.
    monkeypatch.setenv(SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV, "1")
    tess, vlm = _garbled_multicol_case()
    fused, stats = fuse_page(tess, vlm, (612.0, 792.0))
    assert stats["divergence"] >= 0.25
    assert stats["vlm_authoritative"] is True
    # The garbled OCR tails are gone.
    assert not any("garbage" in b["text"] for b in fused)
    # All VLM ordinal markers survive in VLM linear order across the emission.
    joined = " ".join(b["text"] for b in sorted(fused, key=lambda b: b["bbox"][1]))
    order = [joined.index(m) for m in ("zero", "alpha", "bravo", "charlie")]
    assert order == sorted(order)
    assert all(m in joined for m in ("zero", "alpha", "bravo", "charlie"))
    # Bboxes are strictly monotonic single-column bands.
    ys = [b["bbox"][1] for b in sorted(fused, key=lambda b: b["bbox"][1])]
    assert ys == sorted(ys) and len(set(ys)) == len(ys)


def test_authoritative_off_keeps_legacy_behavior_on_high_divergence(monkeypatch):
    # Explicit revert (default-off): the same high-divergence page keeps the
    # legacy fused list — garbled tesseract tails retained, no reorder.
    monkeypatch.setenv(SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV, "0")
    tess, vlm = _garbled_multicol_case()
    fused, stats = fuse_page(tess, vlm, (612.0, 792.0))
    assert stats["divergence"] >= 0.25
    assert stats["vlm_authoritative"] is False
    assert any("garbage" in b["text"] for b in fused)  # tails retained


def test_low_divergence_page_byte_identical_regardless_of_authoritative(monkeypatch):
    # A clean single-column page (all MATCH → divergence 0) is byte-identical
    # whether the authoritative reorder is enabled or reverted — the divergence
    # floor gates it out, so real Tesseract bboxes are preserved.
    tess = [
        _tb(10, 10, 190, 20, "the quick brown fox jumps"),
        _tb(10, 30, 190, 40, "over the lazy dog today"),
        _tb(10, 50, 190, 60, "pack my box with five jugs"),
    ]
    vlm = [
        "the quick brown fox jumps",
        "over the lazy dog today",
        "pack my box with five jugs",
    ]
    monkeypatch.setenv(SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV, "1")
    on, on_stats = fuse_page(copy.deepcopy(tess), list(vlm), (200.0, 300.0))
    monkeypatch.setenv(SEMANTIK_VLM_ORDER_AUTHORITATIVE_ENV, "0")
    off, off_stats = fuse_page(copy.deepcopy(tess), list(vlm), (200.0, 300.0))
    assert on == off  # blocks byte-identical on a clean low-divergence page
    assert on_stats["vlm_authoritative"] is False
    assert on_stats["divergence"] == pytest.approx(0.0)
    # Real Tesseract bboxes preserved (no synthetic single-column rewrite).
    assert on[0]["bbox"] == [10.0, 10.0, 190.0, 20.0]


# --------------------------------------------------------------------------
# OCR-garbage tail drop (SEMANTIK_VLM_DROP_GARBAGE_TAILS, default ON in-fusion)
# --------------------------------------------------------------------------


from semantik_structure.vlm_fusion import (  # noqa: E402
    SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV,
    _looks_like_ocr_garbage,
    resolve_drop_garbage_tails_mode,
)


# Representative evidence lines: unrescued Tesseract-only OCR garbage that
# would otherwise ship verbatim.
_GARBAGE_EVIDENCE = [
    "TRY Tiss ® ©",
    "©) obi Dom -19",
    "© Solution No.",
    "2,791 © 2,795",
]


@pytest.mark.parametrize("line", _GARBAGE_EVIDENCE)
def test_looks_like_ocr_garbage_catches_evidence_lines(line):
    assert _looks_like_ocr_garbage(line) is True


@pytest.mark.parametrize(
    "line",
    [
        "The quick brown fox jumps over the lazy dog.",  # normal sentence
        "See the theorem.",  # short normal sentence, no junk glyph
        r"$|n| \geq 0$ for all numbers",  # real math line (pipes are in a math span)
        "Solution",  # legit short label
        "37. 84 38. 9,696 39. 75",  # clean numeric answer row (no junk glyph)
        "(a) first option",  # leading paren is NOT a junk glyph
    ],
)
def test_looks_like_ocr_garbage_keeps_real_content(line):
    assert _looks_like_ocr_garbage(line) is False


def test_drop_garbage_tails_resolver_default_on(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV, raising=False)
    assert resolve_drop_garbage_tails_mode() is True


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "garbage", "2x", ""])
def test_drop_garbage_tails_resolver_on_unless_explicit_falsey(monkeypatch, val):
    # Default-ON parse posture (mirrors resolve_strip_furniture_mode): unset /
    # truthy / garbage / blank keep it on.
    monkeypatch.setenv(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV, val)
    assert resolve_drop_garbage_tails_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "OFF", " Off "])
def test_drop_garbage_tails_resolver_explicit_falsey_off(monkeypatch, val):
    monkeypatch.setenv(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV, val)
    assert resolve_drop_garbage_tails_mode() is False


def _garbage_tail_case():
    """A clean MATCH anchor + one unrescued tesseract-only OCR-garbage line."""
    tess = [
        _tb(10, 10, 100, 20, "clean anchor line here"),
        _tb(10, 90, 100, 100, "© Solution No.", conf=0.4),
    ]
    vlm = ["clean anchor line here"]
    return tess, vlm


def test_garbage_tail_dropped_when_mode_on(monkeypatch):
    monkeypatch.delenv(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV, raising=False)  # default on
    tess, vlm = _garbage_tail_case()
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    # The garbage tesseract-only block is gone; the drop is tallied but the RAW
    # ledger gap count (tesseract_only) is unchanged (divergence invariant).
    assert not any("Solution No" in b["text"] for b in fused)
    assert stats["garbage_tail_dropped"] == 1
    assert stats["tesseract_only"] == 1
    # The aligned anchor still ships.
    assert any(b["fusion"] == "vlm+tesseract" for b in fused)


def test_garbage_tail_kept_when_mode_off(monkeypatch):
    monkeypatch.setenv(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV, "0")  # escape hatch
    tess, vlm = _garbage_tail_case()
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    only = [b for b in fused if b["fusion"] == "tesseract-only"]
    assert len(only) == 1
    assert only[0]["text"] == "© Solution No."  # byte-verbatim retained
    assert stats["garbage_tail_dropped"] == 0
    assert stats["tesseract_only"] == 1


def test_non_garbage_tesseract_only_kept_even_when_mode_on(monkeypatch):
    # A junk-glyph-free tesseract-only line (real content the VLM dropped) is
    # NEVER touched by the conservative predicate, even with the drop on.
    monkeypatch.delenv(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV, raising=False)
    tess = [
        _tb(10, 10, 100, 20, "clean anchor line here"),
        _tb(10, 90, 100, 100, "ocr only unique zzz", conf=0.71),
    ]
    vlm = ["clean anchor line here"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    only = [b for b in fused if b["fusion"] == "tesseract-only"]
    assert len(only) == 1
    assert only[0]["text"] == "ocr only unique zzz"
    assert stats["garbage_tail_dropped"] == 0


def test_extract_cache_salt_gated_on_garbage_tail_mode():
    # The default-ON garbage drop keys an append-only |gtail salt when fusion
    # AND the drop are both on; the =0 escape hatch keys back to the plain
    # fuse_key (warm cache stays valid), and the no-fusion key omits it. Bumped
    # |gtail1 -> |gtail2 by the ®/© re-OCR duplicate-tail recognizer (drops more
    # blocks, so a gtail1 cache is invalid for a default drop-on run).
    import inspect

    src = inspect.getsource(extract_shared._compute_extract_cache_key)
    assert '"|gtail2"' in src
    assert '"|gtail1"' not in src
    assert "_resolve_drop_garbage_tails_mode" in src


def test_extract_cache_salt_bumped_to_vlmfuse6():
    # The Nemotron-Omni special-token strip changes what fuse_page emits
    # UNCONDITIONALLY when fusion is on (independent of the garbage-tail mode),
    # so a vlmfuse5 cache is invalid — FRESH conversions must key on vlmfuse6.
    import inspect

    src = inspect.getsource(extract_shared._compute_extract_cache_key)
    assert '"|vlmfuse6"' in src
    assert '"|vlmfuse5"' not in src


# --------------------------------------------------------------------------
# FIX 1 — Nemotron-Omni grounding/detection special-token strip (ch03 p.101).
# --------------------------------------------------------------------------


from semantik_structure.vlm_fusion import _strip_vlm_special_tokens  # noqa: E402


def test_strip_vlm_special_tokens_ref_kept_det_dropped():
    # The live-fire ch03 leak: keep the ref inner text, drop the det bbox pair.
    assert (
        _strip_vlm_special_tokens(
            "<|ref|>Question<|/ref|><|det|>[[155, 100, 760, 120]]<|/det|>"
        )
        == "Question"
    )


def test_strip_vlm_special_tokens_ref_inner_text_preserved_in_context():
    # A ref pair embedded in surrounding prose keeps its inner text in place.
    assert (
        _strip_vlm_special_tokens("See <|ref|>Example 3<|/ref|> below")
        == "See Example 3 below"
    )


def test_strip_vlm_special_tokens_stray_unpaired_tokens_removed():
    # Stray unpaired ref/det-family tokens + a bare det bbox payload are scrubbed.
    assert _strip_vlm_special_tokens("Question <|/ref|> tail") == "Question tail"
    assert _strip_vlm_special_tokens("prefix <|det|> [[10, 20, 30, 40]]") == "prefix"
    assert _strip_vlm_special_tokens("<|ref|>Answer") == "Answer"


def test_strip_vlm_special_tokens_token_free_line_byte_identical():
    # No ``<|`` sigil → returned byte-identical (LaTeX, entities, brackets safe).
    for line in (
        r"$-|b|$ when $b = -12$",
        "See section 3(a) for details",
        "A normal [[bracketed]] phrase",
        "",
    ):
        assert _strip_vlm_special_tokens(line) == line


def test_fusion_strips_special_tokens_from_vlm_text():
    # End-to-end through fuse_page: a leaked-token VLM line fuses onto the
    # tesseract bbox with the grounding tokens stripped (ref inner kept).
    tess = [_tb(10, 10, 100, 20, "Question", conf=0.6)]
    vlm = ["<|ref|>Question<|/ref|><|det|>[[155, 100, 760, 120]]<|/det|>"]
    fused, stats = fuse_page(tess, vlm, (200.0, 300.0))
    assert fused[0]["text"] == "Question"
    assert "<|ref|>" not in fused[0]["text"] and "<|det|>" not in fused[0]["text"]
    assert "[[155" not in fused[0]["text"]


def test_vlm_only_insert_strips_special_tokens():
    # The strip applies regardless of alignment: a VLM-only insert is cleaned too.
    tess = [
        _tb(10, 10, 100, 20, "alpha beta gamma"),
        _tb(10, 50, 100, 60, "delta epsilon zeta"),
    ]
    vlm = [
        "alpha beta gamma",
        "<|ref|>inserted only line<|/ref|><|det|>[[1, 2, 3, 4]]<|/det|>",
        "delta epsilon zeta",
    ]
    fused, _ = fuse_page(tess, vlm, (200.0, 300.0))
    insert = [b for b in fused if b["fusion"] == "vlm-only-flagged"]
    assert len(insert) == 1
    assert insert[0]["text"] == "inserted only line"


# --------------------------------------------------------------------------
# FIX 2 — ®/© re-OCR duplicate-tail recognizer.
# --------------------------------------------------------------------------


from semantik_structure.vlm_fusion import _looks_like_reocr_marker_garbage  # noqa: E402


# Regression evidence: a clean-math head fused with a Tesseract re-OCR
# tail enumerated by ®/© mis-reads + junk-glyph pipe-runs.
_REOCR_TAIL_EVIDENCE = (
    "b $-|b|$ when $b = -12$ ® -Iq| when g = -33 © —|b| when b = —12 Add Integers"
)


def test_reocr_marker_garbage_catches_evidence():
    assert _looks_like_reocr_marker_garbage(_REOCR_TAIL_EVIDENCE) is True
    # Reaches the whole predicate too (bypassing the length + math-span guards).
    assert _looks_like_ocr_garbage(_REOCR_TAIL_EVIDENCE) is True


@pytest.mark.parametrize(
    "line",
    [
        # clean LaTeX math ALONE must survive — the ®/© + pipe-junk is the
        # signal, not the LaTeX.
        r"$-|b|$ when $b = -12$",
        r"$-|b|$",
        r"$b = -12$",
        r"$|n| \geq 0$ for all numbers",
        # a table row with pipes but no circled markers is not re-OCR garble
        "value | result | note",
        # circled markers but no pipe-junk (a plain copyright/reg line)
        "© 2026 Example ® brand",
    ],
)
def test_reocr_marker_garbage_keeps_clean_math_and_non_garble(line):
    assert _looks_like_reocr_marker_garbage(line) is False


def test_reocr_tail_dropped_through_fusion(monkeypatch):
    # A tesseract-only block that is a ®/© re-OCR duplicate tail is dropped by
    # the (default-ON) garbage-tail pass; the clean VLM anchor still ships.
    monkeypatch.delenv(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV, raising=False)
    tess = [
        _tb(10, 10, 100, 20, "Simplify each expression"),
        _tb(10, 90, 380, 100, _REOCR_TAIL_EVIDENCE, conf=0.4),
    ]
    vlm = ["Simplify each expression"]
    fused, stats = fuse_page(tess, vlm, (400.0, 300.0))
    assert not any("Add Integers" in b["text"] for b in fused)
    assert not any("-Iq|" in b["text"] for b in fused)
    assert stats["garbage_tail_dropped"] == 1
    assert any(b["fusion"] == "vlm+tesseract" for b in fused)


def test_reocr_tail_kept_when_drop_mode_off(monkeypatch):
    # The escape hatch keeps the re-OCR tail byte-verbatim (no whole-block nuke
    # of real content when an operator opts out).
    monkeypatch.setenv(SEMANTIK_VLM_DROP_GARBAGE_TAILS_ENV, "0")
    tess = [
        _tb(10, 10, 100, 20, "Simplify each expression"),
        _tb(10, 90, 380, 100, _REOCR_TAIL_EVIDENCE, conf=0.4),
    ]
    vlm = ["Simplify each expression"]
    fused, stats = fuse_page(tess, vlm, (400.0, 300.0))
    assert any("Add Integers" in b["text"] for b in fused)
    assert stats["garbage_tail_dropped"] == 0
