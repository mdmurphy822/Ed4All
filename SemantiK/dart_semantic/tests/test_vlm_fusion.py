"""Regression: SEMANTIK_VLM_FUSION deterministically fuses the P0 VLM markdown
source onto the tesseract line blocks (P1 lane).

The VLM endpoint is NEVER called here — P1 consumes the pinned ``page['vlm']``
contract (P0 = SEMANTIK_VLM_EXTRACT), so every test supplies synthetic VLM
lines / a synthetic ``page['vlm']`` dict. No pypdfium2 render, no real OCR, no
model load. CPU-only.

NB: these tests live under ``dart_semantic/tests/`` (the ACTUAL home of the
``test_ocr_render_scale.py`` / ``test_tesseract_config.py`` precedent the lane
map referenced — the map's "dart_semantic/ root" was a mislocation).
"""

from __future__ import annotations

import copy

import pytest

from dart_semantic import extract_shared
from dart_semantic.vlm_fusion import (
    SEMANTIK_VLM_FUSION_ENV,
    fuse_page,
    resolve_vlm_fusion_mode,
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


def test_one_tesseract_to_k_vlm_split():
    # One OCR line unfolded into two VLM lines (a fused title + body). SPLIT →
    # the VLM head hint covers only a PREFIX of the block.
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
    from dart_semantic.vlm_fusion import _sanitize_bbox

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
    from dart_semantic.vlm_fusion import _sanitize_bbox

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
    from dart_semantic.extract import blocks_from_shared

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
    from dart_semantic.extract import blocks_from_shared

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


# --------------------------------------------------------------------------
# Defect 2 — markdown code-fence strip from the fused block text.
# --------------------------------------------------------------------------


def test_strip_markdown_structure_drops_whole_line_fence():
    from dart_semantic.vlm_fusion import _strip_markdown_structure

    assert _strip_markdown_structure("```markdown") == ""
    assert _strip_markdown_structure("```") == ""
    assert _strip_markdown_structure("```math") == ""
    assert _strip_markdown_structure("  ```  ") == ""


def test_strip_markdown_structure_scrubs_stray_inline_fence():
    from dart_semantic.vlm_fusion import _strip_markdown_structure

    # Whole-page wrapper collapsed onto one line with the heading.
    assert (
        _strip_markdown_structure("```markdown Chapter 9 Roots and Radicals")
        == "Chapter 9 Roots and Radicals"
    )


def test_strip_markdown_structure_keeps_inline_latex():
    from dart_semantic.vlm_fusion import _strip_markdown_structure

    assert _strip_markdown_structure(r"$\sqrt{ab}$") == r"$\sqrt{ab}$"


def test_join_vlm_drops_fence_lines():
    from dart_semantic.vlm_fusion import _join_vlm

    lines = ["```markdown", "Chapter 9 Roots and Radicals", "```"]
    assert _join_vlm(lines, [0, 1, 2]) == "Chapter 9 Roots and Radicals"


# --------------------------------------------------------------------------
# Defect 6 — © -> (c) confusable fold (SCORING ONLY).
# --------------------------------------------------------------------------


def test_fold_confusables_copyright_to_paren_c():
    from dart_semantic.vlm_fusion import _fold_confusables, _score_norm

    assert "(c)" in _fold_confusables("©")
    # A VLM "©" option marker scores identically to the OCR "(c)".
    assert _score_norm("©") == _score_norm("(c)")


def test_fold_confusables_scoring_only_output_untouched():
    # The output-text transform (_strip_markdown_structure) must NOT rewrite ©.
    from dart_semantic.vlm_fusion import _strip_markdown_structure

    assert _strip_markdown_structure("© 2026 Example") == "© 2026 Example"


# --------------------------------------------------------------------------
# Defect B (coordinator follow-up) — LaTeX sectioning-wrapper normalization.
# --------------------------------------------------------------------------


def test_strip_markdown_structure_unwraps_latex_section():
    from dart_semantic.vlm_fusion import _strip_markdown_structure

    assert (
        _strip_markdown_structure(r"\section*{REVIEW EXERCISES}")
        == "REVIEW EXERCISES"
    )
    assert _strip_markdown_structure(r"\subsection*{Key Terms}") == "Key Terms"
    assert _strip_markdown_structure(r"\section{PRACTICE TEST}") == "PRACTICE TEST"


def test_latex_section_inline_mention_and_math_untouched():
    from dart_semantic.vlm_fusion import _strip_markdown_structure

    # An inline \section mention inside prose is never rewritten.
    assert (
        _strip_markdown_structure(r"see \section for details")
        == r"see \section for details"
    )
    # Inline $…$ math LaTeX is preserved verbatim.
    assert _strip_markdown_structure(r"$\sqrt{x}$ and $y^2$") == r"$\sqrt{x}$ and $y^2$"


# --------------------------------------------------------------------------
# ch02 defect — literal HTML-entity text scrubbed from VLM lines before fusion.
# --------------------------------------------------------------------------


def test_strip_markdown_structure_scrubs_nbsp_entity_runs():
    from dart_semantic.vlm_fusion import _strip_markdown_structure

    # the audited ch02 shape: blank table spacing transcribed as entity runs
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
    from dart_semantic.vlm_fusion import _strip_markdown_structure

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


def test_extract_cache_salt_bumped_to_vlmfuse5():
    # The entity scrub changes what fuse_page emits: a vlmfuse4 cache may carry
    # literal entity text (corrected downstream by the adapter scrub — current
    # corpus stays on vlmfuse4 deliberately), so FRESH conversions must key on
    # vlmfuse5.
    import inspect

    src = inspect.getsource(extract_shared._compute_extract_cache_key)
    assert '"|vlmfuse5"' in src
    assert '"|vlmfuse4"' not in src
