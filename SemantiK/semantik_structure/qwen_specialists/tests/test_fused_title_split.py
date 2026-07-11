"""Stage-5e fused-section-title SPLIT arm (SEMANTIK_SPLIT_FUSED_SECTION_TITLES).

Lane B: a scan fuses a section title into its first body paragraph so no
standalone heading survives. This suite drives the deterministic detector, the
apply/breadcrumb/audit path, the per-op fail-closed belt, the orthogonality +
overlap precedence with the other Stage-5e arms, and the post-Stage-5e title
promotion hook — all on SYNTHETIC generic fixtures (no course-data text).

The cascade module is import-heavy (axe/playwright); these tests exercise the
Stage-5e helpers directly and never import the cascade.
"""

from __future__ import annotations

from semantik_structure.qwen_specialists import block_resegment as br
from semantik_structure.qwen_specialists import deterministic_structure as ds
from semantik_structure.qwen_specialists.block_resegment import (
    ResegmentOp,
    build_resegment_audit_rows,
    resegment_blocks,
    resolve_split_fused_section_titles_mode,
)
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Synthetic fixtures.
# ---------------------------------------------------------------------------


def _fb(text: str, page: int = 1) -> FeatureBlock:
    raw = RawBlock(
        text=text,
        page=page,
        bbox=(0.0, 0.0, 10.0, 10.0),
        page_width=100.0,
        page_height=100.0,
    )
    return FeatureBlock(
        raw=raw,
        size_bucket="md",
        gap_above=None,
        is_top_of_page=False,
        is_centered=False,
        caps=None,
        indent_bucket=0,
        relative_font_ratio=1.0,
    )


def _reg(kind: str, fb_indices, fbs, *, text=None, css=None, source_region_id=None):
    """Build a Region whose payload text defaults to the joined FB text (so the
    dominant-chapter-ordinal scan sees a realistic region)."""
    if text is None:
        text = " ".join(_fbtext(fbs, i) for i in fb_indices if _fbtext(fbs, i))
    payload = {"text": text}
    if css is not None:
        payload["css_class"] = css
    return Region(
        kind=kind,
        feature_block_indices=tuple(fb_indices),
        payload=payload,
        source_region_id=source_region_id,
    )


def _fbtext(fbs, idx: int) -> str:
    try:
        return (fbs[idx].raw.text or "").strip()
    except (IndexError, AttributeError):
        return ""


class _EmptyState:
    outputs: dict = {}


_PROSE = "A system of two linear equations can be solved together by substitution."


def _happy_regions():
    """One paragraph region: FB0 = a clean 4.2 title, FB1 = a prose sentence.
    Its own payload text (4.2 ...) establishes the dominant ordinal 4."""
    fbs = [_fb("4.2 Solve Linear Systems"), _fb(_PROSE)]
    regions = [_reg("paragraph", [0, 1], fbs)]
    return regions, fbs


# ---------------------------------------------------------------------------
# Resolver — parse-with-fallback.
# ---------------------------------------------------------------------------


def test_resolver_default_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", raising=False)
    assert resolve_split_fused_section_titles_mode() is False


def test_resolver_blank_garbage_falsey_off(monkeypatch):
    for val in ("", "   ", "banana", "0", "false", "no", "off"):
        monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", val)
        assert resolve_split_fused_section_titles_mode() is False


def test_resolver_truthy_on(monkeypatch):
    for val in ("1", "true", "TRUE", "Yes", "on", "ON"):
        monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", val)
        assert resolve_split_fused_section_titles_mode() is True


# ---------------------------------------------------------------------------
# Flag off — byte-identical (no fused-title op, no breadcrumb).
# ---------------------------------------------------------------------------


def test_flag_off_byte_identical(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", raising=False)
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: pin off (default is ON)
    regions, fbs = _happy_regions()
    out, ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    assert ops == []
    assert out == regions  # same partition, no re-tag
    assert (out[0].payload or {}).get("resegment") is None


# ---------------------------------------------------------------------------
# Happy path — N.M title split.
# ---------------------------------------------------------------------------


def test_happy_path_nm_split(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: pin off (default is ON)
    regions, fbs = _happy_regions()
    out, ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)

    assert len(ops) == 1
    assert ops[0].op == "split"
    assert ops[0].subtype == "fused_title"
    assert ops[0].split_at == (1,)

    assert len(out) == 2
    title, prose = out
    assert title.feature_block_indices == (0,)
    assert prose.feature_block_indices == (1,)
    # child0 keeps the parent min-FB id; child1's id == its slice min.
    assert min(title.feature_block_indices) == 0
    assert min(prose.feature_block_indices) == 1
    assert (title.payload or {}).get("text") == "4.2 Solve Linear Systems"
    assert (prose.payload or {}).get("text") == _PROSE
    assert (title.payload or {})["resegment"]["subtype"] == "fused_title"
    assert (title.payload or {})["resegment"]["child_index"] == 0


def test_multi_fb_title_prefix_k2(monkeypatch):
    """Title wrapped over 2 leading FBs -> split after the 2nd FB."""
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    # The section NUMBER is OCR'd as its own FB, so no single-FB prefix is a
    # whole title until k=2 ("4.2" + "Solve Linear Systems").
    fbs = [_fb("4.2"), _fb("Solve Linear Systems"), _fb(_PROSE)]
    regions = [
        _reg("paragraph", [0, 1, 2], fbs, text="4.2 Solve Linear Systems " + _PROSE)
    ]
    ops, intra = br._detect_fused_title_splits(regions, fbs)
    assert intra == 0
    assert len(ops) == 1
    assert ops[0].split_at == (2,)


# ---------------------------------------------------------------------------
# Chapter-consistency + shape rejections (prefix-kind unit level).
# ---------------------------------------------------------------------------


def test_prefix_kind_wrong_ordinal_no_op():
    assert br._fused_title_prefix_kind("5.2 Solve Linear Systems", 4) is None


def test_prefix_kind_none_dominant_disables_section():
    assert br._fused_title_prefix_kind("4.2 Solve Linear Systems", None) is None


def test_prefix_kind_m_out_of_range_no_op():
    assert br._fused_title_prefix_kind("4.0 Solve Systems", 4) is None
    assert br._fused_title_prefix_kind("4.41 Solve Systems", 4) is None


def test_prefix_kind_title_too_long_no_op():
    long_title = "4.2 " + " ".join(f"Word{i}" for i in range(11))  # 11 title words
    assert br._fused_title_prefix_kind(long_title, 4) is None


def test_prefix_kind_lowercase_start_no_op():
    assert br._fused_title_prefix_kind("4.2 solve linear systems", 4) is None


def test_detector_ambiguous_dominant_tie_no_section_op(monkeypatch):
    """A tie in the dominant ordinal disables the N.M arm (no section split)."""
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("4.2 Solve Systems"), _fb(_PROSE), _fb("5.1 Graph Basics"), _fb(_PROSE)]
    regions = [
        _reg("paragraph", [0, 1], fbs, text="4.2 Solve Systems " + _PROSE),
        _reg("paragraph", [2, 3], fbs, text="5.1 Graph Basics " + _PROSE),
    ]
    ops, _ = br._detect_fused_title_splits(regions, fbs)
    assert ops == []  # counts {4:1, 5:1} -> tie -> dominant None -> N.M disabled


# ---------------------------------------------------------------------------
# Prose-remainder rejections.
# ---------------------------------------------------------------------------


def test_remainder_too_short_no_op(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("4.2 Solve Linear Systems"), _fb("Short tail here.")]  # < 8 words
    regions = [_reg("paragraph", [0, 1], fbs, text="4.2 Solve Linear Systems Short tail here.")]
    ops, _ = br._detect_fused_title_splits(regions, fbs)
    assert ops == []


def test_remainder_no_sentence_terminal_no_op(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    no_term = "a clause of at least eight words but lacking any terminal mark"
    fbs = [_fb("4.2 Solve Linear Systems"), _fb(no_term)]
    regions = [_reg("paragraph", [0, 1], fbs, text="4.2 Solve Linear Systems " + no_term)]
    ops, _ = br._detect_fused_title_splits(regions, fbs)
    assert ops == []


def test_single_fb_region_no_split(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("4.2 Solve Linear Systems")]
    regions = [_reg("paragraph", [0], fbs)]
    ops, intra = br._detect_fused_title_splits(regions, fbs)
    assert ops == []
    assert intra == 0  # a bare short title FB is not intra-FB fusion


# ---------------------------------------------------------------------------
# Intra-FB fusion — detect-only diagnostic (unsplittable at an FB boundary).
# ---------------------------------------------------------------------------


def test_intra_fb_fusion_counted_no_op(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fused = "4.2 Solve Linear Systems " + _PROSE  # title + prose in ONE FB
    fbs = [_fb(fused), _fb(_PROSE)]
    regions = [_reg("paragraph", [0, 1], fbs, text=fused + " " + _PROSE)]
    ops, intra = br._detect_fused_title_splits(regions, fbs)
    assert ops == []  # no FB boundary separates title from prose
    assert intra == 1


def test_intra_fb_apparatus_single_fb_counted(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fused = "[y| Introduction Suppose a stone falls from a great height slowly."
    fbs = [_fb(fused)]
    regions = [_reg("paragraph", [0], fbs, text=fused)]
    ops, intra = br._detect_fused_title_splits(regions, fbs)
    assert ops == []
    assert intra == 1


# ---------------------------------------------------------------------------
# Apparatus opener splits.
# ---------------------------------------------------------------------------


def test_apparatus_gutter_prefixed_split(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("[y| Introduction"), _fb("Suppose a stone falls from a great height slowly.")]
    regions = [_reg("paragraph", [0, 1], fbs)]
    ops, intra = br._detect_fused_title_splits(regions, fbs)
    assert len(ops) == 1
    assert ops[0].subtype == "fused_title"
    assert ops[0].split_at == (1,)


def test_apparatus_bare_opener_split(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("Introduction"), _fb("Suppose a stone falls from a great height very slowly.")]
    regions = [_reg("paragraph", [0, 1], fbs)]
    ops, _ = br._detect_fused_title_splits(regions, fbs)
    assert len(ops) == 1
    assert ops[0].split_at == (1,)


def test_apparatus_opener_as_prose_prefix_no_op(monkeypatch):
    """'Introduction to X' as the START of a longer sentence is NOT a whole-match
    apparatus opener -> no op."""
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [
        _fb("Introduction to Polynomials"),
        _fb("Suppose a stone falls from a great height very slowly."),
    ]
    regions = [_reg("paragraph", [0, 1], fbs)]
    ops, _ = br._detect_fused_title_splits(regions, fbs)
    assert ops == []


# ---------------------------------------------------------------------------
# Skips — passthrough + heading regions.
# ---------------------------------------------------------------------------


def test_passthrough_region_never_split(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("4.2 Solve Linear Systems"), _fb(_PROSE)]
    regions = [_reg("table", [0, 1], fbs, source_region_id=7)]
    ops, intra = br._detect_fused_title_splits(regions, fbs)
    assert ops == []
    assert intra == 0


def test_heading_region_never_split(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("4.2 Solve Linear Systems"), _fb(_PROSE)]
    regions = [_reg("heading", [0, 1], fbs)]
    ops, intra = br._detect_fused_title_splits(regions, fbs)
    assert ops == []
    assert intra == 0


# ---------------------------------------------------------------------------
# Per-op fail-closed — a malformed op is dropped individually.
# ---------------------------------------------------------------------------


def test_per_op_fail_closed_drops_malformed(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("4.2 Solve"), _fb(_PROSE), _fb("Another paragraph body of eight or more words here.")]
    regions = [
        _reg("paragraph", [0, 1], fbs, text="4.2 Solve " + _PROSE),
        _reg("paragraph", [2], fbs),
    ]
    good = ResegmentOp(
        op="split", region_indices=(0,), split_at=(1,),
        source_ids=(0,), origin="deterministic", subtype="fused_title",
    )
    # NON-INTERIOR cut: region 1's first FB (2) is never a valid boundary.
    bad = ResegmentOp(
        op="split", region_indices=(1,), split_at=(2,),
        source_ids=(2,), origin="deterministic", subtype="fused_title",
    )
    monkeypatch.setattr(br, "_detect_fused_title_splits", lambda r, f: ([good, bad], 0))
    out, ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    # Only the good op applied: region 0 split into 2, region 1 untouched.
    assert len(ops) == 1
    assert len(out) == 3
    assert out[2].feature_block_indices == (2,)


def test_fused_title_op_valid_predicate():
    fbs = [_fb("a"), _fb("b"), _fb("c")]
    regions = [_reg("paragraph", [0, 1, 2], fbs, text="a b c")]
    ok = ResegmentOp(op="split", region_indices=(0,), split_at=(1,), subtype="fused_title")
    non_interior = ResegmentOp(op="split", region_indices=(0,), split_at=(0,), subtype="fused_title")
    out_of_range = ResegmentOp(op="split", region_indices=(0,), split_at=(9,), subtype="fused_title")
    bad_region = ResegmentOp(op="split", region_indices=(5,), split_at=(1,), subtype="fused_title")
    empty_cut = ResegmentOp(op="split", region_indices=(0,), split_at=(), subtype="fused_title")
    assert br._fused_title_op_valid(ok, regions) is True
    assert br._fused_title_op_valid(non_interior, regions) is False
    assert br._fused_title_op_valid(out_of_range, regions) is False
    assert br._fused_title_op_valid(bad_region, regions) is False
    assert br._fused_title_op_valid(empty_cut, regions) is False


# ---------------------------------------------------------------------------
# Orthogonality with the other Stage-5e arms.
# ---------------------------------------------------------------------------


def test_orthogonality_fused_only(monkeypatch):
    """Fused-title on + block-resegment off + regroup off -> ONLY fused ops."""
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    monkeypatch.delenv("SEMANTIK_BLOCK_RESEGMENT", raising=False)
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: pin off (default is ON)
    regions, fbs = _happy_regions()
    _out, ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    assert ops and all(o.subtype == "fused_title" for o in ops)


def test_orthogonality_block_resegment_on_no_fused(monkeypatch):
    """Block-resegment on + fused-title off -> zero fused-title ops."""
    monkeypatch.delenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", raising=False)
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: pin off (default is ON)
    regions, fbs = _happy_regions()
    _out, ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    assert all(o.subtype != "fused_title" for o in ops)


# ---------------------------------------------------------------------------
# Overlap precedence — fused-title beats a colliding same-kind split.
# ---------------------------------------------------------------------------


def test_overlap_fused_title_wins_over_same_kind(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: pin off (default is ON)
    fbs = [
        _fb("4.2 Solve Systems"),
        _fb(_PROSE),
        _fb("Solution factor it out carefully"),  # interior pedagogical label
    ]
    text = "4.2 Solve Systems " + _PROSE + " Solution factor it out carefully"
    regions = [_reg("paragraph", [0, 1, 2], fbs, text=text)]
    out, ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    # Fused-title split at FB1 wins; the same-kind Solution split at FB2 is
    # dropped whole (first-registered wins), so FB1+FB2 stay one child.
    assert len(out) == 2
    assert out[0].feature_block_indices == (0,)
    assert out[1].feature_block_indices == (1, 2)
    applied_subtypes = {o.subtype for o in ops if o.op == "split" and o.region_indices == (0,)}
    assert "fused_title" in applied_subtypes


# ---------------------------------------------------------------------------
# Breadcrumb discrimination + audit rows.
# ---------------------------------------------------------------------------


def test_same_kind_split_breadcrumb_has_no_subtype(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", raising=False)
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")
    monkeypatch.setenv("SEMANTIK_UNIT_REGROUP", "0")  # ITEM1: pin off (default is ON)
    fbs = [_fb("Body prose sentence one here."), _fb("Solution factor the expression")]
    regions = [_reg("paragraph", [0, 1], fbs, text="Body prose sentence one here. Solution factor the expression")]
    out, ops = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    assert len(out) == 2  # split at the Solution pedagogical label
    for child in out:
        bc = (child.payload or {}).get("resegment")
        assert bc is not None
        assert "subtype" not in bc  # byte-identical to pre-lane-B


def test_audit_rows_fused_title_and_same_kind():
    fused = ResegmentOp(op="split", region_indices=(0,), split_at=(1,), source_ids=(0,), subtype="fused_title")
    same = ResegmentOp(op="split", region_indices=(3,), split_at=(4,), source_ids=(3,))
    rows = build_resegment_audit_rows([fused, same])
    assert rows[0] == {
        "op": "split", "source_ids": [0], "origin": "deterministic",
        "conservation_verified": True, "subtype": "fused_title",
    }
    assert rows[1] == {
        "op": "split", "source_ids": [3], "origin": "deterministic",
        "conservation_verified": True,
    }


# ---------------------------------------------------------------------------
# Promotion hook — post-Stage-5e title child -> heading.
# ---------------------------------------------------------------------------


def _split_output(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    regions, fbs = _happy_regions()
    out, _ = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    return out, fbs


def test_promotion_both_flags_promotes_title_child(monkeypatch):
    out, fbs = _split_output(monkeypatch)
    promoted, diag = ds.promote_fused_title_children(out, fbs)
    assert diag["promoted"] == 1
    title, prose = promoted
    assert title.kind == "heading"
    assert (title.payload or {})["level_hint"] == 2
    assert (title.payload or {})["structure_clean"]["promoted"] == "section_heading"
    assert prose.kind == "paragraph"


def test_promotion_apparatus_child(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", "on")
    fbs = [_fb("[y| Introduction"), _fb("Suppose a stone falls from a great height slowly.")]
    regions = [_reg("paragraph", [0, 1], fbs)]
    out, _ = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    promoted, diag = ds.promote_fused_title_children(out, fbs)
    assert diag["promoted"] == 1
    assert promoted[0].kind == "heading"


def test_promotion_noop_when_no_fused_child(monkeypatch):
    """A plain same-kind split output carries no fused_title child -> no promotion."""
    fbs = [_fb("Body prose sentence one here."), _fb("Solution factor the expression")]
    regions = [_reg("paragraph", [0, 1], fbs, text="Body prose sentence one here. Solution factor the expression")]
    monkeypatch.setenv("SEMANTIK_BLOCK_RESEGMENT", "on")
    monkeypatch.delenv("SEMANTIK_SPLIT_FUSED_SECTION_TITLES", raising=False)
    out, _ = resegment_blocks(regions, fbs, _EmptyState(), runtime=None)
    promoted, diag = ds.promote_fused_title_children(out, fbs)
    assert diag["promoted"] == 0
    assert promoted == out


def test_promotion_fail_closed_on_token_conservation(monkeypatch):
    out, fbs = _split_output(monkeypatch)

    def _boom(*a, **k):
        raise ds.TokenConservationError("forced")

    monkeypatch.setattr(ds, "assert_token_conservation", _boom)
    promoted, diag = ds.promote_fused_title_children(out, fbs)
    assert diag.get("reverted_for_invariant") is True
    assert promoted == out  # whole-revert to input
    assert all(r.kind != "heading" for r in promoted)  # nothing promoted


def test_promotion_empty_regions():
    promoted, diag = ds.promote_fused_title_children([], [])
    assert promoted == []
    assert diag["promoted"] == 0
