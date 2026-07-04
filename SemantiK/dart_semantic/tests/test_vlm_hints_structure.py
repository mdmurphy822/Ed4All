"""P2 — VLM structural hints (NON-AUTHORITATIVE side-channel) unit tests.

Covers the plan (``plans/scan-conversion-improvements-2026-07.md`` Phase 4 P2)
hint CHANNEL: flag + provider-agnostic seat resolution, hint MINTING from VLM
markdown shape, CARRIAGE through RawBlock -> FeatureBlock (surviving the
image-FB interleave), heading-candidate CORROBORATION in structure_graph
Pass-2 (positive + no-bypass negative + level tiebreak), the extract-cache
salt, and the byte-identical-off contract.

The VLM endpoint is NEVER called live here — the seat is exercised as pure
config (the P0 HTTP dispatch calls ``VLMSeat.require_ready`` before any POST).
"""
from __future__ import annotations

import pytest

from dart_semantic.council.cross_reranker import arbitrate
from dart_semantic.council.types import BertOutput, CouncilState, TypedSignal
from dart_semantic.extract import blocks_from_shared
from dart_semantic.extract_shared import (
    VLMSeat,
    VLMSeatError,
    _attach_vlm_struct_hints,
    mint_vlm_hint,
    resolve_vlm_base_url,
    resolve_vlm_extract_mode,
    resolve_vlm_model,
    resolve_vlm_provider,
    resolve_vlm_seat,
    resolve_vlm_struct_hints_mode,
)
from dart_semantic.features import (
    _interleave_image_feature_blocks,
    featurize_from_shared,
)
from dart_semantic.structure_graph import (
    _level_hint,
    _vlm_struct_hints_enabled,
    build_structure_graph,
)
from dart_semantic.types import FeatureBlock, RawBlock

_EXTRACT = "SEMANTIK_VLM_EXTRACT"
_HINTS = "SEMANTIK_VLM_STRUCT_HINTS"


# ===========================================================================
# 1. Flag resolution.
# ===========================================================================


def test_extract_flag_default_off():
    assert resolve_vlm_extract_mode() is False


@pytest.mark.parametrize("val", ["1", "true", "yes", "on", "ON", "True"])
def test_extract_flag_truthy(monkeypatch, val):
    monkeypatch.setenv(_EXTRACT, val)
    assert resolve_vlm_extract_mode() is True


@pytest.mark.parametrize("val", ["0", "false", "no", "off", "", "garbage"])
def test_extract_flag_falsey_or_garbage(monkeypatch, val):
    monkeypatch.setenv(_EXTRACT, val)
    assert resolve_vlm_extract_mode() is False


def test_struct_hints_subordinate_to_extract(monkeypatch):
    # Hints on but extract OFF -> hints inert (no VLM markdown to mint from).
    monkeypatch.setenv(_HINTS, "1")
    monkeypatch.delenv(_EXTRACT, raising=False)
    assert resolve_vlm_struct_hints_mode() is False
    assert _vlm_struct_hints_enabled() is False
    # Both on -> live.
    monkeypatch.setenv(_EXTRACT, "1")
    assert resolve_vlm_struct_hints_mode() is True
    assert _vlm_struct_hints_enabled() is True
    # Extract on but hints off -> off.
    monkeypatch.setenv(_HINTS, "0")
    assert resolve_vlm_struct_hints_mode() is False


# ===========================================================================
# 2. Provider-agnostic seat resolution (mock the boundary, never a live call).
# ===========================================================================


def test_seat_defaults_local_qwen(monkeypatch):
    for k in ("SEMANTIK_VLM_PROVIDER", "SEMANTIK_VLM_BASE_URL",
              "SEMANTIK_VLM_API_KEY", "SEMANTIK_VLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    seat = resolve_vlm_seat()
    assert seat.provider == "local"
    assert seat.base_url == "http://localhost:11434"
    assert seat.model == "qwen2.5vl:7b"
    assert seat.is_local is True
    # A local seat requires no credential.
    assert seat.require_ready() is seat


@pytest.mark.parametrize("alias", ["", "local", "ollama", "gguf", "LOCAL"])
def test_seat_local_aliases(monkeypatch, alias):
    monkeypatch.setenv("SEMANTIK_VLM_PROVIDER", alias)
    assert resolve_vlm_provider() == "local"


def test_seat_explicit_env_wins(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_PROVIDER", "nvidia")
    monkeypatch.setenv("SEMANTIK_VLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.setenv("SEMANTIK_VLM_API_KEY", "secret")
    monkeypatch.setenv("SEMANTIK_VLM_MODEL", "some/vl-model")
    seat = resolve_vlm_seat()
    assert seat.provider == "nvidia"
    assert seat.base_url == "https://api.example.com/v1"
    assert seat.api_key == "secret"
    assert seat.model == "some/vl-model"
    assert seat.is_local is False
    assert seat.require_ready() is seat  # has key -> ready


def test_seat_nonlocal_remote_without_key_fails_loud(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_PROVIDER", "nvidia")
    monkeypatch.setenv("SEMANTIK_VLM_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("SEMANTIK_VLM_API_KEY", raising=False)
    seat = resolve_vlm_seat()
    with pytest.raises(VLMSeatError):
        seat.require_ready()


def test_seat_nonlocal_loopback_without_key_ok(monkeypatch):
    # A non-local provider pointed at localhost (e.g. a local OpenAI-compatible
    # server) needs no credential — loopback is trusted.
    monkeypatch.setenv("SEMANTIK_VLM_PROVIDER", "local-openai")
    monkeypatch.setenv("SEMANTIK_VLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("SEMANTIK_VLM_API_KEY", raising=False)
    seat = resolve_vlm_seat()
    assert seat.require_ready() is seat


def test_seat_garbage_model_falls_back(monkeypatch):
    monkeypatch.setenv("SEMANTIK_VLM_MODEL", "   ")
    assert resolve_vlm_model() == "qwen2.5vl:7b"
    monkeypatch.delenv("SEMANTIK_VLM_BASE_URL", raising=False)
    assert resolve_vlm_base_url() == "http://localhost:11434"


# ===========================================================================
# 3. Hint minting from VLM markdown shape.
# ===========================================================================


@pytest.mark.parametrize("md,level", [
    ("# Chapter 9", 1),
    ("## 9.1 Simplify Radicals", 2),
    ("###   Deep Section", 3),
    ("######  h6", 6),
])
def test_mint_heading(md, level):
    h = mint_vlm_hint(md, "whole_block")
    assert h == {"kind": "heading", "level": level, "marker": None,
                 "coverage": "whole_block"}


@pytest.mark.parametrize("md,marker", [
    ("- a bullet", "-"),
    ("* star bullet", "*"),
    ("+ plus bullet", "+"),
    ("1. ordered item", "1"),
    ("12) paren ordered", "12"),
])
def test_mint_list_item(md, marker):
    h = mint_vlm_hint(md)
    assert h["kind"] == "list_item"
    assert h["marker"] == marker
    assert h["level"] is None


def test_mint_table_from_separator():
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    assert mint_vlm_hint(md)["kind"] == "table"


def test_mint_table_from_pipe_row():
    assert mint_vlm_hint("Name | Value | Unit")["kind"] == "table"


@pytest.mark.parametrize("md", [
    None, "", "   ", "Just ordinary prose with no structural markup at all.",
])
def test_mint_prose_or_empty_is_none(md):
    assert mint_vlm_hint(md) is None


def test_mint_coverage_prefix_preserved():
    h = mint_vlm_hint("## 9.1 Title", "prefix")
    assert h["coverage"] == "prefix"


def test_mint_coverage_unknown_collapses_to_prefix():
    # An UNKNOWN coverage value must fail CONSERVATIVE (non-corroborating
    # "prefix"), never fail-open to the corroboration-eligible "whole_block".
    h = mint_vlm_hint("## 9.1 Title", "garbage")
    assert h["coverage"] == "prefix"


def test_attach_missing_coverage_defaults_to_prefix():
    # A block that carries vlm_md but NO vlm_coverage key must default to the
    # conservative "prefix" — a future stamper omitting the key can never make
    # every fused block heading-corroboration-eligible.
    blocks = [{"text": "## 9.1 Roots", "vlm_md": "## 9.1 Roots"}]
    _attach_vlm_struct_hints(blocks)
    assert blocks[0]["vlm_hint"]["coverage"] == "prefix"


def test_attach_stamps_from_vlm_md():
    blocks = [
        {"text": "## 9.1 Roots", "vlm_md": "## 9.1 Roots", "vlm_coverage": "whole_block"},
        {"text": "body prose"},  # no vlm_md -> no hint
        {"text": "x", "vlm_md": "plain prose line"},  # non-structural -> no hint
    ]
    _attach_vlm_struct_hints(blocks)
    assert blocks[0]["vlm_hint"]["kind"] == "heading"
    assert "vlm_hint" not in blocks[1]
    assert "vlm_hint" not in blocks[2]


# ===========================================================================
# 4. Carriage: merged dict -> RawBlock -> FeatureBlock, image-FB interleave.
# ===========================================================================


def _shared_one_page(blocks: list[dict]) -> dict:
    return {
        "pages": [{
            "page_num": 1, "width": 612.0, "height": 792.0,
            "merged": {"text_blocks": blocks},
        }],
    }


def _mblock(text: str, x0: float, hint: dict | None = None) -> dict:
    b = {
        "text": text, "bbox": [x0, 0.0, x0 + 10.0, 10.0],
        "provenance": ["pdfplumber"], "confidence": 1.0,
    }
    if hint is not None:
        b["vlm_hint"] = hint
    return b


def test_carriage_rawblock_and_none_default():
    hint = {"kind": "heading", "level": 2, "marker": None, "coverage": "whole_block"}
    shared = _shared_one_page([_mblock("H", 0.0, hint), _mblock("body", 20.0)])
    raws = blocks_from_shared(shared)
    assert raws[0].vlm_hint == hint
    assert raws[1].vlm_hint is None  # no key -> None, no KeyError


def test_carriage_featureblock_mirror():
    hint = {"kind": "heading", "level": 2, "marker": None, "coverage": "whole_block"}
    shared = _shared_one_page([_mblock("H", 0.0, hint), _mblock("body", 20.0)])
    fbs = featurize_from_shared(shared)
    assert fbs[0].vlm_hint == hint
    assert fbs[1].vlm_hint is None


def test_carriage_survives_image_fb_interleave():
    # The hint travels ON the FB object, so an image-FB interleave (which shifts
    # indices) never misaligns it — index-keyed side-tables would break here.
    hint = {"kind": "heading", "level": 1, "marker": None, "coverage": "whole_block"}
    top = _fb_at("top heading", y0=0.0)
    top.vlm_hint = hint
    bottom = _fb_at("bottom body", y0=200.0)
    fbs = [top, bottom]

    from types import SimpleNamespace
    # An image candidate whose bbox sorts BETWEEN the two text FBs.
    cand = SimpleNamespace(bbox=(0.0, 100.0, 50.0, 150.0), pages=[1],
                           member_block_indices=[])
    out = _interleave_image_feature_blocks(fbs, [cand])

    # The image FB is inserted between -> the heading FB is no longer index 0's
    # neighbour, but its hint stayed with the object.
    heads = [fb for fb in out if getattr(fb, "vlm_hint", None) == hint]
    assert len(heads) == 1
    assert heads[0].raw.text == "top heading"
    # The synthetic image FB carries no hint.
    imgs = [fb for fb in out if getattr(fb, "is_image", False)]
    assert imgs and all(getattr(fb, "vlm_hint", None) is None for fb in imgs)


# ===========================================================================
# 5. structure_graph Pass-2 heading CORROBORATION.
# ===========================================================================


def _fb_at(text: str, *, ratio: float = 1.0, bold: bool = False,
           y0: float = 0.0) -> FeatureBlock:
    raw = RawBlock(
        text=text, page=1, bbox=(0.0, y0, 100.0, y0 + 20.0),
        page_width=200.0, page_height=800.0, font_size=11.0, is_bold=bold,
    )
    return FeatureBlock(
        raw=raw, size_bucket="lg" if ratio >= 1.3 else "md", gap_above=None,
        is_top_of_page=False, is_centered=False, caps=None, indent_bucket=0,
        relative_font_ratio=ratio, is_image=False,
    )


def _heading_sig(idx: int) -> list[TypedSignal]:
    return [
        TypedSignal("is_heading", idx, ["heading", "body"], [0.97, 0.03]),
        TypedSignal("structural_role", idx, ["heading", "paragraph"], [0.95, 0.05]),
    ]


def _para_sig(idx: int) -> list[TypedSignal]:
    return [
        TypedSignal("is_heading", idx, ["body", "heading"], [0.99, 0.01]),
        TypedSignal("structural_role", idx, ["paragraph", "heading"], [0.95, 0.05]),
    ]


def _graph(fbs, sigs):
    state = CouncilState(outputs={"structure": BertOutput("structure", sigs)})
    decisions = arbitrate(state, [])
    return build_structure_graph(state, fbs, [], decisions)


def _find(regions, kind):
    return [r for r in regions if r.kind == kind]


def test_pass2_corroboration_positive(monkeypatch):
    monkeypatch.setenv(_EXTRACT, "1")
    monkeypatch.setenv(_HINTS, "1")
    fbs = [
        _fb_at("Chapter 1 Introduction", ratio=1.5, bold=True, y0=0.0),
        _fb_at("This is ordinary body prose spanning a full sentence here.", y0=40.0),
    ]
    fbs[0].vlm_hint = {"kind": "heading", "level": 2, "marker": None,
                       "coverage": "whole_block"}
    heads = _find(_graph(fbs, _heading_sig(0) + _para_sig(1)), "heading")
    assert len(heads) == 1
    assert heads[0].payload.get("vlm_corroborated") is True
    assert heads[0].payload["confidence"] > 0.97  # bounded lift recorded


def test_pass2_no_hint_kind_invariant(monkeypatch):
    # Same candidate WITHOUT a hint -> same heading region, NO vlm keys, and
    # confidence is the raw council value (byte-identical decision).
    monkeypatch.setenv(_EXTRACT, "1")
    monkeypatch.setenv(_HINTS, "1")
    fbs = [
        _fb_at("Chapter 1 Introduction", ratio=1.5, bold=True, y0=0.0),
        _fb_at("This is ordinary body prose spanning a full sentence here.", y0=40.0),
    ]
    heads = _find(_graph(fbs, _heading_sig(0) + _para_sig(1)), "heading")
    assert len(heads) == 1
    assert "vlm_corroborated" not in heads[0].payload
    assert heads[0].payload["confidence"] == pytest.approx(0.97)


def test_pass2_flags_off_byte_identical(monkeypatch):
    # A whole-block heading hint present but the flags OFF -> no corroboration,
    # confidence unchanged (the strict gate preserves byte-identical-off).
    monkeypatch.delenv(_EXTRACT, raising=False)
    monkeypatch.delenv(_HINTS, raising=False)
    fbs = [
        _fb_at("Chapter 1 Introduction", ratio=1.5, bold=True, y0=0.0),
        _fb_at("This is ordinary body prose spanning a full sentence here.", y0=40.0),
    ]
    fbs[0].vlm_hint = {"kind": "heading", "level": 2, "marker": None,
                       "coverage": "whole_block"}
    heads = _find(_graph(fbs, _heading_sig(0) + _para_sig(1)), "heading")
    assert len(heads) == 1
    assert "vlm_corroborated" not in heads[0].payload
    assert heads[0].payload["confidence"] == pytest.approx(0.97)


def test_pass2_no_bypass_hint_cannot_admit(monkeypatch):
    # A block FAILING the is_heading threshold (body) + a heading hint -> still
    # NOT a heading (the hint cannot mint a candidate or bypass the gate).
    monkeypatch.setenv(_EXTRACT, "1")
    monkeypatch.setenv(_HINTS, "1")
    fbs = [
        _fb_at("1.1 Anchor Heading", ratio=1.5, bold=True, y0=0.0),
        _fb_at("this is body prose that the council labels body text here.", y0=40.0),
    ]
    fbs[1].vlm_hint = {"kind": "heading", "level": 2, "marker": None,
                       "coverage": "whole_block"}
    regions = _graph(fbs, _heading_sig(0) + _para_sig(1))
    # fb[1] stays prose; no region claims it as a heading + it is not corroborated.
    for r in _find(regions, "heading"):
        assert 1 not in r.feature_block_indices
        assert "vlm_corroborated" not in r.payload


def test_pass2_prefix_coverage_does_not_corroborate(monkeypatch):
    # A prefix-coverage heading hint (a fused-title partial alignment) must NOT
    # corroborate -> no vlm keys even on a real heading candidate.
    monkeypatch.setenv(_EXTRACT, "1")
    monkeypatch.setenv(_HINTS, "1")
    fbs = [
        _fb_at("Chapter 1 Introduction", ratio=1.5, bold=True, y0=0.0),
        _fb_at("This is ordinary body prose spanning a full sentence here.", y0=40.0),
    ]
    fbs[0].vlm_hint = {"kind": "heading", "level": 2, "marker": None,
                       "coverage": "prefix"}
    heads = _find(_graph(fbs, _heading_sig(0) + _para_sig(1)), "heading")
    assert "vlm_corroborated" not in heads[0].payload


# ===========================================================================
# 6. Level tiebreak — SECONDARY only, deterministic wins when unambiguous.
# ===========================================================================


def test_level_hint_ambiguous_uses_vlm():
    # Body-sized font (ratio 1.0 -> font_level 6), no numbered prefix -> the
    # deterministic inference is ambiguous, so the VLM level is consulted.
    fb = _fb_at("Some Short Title Text", ratio=1.0)
    assert _level_hint(fb) == 6
    assert _level_hint(fb, vlm_level=2) == 2


def test_level_hint_unambiguous_font_wins_over_vlm():
    # A visibly enlarged heading (ratio 1.5 -> font_level 2) is unambiguous;
    # a conflicting VLM level does NOT override it.
    fb = _fb_at("Big Heading", ratio=1.5)
    assert _level_hint(fb) == 2
    assert _level_hint(fb, vlm_level=5) == 2


def test_level_hint_numbered_wins_over_vlm():
    # A numbered-section prefix is deterministic -> the VLM level is ignored.
    fb = _fb_at("9.1 Simplify Radicals", ratio=1.0)
    det = _level_hint(fb)
    assert _level_hint(fb, vlm_level=1) == det


def test_level_hint_garbage_vlm_level_ignored():
    fb = _fb_at("Some Short Title Text", ratio=1.0)
    assert _level_hint(fb, vlm_level="x") == 6      # non-int -> font_level
    assert _level_hint(fb, vlm_level=99) == 6       # out of range -> font_level


# ===========================================================================
# 7. Extract-cache salt — flipping SEMANTIK_VLM_EXTRACT invalidates the entry
#    (the SEMANTIK_DETECT_FIGURES in-key precedent, NOT render-scale kept-out).
# ===========================================================================


def test_vlm_flag_salts_extract_cache_key(monkeypatch, tmp_path):
    import dart_semantic.extract_shared as es

    calls = {"n": 0}

    def _stub_extract(pdf_path):
        calls["n"] += 1
        return {"pages": [], "_call": calls["n"]}

    monkeypatch.setattr(es, "extract_shared", _stub_extract)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    cache = tmp_path / "cache"

    monkeypatch.delenv(_EXTRACT, raising=False)
    es.extract_shared_cached(pdf, cache_dir=cache)   # miss (no vlm salt) -> 1
    es.extract_shared_cached(pdf, cache_dir=cache)   # hit  (no vlm salt) -> still 1
    assert calls["n"] == 1

    monkeypatch.setenv(_EXTRACT, "1")
    es.extract_shared_cached(pdf, cache_dir=cache)   # miss (vlm1) -> 2
    assert calls["n"] == 2, "flipping SEMANTIK_VLM_EXTRACT must not serve a stale entry"
    es.extract_shared_cached(pdf, cache_dir=cache)   # hit  (vlm1) -> still 2
    assert calls["n"] == 2


def test_hints_flag_salts_extract_cache_key(monkeypatch, tmp_path):
    # SEMANTIK_VLM_STRUCT_HINTS bakes vlm_hint into the cached merged artifact
    # (attach runs inside _merge_page), so flipping it ON — with EXTRACT already
    # warm — must invalidate the entry, never serve a stale hint-less extraction
    # (the "ship fusion text-only, flip hints later" flow).
    import dart_semantic.extract_shared as es

    calls = {"n": 0}

    def _stub_extract(pdf_path):
        calls["n"] += 1
        return {"pages": [], "_call": calls["n"]}

    monkeypatch.setattr(es, "extract_shared", _stub_extract)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    cache = tmp_path / "cache"

    # EXTRACT on, HINTS off -> key has vlm1 but no vlmhints1.
    monkeypatch.setenv(_EXTRACT, "1")
    monkeypatch.delenv(_HINTS, raising=False)
    es.extract_shared_cached(pdf, cache_dir=cache)   # miss -> 1
    es.extract_shared_cached(pdf, cache_dir=cache)   # hit  -> still 1
    assert calls["n"] == 1

    # Flip HINTS on -> key gains vlmhints1 -> must MISS (no stale hint-less hit).
    monkeypatch.setenv(_HINTS, "1")
    es.extract_shared_cached(pdf, cache_dir=cache)   # miss -> 2
    assert calls["n"] == 2, "flipping SEMANTIK_VLM_STRUCT_HINTS must not serve a stale entry"
    es.extract_shared_cached(pdf, cache_dir=cache)   # hit  -> still 2
    assert calls["n"] == 2


def test_seat_nonlocal_lookalike_host_requires_key(monkeypatch):
    # A non-loopback host that merely CONTAINS "localhost" as a substring
    # (localhost.evil.example) must NOT be trusted as loopback — the seat must
    # still fail loud without a credential (urlparse hostname equality, not a
    # substring scan).
    monkeypatch.setenv("SEMANTIK_VLM_PROVIDER", "evilcorp")
    monkeypatch.setenv("SEMANTIK_VLM_BASE_URL", "https://localhost.evil.example/v1")
    monkeypatch.delenv("SEMANTIK_VLM_API_KEY", raising=False)
    seat = resolve_vlm_seat()
    with pytest.raises(VLMSeatError):
        seat.require_ready()


def test_loopback_helper_hostname_equality():
    # Direct unit coverage of the fail-closed helper: real loopback hosts pass,
    # substring look-alikes / query-string injections do not.
    from dart_semantic.extract_shared import _is_loopback_base_url

    assert _is_loopback_base_url("http://localhost:11434/v1") is True
    assert _is_loopback_base_url("http://127.0.0.1:11434") is True
    assert _is_loopback_base_url("http://[::1]:8000") is True
    assert _is_loopback_base_url("http://0.0.0.0:11434") is True
    assert _is_loopback_base_url("https://localhost.evil.example/v1") is False
    assert _is_loopback_base_url("https://api.example.com/?h=0.0.0.0") is False
    assert _is_loopback_base_url("https://127.0.0.1.evil.example") is False


def test_default_extract_no_hint_keys(monkeypatch):
    # Byte-identical off: featurize a shared page with NO vlm_md and flags unset
    # -> every FeatureBlock.vlm_hint is None (no vlm_* leakage).
    monkeypatch.delenv(_EXTRACT, raising=False)
    monkeypatch.delenv(_HINTS, raising=False)
    shared = _shared_one_page([_mblock("H", 0.0), _mblock("body", 20.0)])
    fbs = featurize_from_shared(shared)
    assert all(fb.vlm_hint is None for fb in fbs)
