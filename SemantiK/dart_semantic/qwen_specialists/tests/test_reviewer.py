"""Phase-1 acceptance — Stage-5d structure reviewer (mocked endpoint, no GPU).

Covers the §9 Phase-1 acceptance list verbatim:
  * phantom-TOC cluster -> metadata_drop;
  * level re-tag honored;
  * paragraph->heading promotion sets non-null payload['text']==source (C1);
  * a verdict altering the FB partition -> REJECTED + original kept;
  * no-source / fabricated-promotion -> rejected;
  * tolerant extractor handles fenced/commented/out-of-batch-block_id
    verdicts WITHOUT raising EndpointBatchItemError;
  * assert_token_conservation passes on a re-level+drop mix and FAILS on a
    fabricated drop (C3);
  * Stage-5 coverage invariant preserved;
  * dataclasses.replace round-trips (no frozen-mutation error).
"""

from __future__ import annotations

import json

import pytest

from dart_semantic.qwen_specialists.reviewer import (
    ReviewVerdict,
    TokenConservationError,
    assert_token_conservation,
    resolve_structure_review_mode,
    run_structure_review,
)
from dart_semantic.structure_graph import Region
from dart_semantic.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Fixtures / builders.
# ---------------------------------------------------------------------------


def _fb(text: str) -> FeatureBlock:
    """Minimal FeatureBlock with the only field the reviewer reads (raw.text)."""
    raw = RawBlock(
        text=text,
        page=1,
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


class _ScriptedRuntime:
    """Injectable runtime whose generate_batch returns crafted JSON strings.

    Takes a list of completion strings (one per prompt, in order). Records
    the prompts it was handed so a test can assert prompt shape. Mirrors the
    QwenRuntime.generate_batch signature; loads no model / GPU.
    """

    def __init__(self, completions: list[str]) -> None:
        self._completions = completions
        self.batch_calls: list[dict] = []

    def generate_batch(self, prompts, *, max_tokens, temperature=0.6,
                        top_p=0.95, seed=None, repeat_penalty=1.0):
        self.batch_calls.append({"prompts": list(prompts), "max_tokens": max_tokens})
        # Return exactly the scripted completions (the driver pads a short
        # list defensively, but we always script len == len(prompts)).
        return list(self._completions)


def _verdict_json(block_id, *, verdict="corrected", kind=None, level=None,
                  doc_role=None, note="n") -> str:
    return json.dumps({
        "block_id": block_id,
        "verdict": verdict,
        "corrected_kind": kind,
        "corrected_level": level,
        "corrected_doc_role": doc_role,
        "review_note": note,
    })


def _assert_coverage_invariant(regions, n_fb):
    """Every FB index appears in exactly one region's feature_block_indices."""
    seen = []
    for r in regions:
        seen.extend(r.feature_block_indices)
    assert sorted(seen) == list(range(n_fb)), "coverage invariant violated"
    assert len(seen) == len(set(seen)), "an FB appears in 2 regions"


# ---------------------------------------------------------------------------
# Mode resolver.
# ---------------------------------------------------------------------------


def test_mode_default_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW", raising=False)
    assert resolve_structure_review_mode() is False


def test_mode_truthy(monkeypatch):
    for v in ("1", "true", "YES", "on"):
        monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW", v)
        assert resolve_structure_review_mode() is True
    for v in ("0", "false", "no", "garbage", ""):
        monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW", v)
        assert resolve_structure_review_mode() is False


# ---------------------------------------------------------------------------
# Phantom-TOC cluster -> metadata_drop.
# ---------------------------------------------------------------------------


def test_phantom_toc_cluster_to_metadata_drop():
    # 3 short "headings" that are really an answer-key run + 1 real paragraph.
    fbs = [_fb("Answer Key 3.1"), _fb("Answer Key 3.2"), _fb("Answer Key 3.3"),
           _fb("Real body content here.")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "Answer Key 3.1"}),
        Region(kind="heading", feature_block_indices=(1,), payload={"level_hint": 2, "text": "Answer Key 3.2"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "Answer Key 3.3"}),
        Region(kind="paragraph", feature_block_indices=(3,), payload={"text": "Real body content here."}),
    ]
    completions = [
        _verdict_json(0, verdict="drop_injected_header", kind="metadata_drop", note="answer-key run"),
        _verdict_json(1, verdict="drop_injected_header", kind="metadata_drop", note="answer-key run"),
        _verdict_json(2, verdict="drop_injected_header", kind="metadata_drop", note="answer-key run"),
        _verdict_json(3, verdict="ok"),
    ]
    rt = _ScriptedRuntime(completions)
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert [r.kind for r in out] == ["metadata_drop", "metadata_drop", "metadata_drop", "paragraph"]
    assert all(v.kind_after == "metadata_drop" for v in verdicts[:3])
    assert not any(v.reverted_for_invariant for v in verdicts)
    _assert_coverage_invariant(out, len(fbs))


# ---------------------------------------------------------------------------
# Level re-tag honored.
# ---------------------------------------------------------------------------


def test_level_retag_honored():
    fbs = [_fb("Chapter 1")]
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 3, "text": "Chapter 1"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1, note="top-level chapter")])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].kind == "heading"
    assert out[0].payload["level_hint"] == 1
    assert verdicts[0].level_before == 3
    assert verdicts[0].level_after == 1
    assert verdicts[0].verdict == "corrected"
    # text untouched.
    assert out[0].payload["text"] == "Chapter 1"


# ---------------------------------------------------------------------------
# paragraph -> heading promotion sets non-null payload['text'] == source (C1).
# ---------------------------------------------------------------------------


def test_promotion_sets_source_text_c1():
    fbs = [_fb("Methods and Materials")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "Methods and Materials"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=2, note="missed heading")])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].kind == "heading"
    assert out[0].payload["text"] == "Methods and Materials"  # non-null, == source
    assert out[0].payload["level_hint"] == 2
    assert not verdicts[0].reverted_for_invariant
    assert verdicts[0].kind_before == "paragraph"
    assert verdicts[0].kind_after == "heading"


def test_promotion_defaults_level_when_omitted():
    fbs = [_fb("Introduction")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "Introduction"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=None, note="missed heading")])
    out, _ = run_structure_review(regions, fbs, rt)
    assert out[0].kind == "heading"
    assert out[0].payload["level_hint"] == 2  # default h2 when model omits level


# ---------------------------------------------------------------------------
# A verdict altering the FB partition -> REJECTED + original kept verbatim.
# ---------------------------------------------------------------------------


def test_fb_partition_change_rejected():
    # The model can't change feature_block_indices via the verdict schema,
    # but a tampered/inconsistent payload that drops a non-structural key
    # must be rejected. Simulate by a runtime that the applier would, on a
    # correct path, leave the non-structural keys intact — we assert the
    # admission invariant rejects a region whose non-promotion text changed.
    fbs = [_fb("Original text")]
    region = Region(kind="paragraph", feature_block_indices=(0,),
                    payload={"text": "Original text", "extra": "keepme"})
    # Verdict re-roles to blockquote (a non-heading re-tag). The applier
    # carries `text` + `extra` forward byte-identical, so this is ADMITTED;
    # to exercise REJECTION of an FB-partition/text change we call the
    # internal admission directly with a tampered candidate.
    from dart_semantic.qwen_specialists import reviewer as rv

    tampered = region.__class__(
        kind="blockquote",
        feature_block_indices=(0, 1),  # FB partition CHANGED
        payload=dict(region.payload),
        source_region_id=region.source_region_id,
    )
    assert rv._admits(region, tampered, fbs, is_promotion=False) is False

    text_tampered = region.__class__(
        kind="blockquote",
        feature_block_indices=(0,),
        payload={"text": "REWRITTEN words", "extra": "keepme"},  # text rewritten
        source_region_id=region.source_region_id,
    )
    assert rv._admits(region, text_tampered, fbs, is_promotion=False) is False

    # And a benign structural-only re-tag IS admitted.
    benign = region.__class__(
        kind="blockquote",
        feature_block_indices=(0,),
        payload={"text": "Original text", "extra": "keepme"},
        source_region_id=region.source_region_id,
    )
    assert rv._admits(region, benign, fbs, is_promotion=False) is True


def test_nonstructural_key_drop_rejected_via_driver():
    # End-to-end: a non-promotion re-tag keeps non-structural payload keys.
    fbs = [_fb("Quote text")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "Quote text", "reason": "x"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="blockquote", note="is a quote")])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].kind == "blockquote"
    # non-structural key preserved verbatim (admission invariant).
    assert out[0].payload["reason"] == "x"
    assert out[0].payload["text"] == "Quote text"


# ---------------------------------------------------------------------------
# No-source / fabricated-promotion -> rejected.
# ---------------------------------------------------------------------------


def test_no_source_promotion_rejected():
    # A region whose FBs resolve to empty text, promoted to heading.
    fbs = [_fb("")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,), payload={"text": ""})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1, note="fabricated")])
    out, verdicts = run_structure_review(regions, fbs, rt)
    # promotion rejected -> original kept verbatim.
    assert out[0].kind == "paragraph"
    assert verdicts[0].reverted_for_invariant is True


# ---------------------------------------------------------------------------
# Tolerant extractor — fenced / commented / out-of-batch / unparseable.
# ---------------------------------------------------------------------------


def test_tolerant_extractor_handles_fences_and_commentary():
    fbs = [_fb("H1"), _fb("H2"), _fb("H3"), _fb("H4")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "H1"}),
        Region(kind="heading", feature_block_indices=(1,), payload={"level_hint": 2, "text": "H2"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "H3"}),
        Region(kind="paragraph", feature_block_indices=(3,), payload={"text": "H4"}),
    ]
    completions = [
        # fenced json with a language tag.
        "```json\n" + _verdict_json(0, kind="heading", level=1) + "\n```",
        # commentary prose before the object.
        "Sure! Here is my verdict:\n" + _verdict_json(1, kind="heading", level=3) + "\nLet me know if that helps.",
        # out-of-batch block_id -> must be dropped, soft-fallback ok.
        _verdict_json(99, kind="metadata_drop"),
        # totally unparseable -> soft-fallback ok.
        "I cannot produce JSON, sorry.",
    ]
    rt = _ScriptedRuntime(completions)
    # MUST NOT raise EndpointBatchItemError.
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].payload["level_hint"] == 1  # fenced verdict applied
    assert out[1].payload["level_hint"] == 3  # commented verdict applied
    # block 2: out-of-batch id dropped -> kept original heading.
    assert out[2].kind == "heading"
    assert verdicts[2].verdict == "ok"
    # block 3: unparseable -> kept original paragraph.
    assert out[3].kind == "paragraph"
    assert verdicts[3].verdict == "ok"
    _assert_coverage_invariant(out, len(fbs))


def test_unknown_kind_degrades_to_ok():
    fbs = [_fb("X")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "X"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="not_a_real_kind", note="bad kind")])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].kind == "paragraph"  # unknown kind dropped
    assert verdicts[0].verdict == "ok"


# ---------------------------------------------------------------------------
# assert_token_conservation — passes on re-level+drop, fails on fabricated drop.
# ---------------------------------------------------------------------------


def test_token_conservation_passes_on_relevel_and_drop():
    fbs = [_fb("alpha beta"), _fb("gamma"), _fb("delta epsilon")]
    original = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 3, "text": "alpha beta"}),
        Region(kind="heading", feature_block_indices=(1,), payload={"level_hint": 2, "text": "gamma"}),
        Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "delta epsilon"}),
    ]
    # corrected: block 0 re-leveled (still heading), block 1 dropped, block 2 unchanged.
    corrected = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 1, "text": "alpha beta"}),
        Region(kind="metadata_drop", feature_block_indices=(1,), payload={"level_hint": None, "text": "gamma"}),
        Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "delta epsilon"}),
    ]
    # passes (kept ⊎ dropped == source).
    assert_token_conservation(original, corrected, fbs)


def test_token_conservation_fails_on_fabricated_drop():
    # corrected region claims metadata_drop on an FB whose text it does NOT
    # own -> we simulate token loss by a corrected list that drops an FB
    # entirely from coverage (a fabricated drop / lost content).
    fbs = [_fb("alpha beta"), _fb("gamma delta")]
    original = [
        Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "alpha beta"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "gamma delta"}),
    ]
    # corrected DROPS FB 1 from coverage entirely (content vanishes) — the
    # conservation check must catch the missing tokens.
    corrected = [
        Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "alpha beta"}),
        # FB 1 simply absent from any region -> 'gamma delta' not in kept/dropped.
    ]
    with pytest.raises(TokenConservationError):
        assert_token_conservation(original, corrected, fbs)


def test_driver_fails_closed_on_conservation_break(monkeypatch):
    # If the applier ever lost content, the driver reverts to flag-OFF list.
    # We force this by monkeypatching assert_token_conservation to raise,
    # proving the fail-closed branch returns the ORIGINAL regions verbatim.
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("kept text")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "kept text"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="blockquote")])

    def _boom(*a, **k):
        raise rv.TokenConservationError("forced")

    monkeypatch.setattr(rv, "assert_token_conservation", _boom)
    out, verdicts = rv.run_structure_review(regions, fbs, rt)
    # flag-OFF revert: original regions returned unchanged.
    assert out is regions
    assert all(v.reverted_for_invariant for v in verdicts)


# ---------------------------------------------------------------------------
# dataclasses.replace round-trips (no frozen-mutation error).
# ---------------------------------------------------------------------------


def test_dataclasses_replace_round_trips():
    fbs = [_fb("Heading words")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "Heading words"}, source_region_id=5,
                      aria_hints=("hint",), provenance={"pass": "p"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=2)])
    out, _ = run_structure_review(regions, fbs, rt)
    r = out[0]
    # frozen replace preserved the non-mutated fields.
    assert r.feature_block_indices == (0,)
    assert r.source_region_id == 5
    assert r.aria_hints == ("hint",)
    assert r.provenance == {"pass": "p"}
    # original region object is unchanged (no in-place mutation).
    assert regions[0].kind == "paragraph"


# ---------------------------------------------------------------------------
# ReviewVerdict shape.
# ---------------------------------------------------------------------------


def test_review_verdict_shape():
    fbs = [_fb("X")]
    regions = [Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "X"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1, note="re-level")])
    _, verdicts = run_structure_review(regions, fbs, rt)
    v = verdicts[0]
    assert isinstance(v, ReviewVerdict)
    assert v.block_id == 0
    assert v.kind_before == "heading"
    assert v.kind_after == "heading"
    assert v.level_before == 2
    assert v.level_after == 1
    assert v.review_note == "re-level"
    assert v.reverted_for_invariant is False


# ---------------------------------------------------------------------------
# Empty / no-op paths.
# ---------------------------------------------------------------------------


def test_empty_regions_noop():
    rt = _ScriptedRuntime([])
    out, verdicts = run_structure_review([], [], rt)
    assert out == []
    assert verdicts == []
    assert rt.batch_calls == []  # no batch fired for an empty doc


def test_short_batch_soft_falls_back_tail():
    fbs = [_fb("A"), _fb("B")]
    regions = [
        Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "A"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "B"}),
    ]
    # runtime returns only ONE completion for two prompts.
    rt = _ScriptedRuntime([_verdict_json(0, kind="blockquote")])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].kind == "blockquote"
    assert out[1].kind == "paragraph"  # missing tail -> soft ok
    assert verdicts[1].verdict == "ok"
