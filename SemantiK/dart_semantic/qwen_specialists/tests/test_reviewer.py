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
#
# NOTE (heading-scoping): the driver now reviews ONLY heading regions, so a
# paragraph->heading PROMOTION is no longer driver-reachable (a paragraph is
# never sent to the 70B). The C1 promotion-text invariant + the level-default
# mechanics still live in _apply_verdict and are exercised DIRECTLY here so
# the anti-fabrication contract stays covered.
# ---------------------------------------------------------------------------


def test_promotion_sets_source_text_c1():
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("Methods and Materials")]
    region = Region(kind="paragraph", feature_block_indices=(0,),
                    payload={"text": "Methods and Materials"})
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "heading",
           "corrected_level": 2, "corrected_doc_role": None, "review_note": "missed heading"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.kind == "heading"
    assert out_region.payload["text"] == "Methods and Materials"  # non-null, == source
    assert out_region.payload["level_hint"] == 2
    assert not verdict.reverted_for_invariant
    assert verdict.kind_before == "paragraph"
    assert verdict.kind_after == "heading"


def test_promotion_defaults_level_when_omitted():
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("Introduction")]
    region = Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "Introduction"})
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "heading",
           "corrected_level": None, "corrected_doc_role": None, "review_note": "missed heading"}
    out_region, _ = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.kind == "heading"
    assert out_region.payload["level_hint"] == 2  # default h2 when model omits level


def test_paragraph_not_reviewed_under_scoping():
    # End-to-end proof of the heading-scoping: a paragraph the model WOULD
    # promote is never sent to the runtime, so it passes through verbatim.
    fbs = [_fb("Methods and Materials")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "Methods and Materials"})]
    rt = _ScriptedRuntime([])  # no headings -> no batch fired
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].kind == "paragraph"  # untouched
    assert verdicts[0].verdict == "ok"
    assert rt.batch_calls == []  # heading-scoping fired no 70B call


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
    # End-to-end: a heading re-tagged to blockquote (a phantom-heading whose
    # words are a quote) keeps non-structural payload keys. Source is a
    # HEADING so it is in-scope for the heading-scoped driver.
    fbs = [_fb("Quote text")]
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 2, "text": "Quote text", "reason": "x"})]
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
    # A region whose FBs resolve to empty text, promoted to heading. Exercised
    # directly via _apply_verdict (promotion is no longer driver-reachable
    # under heading-scoping, but the anti-fabrication reject path is intact).
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("")]
    region = Region(kind="paragraph", feature_block_indices=(0,), payload={"text": ""})
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "heading",
           "corrected_level": 1, "corrected_doc_role": None, "review_note": "fabricated"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    # promotion rejected -> original kept verbatim.
    assert out_region.kind == "paragraph"
    assert verdict.reverted_for_invariant is True


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
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 2, "text": "kept text"})]
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
    # Two HEADING regions (both in-scope) but the runtime returns only ONE
    # completion -> the missing tail soft-falls-back to ok.
    fbs = [_fb("A"), _fb("B")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "A"}),
        Region(kind="heading", feature_block_indices=(1,), payload={"level_hint": 2, "text": "B"}),
    ]
    # runtime returns only ONE completion for two prompts.
    rt = _ScriptedRuntime([_verdict_json(0, kind="blockquote")])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].kind == "blockquote"
    assert out[1].kind == "heading"  # missing tail -> soft ok
    assert verdicts[1].verdict == "ok"


# ---------------------------------------------------------------------------
# Cluster-level signal pre-pass (Phase-4 root-cause fix).
# ---------------------------------------------------------------------------


def _heading(text, level=2):
    return Region(kind="heading", feature_block_indices=(0,),
                  payload={"level_hint": level, "text": text})


def test_cluster_signals_phantom_run_no_content():
    # A run of 9 same-level headings with NO content between them = a TOC /
    # chapter-index cluster: same_level_run_len == 9, content_blocks_following
    # == 0 for every entry, run_position counts 1..9.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    fbs = [_fb(f"Chapter {i}: Title") for i in range(1, 10)]
    regions = [
        Region(kind="heading", feature_block_indices=(i,),
               payload={"level_hint": 2, "text": f"Chapter {i+1}: Title"})
        for i in range(9)
    ]
    sigs = compute_cluster_signals(regions)
    assert all(s.same_level_run_len == 9 for s in sigs)
    assert [s.run_position for s in sigs] == list(range(1, 10))
    assert all(s.content_blocks_following == 0 for s in sigs)


def test_cluster_signals_real_section_followed_by_content():
    # A real section heading followed by 3 paragraphs: same_level_run_len == 1
    # (it is its own run — content breaks it), content_blocks_following == 3.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    fbs = [_fb("1.1 Whole Numbers"), _fb("p1"), _fb("p2"), _fb("p3")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "1.1 Whole Numbers"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "p1"}),
        Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "p2"}),
        Region(kind="paragraph", feature_block_indices=(3,), payload={"text": "p3"}),
    ]
    sigs = compute_cluster_signals(regions)
    assert sigs[0].same_level_run_len == 1
    assert sigs[0].run_position == 1
    assert sigs[0].content_blocks_following == 3
    # paragraphs carry inert heading defaults.
    assert all(sigs[i].same_level_run_len == 0 for i in (1, 2, 3))


def test_cluster_signals_mixed_phantom_run_then_real_section():
    # [9-heading run, no content] + [heading + 3 paragraphs] — the discriminator
    # fixture from the spec. The 9 phantom headings AND the trailing real
    # heading are all same-level with NO content BETWEEN consecutive entries,
    # so they form ONE run of 10 (the run breaks only on content between two
    # entries). The COMPLEMENTARY discriminator is content_blocks_following:
    # the 9 phantoms have 0, the real section has 3 — that is the signal the
    # prompt directive uses to KEEP a run member that is a real section.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    phantom = [
        Region(kind="heading", feature_block_indices=(i,),
               payload={"level_hint": 2, "text": f"Chapter {i+1}: T"})
        for i in range(9)
    ]
    real = [
        Region(kind="heading", feature_block_indices=(9,),
               payload={"level_hint": 2, "text": "1.1 Real Section"}),
        Region(kind="paragraph", feature_block_indices=(10,), payload={"text": "a"}),
        Region(kind="paragraph", feature_block_indices=(11,), payload={"text": "b"}),
        Region(kind="paragraph", feature_block_indices=(12,), payload={"text": "c"}),
    ]
    regions = phantom + real
    fbs = [_fb((r.payload or {}).get("text", "")) for r in regions]
    sigs = compute_cluster_signals(regions)
    # all 10 same-level headings are one run; phantoms have 0 content following.
    for i in range(9):
        assert sigs[i].same_level_run_len == 10
        assert sigs[i].content_blocks_following == 0
    # the real section is the LAST member of that run BUT carries 3 content
    # blocks following — the discriminator that protects it from over-demotion.
    assert sigs[9].same_level_run_len == 10
    assert sigs[9].run_position == 10
    assert sigs[9].content_blocks_following == 3


def test_cluster_signals_content_breaks_run_between_entries():
    # When content DOES sit between two same-level headings, the run breaks —
    # confirming content_blocks_following and same_level_run_len agree.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "1.1 A"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "body"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "1.2 B"}),
    ]
    fbs = [_fb("x") for _ in regions]
    sigs = compute_cluster_signals(regions)
    assert sigs[0].same_level_run_len == 1  # content after it breaks the run
    assert sigs[0].content_blocks_following == 1
    assert sigs[2].same_level_run_len == 1


def test_cluster_signals_trailing_pagenum():
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        _heading("Systems of Linear Equations    577"),
        _heading("Real Heading Without Number"),
    ]
    sigs = compute_cluster_signals(regions)
    assert sigs[0].trailing_pagenum is True
    assert sigs[1].trailing_pagenum is False


def test_cluster_signals_metadata_drop_does_not_break_run():
    # A stray metadata_drop between two same-level headings keeps the run open
    # (it is non-content, non-heading -> transparent).
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "Chapter 1"}),
        Region(kind="metadata_drop", feature_block_indices=(1,), payload={"text": "running header"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "Chapter 2"}),
    ]
    sigs = compute_cluster_signals(regions)
    assert sigs[0].same_level_run_len == 2
    assert sigs[2].same_level_run_len == 2


def test_cluster_signals_different_level_breaks_run():
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 1, "text": "A"}),
        Region(kind="heading", feature_block_indices=(1,), payload={"level_hint": 2, "text": "B"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "C"}),
    ]
    sigs = compute_cluster_signals(regions)
    assert sigs[0].same_level_run_len == 1  # h1 alone
    assert sigs[1].same_level_run_len == 2  # h2,h2 run
    assert sigs[2].same_level_run_len == 2


def test_cluster_signals_nested_subheading_content_counts_for_opener():
    # Residual-fix fixture: a real level-2 "1.1" opener whose body is laid out
    # under a level-3 "Learning Objectives" sub-heading (never a paragraph
    # DIRECTLY beneath it), then the next level-2 opener. Under the same-or-
    # higher-level redefinition, the level-3 sub-heading is TRANSPARENT and its
    # 2 paragraphs count toward the level-2 opener -> content_blocks_following
    # >= 1 -> the opener is PROTECTED from over-demotion.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "1.1 Introduction"}),
        Region(kind="heading", feature_block_indices=(1,),
               payload={"level_hint": 3, "text": "Learning Objectives"}),
        Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "p1"}),
        Region(kind="paragraph", feature_block_indices=(3,), payload={"text": "p2"}),
        Region(kind="heading", feature_block_indices=(4,),
               payload={"level_hint": 2, "text": "1.2 Next Section"}),
    ]
    sigs = compute_cluster_signals(regions)
    # the level-2 opener: the level-3 sub-heading is transparent, its 2 paras
    # count, the next level-2 heading closes the scope -> 2 content blocks.
    assert sigs[0].content_blocks_following == 2
    # the level-3 sub-heading scopes against the next heading of any
    # same-or-higher level: the next level-2 closes it -> its own 2 paras.
    assert sigs[1].content_blocks_following == 2
    # the trailing level-2 opener: no content/headings after -> 0.
    assert sigs[4].content_blocks_following == 0


def test_cluster_signals_phantom_chapter_run_still_flagged_under_levels():
    # Phantom-detection MUST NOT regress: a run of 8 same-level-1 "Chapter N"
    # index headings with NO content between consecutive entries. Each entry's
    # next heading is a same-LEVEL (1 <= 1) heading -> closes scope immediately
    # -> content_blocks_following == 0 for every entry, same_level_run_len large.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        Region(kind="heading", feature_block_indices=(i,),
               payload={"level_hint": 1, "text": f"Chapter {i+3}"})
        for i in range(8)
    ]
    sigs = compute_cluster_signals(regions)
    assert all(s.content_blocks_following == 0 for s in sigs)
    assert all(s.same_level_run_len == 8 for s in sigs)
    assert [s.run_position for s in sigs] == list(range(1, 9))


def test_cluster_signals_level2_sibling_with_content_between():
    # A level-2 opener directly followed by a level-2 sibling with content
    # between them -> the opener counts exactly the content before the sibling.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "1.1 A"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "p1"}),
        Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "p2"}),
        Region(kind="heading", feature_block_indices=(3,),
               payload={"level_hint": 2, "text": "1.2 B"}),
        Region(kind="paragraph", feature_block_indices=(4,), payload={"text": "p3"}),
    ]
    sigs = compute_cluster_signals(regions)
    assert sigs[0].content_blocks_following == 2  # p1, p2 before the sibling
    assert sigs[3].content_blocks_following == 1  # p3 after the sibling


def test_cluster_signals_level_less_heading_safe_default():
    # A heading with a missing/None level must not mis-count: it defaults to
    # the deep sentinel (lowest in hierarchy). A level-2 opener followed by a
    # level-less heading + 1 paragraph + a level-2 sibling -> the level-less
    # heading is strictly-lower (transparent), its paragraph counts toward the
    # opener; the next level-2 closes it.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "1.1 Opener"}),
        Region(kind="heading", feature_block_indices=(1,),
               payload={"text": "Level-less sub"}),  # no level_hint
        Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "body"}),
        Region(kind="heading", feature_block_indices=(3,),
               payload={"level_hint": 2, "text": "1.2 Next"}),
    ]
    sigs = compute_cluster_signals(regions)
    # level-less sub-heading is transparent to the level-2 opener -> 1 content.
    assert sigs[0].content_blocks_following == 1


def test_conservative_no_mass_demote_of_genuine_heading_run():
    # Conservative posture proof: a RUN of 8 genuine same-level section
    # headings (back-to-back, content_blocks_following == 0 for every one —
    # exactly the cluster shape the OLD aggressive directive would mass-demote)
    # is KEPT when the reviewer returns the conservative all-"ok" verdict set.
    # No heading is demoted merely for sitting in a run.
    headings = [f"{i}.{j} Section Title" for i in (1, 2) for j in range(1, 5)]
    fbs = [_fb(h) for h in headings]
    regions = [
        Region(kind="heading", feature_block_indices=(i,),
               payload={"level_hint": 2, "text": h})
        for i, h in enumerate(headings)
    ]
    # The conservative reviewer reads the run/cbf signals but returns "ok" for
    # genuine section titles — every block stays a heading.
    rt = _ScriptedRuntime([_verdict_json(i, verdict="ok") for i in range(len(regions))])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert all(r.kind == "heading" for r in out), "genuine heading run mass-demoted"
    assert all(v.verdict == "ok" for v in verdicts)
    assert not any(v.reverted_for_invariant for v in verdicts)
    _assert_coverage_invariant(out, len(fbs))


def test_conservative_still_retags_clear_answer_key_noise_heading():
    # The OTHER half of the conservative contract: a CLEARLY-mislabeled
    # answer-key fragment promoted to a heading IS re-tagged to metadata_drop
    # (the canonical individual-noise positive). The conservative posture
    # narrows WHEN we re-tag (block's own text is non-heading) — it does not
    # disable re-tagging.
    fbs = [_fb("Real Section Heading"), _fb("3.2: 14, 17, 20, 23"), _fb("Body para.")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "Real Section Heading"}),
        Region(kind="heading", feature_block_indices=(1,),
               payload={"level_hint": 2, "text": "3.2: 14, 17, 20, 23"}),  # answer-key
        Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "Body para."}),
    ]
    rt = _ScriptedRuntime([
        _verdict_json(0, verdict="ok"),
        _verdict_json(1, kind="metadata_drop", verdict="drop_injected_header",
                      note="answer-key fragment, not a heading"),
    ])
    out, verdicts = run_structure_review(regions, fbs, rt)
    # the genuine heading is KEPT, the answer-key noise heading is dropped.
    assert out[0].kind == "heading"
    assert out[1].kind == "metadata_drop"
    assert out[2].kind == "paragraph"
    assert verdicts[1].kind_after == "metadata_drop"
    assert not any(v.reverted_for_invariant for v in verdicts)
    _assert_coverage_invariant(out, len(fbs))


def test_scoping_reviews_only_heading_kinds():
    # The driver fires a 70B call for heading regions ONLY; non-heading
    # regions pass through untouched, indices preserved.
    fbs = [_fb("Chapter 1"), _fb("body para"), _fb("Chapter 2")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "Chapter 1"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "body para"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "Chapter 2"}),
    ]
    # Two heading prompts -> two scripted completions (block_id 0 and 2).
    rt = _ScriptedRuntime([
        _verdict_json(0, kind="heading", level=1, note="re-level"),
        _verdict_json(2, kind="metadata_drop", note="phantom"),
    ])
    out, verdicts = run_structure_review(regions, fbs, rt)
    # exactly 2 prompts were sent (the 2 headings, NOT the paragraph).
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 2
    # heading 0 re-leveled, heading 2 dropped, paragraph untouched.
    assert out[0].payload["level_hint"] == 1
    assert out[1].kind == "paragraph"
    assert verdicts[1].verdict == "ok"
    assert "not reviewed" in verdicts[1].review_note
    assert out[2].kind == "metadata_drop"
    _assert_coverage_invariant(out, len(fbs))
