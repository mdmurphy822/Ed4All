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

from dart_semantic.assembler.types import AssembledDoc
from dart_semantic.qwen_specialists.reviewer import (
    FlaggedBlock,
    ReviewVerdict,
    TokenConservationError,
    VerifierVerdict,
    _content_block_dispatch_gate,
    assert_token_conservation,
    resolve_second_pass_window_tokens,
    resolve_structure_review_mode,
    resolve_structure_review_temperature,
    run_second_pass_verify,
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

    Accepts ``fail_soft`` (the Stage-6 robustness kwarg the reviewer now
    passes) and an OPTIONAL ``fail_indices`` set — prompt positions whose
    completion is returned as the ``None`` SENTINEL (mirroring the endpoint
    runtime's fail_soft behaviour after the bounded retries) so a test can
    simulate a slow/down endpoint per cluster.
    """

    def __init__(
        self,
        completions: list[str],
        *,
        fail_indices: set[int] | None = None,
        queue: bool = False,
    ) -> None:
        self._completions = completions
        self._fail_indices = fail_indices or set()
        # Phase-4 verify dispatch sends a SINGLE-prompt batch per call and may
        # re-dispatch (spot-HTML re-ask) — the default whole-list-per-call mode
        # cannot model a 2-round re-ask. ``queue=True`` returns ONE crafted
        # completion per CALL (so a re-ask gets the NEXT scripted response);
        # ``fail_indices`` then keys on the CALL index. Default False keeps the
        # legacy whole-list-per-call behaviour byte-identical.
        self._queue = queue
        self._call = 0
        self.batch_calls: list[dict] = []

    def generate_batch(self, prompts, *, max_tokens, temperature=0.6,
                        top_p=0.95, seed=None, repeat_penalty=1.0,
                        fail_soft=False):
        self.batch_calls.append({
            "prompts": list(prompts),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "fail_soft": fail_soft,
        })
        if self._queue:
            i = self._call
            self._call += 1
            if i in self._fail_indices:
                return [None]
            raw = self._completions[i] if i < len(self._completions) else ""
            return [raw]
        # Return the scripted completions, overriding any ``fail_indices``
        # slot with the None sentinel (only honoured under fail_soft, exactly
        # as the real endpoint runtime does — but the reviewer always passes
        # fail_soft=True now, so we don't gate the test fixture on it).
        out: list = list(self._completions)
        for i in self._fail_indices:
            if 0 <= i < len(out):
                out[i] = None
        return out


def _verdict_json(block_id, *, verdict="corrected", kind=None, level=None,
                  doc_role=None, semantic_class=None, note="n") -> str:
    obj = {
        "block_id": block_id,
        "verdict": verdict,
        "corrected_kind": kind,
        "corrected_level": level,
        "corrected_doc_role": doc_role,
        "review_note": note,
    }
    if semantic_class is not None:
        obj["semantic_class"] = semantic_class
    return json.dumps(obj)


def _assert_coverage_invariant(regions, n_fb):
    """Every FB index appears in exactly one region's feature_block_indices."""
    seen = []
    for r in regions:
        seen.extend(r.feature_block_indices)
    assert sorted(seen) == list(range(n_fb)), "coverage invariant violated"
    assert len(seen) == len(set(seen)), "an FB appears in 2 regions"


# ---------------------------------------------------------------------------
# SEMANTIK_BLOCK_REVIEW flag-off byte-stability (behavioral).
#
# The durable form of the Phase-0 flag-off contract: with SEMANTIK_BLOCK_REVIEW
# off, run_structure_review's output is byte-identical whether the block-review
# envs are unset or explicitly off with arbitrary window/edge/cache values.
# This replaces the original grep-for-references guard in test_block_review_flags
# (which false-flagged the Phase-1 dead-but-callable edge-tokens consumer in
# reviewer_prompt.py — a reference reachable ONLY from dead code, so it cannot
# change flag-off behavior). Content-block re-types are not driver-reachable
# until Phase 3, so the heading-only path must be unchanged and every
# verdict.role_after (Phase 2) is None on the flag-off path.
# ---------------------------------------------------------------------------


def test_block_review_flag_off_byte_stable(monkeypatch):
    fbs = [_fb("Answer Key 3.1"), _fb("Real body content here.")]

    def _regions():
        return [
            Region(kind="heading", feature_block_indices=(0,),
                   payload={"level_hint": 2, "text": "Answer Key 3.1"}),
            Region(kind="paragraph", feature_block_indices=(1,),
                   payload={"text": "Real body content here."}),
        ]

    def _completions():
        return [
            _verdict_json(0, verdict="drop_injected_header",
                          kind="metadata_drop", note="answer-key"),
            _verdict_json(1, verdict="ok"),
        ]

    for k in ("SEMANTIK_BLOCK_REVIEW", "SEMANTIK_BLOCK_REVIEW_WINDOW",
              "SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", "SEMANTIK_BLOCK_REVIEW_CACHE"):
        monkeypatch.delenv(k, raising=False)
    out_unset, v_unset = run_structure_review(_regions(), fbs, _ScriptedRuntime(_completions()))

    # Flag explicitly off + arbitrary values on the other three envs: none of
    # them are consumed on the flag-off path, so output must be identical.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "0")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "3")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", "99")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", "0")
    out_off, v_off = run_structure_review(_regions(), fbs, _ScriptedRuntime(_completions()))

    # frozen dataclasses -> value equality.
    assert out_unset == out_off
    assert v_unset == v_off
    assert all(v.role_after is None for v in v_unset)


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
# Phase 2 — content-block re-type verdict + invariant-guarded apply.
#
# Content blocks are NOT driver-reachable until Phase 3 widens the heading-only
# review scope, so these exercise _apply_verdict DIRECTLY (mirroring the C1
# promotion tests above). SEMANTIK_BLOCK_REVIEW=1 is set where role_after is
# asserted (it is gated on the flag for flag-off audit byte-stability).
# ---------------------------------------------------------------------------


def test_retype_code_block_to_paragraph_applies(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("Try it: simplify 3(x+2).")]
    region = Region(kind="code_block", feature_block_indices=(0,),
                    payload={"text": "Try it: simplify 3(x+2)."})
    before_text = rv._joined_source_text(region, fbs)
    # The council mis-typed a "TRY IT" exercise as a code_block; re-type to
    # paragraph and assign a doc_role.
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": "example",
           "review_note": "TRY-IT exercise, not code"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.kind == "paragraph"                  # re-typed
    assert verdict.kind_before == "code_block"
    assert verdict.kind_after == "paragraph"
    assert verdict.verdict == "corrected"
    assert not verdict.reverted_for_invariant
    # verbatim source byte-identical (no text mutation through the re-type).
    assert rv._joined_source_text(out_region, fbs) == before_text
    assert out_region.payload["text"] == region.payload["text"]
    # role_after mirrors the already-written doc_role.
    assert out_region.payload["doc_role"] == "example"
    assert verdict.role_after == "example"


def test_retype_table_to_definition_list_applies(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("Term: a definition body")]
    # Passthrough table region (source_region_id set) — a kind-only re-type
    # must NOT move FBs and must NOT trip _admits.
    region = Region(kind="table", feature_block_indices=(0,),
                    payload={"text": "Term: a definition body"},
                    source_region_id=7)
    obj = {"block_id": 0, "verdict": "corrected",
           "corrected_kind": "definition_list", "corrected_level": None,
           "corrected_doc_role": None, "review_note": "key:value, not a grid"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.kind == "definition_list"
    # Passthrough FB partition + source_region_id unchanged.
    assert out_region.feature_block_indices == (0,)
    assert out_region.source_region_id == 7
    assert not verdict.reverted_for_invariant
    # _admits does NOT reject a kind-only passthrough re-type.
    assert rv._admits(region, out_region, fbs, is_promotion=False) is True


def test_retype_text_mutating_verdict_reverts(monkeypatch):
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("Original code")]
    region = Region(kind="code_block", feature_block_indices=(0,),
                    payload={"text": "Original code", "extra": "keep"})
    # A candidate that altered payload text is rejected outright by _admits.
    text_tampered = region.__class__(
        kind="paragraph", feature_block_indices=(0,),
        payload={"text": "REWRITTEN words", "extra": "keep"},
        source_region_id=region.source_region_id,
    )
    assert rv._admits(region, text_tampered, fbs, is_promotion=False) is False
    # And _apply_verdict reverts on ANY admission failure: original kept
    # verbatim, reverted_for_invariant=True, *_after == *_before.
    monkeypatch.setattr(rv, "_admits", lambda *a, **k: False)
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "review_note": "x"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region is region                            # original kept
    assert verdict.reverted_for_invariant is True
    assert verdict.kind_after == verdict.kind_before == "code_block"


def test_retype_unknown_kind_drops_to_ok(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("some code")]
    region = Region(kind="code_block", feature_block_indices=(0,),
                    payload={"text": "some code"})
    # 'exercise' is a deferred pedagogical kind, NOT in REGION_KINDS.
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "exercise",
           "corrected_level": None, "corrected_doc_role": None,
           "review_note": "pedagogical kind not in REGION_KINDS"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.kind == "code_block"                 # unknown kind dropped
    assert verdict.kind_after == "code_block"
    assert verdict.verdict == "ok"


def test_retype_dataclasses_replace_round_trips(monkeypatch):
    # A re-typed Region preserves feature_block_indices + source_region_id (and
    # every other non-mutated frozen field / non-structural payload key).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("grid or list body")]
    region = Region(kind="table", feature_block_indices=(0,),
                    payload={"text": "grid or list body", "cells": [[1]]},
                    source_region_id=9, aria_hints=("h",),
                    provenance={"p": "x"})
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "review_note": "n"}
    out_region, _ = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.kind == "paragraph"
    assert out_region.feature_block_indices == (0,)
    assert out_region.source_region_id == 9
    assert out_region.aria_hints == ("h",)
    assert out_region.provenance == {"p": "x"}
    assert out_region.payload["cells"] == [[1]]            # non-structural key kept
    assert region.kind == "table"                          # original unchanged


def test_audit_asdict_byte_stable_flag_off(monkeypatch):
    # The critic-flagged F1 regression: a bare new role_after field would make
    # dataclasses.asdict emit `role_after: null` on EVERY verdict, breaking the
    # flag-off byte-identical-audit contract. cascade.py excludes role_after
    # ONLY when None, SCOPED so it does NOT drop the legitimately-None level_*
    # keys that today's audit already emits for non-heading regions.
    import dataclasses as _dc
    from dart_semantic.qwen_specialists import reviewer as rv

    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)

    fbs = [_fb("Chapter 1"), _fb("body words")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 3, "text": "Chapter 1"}),
        Region(kind="paragraph", feature_block_indices=(1,),
               payload={"text": "body words"}),
    ]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1)])
    _, verdicts = rv.run_structure_review(regions, fbs, rt)

    # Mirror cascade.py's scoped None-exclusion (role_after + the Phase-5
    # block_review_window field + the Phase-4 semantic_class_after field, each
    # ONLY when None — never a blanket drop).
    def _row(v):
        row = _dc.asdict(v)
        if row.get("role_after") is None:
            row.pop("role_after", None)
        if row.get("block_review_window") is None:
            row.pop("block_review_window", None)
        if row.get("semantic_class_after") is None:
            row.pop("semantic_class_after", None)
        return row

    PRE_PHASE2_KEYS = {
        "block_id", "verdict", "kind_before", "kind_after",
        "level_before", "level_after", "review_note",
        "reverted_for_invariant", "reverted_for_endpoint_failure",
    }
    for v in verdicts:
        assert v.role_after is None                        # flag-off -> never set
        assert v.block_review_window is None               # flag-off -> never set
        row = _row(v)
        assert "role_after" not in row                     # excluded from audit
        assert "block_review_window" not in row            # excluded from audit
        assert set(row) == PRE_PHASE2_KEYS                 # byte-identical key set
    # The paragraph verdict's legitimately-None level_* keys SURVIVE the scoped
    # exclusion (a blanket None-drop would wrongly strip them -> the F1 trap).
    para_row = _row(verdicts[1])
    assert "level_before" in para_row and para_row["level_before"] is None
    assert "level_after" in para_row and para_row["level_after"] is None


# ---------------------------------------------------------------------------
# Phase 4 — ReviewVerdict.semantic_class_after (audit-only, None-drop, byte-stable)
# ---------------------------------------------------------------------------


def test_verdict_semantic_class_after_defaults_none():
    # Mirror the role_after flag-off default assertion: a freshly-constructed
    # verdict carries semantic_class_after=None (Optional-default-None posture,
    # populated only when SEMANTIK_SEMANTIC_CLASS is on — no population yet).
    v = ReviewVerdict(
        block_id=0,
        verdict="ok",
        kind_before="paragraph",
        kind_after="paragraph",
        level_before=None,
        level_after=None,
        review_note="",
    )
    assert v.semantic_class_after is None
    # the sibling audit-only fields keep their None default too.
    assert v.role_after is None
    assert v.block_review_window is None


def test_audit_row_drops_none_semantic_class_after(monkeypatch):
    # The new audit-only field must be EXCLUDED from the audit row when None
    # (SCOPED drop, exactly like role_after / block_review_window) so a heading-
    # only / flag-off run's audit dict stays byte-identical — while the
    # legitimately-None level_before / level_after keys SURVIVE (a blanket
    # None-drop would wrongly strip them: the F1 trap).
    import dataclasses as _dc
    from dart_semantic.qwen_specialists import reviewer as rv

    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)

    fbs = [_fb("Chapter 1"), _fb("body words")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 3, "text": "Chapter 1"}),
        Region(kind="paragraph", feature_block_indices=(1,),
               payload={"text": "body words"}),
    ]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1)])
    _, verdicts = rv.run_structure_review(regions, fbs, rt)

    # Mirror cascade.py::_verdict_audit_row's THREE scoped None-exclusions
    # (role_after + block_review_window + the Phase-4 semantic_class_after),
    # each ONLY when None — never a blanket drop.
    def _row(v):
        row = _dc.asdict(v)
        if row.get("role_after") is None:
            row.pop("role_after", None)
        if row.get("block_review_window") is None:
            row.pop("block_review_window", None)
        if row.get("semantic_class_after") is None:
            row.pop("semantic_class_after", None)
        return row

    PRE_PHASE2_KEYS = {
        "block_id", "verdict", "kind_before", "kind_after",
        "level_before", "level_after", "review_note",
        "reverted_for_invariant", "reverted_for_endpoint_failure",
    }
    for v in verdicts:
        assert v.semantic_class_after is None              # flag-off -> never set
        raw = _dc.asdict(v)
        assert "semantic_class_after" in raw               # bare asdict emits it
        assert raw["semantic_class_after"] is None         # ...as null
        row = _row(v)
        assert "semantic_class_after" not in row           # excluded from audit
        assert set(row) == PRE_PHASE2_KEYS                 # byte-identical key set
    # The paragraph verdict's legitimately-None level_* keys SURVIVE the scoped
    # exclusion (the new drop is SCOPED to semantic_class_after only).
    para_row = _row(verdicts[1])
    assert "level_before" in para_row and para_row["level_before"] is None
    assert "level_after" in para_row and para_row["level_after"] is None


def test_snapshot_byte_stable_with_new_verdict_field():
    # The cache-identity helper (_snapshot in test_block_review_cache.py) keys
    # off dataclasses.asdict(verdict). A bare new field is emitted on BOTH the
    # cache-on and cache-off sides, so the new field cannot break _snapshot
    # equality (cache-on == cache-off). Sanity-check that two equal verdicts
    # asdict-equal WITH the new field present-but-None on both sides.
    import dataclasses as _dc

    def _mk():
        return ReviewVerdict(
            block_id=3,
            verdict="corrected",
            kind_before="code_block",
            kind_after="paragraph",
            level_before=None,
            level_after=None,
            review_note="re-typed",
        )

    a, b = _dc.asdict(_mk()), _dc.asdict(_mk())
    assert a == b                                          # identity holds
    assert a["semantic_class_after"] is None and b["semantic_class_after"] is None


def test_provenance_byte_stable_no_region_field_added():
    # No Region dataclass field was added (storage is payload-only), so
    # _build_region_provenance's hand-built key set is unchanged: no
    # semantic_class / semantic_class_after key leaks into the provenance dict.
    from dart_semantic.cascade import _build_region_provenance

    fbs = [_fb("body words")]
    regions = [
        Region(kind="paragraph", feature_block_indices=(0,),
               payload={"text": "body words"}),
    ]
    prov = _build_region_provenance(
        region_order=[0],
        regions=regions,
        feature_blocks=fbs,
        stage7_results={},
        review_verdicts=None,
    )
    assert len(prov) == 1
    assert "semantic_class" not in prov[0]
    assert "semantic_class_after" not in prov[0]


# ---------------------------------------------------------------------------
# Phase 6 — apply: parse/validate/stamp semantic_class, _admits allowlist,
# EXAMPLE->heading promotion guard.
# ---------------------------------------------------------------------------


def test_semantic_class_stamped_on_payload(monkeypatch):
    # Flag on: a scripted verdict carrying semantic_class='worked_example' on a
    # re-typed content block stamps payload['semantic_class'] + mirrors the
    # audit field, with the verbatim source byte-identical (payload-only — no
    # text mutation).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("Try it: simplify 3(x+2).")]
    region = Region(kind="code_block", feature_block_indices=(0,),
                    payload={"text": "Try it: simplify 3(x+2)."})
    before_text = rv._joined_source_text(region, fbs)
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "semantic_class": "worked_example",
           "review_note": "TRY-IT worked instance"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.payload["semantic_class"] == "worked_example"
    assert verdict.semantic_class_after == "worked_example"
    assert not verdict.reverted_for_invariant
    # verbatim source byte-identical through the stamp.
    assert rv._joined_source_text(out_region, fbs) == before_text
    assert out_region.payload["text"] == region.payload["text"]


def test_unknown_semantic_class_drops_to_none(monkeypatch):
    # An off-catalog token degrades to None (anti-fabrication) and is never
    # stamped, even with the flag on.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("some prose body")]
    region = Region(kind="code_block", feature_block_indices=(0,),
                    payload={"text": "some prose body"})
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "semantic_class": "made_up", "review_note": "bad class"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert "semantic_class" not in out_region.payload     # never stamped
    assert verdict.semantic_class_after is None


def test_admits_allowlist_permits_semantic_class(monkeypatch):
    # MAKE-OR-BREAK #1: a semantic_class-only payload change is ADMITTED (not
    # reverted_for_invariant). The sub-test monkeypatches the allowlist down to
    # the legacy 3-key set and asserts the SAME change then REVERTS — pinning
    # that 'semantic_class' membership is exactly what lets the stamp land.
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("worked example body")]
    region = Region(kind="paragraph", feature_block_indices=(0,),
                    payload={"text": "worked example body"})
    candidate = region.__class__(
        kind="paragraph", feature_block_indices=(0,),
        payload={"text": "worked example body", "semantic_class": "worked_example"},
        source_region_id=region.source_region_id,
    )
    # With 'semantic_class' on the allowlist (the shipped state) -> admitted.
    assert "semantic_class" in rv._ADMIT_STRUCTURAL_PAYLOAD_KEYS
    assert rv._admits(region, candidate, fbs, is_promotion=False) is True

    # Remove it from the allowlist -> the SAME payload-only change is rejected
    # as a non-structural mutation (the silent reverted_for_invariant no-op the
    # allowlist line prevents).
    monkeypatch.setattr(
        rv, "_ADMIT_STRUCTURAL_PAYLOAD_KEYS",
        frozenset({"level_hint", "doc_role", "text"}),
    )
    assert rv._admits(region, candidate, fbs, is_promotion=False) is False


def test_example_label_not_promoted_to_heading(monkeypatch):
    # THE indexing-bug regression test. A content block whose text begins
    # "EXAMPLE 1.3" + a verdict corrected_kind='heading' (flag on) -> the
    # ->heading promotion is REFUSED: kind stays content, semantic_class is
    # stamped 'worked_example', and the verdict reflects no promotion.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("EXAMPLE 1.3 Simplify the expression 3(x + 2) - 5.")]
    region = Region(kind="paragraph", feature_block_indices=(0,),
                    payload={"text": "EXAMPLE 1.3 Simplify the expression 3(x + 2) - 5."})
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "heading",
           "corrected_level": None, "corrected_doc_role": None,
           "review_note": "model wanted to promote to heading"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.kind == "paragraph"                 # promotion refused
    assert verdict.kind_after == "paragraph"
    assert verdict.kind_before == "paragraph"
    assert out_region.payload["semantic_class"] == "worked_example"
    assert verdict.semantic_class_after == "worked_example"
    # No promotion happened -> no minted heading level, no text rewrite.
    assert out_region.payload.get("level_hint") is None
    assert out_region.payload["text"] == region.payload["text"]


def test_solution_step_promotion_division(monkeypatch):
    # Division of responsibility: the reviewer guard's prefix set is
    # {TRY, EXAMPLE, EXERCISE} and does NOT cover Solution / Step N — those
    # label-HEADINGS are demoted UPSTREAM by the always-on deterministic
    # structure-clean pass (deterministic_structure._retag). So a "Solution:"
    # paragraph is NOT caught by _has_pedagogical_label_prefix and the guard
    # does not fire on it.
    from dart_semantic.qwen_specialists import reviewer as rv

    assert "SOLUTION" not in rv._PEDAGOGICAL_LABEL_PREFIXES
    assert "STEP" not in rv._PEDAGOGICAL_LABEL_PREFIXES
    assert rv._has_pedagogical_label_prefix("Solution: combine like terms.") is False
    assert rv._has_pedagogical_label_prefix("Step 3 add 5 to both sides.") is False
    assert rv._pedagogical_label_component("Solution: x = 4") is None
    # Confirm the always-on clean pass owns these label classes.
    from dart_semantic.qwen_specialists import deterministic_structure as ds
    label_classes = {css.lower() for _pat, css in ds._PEDAGOGICAL_LABEL_CLASSES}
    assert any("solution" in c for c in label_classes)
    assert any("step" in c for c in label_classes)


def test_guard_byte_stable_when_semantic_class_off(monkeypatch):
    # SEMANTIK_BLOCK_REVIEW on + SEMANTIK_SEMANTIC_CLASS OFF: the P1 promotion
    # behavior is unchanged — an EXAMPLE paragraph that P1 would promote to a
    # heading STILL promotes (the guard does not fire; the fix ships WITH the
    # feature, not as a silent P1 change).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.delenv("SEMANTIK_SEMANTIC_CLASS", raising=False)
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("EXAMPLE 1.3 Simplify the expression.")]
    region = Region(kind="paragraph", feature_block_indices=(0,),
                    payload={"text": "EXAMPLE 1.3 Simplify the expression."})
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "heading",
           "corrected_level": None, "corrected_doc_role": None,
           "semantic_class": "worked_example",
           "review_note": "promote"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.kind == "heading"                   # P1 promotion preserved
    assert verdict.kind_after == "heading"
    # flag off -> no semantic_class stamped, audit field None.
    assert "semantic_class" not in out_region.payload
    assert verdict.semantic_class_after is None


# ---------------------------------------------------------------------------
# Change 1 — the deterministic FLOOR: when the model leaves semantic_class
# empty, derive it from the clean-pass pedagogy css_class (precedence (a)) or
# the pedagogical-label prefix (precedence (b)). Model > pedagogy-class > prefix.
# ---------------------------------------------------------------------------


def test_floor_assigns_from_pedagogy_class(monkeypatch):
    # A region whose payload carries a clean-pass-stamped pedagogy css_class
    # (deterministic_structure._retag) + a model verdict with NO semantic_class
    # (flag on) -> the floor maps the css_class to its gold component.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("EXAMPLE 1.1 Simplify 3(x + 2).")]
    region = Region(
        kind="code_block", feature_block_indices=(0,),
        payload={"text": "EXAMPLE 1.1 Simplify 3(x + 2).",
                 "css_class": "pedagogy-example"},
    )
    # Model omits semantic_class entirely.
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "review_note": "no class from model"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.payload["semantic_class"] == "worked_example"
    assert verdict.semantic_class_after == "worked_example"
    assert not verdict.reverted_for_invariant
    # source byte-identical through the floored stamp (payload-only).
    assert out_region.payload["text"] == region.payload["text"]
    assert out_region.payload["css_class"] == "pedagogy-example"


def test_floor_assigns_from_prefix_when_no_pedagogy_class(monkeypatch):
    # No pedagogy css_class, but the verbatim source begins with a pedagogical
    # label (TRY IT) + no model semantic_class -> precedence (b) prefix floor.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("TRY IT 1.5 simplify the expression 4(y - 1).")]
    region = Region(
        kind="code_block", feature_block_indices=(0,),
        payload={"text": "TRY IT 1.5 simplify the expression 4(y - 1)."},
    )
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "review_note": "no class, no css_class"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.payload["semantic_class"] == "worked_example"
    assert verdict.semantic_class_after == "worked_example"


def test_floor_does_not_override_model_semantic_class(monkeypatch):
    # Precedence model > pedagogy-class: a VALID model semantic_class is kept
    # even when the css_class would floor to a DIFFERENT component.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("A term and its meaning.")]
    region = Region(
        kind="code_block", feature_block_indices=(0,),
        payload={"text": "A term and its meaning.",
                 "css_class": "pedagogy-example"},  # would floor to worked_example
    )
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "semantic_class": "definition_region",  # model wins
           "review_note": "model supplied a valid class"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert out_region.payload["semantic_class"] == "definition_region"
    assert verdict.semantic_class_after == "definition_region"


def test_floor_only_assigns_valid_catalog_component(monkeypatch):
    # Anti-fabrication: a pedagogy class with NO catalog reconciles_with (and no
    # label prefix) floors to None -> NO semantic_class is stamped.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("Just some ordinary prose body.")]
    region = Region(
        kind="code_block", feature_block_indices=(0,),
        payload={"text": "Just some ordinary prose body.",
                 "css_class": "pedagogy-solution"},  # no reconciles_with -> None
    )
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "review_note": "unmapped pedagogy class, no prefix"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert "semantic_class" not in out_region.payload
    assert verdict.semantic_class_after is None


def test_floor_byte_stable_when_off(monkeypatch):
    # SEMANTIK_SEMANTIC_CLASS OFF: the floor does NOT run, even with a pedagogy
    # css_class + no model semantic_class -> no class stamped, audit field None.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.delenv("SEMANTIK_SEMANTIC_CLASS", raising=False)
    from dart_semantic.qwen_specialists import reviewer as rv

    fbs = [_fb("EXAMPLE 1.1 Simplify 3(x + 2).")]
    region = Region(
        kind="code_block", feature_block_indices=(0,),
        payload={"text": "EXAMPLE 1.1 Simplify 3(x + 2).",
                 "css_class": "pedagogy-example"},
    )
    obj = {"block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
           "corrected_level": None, "corrected_doc_role": None,
           "review_note": "flag off -> no floor"}
    out_region, verdict = rv._apply_verdict(region, 0, obj, fbs)
    assert "semantic_class" not in out_region.payload
    assert verdict.semantic_class_after is None


def test_class_only_run_html_byte_identical(monkeypatch):
    # Decoupled-flag invariant: SEMANTIK_SEMANTIC_CLASS on + SEMANTIK_GOLD_SHELL
    # absent -> the assembler does NOT read payload['semantic_class'], so a
    # region carrying the class produces HTML byte-identical to the both-off
    # baseline (the assembler change is Phase 7).
    from dart_semantic.assembler.api import AssemblerConfig, assemble_document

    def _para(text, idx, *, semantic_class=None):
        payload = {"text": text}
        if semantic_class is not None:
            payload["semantic_class"] = semantic_class
        return Region(kind="paragraph", feature_block_indices=(idx,),
                      payload=payload, source_region_id=idx)

    def _assemble(regions):
        fbs = [_fb(r.payload["text"]) for r in regions]
        top = {i: None for i in range(len(regions))}
        return assemble_document(top, regions, fbs, runtime_mode="mock",
                                 config=AssemblerConfig(skip_gap_fill=True))

    monkeypatch.delenv("SEMANTIK_GOLD_SHELL", raising=False)
    baseline = _assemble([_para("Worked example body.", 0),
                          _para("Second paragraph.", 1)])

    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    classed = _assemble([_para("Worked example body.", 0, semantic_class="worked_example"),
                         _para("Second paragraph.", 1)])

    # HTML byte-identical (the assembler does not read payload['semantic_class']
    # with the shell flag off). The full AssembledDoc is NOT compared: ITEM4's
    # additive, metadata-only ``sub_task_log['containment']`` stash legitimately
    # reflects the unit structure (the classed run's worked_example anchor forms
    # a unit edge), which is exactly the derived-forest metadata — it never
    # touches render bytes. Assert the documented invariant: the HTML.
    assert classed.html == baseline.html                  # shell flag off -> identical
    assert classed.region_provenance == baseline.region_provenance
    assert classed.gaps_found == baseline.gaps_found


def test_audit_byte_stable_flag_off(monkeypatch):
    # The Phase-4 guard still holds at Phase 6: a heading-only / flag-off run's
    # audit dict is byte-identical with the SCOPED semantic_class_after None-drop
    # (None-key excluded; legitimately-None level_* keys survive).
    import dataclasses as _dc
    from dart_semantic.qwen_specialists import reviewer as rv

    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    monkeypatch.delenv("SEMANTIK_SEMANTIC_CLASS", raising=False)

    fbs = [_fb("Chapter 1"), _fb("body words")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 3, "text": "Chapter 1"}),
        Region(kind="paragraph", feature_block_indices=(1,),
               payload={"text": "body words"}),
    ]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1)])
    _, verdicts = rv.run_structure_review(regions, fbs, rt)

    def _row(v):
        row = _dc.asdict(v)
        for k in ("role_after", "block_review_window", "semantic_class_after"):
            if row.get(k) is None:
                row.pop(k, None)
        return row

    PRE_PHASE2_KEYS = {
        "block_id", "verdict", "kind_before", "kind_after",
        "level_before", "level_after", "review_note",
        "reverted_for_invariant", "reverted_for_endpoint_failure",
    }
    for v in verdicts:
        assert v.semantic_class_after is None
        row = _row(v)
        assert "semantic_class_after" not in row
        assert set(row) == PRE_PHASE2_KEYS
    para_row = _row(verdicts[1])
    assert "level_before" in para_row and para_row["level_before"] is None


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


# ---------------------------------------------------------------------------
# Endpoint-failure resilience (Stage-6 robustness parity) — per-cluster
# degrade, whole-document degrade, healthy-endpoint unchanged.
# ---------------------------------------------------------------------------


def test_reviewer_passes_fail_soft_to_generate_batch():
    # The driver must call generate_batch with fail_soft=True so a per-cluster
    # endpoint failure returns the None sentinel instead of raising.
    fbs = [_fb("Chapter 1")]
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 2, "text": "Chapter 1"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1)])
    run_structure_review(regions, fbs, rt)
    assert rt.batch_calls[0]["fail_soft"] is True


# ---------------------------------------------------------------------------
# Deterministic decoding — SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE (default 0.0).
# ---------------------------------------------------------------------------


def test_review_temperature_default_greedy(monkeypatch):
    # Default (env unset) -> greedy decoding: the reviewer must pass
    # temperature=0.0 to generate_batch so the structure-correction pass is
    # deterministic, not sampled at the endpoint's 0.6 default.
    monkeypatch.delenv("SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE", raising=False)
    assert resolve_structure_review_temperature() == 0.0
    fbs = [_fb("Chapter 1")]
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 2, "text": "Chapter 1"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1)])
    run_structure_review(regions, fbs, rt)
    assert rt.batch_calls[0]["temperature"] == 0.0


def test_review_temperature_env_threads_through(monkeypatch):
    # An operator opt-in to sampling threads the resolved value through.
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE", "0.7")
    assert resolve_structure_review_temperature() == 0.7
    fbs = [_fb("Chapter 1")]
    regions = [Region(kind="heading", feature_block_indices=(0,),
                      payload={"level_hint": 2, "text": "Chapter 1"})]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1)])
    run_structure_review(regions, fbs, rt)
    assert rt.batch_calls[0]["temperature"] == 0.7


def test_review_temperature_parse_with_fallback(monkeypatch):
    # Garbage / negative / NaN -> default 0.0 (parse-with-fallback).
    for bad in ("garbage", "-1.0", "nan", ""):
        monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_TEMPERATURE", bad)
        assert resolve_structure_review_temperature() == 0.0


def test_one_cluster_endpoint_failure_degrades_only_that_cluster():
    # (a) One heading's endpoint call fails (None sentinel) -> that cluster
    # reverts to UNREVIEWED (reverted_for_endpoint_failure), the other heading
    # is reviewed normally, the paragraph is untouched, no raise.
    fbs = [_fb("Chapter 1"), _fb("body para"), _fb("Chapter 2")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "Chapter 1"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "body para"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "Chapter 2"}),
    ]
    # Two heading prompts (positions 0 and 1 in the dense prompt list map to
    # region indices 0 and 2). Fail the SECOND prompt (region index 2).
    completions = [
        _verdict_json(0, kind="heading", level=1, note="re-level"),
        _verdict_json(2, kind="metadata_drop", note="would-drop"),
    ]
    rt = _ScriptedRuntime(completions, fail_indices={1})
    out, verdicts = run_structure_review(regions, fbs, rt)

    # heading 0 reviewed normally (re-leveled).
    assert out[0].kind == "heading"
    assert out[0].payload["level_hint"] == 1
    assert verdicts[0].verdict == "corrected"
    assert verdicts[0].reverted_for_endpoint_failure is False

    # paragraph untouched.
    assert out[1].kind == "paragraph"
    assert verdicts[1].verdict == "ok"
    assert verdicts[1].reverted_for_endpoint_failure is False

    # heading 2's endpoint failed -> kept ORIGINAL heading verbatim (NOT
    # dropped), stamped reverted_for_endpoint_failure.
    assert out[2].kind == "heading"
    assert out[2].payload["level_hint"] == 2
    assert verdicts[2].verdict == "ok"
    assert verdicts[2].reverted_for_endpoint_failure is True
    assert verdicts[2].reverted_for_invariant is False
    _assert_coverage_invariant(out, len(fbs))


def test_all_clusters_endpoint_down_degrades_whole_review_no_raise():
    # (b) Every heading's endpoint call fails -> the WHOLE review degrades to
    # unreviewed (anti-crawl short-circuit). Document NOT aborted, no raise,
    # original regions returned verbatim.
    fbs = [_fb("Chapter 1"), _fb("body para"), _fb("Chapter 2"), _fb("Chapter 3")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 2, "text": "Chapter 1"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "body para"}),
        Region(kind="heading", feature_block_indices=(2,), payload={"level_hint": 2, "text": "Chapter 2"}),
        Region(kind="heading", feature_block_indices=(3,), payload={"level_hint": 2, "text": "Chapter 3"}),
    ]
    # 3 heading prompts; fail ALL of them.
    completions = [
        _verdict_json(0, kind="metadata_drop"),
        _verdict_json(2, kind="metadata_drop"),
        _verdict_json(3, kind="metadata_drop"),
    ]
    rt = _ScriptedRuntime(completions, fail_indices={0, 1, 2})
    out, verdicts = run_structure_review(regions, fbs, rt)

    # Whole review degraded -> the original regions list is returned verbatim
    # (byte-stable UNREVIEWED floor); no heading was dropped.
    assert out is regions
    assert [r.kind for r in out] == ["heading", "paragraph", "heading", "heading"]
    # every heading verdict is endpoint-degraded; the paragraph stays ok.
    assert verdicts[0].reverted_for_endpoint_failure is True
    assert verdicts[1].reverted_for_endpoint_failure is False  # paragraph
    assert verdicts[1].verdict == "ok"
    assert verdicts[2].reverted_for_endpoint_failure is True
    assert verdicts[3].reverted_for_endpoint_failure is True
    # No correction leaked through despite the would-drop completions.
    assert not any(v.kind_after == "metadata_drop" for v in verdicts)
    _assert_coverage_invariant(out, len(fbs))


def test_healthy_endpoint_review_unchanged():
    # (c) Healthy endpoint -> review behaves exactly as before this fix: no
    # endpoint-failure reverts, corrections applied normally.
    fbs = [_fb("Chapter 1"), _fb("Chapter 2")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,), payload={"level_hint": 3, "text": "Chapter 1"}),
        Region(kind="heading", feature_block_indices=(1,), payload={"level_hint": 2, "text": "Chapter 2"}),
    ]
    completions = [
        _verdict_json(0, kind="heading", level=1, note="top-level"),
        _verdict_json(1, verdict="ok"),
    ]
    rt = _ScriptedRuntime(completions)  # no fail_indices -> healthy
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].payload["level_hint"] == 1  # corrected
    assert verdicts[0].verdict == "corrected"
    assert not any(v.reverted_for_endpoint_failure for v in verdicts)
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


# ---------------------------------------------------------------------------
# Phase 3 — review-scope predicate + deterministic-first gate + ambiguity
# escalation. Content blocks join the reviewer ONLY when SEMANTIK_BLOCK_REVIEW
# is on AND the confidence gate fires; cluster signals stay byte-identical.
# ---------------------------------------------------------------------------


def _council_state(conf_by_fb):
    """Minimal CouncilState exposing structure.structural_role top-1 per FB idx.

    ``conf_by_fb`` maps a FeatureBlock index -> (label, confidence) so a test
    can drive the council top-1 the gate reads via _get_signal/_top1.
    """
    from types import SimpleNamespace

    signals = [
        SimpleNamespace(
            head_name="structural_role",
            region_id=fb_idx,
            top_k_labels=[label],
            top_k_confidences=[conf],
        )
        for fb_idx, (label, conf) in conf_by_fb.items()
    ]
    return SimpleNamespace(outputs={"structure": SimpleNamespace(signals=signals)})


def test_content_block_in_scope_when_flag_on(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    fbs = [_fb("x = compute(y)")]
    regions = [Region(kind="code_block", feature_block_indices=(0,),
                      payload={"text": "x = compute(y)"})]
    state = _council_state({0: ("code_block", 0.30)})  # below the 0.4 floor
    rt = _ScriptedRuntime([_verdict_json(0, verdict="ok")])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    # the low-confidence code_block WAS dispatched (one prompt, one call).
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 1
    assert verdicts[0].review_note != "non-heading; not reviewed"


def test_content_block_skipped_when_flag_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    fbs = [_fb("x = compute(y)")]
    regions = [Region(kind="code_block", feature_block_indices=(0,),
                      payload={"text": "x = compute(y)"})]
    # A council confidence that WOULD dispatch the block if the flag were on.
    state = _council_state({0: ("code_block", 0.10)})
    rt = _ScriptedRuntime([_verdict_json(0, verdict="ok")])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    # heading-only: no heading present, content block NOT in scope -> no batch.
    assert rt.batch_calls == []
    assert out == regions
    assert verdicts[0].review_note == "non-heading; not reviewed"


def test_cluster_signals_unaffected_by_scope_widen(monkeypatch):
    # compute_cluster_signals reads the UNCHANGED HEADING_KINDS (never the
    # scope predicate / flag), so its output is byte-identical flag-on vs off.
    from dart_semantic.qwen_specialists.reviewer import compute_cluster_signals

    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "Section A"}),
        Region(kind="code_block", feature_block_indices=(1,),
               payload={"text": "def f(): return 1"}),
        Region(kind="paragraph", feature_block_indices=(2,),
               payload={"text": "Body content here."}),
        Region(kind="heading", feature_block_indices=(3,),
               payload={"level_hint": 2, "text": "Section B"}),
    ]
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    sig_off = compute_cluster_signals(regions)
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    sig_on = compute_cluster_signals(regions)
    assert sig_off == sig_on


def test_high_confidence_block_skips_llm(monkeypatch):
    # Deterministic-first for the gated kinds: a high-confidence PARAGRAPH (not
    # a known-weak code_block/table) with no pedagogical prefix gets NO prompt.
    # (code_block/table now dispatch unconditionally — see
    # test_code_block_dispatched_regardless_of_confidence.)
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    fbs = [_fb("Ordinary body sentence with no exercise prefix.")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "Ordinary body sentence with no exercise prefix."})]
    state = _council_state({0: ("paragraph", 0.95)})  # well above the floor
    rt = _ScriptedRuntime([_verdict_json(0, verdict="ok")])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    # deterministic-first: a high-confidence paragraph gets NO prompt.
    assert rt.batch_calls == []
    assert out == regions
    assert verdicts[0].review_note == "non-heading; not reviewed"


def test_code_block_dispatched_regardless_of_confidence(monkeypatch):
    # A high-council-confidence code_block is STILL dispatched — the council
    # mis-types content here systematically at high confidence, so the conf
    # floor is bypassed for the known-weak kinds.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    fbs = [_fb("TRY IT : : 1.27 3 7")]
    regions = [Region(kind="code_block", feature_block_indices=(0,),
                      payload={"text": "TRY IT : : 1.27 3 7"})]
    state = _council_state({0: ("code_block", 0.95)})  # well above the floor
    assert _content_block_dispatch_gate(regions[0], state, fbs) is True
    rt = _ScriptedRuntime([_verdict_json(0, verdict="ok")])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    # the high-confidence code_block WAS dispatched (one prompt).
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 1


def test_table_dispatched_unconditionally(monkeypatch):
    # A table region is dispatched even at high council confidence and with no
    # council_state threaded (tables were NEVER dispatched before the recalibration).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    fbs = [_fb("Commutative Property: a + b = b + a")]
    regions = [Region(kind="table", feature_block_indices=(0,),
                      payload={"text": "Commutative Property: a + b = b + a"})]
    # gate True with no council_state at all (unconditional, conf-independent).
    assert _content_block_dispatch_gate(regions[0], None, fbs) is True
    # and True with a high-confidence council read.
    state = _council_state({0: ("table", 0.99)})
    assert _content_block_dispatch_gate(regions[0], state, fbs) is True
    rt = _ScriptedRuntime([_verdict_json(0, verdict="ok")])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 1


def test_paragraph_not_dispatched_without_prefix_or_lowconf(monkeypatch):
    # A plain, high-confidence paragraph with no pedagogical prefix is NOT
    # dispatched — we do not review every prose block in the document.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    fbs = [_fb("This is an ordinary paragraph of running prose.")]
    regions = [Region(kind="paragraph", feature_block_indices=(0,),
                      payload={"text": "This is an ordinary paragraph of running prose."})]
    state = _council_state({0: ("paragraph", 0.95)})  # above the floor
    assert _content_block_dispatch_gate(regions[0], state, fbs) is False
    rt = _ScriptedRuntime([])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert rt.batch_calls == []
    assert verdicts[0].review_note == "non-heading; not reviewed"


def test_code_block_retyped_to_paragraph_text_byte_identical(monkeypatch):
    # End-to-end: a window op returning corrected_kind=paragraph for a
    # code_block member is APPLIED (code_block -> paragraph) and the verbatim
    # source text is byte-identical through the re-type (no text mutation).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "8")
    src = "TRY IT : : 1.27 simplify the expression three plus seven"
    fbs = [_fb(src), _fb("second exercise body text here")]
    regions = [
        Region(kind="code_block", feature_block_indices=(0,), payload={"text": src}),
        Region(kind="code_block", feature_block_indices=(1,),
               payload={"text": "second exercise body text here"}),
    ]
    state = _council_state({0: ("code_block", 0.95), 1: ("code_block", 0.95)})
    rt = _ScriptedRuntime([_oplist([0, 1], kind="paragraph")])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert out[0].kind == "paragraph"
    assert out[1].kind == "paragraph"
    assert verdicts[0].kind_after == "paragraph"
    # verbatim source survived byte-identical (the reviewer never rewrites text).
    assert fbs[0].raw.text == src
    _assert_coverage_invariant(out, len(fbs))


def test_flag_off_no_dispatch(monkeypatch):
    # Flag off -> no content block is in scope, so even a known-weak code_block
    # / table gets NO prompt (byte-stable heading-only path).
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    fbs = [_fb("TRY IT : : 1.27 3 7"), _fb("a b c")]
    regions = [
        Region(kind="code_block", feature_block_indices=(0,),
               payload={"text": "TRY IT : : 1.27 3 7"}),
        Region(kind="table", feature_block_indices=(1,),
               payload={"text": "a b c"}),
    ]
    rt = _ScriptedRuntime([])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert rt.batch_calls == []
    assert out == regions
    assert all(v.review_note == "non-heading; not reviewed" for v in verdicts)


def test_pedagogical_label_block_dispatched_without_council(monkeypatch):
    # The pedagogical-label OR-arm fires WITHOUT council confidence: a "TRY IT"
    # code_block is dispatched even when no council_state is threaded.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    fbs = [_fb("TRY IT 2.4 simplify the expression")]
    regions = [Region(kind="code_block", feature_block_indices=(0,),
                      payload={"text": "TRY IT 2.4 simplify the expression"})]
    rt = _ScriptedRuntime([_verdict_json(0, verdict="ok")])
    out, verdicts = run_structure_review(regions, fbs, rt)  # no council_state
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 1


def test_ambiguous_block_escalates_to_fuller_text(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_EDGE_TOKENS", "3")  # force truncation
    long_text = " ".join(f"tok{i}" for i in range(40))
    fbs = [_fb(long_text)]
    regions = [Region(kind="code_block", feature_block_indices=(0,),
                      payload={"text": long_text})]
    state = _council_state({0: ("code_block", 0.30)})
    amb = json.dumps({
        "block_id": 0, "verdict": "corrected", "corrected_kind": "paragraph",
        "corrected_level": None, "corrected_doc_role": None,
        "review_note": "edge insufficient", "ambiguous": True,
    })
    rt = _ScriptedRuntime([amb])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    # exactly TWO calls: the first window + the SINGLE fuller-text re-ask.
    assert len(rt.batch_calls) == 2
    assert len(rt.batch_calls[1]["prompts"]) == 1     # one re-ask, for the idx
    first_prompt = rt.batch_calls[0]["prompts"][0]
    esc_prompt = rt.batch_calls[1]["prompts"][0]
    # the re-ask carries the FULL verbatim source (larger window); the first
    # prompt was head/tail-truncated (the full string is NOT present).
    assert long_text in esc_prompt
    assert long_text not in first_prompt
    assert len(esc_prompt) > len(first_prompt)
    # the escalated result is applied through _apply_verdict / _admits.
    assert out[0].kind == "paragraph"
    assert verdicts[0].kind_after == "paragraph"
    _assert_coverage_invariant(out, len(fbs))


def test_flag_off_byte_stable_with_content_blocks(monkeypatch):
    # A doc with a heading + code_block + paragraph: flag-off output is
    # byte-identical whether the flag is unset or explicitly off, the content
    # blocks pass through untouched, and only the heading gets a prompt — even
    # though a council low-confidence would dispatch the code_block IF on.
    fbs = [_fb("Chapter 1"), _fb("def f(): pass"), _fb("Body text.")]

    def _regions():
        return [
            Region(kind="heading", feature_block_indices=(0,),
                   payload={"level_hint": 2, "text": "Chapter 1"}),
            Region(kind="code_block", feature_block_indices=(1,),
                   payload={"text": "def f(): pass"}),
            Region(kind="paragraph", feature_block_indices=(2,),
                   payload={"text": "Body text."}),
        ]

    comps = [_verdict_json(0, verdict="ok")]  # only the heading is prompted
    state = _council_state({1: ("code_block", 0.10)})  # would dispatch IF on

    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    out_unset, v_unset = run_structure_review(
        _regions(), fbs, _ScriptedRuntime(comps), council_state=state)
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "0")
    out_off, v_off = run_structure_review(
        _regions(), fbs, _ScriptedRuntime(comps), council_state=state)

    assert out_unset == out_off == _regions()  # content blocks untouched
    assert v_unset == v_off
    assert all(v.role_after is None for v in v_unset)
    # only the heading fired a prompt (content blocks not in scope flag-off).
    rt = _ScriptedRuntime(comps)
    run_structure_review(_regions(), fbs, rt, council_state=state)
    assert len(rt.batch_calls[0]["prompts"]) == 1


# ---------------------------------------------------------------------------
# Phase 4 — windowed dispatch + idx-keyed op-list parse + 1:1 output contract.
# All reached ONLY when SEMANTIK_BLOCK_REVIEW is on (the flag-off heading-only
# single-block path stays byte-identical — covered by
# test_block_review_flag_off_byte_stable above).
# ---------------------------------------------------------------------------


def _code_regions(n):
    """N code_block regions + their FBs (each its own FeatureBlock index)."""
    fbs = [_fb(f"code chunk {i} body text") for i in range(n)]
    regions = [
        Region(kind="code_block", feature_block_indices=(i,),
               payload={"text": f"code chunk {i} body text"})
        for i in range(n)
    ]
    return fbs, regions


def _low_conf_state(n):
    """Council state placing every FB's code_block role below the 0.4 floor."""
    return _council_state({i: ("code_block", 0.10) for i in range(n)})


def _oplist(window, *, kind="paragraph"):
    """A windowed op-list re-typing every member of ``window`` to ``kind``."""
    return json.dumps([
        {"idx": i, "verdict": "corrected", "corrected_kind": kind,
         "corrected_level": None, "corrected_doc_role": None, "review_note": "p"}
        for i in window
    ])


def test_windowed_dispatch_call_count(monkeypatch):
    # N in-scope blocks, window M, overlap stride -> ceil-based number of
    # window PROMPTS in the single generate_batch call (NOT N per-block calls).
    from dart_semantic.qwen_specialists.reviewer import (
        _BLOCK_REVIEW_WINDOW_OVERLAP,
        _build_windows,
    )

    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "4")
    n = 10
    fbs, regions = _code_regions(n)
    state = _low_conf_state(n)
    windows = _build_windows(list(range(n)), 4, _BLOCK_REVIEW_WINDOW_OVERLAP)
    assert len(windows) == 3  # ceil((10-1)/(4-1)) = 3 (window 4, overlap 1)
    completions = [_oplist(win) for win in windows]
    rt = _ScriptedRuntime(completions)
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    # ONE generate_batch call carrying ceil-based window prompts (not 10).
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 3
    # 1:1 output preserved.
    assert len(out) == n
    assert len(verdicts) == n
    _assert_coverage_invariant(out, len(fbs))


def test_op_list_keyed_by_idx_applies(monkeypatch):
    # A single window completion with two re-type ops -> both applied to the
    # correct members (idx-keyed).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "8")  # one window holds both
    fbs, regions = _code_regions(2)
    state = _low_conf_state(2)
    completion = json.dumps([
        {"idx": 0, "verdict": "corrected", "corrected_kind": "paragraph", "review_note": "a"},
        {"idx": 1, "verdict": "corrected", "corrected_kind": "paragraph", "review_note": "b"},
    ])
    rt = _ScriptedRuntime([completion])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 1  # one window, both members
    assert out[0].kind == "paragraph"
    assert out[1].kind == "paragraph"
    assert verdicts[0].kind_after == "paragraph"
    assert verdicts[1].kind_after == "paragraph"
    _assert_coverage_invariant(out, len(fbs))


def test_block_review_window_metadata_stamped(monkeypatch):
    # Phase 5: every dispatched verdict carries the shared per-window
    # block_review_window capture metadata (window_index + idx-range + model +
    # max_tokens + council kinds/conf); non-dispatched pass-throughs do not.
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "4")
    monkeypatch.setenv("SEMANTIK_STRUCTURE_REVIEW_MODEL", "meta/llama-3.3-70b")
    n = 10
    fbs, regions = _code_regions(n)
    state = _low_conf_state(n)
    from dart_semantic.qwen_specialists.reviewer import (
        _BLOCK_REVIEW_WINDOW_OVERLAP,
        _build_windows,
    )

    windows = _build_windows(list(range(n)), 4, _BLOCK_REVIEW_WINDOW_OVERLAP)
    completions = [_oplist(win) for win in windows]
    rt = _ScriptedRuntime(completions)
    _, verdicts = run_structure_review(regions, fbs, rt, council_state=state)

    stamped = [v for v in verdicts if v.block_review_window is not None]
    assert stamped, "expected windowed verdicts to carry block_review_window"
    seen_windows = set()
    for v in stamped:
        meta = v.block_review_window
        assert meta["model"] == "meta/llama-3.3-70b"
        assert meta["max_tokens"] >= 1
        assert meta["idx_min"] <= v.block_id <= meta["idx_max"]
        assert isinstance(meta["council_kinds"], dict) and meta["council_kinds"]
        seen_windows.add(meta["window_index"])
    # Each window owns its members first-wins -> >1 distinct window stamped.
    assert len(seen_windows) >= 2


def test_op_out_of_window_dropped(monkeypatch):
    # An op whose idx is OUTSIDE the window member set is dropped (no
    # cross-contamination of another region).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "8")
    fbs = [_fb("x=1"), _fb("y=2"), _fb("real prose stays")]
    regions = [
        Region(kind="code_block", feature_block_indices=(0,), payload={"text": "x=1"}),
        Region(kind="code_block", feature_block_indices=(1,), payload={"text": "y=2"}),
        Region(kind="paragraph", feature_block_indices=(2,),
               payload={"text": "real prose stays"}),
    ]
    # Only the two code_blocks are gated in; the plain paragraph (idx 2) is in
    # CONTENT_REVIEW_KINDS but the gate skips it (HIGH council confidence, no
    # pedagogical prefix) -> NOT dispatched (window={0,1}).
    state = _council_state({
        0: ("code_block", 0.10),
        1: ("code_block", 0.10),
        2: ("paragraph", 0.95),
    })
    completion = json.dumps([
        {"idx": 0, "verdict": "corrected", "corrected_kind": "paragraph", "review_note": "ok"},
        {"idx": 2, "verdict": "drop_injected_header",
         "corrected_kind": "metadata_drop", "review_note": "stray"},
    ])
    rt = _ScriptedRuntime([completion])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert out[0].kind == "paragraph"          # in-window op applied
    assert out[2].kind == "paragraph"          # stray op DROPPED -> unchanged
    assert verdicts[2].review_note == "non-heading; not reviewed"
    _assert_coverage_invariant(out, len(fbs))


def test_output_region_count_and_index_stable(monkeypatch):
    # len(reviewed) == len(regions) AND input region i maps to output region i
    # for every i (the cascade zip + C2 cap-safety baseline depend on this).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "3")
    fbs = [_fb("Chapter 1"), _fb("x=1"), _fb("prose"), _fb("y=2"), _fb("Chapter 2")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "Chapter 1"}),
        Region(kind="code_block", feature_block_indices=(1,), payload={"text": "x=1"}),
        Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "prose"}),
        Region(kind="code_block", feature_block_indices=(3,), payload={"text": "y=2"}),
        Region(kind="heading", feature_block_indices=(4,),
               payload={"level_hint": 2, "text": "Chapter 2"}),
    ]
    state = _council_state({1: ("code_block", 0.10), 3: ("code_block", 0.10)})
    # dispatched = [0(heading), 1(code), 3(code), 4(heading)]; paragraph 2 not gated.
    # window=3, overlap=1 -> windows [0,1,3], [3,4].
    completions = [
        json.dumps([{"idx": 1, "verdict": "corrected",
                     "corrected_kind": "paragraph", "review_note": "p"}]),
        json.dumps([{"idx": 4, "verdict": "drop_injected_header",
                     "corrected_kind": "metadata_drop", "review_note": "phantom"}]),
    ]
    rt = _ScriptedRuntime(completions)
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert len(out) == len(regions) == 5
    assert len(verdicts) == 5
    # index-stable: verdict i is about region i.
    assert all(verdicts[i].block_id == i for i in range(5))
    # the ops landed on the right slots; untouched slots unchanged.
    assert out[1].kind == "paragraph"      # re-typed code -> prose
    assert out[2].kind == "paragraph"      # not dispatched, unchanged
    assert out[4].kind == "metadata_drop"  # heading dropped
    assert out[0].kind == "heading"        # heading 0 had no op -> ok
    assert out[3].kind == "code_block"     # code 3 had no op -> ok
    _assert_coverage_invariant(out, len(fbs))


def test_full_window_op_list_not_truncated(monkeypatch):
    # A max-size window's op-list fits within the SCALED max_tokens (the
    # dispatched max_tokens scales with window size; all ops parsed/applied).
    from dart_semantic.qwen_specialists.reviewer import _PER_MEMBER_OP_MAX_TOKENS

    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "24")
    n = 24
    fbs, regions = _code_regions(n)
    state = _low_conf_state(n)
    completion = _oplist(list(range(n)))  # one window of 24, 24 ops
    rt = _ScriptedRuntime([completion])
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert len(rt.batch_calls) == 1
    assert len(rt.batch_calls[0]["prompts"]) == 1  # single 24-member window
    md = rt.batch_calls[0]["max_tokens"]
    # max_tokens scaled to the window size -> exceeds the legacy 512 default.
    assert md == _PER_MEMBER_OP_MAX_TOKENS * n
    assert md > 512
    # every one of the 24 ops parsed + applied (none truncated away).
    assert all(o.kind == "paragraph" for o in out)
    assert all(v.kind_after == "paragraph" for v in verdicts)


def test_window_fail_soft_keeps_deterministic(monkeypatch):
    # A None window (endpoint failure) -> its members kept verbatim,
    # reverted_for_endpoint_failure=True; sibling windows still reviewed.
    from dart_semantic.qwen_specialists.reviewer import _build_windows

    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "3")
    n = 6
    fbs, regions = _code_regions(n)
    state = _low_conf_state(n)
    windows = _build_windows(list(range(n)), 3, 1)  # [0,1,2],[2,3,4],[4,5]
    completions = [_oplist(win) for win in windows]
    rt = _ScriptedRuntime(completions, fail_indices={0})  # window 0 -> None
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    # idx0 is unique to the failed window -> kept verbatim, endpoint-degraded.
    assert out[0].kind == "code_block"
    assert verdicts[0].reverted_for_endpoint_failure is True
    # idx5 is unique to a SUCCESSFUL window -> reviewed (re-typed).
    assert out[5].kind == "paragraph"
    assert verdicts[5].reverted_for_endpoint_failure is False
    _assert_coverage_invariant(out, len(fbs))


def test_anti_crawl_degrade(monkeypatch):
    # >= 80% None windows -> whole review degraded to the byte-stable
    # UNREVIEWED floor (original regions returned verbatim).
    from dart_semantic.qwen_specialists.reviewer import _build_windows

    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_WINDOW", "2")
    n = 6
    fbs, regions = _code_regions(n)
    state = _low_conf_state(n)
    windows = _build_windows(list(range(n)), 2, 1)  # 5 windows
    completions = [_oplist(win) for win in windows]
    rt = _ScriptedRuntime(completions, fail_indices=set(range(len(windows))))  # all None
    out, verdicts = run_structure_review(regions, fbs, rt, council_state=state)
    assert out is regions                       # byte-stable unreviewed floor
    assert all(o.kind == "code_block" for o in out)  # nothing re-typed
    _assert_coverage_invariant(out, len(fbs))


def test_flag_off_single_block_path_byte_stable(monkeypatch):
    # Flag off -> the windowed driver is NEVER entered; the legacy heading
    # single-block path runs (one prompt per heading). Asserting the branch is
    # not taken complements the committed behavioral byte-stability test.
    import dart_semantic.qwen_specialists.reviewer as rv

    def _boom(*a, **k):
        raise AssertionError("windowed path must NOT run with the flag off")

    monkeypatch.setattr(rv, "_run_windowed_block_review", _boom)
    monkeypatch.delenv("SEMANTIK_BLOCK_REVIEW", raising=False)
    fbs = [_fb("Chapter 1"), _fb("body")]
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "Chapter 1"}),
        Region(kind="paragraph", feature_block_indices=(1,), payload={"text": "body"}),
    ]
    rt = _ScriptedRuntime([_verdict_json(0, kind="heading", level=1, note="rl")])
    out, verdicts = run_structure_review(regions, fbs, rt)
    assert out[0].payload["level_hint"] == 1        # legacy heading path ran
    assert len(rt.batch_calls[0]["prompts"]) == 1   # one heading prompt
    _assert_coverage_invariant(out, len(fbs))


# ---------------------------------------------------------------------------
# Phase 3 — frozen Pass-2 verifier verdict schema (SEMANTIK_SECOND_PASS).
# ---------------------------------------------------------------------------


def test_verdict_dataclass_frozen():
    import dataclasses

    from dart_semantic.qwen_specialists.reviewer import FlaggedBlock, VerifierVerdict

    fb = FlaggedBlock(region_index=3, failure_mode="example_as_heading",
                      fix_hint="re-type", fixable=True)
    vv = VerifierVerdict(passed=False, flagged=(fb,), spot_html_requested=(5,))
    assert dataclasses.is_dataclass(fb) and dataclasses.is_dataclass(vv)
    for frozen_obj in (fb, vv):
        with pytest.raises(dataclasses.FrozenInstanceError):
            frozen_obj.region_index = 9  # type: ignore[attr-defined]
    # The verdict carries only index-keyed data, never source text.
    assert vv.flagged[0].region_index == 3
    assert vv.spot_html_requested == (5,)


def test_flagged_block_proposed_regroup_run_optional():
    from dart_semantic.qwen_specialists.reviewer import FlaggedBlock

    # Defaults empty (re-type modes).
    retype = FlaggedBlock(region_index=2, failure_mode="mistyped_component",
                          fix_hint="re-type to paragraph", fixable=True)
    assert retype.proposed_regroup_run == ()
    # Populated for a merge mode (RECONCILIATION DELTA — Phase 6b channel).
    merge = FlaggedBlock(region_index=4, failure_mode="section_no_body",
                         fix_hint="merge", fixable=False,
                         proposed_regroup_run=(4, 5, 6))
    assert merge.proposed_regroup_run == (4, 5, 6)


# ---------------------------------------------------------------------------
# Phase 4 — verifier DISPATCH (run_second_pass_verify) over the shared seat.
#
# Builds the assembled-document digest, dispatches ONE single-prompt verify
# batch (serialized by construction), parses the verdict, and on a need_spot_html
# request re-dispatches ONCE with the section's real HTML. FAIL-SAFE: any error /
# endpoint-soft-None / unparseable -> passed=True (a broken verifier never FAILS
# a good doc). Gated on SEMANTIK_SECOND_PASS (the builder returns "" when off).
# ---------------------------------------------------------------------------


def _verify_json(flags=(), spot=()) -> str:
    """A Pass-2 verifier JSON-array response (one object per flag/spot request)."""
    arr: list = []
    for f in flags:
        arr.append({
            "region_index": f["region_index"],
            "failure_mode": f["failure_mode"],
            "fix_hint": f.get("fix_hint", ""),
            "fixable": f.get("fixable", True),
        })
    for ri in spot:
        arr.append({"region_index": ri, "failure_mode": "need_spot_html"})
    return json.dumps(arr)


def _verify_doc():
    """Minimal assembled doc: a heading (anchored 'sec-1') + a worked_example
    body whose section HTML carries a unique token (BODYUNIQUE) so the spot-HTML
    re-ask is observable. region_provenance=[0,1] (identity)."""
    regions = [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "H"}),
        Region(kind="paragraph", feature_block_indices=(1,),
               payload={"text": "EXAMPLE 1.3 solve x.", "semantic_class": "worked_example"}),
    ]
    fbs = [_fb("H"), _fb("EXAMPLE 1.3 solve x.")]
    doc = AssembledDoc(
        html='<section aria-labelledby="sec-1"><h2 id="sec-1">H</h2>'
             '<p>BODYUNIQUE solve x.</p></section>',
        gaps_found=[], gaps_resolved=[], gaps_fallback=[],
        heading_tree=[(2, "H")], landmarks={}, anchors={"sec-1": 0},
        region_provenance=[0, 1],
        sub_task_log={"region_html": ['<h2 id="sec-1">H</h2>',
                                      "<p>BODYUNIQUE solve x.</p>"]},
    )
    return doc, regions, fbs


def test_verify_pass_clean_doc(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _verify_doc()
    rt = _ScriptedRuntime([_verify_json()], queue=True)  # empty array -> PASS
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    assert verdict.passed is True
    assert verdict.flagged == ()
    assert len(rt.batch_calls) == 1


def test_verify_flags_seeded_example_as_heading(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _verify_doc()
    rt = _ScriptedRuntime([_verify_json(flags=[{
        "region_index": 1, "failure_mode": "example_as_heading",
        "fix_hint": "re-type to worked_example", "fixable": True,
    }])], queue=True)
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    assert verdict.passed is False
    assert len(verdict.flagged) == 1
    flag = verdict.flagged[0]
    assert flag.region_index == 1
    assert flag.failure_mode == "example_as_heading"
    assert flag.fixable is True


def test_verify_spot_html_reask_once(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _verify_doc()
    # Round 1: need_spot_html for the content region; round 2: a real flag.
    rt = _ScriptedRuntime([
        _verify_json(spot=[1]),
        _verify_json(flags=[{"region_index": 1,
                             "failure_mode": "mistyped_component",
                             "fix_hint": "x", "fixable": True}]),
    ], queue=True)
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    # EXACTLY one re-dispatch (two calls total).
    assert len(rt.batch_calls) == 2
    # The 2nd prompt carries the enclosing SECTION's real HTML bytes.
    second_prompt = rt.batch_calls[1]["prompts"][0]
    assert "BODYUNIQUE" in second_prompt
    # The re-ask's flag is returned.
    assert verdict.passed is False
    assert verdict.flagged[0].region_index == 1


def test_verify_endpoint_error_is_failsafe_pass(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _verify_doc()
    # Call 0 returns the None sentinel (endpoint soft-fail) -> fail-safe PASS.
    rt = _ScriptedRuntime([_verify_json(flags=[{
        "region_index": 1, "failure_mode": "mistyped_component",
        "fix_hint": "x", "fixable": True,
    }])], fail_indices={0}, queue=True)
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    assert verdict.passed is True  # never blocks a good doc
    assert verdict.flagged == ()


def test_verify_serialized_concurrency_one(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    monkeypatch.setenv("SEMANTIK_SPECIALIST_CONCURRENCY", "1")
    doc, regions, fbs = _verify_doc()
    rt = _ScriptedRuntime([_verify_json(spot=[1]), _verify_json()], queue=True)
    run_second_pass_verify(doc, regions, fbs, rt)
    # Every dispatch is a SINGLE-prompt batch -> serialized, never fanned out.
    assert rt.batch_calls  # at least one call fired
    for call in rt.batch_calls:
        assert len(call["prompts"]) == 1


def test_verify_no_text_emitted(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _verify_doc()
    rt = _ScriptedRuntime([_verify_json(flags=[{
        "region_index": 1, "failure_mode": "mistyped_component",
        "fix_hint": "re-type the kind", "fixable": True,
    }])], queue=True)
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    flag = verdict.flagged[0]
    # The verdict carries ONLY index-keyed schema fields — no source text leaks.
    assert set(vars(flag).keys()) == {
        "region_index", "failure_mode", "fix_hint", "fixable",
        "proposed_regroup_run",
    }
    assert "EXAMPLE 1.3 solve x." not in flag.fix_hint


# ---------------------------------------------------------------------------
# Section-aligned digest WINDOWING (this session) — run_second_pass_verify now
# windows the digest, dispatches one call per window, and aggregates. A
# multi-section doc + a tiny per-window budget forces multiple windows.
# ---------------------------------------------------------------------------


def _multi_section_doc(n_sections=3):
    """An assembled doc of ``n_sections`` heading+paragraph sections, identity
    region_provenance — its digest splits into ``n_sections`` sections."""
    regions = []
    rh = []
    for s in range(n_sections):
        regions.append(Region(kind="heading", feature_block_indices=(2 * s,),
                              payload={"level_hint": 2, "text": f"Head{s}"}))
        regions.append(Region(kind="paragraph", feature_block_indices=(2 * s + 1,),
                              payload={"text": f"Body para number {s}."}))
        rh.append(f'<h2 id="head-{s}">Head{s}</h2>')
        rh.append(f"<p>Body para number {s}.</p>")
    fbs = []
    for s in range(n_sections):
        fbs.append(_fb(f"Head{s}"))
        fbs.append(_fb(f"Body para number {s}."))
    html = "".join(rh)
    anchors = {f"head-{s}": 2 * s for s in range(n_sections)}
    doc = AssembledDoc(
        html=html, gaps_found=[], gaps_resolved=[], gaps_fallback=[],
        heading_tree=[(2, f"Head{s}") for s in range(n_sections)],
        landmarks={}, anchors=anchors,
        region_provenance=list(range(2 * n_sections)),
        sub_task_log={"region_html": rh},
    )
    return doc, regions, fbs


def _set_one_section_budget(monkeypatch, doc, regions, fbs):
    """Pin SEMANTIK_SECOND_PASS_WINDOW_TOKENS so each section is its own window."""
    from dart_semantic.assembler.verify_digest import (
        _estimate_tokens, _split_into_sections, build_verifier_digest,
    )
    digest = build_verifier_digest(doc, regions, fbs, edge_tokens=12)
    budget = max(_estimate_tokens(s) for s in _split_into_sections(digest["regions"]))
    monkeypatch.setenv("SEMANTIK_SECOND_PASS_WINDOW_TOKENS", str(budget))


def test_verify_aggregates_passed_all(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _multi_section_doc(3)
    _set_one_section_budget(monkeypatch, doc, regions, fbs)
    rt = _ScriptedRuntime([_verify_json(), _verify_json(), _verify_json()], queue=True)
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    assert verdict.passed is True
    assert verdict.flagged == ()
    assert len(rt.batch_calls) == 3  # one dispatch per window
    for call in rt.batch_calls:
        assert len(call["prompts"]) == 1  # serialized single-prompt batches


def test_verify_aggregates_flagged_union_dedup(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _multi_section_doc(3)
    _set_one_section_budget(monkeypatch, doc, regions, fbs)
    # Window 0 (ids 0,1) flags idx 1 TWICE (a dup -> deduped); window 1 (ids 2,3)
    # flags idx 3; window 2 passes.
    win0 = _verify_json(flags=[
        {"region_index": 1, "failure_mode": "mistyped_component",
         "fix_hint": "first", "fixable": True},
        {"region_index": 1, "failure_mode": "mistyped_component",
         "fix_hint": "second", "fixable": True},
    ])
    win1 = _verify_json(flags=[{"region_index": 3, "failure_mode": "example_as_heading",
                                "fix_hint": "B", "fixable": True}])
    rt = _ScriptedRuntime([win0, win1, _verify_json()], queue=True)
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    assert verdict.passed is False
    idxs = sorted(f.region_index for f in verdict.flagged)
    assert idxs == [1, 3]  # union across windows, idx 1 deduped to one
    by_idx = {f.region_index: f for f in verdict.flagged}
    assert by_idx[1].fix_hint == "first"  # first/strongest per index kept


def test_verify_one_window_fails_doc_fails(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _multi_section_doc(3)
    _set_one_section_budget(monkeypatch, doc, regions, fbs)
    win1 = _verify_json(flags=[{"region_index": 3, "failure_mode": "mistyped_component",
                                "fix_hint": "x", "fixable": True}])
    rt = _ScriptedRuntime([_verify_json(), win1, _verify_json()], queue=True)
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    assert verdict.passed is False  # one window failing fails the whole doc
    assert sorted(f.region_index for f in verdict.flagged) == [3]


def test_verify_window_error_is_failsafe_pass(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    doc, regions, fbs = _multi_section_doc(3)
    _set_one_section_budget(monkeypatch, doc, regions, fbs)
    # Window 1 dispatch errors (None sentinel) -> that window treated passed;
    # window 0 flags idx 1, window 2 passes -> the rest aggregate normally.
    win0 = _verify_json(flags=[{"region_index": 1, "failure_mode": "mistyped_component",
                                "fix_hint": "x", "fixable": True}])
    rt = _ScriptedRuntime([win0, _verify_json(), _verify_json()],
                          fail_indices={1}, queue=True)
    verdict = run_second_pass_verify(doc, regions, fbs, rt)
    assert verdict.passed is False
    assert sorted(f.region_index for f in verdict.flagged) == [1]
    # All windows error -> fail-safe PASS (never block a good doc).
    doc2, regions2, fbs2 = _multi_section_doc(3)
    _set_one_section_budget(monkeypatch, doc2, regions2, fbs2)
    rt2 = _ScriptedRuntime([_verify_json(), _verify_json(), _verify_json()],
                           fail_indices={0, 1, 2}, queue=True)
    verdict2 = run_second_pass_verify(doc2, regions2, fbs2, rt2)
    assert verdict2.passed is True
    assert verdict2.flagged == ()


def test_second_pass_window_tokens_flag(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SECOND_PASS_WINDOW_TOKENS", raising=False)
    assert resolve_second_pass_window_tokens() == 8000  # default
    for bad in ("", "  ", "abc", "0", "-5", "3.5"):
        monkeypatch.setenv("SEMANTIK_SECOND_PASS_WINDOW_TOKENS", bad)
        assert resolve_second_pass_window_tokens() == 8000
    monkeypatch.setenv("SEMANTIK_SECOND_PASS_WINDOW_TOKENS", "16000")
    assert resolve_second_pass_window_tokens() == 16000


# ---------------------------------------------------------------------------
# Phase 5 — targeted Pass-1 re-drive seam (restrict_to + feedback_by_idx).
#
# restrict_to narrows dispatch to the flagged subset (AND-composed AFTER
# _in_review_scope); a flagged member short-circuits the conf gate; verifier
# feedback is injected into the prompt (busting the round-1 cache). Default-None
# is byte-identical; 1:1 index-stable is preserved.
# ---------------------------------------------------------------------------


def _dispatched_idxs(rt: _ScriptedRuntime) -> set[int]:
    """Idx set dispatched across all calls (windowed 'idx' or single 'block_id')."""
    out: set[int] = set()
    for call in rt.batch_calls:
        for prompt in call["prompts"]:
            user = prompt.split("\nUSER: ", 1)[1]
            data = json.loads(user)
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict) and "idx" in rec:
                        out.add(rec["idx"])
            elif isinstance(data, dict) and "block_id" in data:
                out.add(data["block_id"])
    return out


def _mixed_regions():
    """1 heading + 5 code_blocks (unconditional-dispatch content kind)."""
    return [
        Region(kind="heading", feature_block_indices=(0,),
               payload={"level_hint": 2, "text": "H0"}),
        Region(kind="code_block", feature_block_indices=(1,),
               payload={"text": "TRY IT 1.1 a"}),
        Region(kind="code_block", feature_block_indices=(2,),
               payload={"text": "TRY IT 1.2 b"}),
        Region(kind="code_block", feature_block_indices=(3,),
               payload={"text": "TRY IT 1.3 c"}),
        Region(kind="code_block", feature_block_indices=(4,),
               payload={"text": "TRY IT 1.4 d"}),
        Region(kind="code_block", feature_block_indices=(5,),
               payload={"text": "TRY IT 1.5 e"}),
    ]


def _mixed_fbs():
    return [_fb("H0"), _fb("TRY IT 1.1 a"), _fb("TRY IT 1.2 b"),
            _fb("TRY IT 1.3 c"), _fb("TRY IT 1.4 d"), _fb("TRY IT 1.5 e")]


def test_restrict_to_dispatches_only_flagged(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", "0")
    regions, fbs = _mixed_regions(), _mixed_fbs()
    rt = _ScriptedRuntime(["[]"])  # no ops -> all dispatched pass through ok
    out, verdicts = run_structure_review(
        regions, fbs, rt, restrict_to=frozenset({2, 5})
    )
    # Only the two flagged indices are dispatched.
    assert _dispatched_idxs(rt) == {2, 5}
    # 1:1 preserved + every non-flagged region passes through ok.
    assert len(out) == len(regions)
    for i, v in enumerate(verdicts):
        if i not in (2, 5):
            assert v.verdict == "ok"


def test_restrict_to_bypasses_conf_gate(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", "0")
    # A plain paragraph + NO council_state -> the dispatch gate returns False
    # (it would be SKIPPED on a whole-document review).
    def _regions():
        return [
            Region(kind="heading", feature_block_indices=(0,),
                   payload={"level_hint": 2, "text": "H"}),
            Region(kind="paragraph", feature_block_indices=(1,),
                   payload={"text": "plain confident prose body"}),
        ]
    fbs = [_fb("H"), _fb("plain confident prose body")]
    # Whole-doc: the paragraph is gated OUT (only the heading is reviewed).
    rt_all = _ScriptedRuntime(["[]"])
    run_structure_review(_regions(), fbs, rt_all)
    assert 1 not in _dispatched_idxs(rt_all)
    # restrict_to={1}: the flagged paragraph IS dispatched despite the gate.
    rt_r = _ScriptedRuntime(["[]"])
    run_structure_review(_regions(), fbs, rt_r, restrict_to=frozenset({1}))
    assert _dispatched_idxs(rt_r) == {1}


def test_feedback_injected_redrive_in_prompt(monkeypatch):
    # The injected feedback reaches the dispatched USER JSON (windowed records).
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", "0")
    regions, fbs = _mixed_regions(), _mixed_fbs()
    rt = _ScriptedRuntime(["[]"])
    run_structure_review(
        regions, fbs, rt,
        restrict_to=frozenset({2, 5}),
        feedback_by_idx={2: "re-type to worked_example"},
    )
    prompt = rt.batch_calls[0]["prompts"][0]
    assert "re-type to worked_example" in prompt
    assert "review_feedback" in prompt


def test_cache_miss_on_feedback_change(monkeypatch, tmp_path):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", "1")  # cache ON
    monkeypatch.setenv("SEMANTIK_CACHE_DIR", str(tmp_path))  # hermetic
    regions, fbs = _mixed_regions(), _mixed_fbs()
    op = json.dumps([{"idx": 2, "verdict": "corrected",
                      "corrected_kind": "paragraph", "review_note": "n"}])
    # Run A (no feedback) -> populates the cache.
    rt_a = _ScriptedRuntime([op])
    run_structure_review(regions, fbs, rt_a, restrict_to=frozenset({2, 5}))
    assert rt_a.batch_calls  # A dispatched
    # Run B (identical, no feedback) -> cache HIT, ZERO dispatch.
    rt_b = _ScriptedRuntime([op])
    run_structure_review(_mixed_regions(), fbs, rt_b, restrict_to=frozenset({2, 5}))
    assert rt_b.batch_calls == []  # full cache hit
    # Run C (feedback injected) -> prompt bytes differ -> cache MISS, re-dispatch.
    rt_c = _ScriptedRuntime([op])
    run_structure_review(
        _mixed_regions(), fbs, rt_c,
        restrict_to=frozenset({2, 5}),
        feedback_by_idx={2: "verifier hint"},
    )
    assert rt_c.batch_calls  # feedback busted the cache


def test_restrict_none_byte_identical(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", "0")
    fbs = _mixed_fbs()
    op = json.dumps([{"idx": 2, "verdict": "corrected",
                      "corrected_kind": "paragraph", "review_note": "n"}])
    rt_base = _ScriptedRuntime([op])
    base_regions, base_verdicts = run_structure_review(_mixed_regions(), fbs, rt_base)
    rt_param = _ScriptedRuntime([op])
    p_regions, p_verdicts = run_structure_review(
        _mixed_regions(), fbs, rt_param, restrict_to=None, feedback_by_idx=None
    )
    # Prompts byte-identical.
    base_prompts = [p for c in rt_base.batch_calls for p in c["prompts"]]
    param_prompts = [p for c in rt_param.batch_calls for p in c["prompts"]]
    assert base_prompts == param_prompts
    # Verdicts byte-identical (kinds + per-verdict fields).
    assert [r.kind for r in base_regions] == [r.kind for r in p_regions]
    assert [v.verdict for v in base_verdicts] == [v.verdict for v in p_verdicts]
    assert [v.kind_after for v in base_verdicts] == [v.kind_after for v in p_verdicts]


def test_one_to_one_preserved_on_subset_redrive(monkeypatch):
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW", "1")
    monkeypatch.setenv("SEMANTIK_BLOCK_REVIEW_CACHE", "0")
    regions, fbs = _mixed_regions(), _mixed_fbs()
    n = len(regions)
    op = json.dumps([{"idx": 2, "verdict": "corrected",
                      "corrected_kind": "paragraph", "review_note": "n"}])
    out, verdicts = run_structure_review(
        regions, fbs, _ScriptedRuntime([op]), restrict_to=frozenset({2})
    )
    assert len(out) == n
    assert len(verdicts) == n


# ---------------------------------------------------------------------------
# ITEM6 — window capture meta gains council_margin_min / _mean (top1-top2 over
# members carrying council_top_k; members lacking the key excluded).
# ---------------------------------------------------------------------------


def test_window_capture_meta_margin_stats():
    from dart_semantic.qwen_specialists.reviewer import _build_window_capture_meta

    record_by_idx = {
        0: {"page": 1, "confidence": 0.7, "council_kind": "code_block",
            "council_top_k": [["code_block", 0.6], ["paragraph", 0.3]]},  # margin 0.3
        1: {"page": 1, "confidence": 0.8, "council_kind": "paragraph",
            "council_top_k": [["paragraph", 0.5], ["list_item", 0.4]]},   # margin 0.1
        2: {"page": 2, "confidence": 0.9, "council_kind": "table"},        # no key -> excluded
    }
    meta = _build_window_capture_meta(
        0, [0, 1, 2], record_by_idx, model="m", max_tokens=256
    )
    assert meta["council_margin_min"] == 0.1
    assert meta["council_margin_mean"] == 0.2


def test_window_capture_meta_margin_absent_without_key():
    from dart_semantic.qwen_specialists.reviewer import _build_window_capture_meta

    record_by_idx = {
        0: {"page": 1, "confidence": 0.7, "council_kind": "paragraph"},
        1: {"page": 1, "confidence": 0.8, "council_kind": "table"},
    }
    meta = _build_window_capture_meta(
        0, [0, 1], record_by_idx, model="m", max_tokens=256
    )
    assert "council_margin_min" not in meta
    assert "council_margin_mean" not in meta
