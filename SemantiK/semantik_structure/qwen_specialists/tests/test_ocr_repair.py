"""Unit tests — SEMANTIK_OCR_CONFUSABLE_REPAIR (plan Phase 5 channels 2+3).

Pin the bounded OCR-confusable micro-repair contract: the flag resolver's
parse-with-fallback, the flag-off byte-stability (zero dispatch), the garble
detector, the DETERMINISTIC accept gate (the auditable owner of every edit),
the strict replay-conservation check, the per-block cap + per-doc runaway
tripwire, the fail-soft windowed dispatch, and the residual-density stats.
"""

from __future__ import annotations

import json

from semantik_structure.qwen_specialists.ocr_repair import (
    _REPAIR_SCORE_CEILING,
    apply_accepted_edits,
    assert_repair_conservation,
    evaluate_edit,
    region_text_is_garbled,
    resolve_ocr_confusable_repair_mode,
    run_ocr_confusable_repair,
    RepairConservationError,
)
from semantik_structure.qwen_specialists.ocr_repair_prompt import (
    build_ocr_repair_request,
    parse_ocr_repair_response,
)
from semantik_structure.structure_graph import Region
from semantik_structure.types import FeatureBlock, RawBlock


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


def _fb(text: str) -> FeatureBlock:
    raw = RawBlock(
        text=text, page=1, bbox=(0.0, 0.0, 10.0, 10.0),
        page_width=100.0, page_height=100.0,
    )
    return FeatureBlock(
        raw=raw, size_bucket="md", gap_above=None, is_top_of_page=False,
        is_centered=False, caps=None, indent_bucket=0, relative_font_ratio=1.0,
    )


def _region(fb_index: int) -> Region:
    return Region(kind="paragraph", feature_block_indices=(fb_index,), payload={})


class _ScriptedRuntime:
    """Records prompts and returns pre-scripted completions per call."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.prompts = []
        self.calls = 0

    def generate_batch(self, prompts, **_kw):
        self.prompts.extend(prompts)
        out = []
        for _ in prompts:
            self.calls += 1
            out.append(self._responses.pop(0) if self._responses else None)
        return out


def _on(monkeypatch):
    monkeypatch.setenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", "on")
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "on")  # window builder gate reuse


# ---------------------------------------------------------------------------
# Flag resolver — parse-with-fallback (default OFF).
# ---------------------------------------------------------------------------


def test_resolver_off_by_default(monkeypatch):
    monkeypatch.delenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", raising=False)
    assert resolve_ocr_confusable_repair_mode() is False
    for v in ("", "  ", "0", "false", "no", "off", "garbage", "2"):
        monkeypatch.setenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", v)
        assert resolve_ocr_confusable_repair_mode() is False


def test_resolver_truthy_tokens(monkeypatch):
    for v in ("1", "true", "yes", "on", "TRUE", "On"):
        monkeypatch.setenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", v)
        assert resolve_ocr_confusable_repair_mode() is True


# ---------------------------------------------------------------------------
# Flag OFF byte-stability — no dispatch, no edits.
# ---------------------------------------------------------------------------


def test_flag_off_zero_dispatch(monkeypatch):
    monkeypatch.delenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", raising=False)
    fbs = [_fb("Solve Vx = 4 for x.")]
    regions = [_region(0)]
    rt = _ScriptedRuntime(['[{"region_index":0,"before":"V","after":"√"}]'])
    result = run_ocr_confusable_repair(regions, fbs, rt)
    assert rt.calls == 0  # build returns "" when the flag is off -> no dispatch
    assert result.edits_by_region == {}


# ---------------------------------------------------------------------------
# Garble detector.
# ---------------------------------------------------------------------------


def test_detector_flags_confusable_patterns():
    assert region_text_is_garbled("Solve Vx = 4") is True  # radical garble
    assert region_text_is_garbled("the answer is 1ike this") is True  # 1/l
    assert region_text_is_garbled("branded ™ term") is True  # decoration
    assert region_text_is_garbled("a curly “quote here") is True


def test_detector_ignores_clean_prose():
    assert region_text_is_garbled("The quick brown fox jumps over.") is False
    assert region_text_is_garbled("Solve the equation for x carefully.") is False


def test_detector_excludes_digit_dense_answer_lines():
    # Answer-list shapes / digit-dense lines never flag.
    assert region_text_is_garbled("3.2: 14, 17, 20, 23") is False
    assert region_text_is_garbled("1. 42 2. 108 3. 256") is False


# ---------------------------------------------------------------------------
# Deterministic accept gate.
# ---------------------------------------------------------------------------


def test_gate_accepts_in_map_edits():
    d = evaluate_edit("Solve Vx = 4", "V", "√")
    assert d.accepted is True and d.reason == "v_to_radical"
    d2 = evaluate_edit("write 1ike this", "1", "l")
    assert d2.accepted is True and d2.reason == "1_to_l"
    d3 = evaluate_edit("turn rnodel around", "rn", "m")
    assert d3.accepted is True and d3.reason == "rn_to_m"


def test_gate_rejects_out_of_map_pair():
    d = evaluate_edit("cat runs", "c", "d")  # c->d not a confusable
    assert d.accepted is False and d.reason == "not_in_map"


def test_gate_rejects_edit_distance_gt_2():
    # 'abcd' -> 'wxyz' is not map-reachable anyway; use a map char with a long
    # target so the map path is not the reason. A single char to a 4-char string
    # is not reachable -> not_in_map already; assert the map guard fires.
    d = evaluate_edit("O here", "O", "0000")
    assert d.accepted is False and d.reason == "not_in_map"


def test_gate_rejects_digit_letter_without_map_entry():
    # 7 <-> a is not in the map -> rejected (char-class enforced by the map).
    d = evaluate_edit("a value", "a", "7")
    assert d.accepted is False and d.reason == "not_in_map"


def test_gate_rejects_numeric_context():
    # Editing inside a digit-dense token (an answer / number) is refused even
    # for an in-map pair.
    d = evaluate_edit("code 12O5 here", "O", "0")
    assert d.accepted is False and d.reason == "numeric_context"


def test_gate_rejects_token_count_change():
    d = evaluate_edit("a b split", "a b", "ab")  # whitespace in before
    assert d.accepted is False and d.reason == "not_within_token"


def test_gate_rejects_identity():
    d = evaluate_edit("Vx here", "V", "V")
    assert d.accepted is False and d.reason == "identity"


def test_gate_rejects_before_not_present():
    d = evaluate_edit("clean text", "V", "√")  # no 'V' in text
    assert d.accepted is False and d.reason == "before_not_found"


# ---------------------------------------------------------------------------
# Replay-conservation check.
# ---------------------------------------------------------------------------


def test_replay_conservation_passes_on_exact_edits():
    original = "Solve Vx = 4"
    edits = [{"before": "V", "after": "√"}]
    repaired = apply_accepted_edits(original, edits)
    assert repaired == "Solve √x = 4"
    assert_repair_conservation(original, repaired, edits)  # no raise


def test_replay_conservation_fails_on_extra_drift():
    original = "Solve Vx = 4"
    edits = [{"before": "V", "after": "√"}]
    tampered = "Solve √y = 9"  # extra unrecorded change
    try:
        assert_repair_conservation(original, tampered, edits)
    except RepairConservationError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected RepairConservationError")


# ---------------------------------------------------------------------------
# Per-block cap + per-doc runaway tripwire.
# ---------------------------------------------------------------------------


def test_per_block_cap_bounds_accepted_edits(monkeypatch):
    _on(monkeypatch)
    # A garbled region with 12 distinct 'Vword_N' tokens (each proposes one
    # V->√ edit, > _MAX_EDITS 8) padded with 600 clean tokens so the runaway
    # fraction stays well under 2% and the PER-BLOCK cap is what bounds it.
    garbled = " ".join(f"Vword{i}" for i in range(12))
    clean = " ".join("clean" for _ in range(600))
    fbs = [_fb(f"{garbled} {clean}")]
    regions = [_region(0)]
    proposals = [
        {"region_index": 0, "before": "V", "after": "√"} for _ in range(12)
    ]
    rt = _ScriptedRuntime([json.dumps(proposals)])
    result = run_ocr_confusable_repair(regions, fbs, rt)
    assert result.stats.get("runaway_reverted") is not True
    # Accepted edits per block are capped at 8.
    assert 0 in result.edits_by_region
    assert result.edits_by_region[0]["ocr_repair"]["n_edits"] == 8


def test_duplicate_proposal_does_not_drift_onto_numeric_token(monkeypatch):
    _on(monkeypatch)
    # Region flagged by the 'Vx' radical garble; it ALSO contains a digit-dense
    # answer token '12V57' (excluded at detect time). TWO identical V->√
    # proposals: the running-text accept gate must land the first on 'Vx' and
    # REJECT the second (its only remaining 'V' is inside the numeric token) —
    # the pre-fix static-first-occurrence gate validated both against 'Vx' while
    # the second silently mutated '12V57' -> '12√57' at apply time.
    text = "Solve Vx = 4 then 12V57 stands as the worked answer value here."
    fbs = [_fb(text)]
    regions = [_region(0)]
    proposals = [
        {"region_index": 0, "before": "V", "after": "√"},
        {"region_index": 0, "before": "V", "after": "√"},
    ]
    rt = _ScriptedRuntime([json.dumps(proposals)])
    result = run_ocr_confusable_repair(regions, fbs, rt)
    assert 0 in result.edits_by_region
    entry = result.edits_by_region[0]
    # Exactly ONE edit landed (the 'Vx' radical); the numeric token is intact.
    assert entry["ocr_repair"]["n_edits"] == 1
    assert entry["repaired_text"] == (
        "Solve √x = 4 then 12V57 stands as the worked answer value here."
    )
    assert "12√57" not in entry["repaired_text"]
    # The duplicate was rejected for landing in a numeric context.
    reasons = {r.get("reason") for r in result.stats["rejected"]}
    assert "numeric_context" in reasons


def test_runaway_tripwire_reverts_whole_pass(monkeypatch):
    _on(monkeypatch)
    # 3 tiny garbled regions, one token each; the accepted edits blow past 2%.
    fbs = [_fb("Vx"), _fb("Vy"), _fb("Vz")]
    regions = [_region(0), _region(1), _region(2)]
    proposals = [
        {"region_index": 0, "before": "V", "after": "√"},
        {"region_index": 1, "before": "V", "after": "√"},
        {"region_index": 2, "before": "V", "after": "√"},
    ]
    rt = _ScriptedRuntime([json.dumps(proposals)])
    result = run_ocr_confusable_repair(regions, fbs, rt)
    assert result.edits_by_region == {}  # whole pass reverted
    assert result.stats.get("runaway_reverted") is True


# ---------------------------------------------------------------------------
# Windowing + fail-soft dispatch.
# ---------------------------------------------------------------------------


def test_window_dispatch_and_accept(monkeypatch):
    _on(monkeypatch)
    text = "Solve Vx = 4 for the value of x in this worked problem statement."
    fbs = [_fb(text)]
    regions = [_region(0)]
    rt = _ScriptedRuntime(['[{"region_index":0,"before":"V","after":"√"}]'])
    result = run_ocr_confusable_repair(regions, fbs, rt)
    assert rt.calls == 1
    assert 0 in result.edits_by_region
    entry = result.edits_by_region[0]
    assert entry["repaired_text"].startswith("Solve √x")
    assert entry["ocr_repair"]["n_edits"] == 1
    assert entry["ocr_repair"]["edits"][0]["rule_id"] == "v_to_radical"


def test_window_failsoft_on_bad_json(monkeypatch):
    _on(monkeypatch)
    fbs = [_fb("Solve Vx = 4 for the value of x in this worked problem.")]
    regions = [_region(0)]
    rt = _ScriptedRuntime(["not json at all {{{"])
    result = run_ocr_confusable_repair(regions, fbs, rt)
    assert result.edits_by_region == {}  # no accepted edits, no raise


def test_window_failsoft_on_none_completion(monkeypatch):
    _on(monkeypatch)
    fbs = [_fb("Solve Vx = 4 for the value of x in this worked problem.")]
    regions = [_region(0)]
    rt = _ScriptedRuntime([None])  # endpoint-soft-None
    result = run_ocr_confusable_repair(regions, fbs, rt)
    assert result.edits_by_region == {}
    assert result.stats["windows_failed"] >= 1


def test_unflagged_region_index_rejected(monkeypatch):
    _on(monkeypatch)
    # Region 0 flagged (garbled); region 1 clean. A proposal targeting region 1
    # (not flagged / not in window) is rejected.
    fbs = [_fb("Solve Vx = 4 for x in this worked example problem statement."),
           _fb("A perfectly clean sentence with no garble at all here.")]
    regions = [_region(0), _region(1)]
    rt = _ScriptedRuntime(['[{"region_index":1,"before":"A","after":"√"}]'])
    result = run_ocr_confusable_repair(regions, fbs, rt)
    assert 1 not in result.edits_by_region
    reasons = {r.get("reason") for r in result.stats["rejected"]}
    assert "region_not_in_window" in reasons


# ---------------------------------------------------------------------------
# Stats bounds.
# ---------------------------------------------------------------------------


def test_stats_score_bounded(monkeypatch):
    _on(monkeypatch)
    fbs = [_fb("The quick brown fox jumps over the lazy sleeping dog today.")]
    regions = [_region(0)]
    rt = _ScriptedRuntime(["[]"])  # clean doc, no edits
    result = run_ocr_confusable_repair(regions, fbs, rt)
    score = result.stats["repair_stats_score"]
    assert 0.0 <= score <= _REPAIR_SCORE_CEILING


# ---------------------------------------------------------------------------
# Prompt builder / parser.
# ---------------------------------------------------------------------------


def test_prompt_builder_off_returns_empty(monkeypatch):
    monkeypatch.delenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", raising=False)
    assert build_ocr_repair_request({"regions": [{"region_index": 0, "text": "Vx"}]}) == ""


def test_prompt_builder_on_emits_system_user(monkeypatch):
    monkeypatch.setenv("SEMANTIK_OCR_CONFUSABLE_REPAIR", "on")
    prompt = build_ocr_repair_request(
        {"regions": [{"region_index": 3, "text": "Vx"}],
         "allowed_region_indices": [3]}
    )
    assert prompt.startswith("SYSTEM:")
    assert "USER:" in prompt
    assert '"region_index":3' in prompt


def test_parser_tolerant_and_fail_soft():
    assert parse_ocr_repair_response(None) == []
    assert parse_ocr_repair_response("garbage") == []
    edits = parse_ocr_repair_response(
        '```json\n[{"region_index":2,"before":"V","after":"√"}]\n```'
    )
    assert edits == [{"region_index": 2, "before": "V", "after": "√"}]
    # Missing keys dropped.
    assert parse_ocr_repair_response('[{"region_index":2}]') == []
