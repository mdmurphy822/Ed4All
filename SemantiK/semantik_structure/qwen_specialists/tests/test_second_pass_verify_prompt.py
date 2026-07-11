"""Phase 3 — Pass-2 verifier prompt builder + fail-soft parser (gated).

The verifier prompt takes the assembled-document digest and mandates a
JSON-ARRAY-ONLY, index-keyed, NEVER-text response; the parser is fail-soft
(``None`` on garbage). Built ONLY when ``SEMANTIK_SECOND_PASS`` is on, so the
flag-off path is byte-identical and the shipped Pass-1 constants are untouched.
"""

from __future__ import annotations

import json

from semantik_structure.qwen_specialists.reviewer_prompt import (
    _SYSTEM_REVIEWER,
    _WINDOW_OPLIST_DIRECTIVE,
    _CONTENT_RETYPE_DIRECTIVE,
    build_second_pass_verify_request,
    build_windowed_reviewer_request,
    parse_verifier_response,
)


def _digest() -> dict:
    return {
        "regions": [
            {"region_index": 0, "kind": "heading", "semantic_class": None,
             "aria_wrapper": None, "head": "EXAMPLE 1.3", "tail": "solve x"},
            {"region_index": 1, "kind": "paragraph", "semantic_class": None,
             "aria_wrapper": None, "head": "Body", "tail": "text"},
        ],
        "heading_outline": [
            {"region_index": 0, "level": 1, "text": "EXAMPLE 1.3", "heading_id": "example-1-3"},
        ],
    }


# ---------------------------------------------------------------------------
# Prompt builder (flag ON).
# ---------------------------------------------------------------------------


def test_verify_prompt_is_json_only_index_keyed(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    prompt = build_second_pass_verify_request(_digest())
    assert prompt
    low = prompt.lower()
    assert "json array only" in low
    assert "region_index" in prompt and "echo" in low
    # NEVER rewrite/return source text.
    assert "never rewrite" in low or "no rewritten text" in low
    # The digest rides the USER turn as JSON.
    assert "USER:" in prompt


def test_verify_prompt_lists_failure_mode_vocabulary(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    prompt = build_second_pass_verify_request(_digest())
    for mode in (
        "example_as_heading",
        "mistyped_component",
        "wrong_semantic_class",
        "under_firing",
        "no_context_answer_fragment_heading",
        "section_no_body",
        "example_misordered_from_body",
    ):
        assert mode in prompt
    # The deferred (fixable=false) modes are present + flagged as non-fixable.
    assert "fixable=false" in prompt
    # The spot-HTML re-ask vocabulary.
    assert "need_spot_html" in prompt


def test_verify_prompt_carries_spot_html_on_reask(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SECOND_PASS", "1")
    prompt = build_second_pass_verify_request(
        _digest(), spot_html_by_idx={3: "<section aria-labelledby='s'>...</section>"}
    )
    assert "spot_html" in prompt
    assert "aria-labelledby" in prompt


# ---------------------------------------------------------------------------
# Parser (fail-soft).
# ---------------------------------------------------------------------------


def test_parse_verifier_response_well_formed():
    raw = json.dumps([
        {"region_index": 0, "failure_mode": "example_as_heading",
         "fix_hint": "re-type to worked_example", "fixable": True},
        {"region_index": 5, "failure_mode": "section_no_body",
         "fix_hint": "merge", "fixable": False, "proposed_regroup_run": [5, 6]},
    ])
    verdict = parse_verifier_response(raw)
    assert verdict is not None
    assert verdict.passed is False
    assert len(verdict.flagged) == 2
    fb0, fb1 = verdict.flagged
    assert fb0.region_index == 0 and fb0.failure_mode == "example_as_heading"
    assert fb0.fixable is True
    assert fb1.fixable is False
    assert fb1.proposed_regroup_run == (5, 6)


def test_parse_verifier_response_empty_array_is_pass():
    verdict = parse_verifier_response("[]")
    assert verdict is not None
    assert verdict.passed is True
    assert verdict.flagged == ()


def test_parse_verifier_response_need_spot_html():
    raw = json.dumps([{"region_index": 3, "failure_mode": "need_spot_html"}])
    verdict = parse_verifier_response(raw)
    assert verdict is not None
    assert verdict.spot_html_requested == (3,)
    assert verdict.flagged == ()
    assert verdict.passed is False  # a spot request is on the path to FAIL


def test_parse_verifier_response_failsoft_none():
    for bad in ("", "   ", "not json at all", "{}", "null", None):
        assert parse_verifier_response(bad) is None


# ---------------------------------------------------------------------------
# Flag-off byte-stability — the verifier prompt is never built; the shipped
# Pass-1 constants + the windowed builder stay byte-identical.
# ---------------------------------------------------------------------------


def test_prompt_not_built_when_flag_off(monkeypatch):
    monkeypatch.delenv("SEMANTIK_SECOND_PASS", raising=False)
    assert build_second_pass_verify_request(_digest()) == ""
    # The shipped Pass-1 prompt constants are untouched (byte-stable).
    assert "Pass-2 VERIFIER" not in _SYSTEM_REVIEWER
    assert "Pass-2 VERIFIER" not in _WINDOW_OPLIST_DIRECTIVE
    assert "Pass-2 VERIFIER" not in _CONTENT_RETYPE_DIRECTIVE
    # And the windowed Pass-1 builder is unchanged by the new directive.
    records = [{"idx": 0, "council_kind": "paragraph", "role": "paragraph",
                "confidence": None, "page": 1, "n_tokens": 3, "text": "a b c"}]
    windowed = build_windowed_reviewer_request(records)
    assert "Pass-2 VERIFIER" not in windowed
    assert "need_spot_html" not in windowed
