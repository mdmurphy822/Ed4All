"""Phase-0 acceptance — reviewer prompt builder (no model / GPU).

Asserts the structure-only / no-rewrite / JSON-only mandate, the ABSENCE
of the Stage-6 generation envelope string, per-kind WCAG rule dispatch,
and exemplar shapes matching the deterministic emitter (`ontology_map.py`).
See plan §9 Phase 0 acceptance + §5/§6.
"""

from __future__ import annotations

import json

from dart_semantic.qwen_specialists.reviewer_prompt import (
    NEIGHBOR_SNIPPET_CHARS,
    _EXEMPLARS,
    _SYSTEM_REVIEWER,
    _VERIFIER_FEEDBACK_DIRECTIVE,
    build_reviewer_request,
    build_windowed_reviewer_request,
)
from dart_semantic.structure_graph import Region


def _heading_region(text: str = "Section One", level: int = 2) -> Region:
    return Region(
        kind="heading",
        feature_block_indices=(0,),
        payload={"level_hint": level, "text": text, "confidence": 0.81},
    )


def _table_region() -> Region:
    return Region(
        kind="table",
        feature_block_indices=(3, 4),
        payload={"text": "a b c"},
        source_region_id=7,
    )


# ---------------------------------------------------------------------------
# Content-type RE-TYPING directive (Phase-6 fix). The CONTENT single-block
# prompt and the WINDOWED prompt now instruct the model to return corrected_kind
# for a mislabeled content block; the HEADING single-block prompt is byte-stable.
# ---------------------------------------------------------------------------

_RETYPE_SIGNATURE = "CONTENT-TYPE RE-TYPING"


def _code_region(text: str = "TRY IT : : 1.27 3 7") -> Region:
    return Region(
        kind="code_block",
        feature_block_indices=(0,),
        payload={"text": text},
    )


def test_content_single_block_prompt_has_retype_instruction():
    prompt = build_reviewer_request(_code_region(), (None, None), 0)
    low = prompt.lower()
    # Names the content-type re-typing job + corrected_kind.
    assert _RETYPE_SIGNATURE in prompt
    assert "corrected_kind" in prompt
    # The code_block -> paragraph exercise example is present.
    assert "code_block" in low and "paragraph" in low
    assert "try it" in low
    assert "exercise" in low
    # Honesty: it is a KIND judgment, not a heading judgment, and still no rewrite.
    assert "content-type judgment" in low
    assert "no 'exercise' kind" in low
    # Only REAL REGION_KINDS are offered (a definition list re-type example).
    assert "definition_list" in low


def test_windowed_prompt_has_retype_instruction():
    records = [
        {"idx": 0, "council_kind": "code_block", "role": "code_block",
         "text": "TRY IT : : 1.27 3 7", "n_tokens": 6, "dup_count": 1},
        {"idx": 1, "council_kind": "table", "role": "table",
         "text": "Commutative Property: a + b = b + a", "n_tokens": 7, "dup_count": 1},
    ]
    prompt = build_windowed_reviewer_request(records)
    low = prompt.lower()
    assert _RETYPE_SIGNATURE in prompt
    assert "corrected_kind" in prompt
    # code_block -> paragraph + table -> definition_list/paragraph examples.
    assert "try it" in low
    assert "definition_list" in low
    assert "no 'exercise' kind" in low


def test_heading_single_block_prompt_byte_stable():
    # The heading single-block prompt must NOT carry the content-type re-typing
    # directive (the legacy byte-stable heading path). Snapshot the heading
    # system: it is exactly the 4 legacy parts, with no retype directive.
    prompt = build_reviewer_request(_heading_region(), (None, None), 0)
    assert _RETYPE_SIGNATURE not in prompt
    # The system turn is the legacy assembly (system + WCAG + heading rule +
    # exemplar) — the retype directive's distinctive examples are absent.
    assert "TRY IT" not in prompt
    assert "council_kind" not in prompt


# ---------------------------------------------------------------------------
# No-rewrite mandate present.
# ---------------------------------------------------------------------------


def test_no_rewrite_mandate_present():
    prompt = build_reviewer_request(_heading_region(), (None, None), 0)
    low = prompt.lower()
    assert "never rewrite" in low or "must never rewrite" in low
    # verbatim text invariant stated.
    assert "verbatim" in low
    # the four forbidden operations.
    for op in ("rewrite", "paraphrase", "add", "remove"):
        assert op in low, f"missing forbidden op: {op}"


# ---------------------------------------------------------------------------
# The generation "convert … fragment" envelope string is ABSENT.
# ---------------------------------------------------------------------------


def test_generation_envelope_absent():
    prompt = build_reviewer_request(_heading_region(), (None, None), 0)
    low = prompt.lower()
    # The Stage-6 _ENVELOPE_DIRECTIVE's signature phrasing must not appear.
    assert "convert the single" not in low
    assert "into one accessible html5 fragment" not in low
    assert "document-conversion specialist" not in low
    # The system directive itself must not carry the envelope.
    assert "convert" not in _SYSTEM_REVIEWER.lower() or "fragment" not in _SYSTEM_REVIEWER.lower()


# ---------------------------------------------------------------------------
# JSON-only mandate present.
# ---------------------------------------------------------------------------


def test_json_only_mandate_present():
    prompt = build_reviewer_request(_heading_region(), (None, None), 0)
    low = prompt.lower()
    assert "json only" in low or "return exactly one json object" in low
    assert "no code fences" in low or "no markdown" in low
    assert "no commentary" in low


# ---------------------------------------------------------------------------
# Per-kind rule dispatch — a heading prompt carries the heading rule and NOT
# the table H43 text; a table prompt carries the table rule.
# ---------------------------------------------------------------------------


def test_per_kind_rule_dispatch_heading():
    prompt = build_reviewer_request(_heading_region(), (None, None), 0)
    low = prompt.lower()
    assert "real heading text" in low
    # heading prompt must NOT carry the table-specific H43 contract.
    assert "h43" not in low
    assert "scope in {col,row" not in low


def test_per_kind_rule_dispatch_table():
    prompt = build_reviewer_request(_table_region(), (None, None), 5)
    low = prompt.lower()
    assert "scope in {col,row" in low or "th> carries scope" in low
    assert "h43" in low
    # table prompt must NOT carry the math allowlist rule.
    assert "50-element allowlist" not in low


def test_per_kind_rule_dispatch_math():
    region = Region(kind="math", feature_block_indices=(1,), payload={"text": "x=2"})
    prompt = build_reviewer_request(region, (None, None), 1)
    low = prompt.lower()
    assert "mathml" in low
    assert "never an image" in low
    assert "0.90" in prompt  # math coverage threshold


# ---------------------------------------------------------------------------
# Exemplar shapes match the deterministic emitter element choices.
# ---------------------------------------------------------------------------


def test_exemplar_shapes_match_ontology_map():
    # table exemplar uses <th scope="col"> (ontology_map _emit_cell shape).
    assert 'scope="col"' in _EXEMPLARS["table"]
    assert "<caption>" in _EXEMPLARS["table"]
    # math exemplar uses <math alttext=...> (mathml gate shape).
    assert "alttext=" in _EXEMPLARS["math"]
    assert "Math/MathML" in _EXEMPLARS["math"]
    # figure exemplar is text-only <figure><figcaption> (no <img src="">).
    assert "<figure>" in _EXEMPLARS["figure"]
    assert "<figcaption>" in _EXEMPLARS["figure"]
    assert "<img" not in _EXEMPLARS["figure"]


def test_exemplar_embedded_in_prompt():
    prompt = build_reviewer_request(_table_region(), (None, None), 5)
    assert 'scope="col"' in prompt


# ---------------------------------------------------------------------------
# USER turn carries the §3 input JSON (block_id, current_kind, text, level,
# neighbors); neighbor snippets are truncated.
# ---------------------------------------------------------------------------


def _split_user(prompt: str) -> dict:
    marker = "\nUSER:"
    idx = prompt.find(marker)
    assert idx != -1
    return json.loads(prompt[idx + len(marker):].strip())


def test_user_json_carries_input_contract():
    region = _heading_region(text="Answer Key 3.2", level=2)
    prompt = build_reviewer_request(region, (None, None), 42)
    obj = _split_user(prompt)
    assert obj["block_id"] == 42
    assert obj["current_kind"] == "heading"
    assert obj["current_level"] == 2
    assert obj["text"] == "Answer Key 3.2"
    assert obj["heading_confidence"] == 0.81
    assert obj["prev_block"] is None
    assert obj["next_block"] is None


def test_neighbor_snippets_present_and_truncated():
    long_text = "x " * 200  # 400 chars
    prev = Region(kind="paragraph", feature_block_indices=(0,), payload={"text": long_text})
    nxt = Region(kind="paragraph", feature_block_indices=(2,), payload={"text": "next para"})
    region = _heading_region()
    prompt = build_reviewer_request(region, (prev, nxt), 1)
    obj = _split_user(prompt)
    assert obj["prev_block"]["kind"] == "paragraph"
    assert len(obj["prev_block"]["text_snippet"]) <= NEIGHBOR_SNIPPET_CHARS
    assert obj["next_block"]["text_snippet"] == "next para"


def test_explicit_text_override_used_for_m2():
    # The reviewer.py caller passes the m2-resolved text explicitly.
    region = Region(kind="paragraph", feature_block_indices=(0,), payload={"text": "stale"})
    prompt = build_reviewer_request(region, (None, None), 0, text="resolved verbatim")
    obj = _split_user(prompt)
    assert obj["text"] == "resolved verbatim"


# ---------------------------------------------------------------------------
# Cluster-level signals carried in the input JSON (Phase-4 root-cause fix).
# ---------------------------------------------------------------------------


class _Sigs:
    """Duck-typed ClusterSignals stand-in (the prompt builder reads by attr)."""

    def __init__(self, run_len, pos, following, pagenum):
        self.same_level_run_len = run_len
        self.run_position = pos
        self.content_blocks_following = following
        self.trailing_pagenum = pagenum


def test_input_json_carries_cluster_signals():
    region = _heading_region(text="Chapter 5: Polynomials")
    sigs = _Sigs(8, 3, 0, True)
    prompt = build_reviewer_request(region, (None, None), 7, cluster_signals=sigs)
    obj = _split_user(prompt)
    assert obj["same_level_run_len"] == 8
    assert obj["run_position"] == 3
    assert obj["content_blocks_following"] == 0
    assert obj["trailing_pagenum"] is True


def test_input_json_cluster_signal_defaults_when_omitted():
    # A caller that passes no cluster_signals still gets the four keys (inert).
    region = _heading_region()
    prompt = build_reviewer_request(region, (None, None), 0)
    obj = _split_user(prompt)
    assert obj["same_level_run_len"] == 0
    assert obj["run_position"] == 0
    assert obj["content_blocks_following"] == 0
    assert obj["trailing_pagenum"] is False


def test_system_directive_carries_cluster_signals_as_informational():
    # The four cluster-signal NAMES are still named in the prompt (they remain
    # informational input the model can read), but they are framed as context,
    # NOT an auto-demote rule.
    prompt = build_reviewer_request(_heading_region(), (None, None), 0)
    low = prompt.lower()
    assert "same_level_run_len" in low
    assert "content_blocks_following" in low
    assert "run_position" in low
    assert "trailing_pagenum" in low
    # The signals are explicitly framed as informational context only.
    assert "informational context only" in low


def test_system_directive_is_conservative_not_mass_demote():
    # The conservative posture is stated: keep genuine headings even in runs;
    # re-tag only on the basis of the block's OWN text being non-heading.
    prompt = build_reviewer_request(_heading_region(), (None, None), 0)
    low = prompt.lower()
    # Conservative default is stated.
    assert "conservative" in low
    assert "do not demote a heading merely because it sits in a run" in low
    # The aggressive auto-demote mandate is GONE: the old prompt mandated that
    # a >=3 same-level run with content_blocks_following == 0 is "almost
    # certainly" a TOC -> metadata_drop. That sentence must no longer appear.
    assert "almost certainly a table-of-contents" not in low
    assert "weigh these structural signals above the local neighbor" not in low
    # And it must NOT instruct keeping/dropping purely on content_blocks_following.
    assert "content_blocks_following >= 1 is a real section" not in low


# ---------------------------------------------------------------------------
# Phase 5 — gold accessible semantic-class menu + optional semantic_class key +
# EXAMPLE-is-a-component-not-heading rule, gated on SEMANTIK_SEMANTIC_CLASS.
# Flag-off byte-identical is the load-bearing invariant.
# ---------------------------------------------------------------------------

_MENU_SIGNATURE = "AVAILABLE SEMANTIC COMPONENTS"


def _window_records():
    return [
        {"idx": 0, "council_kind": "code_block", "role": "code_block",
         "text": "TRY IT : : 1.27 3 7", "n_tokens": 6, "dup_count": 1},
        {"idx": 1, "council_kind": "table", "role": "table",
         "text": "Commutative Property: a + b = b + a", "n_tokens": 7, "dup_count": 1},
    ]


def test_menu_injected_when_flag_on(monkeypatch):
    from dart_semantic.assembler.semantic_catalog import valid_components

    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    content = build_reviewer_request(_code_region(), (None, None), 0)
    windowed = build_windowed_reviewer_request(_window_records())
    tokens = valid_components()
    assert tokens  # catalog is non-empty
    for prompt in (content, windowed):
        assert _MENU_SIGNATURE in prompt
        for token in tokens:
            assert token in prompt, f"menu missing component token: {token}"


def test_prompt_says_always_assign(monkeypatch):
    # Change 2: the directive now instructs the model to ALWAYS assign a
    # semantic_class for every content block (not the old MAY/optional framing).
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    from dart_semantic.qwen_specialists import reviewer_prompt as rp

    directive = rp._semantic_class_directive()
    low = directive.lower()
    # Always-assign instruction present.
    assert "must assign" in low
    assert "for every content block" in low
    # The old "you MAY assign ... optional key" framing is gone.
    assert "you may assign" not in low
    assert "as an optional key" not in low
    # The dispatched-weak-kind nudge is present (the under-firing fix).
    assert "code_block" in directive and "table" in directive


def test_prompt_byte_identical_when_flag_off(monkeypatch):
    # Capture the flag-off baseline for BOTH the content single-block builder
    # and the windowed builder, then toggle the flag ON and back OFF and assert
    # the OFF output is byte-identical across the round-trip (no global state
    # leaks) and never carries any semantic-class artifact.
    monkeypatch.delenv("SEMANTIK_SEMANTIC_CLASS", raising=False)
    off_content = build_reviewer_request(_code_region(), (None, None), 0)
    off_windowed = build_windowed_reviewer_request(_window_records())
    off_heading = build_reviewer_request(_heading_region(), (None, None), 0)

    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    on_content = build_reviewer_request(_code_region(), (None, None), 0)
    on_windowed = build_windowed_reviewer_request(_window_records())
    on_heading = build_reviewer_request(_heading_region(), (None, None), 0)

    monkeypatch.delenv("SEMANTIK_SEMANTIC_CLASS", raising=False)
    off_content_again = build_reviewer_request(_code_region(), (None, None), 0)
    off_windowed_again = build_windowed_reviewer_request(_window_records())

    # Byte-identical flag-off across the on->off round trip.
    assert off_content == off_content_again
    assert off_windowed == off_windowed_again
    # No semantic-class artifacts leak into the flag-off prompts.
    for prompt in (off_content, off_windowed):
        assert _MENU_SIGNATURE not in prompt
        assert "semantic_class" not in prompt
    # Flag-on genuinely DIFFERS for the content + windowed builders.
    assert on_content != off_content
    assert on_windowed != off_windowed
    # The HEADING single-block prompt is byte-identical regardless of the flag
    # (it never reaches the content branch).
    assert on_heading == off_heading
    assert _MENU_SIGNATURE not in on_heading
    assert "semantic_class" not in on_heading


def test_example_rule_present(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    content = build_reviewer_request(_code_region(), (None, None), 0)
    low = content.lower()
    # The EXAMPLE -> worked_example mapping + verdict.
    assert "example 1.3" in low
    assert "worked_example" in content
    assert 'verdict "corrected"' in content
    # The NEVER-heading rule (an example is a component, not a heading).
    assert "an example is a component, not a heading" in low
    assert 'never corrected_kind "heading"' in low


def test_semantic_class_key_in_both_contracts(monkeypatch):
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    content = build_reviewer_request(_code_region(), (None, None), 0)
    windowed = build_windowed_reviewer_request(_window_records())
    # semantic_class documented as an OPTIONAL key on BOTH review surfaces.
    for prompt in (content, windowed):
        assert "semantic_class" in prompt
        assert "optional" in prompt.lower()
        assert "from the menu, or null" in prompt


def test_no_envelope_json_only_still_holds(monkeypatch):
    # The existing no-envelope / JSON-only mandate must still hold with the
    # menu injected (flag on) on the windowed builder.
    monkeypatch.setenv("SEMANTIK_SEMANTIC_CLASS", "1")
    prompt = build_windowed_reviewer_request(_window_records())
    low = prompt.lower()
    assert "into one accessible html5 fragment" not in low
    assert "document-conversion specialist" not in low
    assert "convert the single" not in low
    # JSON-array-only mandate is intact.
    assert "json array only" in low
    assert "no code fences" in low or "no markdown" in low


# ---------------------------------------------------------------------------
# Phase 5 — verifier-feedback injection (targeted Pass-1 re-drive). The
# single-block builder appends _VERIFIER_FEEDBACK_DIRECTIVE; the windowed
# builder stamps a DISTINCT review_feedback record key (no reserved-key
# collision). feedback_by_idx=None / no-matching-index is byte-identical.
# ---------------------------------------------------------------------------

_RESERVED_RECORD_KEYS = {
    "idx", "council_kind", "role", "confidence", "page", "n_tokens",
    "dup_count", "text", "head", "tail", "pages",
}


def test_feedback_injected_single_block():
    base = build_reviewer_request(_code_region(), (None, None), 3)
    fed = build_reviewer_request(
        _code_region(), (None, None), 3,
        feedback_by_idx={3: "re-type to worked_example"},
    )
    assert _VERIFIER_FEEDBACK_DIRECTIVE in fed
    assert "re-type to worked_example" in fed
    # No matching index -> byte-identical to the no-feedback baseline.
    other = build_reviewer_request(
        _code_region(), (None, None), 3, feedback_by_idx={9: "irrelevant"},
    )
    assert other == base
    assert _VERIFIER_FEEDBACK_DIRECTIVE not in base


def test_feedback_injected_windowed_record():
    records = _window_records()
    prompt = build_windowed_reviewer_request(
        records, feedback_by_idx={1: "mis-routed table -> definition_list"},
    )
    # The feedback travels in the USER JSON next to idx 1 (review_feedback key).
    user = prompt.split("\nUSER: ", 1)[1]
    arr = json.loads(user)
    by_idx = {r["idx"]: r for r in arr}
    assert by_idx[1]["review_feedback"] == "mis-routed table -> definition_list"
    # An un-fed member carries NO review_feedback key.
    assert "review_feedback" not in by_idx[0]
    # review_feedback is a DISTINCT key — never a reserved edge-record key.
    assert "review_feedback" not in _RESERVED_RECORD_KEYS
    # No-feedback windowed prompt is byte-identical (no stray key).
    assert build_windowed_reviewer_request(_window_records()) == \
        build_windowed_reviewer_request(_window_records(), feedback_by_idx=None)


# ---------------------------------------------------------------------------
# ITEM6 — windowed prompt appends the council-top-k directive ONLY when a record
# carries council_top_k; single-block (legacy heading) path stays byte-stable.
# ---------------------------------------------------------------------------

from dart_semantic.qwen_specialists.reviewer_prompt import _COUNCIL_TOPK_DIRECTIVE  # noqa: E402


def test_windowed_prompt_appends_topk_directive_only_when_present():
    with_key = [
        {"idx": 0, "council_kind": "code_block", "role": "code_block",
         "text": "x = 1", "n_tokens": 3, "dup_count": 1,
         "council_top_k": [["code_block", 0.51], ["paragraph", 0.3]]},
        {"idx": 1, "council_kind": "paragraph", "role": "paragraph",
         "text": "ordinary prose", "n_tokens": 2, "dup_count": 1},
    ]
    prompt = build_windowed_reviewer_request(with_key)
    assert _COUNCIL_TOPK_DIRECTIVE in prompt
    # The record JSON is serialized verbatim (the additive key rides in USER).
    assert "council_top_k" in prompt


def test_windowed_prompt_no_topk_directive_when_absent():
    no_key = [
        {"idx": 0, "council_kind": "code_block", "role": "code_block",
         "text": "x = 1", "n_tokens": 3, "dup_count": 1},
        {"idx": 1, "council_kind": "table", "role": "table",
         "text": "a b c", "n_tokens": 3, "dup_count": 1},
    ]
    prompt = build_windowed_reviewer_request(no_key)
    assert _COUNCIL_TOPK_DIRECTIVE not in prompt


def test_single_block_prompt_unchanged():
    # The legacy heading single-block builder never carries the council-top-k
    # directive (byte-stable legacy path).
    prompt = build_reviewer_request(_heading_region(), (None, None), 0)
    assert _COUNCIL_TOPK_DIRECTIVE not in prompt
