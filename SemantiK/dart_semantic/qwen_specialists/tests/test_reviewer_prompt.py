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
    build_reviewer_request,
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
