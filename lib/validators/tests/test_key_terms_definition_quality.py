"""W1.5 — KeyTermsDefinitionQualityValidator."""
from __future__ import annotations

from lib.generation.key_terms import render_term_card
from lib.validators.key_terms_definition_quality import (
    KeyTermsDefinitionQualityValidator,
)


def _card(term, definition, *, block_id="t1", block_type="vocab_card", **extra):
    d = {
        "block_type": block_type,
        "block_id": block_id,
        "display": term,
        "definition": definition,
    }
    d.update(extra)
    return d


def _run(blocks, **inputs):
    payload = {"blocks": blocks}
    payload.setdefault("keyterm_def_quality_enabled", True)
    payload.update(inputs)
    return KeyTermsDefinitionQualityValidator().validate(payload)


# ---------------------------------------------------------------- disabled ---
def test_disabled_is_noop_pass():
    block = _card("Prime", "A number with exactly two divisors.")
    res = KeyTermsDefinitionQualityValidator().validate(
        {"blocks": [block], "keyterm_def_quality_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"KEYTERM_DEF_QUALITY_DISABLED"}
    assert res.action is None
    assert res.metadata["key_terms_definition_quality"]["enabled"] is False


def test_disabled_via_env_default(monkeypatch):
    monkeypatch.delenv("ED4ALL_KEYTERM_DEF_QUALITY", raising=False)
    res = KeyTermsDefinitionQualityValidator().validate(
        {"blocks": [_card("Prime", "A number.")]}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"KEYTERM_DEF_QUALITY_DISABLED"}


# ------------------------------------------------------------------- clean ---
def test_clean_definitions_pass():
    blocks = [
        _card("Prime", "A whole number greater than one divisible only by one and itself.", block_id="a"),
        _card("Factor", "A whole number that divides another exactly.", block_id="b"),
    ]
    res = _run(blocks)
    assert res.passed is True
    assert res.action is None
    assert not res.issues
    meta = res.metadata["key_terms_definition_quality"]
    assert meta["cards_audited"] == 2
    assert meta["flagged"] == 0


# ---------------------------------------------------------------- circular ---
def test_circular_definition_flagged():
    block = _card("Prime number", "A prime number is a number that is prime.")
    res = _run([block])
    assert res.passed is True  # warning-day-1
    assert res.action == "regenerate"
    assert any(i.code == "KEYTERM_DEF_CIRCULAR" for i in res.issues)


def test_non_circular_not_flagged():
    block = _card("Prime", "A whole number divisible only by one and itself.")
    res = _run([block])
    assert not any(i.code == "KEYTERM_DEF_CIRCULAR" for i in res.issues)


# ---------------------------------------------------------------- too-long ---
def test_too_long_definition_flagged():
    long_def = "This term " + ("is described at great length. " * 40)
    block = _card("Widget", long_def)
    res = _run([block])
    assert any(i.code == "KEYTERM_DEF_TOO_LONG" for i in res.issues)
    assert res.metadata["key_terms_definition_quality"]["def_char_ceiling"] == 200


def test_body_char_ceiling_override():
    block = _card("Widget", "A small gadget used in examples throughout the text.")
    # A tight ceiling forces the overflow.
    res = _run([block], body_char_ceiling=10)
    assert any(i.code == "KEYTERM_DEF_TOO_LONG" for i in res.issues)


# ------------------------------------------------------------- not-distinct --
def test_shared_definition_flags_both():
    same = "A whole number used in arithmetic examples."
    blocks = [_card("Alpha", same, block_id="a"), _card("Beta", same, block_id="b")]
    res = _run(blocks)
    flagged_ids = {
        i.location for i in res.issues if i.code == "KEYTERM_DEF_NOT_DISTINCT"
    }
    assert flagged_ids == {"a", "b"}
    assert res.metadata["key_terms_definition_quality"]["shared_definition_groups"] == 1


def test_distinct_definitions_not_flagged_dupe():
    blocks = [
        _card("Alpha", "The first letter, used as a variable.", block_id="a"),
        _card("Beta", "The second letter, used as a variable name.", block_id="b"),
    ]
    res = _run(blocks)
    assert not any(i.code == "KEYTERM_DEF_NOT_DISTINCT" for i in res.issues)


# ------------------------------------------------------- rendered-HTML path --
def test_str_content_card_parsed():
    html = render_term_card(
        display="Prime number",
        definition="A prime number is a number that is prime.",
        source_link=None,
        slug="prime-number",
    )
    block = {
        "block_type": "vocab_card",
        "block_id": "h1",
        "template_type": "key_terms",
        "content": html,
    }
    res = _run([block])
    assert any(i.code == "KEYTERM_DEF_CIRCULAR" for i in res.issues)
    assert res.metadata["key_terms_definition_quality"]["cards_audited"] == 1


# ----------------------------------------------------- non-keyterm ignored ---
def test_non_key_terms_block_ignored():
    other = {"block_type": "concept", "block_id": "x", "content": "Some prose."}
    res = _run([other])
    assert res.metadata["key_terms_definition_quality"]["cards_audited"] == 0
    assert not res.issues


def test_template_type_marker_audits_non_vocab_card():
    block = {
        "block_type": "callout",
        "template_type": "key_terms",
        "block_id": "tt",
        "display": "Prime",
        "definition": "A prime is a prime thing.",  # circular
    }
    res = _run([block])
    assert res.metadata["key_terms_definition_quality"]["cards_audited"] == 1
    assert any(i.code == "KEYTERM_DEF_CIRCULAR" for i in res.issues)


# ------------------------------------------------------- decision capture ----
class _CaptureSpy:
    def __init__(self):
        self.calls = []

    def log_decision(self, **kwargs):
        self.calls.append(kwargs)


def test_decision_capture_fires_when_enabled():
    spy = _CaptureSpy()
    res = _run([_card("Prime", "A prime is prime.")], decision_capture=spy)
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["decision_type"] == "content_structure_check"
    assert "key_terms_definition_quality_flagged" in call["decision"]
    assert len(call["rationale"]) >= 20


def test_decision_capture_not_fired_when_disabled():
    spy = _CaptureSpy()
    KeyTermsDefinitionQualityValidator().validate(
        {
            "blocks": [_card("Prime", "A prime is prime.")],
            "keyterm_def_quality_enabled": False,
            "decision_capture": spy,
        }
    )
    assert spy.calls == []
