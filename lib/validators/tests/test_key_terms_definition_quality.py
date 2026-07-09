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


def test_faq_marker_vocab_card_not_audited_as_glossary():
    """A FAQ card reuses the ``vocab_card`` wrapper but carries the ``faq``
    marker — it must NOT be audited as a glossary term/definition (the two
    deterministic card families would otherwise alias on ``block_type``)."""
    faq = {
        "block_type": "vocab_card",
        "template_type": "faq",
        "block_id": "f1",
        # A Q/A "definition" that WOULD trip the circular + too-long checks if
        # it were (wrongly) treated as a glossary definition.
        "display": "What is a fraction?",
        "definition": (
            "A fraction is a fraction that represents a fraction of a whole; "
            "it is written with a numerator over a denominator and describes "
            "how many equal parts of a whole are being counted in total."
        ),
    }
    res = _run([faq])
    assert res.metadata["key_terms_definition_quality"]["cards_audited"] == 0
    assert res.passed is True
    assert not res.issues


def test_plain_vocab_card_without_marker_still_audited():
    """A bare ``vocab_card`` with NO template marker (legacy / planner-emitted
    glossary card) stays in scope — the FAQ exclusion is marker-specific."""
    block = _card("Prime", "A prime is a prime number.")  # circular
    res = _run([block])
    assert res.metadata["key_terms_definition_quality"]["cards_audited"] == 1
    assert any(i.code == "KEYTERM_DEF_CIRCULAR" for i in res.issues)


# -------------------------------------------------- missing math condition ---
def test_missing_nonzero_condition_flagged():
    # A quotient definition with no nonzero-denominator side-condition.
    block = _card("Quotient", "The quotient of one quantity divided by another.")
    res = _run([block])
    assert res.action == "regenerate"
    codes = [i.code for i in res.issues if i.code == "KEYTERM_DEF_MISSING_CONDITION"]
    assert codes == ["KEYTERM_DEF_MISSING_CONDITION"]
    msg = next(
        i.message for i in res.issues if i.code == "KEYTERM_DEF_MISSING_CONDITION"
    )
    assert "nonzero-denominator" in msg


def test_present_nonzero_condition_passes():
    block = _card(
        "Quotient",
        "The result of dividing a by b, where the denominator b is nonzero.",
    )
    res = _run([block])
    assert not any(
        i.code == "KEYTERM_DEF_MISSING_CONDITION" for i in res.issues
    )


def test_missing_nonzero_condition_operator_form_passes():
    # The '!= 0' / '≠ 0' operator forms satisfy the side-condition.
    block = _card("Ratio", "The value a / b for b != 0.")
    res = _run([block])
    assert not any(
        i.code == "KEYTERM_DEF_MISSING_CONDITION" for i in res.issues
    )


def test_slope_missing_distinctness_flagged():
    block = _card(
        "Slope",
        "The slope is the change in y divided by the change in x, "
        "computed as (y2 - y1) / (x2 - x1).",
    )
    res = _run([block])
    msg = next(
        (i.message for i in res.issues if i.code == "KEYTERM_DEF_MISSING_CONDITION"),
        None,
    )
    assert msg is not None
    assert "distinct-points" in msg


def test_slope_present_distinctness_passes():
    block = _card(
        "Slope",
        "The slope of a line through two distinct points is "
        "(y2 - y1) / (x2 - x1) where x2 ≠ x1.",
    )
    res = _run([block])
    assert not any(
        i.code == "KEYTERM_DEF_MISSING_CONDITION" for i in res.issues
    )


def test_non_ratio_definition_no_condition_flag():
    block = _card("Factor", "A whole number that divides another exactly.")
    res = _run([block])
    assert not any(
        i.code == "KEYTERM_DEF_MISSING_CONDITION" for i in res.issues
    )


# -------------------------------------------------- definition-box widening --
_DEF_BOX = (
    '<section data-cf-content-type="explanation"><h2>Fractions</h2>'
    '<p>Some prose introducing the idea.</p>'
    '<div class="definition-box"><strong>{term}</strong> {definition}</div>'
    '</section>'
)


def _concept_block(term, definition, *, block_id="cx", block_type="concept"):
    return {
        "block_type": block_type,
        "block_id": block_id,
        "content": _DEF_BOX.format(term=term, definition=definition),
    }


def test_definition_box_extracted_from_concept_block():
    block = _concept_block(
        "Rational number",
        "A number expressible as a quotient a/b of two integers with b nonzero.",
        block_id="week_01_content_02",
    )
    res = _run([block])
    meta = res.metadata["key_terms_definition_quality"]
    assert meta["cards_audited"] == 1
    # The synthetic unit is keyed off the parent block id (actionable via
    # --block-ids).
    per_block = meta["per_block"]
    unit_key = next(iter(per_block))
    assert unit_key.startswith("week_01_content_02#definition_box")
    assert per_block[unit_key]["location"] == "week_01_content_02"


def test_definition_box_circular_check_fires_via_widening():
    # Proves the widened units feed the EXISTING circular check.
    block = _concept_block(
        "Slope",
        "The slope is the slope of a straight line.",
        block_id="week_02_content_01",
    )
    res = _run([block])
    circular = [
        i for i in res.issues if i.code == "KEYTERM_DEF_CIRCULAR"
    ]
    assert len(circular) == 1
    assert circular[0].location == "week_02_content_01"


def test_definition_box_missing_condition_fires_via_widening():
    block = _concept_block(
        "Quotient",
        "The quotient of one number divided by another.",
        block_id="week_03_content_01",
    )
    res = _run([block])
    assert any(
        i.code == "KEYTERM_DEF_MISSING_CONDITION"
        and i.location == "week_03_content_01"
        for i in res.issues
    )


def test_explanation_block_without_definition_box_is_noop():
    block = {
        "block_type": "explanation",
        "block_id": "e1",
        "content": "<section><h2>Intro</h2><p>Plain prose only.</p></section>",
    }
    res = _run([block])
    assert res.metadata["key_terms_definition_quality"]["cards_audited"] == 0
    assert not res.issues


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
