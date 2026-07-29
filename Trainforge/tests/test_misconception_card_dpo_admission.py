"""Authored misconception cards must reach DPO admission.

These tests drive the PRODUCTION seam chain — ``_focus_chunk_on_objective``
then ``_pair_eligibility_for_mode`` — not ``micro_preference_eligibility``
directly.  A leaf-level test cannot observe the routing, because preference
admission is only reached on a chunk the focus seam already focused, and the
two seams have disagreed before.

The behaviour under test: an authored misconception card carries its
(claim, correction) split in its own markup.  The evidence window collapses a
card into one block, so the split has to be recovered structurally; the legacy
cue-phrase pattern recognises only three fixed openers and authored prose has
no fixed opener.
"""

from __future__ import annotations

import pytest

from Trainforge.generators.synthesis_window_contract import build_evidence_window
from Trainforge.synthesize_training import (
    _focus_chunk_on_objective,
    _pair_eligibility_for_mode,
)

_OBJECTIVE_ID = "co-01"

_OBJECTIVES = {
    _OBJECTIVE_ID: {
        "id": "CO-01",
        "statement": "Distinguish factors from multiples of a whole number.",
        "bloom_level": "understand",
    },
}

_PROSE = (
    "A factor of a whole number divides that number evenly with no "
    "remainder, while a multiple of a whole number is the product of that "
    "number and a counting number. For the number twelve, the factors are "
    "one, two, three, four, six, and twelve, because each divides twelve "
    "evenly. The multiples of twelve are twelve, twenty-four, thirty-six, "
    "and so on, because each is twelve multiplied by a counting number. "
    "Distinguishing factors from multiples of a whole number keeps the two "
    "ideas separate when a problem names both."
)


def _card(
    block_id: str,
    claim: str,
    correction: str,
    *,
    heading: str = "Common Misconception",
) -> str:
    return (
        f'<div class="misconception-card" data-cf-block-id="{block_id}" '
        f'data-cf-objective-id="CO-01">'
        f"<h2>{heading}</h2>"
        f'<p class="misconception-claim">{claim}</p>'
        f'<p class="misconception-correction">{correction}</p>'
        f"</div>"
    )


def _chunk(html_body: str, *, chunk_id: str = "chunk-0001") -> dict:
    return {
        "id": chunk_id,
        "text": _PROSE,
        "html": (
            '<div data-cf-block-id="week_01_content_01#explanation_factors_1" '
            'data-cf-objective-id="CO-01">'
            f"<p>{_PROSE}</p></div>" + html_body
        ),
        "chunk_type": "explanation",
        "concept_tags": ["factor", "multiple"],
        "learning_outcome_refs": ["CO-01"],
        "bloom_level": "understand",
    }


@pytest.fixture(autouse=True)
def _staged_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preference admission is only reachable under a staged contract."""
    monkeypatch.setenv("TRAINFORGE_STAGED_SYNTHESIS_V4", "1")
    monkeypatch.delenv("TRAINFORGE_STAGED_SYNTHESIS_MICRO_V1", raising=False)


def _verdict(chunk: dict, kind: str):
    focused = _focus_chunk_on_objective(chunk, seed=17, objectives=_OBJECTIVES)
    return _pair_eligibility_for_mode(focused, kind=kind)


# The legacy pattern requires "students may incorrectly BELIEVE THAT"; authored
# cards use whatever verb the content calls for.
_NON_CUE_CLAIM = (
    "Students often confuse a factor of a whole number with a multiple of "
    "that same whole number."
)
_CORRECTION = (
    "A factor divides the whole number evenly, while a multiple is the "
    "product of that whole number and a counting number."
)


def test_card_without_a_cue_phrase_opener_is_preference_eligible() -> None:
    verdict = _verdict(
        _chunk(_card("week_01_content_01#misconception_factors_1",
                     _NON_CUE_CLAIM, _CORRECTION)),
        "preference",
    )
    assert verdict.eligible, verdict.reason


def test_card_correction_without_a_mechanism_token_is_still_eligible() -> None:
    """Arm A accepts any non-empty correction as mechanism evidence.

    This arm used to additionally require one of
    correct/rather/instead/means/not/false/true, which rejected correct
    corrective prose that happened to phrase itself differently.
    """
    correction = (
        "A factor divides the whole number evenly, while a multiple is the "
        "product of that whole number and a counting number."
    )
    assert not any(
        token in correction.lower().split()
        for token in ("correct", "rather", "instead", "means", "false", "true")
    )
    assert " not " not in correction.lower()
    verdict = _verdict(
        _chunk(_card("week_01_content_01#misconception_factors_1",
                     _NON_CUE_CLAIM, correction)),
        "preference",
    )
    assert verdict.eligible, verdict.reason


def test_two_identical_cards_in_one_chunk_still_zero_the_chunk() -> None:
    """The duplicate-id invariant is preserved, not relaxed."""
    body = (
        _card("week_01_content_01#misconception_factors_1",
              _NON_CUE_CLAIM, _CORRECTION)
        + _card("week_01_content_01#misconception_factors_2",
                _NON_CUE_CLAIM, _CORRECTION, heading="Watch Out")
    )
    verdict = _verdict(_chunk(body), "preference")
    assert not verdict.eligible
    assert verdict.reason == "preference_misconception_id_collision"


def test_two_distinct_cards_in_one_chunk_both_admit() -> None:
    body = (
        _card("week_01_content_01#misconception_factors_1",
              _NON_CUE_CLAIM, _CORRECTION)
        + _card(
            "week_01_content_01#misconception_factors_2",
            "Learners sometimes read every multiple of a whole number as a "
            "factor of that whole number.",
            "Every whole number divides its own multiples, so the multiple "
            "is the larger product and the factor is the smaller divisor.",
            heading="Watch Out",
        )
    )
    verdict = _verdict(_chunk(body), "preference")
    assert verdict.eligible, verdict.reason


def test_instruction_eligibility_is_unchanged_by_the_card() -> None:
    """The structural recovery must not move the instruction-side verdict."""
    without = _verdict(_chunk(""), "instruction")
    with_card = _verdict(
        _chunk(_card("week_01_content_01#misconception_factors_1",
                     _NON_CUE_CLAIM, _CORRECTION)),
        "instruction",
    )
    assert without.eligible
    assert with_card.eligible


def test_legacy_cue_phrase_block_without_card_markup_still_admits() -> None:
    """Corpora whose blocks carry no structural split keep the old path."""
    legacy = (
        '<div data-cf-block-id="week_01_content_01#misconception_factors_1" '
        'data-cf-objective-id="CO-01"><p>'
        "A common misconception is that every multiple of a whole number is "
        "also a factor of that whole number. A factor divides the whole "
        "number evenly, while a multiple is the product of that whole number "
        "and a counting number."
        "</p></div>"
    )
    verdict = _verdict(_chunk(legacy), "preference")
    assert verdict.eligible, verdict.reason


def test_window_block_shape_is_additive_only() -> None:
    """role / polarity / text / evidence_text must not move."""
    chunk = _chunk(
        _card("week_01_content_01#misconception_factors_1",
              _NON_CUE_CLAIM, _CORRECTION)
    )
    focused = _focus_chunk_on_objective(chunk, seed=17, objectives=_OBJECTIVES)
    window = build_evidence_window(
        focused, focused["synthesis_focus_objective"],
    )
    card_blocks = [
        block for block in window["blocks"]
        if "misconception" in block["block_id"]
    ]
    assert len(card_blocks) == 1
    block = card_blocks[0]
    # The card is still ONE block carrying the whole card text at the same
    # role/polarity, so it still contributes to evidence_text exactly as before.
    assert block["role"] == "evidence"
    assert block["polarity"] == "factual"
    assert _NON_CUE_CLAIM in block["text"]
    assert _CORRECTION in block["text"]
    assert block["misconception_pairs"] == [
        {"claim": _NON_CUE_CLAIM, "correction": _CORRECTION},
    ]


_STRUCTURED_CLAIM = "every multiple of a whole number is also a factor of it"
_STRUCTURED_CORRECTION = (
    "A factor divides the whole number evenly, while a multiple is the "
    "product of that whole number and a counting number."
)


def _alias_chunk(alias: str) -> dict:
    return {
        "id": "chunk-alias",
        "text": (
            "Learners sometimes say every multiple of a whole number is also "
            "a factor of it. " + _PROSE
        ),
        "chunk_type": "explanation",
        "concept_tags": ["factor", "multiple"],
        "learning_outcome_refs": ["CO-01"],
        "bloom_level": "understand",
        "misconceptions": [{
            alias: _STRUCTURED_CLAIM,
            "correction": _STRUCTURED_CORRECTION,
        }],
    }


def test_structured_misconception_arm_stays_reachable_on_a_flat_chunk() -> None:
    """The window's statement-only polarity guard is LOAD-BEARING.

    Arm A of ``micro_preference_eligibility`` requires the claim to be
    source-backed, so a structured DPO misconception is embedded in the
    chunk's own text by construction. If the flat-path polarity guard were
    made symmetric across the two schema-legal spellings, it would refuse the
    whole chunk for exactly the shape Arm A exists to consume.

    This test pins that Arm A remains reachable; a future "alias symmetry"
    cleanup that breaks it will fail here rather than silently zeroing the
    structured preference path.
    """
    verdict = _verdict(_alias_chunk("misconception"), "preference")
    assert verdict.eligible, verdict.reason


def test_statement_keyed_flat_chunk_is_refused_by_the_polarity_guard() -> None:
    """Documents the KNOWN residual asymmetry rather than hiding it.

    Under the ``statement`` spelling the same chunk is refused by the flat
    polarity guard, so Arm A never sees it. This is recorded as current
    behaviour, not endorsed: which spelling means "structured DPO source"
    versus "this prose embeds a known error" is an open owner decision.
    """
    verdict = _verdict(_alias_chunk("statement"), "preference")
    assert not verdict.eligible
    assert verdict.reason == "objective_evidence_window_not_viable"


def test_misconception_carrying_provenance_fields_validates_against_chunk_v4() -> None:
    """The eligibility arm's fields must be representable in the schema.

    $defs.Misconception is additionalProperties:false, so a misconception
    carrying the provenance/mechanism keys micro_preference_eligibility reads
    was schema-INVALID before those keys were declared.
    """
    import json
    from pathlib import Path

    import jsonschema

    schema_path = (
        Path(__file__).resolve().parents[2]
        / "schemas" / "knowledge" / "chunk_v4.schema.json"
    )
    schema = json.loads(schema_path.read_text())
    misconception_schema = dict(schema["$defs"]["Misconception"])
    misconception_schema["$defs"] = schema["$defs"]
    jsonschema.validate(
        {
            "misconception": _STRUCTURED_CLAIM,
            "correction": _STRUCTURED_CORRECTION,
            "mechanism_evidence": _STRUCTURED_CORRECTION,
            "error_mechanism": _STRUCTURED_CORRECTION,
            "source_block_id": "week_01_content_01#misconception_factors_1",
            "id": "mc_0123456789abcdef",
            "bloom_level": "understand",
        },
        misconception_schema,
    )


def test_card_with_no_structural_split_falls_back_not_crashes() -> None:
    """A misconception block with neither markup nor a cue phrase is rejected."""
    bare = (
        '<div data-cf-block-id="week_01_content_01#misconception_factors_1" '
        'data-cf-objective-id="CO-01"><p>'
        "Factors and multiples of a whole number are related but distinct "
        "ideas that learners meet together."
        "</p></div>"
    )
    verdict = _verdict(_chunk(bare), "preference")
    assert not verdict.eligible
    assert verdict.reason == "preference_misconception_candidate_missing"
