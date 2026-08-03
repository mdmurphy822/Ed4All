"""7B→Sonnet parity P2: box vocabulary + teaching-role in rewrite contracts.

Asserts the per-block-type rewrite contracts instruct the Sonnet styled-box
classes, the system prompt references them, and the minted-CURIE
pedagogical-context flip behaves. No LLM, no pipeline run.
"""

import Courseforge.generators.rewrite._rewrite_provider as rp
import MCP.tools.pipeline_tools as pt


# ---------------------------------------------------------------------------
# P2 — box-class instructions in the per-block-type contracts
# ---------------------------------------------------------------------------

def test_example_contract_instructs_example_box():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["example"]
    assert "example-box" in contract
    assert "worked-example" in contract
    assert "step-label" in contract
    assert "solution-line" in contract


def test_concept_contract_instructs_definition_and_key_rule():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["concept"]
    assert "definition-box" in contract
    assert "key-rule" in contract


def test_explanation_contract_instructs_boxes():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["explanation"]
    assert "definition-box" in contract
    assert "key-rule" in contract
    assert "example-box" in contract


def test_self_check_contract_instructs_self_check_item():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["self_check_question"]
    assert "self-check-item" in contract


# ---------------------------------------------------------------------------
# P3 — instruction-block variety: the six previously-bare-<section> block
# types now instruct a distinct styled component class in their contract.
# ---------------------------------------------------------------------------

# block_type -> the distinct component class its rewrite contract must name.
_BLOCK_VARIETY_COMPONENT_CLASSES = {
    "summary_takeaway": "takeaway-card",
    "misconception": "misconception-card",
    "prereq_set": "prereq-card",
    "recap": "recap-box",
    "reflection_prompt": "reflection-prompt",
    "discussion_prompt": "discussion-prompt",
}


def test_block_variety_contracts_instruct_distinct_component_class():
    for block_type, css_class in _BLOCK_VARIETY_COMPONENT_CLASSES.items():
        contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS[block_type]
        assert css_class in contract, (
            f"{block_type} contract must instruct the {css_class!r} "
            "component class so it renders as a distinct styled card "
            f"(found: {contract[:80]!r}...)"
        )
        # The wrapper is a <div class="..."> mirroring scenario/vocab_card.
        assert f'<div class=\\"{css_class}\\"' in contract or (
            f'<div class="{css_class}"' in contract
        ), f"{block_type} should wrap in <div class=\"{css_class}\" ...>"


def test_misconception_contract_keeps_claim_correction_pattern():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["misconception"]
    # The JSON-LD id pattern guidance survives the wrapper addition.
    assert "mc_[0-9a-f]{16}" in contract
    assert "misconception-claim" in contract
    assert "misconception-correction" in contract


def test_system_prompt_references_styled_components():
    sp = rp._REWRITE_SYSTEM_PROMPT
    assert "example-box" in sp
    assert "definition-box" in sp
    assert "key-rule" in sp
    assert "worked-example" in sp


def test_system_prompt_no_longer_orders_verbatim_curie_echo():
    sp = rp._REWRITE_SYSTEM_PROMPT
    # The old "CURIEs (verbatim)" preserve order is gone.
    assert "CURIEs (verbatim)" not in sp
    # And there is an explicit instruction not to echo synthetic CURIE tokens.
    assert "prefix:localname" in sp


# ---------------------------------------------------------------------------
# P2 — data-cf-teaching-role on the section wrapper (block_type mapper)
# ---------------------------------------------------------------------------

def test_teaching_role_resolves_for_prose_block_types():
    assert pt._teaching_role_for_block_type("concept") == "introduce"
    assert pt._teaching_role_for_block_type("example") == "elaborate"
    assert pt._teaching_role_for_block_type("explanation") == "elaborate"
    assert pt._teaching_role_for_block_type("self_check_question") == "assess"
    assert pt._teaching_role_for_block_type("activity") == "transfer"


def test_teaching_role_empty_for_unmapped_block_type():
    assert pt._teaching_role_for_block_type("chrome") == ""
    assert pt._teaching_role_for_block_type("misconception") == ""


def test_teaching_role_in_canonical_enum():
    from lib.ontology.teaching_roles import get_valid_roles

    valid = get_valid_roles()
    for bt in ("concept", "example", "explanation", "self_check_question"):
        role = pt._teaching_role_for_block_type(bt)
        assert role in valid


# ---------------------------------------------------------------------------
# P0/P2 — minted CURIE pedagogical-context flip (code-voice no longer counts)
# ---------------------------------------------------------------------------

def test_minted_curie_code_voice_not_preserved():
    html = "<p>See <code>alg:integers</code> for details.</p>"
    assert rp._curie_in_pedagogical_context(html, "alg:integers", minted=True) is False


def test_rdf_curie_code_voice_still_preserved():
    html = "<p>The predicate <code>rdf:type</code> links resources.</p>"
    assert rp._curie_in_pedagogical_context(html, "rdf:type") is True
