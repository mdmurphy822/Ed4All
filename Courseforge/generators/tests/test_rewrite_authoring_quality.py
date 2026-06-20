"""7B→Sonnet parity: authoring-quality directives in the rewrite contracts.

String-assertable coverage for the measured authoring defects vs the Sonnet
baseline (manual review of a full 7B course). Each test pins a directive in a
per-block-type output contract or the shared rewrite system prompt so a future
edit cannot silently drop it. No LLM, no pipeline run.

Items covered:
- ①  step-label colon INSIDE the badge span.
- ⑤  self_check_question authors MULTIPLE questions, each with a JS-free
     `<details>` reveal.
- ⑥  summary_takeaway authors MULTIPLE distinct takeaway sections.
- ⑦  table guidance for tabular / comparative content.
- ⑫  no raw `$...$` LaTeX in fraction guidance; distractor value ↔ rationale
     self-consistency clause in assessment_item.
- ⑮  content-derived (not filename / generic) headings.
- ⑯  "Key Idea"-style framing on key-rule / callout content.
"""

import Courseforge.generators._rewrite_provider as rp


# ---------------------------------------------------------------------------
# ① — step-label colon INSIDE the badge span
# ---------------------------------------------------------------------------

def test_system_prompt_pins_step_colon_inside_badge():
    sp = rp._REWRITE_SYSTEM_PROMPT
    # The correct form (colon inside) is instructed...
    assert "Step N:</span>" in sp
    # ...and the wrong form (colon outside) is explicitly prohibited.
    assert "Step N</span>:" in sp


def test_example_contract_pins_step_colon_inside_badge():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["example"]
    assert "Step N:</span>" in contract
    # The contract names the wrong form so the model sees the contrast.
    assert "Step N</span>:" in contract


# ---------------------------------------------------------------------------
# ⑤ — self_check authors MULTIPLE questions, each with a JS-free reveal
# ---------------------------------------------------------------------------

def test_self_check_contract_requests_multiple_questions():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["self_check_question"]
    assert "MULTIPLE" in contract
    # A concrete target count steers away from the single-question 7B failure.
    assert "3-5" in contract


def test_self_check_contract_requests_details_reveal_not_js():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["self_check_question"]
    assert "<details>" in contract
    assert "<summary>" in contract
    # JS-button reveals get sanitizer-stripped — the contract must forbid them.
    assert "onclick" in contract or "JavaScript" in contract


def test_self_check_contract_keeps_no_inline_answer_pedagogy():
    # Strengthening to multi-question must NOT drop the formative-assessment
    # guarantee (answer behind a reveal, never inline).
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["self_check_question"]
    assert "NEVER reveal the answer inline" in contract
    assert "self-check-item" in contract


# ---------------------------------------------------------------------------
# ⑥ — summary_takeaway authors MULTIPLE distinct sections
# ---------------------------------------------------------------------------

def test_summary_takeaway_contract_requests_multiple_sections():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["summary_takeaway"]
    assert "MULTIPLE DISTINCT" in contract
    assert "<h3>" in contract
    # A concrete target count, mirroring the self_check directive.
    assert "3-5" in contract


def test_summary_takeaway_contract_keeps_takeaway_card_and_grounding():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["summary_takeaway"]
    # Distinct styled wrapper from the prior session is preserved.
    assert "takeaway-card" in contract
    # Anti-fabrication guidance survives the strengthening.
    assert "no invented" in contract.lower() or "no invented sections" in contract


# ---------------------------------------------------------------------------
# ⑦ — table guidance for tabular / comparative content
# ---------------------------------------------------------------------------

def test_system_prompt_instructs_real_table_for_comparative_content():
    sp = rp._REWRITE_SYSTEM_PROMPT
    assert "<table>" in sp
    assert "<thead>" in sp


def test_concept_contract_instructs_table_for_taxonomy():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["concept"]
    assert "<table>" in contract


# ---------------------------------------------------------------------------
# ⑫ — math rendering (no raw LaTeX) + distractor self-consistency
# ---------------------------------------------------------------------------

def test_system_prompt_prohibits_raw_latex_fractions():
    sp = rp._REWRITE_SYSTEM_PROMPT
    # The directive names the raw-LaTeX failure form so the model recognizes it.
    assert "\\frac" in sp
    # And steers to the repo's Unicode / MathML convention instead.
    assert "MathML" in sp
    assert "Unicode" in sp


def test_assessment_item_contract_has_self_consistency_clause():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["assessment_item"]
    # Distractor stated value must equal the arithmetic of its named error.
    assert "NUMERIC SELF-CHECK" in contract
    assert "contradicts its own rationale" in contract
    # No raw LaTeX in option markup either.
    assert "no LaTeX renderer" in contract


def test_system_prompt_has_numeric_self_consistency_clause():
    sp = rp._REWRITE_SYSTEM_PROMPT
    assert "NUMERIC SELF-CHECK" in sp
    assert "VERIFY the arithmetic" in sp


# ---------------------------------------------------------------------------
# ⑮ — content-derived (not filename / generic) headings
# ---------------------------------------------------------------------------

def test_system_prompt_steers_content_derived_headings():
    sp = rp._REWRITE_SYSTEM_PROMPT
    assert "CONTENT-DERIVED HEADINGS" in sp
    # The raw-filename failure form is named explicitly.
    assert "raw page filename" in sp


def test_concept_contract_steers_content_derived_title():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["concept"]
    assert "content-derived" in contract
    assert "filename" in contract


# ---------------------------------------------------------------------------
# ⑯ — "Key Idea" framing on key-rule / callout content
# ---------------------------------------------------------------------------

def test_system_prompt_has_key_idea_framing_directive():
    sp = rp._REWRITE_SYSTEM_PROMPT
    assert "KEY-IDEA FRAMING" in sp
    assert "Key Idea" in sp


def test_callout_contract_requests_key_idea_framing():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["callout"]
    assert "Key Idea" in contract


def test_concept_contract_leads_key_rule_with_key_idea():
    contract = rp._BLOCK_TYPE_OUTPUT_CONTRACTS["concept"]
    assert "Key Idea" in contract
    assert "key-rule" in contract
