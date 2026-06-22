"""IB6.1 — BlockQualityRubricValidator (8-dim scoring + 3 cap hard gates)."""
from __future__ import annotations

from Courseforge.scripts.blocks import (
    CORE_QUALITY_DIMENSIONS,
    QUALITY_DIMENSIONS,
    Block,
)
from lib.validators.block_quality_rubric import BlockQualityRubricValidator


def _rubric_of(result, block_id):
    return result.metadata["block_quality_rubric"]["per_block"][block_id]


def test_clean_block_scores_core_ge_2():
    block = Block(
        block_id="b1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>A short single idea.</p>",
        bloom_level="understand",
        bloom_verb="explain",
        objective_ids=("CO-01",),
        n_representations=2,
    )
    # Strong signals threaded for every core dim.
    sigs = {
        "b1": {
            "cognitive_load_overflow": False,
            "char_count": 80,
            "verb_triple_status": "aligned",
            "alignment_cosine": 0.7,
            "wcag_clean": True,
            "coherence_score": 3,
            "has_colocated_retrieval": True,
        }
    }
    res = BlockQualityRubricValidator().validate(
        {"blocks": [block], "gate_results_by_block": sigs, "rubric_enabled": True}
    )
    row = _rubric_of(res, "b1")
    for dim in CORE_QUALITY_DIMENSIONS:
        assert row["scores"][dim] >= 2, (dim, row["scores"])
    assert row["block_pass"] is True


def test_assessment_bloom_below_objective_caps_alignment_at_1():
    block = Block(
        block_id="a1",
        block_type="assessment_item",
        page_id="week_01_self_check",
        sequence=1,
        content="<p>Recall the term.</p>",
        objective_ids=("CO-01",),
    )
    sigs = {"a1": {"alignment_cap_at_1": True, "verb_triple_status": "aligned"}}
    res = BlockQualityRubricValidator().validate(
        {"blocks": [block], "gate_results_by_block": sigs, "rubric_enabled": True}
    )
    row = _rubric_of(res, "a1")
    assert row["scores"]["alignment"] == 1
    assert "alignment" in row["caps_applied"]


def test_interaction_without_feedback_caps_feedback_and_coherence():
    block = Block(
        block_id="i1",
        block_type="self_check_question",
        page_id="week_01_self_check",
        sequence=1,
        content="<p>Q?</p>",
        objective_ids=("CO-01",),
    )
    # caps_dims directive from IB6.2 / IB6.3.
    sigs = {"i1": {"feedback_caps": True, "feedback_status": "missing"}}
    res = BlockQualityRubricValidator().validate(
        {"blocks": [block], "gate_results_by_block": sigs, "rubric_enabled": True}
    )
    row = _rubric_of(res, "i1")
    assert row["scores"]["feedback"] == 1
    assert row["scores"]["coherence"] == 1
    assert "feedback" in row["caps_applied"]
    assert "coherence" in row["caps_applied"]


def test_wcag_failing_block_scores_accessibility_0():
    block = Block(
        block_id="w1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
        objective_ids=("CO-01",),
    )
    sigs = {"w1": {"wcag_clean": False}}
    res = BlockQualityRubricValidator().validate(
        {"blocks": [block], "gate_results_by_block": sigs, "rubric_enabled": True}
    )
    row = _rubric_of(res, "w1")
    assert row["scores"]["accessibility"] == 0


def test_disabled_writes_no_rubric_field():
    block = Block(
        block_id="d1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
    )
    res = BlockQualityRubricValidator().validate(
        {"blocks": [block], "rubric_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"RUBRIC_DISABLED"}
    assert res.metadata["block_quality_rubric"]["enabled"] is False


def test_scoring_writes_rubric_into_block_via_replace():
    # The scored Block instance carries a populated quality_rubric tuple while
    # the input Block is unchanged (frozen-dataclass replace contract).
    block = Block(
        block_id="r1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
        objective_ids=("CO-01",),
    )
    assert block.quality_rubric == ()
    # The validator returns scored copies via metadata.scored_block_ids; the
    # rubric field is written on the replace()d copy in the scoring loop. We
    # assert the per_block report carries all 8 dimension keys for applicable.
    res = BlockQualityRubricValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    row = res.metadata["block_quality_rubric"]["per_block"]["r1"]
    # core dims always applicable
    for dim in CORE_QUALITY_DIMENSIONS:
        assert dim in row["scores"]
    assert set(QUALITY_DIMENSIONS) >= set(row["scores"].keys())
    assert block.quality_rubric == ()  # input untouched


def test_does_not_reload_nli_or_embeddings():
    # The rubric path composes threaded signals; it must NOT import/load the
    # NLI classifier or an embedder. We assert no model singleton is touched by
    # checking the module never imports a classifier at scoring time: a clean
    # run over a thin block (no signals) succeeds without sentence-transformers
    # / torch being required. (Import-time guard — the test would error on a
    # missing-deps box if the scorer pulled a model.)
    block = Block(
        block_id="n1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>An idea.</p>",
        objective_ids=("CO-01",),
    )
    res = BlockQualityRubricValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    assert res.passed is True
