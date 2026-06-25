"""IB6.7 — QaChecklistValidator (15-point per-block checklist)."""
from __future__ import annotations

from Courseforge.scripts.blocks import Block
from lib.validators.qa_checklist import QaChecklistValidator


def _checklist(result, block_id):
    return {
        row["item"]: row
        for row in result.metadata["qa_checklist"]["per_block"][block_id]["checklist"]
    }


def _clean_concept(content="<p>One idea.</p>", **kw) -> Block:
    base = dict(
        block_id="b1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content=content,
        bloom_level="understand",
        bloom_verb="explain",
        objective_ids=("CO-01",),
        heading="A concept",
        transition="Next, consider y.",
        n_representations=2,
    )
    base.update(kw)
    return Block(**base)


def test_qa9_signaling_excess():
    # 5 <strong> spans in a single-sentence (1 idea-chunk) body → over-signaling.
    body = "<p>This <strong>a</strong> <strong>b</strong> <strong>c</strong> <strong>d</strong> <strong>e</strong> idea.</p>"
    block = _clean_concept(content=body)
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    rows = _checklist(res, "b1")
    assert rows["QA-9"]["passed"] is False
    assert rows["QA-9"]["code"] == "QA9_SIGNALING_EXCESS"


def test_qa13_single_representation():
    block = _clean_concept(n_representations=1)
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    rows = _checklist(res, "b1")
    assert rows["QA-13"]["passed"] is False
    assert rows["QA-13"]["code"] == "QA13_SINGLE_REPRESENTATION"


def test_qa15_spacing_unverified_without_ib7():
    block = _clean_concept()
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    rows = _checklist(res, "b1")
    # advisory until IB7: passes but carries the unverified code.
    assert rows["QA-15"]["passed"] is True
    assert rows["QA-15"]["code"] == "QA15_SPACING_UNVERIFIED"


def test_fully_clean_block_all_15_pass():
    block = _clean_concept()
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    report = res.metadata["qa_checklist"]["per_block"]["b1"]
    # 15 items resolved; the clean block clears all (QA-15 unverified counts as
    # pass day-1 since it is advisory).
    assert len(report["checklist"]) == 15
    assert report["all_pass"] is True


def test_qa15_massed_fails():
    block = _clean_concept()
    res = QaChecklistValidator().validate(
        {
            "blocks": [block],
            "rubric_enabled": True,
            "spacing_by_block": {"b1": "massed"},
        }
    )
    rows = _checklist(res, "b1")
    assert rows["QA-15"]["passed"] is False
    assert rows["QA-15"]["code"] == "QA15_MASSED"


def _serialized_concept(**kw) -> Block:
    """A block as it re-hydrates from blocks_final.jsonl: the emit-only anatomy
    / UDL / Bloom fields (bloom_verb/bloom_level/heading/purpose_tag/transition/
    n_representations) are NOT serialized, so they come back None / 0."""
    base = dict(
        block_id="s1",
        block_type="concept",
        page_id="week_01_content",
        sequence=1,
        content="<p>One idea.</p>",
        objective_ids=("CO-01",),
    )
    base.update(kw)
    return Block(**base)


def test_absent_emit_only_fields_do_not_false_fire():
    # Regression for the IB6.7 truthiness bug: QA-3 (bloom), QA-6 (activation),
    # QA-13 (UDL >=2), QA-14 (transition) read emit-only Block fields that
    # re-hydrate to None/0 from blocks_final.jsonl. A truthiness check failed
    # EVERY serialized block on absent metadata (not bad content) and all(15)
    # made every block fail unconditionally. Post-fix, an absent field is
    # not-applicable → pass.
    block = _serialized_concept()
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    rows = _checklist(res, "s1")
    for item in ("QA-3", "QA-6", "QA-13", "QA-14"):
        assert rows[item]["passed"] is True, (item, rows[item])
    # The absent-metadata block now clears all 15 (it would have failed before).
    assert res.metadata["qa_checklist"]["per_block"]["s1"]["all_pass"] is True


def test_qa3_populated_but_missing_bloom_still_fires():
    # A block whose bloom fields ARE populated but genuinely empty still fires.
    block = _serialized_concept(bloom_verb="", bloom_level="")
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    rows = _checklist(res, "s1")
    assert rows["QA-3"]["passed"] is False
    assert rows["QA-3"]["code"] == "QA3_NO_BLOOM_TAG"


def test_qa14_populated_but_empty_transition_still_fires():
    block = _serialized_concept(transition="")
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    rows = _checklist(res, "s1")
    assert rows["QA-14"]["passed"] is False
    assert rows["QA-14"]["code"] == "ANATOMY_TRANSITION_MISSING"


def test_qa13_positively_declared_single_representation_still_fires():
    # n_representations == 1 is a POSITIVE single-code declaration (vs the
    # absent 0 sentinel) and still fires QA-13.
    block = _serialized_concept(n_representations=1)
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": True}
    )
    rows = _checklist(res, "s1")
    assert rows["QA-13"]["passed"] is False
    assert rows["QA-13"]["code"] == "QA13_SINGLE_REPRESENTATION"


def test_disabled_is_noop_pass():
    block = _clean_concept()
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"QA_CHECKLIST_DISABLED"}
