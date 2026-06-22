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


def test_disabled_is_noop_pass():
    block = _clean_concept()
    res = QaChecklistValidator().validate(
        {"blocks": [block], "rubric_enabled": False}
    )
    assert res.passed is True
    assert {i.code for i in res.issues} == {"QA_CHECKLIST_DISABLED"}
