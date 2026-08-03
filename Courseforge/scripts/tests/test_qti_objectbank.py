"""Regression net for the QTI 1.2 ``<objectbank>`` item-bank emitter.

``assessment_to_qti`` emits a fixed EXAM (`<assessment>`); this covers the
question-LIBRARY surface (`<objectbank>`) that stores the reusable item pool
with queryable per-item selection metadata.

Contracts pinned here:

* the document is well-formed QTI with the canonical namespace + root tag;
* items are built by the SAME ``question_to_qti_item`` the exam path uses, so
  a bank item is shape-identical to an exam item plus additive metadata;
* the additive ``ed4all_*`` ``qtimetadata`` is emitted only for fields the
  question actually carries (anti-fabrication — never a blank field);
* an empty pool yields a valid empty bank rather than raising.
"""

import xml.etree.ElementTree as ET

import pytest

from Courseforge.scripts.packaging.qti_emitter import (
    QTI_NS,
    assessment_to_objectbank,
    assessment_to_qti,
)

NS = {"q": QTI_NS}


def _mc(**over):
    q = {
        "question_id": "Q-1",
        "question_type": "multiple_choice",
        "stem": "<p>Identify the additive identity.</p>",
        "choices": [
            {"id": "A", "text": "0", "is_correct": True},
            {"id": "B", "text": "1"},
            {"id": "C", "text": "-1"},
        ],
        "correct_answer": "0",
        "objective_id": "CO-01",
        "bloom_level": "remember",
    }
    q.update(over)
    return q


def _bank(questions):
    return {"bank_id": "BANK-demo", "title": "Demo Bank", "questions": questions}


def _items(xml):
    root = ET.fromstring(xml)
    ob = root.find("q:objectbank", NS)
    assert ob is not None, "no <objectbank> element"
    return ob, ob.findall("q:item", NS)


def _meta(item):
    return {
        f.findtext("q:fieldlabel", namespaces=NS):
            f.findtext("q:fieldentry", namespaces=NS)
        for f in item.findall(
            "q:itemmetadata/q:qtimetadata/q:qtimetadatafield", NS
        )
    }


def test_emits_wellformed_objectbank():
    ob, items = _items(assessment_to_objectbank(_bank([_mc()])))
    assert ob.get("ident") == "BANK-demo"
    assert len(items) == 1


def test_root_is_questestinterop_in_qti_namespace():
    root = ET.fromstring(assessment_to_objectbank(_bank([_mc()])))
    assert root.tag == f"{{{QTI_NS}}}questestinterop"


def test_stamps_queryable_selection_metadata():
    _, items = _items(
        assessment_to_objectbank(
            _bank([_mc(item_subtype="error_analysis")])
        )
    )
    meta = _meta(items[0])
    assert meta["ed4all_objective_id"] == "CO-01"
    assert meta["ed4all_bloom_level"] == "remember"
    assert meta["ed4all_item_subtype"] == "error_analysis"
    assert meta["ed4all_question_type"] == "multiple_choice"


def test_absent_fields_are_not_fabricated():
    """A question with no item_subtype must not emit a blank field."""
    _, items = _items(assessment_to_objectbank(_bank([_mc()])))
    meta = _meta(items[0])
    assert "ed4all_item_subtype" not in meta
    # ...but the fields it DOES carry are present.
    assert meta["ed4all_objective_id"] == "CO-01"


def test_blank_valued_field_is_skipped():
    _, items = _items(
        assessment_to_objectbank(_bank([_mc(item_subtype="   ")]))
    )
    assert "ed4all_item_subtype" not in _meta(items[0])


def test_empty_pool_yields_valid_empty_bank():
    ob, items = _items(assessment_to_objectbank(_bank([])))
    assert items == []
    assert ob.get("ident") == "BANK-demo"


def test_accepts_items_key_alias():
    xml = assessment_to_objectbank(
        {"bank_id": "B", "title": "T", "items": [_mc()]}
    )
    _, items = _items(xml)
    assert len(items) == 1


def test_bank_carries_bank_level_metadata():
    ob, _ = _items(assessment_to_objectbank(_bank([_mc()])))
    fields = {
        f.findtext("q:fieldlabel", namespaces=NS):
            f.findtext("q:fieldentry", namespaces=NS)
        for f in ob.findall("q:qtimetadata/q:qtimetadatafield", NS)
    }
    assert fields["ed4all_bank_title"] == "Demo Bank"


@pytest.mark.parametrize(
    "qtype", ["multiple_choice", "true_false", "essay", "short_answer"]
)
def test_item_shape_matches_the_exam_path(qtype):
    """A bank item is the exam item plus additive metadata — same builder."""
    q = _mc(question_type=qtype)
    bank_xml = assessment_to_objectbank(_bank([q]))
    exam_xml = assessment_to_qti(
        {"assessment_id": "A-1", "title": "Exam", "questions": [q]}
    )
    _, bank_items = _items(bank_xml)
    exam_items = ET.fromstring(exam_xml).findall(
        "q:assessment/q:section/q:item", NS
    )
    assert len(bank_items) == len(exam_items) == 1
    # Same presentation subtree (the additive metadata lives in itemmetadata).
    bank_pres = bank_items[0].find("q:presentation", NS)
    exam_pres = exam_items[0].find("q:presentation", NS)
    assert (bank_pres is None) == (exam_pres is None)
    if bank_pres is not None:
        assert ET.tostring(bank_pres) == ET.tostring(exam_pres)
