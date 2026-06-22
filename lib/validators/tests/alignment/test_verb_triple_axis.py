"""IB3.2 — verb-triple EQUALITY axis on BlockObjectiveDeliveryValidator.

Exercises the 4th axis behind ED4ALL_ALIGNMENT_VERB_TRIPLE. NLI is forced
absent (the verb-triple axis is NLI-independent) so these run fast and don't
need the [embedding] extras.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPTS_DIR = _PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.block_objective_delivery import (  # noqa: E402
    BlockObjectiveDeliveryValidator,
    _CODE_VERB_TRIPLE_MISMATCH,
)


class _NoneNli:
    """Stub whose get_or_load is bypassed via nli=None injection."""


def _validate(blocks, objectives, monkeypatch, flag="1"):
    if flag is None:
        monkeypatch.delenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", raising=False)
    else:
        monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", flag)
    # nli stays None → NLI axis skipped; verb-triple axis still runs.
    validator = BlockObjectiveDeliveryValidator(nli=None)
    return validator.validate({
        "blocks": blocks,
        "objectives": objectives,
        "objective_statements": {oid: o.get("statement", "") for oid, o in objectives.items()},
        "return_touched_blocks": True,
    })


def _codes(res):
    return [i.code for i in res.issues]


def test_evaluate_objective_list_activity_mismatch(monkeypatch):
    objectives = {
        "CO-01": {"id": "CO-01", "bloom_level": "evaluate", "bloom_verb": "critique",
                  "statement": "Critique the argument", "abcd": {"behavior": {"verb": "critique"}}},
    }
    block = Block(block_id="p#activity_a_0", block_type="activity", page_id="p",
                  sequence=0, content="List the steps of the process.",
                  objective_ids=("CO-01",))
    res = _validate([block], objectives, monkeypatch)
    assert _CODE_VERB_TRIPLE_MISMATCH in _codes(res)
    entry = res  # find touched block alignment entry
    touched = []  # validate stores _touched_blocks in inputs; reload via re-run
    # Easier: re-run capturing inputs.
    inputs = {
        "blocks": [block], "objectives": objectives,
        "objective_statements": {"CO-01": "Critique the argument"},
        "return_touched_blocks": True,
    }
    BlockObjectiveDeliveryValidator(nli=None).validate(inputs)
    tb = inputs["_touched_blocks"][0]
    ali = tb.objective_alignment[0]
    assert ali["verb_triple_status"] == "mismatch"
    assert "alignment_cap_at_1" not in ali  # not an assessment block


def test_assessment_recall_under_apply_sets_cap(monkeypatch):
    objectives = {
        "CO-01": {"id": "CO-01", "bloom_level": "apply", "bloom_verb": "solve",
                  "statement": "Apply the method", "abcd": {"behavior": {"verb": "solve"}}},
    }
    # Assessment block whose resolved verb is recall ("list"/"define").
    block = Block(block_id="p#assessment_item_a_0", block_type="assessment_item",
                  page_id="p", sequence=0, content="Define the term and list its parts.",
                  objective_ids=("CO-01",))
    inputs = {
        "blocks": [block], "objectives": objectives,
        "objective_statements": {"CO-01": "Apply the method"},
        "return_touched_blocks": True,
    }
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    res = BlockObjectiveDeliveryValidator(nli=None).validate(inputs)
    assert _CODE_VERB_TRIPLE_MISMATCH in [i.code for i in res.issues]
    ali = inputs["_touched_blocks"][0].objective_alignment[0]
    assert ali["verb_triple_status"] == "mismatch"
    assert ali.get("alignment_cap_at_1") is True


def test_matched_verb_no_mismatch(monkeypatch):
    objectives = {
        "CO-01": {"id": "CO-01", "bloom_level": "apply", "bloom_verb": "solve",
                  "statement": "Apply the method", "abcd": {"behavior": {"verb": "solve"}}},
    }
    block = Block(block_id="p#activity_m_0", block_type="activity", page_id="p",
                  sequence=0, content="Solve the practice equation by applying the rule.",
                  objective_ids=("CO-01",))
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    res = BlockObjectiveDeliveryValidator(nli=None).validate({
        "blocks": [block], "objectives": objectives,
        "objective_statements": {"CO-01": "Apply the method"},
    })
    assert _CODE_VERB_TRIPLE_MISMATCH not in [i.code for i in res.issues]


def test_flag_off_byte_stable_entry(monkeypatch):
    """Flag OFF → no verb_triple_status key, no mismatch issue, scenario not audited."""
    objectives = {
        "CO-01": {"id": "CO-01", "bloom_level": "evaluate", "bloom_verb": "critique",
                  "statement": "Critique the argument", "abcd": {"behavior": {"verb": "critique"}}},
    }
    # An audited type (activity) for the entry-shape assertion.
    block = Block(block_id="p#activity_off_0", block_type="activity", page_id="p",
                  sequence=0, content="List the steps.", objective_ids=("CO-01",))
    # A scenario block — only audited when the flag is on.
    scenario = Block(block_id="p#scenario_off_0", block_type="scenario", page_id="p",
                     sequence=1, content="List the parts.", objective_ids=("CO-01",))
    inputs = {
        "blocks": [block, scenario], "objectives": objectives,
        "objective_statements": {"CO-01": "Critique the argument"},
        "return_touched_blocks": True,
    }
    monkeypatch.delenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", raising=False)
    res = BlockObjectiveDeliveryValidator(nli=None).validate(inputs)
    assert _CODE_VERB_TRIPLE_MISMATCH not in [i.code for i in res.issues]
    # The activity block's alignment entry must NOT carry the new key.
    for tb in inputs["_touched_blocks"]:
        for ali in getattr(tb, "objective_alignment", ()):
            assert "verb_triple_status" not in ali
            assert "alignment_cap_at_1" not in ali
    # The scenario block must NOT have been audited (flag off) → no alignment entry.
    scenario_tb = [t for t in inputs["_touched_blocks"]
                   if getattr(t, "block_type", None) == "scenario"][0]
    assert getattr(scenario_tb, "objective_alignment", ()) == ()
