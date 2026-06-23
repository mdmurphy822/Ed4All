"""FR-11 — tests for the anchored-rubric PRODUCER.

Covers:
- the producer authors a valid anchored_rubric for an Evaluate/Create scored
  block (under ED4ALL_ALIGNMENT_VERB_TRIPLE);
- the produced rubric SATISFIES AnchoredRubricValidator (closes the
  validator-without-producer gap);
- None when the flag is off (byte-stable);
- None for a lower-Bloom (apply) block / non-scored block type;
- anti-fabrication: no rubric when nothing can be grounded;
- grounding: criteria/exemplars reference the resolved concept;
- the attach pass is a strict no-op when the flag is off (identity).
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPTS_DIR = _PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.generation.anchored_rubric import (  # noqa: E402
    attach_anchored_rubrics,
    build_anchored_rubric,
)
from lib.validators.alignment.anchored_rubric import (  # noqa: E402
    AnchoredRubricValidator,
)

_FLAG = "ED4ALL_ALIGNMENT_VERB_TRIPLE"


def _objectives_map():
    return {
        # Create objective with a grounded behavior object.
        "CO-01": {
            "id": "CO-01",
            "statement": "Students will design a relational database schema.",
            "bloom_level": "create",
            "abcd": {"behavior": {"verb": "design",
                                  "action_object": "relational database schema"}},
        },
        # Evaluate objective grounded only via the statement.
        "CO-02": {
            "id": "CO-02",
            "statement": "Learners will evaluate competing sorting algorithms.",
            "bloom_level": "evaluate",
            "abcd": {"behavior": {"verb": "evaluate"}},
        },
        # Apply objective — below the Evaluate band.
        "CO-03": {
            "id": "CO-03",
            "statement": "Students will solve linear equations.",
            "bloom_level": "apply",
            "abcd": {"behavior": {"verb": "solve"}},
        },
    }


def _objectives_payload():
    return list(_objectives_map().values())


def _on(monkeypatch):
    monkeypatch.setenv(_FLAG, "1")


def _off(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)


def _scored_block(block_id, obj_id, content="Build it.", block_type="scenario",
                  key_terms=()):
    return Block(
        block_id=block_id, block_type=block_type, page_id="p", sequence=0,
        content=content, objective_ids=(obj_id,), key_terms=tuple(key_terms),
    )


# --------------------------------------------------------------------------- #
# build_anchored_rubric — shape + grounding + anti-fabrication.
# --------------------------------------------------------------------------- #

def test_build_authors_valid_create_rubric():
    block = _scored_block("p#scenario_a_0", "CO-01")
    rubric = build_anchored_rubric(block, _objectives_map())
    assert rubric is not None
    assert rubric["published_before_task"] is True
    assert len(rubric["criteria"]) >= 1
    for crit in rubric["criteria"]:
        assert crit["name"]
        assert len(crit["bands"]) >= 2
        for band in crit["bands"]:
            assert isinstance(band["exemplar"], str) and band["exemplar"].strip()
            assert isinstance(band["descriptor"], str) and band["descriptor"].strip()


def test_build_grounds_criteria_in_concept():
    """The rubric must reference the assessed concept (anti-template)."""
    block = _scored_block("p#scenario_b_0", "CO-01")
    rubric = build_anchored_rubric(block, _objectives_map())
    blob = " ".join(
        crit["name"] + " " + " ".join(b["descriptor"] + " " + b["exemplar"]
                                      for b in crit["bands"])
        for crit in rubric["criteria"]
    ).lower()
    # The grounded behavior object should surface in the criteria/exemplars.
    assert "schema" in blob


def test_build_evaluate_via_statement_concept():
    block = _scored_block("p#assessment_c_0", "CO-02", block_type="assessment_item")
    rubric = build_anchored_rubric(block, _objectives_map())
    assert rubric is not None
    blob = " ".join(
        b["exemplar"] for crit in rubric["criteria"] for b in crit["bands"]
    ).lower()
    # Concept resolved from the statement tail ("sorting algorithms").
    assert "algorithm" in blob or "sorting" in blob


def test_build_none_for_apply_block():
    block = _scored_block("p#scenario_d_0", "CO-03")  # apply objective
    assert build_anchored_rubric(block, _objectives_map()) is None


def test_build_none_for_non_scored_type():
    # A concept block is not a scored/produced-work type.
    block = Block(block_id="p#concept_e_0", block_type="concept", page_id="p",
                  sequence=0, content="x", objective_ids=("CO-01",))
    assert build_anchored_rubric(block, _objectives_map()) is None


def test_build_none_when_nothing_to_anchor():
    """ANTI-FABRICATION: empty objective (no concept, no verb) → None."""
    objectives = {"CO-99": {"id": "CO-99", "statement": "", "bloom_level": "create"}}
    block = Block(block_id="p#scenario_f_0", block_type="scenario", page_id="p",
                  sequence=0, content="", objective_ids=("CO-99",))
    assert build_anchored_rubric(block, objectives) is None


def test_build_none_for_unknown_objective():
    block = _scored_block("p#scenario_g_0", "CO-DOES-NOT-EXIST")
    assert build_anchored_rubric(block, _objectives_map()) is None


# --------------------------------------------------------------------------- #
# Validator-passes-on-produced-block — closes the consumer-without-producer gap.
# --------------------------------------------------------------------------- #

def test_produced_block_passes_validator(monkeypatch):
    _on(monkeypatch)
    block = _scored_block("p#scenario_h_0", "CO-01")
    [produced] = attach_anchored_rubrics(
        blocks=[block], objectives_payload=_objectives_payload(),
    )
    assert produced.anchored_rubric is not None
    res = AnchoredRubricValidator().validate(
        {"blocks": [produced], "objectives": _objectives_map()}
    )
    # No ANCHORED_RUBRIC_* issues — the produced rubric satisfies the consumer.
    assert res.issues == [], [i.code for i in res.issues]
    assert res.passed is True
    assert res.score == 1.0


def test_produced_evaluate_block_passes_validator(monkeypatch):
    _on(monkeypatch)
    block = _scored_block("p#assessment_i_0", "CO-02", block_type="assessment_item")
    [produced] = attach_anchored_rubrics(
        blocks=[block], objectives_payload=_objectives_payload(),
    )
    res = AnchoredRubricValidator().validate(
        {"blocks": [produced], "objectives": _objectives_map()}
    )
    assert res.issues == [], [i.code for i in res.issues]


# --------------------------------------------------------------------------- #
# attach_anchored_rubrics — gating + byte-stability + identity.
# --------------------------------------------------------------------------- #

def test_attach_noop_when_flag_off(monkeypatch):
    _off(monkeypatch)
    block = _scored_block("p#scenario_j_0", "CO-01")
    out = attach_anchored_rubrics(
        blocks=[block], objectives_payload=_objectives_payload(),
    )
    # Identity: same object, anchored_rubric stays None (byte-stable).
    assert out == [block]
    assert out[0].anchored_rubric is None


def test_attach_leaves_lower_bloom_blocks_untouched(monkeypatch):
    _on(monkeypatch)
    apply_block = _scored_block("p#scenario_k_0", "CO-03")  # apply
    concept_block = Block(block_id="p#concept_l_0", block_type="concept",
                          page_id="p", sequence=0, content="x",
                          objective_ids=("CO-01",))
    out = attach_anchored_rubrics(
        blocks=[apply_block, concept_block],
        objectives_payload=_objectives_payload(),
    )
    assert out[0].anchored_rubric is None
    assert out[1].anchored_rubric is None


def test_attach_dict_block_shape(monkeypatch):
    """A dict-shaped block (legacy caller) gets the key set, not replaced."""
    _on(monkeypatch)
    block = {"block_type": "scenario", "block_id": "p#s_0",
             "objective_ids": ["CO-01"], "content": "x"}
    [out] = attach_anchored_rubrics(
        blocks=[block], objectives_payload=_objectives_payload(),
    )
    assert out["anchored_rubric"] is not None


def test_attach_flag_off_then_on_byte_stability(monkeypatch):
    """Flag-off output is byte-identical to the input block (hash unchanged)."""
    block = _scored_block("p#scenario_m_0", "CO-01")
    h_before = block.compute_content_hash()
    _off(monkeypatch)
    out_off = attach_anchored_rubrics(
        blocks=[block], objectives_payload=_objectives_payload(),
    )
    assert out_off[0].compute_content_hash() == h_before
    assert out_off[0].anchored_rubric is None
    # And ON: the rubric is hash-EXCLUDED, so the content hash is STILL stable.
    _on(monkeypatch)
    out_on = attach_anchored_rubrics(
        blocks=[block], objectives_payload=_objectives_payload(),
    )
    assert out_on[0].anchored_rubric is not None
    assert out_on[0].compute_content_hash() == h_before


def test_attach_emits_decision(monkeypatch):
    _on(monkeypatch)

    class _Capture:
        def __init__(self):
            self.events = []

        def log_decision(self, **kw):
            self.events.append(kw)

    cap = _Capture()
    block = _scored_block("p#scenario_n_0", "CO-01")
    attach_anchored_rubrics(
        blocks=[block], objectives_payload=_objectives_payload(),
        capture=cap, course_code="TEST_101",
    )
    assert any(e["decision_type"] == "content_selection" for e in cap.events)
    ev = cap.events[0]
    assert len(ev["rationale"]) >= 20
    assert "authored 1" in ev["rationale"]


def test_attach_empty_blocks(monkeypatch):
    _on(monkeypatch)
    assert attach_anchored_rubrics(blocks=[], objectives_payload=[]) == []
