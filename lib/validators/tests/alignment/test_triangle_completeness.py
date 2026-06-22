"""IB3.5 — TriangleCompletenessValidator tests."""
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

from lib.validators.alignment.triangle_completeness import (  # noqa: E402
    TriangleCompletenessValidator,
    _CODE_NO_ACTIVITY,
    _CODE_NO_ALIGNED_ASSESSMENT,
)


def _objectives():
    return {
        "CO-01": {"id": "CO-01", "bloom_level": "apply",
                  "abcd": {"behavior": {"verb": "solve"}}},
    }


def _validate(blocks, monkeypatch, flag="1"):
    if flag is None:
        monkeypatch.delenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", raising=False)
    else:
        monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", flag)
    return TriangleCompletenessValidator().validate(
        {"blocks": blocks, "objectives": _objectives()}
    )


def _codes(res):
    return {i.code for i in res.issues}


def test_no_activity(monkeypatch):
    # concept + aligned assessment, but no activity.
    concept = Block(block_id="p#concept_a_0", block_type="concept", page_id="p",
                    sequence=0, content="A concept.", objective_ids=("CO-01",))
    assess = Block(block_id="p#assessment_item_a_0", block_type="assessment_item",
                   page_id="p", sequence=1, content="Solve for x.",
                   objective_ids=("CO-01",))
    res = _validate([concept, assess], monkeypatch)
    assert _CODE_NO_ACTIVITY in _codes(res)
    assert _CODE_NO_ALIGNED_ASSESSMENT not in _codes(res)


def test_no_aligned_assessment(monkeypatch):
    # activity present, but assessment is lower-Bloom than the apply objective.
    activity = Block(block_id="p#activity_b_0", block_type="activity", page_id="p",
                     sequence=0, content="Solve the practice problem.",
                     objective_ids=("CO-01",))
    assess = Block(block_id="p#assessment_item_b_0", block_type="assessment_item",
                   page_id="p", sequence=1, content="List the formula's terms.",
                   objective_ids=("CO-01",))
    res = _validate([activity, assess], monkeypatch)
    assert _CODE_NO_ALIGNED_ASSESSMENT in _codes(res)
    assert _CODE_NO_ACTIVITY not in _codes(res)


def test_complete_triangle(monkeypatch):
    activity = Block(block_id="p#activity_c_0", block_type="activity", page_id="p",
                     sequence=0, content="Apply the method to solve this.",
                     objective_ids=("CO-01",))
    assess = Block(block_id="p#assessment_item_c_0", block_type="assessment_item",
                   page_id="p", sequence=1, content="Solve the equation for x.",
                   objective_ids=("CO-01",))
    res = _validate([activity, assess], monkeypatch)
    assert res.issues == []
    assert res.score == 1.0


def test_flag_off_noop(monkeypatch):
    concept = Block(block_id="p#concept_d_0", block_type="concept", page_id="p",
                    sequence=0, content="A concept.", objective_ids=("CO-01",))
    res = _validate([concept], monkeypatch, flag=None)
    assert res.issues == []
    assert res.passed is True
