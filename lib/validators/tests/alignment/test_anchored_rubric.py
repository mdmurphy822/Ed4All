"""IB3.4 — AnchoredRubricValidator tests + Block hash-exclusion."""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_SCRIPTS_DIR = _PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block  # noqa: E402

from lib.validators.alignment.anchored_rubric import (  # noqa: E402
    AnchoredRubricValidator,
    _CODE_MISSING,
    _CODE_NO_EXEMPLAR,
    _CODE_NOT_PUBLISHED_FIRST,
)


def _objectives():
    return {
        "CO-01": {"id": "CO-01", "bloom_level": "create",
                  "abcd": {"behavior": {"verb": "design"}}},
        "CO-02": {"id": "CO-02", "bloom_level": "apply",
                  "abcd": {"behavior": {"verb": "solve"}}},
    }


def _good_rubric():
    return {
        "criteria": [
            {"name": "Originality",
             "bands": [
                 {"level": 3, "descriptor": "novel", "exemplar": "Sample A shows ..."},
                 {"level": 1, "descriptor": "derivative", "exemplar": "Sample B copies ..."},
             ]},
        ],
        "published_before_task": True,
    }


def _validate(blocks, monkeypatch, flag="1"):
    if flag is None:
        monkeypatch.delenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", raising=False)
    else:
        monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", flag)
    return AnchoredRubricValidator().validate(
        {"blocks": blocks, "objectives": _objectives()}
    )


def _codes(res):
    return {i.code for i in res.issues}


def test_create_block_missing_rubric(monkeypatch):
    block = Block(block_id="p#scenario_a_0", block_type="scenario", page_id="p",
                  sequence=0, content="Design a system.", objective_ids=("CO-01",))
    res = _validate([block], monkeypatch)
    assert _CODE_MISSING in _codes(res)
    assert res.passed is True  # warning-only


def test_rubric_no_exemplar(monkeypatch):
    rubric = {
        "criteria": [{"name": "X", "bands": [
            {"level": 3, "descriptor": "good", "exemplar": ""},
            {"level": 1, "descriptor": "bad", "exemplar": "   "},
        ]}],
        "published_before_task": True,
    }
    block = Block(block_id="p#scenario_b_0", block_type="scenario", page_id="p",
                  sequence=0, content="Design.", objective_ids=("CO-01",),
                  anchored_rubric=rubric)
    res = _validate([block], monkeypatch)
    assert _CODE_NO_EXEMPLAR in _codes(res)


def test_rubric_not_published_first(monkeypatch):
    rubric = _good_rubric()
    rubric["published_before_task"] = False
    block = Block(block_id="p#scenario_c_0", block_type="scenario", page_id="p",
                  sequence=0, content="Design.", objective_ids=("CO-01",),
                  anchored_rubric=rubric)
    res = _validate([block], monkeypatch)
    assert _CODE_NOT_PUBLISHED_FIRST in _codes(res)


def test_complete_rubric_passes(monkeypatch):
    block = Block(block_id="p#scenario_d_0", block_type="scenario", page_id="p",
                  sequence=0, content="Design.", objective_ids=("CO-01",),
                  anchored_rubric=_good_rubric())
    res = _validate([block], monkeypatch)
    assert res.issues == []


def test_apply_block_out_of_scope(monkeypatch):
    # An apply objective (CO-02) needs no rubric.
    block = Block(block_id="p#scenario_e_0", block_type="scenario", page_id="p",
                  sequence=0, content="Solve.", objective_ids=("CO-02",))
    res = _validate([block], monkeypatch)
    assert res.issues == []


def test_flag_off_noop(monkeypatch):
    block = Block(block_id="p#scenario_f_0", block_type="scenario", page_id="p",
                  sequence=0, content="Design.", objective_ids=("CO-01",))
    res = _validate([block], monkeypatch, flag=None)
    assert res.issues == []
    assert res.passed is True


def test_anchored_rubric_hash_excluded():
    block = Block(block_id="p#scenario_g_0", block_type="scenario", page_id="p",
                  sequence=0, content="Design.", objective_ids=("CO-01",))
    h1 = block.compute_content_hash()
    block2 = dataclasses.replace(block, anchored_rubric=_good_rubric())
    assert block2.compute_content_hash() == h1
