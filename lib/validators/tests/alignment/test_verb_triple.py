"""IB3.1 — verb-triple resolver helper tests."""
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

from lib.validators.alignment.verb_triple import (  # noqa: E402
    alignment_verb_triple_enabled,
    bloom_below,
    is_apply_plus,
    resolve_block_verb_level,
    resolve_objective_verb_level,
    verbs_share_band,
)


def test_resolve_objective_verb_level_from_abcd():
    lo = {
        "id": "CO-01",
        "bloom_level": "evaluate",
        "abcd": {"behavior": {"verb": "Critique"}},
    }
    verb, level = resolve_objective_verb_level(lo)
    assert verb == "critique"
    assert level == "evaluate"


def test_resolve_objective_verb_level_statement_fallback():
    # ABCD absent → detect from statement.
    lo = {"id": "CO-02", "statement": "Apply the formula to solve problems"}
    verb, level = resolve_objective_verb_level(lo)
    assert level == "apply"
    assert verb in {"apply", "solve"}


def test_evaluate_objective_vs_list_interaction_mismatch():
    lo = {"id": "CO-01", "bloom_level": "evaluate",
          "abcd": {"behavior": {"verb": "critique"}}}
    block = Block(
        block_id="p#activity_x_0", block_type="activity", page_id="p",
        sequence=0, content="List the steps of the process.",
    )
    _, obj_level = resolve_objective_verb_level(lo)
    blk_verb, blk_level = resolve_block_verb_level(block)
    assert (blk_verb, blk_level) == ("list", "remember")
    assert verbs_share_band(obj_level, blk_level) is False
    assert bloom_below(blk_level, obj_level) is True


def test_apply_objective_solve_activity_shares_band():
    block = Block(
        block_id="p#activity_y_0", block_type="activity", page_id="p",
        sequence=0, content="Solve the equation for x.",
    )
    _, blk_level = resolve_block_verb_level(block)
    assert blk_level == "apply"
    assert verbs_share_band("apply", blk_level) is True


def test_resolve_block_prefers_interaction_slot():
    block = Block(
        block_id="p#activity_z_0", block_type="activity", page_id="p",
        sequence=0, content="Some unrelated body text.",
        interaction="Evaluate the two arguments and judge which is stronger.",
    )
    verb, level = resolve_block_verb_level(block)
    assert level == "evaluate"


def test_block_picks_highest_bloom_verb():
    block = Block(
        block_id="p#activity_h_0", block_type="activity", page_id="p",
        sequence=0,
        content="First list the facts, then evaluate the strongest claim.",
    )
    _, level = resolve_block_verb_level(block)
    assert level == "evaluate"  # highest demand wins


def test_is_apply_plus():
    assert is_apply_plus("apply") is True
    assert is_apply_plus("create") is True
    assert is_apply_plus("understand") is False
    assert is_apply_plus("remember") is False
    assert is_apply_plus(None) is False


def test_verbs_share_band_none_safe():
    assert verbs_share_band(None, "apply") is False
    assert verbs_share_band("apply", None) is False
    assert verbs_share_band("bogus", "apply") is False


def test_flag_default_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", raising=False)
    assert alignment_verb_triple_enabled() is False
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "1")
    assert alignment_verb_triple_enabled() is True
    monkeypatch.setenv("ED4ALL_ALIGNMENT_VERB_TRIPLE", "garbage")
    assert alignment_verb_triple_enabled() is False
