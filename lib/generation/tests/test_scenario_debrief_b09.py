"""FR-INT-06 — B09 case/scenario/branching mode + mandatory debrief.

CPU-pinned, no-GPU, no-model unit tests for the B09 scenario_mode field
(case/scenario/branching), the mode-by-Bloom planner selection (gated by
ED4ALL_DYNAMIC_BLOCK_PLAN, identity-no-op off), the deterministic scenario
renderer with a debrief scaffold (gated by ED4ALL_NEW_BLOCK_TYPES, byte-stable
off), and the B09DebriefValidator (fires on a missing debrief, passes when one
is present, no-op when the flag is off).

PRIMARY gate: byte-stability with both flags unset — the renderer returns "",
the planner is a strict identity no-op, and the validator no-ops.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from Courseforge.scripts.blocks import Block  # noqa: E402
from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES  # noqa: E402
from lib.validators.b09_debrief import B09DebriefValidator  # noqa: E402


# --------------------------------------------------------------------------- #
# FR-INT-06.1 — scenario_mode field
# --------------------------------------------------------------------------- #
def test_scenario_block_carries_scenario_mode():
    blk = Block(
        block_id="p#scenario_x_0",
        block_type="scenario",
        page_id="p",
        sequence=0,
        content={"situation": "A patient presents with..."},
        scenario_mode="branching",
    )
    assert blk.scenario_mode == "branching"


def test_scenario_mode_excluded_from_content_hash():
    base = Block(
        block_id="p#scenario_x_0", block_type="scenario", page_id="p",
        sequence=0, content={"situation": "S"},
    )
    with_mode = Block(
        block_id="p#scenario_x_0", block_type="scenario", page_id="p",
        sequence=0, content={"situation": "S"}, scenario_mode="branching",
    )
    assert base.compute_content_hash() == with_mode.compute_content_hash()


# --------------------------------------------------------------------------- #
# FR-INT-06.2 — mode-by-Bloom planner selection (gated, identity-no-op off)
# --------------------------------------------------------------------------- #
def test_planner_scenario_mode_by_bloom_when_on(monkeypatch):
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN", "1")
    from lib.generation.block_planner import _resolve_scenario_modes

    selected = [
        {"block_type": "scenario", "target_bloom": "apply"},
        {"block_type": "scenario", "target_bloom": "evaluate"},
        {"block_type": "scenario", "target_bloom": "create"},
        {"block_type": "concept", "target_bloom": "understand"},
    ]
    out = _resolve_scenario_modes(selected=selected)
    assert out[0]["scenario_mode"] == "case"
    assert out[1]["scenario_mode"] == "scenario"
    assert out[2]["scenario_mode"] == "branching"
    assert "scenario_mode" not in out[3]  # non-scenario untouched


def test_planner_scenario_mode_identity_noop_when_off(monkeypatch):
    monkeypatch.delenv("ED4ALL_DYNAMIC_BLOCK_PLAN", raising=False)
    from lib.generation.block_planner import _resolve_scenario_modes

    selected = [{"block_type": "scenario", "target_bloom": "create"}]
    out = _resolve_scenario_modes(selected=selected)
    assert "scenario_mode" not in out[0]


# --------------------------------------------------------------------------- #
# FR-INT-06.3 — deterministic renderer with debrief (gated, byte-stable off)
# --------------------------------------------------------------------------- #
def _scenario_block(debrief: str | None) -> Block:
    content = {
        "situation": "A bridge must carry a 10-ton load.",
        "prompt": "Which beam do you choose, and why?",
    }
    if debrief is not None:
        content["debrief"] = debrief
    return Block(
        block_id="p#scenario_x_0", block_type="scenario", page_id="p",
        sequence=0, content=content, scenario_mode="scenario",
    )


def test_render_scenario_has_debrief_under_flag(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    from Courseforge.scripts.generate_course import _render_scenario

    html = _render_scenario(_scenario_block("The key tradeoff was strength vs cost."))
    assert 'class="scenario-card"' in html
    assert 'data-cf-scenario-mode="scenario"' in html
    assert 'class="scenario-debrief"' in html
    assert 'data-cf-purpose="debrief"' in html
    assert "The key tradeoff was strength vs cost." in html


def test_render_scenario_byte_stable_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    from Courseforge.scripts.generate_course import _render_scenario

    assert _render_scenario(_scenario_block("x")) == ""


# --------------------------------------------------------------------------- #
# FR-INT-06.4 — B09DebriefValidator (fires / passes / no-op off)
# --------------------------------------------------------------------------- #
def test_b09_validator_noop_when_flag_off():
    res = B09DebriefValidator().validate(
        {"blocks": [_scenario_block(None)], "new_block_types_enabled": False}
    )
    assert res.passed is True
    assert any(i.code == "B09_DEBRIEF_DISABLED" for i in res.issues)


def test_b09_validator_fires_on_missing_debrief():
    blk = Block(
        block_id="p#scenario_x_0", block_type="scenario", page_id="p",
        sequence=0, content={"situation": "S", "prompt": "P"},
    )
    res = B09DebriefValidator().validate(
        {"blocks": [blk], "new_block_types_enabled": True}
    )
    assert res.passed is True  # warning-day-1
    assert any(i.code == "SCENARIO_DEBRIEF_MISSING" for i in res.issues)
    assert res.action == "regenerate"


def test_b09_validator_passes_with_dict_debrief():
    res = B09DebriefValidator().validate(
        {"blocks": [_scenario_block("Debrief: strength wins.")],
         "new_block_types_enabled": True}
    )
    assert res.passed is True
    assert not any(i.code == "SCENARIO_DEBRIEF_MISSING" for i in res.issues)
    assert res.action is None


def test_b09_validator_passes_with_html_debrief():
    blk = Block(
        block_id="p#scenario_x_0", block_type="scenario", page_id="p",
        sequence=0,
        content='<section class="scenario-card"><p>S</p>'
                '<div class="scenario-debrief"><h4>Debrief</h4><p>x</p></div></section>',
    )
    res = B09DebriefValidator().validate(
        {"blocks": [blk], "new_block_types_enabled": True}
    )
    assert not any(i.code == "SCENARIO_DEBRIEF_MISSING" for i in res.issues)


def test_b09_validator_passes_with_transition_slot_debrief():
    blk = Block(
        block_id="p#scenario_x_0", block_type="scenario", page_id="p",
        sequence=0, content={"situation": "S", "prompt": "P"},
        transition="Debrief: what did this case reveal about tradeoffs?",
    )
    res = B09DebriefValidator().validate(
        {"blocks": [blk], "new_block_types_enabled": True}
    )
    assert not any(i.code == "SCENARIO_DEBRIEF_MISSING" for i in res.issues)
