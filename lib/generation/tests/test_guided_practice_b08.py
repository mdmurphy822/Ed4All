"""FR-INT-02 — first-class ``guided_practice`` (B08) block type.

CPU-pinned, no-GPU, no-model unit tests for the B08 first-class guided-practice
block type (distinct from the B07 self_check_question and the generic B08
activity/problem). Covers token registration + catalog/policy/bounds/framework-
map round-trip to B08 + the deterministic faded-scaffold renderer (gated by
ED4ALL_NEW_BLOCK_TYPES, byte-stable off) + the B08SequenceValidator (fires on a
bad sequence, passes on a good one, no-op when the flag is off).

PRIMARY gate: byte-stability with ED4ALL_NEW_BLOCK_TYPES unset — the type is
never rendered (renderer returns "") and the validator no-ops.
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

from Courseforge.scripts.blocks import BLOCK_TYPES, Block  # noqa: E402
from Courseforge.router.policy import DEFAULT_BLOCK_ROUTING  # noqa: E402
from lib.generation.block_catalog import load_block_catalog  # noqa: E402
from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES  # noqa: E402
from lib.ontology.framework_blocks import framework_block_for  # noqa: E402
from lib.validators.b08_sequence import B08SequenceValidator  # noqa: E402


# --------------------------------------------------------------------------- #
# FR-INT-02.1 — token + catalog/policy/bounds/framework round-trip to B08
# --------------------------------------------------------------------------- #
def test_guided_practice_in_block_types():
    assert "guided_practice" in BLOCK_TYPES


def test_guided_practice_constructs():
    blk = Block(
        block_id="week_01_application#guided_practice_x_0",
        block_type="guided_practice",
        page_id="week_01_application",
        sequence=0,
        content={"items": [{"prompt": "Solve 2x+1=7"}]},
        fade_state="completion",
    )
    assert blk.block_type == "guided_practice"
    assert blk.fade_state == "completion"


def test_guided_practice_maps_to_b08_via_sot_loader():
    assert framework_block_for("guided_practice") == "B08"


def test_guided_practice_has_catalog_entry_with_b08():
    entry = next(
        (e for e in load_block_catalog() if e["block_type"] == "guided_practice"),
        None,
    )
    assert entry is not None
    assert entry["framework_block"] == "B08"
    assert entry.get("use_when")
    assert entry.get("bloom_fit")
    assert entry.get("bloom_ceiling") == "create"


def test_guided_practice_in_default_block_routing():
    assert "guided_practice" in DEFAULT_BLOCK_ROUTING
    matrix = DEFAULT_BLOCK_ROUTING["guided_practice"]
    assert "source_ref" in matrix["required"]
    assert matrix["fail_action"] in {"regenerate", "escalate", "block"}


def test_guided_practice_in_outline_bounds():
    from Courseforge.generators._outline_provider import _OUTLINE_KIND_BOUNDS

    assert "guided_practice" in _OUTLINE_KIND_BOUNDS


# --------------------------------------------------------------------------- #
# FR-INT-02.2 — deterministic renderer (gated + reuses fade_state)
# --------------------------------------------------------------------------- #
def _gp_block(fade: str = "completion") -> Block:
    return Block(
        block_id="week_01_application#guided_practice_x_0",
        block_type="guided_practice",
        page_id="week_01_application",
        sequence=0,
        content={
            "prompt": "Now practice with fading support.",
            "items": [
                {"prompt": "Solve 3x = 9", "support": "Divide both sides by 3."},
                {"prompt": "Solve 5x = 20"},
            ],
        },
        fade_state=fade,
    )


def test_render_guided_practice_under_flag(monkeypatch):
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    from Courseforge.scripts.generate_course import _render_guided_practice

    html = _render_guided_practice(_gp_block("completion"))
    assert 'class="guided-practice"' in html
    assert 'data-cf-fade-state="completion"' in html  # reuses fade_state
    assert "<ol" in html and "</ol>" in html
    assert "Solve 3x = 9" in html
    assert "Divide both sides by 3." in html  # support gloss rendered


def test_render_guided_practice_byte_stable_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    from Courseforge.scripts.generate_course import _render_guided_practice

    assert _render_guided_practice(_gp_block()) == ""


# --------------------------------------------------------------------------- #
# FR-INT-02.3 — B08SequenceValidator (fires / passes / no-op off)
# --------------------------------------------------------------------------- #
def _block(block_type: str, idx: int, *, fade: str | None = None) -> Block:
    return Block(
        block_id=f"p#{block_type}_x_{idx}",
        block_type=block_type,
        page_id="p",
        sequence=idx,
        content={"items": []} if block_type == "guided_practice" else "body",
        fade_state=fade,
    )


def test_b08_validator_noop_when_flag_off():
    res = B08SequenceValidator().validate(
        {"blocks": [_block("guided_practice", 0)], "new_block_types_enabled": False}
    )
    assert res.passed is True
    assert any(i.code == "B08_SEQUENCE_DISABLED" for i in res.issues)


def test_b08_validator_fires_when_not_after_worked():
    # guided_practice with NO worked_example before it + no fade_state.
    res = B08SequenceValidator().validate(
        {
            "blocks": [
                _block("concept", 0),
                _block("guided_practice", 1),
            ],
            "new_block_types_enabled": True,
        }
    )
    assert res.passed is True  # warning-day-1
    codes = {i.code for i in res.issues}
    assert "B08_PRACTICE_NOT_AFTER_WORKED" in codes
    assert "B08_PRACTICE_NO_FADE_STATE" in codes
    assert res.action == "regenerate"


def test_b08_validator_passes_well_formed_sequence():
    res = B08SequenceValidator().validate(
        {
            "blocks": [
                _block("worked_example", 0, fade="worked"),
                _block("guided_practice", 1, fade="completion"),
            ],
            "new_block_types_enabled": True,
        }
    )
    assert res.passed is True
    codes = {i.code for i in res.issues}
    assert "B08_PRACTICE_NOT_AFTER_WORKED" not in codes
    assert "B08_PRACTICE_NO_FADE_STATE" not in codes
    assert res.action is None
