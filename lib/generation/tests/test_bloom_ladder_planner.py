"""Bloom-ladder initiative, WI-06 — planner ladder-selection pass tests.

Covers ``lib/generation/block_planner.py``'s ``ED4ALL_BLOOM_LADDER``-gated
ladder pass:

- byte-identical / strict-identity no-op when the flag is off;
- the CEILING SOURCE is the CO's OWN synthesized ``bloom_level``
  (``bloom_level`` / ``bloomLevel``-fallback), never
  ``_resolve_target_bloom``'s per-BLOCK declared/catalog arms;
- ``rungs_up_to(objective_ceiling)`` is walked ONLY — no rung above the
  objective's own ceiling is ever offered;
- exactly one misconception-probe block is injected per rung that the
  taxonomy actually names ``misconception`` at (today: only ``analyze``);
- every injected candidate is routed through the existing
  ``_apply_bloom_ceilings`` re-route machinery, preserving ``target_co_ids``;
- ``ladder_rung`` / ``mc_bloom_rung`` are stamped via the additive
  ``_to_page_plan`` tuple convention — appended ONLY when set;
- the two WI-06 decision events (``bloom_ladder_block_selection`` /
  ``bloom_ladder_ceiling_enforcement``) fire with dynamic, replayable
  rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lib.generation.block_catalog import load_block_catalog  # noqa: E402
from lib.generation.block_planner import (  # noqa: E402
    _BLOOM_CEILING_ENV,
    _apply_bloom_ladder_selection,
    _bloom_ladder_objective_ceiling,
    plan_week_blocks,
)
from lib.generation.bloom_ladder_blocks import ENV_BLOOM_LADDER  # noqa: E402
from lib.ontology.bloom import BLOOM_LEVELS  # noqa: E402


def _catalog_by_type():
    return {
        str(e.get("block_type")): e
        for e in load_block_catalog() if e.get("block_type")
    }


def _block_types():
    return __import__(
        "Courseforge.scripts.blocks", fromlist=["BLOCK_TYPES"]
    ).BLOCK_TYPES


class _RecordingCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


_TO = {"id": "TO-01", "statement": "Solve and graph linear equations"}


# --------------------------------------------------------------------------- #
# _bloom_ladder_objective_ceiling — tolerant bloom_level / bloomLevel read
# --------------------------------------------------------------------------- #
def test_ceiling_reads_bloom_level():
    assert _bloom_ladder_objective_ceiling({"bloom_level": "apply"}) == "apply"


def test_ceiling_falls_back_to_camel_case():
    assert _bloom_ladder_objective_ceiling({"bloomLevel": "Analyze"}) == "analyze"


def test_ceiling_strips_and_lowercases():
    assert _bloom_ladder_objective_ceiling({"bloom_level": "  Understand  "}) == "understand"


def test_ceiling_none_when_missing():
    assert _bloom_ladder_objective_ceiling({}) is None


def test_ceiling_none_when_invalid():
    assert _bloom_ladder_objective_ceiling({"bloom_level": "not_a_level"}) is None


def test_ceiling_none_when_not_a_string():
    assert _bloom_ladder_objective_ceiling({"bloom_level": 3}) is None


# --------------------------------------------------------------------------- #
# _apply_bloom_ladder_selection — flag-off byte-identity
# --------------------------------------------------------------------------- #
def test_selection_identity_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
    ]
    out = _apply_bloom_ladder_selection(
        selected=selected,
        chapter_objectives=[{"id": "CO-01", "bloom_level": "create"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    # Strict identity — the exact same object, not just an equal copy.
    assert out is selected


def test_selection_noop_when_no_co_has_a_ceiling(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
    ]
    out = _apply_bloom_ladder_selection(
        selected=selected,
        chapter_objectives=[{"id": "CO-01"}],  # no bloom_level at all
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    assert out is selected


# --------------------------------------------------------------------------- #
# Ceiling capping — never draws from a rung above the objective's own level.
# --------------------------------------------------------------------------- #
def test_ladder_capped_at_objective_own_ceiling(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-01", "bloom_level": "apply"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    injected = [b for b in out if b.get("ladder_rung")]
    assert injected, "expected the ladder pass to inject at least one block"
    ceiling_idx = BLOOM_LEVELS.index("apply")
    for blk in injected:
        assert BLOOM_LEVELS.index(blk["ladder_rung"]) <= ceiling_idx
    rungs_seen = {blk["ladder_rung"] for blk in injected}
    assert rungs_seen == {"remember", "understand", "apply"}
    # No misconception block at this ceiling (misconception is only
    # taxonomy-named at the analyze rung, above this CO's own ceiling).
    assert not any(b.get("block_type") == "misconception" for b in injected)
    assert not any(b.get("mc_bloom_rung") for b in injected)


def test_ladder_single_rung_at_remember(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-01", "bloom_level": "remember"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    rungs_seen = {b["ladder_rung"] for b in out if b.get("ladder_rung")}
    assert rungs_seen == {"remember"}


def test_ladder_full_ladder_at_create(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-01", "bloom_level": "create"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    rungs_seen = {b["ladder_rung"] for b in out if b.get("ladder_rung")}
    assert rungs_seen == set(BLOOM_LEVELS)
    # Exactly one misconception-probe block across the whole walk — the
    # taxonomy names misconception at exactly one rung (analyze) today.
    mc_blocks = [b for b in out if b.get("mc_bloom_rung")]
    assert len(mc_blocks) == 1
    assert mc_blocks[0]["mc_bloom_rung"] == "analyze"
    assert mc_blocks[0]["block_type"] == "misconception"
    assert mc_blocks[0]["target_co_ids"] == ["CO-01"]


def test_ladder_deterministic_type_per_rung(monkeypatch):
    """The alphabetically-first taxonomy-permitted non-misconception type is
    picked per rung — deterministic, not frozenset-iteration-order-dependent."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-01", "bloom_level": "create"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    by_rung = {}
    for blk in out:
        rung = blk.get("ladder_rung")
        if rung and not blk.get("mc_bloom_rung"):
            by_rung[rung] = blk["block_type"]
    assert by_rung == {
        "remember": "acronym",
        "understand": "callout",
        "apply": "activity",
        "analyze": "diagram",
        "evaluate": "assessment_item",
        "create": "activity",
    }


def test_ladder_independent_per_co(monkeypatch):
    """Each CO's ceiling is resolved independently — a low-ceiling CO does not
    borrow a high-ceiling sibling's rungs, and vice versa."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[
            {"id": "CO-LOW", "bloom_level": "remember"},
            {"id": "CO-HIGH", "bloom_level": "analyze"},
        ],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    low_rungs = {
        b["ladder_rung"] for b in out
        if b.get("ladder_rung") and b["target_co_ids"] == ["CO-LOW"]
    }
    high_rungs = {
        b["ladder_rung"] for b in out
        if b.get("ladder_rung") and b["target_co_ids"] == ["CO-HIGH"]
    }
    assert low_rungs == {"remember"}
    assert high_rungs == {"remember", "understand", "apply", "analyze"}


def test_ladder_never_invents_a_co_id(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-01", "bloom_level": "create"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    for blk in out:
        assert blk["target_co_ids"] == ["CO-01"]


# --------------------------------------------------------------------------- #
# Ceiling re-route machinery reuse (:2711-2759) — target_co_ids preserved.
# --------------------------------------------------------------------------- #
def test_ladder_misconception_rerouted_when_ceiling_reroute_on(monkeypatch):
    """misconception's own catalog bloom_ceiling (apply) is BELOW the analyze
    rung it is taxonomy-named at, so when ED4ALL_PLANNER_BLOOM_CEILING is ALSO
    on, the existing _apply_bloom_ceilings machinery re-routes it — exactly
    like any other over-ceiling block, target_co_ids preserved."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    monkeypatch.setenv(_BLOOM_CEILING_ENV, "1")
    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-01", "bloom_level": "analyze"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    mc_blocks = [b for b in out if b.get("mc_bloom_rung") == "analyze"]
    assert len(mc_blocks) == 1
    rerouted = mc_blocks[0]
    assert rerouted["block_type"] in {"scenario", "problem", "assessment_item"}
    assert rerouted["block_type"] != "misconception"
    assert rerouted["target_co_ids"] == ["CO-01"]
    # The ladder stamp survives the re-route (only block_type/page_type/
    # content_focus are mutated by _apply_bloom_ceilings).
    assert rerouted["mc_bloom_rung"] == "analyze"
    assert rerouted["ladder_rung"] == "analyze"


def test_ladder_misconception_kept_when_ceiling_reroute_off(monkeypatch):
    """Without ED4ALL_PLANNER_BLOOM_CEILING, the existing re-route machinery
    is itself a no-op, so the misconception candidate stays block_type
    'misconception' — the ladder pass never duplicates that gate's logic."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    monkeypatch.delenv(_BLOOM_CEILING_ENV, raising=False)
    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-01", "bloom_level": "analyze"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
    )
    mc_blocks = [b for b in out if b.get("mc_bloom_rung") == "analyze"]
    assert len(mc_blocks) == 1
    assert mc_blocks[0]["block_type"] == "misconception"
    assert mc_blocks[0]["target_co_ids"] == ["CO-01"]


# --------------------------------------------------------------------------- #
# Decision capture — bloom_ladder_block_selection / bloom_ladder_ceiling_enforcement
# --------------------------------------------------------------------------- #
def test_decision_capture_fires_with_dynamic_rationale(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    cap = _RecordingCapture()
    _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-42", "bloom_level": "analyze"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
        capture=cap,
        course_code="TEST101",
        to_id="TO-01",
    )
    types = [e["decision_type"] for e in cap.events]
    assert "bloom_ladder_block_selection" in types
    assert "bloom_ladder_ceiling_enforcement" in types

    selection_ev = next(
        e for e in cap.events if e["decision_type"] == "bloom_ladder_block_selection"
    )
    assert len(selection_ev["rationale"]) >= 20
    assert "CO-42" in selection_ev["rationale"]
    assert "analyze" in selection_ev["rationale"]
    assert "TO-01" in selection_ev["rationale"]

    ceiling_ev = next(
        e for e in cap.events if e["decision_type"] == "bloom_ladder_ceiling_enforcement"
    )
    assert len(ceiling_ev["rationale"]) >= 20
    assert "TO-01" in ceiling_ev["rationale"]
    assert "re-route" in ceiling_ev["rationale"].lower()


def test_decision_capture_silent_when_flag_off(monkeypatch):
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    cap = _RecordingCapture()
    _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-42", "bloom_level": "analyze"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
        capture=cap,
        course_code="TEST101",
        to_id="TO-01",
    )
    assert cap.events == []


def test_decision_capture_survives_capture_exception(monkeypatch):
    """A raising capture must never break the ladder pass (best-effort)."""
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")

    class _RaisingCapture:
        def log_decision(self, **kwargs):  # noqa: ARG002
            raise RuntimeError("capture backend down")

    out = _apply_bloom_ladder_selection(
        selected=[],
        chapter_objectives=[{"id": "CO-01", "bloom_level": "apply"}],
        catalog_by_type=_catalog_by_type(),
        block_types=_block_types(),
        capture=_RaisingCapture(),
        to_id="TO-01",
    )
    assert any(b.get("ladder_rung") for b in out)


# --------------------------------------------------------------------------- #
# _to_page_plan extras convention — ladder_rung / mc_bloom_rung appended ONLY
# when set (additive tuple, byte-stable flag-off).
# --------------------------------------------------------------------------- #
def test_page_plan_carries_ladder_extras_when_on(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=[{"id": "CO-01", "bloom_level": "analyze"}],
        provider=None,  # deterministic fixed-plan fallback path
    )
    extras_tuples = [
        entry
        for page in plan.page_plan.values()
        for entry in page
        if len(entry) >= 6
    ]
    assert extras_tuples, "expected at least one ladder-stamped tuple"
    ladder_stamped = [t for t in extras_tuples if "ladder_rung" in t[5]]
    assert ladder_stamped
    mc_stamped = [t for t in extras_tuples if "mc_bloom_rung" in t[5]]
    assert mc_stamped
    assert mc_stamped[0][5]["mc_bloom_rung"] == "analyze"


def test_page_plan_no_ladder_extras_when_off(monkeypatch):
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=[{"id": "CO-01", "bloom_level": "analyze"}],
        provider=None,
    )
    for page in plan.page_plan.values():
        for entry in page:
            if len(entry) >= 6:
                assert "ladder_rung" not in entry[5]
                assert "mc_bloom_rung" not in entry[5]


# --------------------------------------------------------------------------- #
# Full plan_week_blocks integration — byte-identical fallback path when off.
# --------------------------------------------------------------------------- #
def test_plan_week_blocks_fallback_byte_identical_when_off(monkeypatch):
    monkeypatch.delenv(ENV_BLOOM_LADDER, raising=False)
    cos = [{"id": "CO-01", "statement": "isolate the variable", "bloom_level": "apply"}]
    plan_a = plan_week_blocks(
        terminal_objective=_TO, chapter_objectives=cos, provider=None,
    )
    plan_b = plan_week_blocks(
        terminal_objective=_TO, chapter_objectives=cos, provider=None,
    )
    assert plan_a.selected == plan_b.selected
    assert plan_a.page_plan == plan_b.page_plan
    assert not any(b.get("ladder_rung") for b in plan_a.selected)


def test_plan_week_blocks_fallback_injects_ladder_when_on(monkeypatch):
    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")
    cos = [{"id": "CO-01", "statement": "isolate the variable", "bloom_level": "apply"}]
    plan = plan_week_blocks(
        terminal_objective=_TO, chapter_objectives=cos, provider=None,
    )
    assert any(b.get("ladder_rung") for b in plan.selected)
    assert plan.fallback_used is True


def test_plan_week_blocks_success_path_injects_ladder_when_on(monkeypatch):
    """The ladder pass also fires on the real LLM-selection success path
    (not just the fixed-plan fallback)."""
    import json

    monkeypatch.setenv(ENV_BLOOM_LADDER, "1")

    class _MockProvider:
        _model = "meta/llama-3.3-70b-instruct"

        def plan_blocks(self, prompt):  # noqa: ARG002
            return json.dumps({"blocks": [
                {"block_type": "objective", "target_co_ids": ["CO-01"],
                 "page_type": "overview", "content_focus": "state outcome"},
                {"block_type": "example", "target_co_ids": ["CO-01"],
                 "page_type": "content", "content_focus": "worked solve"},
                {"block_type": "self_check_question", "target_co_ids": ["CO-01"],
                 "page_type": "self_check", "content_focus": "checkpoint"},
                {"block_type": "summary_takeaway", "target_co_ids": ["CO-01"],
                 "page_type": "summary", "content_focus": "recap"},
            ]})

    cos = [{"id": "CO-01", "statement": "isolate the variable", "bloom_level": "understand"}]
    plan = plan_week_blocks(
        terminal_objective=_TO, chapter_objectives=cos,
        provider=_MockProvider(),
    )
    assert plan.fallback_used is False
    ladder_rungs = {b["ladder_rung"] for b in plan.selected if b.get("ladder_rung")}
    assert ladder_rungs == {"remember", "understand"}


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
