"""Unit tests for the Wave-2 Part 3 content-aware block planner.

Mocks the 70B (NO live call). Covers:

- planner returns an ordered, validated block list;
- unknown ``block_type`` is dropped;
- the block count is clamped to the budget;
- a dropped CO gets a default coverage block (coverage holds);
- an LLM error degrades to the deterministic fixed-plan fallback;
- an unparseable / empty response degrades to fallback;
- the ``WeekBlockPlan`` projects to the canonical five page types;
- a ``block_plan`` decision event fires per TO.
"""

from __future__ import annotations

import json

import pytest

from lib.generation.block_planner import (
    CANONICAL_PAGE_TYPES,
    _PAGE_TYPE_FLOORS,
    WeekBlockPlan,
    plan_week_blocks,
)


# --------------------------------------------------------------------------- #
# Mock provider seams.
# --------------------------------------------------------------------------- #
class _MockProvider:
    """Returns a canned JSON plan via ``plan_blocks``."""

    _model = "meta/llama-3.3-70b-instruct"

    def __init__(self, payload):
        self._payload = payload
        self.calls = 0

    def plan_blocks(self, prompt):  # noqa: ARG002
        self.calls += 1
        return self._payload


class _RaisingProvider:
    _model = "meta/llama-3.3-70b-instruct"

    def plan_blocks(self, prompt):  # noqa: ARG002
        raise RuntimeError("simulated 70B dispatch failure")


class _DispatchOnlyProvider:
    """Exercises the ``_dispatch_call`` seam (no ``plan_blocks``)."""

    _model = "meta/llama-3.3-70b-instruct"

    def __init__(self, payload):
        self._payload = payload

    def _dispatch_call(self, prompt):  # noqa: ARG002
        return self._payload, 0


class _RecordingCapture:
    def __init__(self):
        self.events = []

    def log_decision(self, **kwargs):
        self.events.append(kwargs)


_COS = [
    {"id": "CO-01", "statement": "isolate the variable", "bloom_level": "apply"},
    {"id": "CO-02", "statement": "check the solution", "bloom_level": "analyze"},
    {"id": "CO-03", "statement": "graph the line", "bloom_level": "understand"},
]
_TO = {"id": "TO-01", "statement": "Solve and graph linear equations"}


def _plan_payload(blocks):
    return json.dumps({"blocks": blocks})


# --------------------------------------------------------------------------- #
# Tests.
# --------------------------------------------------------------------------- #
def test_planner_returns_ordered_validated_list():
    payload = _plan_payload([
        {"block_type": "objective", "target_co_ids": ["CO-01"],
         "page_type": "overview", "content_focus": "state outcome"},
        {"block_type": "vocab_card", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": "slope term"},
        {"block_type": "example", "target_co_ids": ["CO-02"],
         "page_type": "content", "content_focus": "worked solve"},
        {"block_type": "scenario", "target_co_ids": ["CO-03"],
         "page_type": "application", "content_focus": "real-world line"},
        {"block_type": "self_check_question", "target_co_ids": ["CO-02"],
         "page_type": "self_check", "content_focus": "checkpoint"},
        {"block_type": "summary_takeaway", "target_co_ids": ["CO-01"],
         "page_type": "summary", "content_focus": "recap"},
    ])
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(payload),
    )
    assert isinstance(plan, WeekBlockPlan)
    assert plan.fallback_used is False
    types = [b["block_type"] for b in plan.selected]
    # Every planner-selected type survives (floors only ADD, never drop).
    for bt in (
        "objective", "vocab_card", "example", "scenario",
        "self_check_question", "summary_takeaway",
    ):
        assert bt in types
    # Floors reflatten the selection in canonical page-emit order, so the
    # overview ``objective`` leads and the summary blocks trail.
    overview_types = [bt for bt, *_ in plan.page_plan["overview"]]
    summary_types = [bt for bt, *_ in plan.page_plan["summary"]]
    assert overview_types and overview_types[0] == "objective"
    assert "summary_takeaway" in summary_types
    # All five canonical page types present as keys.
    assert set(plan.page_plan) == set(CANONICAL_PAGE_TYPES)
    # The vocab_card landed on the content page.
    assert any(bt == "vocab_card" for bt, *_ in plan.page_plan["content"])


def test_unknown_block_type_dropped():
    payload = _plan_payload([
        {"block_type": "concept", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": "x"},
        {"block_type": "NOT_A_REAL_TYPE", "target_co_ids": ["CO-02"],
         "page_type": "content", "content_focus": "y"},
        {"block_type": "example", "target_co_ids": ["CO-03"],
         "page_type": "content", "content_focus": "z"},
    ])
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(payload),
    )
    types = [b["block_type"] for b in plan.selected]
    assert "NOT_A_REAL_TYPE" not in types
    assert "concept" in types and "example" in types


def test_budget_clamped_max():
    # 15 content blocks returned; budget max 6 → the PLANNER's own
    # selections are clamped to 6 content blocks. Page-type FLOORS then add
    # page-appropriate fillers on top (floors take precedence over the
    # budget max, by design — the default budget is raised to fund them).
    blocks = [
        {"block_type": "explanation", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": f"b{i}"}
        for i in range(15)
    ]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=[_COS[0]],  # single CO so coverage is trivial
        provider=_MockProvider(_plan_payload(blocks)),
        budget=(2, 6),
    )
    assert plan.fallback_used is False
    # The planner's CONTENT (non-floor) blocks were clamped to the max of 6.
    content_explanations = [
        bt for bt, *_ in plan.page_plan["content"] if bt == "explanation"
    ]
    assert len(content_explanations) <= 6


def test_budget_topup_min():
    # Only 1 block returned; budget min 5 → top-up to 5.
    blocks = [
        {"block_type": "concept", "target_co_ids": ["CO-01", "CO-02", "CO-03"],
         "page_type": "content", "content_focus": "all"},
    ]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload(blocks)),
        budget=(5, 12),
    )
    assert len(plan.selected) >= 5


def test_dropped_co_gets_default_block():
    # Plan covers CO-01 only; CO-02 + CO-03 must get default coverage blocks.
    blocks = [
        {"block_type": "concept", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": "only co1"},
    ]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload(blocks)),
        budget=(1, 12),
    )
    covered = set()
    for b in plan.selected:
        covered.update(b["target_co_ids"])
    assert {"CO-01", "CO-02", "CO-03"} <= covered


def test_invalid_page_type_repaired_not_dropped():
    blocks = [
        {"block_type": "misconception", "target_co_ids": ["CO-01"],
         "page_type": "bananas", "content_focus": "error"},
    ]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=[_COS[0]],
        provider=_MockProvider(_plan_payload(blocks)),
        budget=(1, 12),
    )
    misc = [b for b in plan.selected if b["block_type"] == "misconception"]
    assert misc, "misconception block should survive page_type repair"
    assert misc[0]["page_type"] in CANONICAL_PAGE_TYPES


def test_llm_error_falls_back_to_fixed_plan():
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_RaisingProvider(),
    )
    assert plan.fallback_used is True
    # Fixed plan now deploys the previously-unused types and is floor-compliant.
    content = plan.page_block_plan_for("content")
    content_types = [bt for bt, *_ in content]
    assert content_types[:2] == ["concept", "explanation"]
    assert "example" in content_types


def test_unparseable_response_falls_back():
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider("this is not json at all <<<"),
    )
    assert plan.fallback_used is True


def test_empty_blocks_falls_back():
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload([])),
    )
    assert plan.fallback_used is True


def test_no_provider_is_fixed_plan():
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=None,
    )
    assert plan.fallback_used is True
    assert set(plan.page_plan) == set(CANONICAL_PAGE_TYPES)


def test_dispatch_call_seam():
    payload = _plan_payload([
        {"block_type": "concept", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": "x"},
        {"block_type": "example", "target_co_ids": ["CO-02"],
         "page_type": "content", "content_focus": "y"},
    ])
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_DispatchOnlyProvider(payload),
        budget=(1, 12),
    )
    assert plan.fallback_used is False
    types = [b["block_type"] for b in plan.selected]
    assert "concept" in types and "example" in types


def test_json_with_code_fence_parses():
    fenced = (
        "```json\n"
        + _plan_payload([
            {"block_type": "concept", "target_co_ids": ["CO-01"],
             "page_type": "content", "content_focus": "x"},
        ])
        + "\n```"
    )
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=[_COS[0]],
        provider=_MockProvider(fenced),
        budget=(1, 12),
    )
    assert plan.fallback_used is False


def test_block_plan_decision_event_fires():
    cap = _RecordingCapture()
    payload = _plan_payload([
        {"block_type": "concept", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": "x"},
        {"block_type": "example", "target_co_ids": ["CO-02"],
         "page_type": "content", "content_focus": "y"},
        {"block_type": "scenario", "target_co_ids": ["CO-03"],
         "page_type": "application", "content_focus": "z"},
    ])
    plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(payload),
        capture=cap,
        course_code="TEST_101",
        budget=(1, 12),
    )
    events = [e for e in cap.events if e.get("decision_type") == "block_plan"]
    assert len(events) == 1
    ev = events[0]
    assert len(ev["rationale"]) >= 20
    assert "TO-01" in ev["rationale"] or "TO-01" in ev["decision"]


def test_fallback_emits_block_plan_event():
    cap = _RecordingCapture()
    plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_RaisingProvider(),
        capture=cap,
        course_code="TEST_101",
    )
    events = [e for e in cap.events if e.get("decision_type") == "block_plan"]
    assert len(events) == 1
    assert events[0]["ml_features"]["fallback_used"] is True


def test_page_plan_for_unknown_page_type_defaults_to_content():
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=None,
    )
    # Unknown page type → content plan fallback.
    assert plan.page_block_plan_for("nonexistent") == plan.page_plan["content"]


# --------------------------------------------------------------------------- #
# Per-page-type FLOOR tests (findings 2/6/7/8/16/18).
# --------------------------------------------------------------------------- #
def _page_counts(plan):
    return {ptype: len(blocks) for ptype, blocks in plan.page_plan.items()}


def _all_types(plan):
    return {b["block_type"] for b in plan.selected}


# A deliberately STARVED plan: the 70B pours everything onto content and
# emits 1 thin block on every other page (the observed 7B failure mode).
_STARVED_PLAN = [
    {"block_type": "objective", "target_co_ids": ["CO-01"],
     "page_type": "overview", "content_focus": "to only"},
    {"block_type": "concept", "target_co_ids": ["CO-01"],
     "page_type": "content", "content_focus": "c1"},
    {"block_type": "explanation", "target_co_ids": ["CO-02"],
     "page_type": "content", "content_focus": "c2"},
    {"block_type": "activity", "target_co_ids": ["CO-03"],
     "page_type": "application", "content_focus": "one app"},
    {"block_type": "self_check_question", "target_co_ids": ["CO-01"],
     "page_type": "self_check", "content_focus": "one q"},
    {"block_type": "summary_takeaway", "target_co_ids": ["CO-02"],
     "page_type": "summary", "content_focus": "one takeaway"},
]


def test_every_page_type_meets_its_floor():
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload(_STARVED_PLAN)),
    )
    assert plan.fallback_used is False
    counts = _page_counts(plan)
    for ptype, floor in _PAGE_TYPE_FLOORS.items():
        assert counts[ptype] >= floor, (
            f"{ptype} has {counts[ptype]} blocks, floor is {floor}"
        )


def test_self_check_has_at_least_four_questions():
    # finding 7 — self_check must carry >= 4 self_check_question blocks.
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload(_STARVED_PLAN)),
    )
    q_count = sum(
        1 for bt, *_ in plan.page_plan["self_check"]
        if bt == "self_check_question"
    )
    assert q_count >= 4


def test_application_and_summary_meet_floor():
    # findings 2/8 — application + summary each >= 4 blocks.
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload(_STARVED_PLAN)),
    )
    counts = _page_counts(plan)
    assert counts["application"] >= 4
    assert counts["summary"] >= 4


def test_overview_carries_objective_enumeration():
    # finding 18 — the overview must lead with the objective enumeration
    # block (TO + all COs), even when the planner forgot it.
    starved_no_objective = [
        b for b in _STARVED_PLAN if b["block_type"] != "objective"
    ] + [
        {"block_type": "explanation", "target_co_ids": ["CO-01"],
         "page_type": "overview", "content_focus": "prose only"},
    ]
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload(starved_no_objective)),
    )
    overview_types = [bt for bt, *_ in plan.page_plan["overview"]]
    assert overview_types, "overview must not be empty"
    assert overview_types[0] == "objective"
    assert len(overview_types) >= _PAGE_TYPE_FLOORS["overview"]


def test_unused_block_types_become_reachable():
    # findings 6/16 — the contracted-but-never-authored types appear via the
    # floor fillers even when the planner never selected them.
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload(_STARVED_PLAN)),
    )
    deployed = _all_types(plan)
    # The floor fillers deploy these previously-0-count types.
    assert "reflection_prompt" in deployed   # finding 6
    assert "discussion_prompt" in deployed
    assert "prereq_set" in deployed
    # At least one of callout (finding 16) / misconception / flip_card_grid
    # is reachable via a floor top-up.
    assert deployed & {"callout", "misconception", "flip_card_grid", "checklist"}


def test_floors_apply_to_fixed_fallback():
    # A planner FAILURE must still yield a floor-compliant, balanced week.
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_RaisingProvider(),
    )
    assert plan.fallback_used is True
    counts = _page_counts(plan)
    for ptype, floor in _PAGE_TYPE_FLOORS.items():
        assert counts[ptype] >= floor
    # The fixed fallback also deploys the previously-unused types.
    deployed = _all_types(plan)
    assert {"reflection_prompt", "discussion_prompt", "prereq_set"} <= deployed


def test_floors_preserve_co_coverage():
    # Floors must never break the CO-coverage guarantee.
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(_plan_payload(_STARVED_PLAN)),
    )
    covered = set()
    for b in plan.selected:
        covered.update(b["target_co_ids"])
    assert {"CO-01", "CO-02", "CO-03"} <= covered


# --------------------------------------------------------------------------- #
# CO-id fan-out fix: page_block_plan_for surfaces per-block target_co_ids.
# --------------------------------------------------------------------------- #
def test_page_block_plan_for_surfaces_target_co_ids():
    """Each ``page_block_plan_for`` entry is a 3-tuple carrying the planner's
    per-block ``target_co_ids`` (the CO-id fan-out fix)."""
    payload = _plan_payload([
        {"block_type": "vocab_card", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": "slope term"},
        {"block_type": "example", "target_co_ids": ["CO-02"],
         "page_type": "content", "content_focus": "worked solve"},
        {"block_type": "scenario", "target_co_ids": ["CO-03"],
         "page_type": "application", "content_focus": "real-world line"},
    ])
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(payload),
    )
    content = plan.page_block_plan_for("content")
    # Every entry is a 3-tuple (block_type, target_bloom, target_co_ids).
    for entry in content:
        assert len(entry) == 3
        assert isinstance(entry[2], list)
    co_ids_by_type = {bt: co_ids for bt, _bloom, co_ids in content}
    assert co_ids_by_type.get("vocab_card") == ["CO-01"]
    assert co_ids_by_type.get("example") == ["CO-02"]
    app = plan.page_block_plan_for("application")
    app_by_type = {bt: co_ids for bt, _bloom, co_ids in app}
    assert app_by_type.get("scenario") == ["CO-03"]


def test_page_block_plan_for_synthetic_weekblockplan_3tuple():
    """A hand-built ``WeekBlockPlan`` surfaces the 3-tuple verbatim (no
    planner / no model load)."""
    plan = WeekBlockPlan(
        page_plan={
            "overview": [],
            "content": [
                ("concept", "understand", ["CO-02"]),
                ("explanation", "apply", []),
            ],
            "application": [],
            "self_check": [],
            "summary": [],
        }
    )
    entries = plan.page_block_plan_for("content")
    assert entries == [
        ("concept", "understand", ["CO-02"]),
        ("explanation", "apply", []),
    ]


def test_fallback_plan_entries_are_3tuple_empty_co_ids():
    """The deterministic fixed-plan fallback emits 3-tuples whose
    ``target_co_ids`` is always empty — the consumer's "use week-TO" signal,
    keeping the fixed-plan path byte-identical in objective_ids terms."""
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=None,  # None provider → fixed fallback
    )
    assert plan.fallback_used is True
    for page_type in CANONICAL_PAGE_TYPES:
        for entry in plan.page_block_plan_for(page_type):
            assert len(entry) == 3
            assert entry[2] == []


# --------------------------------------------------------------------------- #
# ITEM 7 (plan side) — a table-bearing / comparative-grid block is selectable
# and deployed. There is no dedicated "table" BLOCK_TYPE (the <table> HTML
# rides on a content section); the comparative/tabular block the planner can
# SELECT is flip_card_grid (a grid of paired cards).
# --------------------------------------------------------------------------- #
def test_flip_card_grid_is_selectable_on_content():
    # The planner SELECTS flip_card_grid for a comparison and it survives onto
    # the content page (floors never drop a planner-chosen block).
    payload = _plan_payload([
        {"block_type": "flip_card_grid", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": "compare slope-intercept vs point-slope"},
        {"block_type": "concept", "target_co_ids": ["CO-02"],
         "page_type": "content", "content_focus": "definition"},
        {"block_type": "example", "target_co_ids": ["CO-03"],
         "page_type": "content", "content_focus": "worked"},
    ])
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_MockProvider(payload),
    )
    assert plan.fallback_used is False
    content_types = [bt for bt, *_ in plan.page_plan["content"]]
    assert "flip_card_grid" in content_types


def test_fixed_fallback_deploys_comparative_grid():
    # ITEM 7 — even a planner FAILURE deploys the comparative/grid block on the
    # content page (the Sonnet-table analogue), via the fixed-plan fallback.
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        provider=_RaisingProvider(),
    )
    assert plan.fallback_used is True
    content_types = [bt for bt, *_ in plan.page_block_plan_for("content")]
    assert "flip_card_grid" in content_types


def test_prompt_steers_comparative_content_to_grid():
    # The planner prompt carries explicit comparative/tabular guidance so the
    # live 70B steers comparison content to flip_card_grid (plan-side ITEM 7).
    from lib.generation.block_planner import _build_prompt
    from lib.generation.block_catalog import load_block_catalog

    prompt = _build_prompt(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=[],
        catalog=load_block_catalog(),
        budget=(5, 24),
    )
    lowered = prompt.lower()
    assert "comparative" in lowered or "tabular" in lowered
    assert "flip_card_grid" in prompt


# --------------------------------------------------------------------------- #
# IB5 — planner selection of the four framework-aligned types (gated).
# --------------------------------------------------------------------------- #
def _build_ib5_prompt(source_chunks):
    from lib.generation.block_planner import _build_prompt
    from lib.generation.block_catalog import load_block_catalog

    return _build_prompt(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=source_chunks,
        catalog=load_block_catalog(),
        budget=(5, 24),
    )


def test_ib5_nudges_absent_when_flag_off(monkeypatch):
    """The byte-stability guard: flag off → no IB5 content-shape NUDGE lines.

    The catalog MENU lists the four new types unconditionally once IB5.2 lands
    (it iterates the catalog) — exactly the I6 precedent — so the menu naming
    them flag-off is expected. What MUST be gated is the IB5 nudge lines: with
    the flag off the planner never appends a worked_example / multimedia /
    diagram / hook nudge, so the DETECTED CONTENT SHAPES section carries no IB5
    steering even on a procedure-shaped source."""
    from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES

    procedure_src = [{"text": "Step 1: do X. Step 2: do Y. Step 3: solve for z."}]
    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    prompt_off = _build_ib5_prompt(procedure_src)
    # The IB5 nudge phrasings are absent (these strings appear ONLY in the
    # gated DETECTED CONTENT SHAPES nudge lines, never in the catalog menu).
    assert "describes a step-by-step PROCEDURE" not in prompt_off
    assert "mandatory captions / audio-description" not in prompt_off
    assert "OPEN this objective with a `hook`" not in prompt_off
    # On a procedure source with the flag ON, the worked_example nudge appears.
    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    prompt_on = _build_ib5_prompt(procedure_src)
    assert "describes a step-by-step PROCEDURE" in prompt_on


def test_ib5_procedure_nudges_worked_example_when_on(monkeypatch):
    from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES

    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    prompt = _build_ib5_prompt(
        [{"text": "Follow these steps: Step 1: isolate. Step 2: divide."}]
    )
    assert "worked_example" in prompt


def test_ib5_media_nudges_multimedia_when_on(monkeypatch):
    from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES

    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    prompt = _build_ib5_prompt([{"text": "Watch the video explaining factoring."}])
    assert "multimedia" in prompt


def test_ib5_diagram_nudges_diagram_when_on(monkeypatch):
    from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES

    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    prompt = _build_ib5_prompt([{"text": "As shown in the flowchart, the figure maps each step."}])
    assert "diagram" in prompt


def test_ib5_hook_nudge_always_present_when_on(monkeypatch):
    from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES

    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    prompt = _build_ib5_prompt([{"text": "plain prose about fractions"}])
    assert "hook" in prompt


def test_ib5_detectors_precision():
    from lib.generation.block_planner import (
        detect_procedure,
        detect_media_reference,
        detect_diagram_reference,
    )

    assert detect_procedure("Step 1: do X. Step 2: do Y.")
    assert not detect_procedure("A flat prose paragraph about numbers.")
    assert detect_media_reference("Watch the video below.")
    assert detect_media_reference("see https://x.test/clip.mp4")
    assert not detect_media_reference("read the textbook chapter")
    assert detect_diagram_reference("see Figure 3 below")
    assert detect_diagram_reference("the flowchart shows the process")
    assert not detect_diagram_reference("a list of vocabulary terms")


def test_ib5_injection_deploys_types_when_flag_on(monkeypatch):
    """Flag on + a procedure source → worked_example reaches the page plan."""
    from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES

    monkeypatch.setenv(ENV_NEW_BLOCK_TYPES, "1")
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=[{"text": "Step 1: isolate the variable. Step 2: divide both sides."}],
        provider=None,  # fixed-plan fallback path; injection still runs
    )
    deployed = {b["block_type"] for b in plan.selected}
    assert "hook" in deployed
    assert "worked_example" in deployed


def test_ib5_injection_noop_when_flag_off(monkeypatch):
    """Flag off → none of the four IB5 types are injected (byte-stability)."""
    from lib.generation.new_block_types import ENV_NEW_BLOCK_TYPES

    monkeypatch.delenv(ENV_NEW_BLOCK_TYPES, raising=False)
    plan = plan_week_blocks(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=[{"text": "Step 1: isolate the variable. Step 2: divide both sides."}],
        provider=None,
    )
    deployed = {b["block_type"] for b in plan.selected}
    assert not ({"hook", "multimedia", "worked_example", "diagram"} & deployed)


# --------------------------------------------------------------------------- #
# IB7 — planner pedagogy (Bloom-climb / lifecycle / spacing / ceiling / heading
# / local seat / decision-capture). Each new pass is default-OFF → identity, so
# the byte-stability assertions below guard the off path.
# --------------------------------------------------------------------------- #

from lib.generation.block_planner import (  # noqa: E402
    _BLOOM_CEILING_ENV,
    _BLOOM_CLIMB_ENV,
    _DEFAULT_PLANNER_MAX_TOKENS,
    _ENV_PLANNER_MAX_TOKENS,
    _LIFECYCLE_ENV,
    _SPACING_ENV,
    _apply_bloom_ceilings,
    _apply_bloom_climb,
    _apply_spacing,
    _build_prompt,
    _ensure_lifecycle_endpoints,
    _resolve_block_plan_max_tokens,
    _resolve_planner_model,
    _source_text_blob,
    build_planner_provider,
)
from lib.generation.block_catalog import load_block_catalog  # noqa: E402
from lib.ontology.bloom import BLOOM_LEVELS  # noqa: E402


def _all_ib7_off(monkeypatch):
    for env in (_BLOOM_CLIMB_ENV, _LIFECYCLE_ENV, _SPACING_ENV, _BLOOM_CEILING_ENV):
        monkeypatch.delenv(env, raising=False)


def _ib7_catalog_by_type():
    return {
        str(e.get("block_type")): e
        for e in load_block_catalog() if e.get("block_type")
    }


def _block_types():
    return __import__(
        "Courseforge.scripts.blocks", fromlist=["BLOCK_TYPES"]
    ).BLOCK_TYPES


# ---- IB7.1 — source-chunk heading threaded into the digest ---------------- #
def test_ib7_heading_in_prompt_and_blob():
    chunks = [{"id": "c1", "text": "Compare proper and improper fractions.",
               "heading": "Types of Fractions"}]
    prompt = _build_prompt(
        terminal_objective=_TO,
        chapter_objectives=_COS,
        source_chunks=chunks,
        catalog=load_block_catalog(),
        budget=(5, 24),
    )
    assert "[Types of Fractions] " in prompt
    blob = _source_text_blob(chunks)
    assert "Types of Fractions" in blob


# ---- IB7.2 — license-clean local planner seat ----------------------------- #
def test_ib7_local_seat_model_resolution(monkeypatch):
    monkeypatch.delenv("ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    assert _resolve_planner_model("local") == "qwen2.5:14b-instruct-q4_K_M"
    monkeypatch.setenv("ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL", "qwen2.5:32b")
    assert _resolve_planner_model("local") == "qwen2.5:32b"


def test_ib7_local_seat_constructs_without_nvidia_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    prov = build_planner_provider(provider="local")
    assert prov is not None
    assert getattr(prov, "_provider", "") == "local"


# ---- big-model-overflow-fix-2026-07 — planner max_tokens env override ----- #
def test_resolve_block_plan_max_tokens_default_when_env_unset(monkeypatch):
    """Env unset resolves to the 4096 default (bumped from the legacy 2048;
    a truncated week plan corrupts the whole week upstream of authoring)."""
    monkeypatch.delenv(_ENV_PLANNER_MAX_TOKENS, raising=False)
    assert _DEFAULT_PLANNER_MAX_TOKENS == 4096
    assert _resolve_block_plan_max_tokens(None) == 4096


def test_resolve_block_plan_max_tokens_env_positive_int(monkeypatch):
    monkeypatch.setenv(_ENV_PLANNER_MAX_TOKENS, "8192")
    assert _resolve_block_plan_max_tokens(None) == 8192


@pytest.mark.parametrize("bad", ["", "  ", "not-an-int", "0", "-5", "3.5"])
def test_resolve_block_plan_max_tokens_garbage_falls_back(monkeypatch, bad):
    """Garbage / non-positive env → the 4096 default (parse-with-fallback)."""
    monkeypatch.setenv(_ENV_PLANNER_MAX_TOKENS, bad)
    assert _resolve_block_plan_max_tokens(None) == 4096


def test_resolve_block_plan_max_tokens_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv(_ENV_PLANNER_MAX_TOKENS, "8192")
    assert _resolve_block_plan_max_tokens(4096) == 4096


def test_build_planner_provider_threads_resolved_max_tokens(monkeypatch):
    """The resolved cap reaches the constructed planner provider (env path)
    and an explicit kwarg still wins."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("ED4ALL_DYNAMIC_BLOCK_PLAN_MODEL", raising=False)
    monkeypatch.setenv("LOCAL_SYNTHESIS_MODEL", "qwen2.5:14b-instruct-q4_K_M")
    monkeypatch.setenv(_ENV_PLANNER_MAX_TOKENS, "6000")
    prov = build_planner_provider(provider="local")
    assert prov is not None and prov._max_tokens == 6000
    prov2 = build_planner_provider(provider="local", max_tokens=1234)
    assert prov2 is not None and prov2._max_tokens == 1234


# ---- IB7.3 — programmatic Bloom-climb re-sort ----------------------------- #
def test_ib7_bloom_climb_resort(monkeypatch):
    monkeypatch.setenv(_BLOOM_CLIMB_ENV, "1")
    selected = [
        {"block_type": "summary_takeaway", "page_type": "summary",
         "target_co_ids": [], "target_bloom": "understand"},
        {"block_type": "self_check_question", "page_type": "self_check",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-02"], "target_bloom": "analyze"},
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-03"], "target_bloom": "understand"},
        {"block_type": "vocab_card", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "remember"},
        {"block_type": "hook", "page_type": "overview",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "example", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "scenario", "page_type": "application",
         "target_co_ids": ["CO-02"], "target_bloom": "analyze"},
    ]
    out = _apply_bloom_climb(selected=selected)
    types = [b["block_type"] for b in out]
    assert types[0] == "hook"
    assert types[-1] == "summary_takeaway"
    assert types.index("vocab_card") < types.index("concept")
    concept_blooms = [
        b["target_bloom"] for b in out if b["block_type"] == "concept"
    ]
    idxs = [BLOOM_LEVELS.index(x) for x in concept_blooms]
    assert idxs == sorted(idxs)
    assert types.index("example") < types.index("scenario")
    assert types.index("scenario") < types.index("self_check_question")


def test_ib7_bloom_climb_identity_when_off(monkeypatch):
    monkeypatch.delenv(_BLOOM_CLIMB_ENV, raising=False)
    selected = [
        {"block_type": "summary_takeaway", "page_type": "summary",
         "target_co_ids": [], "target_bloom": "understand"},
        {"block_type": "hook", "page_type": "overview",
         "target_co_ids": [], "target_bloom": "understand"},
    ]
    out = _apply_bloom_climb(selected=selected)
    assert [b["block_type"] for b in out] == ["summary_takeaway", "hook"]


# ---- IB7.4 — lifecycle open/close + slot-edit escalation ------------------ #
def test_ib7_lifecycle_opens_and_closes(monkeypatch):
    monkeypatch.setenv(_LIFECYCLE_ENV, "1")
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "example", "page_type": "content",
         "target_co_ids": ["CO-02"], "target_bloom": "apply"},
    ]
    out = _ensure_lifecycle_endpoints(
        selected=selected, chapter_objectives=_COS,
        catalog_by_type=_ib7_catalog_by_type(), block_types=_block_types(),
    )
    types = [b["block_type"] for b in out]
    assert types[0] in {"hook", "objective", "prereq_set"}
    assert types[-1] in {"summary_takeaway", "recap", "checklist",
                         "reflection_prompt"}
    assert out[0]["target_co_ids"] == ["CO-01"]


def test_ib7_lifecycle_slot_edit_before_type_swap(monkeypatch):
    monkeypatch.setenv(_LIFECYCLE_ENV, "1")
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "analyze"},
        {"block_type": "objective", "page_type": "overview",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "summary_takeaway", "page_type": "summary",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
    ]
    out = _ensure_lifecycle_endpoints(
        selected=selected, chapter_objectives=_COS,
        catalog_by_type=_ib7_catalog_by_type(), block_types=_block_types(),
    )
    over = next(b for b in out if b["block_type"] == "concept")
    assert over["block_type"] == "concept"
    assert isinstance(over.get("anatomy_slot_weights"), dict)
    assert over["anatomy_slot_weights"].get("interaction") == "heavy"


def test_ib7_lifecycle_identity_when_off(monkeypatch):
    monkeypatch.delenv(_LIFECYCLE_ENV, raising=False)
    selected = [{"block_type": "concept", "page_type": "content",
                 "target_co_ids": ["CO-01"], "target_bloom": "understand"}]
    out = _ensure_lifecycle_endpoints(
        selected=selected, chapter_objectives=_COS,
        catalog_by_type=_ib7_catalog_by_type(), block_types=_block_types(),
    )
    assert out == selected


# ---- IB7.5a — within-module temporal spacing ------------------------------ #
def test_ib7_spacing_moves_adjacent_check(monkeypatch):
    monkeypatch.setenv(_SPACING_ENV, "1")
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "self_check_question", "page_type": "self_check",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-02"], "target_bloom": "understand"},
    ]
    out = _apply_spacing(selected=selected)
    types = [b["block_type"] for b in out]
    chk = types.index("self_check_question")
    assert not (
        out[chk - 1]["block_type"] == "concept"
        and set(out[chk - 1].get("target_co_ids") or []) == {"CO-01"}
    )


def test_ib7_spacing_identity_when_off(monkeypatch):
    monkeypatch.delenv(_SPACING_ENV, raising=False)
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
        {"block_type": "self_check_question", "page_type": "self_check",
         "target_co_ids": ["CO-01"], "target_bloom": "apply"},
    ]
    out = _apply_spacing(selected=selected)
    assert [b["block_type"] for b in out] == ["concept", "self_check_question"]


# ---- IB7.6 — per-type Bloom-range ceiling re-route ------------------------ #
def test_ib7_bloom_ceiling_reroute(monkeypatch):
    monkeypatch.setenv(_BLOOM_CEILING_ENV, "1")
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "analyze"},
    ]
    out = _apply_bloom_ceilings(
        selected=selected, catalog_by_type=_ib7_catalog_by_type(),
        block_types=_block_types(),
    )
    assert out[0]["block_type"] in {"scenario", "problem", "assessment_item"}
    assert out[0]["target_co_ids"] == ["CO-01"]


def test_ib7_bloom_ceiling_in_range_kept(monkeypatch):
    monkeypatch.setenv(_BLOOM_CEILING_ENV, "1")
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "understand"},
    ]
    out = _apply_bloom_ceilings(
        selected=selected, catalog_by_type=_ib7_catalog_by_type(),
        block_types=_block_types(),
    )
    assert out[0]["block_type"] == "concept"


def test_ib7_bloom_ceiling_identity_when_off(monkeypatch):
    monkeypatch.delenv(_BLOOM_CEILING_ENV, raising=False)
    selected = [
        {"block_type": "concept", "page_type": "content",
         "target_co_ids": ["CO-01"], "target_bloom": "analyze"},
    ]
    out = _apply_bloom_ceilings(
        selected=selected, catalog_by_type=_ib7_catalog_by_type(),
        block_types=_block_types(),
    )
    assert out[0]["block_type"] == "concept"


# ---- IB7.9 — DecisionCapture fires (success + fallback) ------------------- #
def test_ib7_decision_capture_fires_success(monkeypatch):
    monkeypatch.setenv(_BLOOM_CLIMB_ENV, "1")
    monkeypatch.setenv(_LIFECYCLE_ENV, "1")
    cap = _RecordingCapture()
    payload = _plan_payload([
        {"block_type": "concept", "target_co_ids": ["CO-01"],
         "page_type": "content", "content_focus": "x"},
        {"block_type": "example", "target_co_ids": ["CO-02"],
         "page_type": "content", "content_focus": "y"},
        {"block_type": "scenario", "target_co_ids": ["CO-03"],
         "page_type": "application", "content_focus": "z"},
    ])
    plan = plan_week_blocks(
        terminal_objective=_TO, chapter_objectives=_COS,
        provider=_MockProvider(payload), capture=cap,
    )
    assert plan.fallback_used is False
    events = [e for e in cap.events if e.get("decision_type") == "block_plan"]
    assert len(events) == 1
    ev = events[0]
    assert len(ev["rationale"]) >= 20
    assert "TO-01" in ev["rationale"]
    assert "climb=" in ev["rationale"]
    feats = ev["ml_features"]
    assert feats["terminal_objective_id"] == "TO-01"
    assert "n_blocks" in feats
    for key in (
        "bloom_climb_applied", "lifecycle_opened", "lifecycle_closed",
        "spacing_moves", "bloom_ceiling_reroutes", "slot_weight_edits",
        "planner_seat", "source_headings_present",
    ):
        assert key in feats
    assert feats["bloom_climb_applied"] is True


def test_ib7_decision_capture_fires_on_fallback():
    cap = _RecordingCapture()
    plan = plan_week_blocks(
        terminal_objective=_TO, chapter_objectives=_COS,
        provider=None, capture=cap,
    )
    assert plan.fallback_used is True
    events = [e for e in cap.events if e.get("decision_type") == "block_plan"]
    assert len(events) == 1
    assert len(events[0]["rationale"]) >= 20
    assert "IB7" in events[0]["rationale"]


def test_ib7_all_passes_identity_when_all_flags_off(monkeypatch):
    _all_ib7_off(monkeypatch)
    payload = _plan_payload([
        {"block_type": "summary_takeaway", "target_co_ids": ["CO-01"],
         "page_type": "summary", "content_focus": "a"},
        {"block_type": "concept", "target_co_ids": ["CO-02"],
         "page_type": "content", "content_focus": "b"},
        {"block_type": "self_check_question", "target_co_ids": ["CO-03"],
         "page_type": "self_check", "content_focus": "c"},
    ])
    plan = plan_week_blocks(
        terminal_objective=_TO, chapter_objectives=_COS,
        provider=_MockProvider(payload),
    )
    types = [b["block_type"] for b in plan.selected]
    assert all(b.get("anatomy_slot_weights") is None for b in plan.selected)
    assert types
    assert set(plan.page_plan) == set(CANONICAL_PAGE_TYPES)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
