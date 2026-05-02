"""End-to-end integration test for the Phase 3 two-pass router (Subtask 58).

Exercises the three new Phase-3 surfaces together against a mini course
fixture (1 course / 2 weeks / 4 block types):

1. Outline tier — ``OutlineProvider`` mocked to return canned JSON.
2. Inter-tier validation — Block-input validator chain in
   ``Courseforge.router.inter_tier_gates`` (BlockCurieAnchoringValidator,
   BlockContentTypeValidator, BlockPageObjectivesValidator,
   BlockSourceRefValidator).
3. Rewrite tier — ``RewriteProvider`` mocked to return canned HTML.

Exercise path notes (limitation documentation):

- The plan calls for running the three new workflow phases via
  ``WorkflowRunner.run_workflow(..., workflow_id="textbook_to_course")``
  with ``COURSEFORGE_TWO_PASS=true``. The new phases
  (``content_generation_outline`` / ``inter_tier_validation`` /
  ``content_generation_rewrite``) are declared in
  ``config/workflows.yaml`` and the param-routing tables in
  ``MCP/core/workflow_runner.py`` (lines 113–143, 213–222), but the
  underlying tool-level Python hooks for these phases are not yet
  wired in ``MCP/tools/pipeline_tools.py``. Running the full workflow
  runner today would dispatch the new phases as Wave-74-style subagent
  calls, which is not the surface we want to assert on.
- This test therefore exercises the routing end-to-end at the
  ``CourseforgeRouter.route_all`` level — the layer that the future
  ``_run_inter_tier_validation`` / ``_run_content_generation_outline``
  / ``_run_content_generation_rewrite`` hooks will drive when they
  land. Per Worker H's flagged followup, ``route_all`` calls
  ``route(block, tier=...)`` directly rather than routing through
  ``route_with_self_consistency``; this test thus does NOT exercise
  the self-consistency loop. Subtask 58's scope is the pass / fail /
  touch-chain end-to-end contract; self-consistency coverage lives
  in ``Courseforge/router/tests/test_self_consistency.py``.

Assertions (per plan §M Subtask 58):

- Every emitted block carries ``Touch`` entries for both tiers
  (``outline`` + ``rewrite``), unless the outline tier failed
  (failed-outline blocks have an ``escalation_marker`` set and a
  single outline-tier Touch only — verified per the plan's "failed
  blocks do NOT have a rewrite-tier Touch" assertion).
- Every CURIE declared in the outline payload is preserved on the
  final HTML emitted by the rewrite tier.
- All decision-capture events validate against
  ``schemas/events/decision_event.schema.json`` (lenient mode).
- The ``inter_tier_validation`` step emits both
  ``blocks_validated_path`` and ``blocks_failed_path`` outputs.
- Failed blocks are detected via the ``escalation_marker`` field
  (Worker J's deviation: Block has no ``status`` field; failed blocks
  carry ``escalation_marker="structural_unfixable"`` instead).

Subtask 60 extends this file with a strict-mode decision-event test.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_SCRIPTS_DIR = PROJECT_ROOT / "Courseforge" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from blocks import Block, Touch  # noqa: E402

from Courseforge.router.inter_tier_gates import (  # noqa: E402
    BlockContentTypeValidator,
    BlockCurieAnchoringValidator,
    BlockPageObjectivesValidator,
    BlockSourceRefValidator,
)
from Courseforge.router.router import CourseforgeRouter  # noqa: E402


# ---------------------------------------------------------------------------
# Mock providers — return canned payloads with no httpx in flight.
# ---------------------------------------------------------------------------


class _MockOutlineProvider:
    """Returns a canned outline-dict + a tier=outline Touch.

    Records the dispatched ``block.block_id`` so tests can assert the
    full per-block invocation set.
    """

    def __init__(
        self,
        *,
        canned_outlines: Dict[str, Dict[str, Any]],
        capture: Optional[Any] = None,
        fail_for: Optional[set] = None,
    ) -> None:
        self.canned_outlines = canned_outlines
        self._capture = capture
        self._fail_for = fail_for or set()
        self.calls: List[str] = []
        self._provider = "local"
        self._model = "qwen2.5:7b-instruct-q4_K_M"

    def generate_outline(
        self,
        block: Block,
        *,
        source_chunks: List[Any],
        objectives: List[Any],
    ) -> Block:
        self.calls.append(block.block_id)
        if block.block_id in self._fail_for:
            raise RuntimeError(
                f"_MockOutlineProvider: scripted failure for {block.block_id}"
            )
        payload = self.canned_outlines[block.block_id]
        if self._capture is not None:
            self._capture.log_decision(
                decision_type="block_outline_call",
                decision=f"outline_call:{block.block_type}:{block.block_id}:success",
                rationale=(
                    f"block_id={block.block_id}; block_type={block.block_type}; "
                    f"page_id={block.page_id}; provider={self._provider}; "
                    f"model={self._model}; output_chars={len(json.dumps(payload))}; "
                    f"retry_count=0; attempts=1; success=True"
                ),
            )
        touch = Touch(
            model=self._model,
            provider="local",
            tier="outline",
            timestamp="2026-05-02T00:00:00Z",
            decision_capture_id="in-memory:0",
            purpose="draft",
        )
        new_block = dataclasses.replace(block, content=payload)
        return new_block.with_touch(touch)


class _MockRewriteProvider:
    """Returns a canned HTML string that preserves every CURIE in the outline."""

    def __init__(
        self,
        *,
        canned_html_by_block_id: Dict[str, str],
        capture: Optional[Any] = None,
    ) -> None:
        self.canned_html = canned_html_by_block_id
        self._capture = capture
        self.calls: List[str] = []
        self._provider = "local"
        self._model = "claude-sonnet-4-6-stub"

    def generate_rewrite(
        self,
        block: Block,
        *,
        source_chunks: List[Any],
        objectives: List[Any],
    ) -> Block:
        self.calls.append(block.block_id)
        html = self.canned_html.get(
            block.block_id,
            "<section><p>Default rewrite stub.</p></section>",
        )
        if self._capture is not None:
            self._capture.log_decision(
                decision_type="block_rewrite_call",
                decision=f"rewrite_call:{block.block_type}:{block.block_id}:success",
                rationale=(
                    f"block_id={block.block_id}; block_type={block.block_type}; "
                    f"page_id={block.page_id}; provider={self._provider}; "
                    f"model={self._model}; output_chars={len(html)}; "
                    f"retry_count=0; attempts=1; success=True"
                ),
            )
        touch = Touch(
            model=self._model,
            provider="local",
            tier="rewrite",
            timestamp="2026-05-02T00:01:00Z",
            decision_capture_id="in-memory:1",
            purpose="pedagogical_depth",
        )
        new_block = dataclasses.replace(block, content=html)
        return new_block.with_touch(touch)


class _FakeCapture:
    """Capture stub that records events for downstream schema validation."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        # Mirror the canonical event-shape required by the strict schema —
        # ``run_id`` / ``timestamp`` / ``operation`` / ``decision_type`` /
        # ``decision`` / ``rationale`` are required top-level fields per
        # ``schemas/events/decision_event.schema.json``. The production
        # ``DecisionCapture._write_with_facade`` path writes those for
        # every event; the test seeds them here so the strict-validation
        # test (Subtask 60) exercises them without spinning up a real
        # capture instance.
        record = dict(kwargs)
        record.setdefault("course_id", "TWOPASS-MINI")
        record.setdefault("phase", "courseforge-content-generator")
        record.setdefault("tool", "courseforge")
        record.setdefault("timestamp", "2026-05-02T00:00:00Z")
        record.setdefault("run_id", "TWOPASS_MINI_20260502_000000")
        # Operation: derive a stable label from decision_type.
        decision_type = record.get("decision_type", "decide")
        record.setdefault("operation", f"decide_{decision_type}")
        self.events.append(record)


# ---------------------------------------------------------------------------
# Fixture course — 1 course, 2 weeks, 4 block types
# ---------------------------------------------------------------------------


def _build_mini_course() -> Tuple[List[Block], Dict[str, Dict[str, Any]], Dict[str, str], List[Dict[str, Any]]]:
    """Construct the mini-course fixture.

    Returns:
        - ``blocks``: ordered Block list (4 outline-shaped Blocks
          across 2 pages = 2 weeks).
        - ``outlines``: mock-outline payload keyed by ``block_id``.
        - ``htmls``: mock-rewrite HTML keyed by ``block_id``.
        - ``objectives``: shared canonical objective list.
    """
    objectives = [
        {"id": "TO-01", "statement": "Define the central concept."},
        {"id": "CO-02", "statement": "Explain the worked example."},
    ]

    # Week 1 / page 1 — concept block + assessment_item.
    # Week 2 / page 2 — example block + summary_takeaway.
    block_specs = [
        ("week01_module01#concept_intro_0", "concept", "week01_module01", "TO-01"),
        ("week01_module01#assessment_item_q1_1", "assessment_item", "week01_module01", "TO-01"),
        ("week02_module01#example_worked_0", "example", "week02_module01", "CO-02"),
        ("week02_module01#summary_takeaway_recap_1", "summary_takeaway", "week02_module01", "CO-02"),
    ]
    blocks: List[Block] = []
    outlines: Dict[str, Dict[str, Any]] = {}
    htmls: Dict[str, str] = {}

    for idx, (block_id, block_type, page_id, obj_ref) in enumerate(block_specs):
        b = Block(
            block_id=block_id,
            block_type=block_type,
            page_id=page_id,
            sequence=idx,
            content="",
            objective_ids=(obj_ref,),
            source_ids=("dart:slug#blk1",),
        )
        blocks.append(b)
        # Map block_type → a content_type drawn from the canonical
        # ChunkType taxonomy enforced by ``BlockContentTypeValidator``
        # (``schemas/taxonomies/content_type.json::ChunkType.enum``).
        # The 8 valid values are: assessment_item, common_pitfall,
        # example, exercise, explanation, overview, problem_solution,
        # procedure, real_world_scenario, summary.
        content_type = {
            "concept": "explanation",
            "example": "example",
            "assessment_item": "assessment_item",
            "summary_takeaway": "summary",
        }.get(block_type, "explanation")
        outline_payload: Dict[str, Any] = {
            "block_id": block_id,
            "block_type": block_type,
            "content_type": content_type,
            "bloom_level": "understand",
            "objective_refs": [obj_ref],
            "curies": ["sh:NodeShape", "rdf:type"],
            "key_claims": [
                f"The {block_type} explains sh:NodeShape via rdf:type.",
            ],
            "section_skeleton": (
                [{"heading": "Definition"}] if block_type != "summary_takeaway"
                else [{"heading": "Recap"}]
            ),
            "source_refs": [{"sourceId": "dart:slug#blk1", "role": "primary"}],
            "structural_warnings": [],
        }
        if block_type == "assessment_item":
            outline_payload["stem"] = (
                f"Which CURIE specifies node shape constraints (objective {obj_ref})?"
            )
            outline_payload["answer_key"] = (
                f"sh:NodeShape (TO-01 / {obj_ref})"
            )
        outlines[block_id] = outline_payload
        # The rewrite-tier HTML preserves both CURIEs verbatim so the
        # CURIE-survival assertion below holds.
        htmls[block_id] = (
            "<section data-cf-source-ids=\"dart:slug#blk1\">"
            f"<h2>{block_type.title()}</h2>"
            f"<p>This block grounds <code>sh:NodeShape</code> via "
            f"<code>rdf:type</code> against {obj_ref}.</p>"
            "</section>"
        )

    return blocks, outlines, htmls, objectives


# ---------------------------------------------------------------------------
# Validator chain runner — emulates the inter_tier_validation phase.
# ---------------------------------------------------------------------------


def _run_inter_tier_validation(
    outline_blocks: List[Block],
    *,
    objectives: List[Dict[str, Any]],
    valid_source_ids: List[str],
    output_dir: Path,
) -> Dict[str, Any]:
    """Mirror the workflow's ``inter_tier_validation`` phase.

    Runs the four Block-input validators against the outline-tier
    Block list, partitions into pass/fail by block_id, and writes
    JSON sidecars at ``blocks_validated_path`` / ``blocks_failed_path``.
    Returns the canonical phase-output dict shape (matches
    ``MCP/core/workflow_runner.py::_LEGACY_PHASE_OUTPUT_KEYS["inter_tier_validation"]``).
    """
    objective_ids = [obj["id"] for obj in objectives]

    validators = [
        (
            BlockCurieAnchoringValidator(),
            {"blocks": outline_blocks, "gate_id": "outline_curie_anchoring"},
        ),
        (
            BlockContentTypeValidator(),
            {"blocks": outline_blocks, "gate_id": "outline_content_type"},
        ),
        (
            BlockPageObjectivesValidator(),
            {
                "blocks": outline_blocks,
                "gate_id": "outline_page_objectives",
                # Direct seed via the validator's ``valid_objective_ids``
                # input seam (see inter_tier_gates.py:379-381).
                "valid_objective_ids": objective_ids,
            },
        ),
        (
            BlockSourceRefValidator(),
            {
                "blocks": outline_blocks,
                "gate_id": "outline_source_refs",
                "valid_source_ids": valid_source_ids,
            },
        ),
    ]

    failed_block_ids: set = set()
    for validator, inputs in validators:
        result = validator.validate(inputs)
        if not result.passed:
            for issue in result.issues:
                if issue.location and issue.location in {b.block_id for b in outline_blocks}:
                    failed_block_ids.add(issue.location)

    validated = [b for b in outline_blocks if b.block_id not in failed_block_ids]
    failed = [b for b in outline_blocks if b.block_id in failed_block_ids]

    output_dir.mkdir(parents=True, exist_ok=True)
    validated_path = output_dir / "blocks_validated.json"
    failed_path = output_dir / "blocks_failed.json"
    validated_path.write_text(
        json.dumps([b.block_id for b in validated], indent=2)
    )
    failed_path.write_text(
        json.dumps([b.block_id for b in failed], indent=2)
    )
    return {
        "blocks_validated_path": str(validated_path),
        "blocks_failed_path": str(failed_path),
        "validated_blocks": validated,
        "failed_blocks": failed,
    }


# ---------------------------------------------------------------------------
# Schema validator helper — used by the strict-mode test (Subtask 60).
# ---------------------------------------------------------------------------


def _load_decision_event_schema() -> Dict[str, Any]:
    schema_path = (
        PROJECT_ROOT / "schemas" / "events" / "decision_event.schema.json"
    )
    return json.loads(schema_path.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_two_pass_end_to_end_yields_outline_and_rewrite_touches(
    monkeypatch, tmp_path
):
    """End-to-end: route 4 mini-course blocks through outline → inter-tier →
    rewrite. Assert every block has both an outline-tier and a
    rewrite-tier Touch on its ``touched_by`` chain, every declared
    CURIE survives to the final HTML, and the inter-tier phase emits
    both ``blocks_validated_path`` and ``blocks_failed_path``."""
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.delenv("DECISION_VALIDATION_STRICT", raising=False)

    blocks, outlines, htmls, objectives = _build_mini_course()
    capture = _FakeCapture()
    outline_provider = _MockOutlineProvider(
        canned_outlines=outlines, capture=capture
    )
    rewrite_provider = _MockRewriteProvider(
        canned_html_by_block_id=htmls, capture=capture
    )
    router = CourseforgeRouter(
        outline_provider=outline_provider,
        rewrite_provider=rewrite_provider,
        capture=capture,
    )

    # Pass 1: outline tier (driven by route_all's first internal pass).
    # We dispatch outline-only here so the inter-tier validator sees
    # outline-shaped Blocks (route_all's internal pass-2 currently
    # collapses validate + rewrite together; per the plan's documented
    # limitation we drive the three phases explicitly here).
    outlined_blocks: List[Block] = []
    for block in blocks:
        outlined = router.route(
            block,
            tier="outline",
            source_chunks=[],
            objectives=objectives,
        )
        outlined_blocks.append(outlined)

    assert len(outline_provider.calls) == 4
    assert all(
        isinstance(b.content, dict) for b in outlined_blocks
    ), "outline-tier Blocks must carry dict content"

    # Phase 2: inter-tier validation.
    inter_outputs = _run_inter_tier_validation(
        outlined_blocks,
        objectives=objectives,
        valid_source_ids=["dart:slug#blk1"],
        output_dir=tmp_path / "inter_tier",
    )
    assert "blocks_validated_path" in inter_outputs
    assert "blocks_failed_path" in inter_outputs
    assert Path(inter_outputs["blocks_validated_path"]).exists()
    assert Path(inter_outputs["blocks_failed_path"]).exists()

    validated_blocks = inter_outputs["validated_blocks"]
    failed_blocks = inter_outputs["failed_blocks"]
    # Mini-course outlines are constructed to pass every gate so the
    # pass-list captures every block.
    assert len(validated_blocks) == 4
    assert len(failed_blocks) == 0

    # Phase 3: rewrite tier — only for validated blocks.
    rewritten_blocks: List[Block] = []
    for block in validated_blocks:
        rewritten = router.route(
            block,
            tier="rewrite",
            source_chunks=[],
            objectives=objectives,
        )
        rewritten_blocks.append(rewritten)

    assert len(rewritten_blocks) == 4
    for b in rewritten_blocks:
        # 1. Both tiers emitted a touch on this block.
        tiers = {t.tier for t in b.touched_by}
        assert "outline" in tiers
        assert "rewrite" in tiers
        # 2. The rewritten block carries HTML content.
        assert isinstance(b.content, str)
        assert "<section" in b.content
        # 3. Every CURIE declared in the outline survives to the HTML.
        outline_payload = outlines[b.block_id]
        for curie in outline_payload["curies"]:
            assert curie in b.content, (
                f"CURIE {curie!r} declared in outline for {b.block_id} "
                f"missing from final HTML"
            )

    # 4. Decision events: at minimum, one outline-call + one rewrite-call
    # per block (8 events) plus per-block router events. Verify the
    # outline + rewrite call types are present.
    decision_types = {e["decision_type"] for e in capture.events}
    assert "block_outline_call" in decision_types
    assert "block_rewrite_call" in decision_types

    # 5. Lenient-schema validation: each event has the required
    # top-level keys (decision_type, decision, rationale).
    for event in capture.events:
        assert event["decision_type"]
        assert event["decision"]
        assert event["rationale"]
        assert len(event["rationale"]) >= 20


def test_two_pass_failed_outline_block_skips_rewrite_tier(
    monkeypatch, tmp_path
):
    """A block that fails the inter-tier validators (e.g. its
    ``content_type`` is outside the canonical taxonomy) is partitioned
    into ``blocks_failed_path`` and is NOT dispatched to the rewrite
    tier. Verifies the "failed blocks do NOT have a rewrite-tier
    Touch" assertion in plan §M Subtask 58."""
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")

    blocks, outlines, htmls, objectives = _build_mini_course()
    # Sabotage the assessment_item block — its outline declares an
    # invalid content_type so BlockContentTypeValidator fails it.
    bad_block_id = "week01_module01#assessment_item_q1_1"
    outlines[bad_block_id]["content_type"] = "totally-bogus-type"

    capture = _FakeCapture()
    outline_provider = _MockOutlineProvider(
        canned_outlines=outlines, capture=capture
    )
    rewrite_provider = _MockRewriteProvider(
        canned_html_by_block_id=htmls, capture=capture
    )
    router = CourseforgeRouter(
        outline_provider=outline_provider,
        rewrite_provider=rewrite_provider,
        capture=capture,
    )

    outlined_blocks = [
        router.route(
            b, tier="outline", source_chunks=[], objectives=objectives
        )
        for b in blocks
    ]

    inter_outputs = _run_inter_tier_validation(
        outlined_blocks,
        objectives=objectives,
        valid_source_ids=["dart:slug#blk1"],
        output_dir=tmp_path / "inter_tier",
    )
    failed_ids = [b.block_id for b in inter_outputs["failed_blocks"]]
    assert bad_block_id in failed_ids, (
        f"Expected {bad_block_id} to fail inter-tier validation; got "
        f"failed list {failed_ids}"
    )
    validated_blocks = inter_outputs["validated_blocks"]
    failed_blocks = inter_outputs["failed_blocks"]

    # Rewrite tier dispatches only for validated blocks.
    rewritten_blocks = [
        router.route(
            b, tier="rewrite", source_chunks=[], objectives=objectives
        )
        for b in validated_blocks
    ]
    assert bad_block_id not in {b.block_id for b in rewritten_blocks}

    # The failed Block carries an outline-tier Touch but NO
    # rewrite-tier Touch — closes the plan's "failed blocks do NOT
    # have a rewrite-tier Touch" assertion. (Worker J's deviation:
    # we detect the failure via the failed_blocks partition, not via
    # a Block.status field.)
    failed_block = next(b for b in failed_blocks if b.block_id == bad_block_id)
    failed_tiers = {t.tier for t in failed_block.touched_by}
    assert "outline" in failed_tiers
    assert "rewrite" not in failed_tiers, (
        f"Failed block {bad_block_id} unexpectedly carries a rewrite-tier "
        f"Touch ({failed_block.touched_by!r})"
    )


def test_two_pass_outline_dispatch_failure_marks_escalation(
    monkeypatch, tmp_path
):
    """When the outline provider raises for one block, ``route_all``
    captures the failure as ``escalation_marker="outline_budget_exhausted"``
    and skips the rewrite tier for that block. Per Worker J's
    deviation: failed blocks are detected via ``escalation_marker``,
    not a ``status`` field."""
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")

    blocks, outlines, htmls, objectives = _build_mini_course()
    target_id = "week02_module01#example_worked_0"
    capture = _FakeCapture()
    outline_provider = _MockOutlineProvider(
        canned_outlines=outlines,
        capture=capture,
        fail_for={target_id},
    )
    rewrite_provider = _MockRewriteProvider(
        canned_html_by_block_id=htmls, capture=capture
    )
    router = CourseforgeRouter(
        outline_provider=outline_provider,
        rewrite_provider=rewrite_provider,
        capture=capture,
    )

    # Use route_all here so the failure-handling path runs (it's the
    # surface the production runner will drive).
    out = router.route_all(
        blocks,
        source_chunks_by_block_id={},
        objectives=objectives,
    )
    by_id = {b.block_id: b for b in out}
    failed_block = by_id[target_id]
    assert failed_block.escalation_marker == "outline_budget_exhausted"
    failed_tiers = {t.tier for t in failed_block.touched_by}
    assert "rewrite" not in failed_tiers, (
        f"Block {target_id} dispatched to rewrite despite outline-tier "
        f"failure; touches={failed_block.touched_by!r}"
    )

    # Every other block carried both touches.
    for b in out:
        if b.block_id == target_id:
            continue
        tiers = {t.tier for t in b.touched_by}
        assert "outline" in tiers
        assert "rewrite" in tiers


# ---------------------------------------------------------------------------
# Subtask 60: strict-schema decision-event validation
# ---------------------------------------------------------------------------


def test_all_phase3_decision_events_pass_strict_schema_validation(
    monkeypatch, tmp_path
):
    """With ``DECISION_VALIDATION_STRICT=true`` set, every decision
    event captured during the two-pass workflow validates against
    ``schemas/events/decision_event.schema.json``. Closes the
    regression class Wave 120 Phase A re-fixed for the curriculum
    surface.

    The strict validator checks (per schema):
    - ``decision_type`` is in the canonical enum (Phase 3 added
      ``block_outline_call`` / ``block_rewrite_call`` /
      ``block_validation_action`` / ``block_escalation`` to the enum).
    - ``phase`` is in the canonical hyphenated enum.
    - ``course_id`` matches ``^[A-Z][A-Z0-9_-]{1,}$`` (Wave-120 fix).
    - ``run_id`` / ``timestamp`` / ``operation`` / ``decision`` /
      ``rationale`` are non-empty (top-level required fields).
    """
    import jsonschema  # noqa: PLC0415  — soft dep present via pytest deps

    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")
    monkeypatch.setenv("DECISION_VALIDATION_STRICT", "true")

    blocks, outlines, htmls, objectives = _build_mini_course()
    capture = _FakeCapture()
    outline_provider = _MockOutlineProvider(
        canned_outlines=outlines, capture=capture
    )
    rewrite_provider = _MockRewriteProvider(
        canned_html_by_block_id=htmls, capture=capture
    )
    router = CourseforgeRouter(
        outline_provider=outline_provider,
        rewrite_provider=rewrite_provider,
        capture=capture,
    )

    out = router.route_all(
        blocks,
        source_chunks_by_block_id={},
        objectives=objectives,
    )
    assert len(out) == 4
    assert capture.events, "expected at least one captured decision event"

    schema = _load_decision_event_schema()
    validator = jsonschema.Draft202012Validator(schema)
    for idx, event in enumerate(capture.events):
        errors = list(validator.iter_errors(event))
        assert not errors, (
            f"event[{idx}] (decision_type={event.get('decision_type')!r}) "
            f"failed strict schema validation: "
            f"{[(list(e.absolute_path), e.message) for e in errors]}"
        )
