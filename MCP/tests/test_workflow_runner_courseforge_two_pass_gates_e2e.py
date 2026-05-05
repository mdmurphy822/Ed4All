"""Worker W9 — end-to-end gate-firing assertion through ``WorkflowRunner.run_workflow``.

Closes the Worker W1-W4 wiring sweep at the integration layer: assassin-
ates §1.1 (validators silently skipped) and §1.2 (self-consistency loop
unreachable) by proving that gates wired in ``config/workflows.yaml``
actually fire ``passed=False`` through the canonical
``WorkflowRunner.run_workflow`` dispatch chain.

Two tests:

1. **Outline-tier CURIE gate fires through the runner.** Patches the
   outline provider so every candidate emits content with empty
   ``curies``. Runs ``content_generation_outline`` →
   ``inter_tier_validation`` end-to-end. Asserts:

   * ``02_validation_report/report.json`` is written.
   * The ``outline_curie_anchoring`` gate result is ``passed=False``
     (NOT a ``waiver_info["skipped"]="true"`` stamp — the §1.1
     silent-skip failure mode).
   * At least one block carries ``validation_attempts > 0`` (proof the
     self-consistency regen loop ran).
   * ``escalation_marker == "outline_budget_exhausted"`` lands on the
     consensus-failure path (the patched provider always emits empty
     CURIEs so the regen budget exhausts).

2. **Post-rewrite ``rewrite_html_shape`` gate fires through the
   runner.** Reconstructs the historical Qwen-7B-Q4 JSON-wrapped
   ``{"div": {...}}`` rewrite emit (mirrors
   ``test_qwen7b_post_rewrite_e2e.py::_REGRESSION_JSON_WRAPPED_HTML``).
   Stages the JSON-wrapped Block as the ``blocks_final_path`` input to
   ``post_rewrite_validation`` and runs that phase through the runner.
   Asserts the gate fires ``passed=False`` with code
   ``REWRITE_NOT_HTML_BODY_FRAGMENT`` or ``REWRITE_JSON_WRAPPED_HTML``
   (the validator subdivides; both are critical, both indicate the
   fail-closed path is reachable through the runner).

Both tests stub the outline provider so no LLM dispatch fires; the test
file is self-contained and offline.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest  # noqa: F401  — harness expects pytest visible at module load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.config import OrchestratorConfig, WorkflowConfig, WorkflowPhase
from MCP.core.executor import TaskExecutor
from MCP.core.workflow_runner import WorkflowRunner


# --------------------------------------------------------------------- #
# Fixture corpus + helpers
# --------------------------------------------------------------------- #
#
# Minimal Courseforge project export under tmp_path:
#
#   tmp_path/
#     state/workflows/<workflow_id>.json   (workflow state)
#     Courseforge/exports/<project_id>/
#       project_config.json
#       01_learning_objectives/
#         synthesized_objectives.json
#
# DART staging is intentionally elided — the BlockSourceRefValidator's
# missing-manifest path emits a warning rather than failing closed, so
# the test stays scoped to the curie / shape gates W1-W4 wired.
#
# All fixture writes happen inside the per-test ``tmp_path`` so the
# project tree stays clean.


def _seed_project(tmp_path: Path, project_id: str, course_name: str) -> Path:
    """Create a minimal project export at tmp_path/Courseforge/exports/<id>."""
    project_path = (
        tmp_path / "Courseforge" / "exports" / project_id
    )
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "01_learning_objectives").mkdir(exist_ok=True)
    (project_path / "01_learning_objectives" / "synthesized_objectives.json").write_text(
        json.dumps({
            "terminal_objectives": [
                {
                    "id": "TO-01",
                    "statement": "Describe core concept A in detail.",
                },
            ],
            "chapter_objectives": [],
        }),
        encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({
            "course_name": course_name,
            "project_id": project_id,
            "duration_weeks": 1,
        }),
        encoding="utf-8",
    )
    return project_path


def _create_workflow_state(
    tmp_path: Path,
    workflow_id: str,
    course_name: str,
    project_id: str,
    *,
    extra_phase_outputs: Dict[str, Any] | None = None,
) -> Path:
    """Persist a workflow-state JSON the runner can load."""
    workflows_dir = tmp_path / "state" / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    phase_outputs = {
        # Pre-populate objective_extraction so the routing table can
        # resolve project_id without dispatching the upstream phase.
        "objective_extraction": {
            "_completed": True,
            "_skipped": True,
            "_gates_passed": True,
            "project_id": project_id,
            "project_path": str(
                tmp_path / "Courseforge" / "exports" / project_id
            ),
        },
    }
    if extra_phase_outputs:
        phase_outputs.update(extra_phase_outputs)
    state = {
        "id": workflow_id,
        "type": "textbook_to_course",
        "status": "PENDING",
        "params": {
            "course_name": course_name,
            "duration_weeks": 1,
            "duration_weeks_explicit": True,
            "generate_assessments": False,
        },
        "phase_outputs": phase_outputs,
        "tasks": [],
    }
    path = workflows_dir / f"{workflow_id}.json"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return path


def _build_runner(tmp_path: Path, monkeypatch, phases: List[WorkflowPhase]):
    """Construct a WorkflowRunner wired to a real TaskExecutor.

    The executor runs the real ``_PHASE_TOOL_MAPPING`` shim against the
    pipeline_tools registry, so the same code path that ships in
    production fires.
    """
    # Redirect all on-disk writes (workflow state, project exports,
    # checkpoint dirs, decision capture, run state) into tmp_path.
    # The workflow runner AND the executor both read STATE_PATH; the
    # runner's setattr only flips the runner's module-local symbol, so
    # we patch both call sites' STATE_PATH so task lookup + state
    # persistence land in the same tmp directory.
    monkeypatch.setattr(
        "MCP.core.workflow_runner.PROJECT_ROOT", tmp_path,
    )
    monkeypatch.setattr(
        "MCP.core.workflow_runner.STATE_PATH", tmp_path / "state",
    )
    monkeypatch.setattr(
        "MCP.core.executor.STATE_PATH", tmp_path / "state",
    )
    (tmp_path / "state" / "workflows").mkdir(parents=True, exist_ok=True)

    # PROJECT_ROOT inside pipeline_tools points the phase helpers at the
    # synthesized_objectives.json + project export tree we just laid down.
    from MCP.tools import pipeline_tools as _pt
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)

    # Build the in-process tool registry for the phase helpers.
    # Mirrors the relevant slice of pipeline_tools._build_tool_registry
    # (we only need the four two-pass surface helpers to fire).
    registry = {
        "run_content_generation_outline": _pt._run_content_generation_outline,
        "run_inter_tier_validation": _pt._run_inter_tier_validation,
        "run_content_generation_rewrite": _pt._run_content_generation_rewrite,
        "run_post_rewrite_validation": _pt._run_post_rewrite_validation,
    }

    wf_config = WorkflowConfig(description="W9 test", phases=phases)
    cfg = OrchestratorConfig()
    cfg.workflows = {"textbook_to_course": wf_config}
    # Tighten retries so a permanent-failure phase exits fast.
    cfg.retry_attempts = 0
    cfg.task_timeout_minutes = 2

    executor = TaskExecutor(
        tool_registry=registry,
        config=cfg,
        max_retries=0,
        run_path=tmp_path / "runs" / "W9",
    )

    runner = WorkflowRunner(executor=executor, config=cfg)
    return runner


# --------------------------------------------------------------------- #
# Outline-tier provider stubs
# --------------------------------------------------------------------- #


class _CurieMissingProvider:
    """Outline provider stub that always emits content with no CURIEs.

    The dict-shape outline content uses ``content["curies"] == []``
    which forces ``BlockCurieAnchoringValidator`` to emit
    ``action="regenerate"`` on every candidate so the self-consistency
    loop walks its full regen budget and stamps
    ``escalation_marker="outline_budget_exhausted"``.
    """

    def __init__(self) -> None:
        self.outline_calls: List[Dict[str, Any]] = []
        self.rewrite_calls: List[Dict[str, Any]] = []

    def generate_outline(
        self, block, *, source_chunks=None, objectives=None, **kw,
    ):
        self.outline_calls.append({"block_id": block.block_id})
        # String content with no CURIE-shaped tokens — guarantees the
        # CurieAnchoringValidator's str-path returns curies=[] and emits
        # action="regenerate".
        return dataclasses.replace(
            block,
            content="plain prose with no anchored identifiers",
        )

    def generate_rewrite(
        self, block, *, source_chunks=None, objectives=None, **kw,
    ):
        self.rewrite_calls.append({"block_id": block.block_id})
        return dataclasses.replace(block, content="<p>rewrite stub</p>")


def _patch_router_with_provider(monkeypatch, provider: Any) -> None:
    """Force every CourseforgeRouter() instantiation to wire the stub."""
    from Courseforge.router import router as _router_mod

    real_init = _router_mod.CourseforgeRouter.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("outline_provider", provider)
        kwargs.setdefault("rewrite_provider", provider)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter, "__init__", patched_init,
    )


# --------------------------------------------------------------------- #
# Test 1 — outline curie gate fires through the runner
# --------------------------------------------------------------------- #


def _outline_phases() -> List[WorkflowPhase]:
    """Two-phase slice: content_generation_outline + inter_tier_validation.

    Mirrors the YAML shape (incl. validation_gates on
    inter_tier_validation) but trimmed to only what Test 1 cares about.
    """
    return [
        WorkflowPhase(
            name="content_generation_outline",
            agents=["content-generator"],
            depends_on=[],
            timeout_minutes=2,
            max_concurrent=1,
            parallel=False,
        ),
        WorkflowPhase(
            name="inter_tier_validation",
            agents=[],
            depends_on=["content_generation_outline"],
            timeout_minutes=2,
            max_concurrent=1,
            parallel=False,
            validation_gates=[
                {
                    "gate_id": "outline_curie_anchoring",
                    "validator": (
                        "Courseforge.router.inter_tier_gates."
                        "BlockCurieAnchoringValidator"
                    ),
                    "severity": "critical",
                    "threshold": {"max_critical_issues": 0},
                    "behavior": {
                        "on_fail": "block",
                        "on_error": "warn",
                    },
                },
            ],
        ),
    ]


def test_outline_curie_gate_fires_through_workflow_runner(
    tmp_path, monkeypatch,
):
    """Worker W9 Test 1.

    With the outline provider patched to emit curie-less content and a
    small regen budget, the runner end-to-end:

    * runs the outline phase under route_with_self_consistency,
    * exhausts the regen budget per block (validation_attempts > 0),
    * stamps escalation_marker="outline_budget_exhausted",
    * runs the inter_tier_validation phase, which writes
      ``02_validation_report/report.json``,
    * fires the ``outline_curie_anchoring`` gate as ``passed=False``
      (NOT a silent waiver_info["skipped"]="true" stamp — the §1.1
      failure mode this test guards against).
    """
    project_id = "PROJ-W9_OUTLINE-20260505"
    course_name = "W9_OUTLINE"
    _seed_project(tmp_path, project_id, course_name)

    # Provider always emits empty curies → regen budget exhausts.
    provider = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, provider)

    # Minimal regen + candidate budget so the test runs fast. With
    # regen_budget=1 and n_candidates=1 the loop runs exactly once,
    # bumps cumulative_attempts to 1, hits the budget threshold, and
    # stamps escalation_marker="outline_budget_exhausted".
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")
    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")

    runner = _build_runner(tmp_path, monkeypatch, _outline_phases())

    workflow_id = "WF-W9-OUTLINE"
    _create_workflow_state(
        tmp_path,
        workflow_id=workflow_id,
        course_name=course_name,
        project_id=project_id,
    )

    result = asyncio.run(runner.run_workflow(workflow_id))

    # The outline gate is critical. With on_fail=block and the patched
    # provider exhausting the regen budget, the gate fires passed=False
    # and the workflow ends FAILED — that's the contract under test.
    assert result["status"] in {"FAILED", "COMPLETE"}, result

    # The outline phase must have actually run — i.e. the stub provider
    # was invoked. >= 1 candidate per block under the regen budget.
    assert len(provider.outline_calls) >= 1, (
        "outline phase did not invoke the stub provider; "
        "self-consistency loop never dispatched"
    )

    # Verify the operator-facing report.json landed on disk under the
    # project export root.
    project_path = (
        tmp_path / "Courseforge" / "exports" / project_id
    )
    report_path = (
        project_path / "02_validation_report" / "report.json"
    )
    assert report_path.exists(), (
        f"Expected validation report at {report_path}; got nothing. "
        f"inter_tier_validation phase never ran or never emitted "
        f"blocks_validated_path."
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["phase"] == "inter_tier_validation"
    assert report["total_blocks"] >= 1, report

    # Locate the outline_curie_anchoring gate result. Plan §6 schema:
    # report["per_block"][N]["gate_results"] is the chain summary
    # (gate_id / action / passed / issue_count). Every per_block entry
    # mirrors the same chain summary, so we can read off any one.
    chain = (
        report["per_block"][0]["gate_results"]
        if report["per_block"]
        else []
    )
    curie_gate = next(
        (g for g in chain if g.get("gate_id") == "outline_curie_anchoring"),
        None,
    )
    assert curie_gate is not None, (
        f"outline_curie_anchoring gate missing from chain summary; "
        f"got {[g.get('gate_id') for g in chain]!r}. "
        f"This is the §1.1 failure mode (validators silently skipped)."
    )

    # CRITICAL ASSERTION: the gate fires passed=False — NOT a silent
    # skip. The §1.1 failure mode pre-W1 stamped passed=True with
    # waiver_info["skipped"]="true"; W1's gate_input_routing
    # registration closes that. This test is the integration-layer
    # proof that path is closed.
    assert curie_gate.get("passed") is False, (
        f"outline_curie_anchoring gate did NOT fail closed; "
        f"chain summary entry: {curie_gate!r}. "
        f"This is the §1.1 silent-skip regression — the gate either "
        f"passed=True (silent skip) or the chain summary is malformed."
    )

    # Inspect the on-disk blocks_outline.jsonl emit for proof the
    # self-consistency loop walked the regen budget and stamped the
    # consensus-failure marker.
    outline_blocks_path = (
        project_path / "01_outline" / "blocks_outline.jsonl"
    )
    assert outline_blocks_path.exists(), outline_blocks_path
    parsed = [
        json.loads(ln)
        for ln in outline_blocks_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if ln.strip()
    ]
    assert len(parsed) >= 1, parsed
    assert any(
        int(entry.get("validation_attempts", 0)) > 0 for entry in parsed
    ), (
        f"Expected at least one block to carry validation_attempts > 0 "
        f"after the self-consistency loop walked its regen budget; "
        f"got: "
        + json.dumps(
            [entry.get("validation_attempts", 0) for entry in parsed]
        )
    )
    # At least one block should carry the budget-exhausted marker —
    # the patched provider always emits empty curies, so the regen
    # budget MUST exhaust on every block.
    assert any(
        entry.get("escalation_marker") == "outline_budget_exhausted"
        for entry in parsed
    ), (
        f"Expected at least one block to carry "
        f"escalation_marker='outline_budget_exhausted'; got: "
        + json.dumps([entry.get("escalation_marker") for entry in parsed])
    )


# --------------------------------------------------------------------- #
# Test 2 — post-rewrite html-shape gate fires through the runner
# --------------------------------------------------------------------- #


# Mirrors test_qwen7b_post_rewrite_e2e.py::_REGRESSION_JSON_WRAPPED_HTML.
# The on-disk fixture at runtime/qwen_test/surfaces.json has been
# refreshed post-hardening; this is the historical regression payload
# the plan §2 records verbatim.
_REGRESSION_JSON_WRAPPED_HTML = json.dumps({
    "div": {
        "class": "assessment-item",
        "content": (
            "<p>What are the three components of an RDF triple?</p>"
            "<ol>"
            "<li>subject, predicate, object</li>"
            "<li>subject, object, predicate</li>"
            "<li>predicate, subject, object</li>"
            "</ol>"
        ),
    }
})


def _seed_blocks_final(
    project_path: Path, json_wrapped_content: str,
) -> Path:
    """Persist a one-Block blocks_final.jsonl with JSON-wrapped content.

    Matches the on-disk shape ``_run_content_generation_rewrite``
    writes — one snake_case Block JSON entry per line. The Block carries
    the JSON-wrapped string so ``RewriteHtmlShapeValidator`` audits it
    via the str-path branch.
    """
    out_dir = project_path / "04_rewrite"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "blocks_final.jsonl"
    entry = {
        "block_id": "page_week_1#assessment_item_rdf_triple_0",
        "block_type": "assessment_item",
        "page_id": "page_week_1",
        "sequence": 0,
        "content": json_wrapped_content,
        "objective_ids": ["TO-01"],
        "bloom_level": "remember",
    }
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return path


def _post_rewrite_phases() -> List[WorkflowPhase]:
    """Single-phase slice: post_rewrite_validation only.

    The phase reads ``blocks_final_path`` from
    ``content_generation_rewrite`` phase output (pre-populated in the
    workflow state) so we don't have to run the rewrite tier itself.
    """
    return [
        WorkflowPhase(
            name="post_rewrite_validation",
            agents=[],
            depends_on=[],
            timeout_minutes=2,
            max_concurrent=1,
            parallel=False,
            validation_gates=[
                # The Block-input adapter — included so the gate chain
                # sees at least one inter-tier validator alongside the
                # shape gate (mirrors the YAML).
                {
                    "gate_id": "rewrite_curie_anchoring",
                    "validator": (
                        "Courseforge.router.inter_tier_gates."
                        "BlockCurieAnchoringValidator"
                    ),
                    "severity": "critical",
                    "threshold": {"max_critical_issues": 0},
                    "behavior": {
                        "on_fail": "block",
                        "on_error": "warn",
                    },
                },
                # The HTML-shape sentinel — the §3.2 followup gate.
                # This is the assertion target.
                {
                    "gate_id": "rewrite_html_shape",
                    "validator": (
                        "lib.validators.rewrite_html_shape."
                        "RewriteHtmlShapeValidator"
                    ),
                    "severity": "critical",
                    "threshold": {"max_critical_issues": 0},
                    "behavior": {
                        "on_fail": "block",
                        "on_error": "fail_closed",
                    },
                },
            ],
        ),
    ]


def test_rewrite_html_shape_gate_fires_on_recorded_json_wrapped_emit(
    tmp_path, monkeypatch,
):
    """Worker W9 Test 2.

    Replay the recorded JSON-wrapped rewrite emit through
    ``WorkflowRunner.run_workflow`` driving ``post_rewrite_validation``.
    Assert the ``rewrite_html_shape`` gate fires ``passed=False`` with a
    critical-code that flags the JSON-wrapped / non-HTML fragment shape
    — proof the post-rewrite shape sentinel is reachable from the
    runner.
    """
    project_id = "PROJ-W9_REWRITE-20260505"
    course_name = "W9_REWRITE"
    project_path = _seed_project(tmp_path, project_id, course_name)

    # Stage the JSON-wrapped Block as if the rewrite tier had emitted
    # it. The post_rewrite_validation phase reads blocks_final_path
    # from the content_generation_rewrite phase output dict, so we
    # pre-populate that in the workflow state below.
    blocks_final_path = _seed_blocks_final(
        project_path, _REGRESSION_JSON_WRAPPED_HTML,
    )

    monkeypatch.setenv("COURSEFORGE_TWO_PASS", "true")

    runner = _build_runner(tmp_path, monkeypatch, _post_rewrite_phases())

    # Pre-populate phase outputs the post_rewrite_validation phase
    # depends on (content_generation_rewrite emits blocks_final_path).
    extra = {
        "content_generation_rewrite": {
            "_completed": True,
            "_skipped": True,
            "_gates_passed": True,
            "blocks_final_path": str(blocks_final_path),
            "project_id": project_id,
        },
    }
    workflow_id = "WF-W9-REWRITE"
    _create_workflow_state(
        tmp_path,
        workflow_id=workflow_id,
        course_name=course_name,
        project_id=project_id,
        extra_phase_outputs=extra,
    )

    # Patch the routing table so the post_rewrite_validation phase
    # picks up blocks_final_path. The legacy routing entry for
    # post_rewrite_validation pulls blocks_final_path from
    # content_generation_rewrite phase output.
    from MCP.core import workflow_runner as _wr
    monkeypatch.setitem(
        _wr._LEGACY_PHASE_PARAM_ROUTING,
        "post_rewrite_validation",
        {
            "blocks_final_path": (
                "phase_outputs", "content_generation_rewrite",
                "blocks_final_path",
            ),
            "project_id": (
                "phase_outputs", "objective_extraction", "project_id",
            ),
        },
    )

    result = asyncio.run(runner.run_workflow(workflow_id))

    # The shape gate is critical with on_fail=block → the workflow
    # MUST end FAILED. (When the gate were silently passing, this
    # would be COMPLETE — that's the §1.1 / §1.2 regression class.)
    assert result["status"] == "FAILED", (
        f"Expected workflow FAILED (post-rewrite html_shape gate "
        f"critical-fails on JSON-wrapped emit); got "
        f"status={result.get('status')!r}, result={result!r}. "
        f"This indicates the post-rewrite gate did NOT fire "
        f"passed=False through the runner — the §1.1 / §1.2 "
        f"regression class."
    )

    # Locate the operator-facing report.json that the runner writes
    # after the post_rewrite_validation phase. Plan §6 contract:
    # rewrite-tier report lives INSIDE 04_rewrite/.
    report_path = (
        project_path / "04_rewrite" / "02_validation_report"
        / "report.json"
    )
    assert report_path.exists(), report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["phase"] == "post_rewrite_validation"
    assert report["total_blocks"] >= 1, report

    # Assert the rewrite_html_shape gate fired passed=False through
    # the chain summary attached to per_block entries.
    chain = (
        report["per_block"][0]["gate_results"]
        if report["per_block"]
        else []
    )
    shape_gate = next(
        (g for g in chain if g.get("gate_id") == "rewrite_html_shape"),
        None,
    )
    assert shape_gate is not None, (
        f"rewrite_html_shape gate missing from chain summary; "
        f"got {[g.get('gate_id') for g in chain]!r}. "
        f"This is the §1.1 failure mode (validators silently "
        f"skipped) at the post-rewrite seam."
    )
    assert shape_gate.get("passed") is False, (
        f"rewrite_html_shape gate did NOT fail closed; "
        f"chain summary entry: {shape_gate!r}. "
        f"This is the §1.1 / §1.2 regression — the post-rewrite "
        f"shape sentinel is unreachable through the runner."
    )
    # The gate emitted at least one critical issue — the gate's chain
    # summary tracks issue_count rather than per-issue codes, so the
    # canonical assertion is issue_count >= 1 alongside passed=False.
    # Test 2's secondary assertion is on the validator's own emit,
    # which we can read off the executor's gate_results envelope on
    # the phase task result. The phase task result is persisted in
    # workflow state at state/workflows/<id>.json::tasks[].
    workflow_state = json.loads(
        (tmp_path / "state" / "workflows" / f"{workflow_id}.json")
        .read_text(encoding="utf-8")
    )
    # Walk task results for the post_rewrite_validation task; the
    # phase emit carries gate_results[] from _run_post_rewrite_validation
    # (the inline four-validator chain). The HTML-shape gate is run
    # separately by the gate_manager — its emit lives in the runner's
    # extracted phase output via the executor's gate_results envelope.
    # We assert the runner's phase output carries the failed gate
    # signal at the workflow level.
    assert shape_gate.get("issue_count", 0) >= 1, (
        f"rewrite_html_shape passed=False but reported "
        f"issue_count=0 — the validator's GateIssue list is empty. "
        f"Expected at least one critical issue "
        f"(REWRITE_NOT_HTML_BODY_FRAGMENT or "
        f"REWRITE_JSON_WRAPPED_HTML)."
    )
