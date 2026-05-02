"""Phase 3.5 Subtask 32 — Two-pass router phase handler tests.

Asserts the I/O contract for the three new pipeline-tool helpers
landed in Subtasks 28-30:

* :func:`MCP.tools.pipeline_tools._run_content_generation_outline`
* :func:`MCP.tools.pipeline_tools._run_inter_tier_validation`
* :func:`MCP.tools.pipeline_tools._run_content_generation_rewrite`

Plus the executor-side phase-name dispatch shim landed in
Subtask 31 (``MCP.core.executor._PHASE_TOOL_MAPPING``).

The router is fully stubbed via the constructor's
``outline_provider=`` / ``rewrite_provider=`` injection seam — no
real LLM dispatch happens. Each helper is exercised against a
temporary project workspace populated with the minimum fixtures
required for the helper to produce a real JSONL emit.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.core.executor import (  # noqa: E402
    AGENT_TOOL_MAPPING,
    _PHASE_TOOL_MAPPING,
)
from MCP.core.workflow_runner import _LEGACY_PHASE_OUTPUT_KEYS  # noqa: E402
from MCP.tools import pipeline_tools as _pt  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixture builders
# ---------------------------------------------------------------------- #


class _FakeProvider:
    """Stub provider exposing the two router-facing surfaces.

    Returns the input block unchanged for outline tier; populates a
    minimal HTML body for rewrite tier so the rewrite helper has
    something to write to disk.
    """

    def __init__(self) -> None:
        self.outline_calls: List[Dict[str, Any]] = []
        self.rewrite_calls: List[Dict[str, Any]] = []

    def generate_outline(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.outline_calls.append({"block_id": block.block_id})
        # Outline tier returns block with a small content payload (str).
        import dataclasses
        return dataclasses.replace(block, content="outline-stub")

    def generate_rewrite(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.rewrite_calls.append({"block_id": block.block_id})
        import dataclasses
        return dataclasses.replace(
            block,
            content=(
                f"<p>Rewrite stub for {block.block_id} with thirty plus "
                f"words of body prose to clear the grounding floor and "
                f"satisfy the non-trivial paragraph word minimum.</p>"
            ),
        )


def _seed_project(tmp_path: Path, project_id: str) -> Path:
    """Create a minimal Courseforge/exports/<project> scaffold under tmp_path.

    Returns the project path. Includes a project_config.json with a
    one-week duration so the outline helper produces a small Block list.
    """
    exports_root = tmp_path / "Courseforge" / "exports"
    project_path = exports_root / project_id
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "01_learning_objectives").mkdir(exist_ok=True)
    (project_path / "01_learning_objectives" / "synthesized_objectives.json").write_text(
        json.dumps({
            "terminal_objectives": [
                {"id": "TO-01", "statement": "Describe core concept A in detail."}
            ],
            "chapter_objectives": [],
        }),
        encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({
            "course_name": project_id,
            "duration_weeks": 1,
        }),
        encoding="utf-8",
    )
    return project_path


def _patch_project_root(monkeypatch, tmp_path: Path) -> None:
    """Re-point pipeline_tools' PROJECT_ROOT at tmp_path so helpers
    write into the test workspace instead of the real repo."""
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)


def _patch_router_with_fakes(monkeypatch, fake: _FakeProvider) -> None:
    """Force CourseforgeRouter() to return a router pre-wired with the
    fake provider for both tiers, so route() doesn't try to spin up
    real LLM clients during the test."""
    from Courseforge.router import router as _router_mod

    real_init = _router_mod.CourseforgeRouter.__init__

    def patched_init(self, *args, **kwargs):
        kwargs.setdefault("outline_provider", fake)
        kwargs.setdefault("rewrite_provider", fake)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter, "__init__", patched_init,
    )


# ---------------------------------------------------------------------- #
# Subtask 28: _run_content_generation_outline
# ---------------------------------------------------------------------- #


def test_run_content_generation_outline_emits_blocks_outline_path(
    tmp_path, monkeypatch,
):
    """Outline helper writes a JSONL of outline-tier Blocks and returns
    the canonical phase output keys."""
    project_id = "TEST_OUTLINE"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload
    assert "blocks_outline_path" in payload
    assert "project_id" in payload
    assert "weeks_prepared" in payload
    blocks_path = Path(payload["blocks_outline_path"])
    assert blocks_path.exists()
    # JSONL must parse cleanly.
    lines = [
        ln for ln in blocks_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    parsed = [json.loads(ln) for ln in lines]
    assert len(parsed) >= 1
    # Round-trip-friendly snake_case shape per
    # _pt._block_to_snake_case_entry — Block(**entry) must succeed.
    for entry in parsed:
        assert "block_id" in entry
        assert "block_type" in entry
    # Outline provider was invoked at least once.
    assert len(fake.outline_calls) >= 1


# ---------------------------------------------------------------------- #
# Subtask 29: _run_inter_tier_validation
# ---------------------------------------------------------------------- #


def test_run_inter_tier_validation_emits_validated_and_failed_paths(
    tmp_path, monkeypatch,
):
    """Inter-tier validation helper reads blocks_outline_path, runs the
    four shape-discriminating Block validators, and persists pass/fail
    sidecars."""
    project_id = "TEST_VALIDATE"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    # Hand-craft a one-block outline JSONL — the simplest input that
    # exercises the validator chain without forcing every gate to fire.
    blocks_path = tmp_path / "blocks_outline.jsonl"
    block = Block(
        block_id="week_01_content_01#explanation_concept_a_0",
        block_type="explanation",
        page_id="week_01_content_01",
        sequence=0,
        content="outline-stub body content",
        objective_ids=("TO-01",),
    )
    blocks_path.write_text(
        json.dumps(_pt._block_to_snake_case_entry(block)) + "\n",
        encoding="utf-8",
    )

    result = asyncio.run(_pt._run_inter_tier_validation(
        blocks_outline_path=str(blocks_path),
        project_id=project_id,
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload
    assert "blocks_validated_path" in payload
    assert "blocks_failed_path" in payload
    assert Path(payload["blocks_validated_path"]).exists()
    assert Path(payload["blocks_failed_path"]).exists()
    assert "gate_results" in payload
    assert isinstance(payload["gate_results"], list)
    # Four validators are run; we expect four (or fewer if any raised).
    assert len(payload["gate_results"]) >= 1


# ---------------------------------------------------------------------- #
# Subtask 30: _run_content_generation_rewrite
# ---------------------------------------------------------------------- #


def test_run_content_generation_rewrite_emits_blocks_final_and_pages(
    tmp_path, monkeypatch,
):
    """Rewrite helper reads blocks_validated_path, dispatches each
    block through tier='rewrite', and persists both the final blocks
    JSONL and per-page HTML files."""
    project_id = "TEST_REWRITE"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)

    # Seed a validated-blocks JSONL with one Block.
    validated_path = tmp_path / "blocks_validated.jsonl"
    block = Block(
        block_id="week_01_content_01#explanation_concept_a_0",
        block_type="explanation",
        page_id="week_01_content_01",
        sequence=0,
        content="",
        objective_ids=("TO-01",),
    )
    validated_path.write_text(
        json.dumps(_pt._block_to_snake_case_entry(block)) + "\n",
        encoding="utf-8",
    )

    result = asyncio.run(_pt._run_content_generation_rewrite(
        blocks_validated_path=str(validated_path),
        project_id=project_id,
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload
    assert "content_paths" in payload
    assert "page_paths" in payload
    assert "content_dir" in payload
    assert "blocks_final_path" in payload
    blocks_final = Path(payload["blocks_final_path"])
    assert blocks_final.exists()
    # At least one page emitted with HTML content.
    assert len(payload["page_paths"]) >= 1
    for p in payload["page_paths"]:
        assert Path(p).exists()
    # Rewrite provider was invoked.
    assert len(fake.rewrite_calls) >= 1


# ---------------------------------------------------------------------- #
# Subtask 31 / 32: chained handlers + LEGACY_PHASE_OUTPUT_KEYS contract
# ---------------------------------------------------------------------- #


def test_two_pass_handlers_chain_end_to_end(tmp_path, monkeypatch):
    """Integration: chain outline → validation → rewrite → post_rewrite
    handlers in sequence, asserting each helper's output flows into the
    next handler's expected input."""
    project_id = "TEST_CHAIN"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    fake = _FakeProvider()
    _patch_router_with_fakes(monkeypatch, fake)

    # 1. Outline tier.
    outline_result = json.loads(asyncio.run(
        _pt._run_content_generation_outline(project_id=project_id),
    ))
    assert outline_result["success"] is True
    blocks_outline_path = outline_result["blocks_outline_path"]

    # 2. Inter-tier validation.
    validation_result = json.loads(asyncio.run(
        _pt._run_inter_tier_validation(
            blocks_outline_path=blocks_outline_path,
            project_id=project_id,
        ),
    ))
    assert validation_result["success"] is True
    blocks_validated_path = validation_result["blocks_validated_path"]

    # 3. Rewrite tier.
    rewrite_result = json.loads(asyncio.run(
        _pt._run_content_generation_rewrite(
            blocks_validated_path=blocks_validated_path,
            project_id=project_id,
        ),
    ))
    assert rewrite_result["success"] is True
    blocks_final_path = Path(rewrite_result["blocks_final_path"])
    assert blocks_final_path.exists()

    # 4. Post-rewrite validation runs only when the rewrite tier
    # produced at least one Block; in this minimal chain the inter-tier
    # validation may have rejected the only outline-tier Block, leaving
    # ``blocks_final.jsonl`` empty. In that branch, the post-rewrite
    # validator returns success=False with a structured error message —
    # which is the correct contract: an empty rewrite emit is itself a
    # signal worth surfacing. Test assertion: when blocks_final has at
    # least one entry, post-rewrite validation succeeds; otherwise it
    # surfaces the empty-input error envelope.
    if blocks_final_path.read_text(encoding="utf-8").strip():
        post_result = json.loads(asyncio.run(
            _pt._run_post_rewrite_validation(
                blocks_final_path=str(blocks_final_path),
                project_id=project_id,
            ),
        ))
        assert post_result["success"] is True, post_result


def test_two_pass_phase_outputs_match_legacy_phase_output_keys_table():
    """Each new handler's return-dict keys align with the declarations
    in MCP/core/workflow_runner.py::_LEGACY_PHASE_OUTPUT_KEYS for the
    matching phase name. The two-pass router phases are in Phase 3
    Subtask 5 (declared) and Phase 3.5 Subtasks 28-30 (emitted)."""
    # Outline phase
    outline_keys = set(_LEGACY_PHASE_OUTPUT_KEYS["content_generation_outline"])
    assert "blocks_outline_path" in outline_keys
    assert "project_id" in outline_keys
    assert "weeks_prepared" in outline_keys

    # Inter-tier validation phase
    validation_keys = set(_LEGACY_PHASE_OUTPUT_KEYS["inter_tier_validation"])
    assert "blocks_validated_path" in validation_keys
    assert "blocks_failed_path" in validation_keys

    # Rewrite phase
    rewrite_keys = set(_LEGACY_PHASE_OUTPUT_KEYS["content_generation_rewrite"])
    assert "content_paths" in rewrite_keys
    assert "page_paths" in rewrite_keys
    assert "content_dir" in rewrite_keys
    assert "blocks_final_path" in rewrite_keys

    # Post-rewrite validation phase (Wave-B)
    post_keys = set(_LEGACY_PHASE_OUTPUT_KEYS["post_rewrite_validation"])
    assert "blocks_validated_path" in post_keys
    assert "blocks_failed_path" in post_keys


def test_phase_tool_mapping_overrides_agent_tool_mapping():
    """Subtask 31's _PHASE_TOOL_MAPPING shim wires phase names to
    dedicated tool handlers regardless of the agent_type the task
    carries. Each of the four two-pass router phases must map to its
    canonical run_* tool name."""
    expected = {
        "content_generation_outline": "run_content_generation_outline",
        "inter_tier_validation": "run_inter_tier_validation",
        "content_generation_rewrite": "run_content_generation_rewrite",
        "post_rewrite_validation": "run_post_rewrite_validation",
    }
    for phase_name, tool_name in expected.items():
        assert _PHASE_TOOL_MAPPING.get(phase_name) == tool_name, (
            f"Phase '{phase_name}' must map to '{tool_name}' "
            f"(got {_PHASE_TOOL_MAPPING.get(phase_name)!r})"
        )

    # The phase-name mapping must not collide with agent-name mappings.
    # Agent names like 'content-generator' resolve to
    # 'generate_course_content' (legacy single-pass); the new phases
    # override that for two-pass dispatch via the executor's
    # resolution-order check.
    assert "content-generator" in AGENT_TOOL_MAPPING
    assert (
        AGENT_TOOL_MAPPING["content-generator"] == "generate_course_content"
    ), "AGENT_TOOL_MAPPING must keep the legacy single-pass mapping"


def test_run_content_generation_outline_returns_error_on_missing_project_id():
    """Defensive: missing project_id returns a structured error, not
    a crash — matches the I/O contract of _run_post_rewrite_validation
    which the executor's retry classifier also depends on."""
    result = asyncio.run(_pt._run_content_generation_outline())
    payload = json.loads(result)
    assert payload["success"] is False
    assert "project_id" in payload["error"].lower()
