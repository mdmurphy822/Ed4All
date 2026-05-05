"""Worker W2 regression — outline phase drives self-consistency.

Asserts the I/O contract for the Phase 3 self-consistency wiring
landed by Worker W2 in ``MCP/tools/pipeline_tools.py``:

* ``_run_content_generation_outline`` dispatches each Block through
  ``CourseforgeRouter.route_with_self_consistency(...)`` (NOT the pre-
  W2 single-shot ``router.route(blk, tier="outline", ...)``).
* The validator chain is resolved from the workflow YAML's
  ``inter_tier_validation`` phase via
  ``_resolve_inter_tier_validators``; failed candidates re-roll under
  the regen budget so the resulting Block carries
  ``validation_attempts > 0``.
* Two sidecars are persisted next to ``blocks_outline.jsonl``:
  ``outline_chunks.json`` + ``outline_objectives.json``.
* The phase output envelope surfaces both sidecar paths under
  ``outline_chunks_path`` and ``outline_objectives_path``.

The router runs end-to-end; we stub only the outline provider (so no
real LLM dispatch fires) and pre-resolve a validator chain that
always emits ``action="regenerate"`` so the self-consistency loop
exhausts its budget.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Courseforge.scripts.blocks import Block  # noqa: E402
from MCP.tools import pipeline_tools as _pt  # noqa: E402


# ---------------------------------------------------------------------- #
# Fixture builders (mirror test_pipeline_tools_phase3_handlers.py)
# ---------------------------------------------------------------------- #


class _CurieMissingProvider:
    """Outline provider stub that never produces CURIEs.

    Triggers ``BlockCurieAnchoringValidator`` to emit
    ``action="regenerate"`` on every candidate so the self-consistency
    loop walks its full regen budget.
    """

    def __init__(self) -> None:
        self.outline_calls: List[Dict[str, Any]] = []
        self.rewrite_calls: List[Dict[str, Any]] = []

    def generate_outline(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.outline_calls.append({"block_id": block.block_id})
        # String content with no CURIE-shaped tokens — guarantees the
        # CurieAnchoringValidator's str-path returns [] and emits
        # action="regenerate".
        return dataclasses.replace(
            block, content="plain prose with no anchored identifiers",
        )

    def generate_rewrite(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        # Rewrite tier shouldn't be reached in this test, but provide
        # a stub so the router's constructor doesn't trip.
        self.rewrite_calls.append({"block_id": block.block_id})
        return dataclasses.replace(block, content="<p>rewrite stub</p>")


def _seed_project(tmp_path: Path, project_id: str) -> Path:
    """Create a minimal Courseforge/exports/<project> scaffold under tmp_path."""
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
    monkeypatch.setattr(_pt, "PROJECT_ROOT", tmp_path)


def _patch_router_with_provider(monkeypatch, fake: Any) -> None:
    """Force CourseforgeRouter() to wire fake provider on both tiers."""
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
# Worker W2: route_with_self_consistency wiring
# ---------------------------------------------------------------------- #


def test_outline_phase_drives_self_consistency(tmp_path, monkeypatch):
    """_run_content_generation_outline routes each block through
    ``route_with_self_consistency`` with the resolved validator chain;
    candidates that fail CURIE anchoring re-roll under the regen
    budget; sidecars are written next to ``blocks_outline.jsonl``.
    """
    project_id = "TEST_W2_SELF_CONSISTENCY"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)

    # Cap the regen budget at a small value so the test is fast — the
    # provider always emits CURIE-less content, so every candidate will
    # fire action="regenerate" and the loop will exhaust the budget.
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "2")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "2")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    # Sidecar paths are surfaced through the phase-output envelope.
    assert "outline_chunks_path" in payload
    assert "outline_objectives_path" in payload
    chunks_sidecar = Path(payload["outline_chunks_path"])
    objectives_sidecar = Path(payload["outline_objectives_path"])
    assert chunks_sidecar.exists(), chunks_sidecar
    assert objectives_sidecar.exists(), objectives_sidecar

    # Sidecars are diff-friendly JSON (indent=2, sort_keys=True).
    chunks_data = json.loads(chunks_sidecar.read_text(encoding="utf-8"))
    objectives_data = json.loads(objectives_sidecar.read_text(encoding="utf-8"))
    assert isinstance(chunks_data, dict)
    assert isinstance(objectives_data, list)
    # Objectives sidecar carries the canonical TO-01 we seeded.
    assert any(
        isinstance(o, dict) and o.get("id") == "TO-01"
        for o in objectives_data
    )

    # Outline JSONL exists and the resulting blocks have non-zero
    # validation_attempts (proof the self-consistency loop ran the
    # validator chain inside the regen budget).
    blocks_path = Path(payload["blocks_outline_path"])
    assert blocks_path.exists()
    parsed = [
        json.loads(ln) for ln in
        blocks_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(parsed) >= 1
    # At least one block carries validation_attempts > 0 — the loop
    # walked its budget against a validator that fires regenerate.
    assert any(
        int(entry.get("validation_attempts", 0)) > 0 for entry in parsed
    ), (
        "Expected at least one block to carry validation_attempts > 0 "
        "after the self-consistency loop walked its regen budget; "
        "got: " + json.dumps(
            [entry.get("validation_attempts", 0) for entry in parsed]
        )
    )

    # The outline provider was invoked more than once per block (one
    # call per candidate; n_candidates=2 above), confirming the
    # self-consistency loop dispatched multiple attempts. With one
    # block in the fixture we expect ~2 calls.
    assert len(fake.outline_calls) >= 2, (
        f"Expected >= 2 outline calls (n_candidates=2 over 1 block); "
        f"got {len(fake.outline_calls)}"
    )


def test_resolve_inter_tier_validators_returns_empty_on_unknown_workflow(
    tmp_path, monkeypatch,
):
    """``_resolve_inter_tier_validators`` falls back to [] when the
    workflow_type is empty or unknown — preserving pre-W2 behavior on
    legacy direct callers."""
    # Empty workflow_type -> []
    assert _pt._resolve_inter_tier_validators("") == []
    # Unknown workflow_type -> [] (with a warning, but no exception)
    assert _pt._resolve_inter_tier_validators("nonexistent_workflow") == []


def test_resolve_inter_tier_validators_imports_yaml_declared_chain():
    """``_resolve_inter_tier_validators`` walks the
    ``inter_tier_validation`` phase's ``validation_gates`` and
    instantiates each declared validator. The W4 gate fixes pointed
    three gates at the Block-shape ``Courseforge.router.inter_tier_gates.Block*Validator``
    classes; we expect at least one of those to land in the resolved
    list."""
    validators = _pt._resolve_inter_tier_validators("textbook_to_course")
    # Should resolve a non-empty list (the gates the YAML declares).
    assert len(validators) >= 1, (
        "Expected the textbook_to_course workflow's "
        "inter_tier_validation phase to declare at least one validator"
    )
    # Validator instances expose a ``validate`` callable — sanity-check
    # the duck-type contract the router consumes.
    for v in validators:
        assert callable(getattr(v, "validate", None)), (
            f"Resolved entry {v!r} does not expose validate(); "
            f"_resolve_inter_tier_validators returned a non-validator"
        )
