"""Worker W3 regression — rewrite phase drives remediation loop.

Asserts the I/O contract for the Phase 3.5 remediation wiring landed
by Worker W3 in ``MCP/tools/pipeline_tools.py``:

* ``_run_content_generation_rewrite`` dispatches each Block through
  ``CourseforgeRouter.route_rewrite_with_remediation(...)`` (NOT the
  pre-W3 single-shot ``router.route(blk, tier="rewrite",
  source_chunks=[], objectives=[])`` short-circuit that broke the
  inter-tier seam).
* The validator chain is resolved from the workflow YAML's
  ``post_rewrite_validation`` phase via
  ``_resolve_post_rewrite_validators``; absent workflow_type yields
  ``[]`` (preserves pre-W3 semantics).
* ``source_chunks`` and ``objectives`` are rehydrated from the
  W2-persisted ``outline_chunks.json`` + ``outline_objectives.json``
  sidecars when the workflow_runner threads ``outline_chunks_path`` /
  ``outline_objectives_path`` through as kwargs.
* Missing sidecars emit a ``rewrite_grounding_missing`` warning
  decision capture (gate_id ``_run_content_generation_rewrite``) and
  fall through with empty source_chunks / objectives — matching pre-W3
  behavior so the rewrite phase does not crash on legacy direct calls.
* When ``route_rewrite_with_remediation`` raises (consensus failure
  surface), the resulting block carries ``escalation_marker=
  "validator_consensus_fail"``.

Mirrors the fixture pattern from
``test_pipeline_tools_outline_self_consistency.py``.
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
# Fixture builders (mirror test_pipeline_tools_outline_self_consistency.py)
# ---------------------------------------------------------------------- #


class _CapturingProvider:
    """Provider stub that records every (block_id, source_chunks,
    objectives) tuple passed into ``generate_rewrite``.

    Returns the block unchanged so the post-rewrite validator chain is
    free to either accept or fire ``action="regenerate"`` depending on
    the validator list resolved from the YAML.
    """

    def __init__(self) -> None:
        self.outline_calls: List[Dict[str, Any]] = []
        self.rewrite_calls: List[Dict[str, Any]] = []

    def generate_outline(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.outline_calls.append({"block_id": block.block_id})
        return dataclasses.replace(block, content="<p>outline stub</p>")

    def generate_rewrite(
        self, block: Block, *, source_chunks: Any, objectives: Any, **kw: Any,
    ) -> Block:
        self.rewrite_calls.append({
            "block_id": block.block_id,
            "source_chunks": list(source_chunks or []),
            "objectives": list(objectives or []),
        })
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


def _seed_outline_blocks(
    project_path: Path,
    blocks: List[Block],
) -> Path:
    """Persist a ``blocks_outline.jsonl`` under ``project_path/01_outline``
    and return its path.

    Mirrors the on-disk shape ``_run_content_generation_outline``
    writes (one snake_case JSON entry per line) so the rewrite phase
    can rehydrate via its ``_entry_to_block`` helper.
    """
    out_dir = project_path / "01_outline"
    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_path = out_dir / "blocks_outline.jsonl"
    with blocks_path.open("w", encoding="utf-8") as fh:
        for blk in blocks:
            fh.write(json.dumps(
                _pt._block_to_snake_case_entry(blk), ensure_ascii=False,
            ))
            fh.write("\n")
    return blocks_path


def _seed_outline_sidecars(
    project_path: Path,
    chunks_lookup: Dict[str, Any],
    objectives: List[Any],
) -> Dict[str, Path]:
    """Persist the W2 ``outline_chunks.json`` + ``outline_objectives.json``
    sidecars and return their paths."""
    out_dir = project_path / "01_outline"
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = out_dir / "outline_chunks.json"
    objectives_path = out_dir / "outline_objectives.json"
    chunks_path.write_text(
        json.dumps(chunks_lookup, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    objectives_path.write_text(
        json.dumps(objectives, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {"chunks": chunks_path, "objectives": objectives_path}


def _make_block(block_id: str, *, page_id: str = "week_01_content_01") -> Block:
    return Block(
        block_id=block_id,
        block_type="objective",
        page_id=page_id,
        sequence=0,
        content="<p>outline stub</p>",
        objective_ids=("TO-01",),
        bloom_level="remember",
    )


# ---------------------------------------------------------------------- #
# Worker W3: rewrite remediation wiring
# ---------------------------------------------------------------------- #


def test_rewrite_phase_threads_source_chunks_and_validators(
    tmp_path, monkeypatch,
):
    """``_run_content_generation_rewrite`` calls
    ``route_rewrite_with_remediation`` with the YAML-resolved validator
    list AND the rehydrated outline_chunks sidecar payload (NOT
    ``source_chunks=[]``)."""
    project_id = "TEST_W3_REMEDIATION"
    project_path = _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    blk = _make_block("week_01_content_01#objective_to-01_0")
    blocks_path = _seed_outline_blocks(project_path, [blk])

    # Seed sidecars: one chunk per block + non-empty objectives.
    chunks_lookup = {
        blk.block_id: [
            {"chunk_id": "c1", "text": "Source chunk text for grounding."}
        ],
    }
    objectives_payload = [
        {"id": "TO-01", "statement": "Describe core concept A in detail."},
    ]
    sidecar_paths = _seed_outline_sidecars(
        project_path, chunks_lookup, objectives_payload,
    )

    # Stub the router to capture call args so we can assert what was
    # actually threaded into route_rewrite_with_remediation.
    captured_calls: List[Dict[str, Any]] = []

    from Courseforge.router import router as _router_mod

    def fake_remediation(
        self, block, *, validators=None, source_chunks=None,
        objectives=None, **kw,
    ):
        captured_calls.append({
            "block_id": block.block_id,
            "validators": list(validators or []),
            "source_chunks": list(source_chunks or []),
            "objectives": list(objectives or []),
        })
        return dataclasses.replace(block, content="<p>rewrite stub</p>")

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter,
        "route_rewrite_with_remediation",
        fake_remediation,
    )

    result = asyncio.run(_pt._run_content_generation_rewrite(
        project_id=project_id,
        blocks_validated_path=str(blocks_path),
        workflow_type="textbook_to_course",
        outline_chunks_path=str(sidecar_paths["chunks"]),
        outline_objectives_path=str(sidecar_paths["objectives"]),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    # The router was invoked once per block via the remediation entry
    # point — NOT the legacy router.route(tier="rewrite") shortcut.
    assert len(captured_calls) == 1, captured_calls
    call = captured_calls[0]
    assert call["block_id"] == blk.block_id
    # Source chunks rehydrated from the sidecar (not the pre-W3 [] default).
    assert call["source_chunks"] == [
        {"chunk_id": "c1", "text": "Source chunk text for grounding."}
    ]
    # Objectives rehydrated from the sidecar (not the pre-W3 [] default).
    assert call["objectives"] == objectives_payload
    # Validator chain resolved from the YAML post_rewrite_validation
    # phase. The textbook_to_course workflow declares >= 1 validator
    # (rewrite_curie_anchoring + content_type + page_objectives +
    # source_refs + html_shape + source_grounding).
    assert len(call["validators"]) >= 1, (
        "Expected the textbook_to_course workflow's "
        "post_rewrite_validation phase to declare at least one validator"
    )


class _RecordingCapture:
    """Minimal DecisionCapture-compatible stub.

    Records every ``log_decision`` call so tests can assert the
    rewrite-phase grounding-missing helper emits the expected
    structured warning. Avoids the global on-disk DecisionCapture
    surface entirely (whose base-dir paths are resolved at module
    import and hard to redirect from a test fixture)."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    def log_decision(self, **kwargs: Any) -> None:
        self.events.append(dict(kwargs))


def test_rewrite_phase_emits_warning_when_sidecars_missing():
    """When ``outline_chunks_path`` / ``outline_objectives_path`` are
    absent from the phase_outputs dict, ``_load_outline_chunks`` /
    ``_load_outline_objectives`` emit a structured
    ``rewrite_grounding_missing`` warning capture (gate_id
    ``_run_content_generation_rewrite``) and fall through to empty
    chunks-dict / empty objectives-list — matching pre-W3
    fall-through semantics so the phase doesn't crash on legacy
    direct calls."""
    capture = _RecordingCapture()

    # Missing key — phase_outputs has no content_generation_outline.
    chunks = _pt._load_outline_chunks({}, capture)
    objectives = _pt._load_outline_objectives({}, capture)
    assert chunks == {}
    assert objectives == []

    # Both helpers emitted one rewrite_grounding_missing event each.
    grounding_events = [
        e for e in capture.events
        if e.get("decision_type") == "rewrite_grounding_missing"
    ]
    assert len(grounding_events) >= 2, capture.events

    # Each event carries the gate_id sentinel + missing_key context.
    missing_keys = {
        (e.get("ml_features") or {}).get("missing_key")
        for e in grounding_events
    }
    assert "outline_chunks_path" in missing_keys
    assert "outline_objectives_path" in missing_keys
    assert all(
        (e.get("ml_features") or {}).get("gate_id")
        == "_run_content_generation_rewrite"
        for e in grounding_events
    )


def test_rewrite_phase_falls_through_when_sidecar_kwargs_absent(
    tmp_path, monkeypatch,
):
    """End-to-end: ``_run_content_generation_rewrite`` does NOT crash
    when ``outline_chunks_path`` / ``outline_objectives_path`` are
    absent from kwargs (legacy direct call); the router is invoked
    with empty source_chunks + empty objectives so downstream gates
    light up rather than the phase itself failing closed."""
    project_id = "TEST_W3_MISSING_SIDECARS"
    project_path = _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    blk = _make_block("week_01_content_01#objective_to-01_0")
    blocks_path = _seed_outline_blocks(project_path, [blk])

    captured_calls: List[Dict[str, Any]] = []

    from Courseforge.router import router as _router_mod

    def fake_remediation(
        self, block, *, validators=None, source_chunks=None,
        objectives=None, **kw,
    ):
        captured_calls.append({
            "source_chunks": list(source_chunks or []),
            "objectives": list(objectives or []),
        })
        return dataclasses.replace(block, content="<p>rewrite stub</p>")

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter,
        "route_rewrite_with_remediation",
        fake_remediation,
    )

    result = asyncio.run(_pt._run_content_generation_rewrite(
        project_id=project_id,
        blocks_validated_path=str(blocks_path),
        workflow_type="textbook_to_course",
        # outline_chunks_path + outline_objectives_path intentionally absent
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    # Router was called with empty source_chunks + empty objectives —
    # the legacy fall-through path the helper guarantees.
    assert len(captured_calls) == 1, captured_calls
    assert captured_calls[0]["source_chunks"] == []
    assert captured_calls[0]["objectives"] == []


def test_rewrite_phase_consensus_fail_stamps_marker(tmp_path, monkeypatch):
    """When ``route_rewrite_with_remediation`` raises (rewrite-tier
    consensus failure), the resulting block carries
    ``escalation_marker="validator_consensus_fail"``."""
    project_id = "TEST_W3_CONSENSUS_FAIL"
    project_path = _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    blk = _make_block("week_01_content_01#objective_to-01_0")
    blocks_path = _seed_outline_blocks(project_path, [blk])
    sidecar_paths = _seed_outline_sidecars(
        project_path, {blk.block_id: []}, [],
    )

    from Courseforge.router import router as _router_mod

    def raising_remediation(self, block, *, validators=None,
                             source_chunks=None, objectives=None, **kw):
        raise RuntimeError("simulated rewrite-tier consensus failure")

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter,
        "route_rewrite_with_remediation",
        raising_remediation,
    )

    result = asyncio.run(_pt._run_content_generation_rewrite(
        project_id=project_id,
        blocks_validated_path=str(blocks_path),
        workflow_type="textbook_to_course",
        outline_chunks_path=str(sidecar_paths["chunks"]),
        outline_objectives_path=str(sidecar_paths["objectives"]),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    # Inspect blocks_final.jsonl for the escalation_marker.
    blocks_final = Path(payload["blocks_final_path"])
    assert blocks_final.exists()
    parsed = [
        json.loads(ln)
        for ln in blocks_final.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert any(
        entry.get("escalation_marker") == "validator_consensus_fail"
        for entry in parsed
    ), parsed


def test_rewrite_phase_preserves_outline_escalation_marker(
    tmp_path, monkeypatch,
):
    """Pre-existing escalation_marker on a block (set at the outline
    tier) short-circuits through the rewrite loop unchanged — the
    block rides through to the final list with the marker intact so
    packaging persists it for re-execution."""
    project_id = "TEST_W3_PRE_SKIP"
    project_path = _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    # Block already carries an outline-tier escalation marker.
    blk = dataclasses.replace(
        _make_block("week_01_content_01#objective_to-01_0"),
        escalation_marker="outline_budget_exhausted",
    )
    blocks_path = _seed_outline_blocks(project_path, [blk])
    sidecar_paths = _seed_outline_sidecars(
        project_path, {}, [],
    )

    invoked: List[str] = []
    from Courseforge.router import router as _router_mod

    def fake_remediation(self, block, **kw):
        invoked.append(block.block_id)
        return dataclasses.replace(block, content="<p>SHOULD NOT RUN</p>")

    monkeypatch.setattr(
        _router_mod.CourseforgeRouter,
        "route_rewrite_with_remediation",
        fake_remediation,
    )

    result = asyncio.run(_pt._run_content_generation_rewrite(
        project_id=project_id,
        blocks_validated_path=str(blocks_path),
        workflow_type="textbook_to_course",
        outline_chunks_path=str(sidecar_paths["chunks"]),
        outline_objectives_path=str(sidecar_paths["objectives"]),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    # Router was NOT invoked for the pre-marked block.
    assert invoked == [], (
        f"Expected pre-marked outline block to short-circuit through "
        f"the rewrite loop, but route_rewrite_with_remediation was "
        f"invoked for {invoked}"
    )
    # The marker is preserved on disk.
    blocks_final = Path(payload["blocks_final_path"])
    parsed = [
        json.loads(ln)
        for ln in blocks_final.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert any(
        entry.get("escalation_marker") == "outline_budget_exhausted"
        for entry in parsed
    ), parsed


def test_resolve_post_rewrite_validators_returns_empty_on_unknown_workflow():
    """``_resolve_post_rewrite_validators`` falls back to [] when the
    workflow_type is empty or unknown — preserving pre-W3 behavior on
    legacy direct callers."""
    assert _pt._resolve_post_rewrite_validators("") == []
    assert _pt._resolve_post_rewrite_validators("nonexistent_workflow") == []


def test_resolve_post_rewrite_validators_imports_yaml_declared_chain():
    """``_resolve_post_rewrite_validators`` walks the
    ``post_rewrite_validation`` phase's ``validation_gates`` and
    instantiates each declared validator — including the W1-registered
    ``RewriteHtmlShapeValidator`` + ``RewriteSourceGroundingValidator``
    plus the four shape-discriminating Block*Validator adapters."""
    validators = _pt._resolve_post_rewrite_validators("textbook_to_course")
    assert len(validators) >= 1, (
        "Expected the textbook_to_course workflow's "
        "post_rewrite_validation phase to declare at least one validator"
    )
    for v in validators:
        assert callable(getattr(v, "validate", None)), (
            f"Resolved entry {v!r} does not expose validate(); "
            f"_resolve_post_rewrite_validators returned a non-validator"
        )
