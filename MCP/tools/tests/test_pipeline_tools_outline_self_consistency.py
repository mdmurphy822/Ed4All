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
import re
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


def _patch_empty_escalation_policy(monkeypatch) -> None:
    """Strip the Wave-7 ``escalate_immediately`` flags from the loaded policy.

    Wave-7/7b (commits ``101d1e5`` + ``8dc44e1``) flipped the
    ``objective`` (and ``prereq_set``) block-types in
    ``Courseforge/config/block_routing.yaml`` to
    ``escalate_immediately: true``. The outline handler stamps every stub
    block ``block_type="objective"`` (``pipeline_tools.py`` ~4346-4361),
    so under the real policy the router short-circuits EVERY block at the
    ``escalate_immediately`` gate (``router.py`` ~774-824) — the outline
    provider is never invoked and no self-consistency regen loop runs.

    These self-consistency tests were written to prove the
    ``route_with_self_consistency`` wiring (provider dispatch + regen
    budget + validator chain). To re-exercise that wiring we override
    ``Courseforge.router.policy.load_block_routing_policy`` (the deferred
    import bound inside ``_run_content_generation_outline`` at call time)
    to return a policy whose ``escalate_immediately_by_block_type`` set is
    EMPTY, so the ``objective`` stubs fall through to the outline tier and
    the regen loop actually fires.

    The Wave-7 production default itself is pinned separately by
    :func:`test_outline_objective_blocks_escalate_immediately_under_real_policy`.
    """
    import dataclasses as _dc

    from Courseforge.router import policy as _policy_mod

    real_loader = _policy_mod.load_block_routing_policy

    def _empty_escalation_loader(*args: Any, **kwargs: Any):
        loaded = real_loader(*args, **kwargs)
        return _dc.replace(loaded, escalate_immediately_by_block_type={})

    monkeypatch.setattr(
        _policy_mod, "load_block_routing_policy", _empty_escalation_loader,
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

    # Wave-7/7b production pivot: ``objective`` blocks carry
    # ``escalate_immediately: true`` in block_routing.yaml, and the
    # outline handler stamps every stub ``block_type="objective"`` —
    # so under the real policy the router short-circuits every block at
    # the outline seam and the provider is never invoked. Strip the
    # escalate flags so the self-consistency wiring this test was
    # written to prove (provider dispatch + regen budget + validator
    # chain) actually runs. The Wave-7 default is pinned separately by
    # test_outline_objective_blocks_escalate_immediately_under_real_policy.
    _patch_empty_escalation_policy(monkeypatch)

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


def test_outline_objective_blocks_escalate_immediately_under_real_policy(
    tmp_path, monkeypatch,
):
    """Wave-7/7b production-default pin (commits ``101d1e5`` + ``8dc44e1``),
    re-pinned for the hollow-course keystone fix.

    With the REAL ``block_routing.yaml`` policy (no escalate-flag strip),
    each page emits the ``_PAGE_BLOCK_PLAN`` block types. The ``objective``
    block carries ``escalate_immediately: true``, so ``CourseforgeRouter``
    fires the outline-tier escalate short-circuit for it (no provider
    dispatch, ``escalation_marker="outline_budget_exhausted"``). The prose
    block types (``explanation`` / ``example``) are NOT escalate-immediate,
    so they DISPATCH through the outline tier — that dispatch is exactly
    what produces the instructional prose the hollow-course defect was
    missing.

    Pins:
    * ``objective`` blocks short-circuit (no provider call for them) and
      carry the budget-exhausted marker;
    * ``explanation`` / ``example`` blocks DO invoke the outline provider
      (the prose path is live under the real policy).

    This is the complement of
    :func:`test_outline_phase_drives_self_consistency`, which strips the
    escalate flags to re-exercise the self-consistency wiring on every type.
    """
    project_id = "TEST_WAVE7_ESCALATE_DEFAULT"
    _seed_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    # NOTE: deliberately NO _patch_empty_escalation_policy — we want the
    # real Wave-7 policy in force.

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    blocks_path = Path(payload["blocks_outline_path"])
    parsed = [
        json.loads(ln) for ln in
        blocks_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(parsed) >= 1, parsed

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for entry in parsed:
        by_type.setdefault(entry.get("block_type"), []).append(entry)

    # The page block plan emitted both the objective and the prose blocks.
    assert "objective" in by_type, parsed
    assert "explanation" in by_type or "example" in by_type, (
        "Keystone fix: each page MUST emit at least one prose block "
        f"(explanation/example); got types {sorted(by_type)}"
    )

    # Every OBJECTIVE block rides through carrying the budget-exhausted
    # marker the escalate-immediately short-circuit stamps.
    assert all(
        e.get("escalation_marker") == "outline_budget_exhausted"
        for e in by_type["objective"]
    ), (
        "Wave-7 default: every objective block must carry "
        "escalation_marker='outline_budget_exhausted' from the "
        "escalate_immediately short-circuit; got: "
        + json.dumps([e.get("escalation_marker") for e in by_type["objective"]])
    )

    # The prose blocks DISPATCHED through the outline tier — the provider
    # WAS invoked (proof the prose path is live; the lone-objective hollow
    # course never reached this dispatch).
    assert fake.outline_calls, (
        "Keystone fix: explanation/example blocks are not escalate-"
        "immediate, so the outline provider MUST be invoked for them; "
        f"got {len(fake.outline_calls)} outline calls"
    )


def _seed_grounded_project(tmp_path: Path, project_id: str) -> Path:
    """Project scaffold whose objectives carry grounded source_refs[].chunk_ids.

    Mirrors the production ``synthesized_objectives.json`` shape: a
    terminal objective + per-week ``chapter_objectives`` groups, each
    objective carrying structured ``source_refs[{ref, chunk_ids[]}]``
    referencing the course chunkset (built by ``_seed_chunkset`` below).
    """
    exports_root = tmp_path / "Courseforge" / "exports"
    project_path = exports_root / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    objectives = {
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Describe core concept A in detail.",
                "source_refs": [
                    {"ref": "ch1", "chunk_ids": ["course_chunk_00001"]}
                ],
            }
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {
                        "id": "CO-01",
                        "statement": "Explain the first chapter idea.",
                        "bloom_level": "understand",
                        "source_refs": [
                            {
                                "ref": "ch1",
                                "chunk_ids": [
                                    "course_chunk_00002",
                                    "course_chunk_00003",
                                ],
                            }
                        ],
                    }
                ],
            },
            {
                "chapter": "Week 2",
                "objectives": [
                    {
                        "id": "CO-02",
                        "statement": "Apply the second chapter method.",
                        "bloom_level": "apply",
                        "source_refs": [
                            {"ref": "ch2", "chunk_ids": ["course_chunk_00004"]}
                        ],
                    }
                ],
            },
        ],
    }
    (project_path / "01_learning_objectives" / "synthesized_objectives.json").write_text(
        json.dumps(objectives), encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": 2}),
        encoding="utf-8",
    )
    return project_path


def _seed_chunkset(libv2_root: Path, course_slug: str) -> None:
    """Write a tiny dart_chunks/chunks.jsonl whose ids match the objectives."""
    chunks_dir = libv2_root / "courses" / course_slug / "dart_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        {"id": "course_chunk_00001", "text": "Body of chunk one about concept A."},
        {"id": "course_chunk_00002", "text": "Body of chunk two, chapter one idea."},
        {"id": "course_chunk_00003", "text": "Body of chunk three, more chapter one."},
        {"id": "course_chunk_00004", "text": "Body of chunk four, chapter two method."},
    ]
    (chunks_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in lines) + "\n", encoding="utf-8",
    )


def test_outline_resolves_source_chunks_from_grounded_objectives(
    tmp_path, monkeypatch,
):
    """Keystone fix: per-block ``chunks_lookup`` / ``outline_chunks.json``
    carries ``{id, text}`` dicts whose ids equal the block's week's
    objective ``source_refs[].chunk_ids`` — NOT the legacy no-id
    topic-paragraph shape.
    """
    project_id = "TEST_GROUNDED_CHUNKS"
    _seed_grounded_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    # Resolve libv2 root to a tmp dir and seed the matching chunkset.
    libv2_root = tmp_path / "libv2"
    course_slug = project_id.lower().replace("_", "-").replace(" ", "-")
    _seed_chunkset(libv2_root, course_slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))

    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    _patch_empty_escalation_policy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    chunks_sidecar = Path(payload["outline_chunks_path"])
    chunks_data = json.loads(chunks_sidecar.read_text(encoding="utf-8"))
    assert isinstance(chunks_data, dict) and chunks_data

    # Collect every resolved chunk-id across all blocks.
    all_ids: set = set()
    saw_id_text_shape = False
    for block_id, entries in chunks_data.items():
        assert isinstance(entries, list)
        for entry in entries:
            assert isinstance(entry, dict)
            if "id" in entry and "text" in entry:
                saw_id_text_shape = True
                # JSON-serializable str->str contract.
                assert isinstance(entry["id"], str) and entry["id"]
                assert isinstance(entry["text"], str) and entry["text"]
                all_ids.add(entry["id"])

    assert saw_id_text_shape, (
        "Expected per-block chunks to carry the grounded {id, text} shape "
        f"resolved from objectives; got: {json.dumps(chunks_data)[:500]}"
    )
    # The resolved ids are EXACTLY the objective source_refs chunk_ids
    # (the TO + per-week CO chunk_ids), never fabricated.
    expected = {
        "course_chunk_00001",  # TO-01 (week 1 terminal slice)
        "course_chunk_00002",  # CO-01 week 1
        "course_chunk_00003",  # CO-01 week 1
        "course_chunk_00004",  # CO-02 week 2
    }
    assert all_ids <= expected, (
        f"Resolved ids must be a subset of the objective chunk_ids "
        f"{expected}; got extras: {all_ids - expected}"
    )
    assert all_ids, "Expected at least one objective-grounded chunk id"
    # Regression: the per-week CHAPTER-OBJECTIVE chunk_ids (CO-01 week 1 /
    # CO-02 week 2) must resolve, NOT just the terminal-objective slice.
    # The pre-fix index consumed the FLATTENED ``load_objectives_json``
    # output (which strips the per-group ``chapter`` label), so every CO
    # chunk_id was dropped and only ``course_chunk_00001`` (the terminal
    # slice) survived. Asserting the CO ids resolve is what proves the
    # grouped-shape week→chunk_ids index works end-to-end.
    co_chunk_ids = {
        "course_chunk_00002",  # CO-01 week 1
        "course_chunk_00003",  # CO-01 week 1
        "course_chunk_00004",  # CO-02 week 2
    }
    assert co_chunk_ids <= all_ids, (
        "Expected the per-week chapter-objective chunk_ids to resolve "
        f"(the keystone fix); missing: {co_chunk_ids - all_ids}"
    )

    # Contract verification: _block_source_chunk_ids reads id/chunk_id and
    # must return the resolved ids for a resolved block.
    from Courseforge.generators._outline_provider import (
        _block_source_chunk_ids,
    )
    resolved_block = next(
        entries for entries in chunks_data.values()
        if any("id" in e for e in entries)
    )
    ids_back = _block_source_chunk_ids(resolved_block)
    assert ids_back, "_block_source_chunk_ids returned empty for a resolved block"
    assert set(ids_back) <= expected


def test_outline_falls_back_to_topic_shape_when_chunkset_missing(
    tmp_path, monkeypatch,
):
    """Fail-SAFE: a missing chunkset never crashes the phase; blocks fall
    back to the legacy topic-paragraph (no-id) shape and the run succeeds.
    """
    project_id = "TEST_NO_CHUNKSET"
    _seed_grounded_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    # Point libv2 at an empty tmp dir — no chunkset present.
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(tmp_path / "empty_libv2"))

    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    _patch_empty_escalation_policy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    chunks_data = json.loads(
        Path(payload["outline_chunks_path"]).read_text(encoding="utf-8")
    )
    # No entry carries an ``id`` (chunkset absent → topic-paragraph shape).
    for entries in chunks_data.values():
        for entry in entries:
            assert "id" not in entry, (
                "Missing chunkset must fall back to the no-id topic shape; "
                f"got an id-bearing entry: {entry}"
            )
            assert "heading" in entry


# ---------------------------------------------------------------------- #
# _build_week_chunk_id_index — grouped raw shape + positional fallback
# ---------------------------------------------------------------------- #


def test_build_week_chunk_id_index_resolves_grouped_week_labels():
    """The index reads the RAW GROUPED ``{chapter, objectives}`` shape and
    unions every objective's ``source_refs[].chunk_ids`` per parsed week
    label ("Week N" → N).
    """
    groups = [
        {
            "chapter": "Week 1",
            "objectives": [
                {"id": "CO-01", "source_refs": [
                    {"ref": "ch1", "chunk_ids": ["c1", "c2"]}
                ]},
            ],
        },
        {
            "chapter": "Week 2",
            "objectives": [
                {"id": "CO-02", "source_refs": [
                    {"ref": "ch2", "chunk_ids": ["c3"]}
                ]},
                {"id": "CO-03", "source_refs": [
                    {"ref": "ch2", "chunk_ids": ["c3", "c4"]}  # dedupe c3
                ]},
            ],
        },
    ]
    idx = _pt._build_week_chunk_id_index(groups)
    assert idx == {1: ["c1", "c2"], 2: ["c3", "c4"]}, idx


def test_build_week_chunk_id_index_positional_fallback_when_label_missing():
    """When ``chapter`` does not parse to a week number, the i-th group
    (0-based) maps to week ``i+1`` — the positional fallback.
    """
    groups = [
        {  # unparseable label → positional week 1
            "chapter": "Introduction & Foundations",
            "objectives": [
                {"source_refs": [{"chunk_ids": ["a1"]}]},
            ],
        },
        {  # no chapter key at all → positional week 2
            "objectives": [
                {"source_refs": [{"chunk_ids": ["a2", "a3"]}]},
            ],
        },
    ]
    idx = _pt._build_week_chunk_id_index(groups)
    assert idx == {1: ["a1"], 2: ["a2", "a3"]}, idx


def test_build_week_chunk_id_index_folds_terminal_chunk_ids():
    """Per-week terminal-objective chunk_ids fold in after the chapter
    objectives (order-preserving, deduped)."""
    groups = [
        {"chapter": "Week 1", "objectives": [
            {"source_refs": [{"chunk_ids": ["c1"]}]},
        ]},
    ]
    idx = _pt._build_week_chunk_id_index(
        groups, week_terminal_chunk_ids={1: ["c1", "t1"], 2: ["t2"]},
    )
    assert idx == {1: ["c1", "t1"], 2: ["t2"]}, idx


def test_build_week_chunk_id_index_empty_groups_returns_empty():
    """No groups (absent / unreadable objectives) → empty index (caller
    degrades to topic-paragraph shape)."""
    assert _pt._build_week_chunk_id_index([]) == {}
    assert _pt._build_week_chunk_id_index(None) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# Fix 2A — _build_week_co_chunk_index: per-week PER-CO chunk universe
# ---------------------------------------------------------------------- #


def test_build_week_co_chunk_index_preserves_per_co_split():
    """The per-CO index keeps each objective's chunk_ids in its OWN inner
    list (group order), rather than collapsing to the week-wide union — the
    split that lets the caller narrow a page to a single objective."""
    groups = [
        {
            "chapter": "Week 1",
            "objectives": [
                {"id": "CO-01", "source_refs": [
                    {"ref": "ch1", "chunk_ids": ["c1", "c2"]},
                ]},
                {"id": "CO-02", "source_refs": [
                    {"ref": "ch1", "chunk_ids": ["c3", "c4"]},
                ]},
            ],
        },
    ]
    idx = _pt._build_week_co_chunk_index(groups)
    # One inner list per CO, in group order; NOT the union ["c1".."c4"].
    assert idx == {1: [["c1", "c2"], ["c3", "c4"]]}, idx


def test_build_week_co_chunk_index_dedupes_within_co_keeps_cross_co_overlap():
    """Chunk ids dedupe WITHIN a CO but recur ACROSS COs — the overlap the
    week-wide union collapses and the per-CO split deliberately keeps."""
    groups = [
        {
            "chapter": "Week 1",
            "objectives": [
                {"id": "CO-01", "source_refs": [
                    {"chunk_ids": ["c1", "c1", "c2"]},  # in-CO dedupe → c1,c2
                ]},
                {"id": "CO-02", "source_refs": [
                    {"chunk_ids": ["c2", "c3"]},  # c2 recurs across COs (kept)
                ]},
            ],
        },
    ]
    idx = _pt._build_week_co_chunk_index(groups)
    assert idx == {1: [["c1", "c2"], ["c2", "c3"]]}, idx


def test_build_week_co_chunk_index_positional_fallback_and_empty_co():
    """Unparseable label → positional week; a CO with no chunk_ids
    contributes an empty inner list (the caller's floor handles it)."""
    groups = [
        {  # unparseable label → positional week 1
            "chapter": "Foundations",
            "objectives": [
                {"id": "CO-01", "source_refs": [{"chunk_ids": ["a1"]}]},
                {"id": "CO-02", "source_refs": []},  # no chunks → []
            ],
        },
    ]
    idx = _pt._build_week_co_chunk_index(groups)
    assert idx == {1: [["a1"], []]}, idx


def test_build_week_co_chunk_index_empty_groups_returns_empty():
    """No groups → empty index (caller degrades to the week-wide union)."""
    assert _pt._build_week_co_chunk_index([]) == {}
    assert _pt._build_week_co_chunk_index(None) == {}  # type: ignore[arg-type]


def test_build_week_co_chunk_index_subset_of_week_union():
    """Anti-fabrication: the union of all per-CO slices equals the chapter-
    objective portion of the week-wide union — the per-CO split only ever
    SUBSETS real chunk sets, never invents an id."""
    groups = [
        {"chapter": "Week 1", "objectives": [
            {"source_refs": [{"chunk_ids": ["c1", "c2"]}]},
            {"source_refs": [{"chunk_ids": ["c2", "c3"]}]},
        ]},
        {"chapter": "Week 2", "objectives": [
            {"source_refs": [{"chunk_ids": ["c4"]}]},
        ]},
    ]
    co_idx = _pt._build_week_co_chunk_index(groups)
    union_idx = _pt._build_week_chunk_id_index(groups)  # no terminal fold
    for week, slices in co_idx.items():
        flat = {cid for sl in slices for cid in sl}
        assert flat <= set(union_idx.get(week, [])), (
            f"week {week}: per-CO ids {flat} escaped the week union "
            f"{set(union_idx.get(week, []))}"
        )


# ---------------------------------------------------------------------- #
# Donor / non-content heading filter — all-filtered week falls through to
# a neutral week_NN block (donor name never seeds a block_id slug).
# ---------------------------------------------------------------------- #


def test_all_noncontent_week_does_not_seed_donor_block_id(
    tmp_path, monkeypatch,
):
    """A week whose ONLY topic is a non-content donor heading must NOT
    re-admit that heading as a block slug. Instead the week falls through
    to the ``page_count = 1`` minimum and emits ONE block with a neutral
    ``week_NN`` heading (grounding still comes from the week's objectives).
    """
    project_id = "TEST_DONOR_FILTER"
    project_path = _seed_grounded_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    # Seed a chunkset so week-1 objectives resolve (block still gets built).
    libv2_root = tmp_path / "libv2"
    course_slug = project_id.lower().replace("_", "-").replace(" ", "-")
    _seed_chunkset(libv2_root, course_slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))

    # Stage a single DART HTML file whose only week-1 heading is a donor
    # acknowledgments line (the live-run leak). The staging dir is passed
    # explicitly so collect_staged_html finds it.
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    # DART-shaped: a <section> wrapping the donor <h2> + a paragraph long
    # enough (>= 30 words) to survive parse_dart_html_files' min-signal
    # filter, so the donor topic actually reaches the week grouping (and
    # the non-content heading filter).
    (staging_dir / "donor.html").write_text(
        "<html><body><section>"
        "<h2>Charles Koch Foundation The Stuart Family Foundation</h2>"
        "<p>We gratefully acknowledge the generous support of our donors "
        "and benefactors who made this open educational resource possible "
        "for students and instructors everywhere, year after year after "
        "year, in many different countries and institutions worldwide.</p>"
        "</section></body></html>",
        encoding="utf-8",
    )

    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    _patch_empty_escalation_policy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
        staging_dir=str(staging_dir),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    blocks_path = Path(payload["blocks_outline_path"])
    parsed = [
        json.loads(ln) for ln in
        blocks_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert parsed, "expected at least one block"
    # No block_id (or page slug) carries any donor-name token.
    blob = json.dumps(parsed).lower()
    for token in ("koch", "stuart", "foundation"):
        assert token not in blob, (
            f"donor token {token!r} leaked into an outline block: "
            + json.dumps([e.get("block_id") for e in parsed])
        )


# ---------------------------------------------------------------------- #
# Keystone hollow-course fix — block-plan emits prose block(s) per page
# ---------------------------------------------------------------------- #


def _outline_block_types(payload: Dict[str, Any]) -> List[str]:
    """Read the emitted outline blocks' block_types from the JSONL."""
    blocks_path = Path(payload["blocks_outline_path"])
    return [
        json.loads(ln).get("block_type")
        for ln in blocks_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]


def test_outline_block_plan_emits_prose_blocks_topic_rich_week(
    tmp_path, monkeypatch,
):
    """KEYSTONE regression (hollow-course defect): a topic-rich week's
    block plan emits non-objective prose block(s) per page — the
    distribution is NOT 100% ``objective``.

    Pre-fix, every page emitted exactly one ``block_type="objective"``
    stub, so the published course carried only the LO sentence and zero
    instructional prose. The fix emits ``explanation`` + ``example`` prose
    blocks alongside the objective on every page.
    """
    project_id = "TEST_PLAN_TOPIC_RICH"
    _seed_grounded_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    libv2_root = tmp_path / "libv2"
    course_slug = project_id.lower().replace("_", "-").replace(" ", "-")
    _seed_chunkset(libv2_root, course_slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))

    # Stage a topic-rich DART file so week 1 has a real content topic
    # (page heading derived from the topic, not the week_NN fallback).
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "topic.html").write_text(
        "<html><body><section>"
        "<h2>Solving Linear Equations</h2>"
        "<p>A linear equation in one variable can be solved by isolating "
        "the variable through inverse operations applied to both sides, "
        "step by step, until the variable stands alone on one side of the "
        "equality, yielding the unique solution value for the unknown.</p>"
        "</section></body></html>",
        encoding="utf-8",
    )

    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    _patch_empty_escalation_policy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
        staging_dir=str(staging_dir),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    types = _outline_block_types(payload)
    assert types, "expected at least one outline block"

    # Distribution is NOT all-objective: prose blocks are present.
    distinct = set(types)
    assert distinct != {"objective"}, (
        "Hollow-course regression: block plan emitted ONLY objective "
        f"blocks; got distribution {sorted(distinct)}"
    )
    # Both the objective and at least one prose block_type are present.
    assert "objective" in distinct, sorted(distinct)
    assert distinct & {"explanation", "example"}, (
        "Expected at least one prose block_type (explanation/example) per "
        f"page; got {sorted(distinct)}"
    )

    # No page is hollow — every page carries at least one instructional
    # (non-objective) block. The week now emits five canonical page types
    # (overview / content / application / self_check / summary; see
    # ``_WEEK_PAGE_TYPES``), so a page need not carry ``explanation``/
    # ``example`` specifically — a self_check page's ``self_check_question``
    # or a summary page's ``summary_takeaway``/``recap`` is substantive
    # instructional content, not the hollow LO-sentence-only defect. The
    # course-level prose assertion above still guarantees explanation/example
    # are present somewhere.
    by_page: Dict[str, set] = {}
    blocks_path = Path(payload["blocks_outline_path"])
    for ln in blocks_path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        entry = json.loads(ln)
        by_page.setdefault(entry.get("page_id"), set()).add(
            entry.get("block_type")
        )
    for page_id, page_types in by_page.items():
        assert page_types - {"objective"}, (
            f"page {page_id!r} is hollow — it carries only an objective "
            f"block with no instructional content: {sorted(page_types)}"
        )


def test_outline_block_plan_emits_prose_blocks_topic_empty_week(
    tmp_path, monkeypatch,
):
    """KEYSTONE regression: the topic-empty (week_NN) fallback page also
    emits prose block(s), not just an objective.

    When a week has no usable content topic (no staged DART, or every
    topic filtered as non-content noise), the builder falls through to the
    ``page_count = max(0, 1) = 1`` minimum with a neutral ``week_NN``
    heading. That fallback page MUST still carry the prose block plan —
    otherwise the fallback weeks render hollow.
    """
    project_id = "TEST_PLAN_TOPIC_EMPTY"
    _seed_grounded_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)

    libv2_root = tmp_path / "libv2"
    course_slug = project_id.lower().replace("_", "-").replace(" ", "-")
    _seed_chunkset(libv2_root, course_slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))

    # NO staging_dir → every week falls through to the week_NN fallback
    # page (topic-empty path).
    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    _patch_empty_escalation_policy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    types = _outline_block_types(payload)
    assert types, "expected at least one outline block"

    distinct = set(types)
    assert distinct != {"objective"}, (
        "Hollow-course regression (fallback week): topic-empty page "
        f"emitted ONLY objective blocks; got {sorted(distinct)}"
    )
    assert "objective" in distinct, sorted(distinct)
    assert distinct & {"explanation", "example"}, (
        "Topic-empty fallback page must still carry a prose block_type; "
        f"got {sorted(distinct)}"
    )

    # Confirm the fallback page id is the neutral week_NN form (no donor /
    # topic heading leaked) and still carries prose.
    blocks_path = Path(payload["blocks_outline_path"])
    page_ids = {
        json.loads(ln).get("page_id")
        for ln in blocks_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }
    assert any(
        pid and pid.startswith("week_") for pid in page_ids
    ), sorted(p for p in page_ids if p)


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


# ---------------------------------------------------------------------- #
# Fix 2A — end-to-end page-level objective scoping (Step 2b)
# ---------------------------------------------------------------------- #


def _seed_two_co_one_week_project(tmp_path: Path, project_id: str) -> Path:
    """Scaffold: ONE week carrying TWO grounded COs, each with a focused
    chunk set (CO-01 → chunks 1-3, CO-02 → chunks 4-6, no overlap). Paired
    with two staged DART topics so the week emits TWO pages — page 0 maps
    positionally to CO-01, page 1 to CO-02.
    """
    exports_root = tmp_path / "Courseforge" / "exports"
    project_path = exports_root / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    objectives = {
        "terminal_objectives": [
            {"id": "TO-01", "statement": "Master the whole-numbers unit.",
             "source_refs": []},
        ],
        "chapter_objectives": [
            {
                "chapter": "Week 1",
                "objectives": [
                    {
                        "id": "CO-01",
                        "statement": "Add and subtract fractions.",
                        "bloom_level": "apply",
                        "source_refs": [{"ref": "ch1", "chunk_ids": [
                            "course_chunk_00001",
                            "course_chunk_00002",
                            "course_chunk_00003",
                        ]}],
                    },
                    {
                        "id": "CO-02",
                        "statement": "Perform prime factorization and find the LCM.",
                        "bloom_level": "apply",
                        "source_refs": [{"ref": "ch1", "chunk_ids": [
                            "course_chunk_00004",
                            "course_chunk_00005",
                            "course_chunk_00006",
                        ]}],
                    },
                ],
            },
        ],
    }
    (project_path / "01_learning_objectives" / "synthesized_objectives.json").write_text(
        json.dumps(objectives), encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": 1}),
        encoding="utf-8",
    )
    return project_path


def _seed_six_chunk_set(libv2_root: Path, course_slug: str) -> None:
    chunks_dir = libv2_root / "courses" / course_slug / "dart_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        {"id": f"course_chunk_{n:05d}", "text": f"Body of chunk {n}."}
        for n in range(1, 7)
    ]
    (chunks_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(c) for c in lines) + "\n", encoding="utf-8",
    )


def _stage_two_topics(tmp_path: Path) -> Path:
    """Two content-rich DART topics → week 1 emits two pages.

    The headings (``Photosynthesis in Plants`` / ``The Krebs Cycle``) are
    chosen to clear ``_content_gen_helpers._is_low_signal_heading`` so they
    survive as distinct topics → distinct pages (many "obvious" math titles
    like "Adding Fractions" are blocklisted as low-signal chrome). The
    topic TEXT is irrelevant to Fix 2A: page→CO scoping is POSITIONAL by
    page index, not by topic↔objective similarity.
    """
    staging_dir = tmp_path / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    (staging_dir / "topics.html").write_text(
        "<html><body>"
        "<section><h2>Photosynthesis in Plants</h2>"
        "<p>Photosynthesis is the biochemical process by which green plants "
        "convert light energy into chemical energy stored in glucose, "
        "drawing on carbon dioxide and water while releasing oxygen as a "
        "byproduct of the light-dependent reactions in the chloroplast.</p>"
        "</section>"
        "<section><h2>The Krebs Cycle</h2>"
        "<p>The Krebs cycle is a series of enzyme-catalyzed reactions that "
        "oxidize acetyl coenzyme A to carbon dioxide, generating the "
        "electron carriers that subsequently drive oxidative phosphorylation "
        "and the bulk of cellular ATP production in the mitochondrion.</p>"
        "</section>"
        "</body></html>",
        encoding="utf-8",
    )
    return staging_dir


def _chunks_sidecar_by_page(payload: Dict[str, Any]) -> Dict[str, set]:
    """Map page_id → set of resolved chunk ids from the outline_chunks sidecar.

    The sidecar is keyed by block_id; block_ids start with the page_id
    (``week_NN_content_MM_...``), so we group by the ``week_NN_content_MM``
    prefix.
    """
    chunks_data = json.loads(
        Path(payload["outline_chunks_path"]).read_text(encoding="utf-8")
    )
    by_page: Dict[str, set] = {}
    for block_id, entries in chunks_data.items():
        m = re.match(r"(week_\d+_content_\d+)", block_id)
        page_id = m.group(1) if m else block_id
        bucket = by_page.setdefault(page_id, set())
        for e in entries:
            if isinstance(e, dict) and e.get("id"):
                bucket.add(e["id"])
    return by_page


def test_outline_page_offered_only_its_objectives_chunks(tmp_path, monkeypatch):
    """Fix 2A: a page whose objective has a focused chunk set is offered
    ONLY that objective's chunks — NOT the whole-week union (the semantic
    grab-bag). Page 0 (CO-01) sees chunks 1-3; page 1 (CO-02) sees 4-6.
    """
    project_id = "TEST_2A_OBJ_SCOPED"
    _seed_two_co_one_week_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    libv2_root = tmp_path / "libv2"
    course_slug = project_id.lower().replace("_", "-").replace(" ", "-")
    _seed_six_chunk_set(libv2_root, course_slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    staging_dir = _stage_two_topics(tmp_path)

    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    _patch_empty_escalation_policy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
        staging_dir=str(staging_dir),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    by_page = _chunks_sidecar_by_page(payload)
    # Fix 2A scopes the per-topic CONTENT pages to their positional CO's
    # chunks. The week also emits overview/application/self_check/summary
    # pages (see ``_WEEK_PAGE_TYPES``) that legitimately span the whole
    # week's TO and therefore see the full week union — they are NOT
    # positional-CO content pages, so they are out of scope for this
    # grab-bag guard. Restrict the assertion to the two content pages.
    content_pages = {
        pid: chunks
        for pid, chunks in by_page.items()
        if re.fullmatch(r"week_\d+_content_\d+", pid or "")
    }
    assert set(content_pages) == {"week_01_content_01", "week_01_content_02"}, (
        f"expected exactly the two positional content pages "
        f"(week_01_content_01 / week_01_content_02); got "
        f"{sorted(content_pages)}"
    )
    page0 = content_pages["week_01_content_01"]
    page1 = content_pages["week_01_content_02"]
    co1 = {"course_chunk_00001", "course_chunk_00002", "course_chunk_00003"}
    co2 = {"course_chunk_00004", "course_chunk_00005", "course_chunk_00006"}
    assert page0 == co1, (
        f"page 0 must be offered ONLY CO-01's chunks {co1}; got {page0} "
        f"(grab-bag would also include {co2})"
    )
    assert page1 == co2, (
        f"page 1 must be offered ONLY CO-02's chunks {co2}; got {page1}"
    )
    # The grab-bag is gone: neither page sees the OTHER objective's chunks.
    assert not (page0 & co2), f"page 0 leaked CO-02 chunks: {page0 & co2}"
    assert not (page1 & co1), f"page 1 leaked CO-01 chunks: {page1 & co1}"


def test_outline_page_falls_back_to_week_union_when_co_below_floor(
    tmp_path, monkeypatch,
):
    """Fix 2A graceful fallback: when a page's positional CO is grounded by
    fewer than the floor (2) chunks, the page reverts to the WEEK-WIDE union
    so the block is never starved of grounding.

    CO-01 has a single chunk (below the floor) → page 0 must fall back to the
    full week union {chunk1..chunk4}, not be starved to just {chunk1}.
    """
    project_id = "TEST_2A_FALLBACK"
    exports_root = tmp_path / "Courseforge" / "exports"
    project_path = exports_root / project_id
    (project_path / "01_learning_objectives").mkdir(parents=True, exist_ok=True)
    objectives = {
        "terminal_objectives": [{"id": "TO-01", "statement": "Unit.",
                                 "source_refs": []}],
        "chapter_objectives": [
            {"chapter": "Week 1", "objectives": [
                {"id": "CO-01", "statement": "Thinly grounded objective.",
                 "bloom_level": "understand",
                 "source_refs": [{"chunk_ids": ["course_chunk_00001"]}]},
                {"id": "CO-02", "statement": "Well grounded objective.",
                 "bloom_level": "apply",
                 "source_refs": [{"chunk_ids": [
                     "course_chunk_00002", "course_chunk_00003",
                     "course_chunk_00004",
                 ]}]},
            ]},
        ],
    }
    (project_path / "01_learning_objectives" / "synthesized_objectives.json").write_text(
        json.dumps(objectives), encoding="utf-8",
    )
    (project_path / "project_config.json").write_text(
        json.dumps({"course_name": project_id, "duration_weeks": 1}),
        encoding="utf-8",
    )
    _patch_project_root(monkeypatch, tmp_path)
    libv2_root = tmp_path / "libv2"
    course_slug = project_id.lower().replace("_", "-").replace(" ", "-")
    chunks_dir = libv2_root / "courses" / course_slug / "dart_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / "chunks.jsonl").write_text(
        "\n".join(json.dumps(
            {"id": f"course_chunk_{n:05d}", "text": f"Body {n}."}
        ) for n in range(1, 5)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    staging_dir = _stage_two_topics(tmp_path)

    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    _patch_empty_escalation_policy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
        staging_dir=str(staging_dir),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    by_page = _chunks_sidecar_by_page(payload)
    week_union = {f"course_chunk_{n:05d}" for n in range(1, 5)}
    page0 = by_page["week_01_content_01"]
    # CO-01's lone chunk is below the floor → page 0 falls back to the week
    # union (NOT starved to {chunk1}).
    assert page0 == week_union, (
        f"thin-CO page 0 must fall back to the week-wide union {week_union}; "
        f"got {page0} (starvation would yield just course_chunk_00001)"
    )
    # CO-02 clears the floor → page 1 stays objective-scoped to CO-02.
    page1 = by_page["week_01_content_02"]
    co2 = {"course_chunk_00002", "course_chunk_00003", "course_chunk_00004"}
    assert page1 == co2, (
        f"well-grounded page 1 should remain objective-scoped to CO-02 {co2}; "
        f"got {page1}"
    )


def test_outline_offered_chunks_subset_of_real_chunkset(tmp_path, monkeypatch):
    """Anti-fabrication: every offered chunk id resolves to a REAL chunk in
    the chunkset — narrowing never invents an id, in either the scoped or
    the fallback path."""
    project_id = "TEST_2A_NO_FABRICATION"
    _seed_two_co_one_week_project(tmp_path, project_id)
    _patch_project_root(monkeypatch, tmp_path)
    libv2_root = tmp_path / "libv2"
    course_slug = project_id.lower().replace("_", "-").replace(" ", "-")
    _seed_six_chunk_set(libv2_root, course_slug)
    monkeypatch.setenv("ED4ALL_LIBV2_ROOT", str(libv2_root))
    staging_dir = _stage_two_topics(tmp_path)

    fake = _CurieMissingProvider()
    _patch_router_with_provider(monkeypatch, fake)
    _patch_empty_escalation_policy(monkeypatch)
    monkeypatch.setenv("COURSEFORGE_OUTLINE_REGEN_BUDGET", "1")
    monkeypatch.setenv("COURSEFORGE_OUTLINE_N_CANDIDATES", "1")

    result = asyncio.run(_pt._run_content_generation_outline(
        project_id=project_id,
        workflow_type="textbook_to_course",
        staging_dir=str(staging_dir),
    ))
    payload = json.loads(result)
    assert payload["success"] is True, payload

    real_ids = {f"course_chunk_{n:05d}" for n in range(1, 7)}
    by_page = _chunks_sidecar_by_page(payload)
    offered: set = set()
    for ids in by_page.values():
        offered |= ids
    assert offered, "expected at least one offered chunk id"
    assert offered <= real_ids, (
        f"offered ids must be a subset of the real chunkset {real_ids}; "
        f"fabricated extras: {offered - real_ids}"
    )
