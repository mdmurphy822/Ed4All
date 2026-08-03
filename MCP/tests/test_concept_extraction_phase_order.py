"""Phase-ordering fix (Option A1) — concept_extraction runs AFTER course_planning.

Locks the mission fix: on a fresh ``textbook_to_course`` run (no
``--reuse-objectives``), the ``concept_extraction`` phase must run AFTER
``course_planning`` so the synthesized objectives + LO-backfilled SemantiK
chunks are available when the semantic concept graph is built. Pre-fix
``concept_extraction`` ran BEFORE ``course_planning``, so fresh runs had
no synthesized objectives and no chunk ``learning_outcome_refs`` at
graph-build time, producing the degraded ``related-to``-dominated graph.

Covers:
 1. YAML topology — declared order course_planning < concept_extraction
    < content_generation; config loads without raising; libv2_archival
    still routes concept_graph_sha256 from concept_extraction; the
    WorkflowRunner topological sort preserves the ordering.
 2. Fresh-run simulation — disk state exactly as course_planning leaves
    it (synthesized_objectives.json + LO-ref'd chunks), no reuse kwarg →
    real objectives_source + LO-driven edges.
 3. Reuse-run regression — reuse objectives_path kwarg wins over the
    course_planning synthesized_objectives_path.
 4. Linker-in-phase — the relocated concept-objective linker enriches
    key_concepts and writes them back to the project-export
    synthesized_objectives.json; the targets-concept edge target is a
    graph-resident node id.
 5. Old-order checkpoint resume — a workflow-state with concept_extraction
    completed and course_planning absent still runs course_planning
    (deps met) without a _dependencies_met failure.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.config import WorkflowPhase  # noqa: E402
from MCP.core.workflow_runner import (  # noqa: E402
    WorkflowRunner,
    _get_phase_param_routing,
    _load_workflows_config,
)
from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_SYNTHESIZED_OBJECTIVES: Dict[str, Any] = {
    "course_name": "FXPHASE_101",
    "mint_method": "fixture",
    "duration_weeks": 8,
    "learning_outcomes": [
        {
            "id": "TO-01",
            "statement": "Explain vector spaces and their axioms.",
            "hierarchy_level": "terminal",
            "key_concepts": ["vector space"],
        },
        {
            "id": "TO-02",
            "statement": "Apply linear maps to coordinate transforms.",
            "hierarchy_level": "terminal",
            "key_concepts": ["linear map"],
        },
        {
            "id": "CO-01",
            "statement": "Identify eigenvalues of a square matrix.",
            "hierarchy_level": "chapter",
            "key_concepts": ["eigenvalue"],
        },
        {
            "id": "CO-02",
            "statement": "Evaluate determinant-based rank arguments.",
            "hierarchy_level": "chapter",
            "key_concepts": ["determinant"],
        },
    ],
    "terminal_objectives": [
        {"id": "TO-01", "statement": "Explain vector spaces and their axioms.",
         "key_concepts": ["vector space"]},
        {"id": "TO-02", "statement": "Apply linear maps to coordinate transforms.",
         "key_concepts": ["linear map"]},
    ],
    "chapter_objectives": [
        {"chapter": "Week 1", "objectives": [
            {"id": "CO-01", "statement": "Identify eigenvalues of a square matrix.",
             "key_concepts": ["eigenvalue"]}]},
        {"chapter": "Week 2", "objectives": [
            {"id": "CO-02", "statement": "Evaluate determinant-based rank arguments.",
             "key_concepts": ["determinant"]}]},
    ],
}


def _lo_tagged_chunkset() -> List[Dict[str, Any]]:
    """Chunks carrying learning_outcome_refs + concept_tags exactly as the
    chunking phase + course_planning LO-backfill leave them on disk."""
    base = {"module_id": "week_01", "item_path": "week_01/page_001.html",
            "course_id": "FXPHASE_101"}
    return [
        {
            "id": "pograph_chunk_00001",
            "text": "A vector space is closed under addition and scalar "
                    "multiplication, defined by its axiom set.",
            "chunk_type": "explanation",
            "concept_tags": ["vector-space", "axiom-set"],
            "learning_outcome_refs": ["TO-01"],
            "bloom_level": "understand",
            "source": dict(base),
        },
        {
            "id": "pograph_chunk_00002",
            "text": "An eigenvalue scales an eigenvector under a linear map "
                    "acting on a vector space.",
            "chunk_type": "explanation",
            "concept_tags": ["eigenvalue", "vector-space"],
            "learning_outcome_refs": ["CO-01"],
            "bloom_level": "apply",
            "source": {**base, "module_id": "week_02",
                       "item_path": "week_02/page_002.html"},
        },
        {
            "id": "pograph_chunk_00003",
            "text": "The determinant decides invertibility; rank arguments "
                    "build on eigenvalue structure.",
            "chunk_type": "explanation",
            "concept_tags": ["determinant", "eigenvalue"],
            "learning_outcome_refs": ["CO-02"],
            "bloom_level": "evaluate",
            "source": {**base, "module_id": "week_03",
                       "item_path": "week_03/page_003.html"},
        },
    ]


def _write_chunkset(path: Path, chunks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")


def _edges_of_type(graph: Dict[str, Any], edge_type: str) -> List[Dict[str, Any]]:
    return [
        e for e in graph.get("edges") or []
        if (e.get("type") or e.get("relation")) == edge_type
    ]


def _run_concept_extraction(
    tmp_path: Path,
    *,
    objectives_path: str = "",
    synthesized_objectives_path: str = "",
) -> Dict[str, Any]:
    chunks_path = tmp_path / "semantik_chunks" / "chunks.jsonl"
    _write_chunkset(chunks_path, _lo_tagged_chunkset())
    custom_libv2 = tmp_path / "libv2"

    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name="FXPHASE_101",
            staging_dir="",
            dart_chunks_path=str(chunks_path),
            libv2_root=str(custom_libv2),
            run_id="WF-TEST-PHASE-ORDER",
            objectives_path=objectives_path,
            synthesized_objectives_path=synthesized_objectives_path,
        )
    )
    payload = json.loads(result)
    graph_path = Path(payload["concept_graph_path"])
    payload["_graph"] = json.loads(graph_path.read_text(encoding="utf-8"))
    return payload


# ---------------------------------------------------------------------------
# 1. YAML topology
# ---------------------------------------------------------------------------


def _textbook_phases() -> List[Dict[str, Any]]:
    cfg = _load_workflows_config(force_reload=True)
    return cfg["workflows"]["textbook_to_course"]["phases"]


def test_declared_order_course_planning_before_concept_extraction():
    names = [p["name"] for p in _textbook_phases()]
    assert names.index("course_planning") < names.index("concept_extraction"), (
        "course_planning must be declared BEFORE concept_extraction so the "
        "synthesized objectives exist at graph-build time."
    )
    assert names.index("concept_extraction") < names.index("content_generation"), (
        "concept_extraction must be declared before content_generation so "
        "Kahn's declaration-order tie-break keeps it earlier."
    )


def test_config_loads_without_raising():
    # force_reload re-runs _validate_inputs_from_references; a dangling
    # phase_outputs reference (e.g. course_planning -> concept_extraction
    # after the move) would raise here.
    cfg = _load_workflows_config(force_reload=True)
    assert "textbook_to_course" in cfg["workflows"]


def test_libv2_archival_still_routes_concept_graph_sha256():
    routing = _get_phase_param_routing("libv2_archival")
    assert routing.get("concept_graph_sha256") == (
        "phase_outputs", "concept_extraction", "concept_graph_sha256",
    ), "libv2_archival must still source concept_graph_sha256 from concept_extraction."


def test_topological_sort_preserves_ordering():
    phases_raw = _textbook_phases()
    phases = [
        WorkflowPhase(
            name=p["name"],
            agents=list(p.get("agents", [])),
            depends_on=list(p.get("depends_on", [])),
        )
        for p in phases_raw
    ]
    runner = object.__new__(WorkflowRunner)  # no executor/config needed
    sorted_phases = runner._topological_sort(phases)
    order = [p.name for p in sorted_phases]
    assert order.index("course_planning") < order.index("concept_extraction")
    assert order.index("concept_extraction") < order.index("content_generation")
    # source_mapping + chunking both precede course_planning (its deps).
    assert order.index("source_mapping") < order.index("course_planning")
    assert order.index("chunking") < order.index("course_planning")


# ---------------------------------------------------------------------------
# 2. Fresh-run simulation (the mission regression)
# ---------------------------------------------------------------------------


def test_fresh_run_resolves_synthesized_objectives_and_edges(tmp_path: Path):
    """Disk state exactly as course_planning leaves it; NO reuse kwarg.

    The synthesized_objectives_path kwarg (routed from course_planning)
    must drive a real objectives_source + LO-driven edges.
    """
    synth_path = tmp_path / "synthesized_objectives.json"
    synth_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    payload = _run_concept_extraction(
        tmp_path, synthesized_objectives_path=str(synth_path)
    )

    assert payload["objectives_source"].endswith("synthesized_objectives.json")
    assert payload["learning_outcome_count"] > 0
    graph = payload["_graph"]
    assert _edges_of_type(graph, "prerequisite"), (
        "Fresh run with LO-ref'd chunks must emit >=1 prerequisite edge."
    )
    assert _edges_of_type(graph, "targets-concept"), (
        "Fresh run must emit >=1 targets-concept edge from LO key_concepts."
    )


# ---------------------------------------------------------------------------
# 3. Reuse-run regression
# ---------------------------------------------------------------------------


def test_reuse_objectives_path_wins_over_synthesized(tmp_path: Path):
    reuse_path = tmp_path / "reuse_objectives.json"
    reuse_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")
    synth_path = tmp_path / "synthesized_objectives.json"
    synth_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    payload = _run_concept_extraction(
        tmp_path,
        objectives_path=str(reuse_path),
        synthesized_objectives_path=str(synth_path),
    )
    assert payload["objectives_source"] == str(reuse_path), (
        "The reuse objectives_path kwarg must beat the course_planning "
        "synthesized_objectives_path candidate."
    )


# ---------------------------------------------------------------------------
# 4. Linker-in-phase + write-back
# ---------------------------------------------------------------------------


def test_linker_enriches_and_writes_back_key_concepts(tmp_path: Path):
    """The relocated concept-objective linker enriches key_concepts and
    merges them back into the project-export synthesized_objectives.json;
    the targets-concept edge target is a graph-resident node id."""
    synth_path = tmp_path / "synthesized_objectives.json"
    synth_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    payload = _run_concept_extraction(
        tmp_path, synthesized_objectives_path=str(synth_path)
    )

    assert payload.get("key_concepts_linked", 0) > 0, (
        "The relocated linker must enrich >=1 LO's key_concepts."
    )

    # Write-back: the on-disk synthesized_objectives.json LOs carry
    # key_concepts after the run.
    doc = json.loads(synth_path.read_text(encoding="utf-8"))
    enriched = [
        lo for lo in doc.get("learning_outcomes") or []
        if lo.get("key_concepts") or lo.get("keyConcepts")
    ]
    assert enriched, (
        "Linker-enriched key_concepts must be written back into the "
        "project-export synthesized_objectives.json LO entries."
    )

    # At least one targets-concept edge resolves to a graph-resident
    # concept node — i.e. the linker's enriched key_concepts produced an
    # edge whose target is a real node built from the chunkset's
    # concept_tags (vector-space / eigenvalue). The rule legitimately also
    # emits edges to LO-declared concepts that never became nodes (e.g.
    # linear-map, which no chunk tags), so we assert "at least one
    # resident" rather than "all resident".
    graph = payload["_graph"]
    node_ids = {str(n.get("id")) for n in graph.get("nodes") or []}
    tc_edges = _edges_of_type(graph, "targets-concept")
    assert tc_edges, "expected >=1 targets-concept edge"
    assert any(str(e["target"]) in node_ids for e in tc_edges), (
        f"No targets-concept edge resolves to a graph-resident node. "
        f"targets={[e['target'] for e in tc_edges]}; nodes={sorted(node_ids)}"
    )


def test_reuse_path_is_never_mutated_by_writeback(tmp_path: Path):
    """The write-back must NEVER mutate a user-supplied reuse file."""
    reuse_path = tmp_path / "reuse_objectives.json"
    original = json.dumps(_SYNTHESIZED_OBJECTIVES, sort_keys=True)
    reuse_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    _run_concept_extraction(tmp_path, objectives_path=str(reuse_path))

    after = json.dumps(
        json.loads(reuse_path.read_text(encoding="utf-8")), sort_keys=True
    )
    assert after == original, (
        "The reuse objectives file must be byte-stable — write-back only "
        "ever touches the project-export synthesized_objectives.json."
    )


# ---------------------------------------------------------------------------
# 5. Old-order checkpoint resume
# ---------------------------------------------------------------------------


def test_old_order_checkpoint_resume_runs_course_planning():
    """A stale workflow-state with concept_extraction completed and
    course_planning absent must still let course_planning run (its new
    deps — source_mapping + chunking — are met) without tripping
    _dependencies_met on the removed concept_extraction edge."""
    runner = object.__new__(WorkflowRunner)
    phases_raw = {p["name"]: p for p in _textbook_phases()}
    cp = WorkflowPhase(
        name="course_planning",
        agents=[],
        depends_on=list(phases_raw["course_planning"]["depends_on"]),
    )
    # Simulated old-order checkpoint: concept_extraction already
    # "completed" (from the pre-fix run), course_planning never ran.
    phase_outputs = {
        "source_mapping": {"_completed": True},
        "chunking": {"_completed": True},
        "concept_extraction": {"_completed": True},
        # course_planning intentionally absent.
    }
    assert runner._dependencies_met(cp, phase_outputs), (
        "course_planning deps (source_mapping + chunking) are met, so it "
        "must run even when a stale concept_extraction checkpoint exists."
    )
    # And concept_extraction now depends on course_planning, which is NOT
    # yet complete → it must NOT be considered runnable until planning runs.
    ce = WorkflowPhase(
        name="concept_extraction",
        agents=[],
        depends_on=list(phases_raw["concept_extraction"]["depends_on"]),
    )
    assert not runner._dependencies_met(ce, phase_outputs), (
        "concept_extraction must wait for course_planning under the new "
        "dependency edge."
    )
