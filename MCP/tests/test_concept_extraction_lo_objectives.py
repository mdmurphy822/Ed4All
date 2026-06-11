"""Regression — `_run_concept_extraction` threads course objectives into
the typed-edge build (NVIDIA-KG improvement queue, item 1).

Locks the contract that the ``concept_extraction`` workflow phase
(``MCP/tools/pipeline_tools.py::_run_concept_extraction``) resolves the
synthesized course objectives and passes a ``course.json``-shaped dict
(``learning_outcomes: [{id, ...}, ...]``) into
``Trainforge.rag.typed_edge_inference.build_semantic_graph`` — plus the
derived ``objectives_metadata`` for the ``targets-concept`` rule.

Why this exists: pre-fix the phase called ``build_semantic_graph(...,
course=None, objectives_metadata=None)`` unconditionally, so the
LO-dependent rules (``prerequisite_from_lo_order``,
``targets_concept_from_lo``) early-returned with zero edges on EVERY
course. Verified on a real corpus: ~980 edges, ~760
generic ``related-to`` / ~170 ``defined-by`` / ~50 ``exemplifies``, zero
prerequisite edges — even though the on-disk objectives doc and
LO-tagged chunkset both existed.

Resolution chain under test (first hit wins):

1. Explicit ``objectives_path`` kwarg (routed from the
   ``reuse_objectives_path`` workflow param via ``inputs_from``).
2. ``<project export>/01_learning_objectives/synthesized_objectives.json``.
3. ``<project export>/03_content_development/course.json``.
4. ``LibV2/courses/<slug>/course.json`` (canonical Trainforge form).

Missing / malformed objectives MUST degrade to the legacy
``course=None`` behavior — never fail the phase.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import _build_tool_registry  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# LO ordering for the prerequisite rule: terminal-first (TO-01, TO-02),
# then chapter objectives in chapter order (CO-01, CO-02).
_SYNTHESIZED_OBJECTIVES: Dict[str, Any] = {
    "course_name": "LOGRAPH_101",
    "mint_method": "fixture",
    "duration_weeks": 8,
    "terminal_objectives": [
        {
            "id": "TO-01",
            "statement": "Explain vector spaces and their axioms.",
            "bloom_verb": "Explain",
            "key_concepts": ["vector space", "axiom set"],
        },
        {
            "id": "TO-02",
            "statement": "Apply linear maps to coordinate transforms.",
            "bloom_verb": "Apply",
            "key_concepts": ["linear map"],
        },
    ],
    "chapter_objectives": [
        {
            "chapter": "Week 1",
            "objectives": [
                {
                    "id": "CO-01",
                    "statement": "Identify eigenvalues of a square matrix.",
                    "bloom_verb": "Identify",
                    "key_concepts": ["eigenvalue"],
                },
            ],
        },
        {
            "chapter": "Week 2",
            "objectives": [
                {
                    "id": "CO-02",
                    "statement": "Evaluate determinant-based rank arguments.",
                    "bloom_verb": "Evaluate",
                    "key_concepts": ["determinant"],
                },
            ],
        },
    ],
}

# LibV2-archive alias form (terminal_outcomes / component_objectives) —
# the resolver must tolerate it (root CLAUDE.md --reuse-objectives doc).
_ARCHIVE_FORM_OBJECTIVES: Dict[str, Any] = {
    "course_name": "LOGRAPH_101",
    "terminal_outcomes": _SYNTHESIZED_OBJECTIVES["terminal_objectives"],
    "component_objectives": _SYNTHESIZED_OBJECTIVES["chapter_objectives"],
}


def _lo_tagged_chunkset() -> List[Dict[str, Any]]:
    """Chunks whose ``learning_outcome_refs`` + ``concept_tags`` make the
    prerequisite rule fire.

    ``vector-space`` first appears at LO position 0 (TO-01);
    ``eigenvalue`` first appears at LO position 2 (CO-01); the two
    co-occur in chunk 2 → expected edge
    ``eigenvalue --prerequisite--> vector-space``.
    """
    base_source = {
        "module_id": "week_01",
        "item_path": "week_01/page_001.html",
        "course_id": "LOGRAPH_101",
    }
    chunks: List[Dict[str, Any]] = [
        {
            "id": "lograph_chunk_00001",
            "text": "A vector space is a set closed under addition and "
                    "scalar multiplication, defined by its axiom set.",
            "chunk_type": "explanation",
            "concept_tags": ["vector-space", "axiom-set"],
            "learning_outcome_refs": ["TO-01"],
            "bloom_level": "understand",
            "source": dict(base_source),
        },
        {
            "id": "lograph_chunk_00002",
            "text": "An eigenvalue scales an eigenvector under a linear "
                    "map acting on a vector space.",
            "chunk_type": "explanation",
            "concept_tags": ["eigenvalue", "vector-space"],
            "learning_outcome_refs": ["CO-01"],
            "bloom_level": "apply",
            "source": {**base_source,
                       "module_id": "week_02",
                       "item_path": "week_02/page_002.html"},
        },
        {
            "id": "lograph_chunk_00003",
            "text": "The determinant decides invertibility; rank arguments "
                    "build on eigenvalue structure.",
            "chunk_type": "explanation",
            "concept_tags": ["determinant", "eigenvalue"],
            "learning_outcome_refs": ["CO-02"],
            "bloom_level": "evaluate",
            "source": {**base_source,
                       "module_id": "week_03",
                       "item_path": "week_03/page_003.html"},
        },
    ]
    return chunks


def _write_chunkset(path: Path, chunks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")


def _run(
    tmp_path: Path,
    chunks: List[Dict[str, Any]],
    *,
    objectives_path: Optional[Path] = None,
    synthesized_objectives_path: Optional[Path] = None,
    libv2_course_json: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    chunks_path = tmp_path / "dart_chunks" / "chunks.jsonl"
    _write_chunkset(chunks_path, chunks)
    custom_libv2 = tmp_path / "libv2"

    if libv2_course_json is not None:
        # Slug transform mirrors _run_concept_extraction:
        # lower + underscores→dashes.
        course_dir = custom_libv2 / "courses" / "lograph-101"
        course_dir.mkdir(parents=True, exist_ok=True)
        (course_dir / "course.json").write_text(
            json.dumps(libv2_course_json), encoding="utf-8"
        )

    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name="LOGRAPH_101",
            staging_dir="",
            dart_chunks_path=str(chunks_path),
            libv2_root=str(custom_libv2),
            run_id="WF-TEST-LO-OBJECTIVES",
            objectives_path=str(objectives_path) if objectives_path else "",
            synthesized_objectives_path=(
                str(synthesized_objectives_path)
                if synthesized_objectives_path
                else ""
            ),
        )
    )
    payload = json.loads(result)
    graph_path = Path(payload["concept_graph_path"])
    payload["_graph"] = json.loads(graph_path.read_text(encoding="utf-8"))
    return payload


def _edges_of_type(graph: Dict[str, Any], edge_type: str) -> List[Dict[str, Any]]:
    return [
        e for e in graph.get("edges") or []
        if (e.get("type") or e.get("relation")) == edge_type
    ]


# ---------------------------------------------------------------------------
# Explicit objectives_path kwarg (resolution candidate 1)
# ---------------------------------------------------------------------------


def test_prerequisite_edges_fire_with_explicit_objectives_path(
    tmp_path: Path,
) -> None:
    obj_path = tmp_path / "synthesized_objectives.json"
    obj_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    payload = _run(tmp_path, _lo_tagged_chunkset(), objectives_path=obj_path)
    prereqs = _edges_of_type(payload["_graph"], "prerequisite")
    assert prereqs, (
        "With a resolvable objectives doc and LO-tagged chunks, the "
        "prerequisite_from_lo_order rule must emit edges; got zero. "
        "course=None starvation regressed."
    )
    pairs = {(e["source"], e["target"]) for e in prereqs}
    assert ("eigenvalue", "vector-space") in pairs, (
        f"Expected eigenvalue --prerequisite--> vector-space (LO order "
        f"TO-01 < CO-01); got pairs {sorted(pairs)}."
    )


def test_targets_concept_edges_fire_from_key_concepts(tmp_path: Path) -> None:
    obj_path = tmp_path / "synthesized_objectives.json"
    obj_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    payload = _run(tmp_path, _lo_tagged_chunkset(), objectives_path=obj_path)
    targets = _edges_of_type(payload["_graph"], "targets-concept")
    assert targets, (
        "objectives_metadata derived from the synthesized doc's "
        "key_concepts must make targets_concept_from_lo emit edges."
    )
    pairs = {(e["source"], e["target"]) for e in targets}
    assert ("to-01", "vector-space") in pairs, (
        f"Expected to-01 --targets-concept--> vector-space; got {sorted(pairs)}."
    )


def test_envelope_surfaces_objectives_source(tmp_path: Path) -> None:
    obj_path = tmp_path / "synthesized_objectives.json"
    obj_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    payload = _run(tmp_path, _lo_tagged_chunkset(), objectives_path=obj_path)
    assert payload.get("objectives_source") == str(obj_path)
    assert payload.get("learning_outcome_count") == 4


# ---------------------------------------------------------------------------
# Phase-ordering fix (Option A1): synthesized_objectives_path kwarg
# (resolution candidate #2 — the fresh-run source routed from
# course_planning). The reuse objectives_path kwarg (candidate #1) still
# wins when BOTH are supplied.
# ---------------------------------------------------------------------------


def test_synthesized_objectives_path_resolves_on_fresh_run(
    tmp_path: Path,
) -> None:
    """With NO reuse objectives_path but a course_planning-emitted
    synthesized_objectives_path, the resolver picks candidate #2 so the
    LO-driven edges fire on a fresh run."""
    synth_path = tmp_path / "synthesized_objectives.json"
    synth_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    payload = _run(
        tmp_path,
        _lo_tagged_chunkset(),
        synthesized_objectives_path=synth_path,
    )
    assert payload.get("objectives_source") == str(synth_path)
    assert payload.get("learning_outcome_count") == 4
    assert _edges_of_type(payload["_graph"], "prerequisite"), (
        "Fresh-run synthesized_objectives_path candidate must give the "
        "prerequisite_from_lo_order rule a learning-outcomes ordering."
    )


def test_reuse_objectives_path_beats_synthesized_candidate(
    tmp_path: Path,
) -> None:
    """When BOTH the reuse objectives_path (candidate #1) and the
    course_planning synthesized_objectives_path (candidate #2) are
    supplied, the reuse path wins."""
    reuse_path = tmp_path / "reuse_objectives.json"
    reuse_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")
    synth_path = tmp_path / "synthesized_objectives.json"
    synth_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")

    payload = _run(
        tmp_path,
        _lo_tagged_chunkset(),
        objectives_path=reuse_path,
        synthesized_objectives_path=synth_path,
    )
    assert payload.get("objectives_source") == str(reuse_path), (
        "The reuse objectives_path kwarg must remain candidate #1 ahead "
        "of the course_planning-emitted synthesized_objectives_path."
    )


# ---------------------------------------------------------------------------
# LibV2 archive alias form (terminal_outcomes / component_objectives)
# ---------------------------------------------------------------------------


def test_archive_form_objectives_are_normalized(tmp_path: Path) -> None:
    obj_path = tmp_path / "objectives_archive_form.json"
    obj_path.write_text(json.dumps(_ARCHIVE_FORM_OBJECTIVES), encoding="utf-8")

    payload = _run(tmp_path, _lo_tagged_chunkset(), objectives_path=obj_path)
    prereqs = _edges_of_type(payload["_graph"], "prerequisite")
    assert prereqs, (
        "The LibV2 archive form (terminal_outcomes/component_objectives) "
        "must normalize to the course.json shape; prerequisite rule "
        "emitted zero edges."
    )


# ---------------------------------------------------------------------------
# LibV2 course-dir course.json fallback (resolution candidate 4)
# ---------------------------------------------------------------------------


def test_libv2_course_json_fallback(tmp_path: Path) -> None:
    course_json = {
        "course_code": "LOGRAPH_101",
        "title": "Linear Algebra via Graphs",
        "learning_outcomes": [
            {"id": "TO-01", "statement": "Explain vector spaces."},
            {"id": "TO-02", "statement": "Apply linear maps."},
            {"id": "CO-01", "statement": "Identify eigenvalues."},
            {"id": "CO-02", "statement": "Evaluate determinants."},
        ],
    }
    payload = _run(
        tmp_path, _lo_tagged_chunkset(), libv2_course_json=course_json
    )
    prereqs = _edges_of_type(payload["_graph"], "prerequisite")
    assert prereqs, (
        "With no explicit objectives_path and no project export, the "
        "phase must fall back to LibV2/courses/<slug>/course.json."
    )


# ---------------------------------------------------------------------------
# Graceful degradation (course=None legacy path)
# ---------------------------------------------------------------------------


def test_missing_objectives_degrades_gracefully(tmp_path: Path) -> None:
    payload = _run(tmp_path, _lo_tagged_chunkset())
    assert payload.get("success") is True, (
        "Missing objectives must never fail the phase."
    )
    graph = payload["_graph"]
    assert graph.get("nodes"), "Graph must still build without objectives."
    assert not _edges_of_type(graph, "prerequisite"), (
        "Without objectives the prerequisite rule has no LO ordering; "
        "legacy zero-edge behavior expected."
    )
    assert payload.get("objectives_source") in ("", None)


def test_malformed_objectives_degrades_gracefully(tmp_path: Path) -> None:
    obj_path = tmp_path / "broken.json"
    obj_path.write_text("{not valid json", encoding="utf-8")

    payload = _run(tmp_path, _lo_tagged_chunkset(), objectives_path=obj_path)
    assert payload.get("success") is True, (
        "A malformed objectives doc must degrade to course=None, not "
        "fail the phase."
    )
    assert not _edges_of_type(payload["_graph"], "prerequisite")


# ---------------------------------------------------------------------------
# Fix-2 — LO key_concepts seed the in-memory chunk retag BEFORE the
# co-occurrence build, so a key_concept present in ≥2 chunks' prose mints a
# real co-occurrence node (chunk-anchored, frequency>0) and its
# targets-concept edge resolves to that grounded node instead of a
# frequency=0 ``lo_key_concept`` orphan.
# ---------------------------------------------------------------------------

# Objectives whose key_concepts are NOT pre-present in any chunk's
# concept_tags but DO appear (multi-word, spaced) in the chunk text.
_FIX2_OBJECTIVES: Dict[str, Any] = {
    "course_name": "FIX2_101",
    "mint_method": "fixture",
    "duration_weeks": 8,
    "terminal_objectives": [
        {
            "id": "TO-01",
            "statement": "Explain visual hierarchy in interface design.",
            "bloom_verb": "Explain",
            # multi-word key_concept → slug ``visual-hierarchy``; its
            # de-slugged alias "visual hierarchy" must match the prose.
            "key_concepts": ["visual hierarchy"],
        },
    ],
    "chapter_objectives": [
        {
            "chapter": "Week 1",
            "objectives": [
                {
                    "id": "CO-01",
                    "statement": "Apply white space to reduce clutter.",
                    "bloom_verb": "Apply",
                    "key_concepts": ["white space"],
                },
            ],
        },
    ],
}


def _fix2_chunkset() -> List[Dict[str, Any]]:
    """Chunks whose ``concept_tags`` do NOT pre-list the LO key_concepts,
    but whose ``text`` carries the spaced surface form in ≥2 chunks each.

    "visual hierarchy" appears in chunks 1 + 2; "white space" appears in
    chunks 2 + 3 — both clear the co-occurrence ``min_freq=2`` floor once
    the Fix-2 retag tags them.
    """
    base = {
        "module_id": "week_01",
        "item_path": "week_01/page.html",
        "course_id": "FIX2_101",
    }
    return [
        {
            "id": "fix2_chunk_00001",
            "text": "Good interfaces use visual hierarchy to guide the "
                    "eye through the layout and emphasize key actions.",
            "chunk_type": "explanation",
            "concept_tags": ["design"],
            "learning_outcome_refs": ["TO-01"],
            "bloom_level": "understand",
            "source": dict(base),
        },
        {
            "id": "fix2_chunk_00002",
            "text": "A strong visual hierarchy together with deliberate "
                    "white space separates distinct content regions.",
            "chunk_type": "explanation",
            "concept_tags": ["design"],
            "learning_outcome_refs": ["TO-01"],
            "bloom_level": "understand",
            "source": {**base,
                       "module_id": "week_01",
                       "item_path": "week_01/page2.html"},
        },
        {
            "id": "fix2_chunk_00003",
            "text": "Designers apply white space deliberately to reduce "
                    "visual clutter and improve readability.",
            "chunk_type": "explanation",
            "concept_tags": ["clutter"],
            "learning_outcome_refs": ["CO-01"],
            "bloom_level": "apply",
            "source": {**base,
                       "module_id": "week_02",
                       "item_path": "week_02/page3.html"},
        },
    ]


def _node_by_id(graph: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    for n in graph.get("nodes") or []:
        if isinstance(n, dict) and n.get("id") == node_id:
            return n
    return None


def test_lo_key_concepts_mint_real_cooccurrence_nodes(tmp_path: Path) -> None:
    """A multi-word LO key_concept present in ≥2 chunks' prose mints a
    real co-occurrence node (frequency>0, NOT node_provenance=lo_key_concept)
    and a chunk-anchored edge, with the targets-concept edge resolving to it.
    """
    obj_path = tmp_path / "objectives.json"
    obj_path.write_text(json.dumps(_FIX2_OBJECTIVES), encoding="utf-8")

    payload = _run(tmp_path, _fix2_chunkset(), objectives_path=obj_path)

    # The retag fired and seeded the LO key_concepts.
    assert payload.get("lo_key_concept_seed_count") == 2, (
        "Fix-2: both LO key_concepts should compile into seeds."
    )
    assert payload.get("lo_key_concept_chunks_retagged", 0) >= 2

    graph = payload["_graph"]
    vh = _node_by_id(graph, "visual-hierarchy")
    assert vh is not None, (
        "Fix-2: 'visual hierarchy' (in 2 chunks' prose) must mint a "
        "co-occurrence node after the LO key_concept retag."
    )
    # Real, chunk-anchored node — NOT a frequency=0 lo_key_concept orphan.
    assert vh.get("node_provenance") != "lo_key_concept", (
        f"Expected a chunk-anchored node, got an LO-orphan: {vh}"
    )
    assert (vh.get("frequency") or 0) > 0, (
        f"A real co-occurrence node must carry frequency>0; got {vh}"
    )

    # The targets-concept edge resolves to the grounded node.
    targets = _edges_of_type(graph, "targets-concept")
    pairs = {(e["source"], e["target"]) for e in targets}
    assert ("to-01", "visual-hierarchy") in pairs, (
        f"Expected to-01 --targets-concept--> visual-hierarchy; got "
        f"{sorted(pairs)}."
    )
    # And NO frequency=0 lo_key_concept orphan was minted for it.
    assert payload.get("lo_concept_nodes_materialized") == 0, (
        "Both LO key_concepts are prose-grounded, so zero orphan "
        "lo_key_concept nodes should be materialized."
    )


def test_lo_key_concept_chunk_anchored_after_retag(tmp_path: Path) -> None:
    """The minted node is reachable through chunk-anchored edges (it
    participates in the co-occurrence + typed-edge structure rather than
    dangling)."""
    obj_path = tmp_path / "objectives.json"
    obj_path.write_text(json.dumps(_FIX2_OBJECTIVES), encoding="utf-8")

    payload = _run(tmp_path, _fix2_chunkset(), objectives_path=obj_path)
    graph = payload["_graph"]

    incident = [
        e for e in graph.get("edges") or []
        if "visual-hierarchy" in (e.get("source"), e.get("target"))
    ]
    assert incident, (
        "Fix-2: the minted visual-hierarchy node must carry incident "
        "edges (it is grounded, not a dangling endpoint)."
    )


def test_prose_absent_key_concept_falls_through_to_lo_orphan(
    tmp_path: Path,
) -> None:
    """Behavior preserved: a key_concept that does NOT appear in any
    chunk's prose still materializes as the legacy frequency=0
    ``node_provenance=lo_key_concept`` endpoint."""
    objectives = {
        "course_name": "FIX2_101",
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Explain a concept absent from the corpus.",
                "bloom_verb": "Explain",
                # This phrase appears in NO chunk text below.
                "key_concepts": ["nonexistent absent concept"],
            },
        ],
    }
    obj_path = tmp_path / "objectives.json"
    obj_path.write_text(json.dumps(objectives), encoding="utf-8")

    # Chunks already carry a (different) concept_tag so the graph builds,
    # but none of them mention the LO key_concept.
    chunks = [
        {
            "id": "fix2_chunk_00001",
            "text": "Good interfaces use layout to guide the eye.",
            "chunk_type": "explanation",
            "concept_tags": ["layout", "design"],
            "learning_outcome_refs": ["TO-01"],
            "bloom_level": "understand",
            "source": {"module_id": "w1", "item_path": "w1/p.html",
                       "course_id": "FIX2_101"},
        },
        {
            "id": "fix2_chunk_00002",
            "text": "Layout and design choices shape the reading order.",
            "chunk_type": "explanation",
            "concept_tags": ["layout", "design"],
            "learning_outcome_refs": ["TO-01"],
            "bloom_level": "understand",
            "source": {"module_id": "w1", "item_path": "w1/p2.html",
                       "course_id": "FIX2_101"},
        },
    ]

    payload = _run(tmp_path, chunks, objectives_path=obj_path)
    graph = payload["_graph"]

    orphan = _node_by_id(graph, "nonexistent-absent-concept")
    assert orphan is not None, (
        "A prose-absent key_concept must still materialize as the legacy "
        "lo_key_concept endpoint (behavior preserved)."
    )
    assert orphan.get("node_provenance") == "lo_key_concept"
    assert orphan.get("frequency") == 0
    assert payload.get("lo_concept_nodes_materialized", 0) >= 1
