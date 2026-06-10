"""Bloom-level fallback in `_normalize_objectives_payload_to_course` +
end-to-end LO-concept materialization through `_run_concept_extraction`.

Merge-vs-LO-edges fix, Steps 3 + 4 + 11.

Step 3 — when an LO carries ``key_concepts`` but neither an explicit canonical
``bloom_level`` nor a verb ``detect_bloom_level`` can resolve from its
statement, the normalizer falls back to ``_FALLBACK_BLOOM_LEVEL`` ("apply")
with a logged warning rather than dropping the LO's targetedConcepts.

Step 11 — under the flagged config (TRAINFORGE_MERGE_DUPLICATE_CONCEPTS=true),
the persisted graph keeps every pre-merge ``targets-concept`` edge (phantom
endpoints are materialized, not dropped) and the envelope surfaces
``lo_concept_nodes_materialized > 0``.

Covers:

 9. LO statement "Integrate …" (no canonical verb, no explicit bloom_level)
    → objectives_metadata entry present with bloomLevel "apply"; warning logged.
10. LO with an explicit canonical bloom_level → unchanged (no fallback).
11. End-to-end flagged run through _run_concept_extraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.tools.pipeline_tools import (  # noqa: E402
    _build_tool_registry,
    _normalize_objectives_payload_to_course,
)


# ---------------------------------------------------------------------------
# 9. Bloom fallback fires
# ---------------------------------------------------------------------------


def test_bloom_fallback_keeps_targeted_concepts(caplog) -> None:
    """An LO with key_concepts but no resolvable Bloom level still contributes
    a targetedConcepts entry at the fallback level."""
    payload = {
        "course_name": "BLOOM_FB_101",
        "terminal_objectives": [
            {
                # "Integrate" is not in the Bloom verb table, so neither the
                # explicit bloom_level nor detect_bloom_level resolves a level.
                "id": "TO-01",
                "statement": "Integrate the gizmo with the doohickey subsystem.",
                "key_concepts": ["gizmo", "doohickey subsystem"],
            },
        ],
        "chapter_objectives": [],
    }
    with caplog.at_level(logging.WARNING):
        course, objectives_metadata = _normalize_objectives_payload_to_course(
            payload, "BLOOM_FB_101"
        )

    assert course is not None
    assert objectives_metadata, "fallback must preserve the LO's targetedConcepts"
    entry = next(e for e in objectives_metadata if e["id"] == "TO-01")
    assert entry["targetedConcepts"], "concepts must survive the fallback"
    for tc in entry["targetedConcepts"]:
        assert tc["bloomLevel"] == "apply"

    assert any(
        "no resolvable bloom_level" in rec.getMessage()
        and "TO-01" in rec.getMessage()
        for rec in caplog.records
    ), "a warning naming the LO must be logged when the fallback fires"


# ---------------------------------------------------------------------------
# 10. Canonical bloom_level → no fallback
# ---------------------------------------------------------------------------


def test_canonical_bloom_level_unchanged(caplog) -> None:
    payload = {
        "course_name": "BLOOM_FB_101",
        "terminal_objectives": [
            {
                "id": "TO-01",
                "statement": "Some statement.",
                "bloom_level": "analyze",
                "key_concepts": ["widget"],
            },
        ],
        "chapter_objectives": [],
    }
    with caplog.at_level(logging.WARNING):
        _course, objectives_metadata = _normalize_objectives_payload_to_course(
            payload, "BLOOM_FB_101"
        )

    entry = next(e for e in objectives_metadata if e["id"] == "TO-01")
    assert all(tc["bloomLevel"] == "analyze" for tc in entry["targetedConcepts"])
    assert not any(
        "no resolvable bloom_level" in rec.getMessage() for rec in caplog.records
    ), "no fallback warning when an explicit canonical level is present"


# ---------------------------------------------------------------------------
# 11. End-to-end flagged run through _run_concept_extraction
# ---------------------------------------------------------------------------


# Reuses the fixture pattern from test_concept_extraction_lo_objectives.py
# (test_targets_concept_edges_fire_from_key_concepts). TO-02's key_concept
# "linear map" (→ slug "linear-map") is NOT in any chunk's concept_tags, so it
# is a phantom targets-concept target that must be materialized rather than
# dropped by the merge pass.
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
            # "linear map" is NOT tagged on any chunk → phantom endpoint.
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
    ],
}


def _lo_tagged_chunkset() -> List[Dict[str, Any]]:
    base_source = {
        "module_id": "week_01",
        "item_path": "week_01/page_001.html",
        "course_id": "LOGRAPH_101",
    }
    return [
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


def _write_chunkset(path: Path, chunks: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk) + "\n")


def test_flagged_run_preserves_targets_concept_edges_and_materializes(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("TRAINFORGE_MERGE_DUPLICATE_CONCEPTS", "true")
    monkeypatch.setenv("TRAINFORGE_RELATED_FANOUT_CAP", "8")
    monkeypatch.setenv("TRAINFORGE_INTRA_CHUNK_LINKS", "true")

    chunks = _lo_tagged_chunkset()
    chunks_path = tmp_path / "dart_chunks" / "chunks.jsonl"
    _write_chunkset(chunks_path, chunks)
    obj_path = tmp_path / "synthesized_objectives.json"
    obj_path.write_text(json.dumps(_SYNTHESIZED_OBJECTIVES), encoding="utf-8")
    custom_libv2 = tmp_path / "libv2"

    registry = _build_tool_registry()
    tool = registry["run_concept_extraction"]
    result = asyncio.run(
        tool(
            project_id="",
            course_name="LOGRAPH_101",
            staging_dir="",
            dart_chunks_path=str(chunks_path),
            libv2_root=str(custom_libv2),
            run_id="WF-TEST-BLOOM-FALLBACK",
            objectives_path=str(obj_path),
            synthesized_objectives_path="",
        )
    )
    payload = json.loads(result)
    graph = json.loads(
        Path(payload["concept_graph_path"]).read_text(encoding="utf-8")
    )

    # Pre-merge targets-concept emit count == one edge per (LO, key_concept):
    # TO-01 → vector-space, axiom-set; TO-02 → linear-map; CO-01 → eigenvalue.
    targets = [
        e for e in graph.get("edges") or []
        if (e.get("type") or e.get("relation")) == "targets-concept"
    ]
    expected_pairs = {
        ("to-01", "vector-space"),
        ("to-01", "axiom-set"),
        ("to-02", "linear-map"),
        ("co-01", "eigenvalue"),
    }
    persisted_pairs = {(e["source"], e["target"]) for e in targets}
    assert expected_pairs <= persisted_pairs, (
        f"flagged merge pass dropped targets-concept edges: "
        f"missing {sorted(expected_pairs - persisted_pairs)}"
    )

    # Every targets-concept edge endpoint resolves to a node (no dangling).
    node_ids = {
        n["id"] for n in graph.get("nodes") or []
        if isinstance(n, dict) and isinstance(n.get("id"), str)
    }
    for e in targets:
        assert e["target"] in node_ids, (
            f"targets-concept target {e['target']!r} has no node"
        )

    # "linear-map" was a phantom key_concept → materialized DomainConcept.
    materialized = [
        n for n in graph.get("nodes") or []
        if isinstance(n, dict) and n.get("node_provenance") == "lo_key_concept"
    ]
    materialized_ids = {n["id"] for n in materialized}
    assert "linear-map" in materialized_ids
    for n in materialized:
        assert n["class"] == "DomainConcept"
        assert n.get("frequency") == 0

    # Envelope surfaces the count.
    assert payload.get("lo_concept_nodes_materialized", 0) > 0
