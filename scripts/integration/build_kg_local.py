#!/usr/bin/env python3
"""Deterministic concept-knowledge-graph builder + KG-quality reporter.

Standalone, reproducible reconstruction of the production
``MCP/tools/pipeline_tools.py::_run_concept_extraction`` graph-build
sequence, with NO live LLM dispatch. Builds the semantic concept graph
for a course from its on-disk ``semantik_chunks/chunks.jsonl`` plus an
optional ``synthesized_objectives.json``, runs the edge-consensus
aggregator and the KG-quality reporter, and prints the four KG-quality
dimension scores (completeness / consistency / accuracy / coverage)
against the canonical thresholds (0.95 / 0.95 / 0.95 / 0.50), plus the
overall pass/fail.

The build path mirrors ``_run_concept_extraction`` faithfully:

    1. load chunks (jsonl) — chunks already carry concept_tags, so NO
       Stage-3 LLM vocabulary pass and NO lexical-seed re-tag is needed
       (those only fire when chunks lack tags).
    2. load + normalize objectives via
       ``_normalize_objectives_payload_to_course`` (Courseforge
       synthesized form OR LibV2 archive form).
    3. set the measured-best graph-shaping env defaults in-process via
       ``os.environ.setdefault`` (the same set
       ``MCP/core/workflow_runner.py::_apply_corpus_generalization_defaults``
       turns on for a pipeline run — page grouping, merge, intra-chunk
       links, fan-out cap, normalize labels, prune/seed/fragment-filter).
    4. ``build_cooccurrence_graph`` (group_by from env) ->
       concept-objective linker -> ``build_semantic_graph`` ->
       merge-duplicate-concepts -> intra-chunk links -> related-to
       fan-out cap (each env-gated, matching the handler).
    5. ``EdgeConsensusAggregator.apply_to_graph`` (stamp edge_status).
    6. ``KGQualityReporter.compute_metrics_only`` (authoring-time entry
       point; the SAME call the handler uses to populate kg_quality) +
       a sibling write of the asserted ``concept_graph.json`` so the
       gate-path ``KGQualityValidator`` can also be run honestly.

Usage:
    python -m scripts.integration.build_kg_local \
        --chunks LibV2/courses/<slug>/semantik_chunks/chunks.jsonl \
        --course-slug <slug> \
        --objectives LibV2/courses/<slug>/sources/objectives/synthesized_objectives.json \
        --out-dir /tmp/kg-verify

Exit 0 when every dimension clears its threshold; 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


# Canonical thresholds (lib/validators/kg_quality.py).
_THRESHOLDS: Dict[str, float] = {
    "completeness": 0.95,
    "consistency": 0.95,
    "accuracy": 0.95,
    "coverage": 0.50,
}
_DIMENSIONS = ("completeness", "consistency", "accuracy", "coverage")


# Measured-best graph-shaping env defaults — byte-identical to
# MCP/core/workflow_runner.py::_CORPUS_GENERALIZATION_ENV_DEFAULTS, minus
# the two licensing-sensitive LLM-provider envs (TEXTBOOK_SYNTHESIS_PROVIDER
# / TRAINFORGE_SYNTHESIS_PROVIDER), which would dispatch a live LLM. The
# chunks already carry concept_tags, so Stage-3 synthesis / lexical-seed
# re-tag is unnecessary and the graph build is fully deterministic.
_GRAPH_SHAPING_ENV_DEFAULTS: Dict[str, str] = {
    "TRAINFORGE_PRUNE_SCAFFOLDING_CONCEPTS": "true",
    "TRAINFORGE_SEED_TECH_CONCEPTS": "true",
    "TRAINFORGE_FILTER_FRAGMENT_CONCEPTS": "true",
    "TRAINFORGE_CHUNK_TYPE_CONTENT_AWARE": "true",
    "TRAINFORGE_MERGE_DUPLICATE_CONCEPTS": "true",
    "TRAINFORGE_INTRA_CHUNK_LINKS": "true",
    "TRAINFORGE_RELATED_FANOUT_CAP": "8",
    "TRAINFORGE_NORMALIZE_LABELS": "true",
    "TRAINFORGE_COOCCURRENCE_GROUP_BY": "page",
}


def _repo_root() -> Path:
    """Resolve the Ed4All repo root (this file lives at scripts/integration/)."""
    return Path(__file__).resolve().parents[2]


def _load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    """Load a chunks.jsonl into a list of dicts (skip blank / malformed)."""
    chunks: List[Dict[str, Any]] = []
    with chunks_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except ValueError as exc:
                print(
                    f"  WARN: skipping malformed chunk at line {line_no}: {exc}",
                    file=sys.stderr,
                )
    return chunks


def _derive_misconceptions(
    chunks: List[Dict[str, Any]], course_code: str
) -> List[Dict[str, Any]]:
    """Mirror _run_concept_extraction._derive_misconceptions."""
    try:
        from Trainforge.rag.typed_edge_inference import _make_concept_id
        from Trainforge.pipeline.process_course import _route_misconception_to_tag
        from lib.ontology.misconception_id import canonical_mc_id
    except Exception:  # noqa: BLE001 — rule self-skips on []
        return []
    entities: List[Dict[str, Any]] = []
    seen: set = set()
    cid_course = course_code.upper() or ""
    for chunk in chunks:
        raw = chunk.get("misconceptions") or []
        if not raw:
            continue
        tags = [t for t in (chunk.get("concept_tags") or []) if t]
        for entry in raw:
            if isinstance(entry, dict):
                statement = (entry.get("misconception") or "").strip()
                correction = (entry.get("correction") or "").strip()
                explicit_cid = (entry.get("concept_id") or "").strip() or None
                bloom_level = (entry.get("bloom_level") or "").strip().lower()
                cognitive_domain = (entry.get("cognitive_domain") or "").strip()
            elif isinstance(entry, str):
                statement, correction, explicit_cid = entry.strip(), "", None
                bloom_level, cognitive_domain = "", ""
            else:
                continue
            if not statement:
                continue
            mc_id = canonical_mc_id(statement, correction, bloom_level)
            if mc_id in seen:
                continue
            seen.add(mc_id)
            entity: Dict[str, Any] = {
                "id": mc_id,
                "misconception": statement,
                "correction": correction or statement,
            }
            if bloom_level:
                entity["bloom_level"] = bloom_level
            if cognitive_domain:
                entity["cognitive_domain"] = cognitive_domain
            concept_id = explicit_cid
            if not concept_id and tags:
                routed_tag = _route_misconception_to_tag(statement, tags)
                if routed_tag:
                    concept_id = _make_concept_id(routed_tag, cid_course)
            if concept_id:
                entity["concept_id"] = concept_id
            entities.append(entity)
    return entities


def _derive_questions(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mirror _run_concept_extraction._derive_questions."""
    questions: List[Dict[str, Any]] = []
    for chunk in chunks:
        if chunk.get("chunk_type") != "assessment_item":
            continue
        chunk_id = chunk.get("id")
        if not chunk_id:
            continue
        for ref in chunk.get("learning_outcome_refs") or []:
            if not ref:
                continue
            questions.append({
                "id": f"q_{chunk_id}_{ref}",
                "objective_id": ref,
                "source_chunk_id": chunk_id,
            })
    return questions


def build_kg(
    *,
    chunks: List[Dict[str, Any]],
    course_slug: str,
    objectives_payload: Optional[Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    """Build the semantic concept graph — faithful to _run_concept_extraction.

    Returns the graph dict (post edge-consensus stamping + kg_quality
    metrics-only computation).
    """
    from lib.ontology.cooccurrence_graph import build_cooccurrence_graph
    from Trainforge.rag.typed_edge_inference import build_semantic_graph

    course_code = course_slug.upper().replace("-", "_")

    # --- normalize objectives to the course.json graph-input shape ----------
    course_for_graph: Optional[Dict[str, Any]] = None
    objectives_meta_for_graph: Optional[List[Dict[str, Any]]] = None
    if objectives_payload is not None:
        from MCP.tools.pipeline_tools import (
            _normalize_objectives_payload_to_course,
        )
        course_for_graph, objectives_meta_for_graph = (
            _normalize_objectives_payload_to_course(
                objectives_payload, course_code
            )
        )
        if course_for_graph is not None:
            lo_count = len(course_for_graph.get("learning_outcomes") or [])
            print(f"  objectives normalized: {lo_count} canonical LOs")
        else:
            print("  objectives produced ZERO canonical LOs (LO rules skip)")

    # --- co-occurrence graph (page-grouped by env) --------------------------
    group_by = os.environ.get("TRAINFORGE_COOCCURRENCE_GROUP_BY", "").strip() or "chunk"
    cooccurrence_graph = build_cooccurrence_graph(
        chunks, course_code, graph_kind="concept", group_by=group_by,
    )
    print(
        f"  cooccurrence graph (group_by={group_by}): "
        f"{len(cooccurrence_graph.get('nodes', []))} nodes, "
        f"{len(cooccurrence_graph.get('edges', []))} edges"
    )

    # --- concept-objective linker (enrich LO key_concepts) ------------------
    if course_for_graph is not None:
        try:
            from lib.ontology.concept_objective_linker import (
                link_concepts_to_objectives,
            )
            enriched_los = link_concepts_to_objectives(
                course_for_graph.get("learning_outcomes") or [],
                cooccurrence_graph,
            )
            from MCP.tools.pipeline_tools import (
                _normalize_objectives_payload_to_course as _renorm,
            )
            relinked_course, relinked_meta = _renorm(
                {
                    "learning_outcomes": enriched_los,
                    "course_code": course_code,
                },
                course_code,
            )
            if relinked_course is not None:
                course_for_graph = relinked_course
                objectives_meta_for_graph = relinked_meta
                linked = sum(
                    1 for lo in enriched_los
                    if lo.get("key_concepts") or lo.get("keyConcepts")
                )
                print(
                    f"  concept-objective linker enriched {linked}/"
                    f"{len(enriched_los)} LOs"
                )
        except Exception as exc:  # noqa: BLE001 — fail-soft (matches handler)
            print(f"  WARN: concept-objective linker failed: {exc}")

    # --- typed-edge semantic graph ------------------------------------------
    misconceptions = _derive_misconceptions(chunks, course_code)
    questions = _derive_questions(chunks)
    graph = build_semantic_graph(
        chunks,
        course=course_for_graph,
        concept_graph=cooccurrence_graph,
        misconceptions=misconceptions or None,
        questions=questions or None,
        objectives_metadata=objectives_meta_for_graph,
        run_id=run_id,
    )
    if isinstance(graph, dict):
        graph.setdefault("course_id", course_code)

    # --- KG-quality post-build passes (env-gated, order load-bearing) -------
    if isinstance(graph, dict):
        if os.getenv("TRAINFORGE_MERGE_DUPLICATE_CONCEPTS", "").lower() == "true":
            try:
                from lib.ontology.concept_node_merge import (
                    merge_duplicate_concept_nodes,
                )
                graph = merge_duplicate_concept_nodes(graph)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: concept-merge pass failed: {exc}")
        if os.getenv("TRAINFORGE_INTRA_CHUNK_LINKS", "").lower() == "true":
            try:
                from lib.ontology.intra_chunk_linker import (
                    link_intra_chunk_concepts,
                )
                graph = link_intra_chunk_concepts(graph)
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: intra-chunk linking pass failed: {exc}")
        fanout_k = os.getenv("TRAINFORGE_RELATED_FANOUT_CAP", "").strip()
        if fanout_k.isdigit() and int(fanout_k) > 0:
            try:
                from lib.ontology.related_edge_cap import cap_related_fanout
                graph = cap_related_fanout(graph, int(fanout_k))
            except Exception as exc:  # noqa: BLE001
                print(f"  WARN: related-to fan-out cap failed: {exc}")

    return graph


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically build a course concept knowledge graph from "
            "on-disk semantik_chunks + synthesized objectives and run the "
            "KG-quality reporter/validator (no live LLM)."
        )
    )
    parser.add_argument(
        "--chunks", required=True,
        help="Path to semantik_chunks/chunks.jsonl",
    )
    parser.add_argument(
        "--course-slug", required=True,
        help="Course slug (e.g. my-course-101)",
    )
    parser.add_argument(
        "--objectives", default=None,
        help=(
            "Optional path to synthesized_objectives.json (Courseforge "
            "synthesized form or LibV2 archive form)"
        ),
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output dir; writes concept_graph/ + quality/ beneath it",
    )
    parser.add_argument(
        "--run-id", default="kg-local-build",
        help="Run identifier stamped on nodes/edges (default kg-local-build)",
    )
    args = parser.parse_args(argv)

    # Make repo imports resolvable when invoked as a plain script.
    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    chunks_path = Path(args.chunks).resolve()
    if not chunks_path.is_file():
        print(f"ERROR: chunks file not found: {chunks_path}", file=sys.stderr)
        return 1

    # Set the measured-best graph-shaping env defaults (setdefault — an
    # operator-pinned value wins, matching _apply_corpus_generalization_defaults).
    for env_var, value in _GRAPH_SHAPING_ENV_DEFAULTS.items():
        os.environ.setdefault(env_var, value)

    print(f"Course slug : {args.course_slug}")
    print(f"Chunks      : {chunks_path}")
    chunks = _load_chunks(chunks_path)
    if not chunks:
        print("ERROR: no chunks loaded", file=sys.stderr)
        return 1
    distinct_tags = {t for c in chunks for t in (c.get("concept_tags") or [])}
    print(f"  loaded {len(chunks)} chunks; {len(distinct_tags)} distinct concept_tags")

    objectives_payload: Optional[Dict[str, Any]] = None
    if args.objectives:
        obj_path = Path(args.objectives).resolve()
        if not obj_path.is_file():
            print(f"ERROR: objectives file not found: {obj_path}", file=sys.stderr)
            return 1
        print(f"Objectives  : {obj_path}")
        try:
            objectives_payload = json.loads(obj_path.read_text(encoding="utf-8"))
        except ValueError as exc:
            print(f"ERROR: objectives JSON parse failed: {exc}", file=sys.stderr)
            return 1

    # --- build the graph ----------------------------------------------------
    print("\nBuilding semantic concept graph ...")
    graph = build_kg(
        chunks=chunks,
        course_slug=args.course_slug,
        objectives_payload=objectives_payload,
        run_id=args.run_id,
    )
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    print(f"  built graph: {len(nodes)} nodes, {len(edges)} edges")
    if not edges:
        print("ERROR: empty graph (no edges) — cannot score", file=sys.stderr)
        return 1

    # --- edge-consensus stamping (apply_to_graph, in place) -----------------
    from lib.aggregators.edge_consensus import EdgeConsensusAggregator
    try:
        EdgeConsensusAggregator(
            semantic_graph_path=None,
            course_slug=args.course_slug,
            run_id=args.run_id,
        ).apply_to_graph(graph)
    except Exception as exc:  # noqa: BLE001 — fail-soft (matches handler)
        print(f"  WARN: edge-consensus stamping failed: {exc}")

    # --- KG-quality (authoring-time entry point) ----------------------------
    from Trainforge.rag.kg_quality_report import KGQualityReporter

    out_dir = Path(args.out_dir).resolve()
    graph_dir = out_dir / "concept_graph"
    quality_dir = out_dir / "quality"
    graph_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    reporter = KGQualityReporter(
        course_slug=args.course_slug,
        run_id=args.run_id,
        output_dir=quality_dir,
    )
    report = reporter.compute_metrics_only(graph)

    # Stamp the compact four-dimension score block onto the graph BEFORE
    # serialization (mirrors _run_concept_extraction so the in-graph
    # kg_quality field is populated, not null).
    _stamp_dims = report.get("dimensions") or {}
    graph["kg_quality"] = {
        dim: float((_stamp_dims.get(dim) or {}).get("score", 0.0))
        for dim in _DIMENSIONS
    }

    # --- write artifacts ----------------------------------------------------
    semantic_path = graph_dir / "concept_graph_semantic.json"
    semantic_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # The production _run_concept_extraction handler deliberately does NOT
    # write a separate asserted concept_graph.json — it emits only the
    # semantic graph + manifest + edge_consensus_report + kg_quality_report
    # (computed via compute_metrics_only). The gate-path KGQualityValidator
    # at libv2_archival routes concept_graph_path -> a SIBLING
    # concept_graph.json that does not exist, so its compute() reads an empty
    # asserted graph. We deliberately do NOT fabricate that file: the
    # gate-path cross-check below points concept_graph_path at the (absent)
    # sibling so it reflects exactly what production scores.
    concept_path = graph_dir / "concept_graph.json"

    report_path = quality_dir / "kg_quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Sibling edge-consensus report.
    try:
        EdgeConsensusAggregator(
            semantic_graph_path=semantic_path,
            course_slug=args.course_slug,
            run_id=args.run_id,
        ).write(graph_dir / "edge_consensus_report.json")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: edge_consensus_report.json write failed: {exc}")

    # --- print dimension scores (authoring-time reporter) -------------------
    dims = report.get("dimensions", {})
    scores: Dict[str, float] = {
        dim: float((dims.get(dim) or {}).get("score", 0.0))
        for dim in _DIMENSIONS
    }

    cov_detail = dims.get("coverage", {})
    print("\n" + "=" * 64)
    print(f"KG-quality (compute_metrics_only) — course={args.course_slug}")
    print("=" * 64)
    all_pass = True
    for dim in _DIMENSIONS:
        score = scores[dim]
        thr = _THRESHOLDS[dim]
        ok = score >= thr
        all_pass = all_pass and ok
        print(
            f"  {dim:<13} {score:.4f}  vs  {thr:.2f}  "
            f"[{'PASS' if ok else 'FAIL'}]"
        )
    print(
        f"\n  coverage detail: grounded={cov_detail.get('grounded_node_count')}"
        f" / concept_nodes={cov_detail.get('concept_node_count')}"
        f"  (asserted_edge_share={cov_detail.get('asserted_edge_share')})"
    )
    print(f"  contradiction_rate: {report.get('contradiction_rate')}")
    composite = sum(scores.values()) / len(_DIMENSIONS)
    print(f"  composite (mean)  : {composite:.4f}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 64)

    # --- ALSO run the gate-path KGQualityValidator (honest cross-check) -----
    print("\nGate-path KGQualityValidator.validate() cross-check:")
    try:
        from lib.validators.kg_quality import KGQualityValidator
        # Route the gate's own report write to a separate subdir so it does
        # NOT clobber the authoritative authoring kg_quality_report.json
        # (the gate's compute() emits the legacy edge-share coverage).
        gate_quality_dir = quality_dir / "gate_path"
        gate_quality_dir.mkdir(parents=True, exist_ok=True)
        gate = KGQualityValidator()
        gate_result = gate.validate({
            "gate_id": "kg_quality_report",
            "course_slug": args.course_slug,
            "run_id": args.run_id,
            "output_dir": str(gate_quality_dir),
            "concept_graph_path": str(concept_path),
            "semantic_graph_path": str(semantic_path),
            "min_completeness": _THRESHOLDS["completeness"],
            "min_consistency": _THRESHOLDS["consistency"],
            "min_accuracy": _THRESHOLDS["accuracy"],
            "min_coverage": _THRESHOLDS["coverage"],
        })
        print(
            f"  gate passed={gate_result.passed} "
            f"score={getattr(gate_result, 'score', None)}"
        )
        for issue in (gate_result.issues or []):
            print(f"    - [{issue.severity}] {issue.code}: {issue.message}")
        print(
            "  NOTE: the gate path's compute() now uses the SAME "
            "node-grounding coverage as compute_metrics_only (computed over "
            "the semantic graph), so the gate verdict matches the authoring "
            "reporter even when no asserted concept_graph.json exists. "
            "completeness falls back to the semantic graph's nodes when the "
            "asserted graph is absent."
        )
    except Exception as exc:  # noqa: BLE001 — cross-check is best-effort
        print(f"  WARN: gate cross-check failed: {exc}")

    print(f"\nArtifacts written under: {out_dir}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
