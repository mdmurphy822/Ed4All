#!/usr/bin/env python3
"""Stage-3 (7B-synthesized) concept-knowledge-graph builder + KG-quality reporter.

Standalone, reproducible reconstruction of the PRODUCTION
``MCP/tools/pipeline_tools.py::_run_concept_extraction`` Stage-3 path —
the branch that fires when ``TEXTBOOK_SYNTHESIS_PROVIDER`` is set. Unlike
the sibling ``build_kg_local.py`` (which is fully deterministic and trusts
the chunks' EXISTING ``concept_tags``), this runner dispatches the live
per-chapter ``TextbookSynthesisProvider.synthesize_concepts`` LLM calls
over ``textbook_structure.json::chapters[].chapter_text``, builds a
``domain_concept_vocabulary.json``, compiles ``domain_concept_seeds``, and
RE-TAGS the loaded chunks IN-MEMORY before the co-occurrence / semantic
graph build — so a corpus whose chunks carry only a handful of distinct
``concept_tags`` (e.g. a real full-book course: 2 distinct tags → coverage 0.133,
degenerate 2-node graph) gets real ``DomainConcept`` nodes and clears the
chunk-anchored coverage floor.

This is the production fix for the real-corpus 0-node failure, exercised
standalone so it can be driven directly against an existing course's
on-disk artifacts.

The build path mirrors ``_run_concept_extraction`` faithfully and REUSES
the same production helpers (no divergent re-implementation):

    1. load chunks (jsonl).
    2. Stage-3 concept synthesis (THIS runner's reason to exist):
       (a) read ``textbook_structure.json``, pull ``chapters[]`` with
           ``chapter_text``;
       (b) N per-chapter ``provider.synthesize_concepts`` calls, batched
           ≤10 via ``provider.batch_chapters`` (per-chapter failure
           isolation — one bad chapter degrades, doesn't abort);
       (c) merge per-chapter concepts → course vocabulary, de-dup on
           ``canonical_slug``;
       (d) compile via ``Trainforge.process_course.compile_domain_concept_seeds``
           (with the same de-slug surface-form alias the handler appends);
       (e) re-tag each chunk via
           ``lib.ontology.concept_tagging.extract_concept_tags(..., domain_concept_seeds=seeds)``,
           UNION into ``chunk["concept_tags"]`` (in-memory only — the
           on-disk chunks.jsonl is NOT rewritten).
    3. normalize objectives via ``_normalize_objectives_payload_to_course``.
    4. set the measured-best graph-shaping env defaults via setdefault.
    5. ``build_cooccurrence_graph`` (group_by from env) → concept-objective
       linker → ``build_semantic_graph`` → merge → intra-chunk links →
       related-to fan-out cap (each env-gated, matching the handler).
    6. ``EdgeConsensusAggregator.apply_to_graph`` (stamp edge_status).
    7. ``KGQualityReporter.compute_metrics_only`` (the SAME authoring-time
       entry point the handler uses to populate kg_quality).

Usage:
    TEXTBOOK_SYNTHESIS_PROVIDER=local \
    TEXTBOOK_SYNTHESIS_MODEL=qwen2.5-7b-8k \
    LOCAL_SYNTHESIS_BASE_URL=http://localhost:11434/v1 \
    python -m scripts.integration.build_kg_stage3_local \
        --chunks LibV2/courses/<slug>/semantik_chunks/chunks.jsonl \
        --course-slug <slug> \
        --objectives <project>/01_learning_objectives/synthesized_objectives.json \
        --textbook-structure <project>/.../textbook_structure.json \
        --out-dir /tmp/kg-stage3-verify

Exit 0 when every KG-quality dimension clears its threshold; 1 otherwise.
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
# MCP/core/workflow_runner.py::_apply_corpus_generalization_defaults's
# graph-shaping set. We DELIBERATELY do NOT default
# TEXTBOOK_SYNTHESIS_PROVIDER here: that env is what selects the live 7B
# Stage-3 path, and the operator must opt in explicitly (no silent LLM
# dispatch). When it's unset the runner fails loud (see main()).
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
    # Lexical fallback ON so a degraded-Stage-3 run (all chapters fail / no
    # concepts) still recovers a real DomainConcept node set instead of a
    # degenerate graph — exactly the auto-on behavior of a pipeline run.
    # No-op when Stage-3 produced seeds.
    "TRAINFORGE_LEXICAL_CONCEPT_SEEDS": "true",
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


def run_stage3_concept_synthesis(
    *,
    chunks: List[Dict[str, Any]],
    textbook_structure_path: Path,
    course_name: str,
    course_slug: str,
    provider_factory: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """Stage-3 concept synthesis + in-memory re-tag.

    FAITHFUL standalone mirror of
    ``_run_concept_extraction._run_stage3_concept_synthesis``. Reuses the
    production ``TextbookSynthesisProvider``, ``compile_domain_concept_seeds``,
    and ``extract_concept_tags`` helpers; re-tags ``chunks`` IN PLACE.

    Args:
        chunks: loaded SemantiK chunks (mutated in place — concept_tags UNIONed).
        textbook_structure_path: path to textbook_structure.json (chapter
            text source).
        course_name: canonical course name (drives the capture rationale).
        course_slug: course slug (stamped onto the vocabulary artifact).
        provider_factory: TEST SEAM. A zero-arg callable returning an object
            exposing ``synthesize_concepts(chapter, course_name=...)`` +
            ``batch_chapters(chapters)``. Default ``None`` constructs the
            real ``TextbookSynthesisProvider`` (live LLM dispatch).

    Returns:
        The vocabulary dict (also carries ``_chunks_retagged``), or ``None``
        when no readable textbook structure / no chapters are available.
    """
    # --- read textbook_structure.json --------------------------------------
    if not textbook_structure_path.is_file():
        print(
            f"  WARN: textbook_structure not found ({textbook_structure_path}); "
            "skipping Stage 3.",
            file=sys.stderr,
        )
        return None
    try:
        textbook_structure = json.loads(
            textbook_structure_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        print(
            f"  WARN: failed to parse textbook_structure ({exc}); skipping "
            "Stage 3.",
            file=sys.stderr,
        )
        return None
    chapters: List[Dict[str, Any]] = []
    if isinstance(textbook_structure, dict):
        raw_chapters = textbook_structure.get("chapters")
        if isinstance(raw_chapters, list):
            chapters = [c for c in raw_chapters if isinstance(c, dict)]
    if not chapters:
        print(
            "  WARN: textbook_structure carries no chapters[]; skipping Stage 3.",
            file=sys.stderr,
        )
        return None

    # --- construct the provider --------------------------------------------
    from Courseforge.generators.outline._textbook_synthesis_provider import (
        TextbookSynthesisProviderError,
    )

    if provider_factory is not None:
        provider = provider_factory()
    else:
        from Courseforge.generators.outline._textbook_synthesis_provider import (
            TextbookSynthesisProvider,
        )
        provider = TextbookSynthesisProvider(capture=None)

    # --- N per-chapter calls, batched ≤10 (per-chapter failure isolation) --
    per_chapter_concepts: List[Dict[str, Any]] = []
    chapter_synthesis_failures: List[str] = []
    chapters_synthesized = 0

    batches = provider.batch_chapters(chapters)
    for batch in batches:
        for chapter in batch:
            cid = str(chapter.get("id") or "")
            try:
                res = provider.synthesize_concepts(
                    chapter, course_name=course_name
                )
            except TextbookSynthesisProviderError as exc:
                print(
                    f"  WARN: Stage-3 chapter {cid!r} exhausted ({exc}); "
                    "degrading per-chapter.",
                    file=sys.stderr,
                )
                chapter_synthesis_failures.append(cid)
                continue
            except Exception as exc:  # noqa: BLE001 — isolate any raise
                print(
                    f"  WARN: Stage-3 chapter {cid!r} raised ({exc}); "
                    "degrading per-chapter.",
                    file=sys.stderr,
                )
                chapter_synthesis_failures.append(cid)
                continue
            if res is None:
                chapter_synthesis_failures.append(cid)
                continue
            chapters_synthesized += 1
            for concept in res.get("concepts") or []:
                if isinstance(concept, dict):
                    per_chapter_concepts.append(concept)

    # --- merge + de-dup on canonical_slug ----------------------------------
    from lib.ontology.slugs import canonical_slug as _canonical_slug

    merged: Dict[str, Dict[str, Any]] = {}
    for concept in per_chapter_concepts:
        canonical_raw = str(concept.get("canonical") or "").strip()
        if not canonical_raw:
            continue
        slug = _canonical_slug(canonical_raw)
        if not slug:
            continue
        aliases = [
            str(a).strip()
            for a in (concept.get("aliases") or [])
            if isinstance(a, str) and str(a).strip()
        ]
        # Preserve the LLM's raw canonical surface form as an alias before
        # the slug overwrites it (so the compiled seed also matches the
        # natural-language form the model emitted in prose).
        if canonical_raw and canonical_raw not in aliases:
            aliases.append(canonical_raw)
        chapter_ids = [
            str(c)
            for c in (concept.get("chapter_ids") or [])
            if isinstance(c, (str, int))
        ]
        hint = str(concept.get("definition_hint") or "").strip()
        if slug in merged:
            existing = merged[slug]
            for a in aliases:
                if a not in existing["aliases"]:
                    existing["aliases"].append(a)
            for c in chapter_ids:
                if c not in existing["chapter_ids"]:
                    existing["chapter_ids"].append(c)
            if not existing.get("definition_hint") and hint:
                existing["definition_hint"] = hint
        else:
            merged[slug] = {
                "canonical": slug,
                "aliases": aliases,
                "chapter_ids": chapter_ids,
                "definition_hint": hint,
            }

    concepts_out = list(merged.values())
    vocabulary: Dict[str, Any] = {
        "schema_version": "v1",
        "course_id": course_name.upper(),
        "course_slug": course_slug,
        "provider": getattr(provider, "_provider", ""),
        "model": getattr(provider, "_model", "") or "",
        "chapter_call_count": len(chapters),
        "chapter_synthesis_failures": chapter_synthesis_failures,
        "concept_count": len(concepts_out),
        "concepts": concepts_out,
    }

    # --- all-fail → empty-seed fallback ------------------------------------
    if chapters_synthesized == 0 or not concepts_out:
        print(
            f"  WARN: Stage-3 produced no concepts "
            f"(chapters_synthesized={chapters_synthesized}, "
            f"concepts={len(concepts_out)}); falling back to empty seeds.",
            file=sys.stderr,
        )
        vocabulary["_chunks_retagged"] = 0
        return vocabulary

    # --- compile into (canonical, [regex]) seed pairs ----------------------
    from Trainforge.process_course import compile_domain_concept_seeds

    _stage3_seed_specs: List[Dict[str, Any]] = []
    for c in concepts_out:
        _cid = c["canonical"]
        _aliases = list(c.get("aliases") or [])
        _deslug = _cid.replace("-", " ")
        if _deslug != _cid and _deslug not in _aliases:
            _aliases.append(_deslug)
        _stage3_seed_specs.append({"id": _cid, "aliases": _aliases})
    seeds = compile_domain_concept_seeds(_stage3_seed_specs)

    # --- re-tag each loaded chunk in-memory (UNION) ------------------------
    from lib.ontology.concept_tagging import extract_concept_tags

    retagged = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        text = str(chunk.get("text") or "")
        if not text:
            continue
        try:
            new_tags = extract_concept_tags(
                text, chunk, domain_concept_seeds=seeds
            )
        except Exception as exc:  # noqa: BLE001 — isolate per chunk
            print(
                f"  WARN: Stage-3 re-tag raised on chunk {chunk.get('id')!r} "
                f"({exc}); leaving its tags untouched.",
                file=sys.stderr,
            )
            continue
        existing = chunk.get("concept_tags")
        existing = existing if isinstance(existing, list) else []
        union = list(existing)
        for tag in new_tags:
            if tag not in union:
                union.append(tag)
        if union != existing:
            retagged += 1
        chunk["concept_tags"] = union

    vocabulary["_chunks_retagged"] = retagged
    return vocabulary


def _derive_misconceptions(
    chunks: List[Dict[str, Any]], course_code: str
) -> List[Dict[str, Any]]:
    """Mirror _run_concept_extraction._derive_misconceptions."""
    try:
        from Trainforge.rag.typed_edge_inference import _make_concept_id
        from Trainforge.process_course import _route_misconception_to_tag
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

    Consumes the (already Stage-3-retagged) ``chunks``. Returns the graph
    dict (post edge-consensus stamping + kg_quality metrics-only).
    """
    from lib.ontology.cooccurrence_graph import build_cooccurrence_graph
    from Trainforge.rag.typed_edge_inference import build_semantic_graph

    course_code = course_slug.upper().replace("-", "_")

    # --- normalize objectives ----------------------------------------------
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

    # --- co-occurrence graph (page-grouped by env) -------------------------
    group_by = (
        os.environ.get("TRAINFORGE_COOCCURRENCE_GROUP_BY", "").strip() or "chunk"
    )
    cooccurrence_graph = build_cooccurrence_graph(
        chunks, course_code, graph_kind="concept", group_by=group_by,
    )
    print(
        f"  cooccurrence graph (group_by={group_by}): "
        f"{len(cooccurrence_graph.get('nodes', []))} nodes, "
        f"{len(cooccurrence_graph.get('edges', []))} edges"
    )

    # --- concept-objective linker ------------------------------------------
    if course_for_graph is not None:
        try:
            from lib.ontology.concept_objective_linker import (
                link_concepts_to_objectives,
            )
            from MCP.tools.pipeline_tools import (
                _normalize_objectives_payload_to_course as _renorm,
            )
            enriched_los = link_concepts_to_objectives(
                course_for_graph.get("learning_outcomes") or [],
                cooccurrence_graph,
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

    # --- typed-edge semantic graph -----------------------------------------
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

    # --- KG-quality post-build passes (env-gated, order load-bearing) ------
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
            "Build a course concept knowledge graph with 7B-synthesized "
            "domain concepts (Stage-3 path), re-tag chunks in-memory, then "
            "run the KG-quality reporter/validator."
        )
    )
    parser.add_argument(
        "--chunks", required=True, help="Path to semantik_chunks/chunks.jsonl",
    )
    parser.add_argument(
        "--course-slug", required=True, help="Course slug (e.g. my-course-101)",
    )
    parser.add_argument(
        "--objectives", default=None,
        help=(
            "Optional path to synthesized_objectives.json (Courseforge "
            "synthesized form or LibV2 archive form)"
        ),
    )
    parser.add_argument(
        "--textbook-structure", required=True,
        help=(
            "Path to textbook_structure.json — the chapters[].chapter_text "
            "source for the Stage-3 synthesize_concepts calls"
        ),
    )
    parser.add_argument(
        "--course-name", default=None,
        help=(
            "Canonical course name (defaults to the upper-cased, "
            "underscore-normalized slug)"
        ),
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Output dir; writes concept_graph/ + quality/ beneath it",
    )
    parser.add_argument(
        "--run-id", default="kg-stage3-local-build",
        help="Run identifier stamped on nodes/edges",
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
    ts_path = Path(args.textbook_structure).resolve()
    if not ts_path.is_file():
        print(
            f"ERROR: textbook_structure file not found: {ts_path}",
            file=sys.stderr,
        )
        return 1

    # The Stage-3 path is what this runner is FOR — fail loud if the
    # operator forgot to select a synthesis provider (no silent fall-through
    # to the deterministic existing-tags build; that's build_kg_local.py).
    if not os.environ.get("TEXTBOOK_SYNTHESIS_PROVIDER", "").strip():
        print(
            "ERROR: TEXTBOOK_SYNTHESIS_PROVIDER is not set. This runner "
            "dispatches the live Stage-3 synthesize_concepts LLM calls; set "
            "TEXTBOOK_SYNTHESIS_PROVIDER (e.g. =local) + TEXTBOOK_SYNTHESIS_MODEL "
            "+ LOCAL_SYNTHESIS_BASE_URL. For a deterministic build from the "
            "chunks' EXISTING concept_tags, use build_kg_local.py instead.",
            file=sys.stderr,
        )
        return 1

    # Measured-best graph-shaping defaults (setdefault — operator-pinned wins).
    for env_var, value in _GRAPH_SHAPING_ENV_DEFAULTS.items():
        os.environ.setdefault(env_var, value)

    course_slug = args.course_slug
    course_name = (
        args.course_name
        or course_slug.upper().replace("-", "_")
    )

    print(f"Course slug : {course_slug}")
    print(f"Course name : {course_name}")
    print(f"Chunks      : {chunks_path}")
    print(f"Textbook    : {ts_path}")
    print(
        f"Provider    : {os.environ.get('TEXTBOOK_SYNTHESIS_PROVIDER')} "
        f"(model={os.environ.get('TEXTBOOK_SYNTHESIS_MODEL', '(default)')})"
    )

    chunks = _load_chunks(chunks_path)
    if not chunks:
        print("ERROR: no chunks loaded", file=sys.stderr)
        return 1
    tags_before = {t for c in chunks for t in (c.get("concept_tags") or [])}
    print(
        f"  loaded {len(chunks)} chunks; "
        f"{len(tags_before)} distinct concept_tags BEFORE Stage-3"
    )

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

    # --- Stage-3 concept synthesis + in-memory re-tag ----------------------
    print("\nStage-3 concept synthesis (live 7B per-chapter calls) ...")
    vocabulary = run_stage3_concept_synthesis(
        chunks=chunks,
        textbook_structure_path=ts_path,
        course_name=course_name,
        course_slug=course_slug,
    )
    synthesized_concept_count = 0
    if vocabulary is not None:
        synthesized_concept_count = int(vocabulary.get("concept_count", 0) or 0)
        print(
            f"  Stage-3 vocabulary: {synthesized_concept_count} concepts, "
            f"{vocabulary.get('_chunks_retagged', 0)} chunks re-tagged, "
            f"{len(vocabulary.get('chapter_synthesis_failures') or [])} "
            f"chapter failures"
        )
    else:
        print("  Stage-3 produced no vocabulary (no chapters / unreadable).")
    tags_after = {t for c in chunks for t in (c.get("concept_tags") or [])}
    print(
        f"  distinct concept_tags AFTER Stage-3: {len(tags_after)} "
        f"(+{len(tags_after) - len(tags_before)})"
    )

    # --- build the graph ---------------------------------------------------
    print("\nBuilding semantic concept graph ...")
    graph = build_kg(
        chunks=chunks,
        course_slug=course_slug,
        objectives_payload=objectives_payload,
        run_id=args.run_id,
    )
    nodes = graph.get("nodes", []) if isinstance(graph, dict) else []
    edges = graph.get("edges", []) if isinstance(graph, dict) else []
    print(f"  built graph: {len(nodes)} nodes, {len(edges)} edges")
    if not edges:
        print("ERROR: empty graph (no edges) — cannot score", file=sys.stderr)
        return 1

    # --- edge-consensus stamping (apply_to_graph, in place) ----------------
    from lib.aggregators.edge_consensus import EdgeConsensusAggregator
    try:
        EdgeConsensusAggregator(
            semantic_graph_path=None,
            course_slug=course_slug,
            run_id=args.run_id,
        ).apply_to_graph(graph)
    except Exception as exc:  # noqa: BLE001 — fail-soft (matches handler)
        print(f"  WARN: edge-consensus stamping failed: {exc}")

    # --- KG-quality (authoring-time entry point) ---------------------------
    from Trainforge.rag.kg_quality_report import KGQualityReporter

    out_dir = Path(args.out_dir).resolve()
    graph_dir = out_dir / "concept_graph"
    quality_dir = out_dir / "quality"
    graph_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    reporter = KGQualityReporter(
        course_slug=course_slug,
        run_id=args.run_id,
        output_dir=quality_dir,
    )
    report = reporter.compute_metrics_only(graph)

    # Stamp the compact four-dimension score block onto the graph.
    _stamp_dims = report.get("dimensions") or {}
    graph["kg_quality"] = {
        dim: float((_stamp_dims.get(dim) or {}).get("score", 0.0))
        for dim in _DIMENSIONS
    }

    # --- write artifacts ---------------------------------------------------
    semantic_path = graph_dir / "concept_graph_semantic.json"
    semantic_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    concept_path = graph_dir / "concept_graph.json"  # gate-path sibling (absent)

    # domain_concept_vocabulary.json — sibling of the semantic graph (matches
    # the production handler's persistence location).
    if vocabulary is not None:
        (graph_dir / "domain_concept_vocabulary.json").write_text(
            json.dumps(vocabulary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    report_path = quality_dir / "kg_quality_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Sibling edge-consensus report.
    try:
        EdgeConsensusAggregator(
            semantic_graph_path=semantic_path,
            course_slug=course_slug,
            run_id=args.run_id,
        ).write(graph_dir / "edge_consensus_report.json")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARN: edge_consensus_report.json write failed: {exc}")

    # --- print dimension scores --------------------------------------------
    dims = report.get("dimensions", {})
    scores: Dict[str, float] = {
        dim: float((dims.get(dim) or {}).get("score", 0.0))
        for dim in _DIMENSIONS
    }
    cov_detail = dims.get("coverage", {})

    # Count how many of the 7B-synthesized concepts became grounded nodes.
    grounded_synth_concepts = _count_grounded_synthesized_concepts(
        graph, vocabulary
    )

    print("\n" + "=" * 64)
    print(f"KG-quality (compute_metrics_only) — course={course_slug}")
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
    print(
        f"  7B-synthesized concepts that became grounded nodes: "
        f"{grounded_synth_concepts}/{synthesized_concept_count}"
    )
    composite = sum(scores.values()) / len(_DIMENSIONS)
    print(f"  composite (mean)  : {composite:.4f}")
    print(f"\n  OVERALL: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 64)

    # --- gate-path KGQualityValidator cross-check --------------------------
    print("\nGate-path KGQualityValidator.validate() cross-check:")
    try:
        from lib.validators.kg_quality import KGQualityValidator
        gate_quality_dir = quality_dir / "gate_path"
        gate_quality_dir.mkdir(parents=True, exist_ok=True)
        gate = KGQualityValidator()
        gate_result = gate.validate({
            "gate_id": "kg_quality_report",
            "course_slug": course_slug,
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
    except Exception as exc:  # noqa: BLE001 — cross-check best-effort
        print(f"  WARN: gate cross-check failed: {exc}")

    print(f"\nArtifacts written under: {out_dir}")
    return 0 if all_pass else 1


def _count_grounded_synthesized_concepts(
    graph: Dict[str, Any],
    vocabulary: Optional[Dict[str, Any]],
) -> int:
    """Count Stage-3 concept slugs that became chunk-grounded graph nodes.

    A synthesized concept is "grounded" when a DomainConcept node whose id
    or label slug matches the concept's canonical slug carries ≥1 chunk
    occurrence (frequency>0 or occurrences[]). Best-effort/diagnostic.
    """
    if not vocabulary:
        return 0
    from lib.ontology.slugs import canonical_slug

    synth_slugs = {
        canonical_slug(str(c.get("canonical") or ""))
        for c in (vocabulary.get("concepts") or [])
        if str(c.get("canonical") or "").strip()
    }
    synth_slugs.discard("")
    if not synth_slugs:
        return 0

    grounded: set = set()
    for node in graph.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        # node id often "<COURSE>:slug" or just slug; also check label slug.
        candidates: set = set()
        nid = str(node.get("id") or "")
        if nid:
            candidates.add(canonical_slug(nid.split(":")[-1]))
        label = str(node.get("label") or "")
        if label:
            candidates.add(canonical_slug(label))
        match = candidates & synth_slugs
        if not match:
            continue
        freq = node.get("frequency")
        occ = node.get("occurrences")
        is_grounded = (
            (isinstance(freq, (int, float)) and freq > 0)
            or (isinstance(occ, list) and len(occ) > 0)
        )
        if is_grounded:
            grounded |= match
    return len(grounded)


if __name__ == "__main__":
    raise SystemExit(main())
