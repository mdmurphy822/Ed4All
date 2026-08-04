"""SFT-D B3/B4 — deterministic graph-layout normalization for a LibV2 course.

The modern ``textbook_to_course`` pipeline writes the semantic concept graph
to ``<course>/concept_graph/concept_graph_semantic.json`` (the
``_run_concept_extraction`` phase output dir) and never emits a
``pedagogy_graph.json`` at all. But the whole downstream training + retrieval
surface reads from ``<course>/graph/``:

* ``Trainforge.training.runner._compute_provenance`` treats
  ``graph/pedagogy_graph.json`` as a CRITICAL provenance artifact — its
  absence fails a training run closed.
* ``Trainforge.eval.holdout_builder.HoldoutBuilder`` loads the pedagogy graph
  from ``graph/pedagogy_graph.json`` to build the Tier-2 edge holdout.
* ``LibV2`` retrieval scoring and cross-package indexing, and
  ``rdf_export`` read ``graph/concept_graph_semantic.json``.

So a fresh pipeline course is un-trainable (missing pedagogy graph) and its
concept graph is invisible to retrieval (wrong dir). This module closes both
gaps deterministically (pure file ops + the existing CPU-only
``build_pedagogy_graph`` — NO LLM, NO GPU, NO network):

* **B4** — if ``graph/concept_graph_semantic.json`` is absent but
  ``concept_graph/concept_graph_semantic.json`` exists, copy it into
  ``graph/`` (byte-identical; the concept-extraction output is the source of
  truth).
* **B3** — if ``graph/pedagogy_graph.json`` is absent, deterministically
  re-emit it from the course's chunkset (+ objectives + the concept graph for
  the concept-class map) via ``build_pedagogy_graph``.

Idempotent: a course that already carries both artifacts is left byte-identical
(nothing is overwritten). Fail-soft per artifact: a missing chunkset means the
pedagogy graph can't be built, which is logged and skipped rather than raised —
the caller (runner / archival) decides whether the absence is fatal.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Chunkset candidates, most-canonical first (mirrors the runner's provenance
# resolution + the Phase-7c imscc_chunks rename).
_CHUNK_CANDIDATES = (
    "imscc_chunks/chunks.jsonl",
    "semantik_chunks/chunks.jsonl",
    "dart_chunks/chunks.jsonl",
    "corpus/chunks.jsonl",
)

# concept_graph_semantic.json candidates (the concept-extraction output dir
# first, then any already-in-place graph/ copy).
_SEMANTIC_GRAPH_CANDIDATES = (
    "concept_graph/concept_graph_semantic.json",
    "graph/concept_graph_semantic.json",
    "imscc_chunks/concept_graph_semantic.json",
)

# concept_graph.json supplies the optional concept-classes map consumed by
# build_pedagogy_graph.
_CONCEPT_CLASSES_CANDIDATES = (
    "graph/concept_graph.json",
    "concept_graph/concept_graph.json",
)


def _first_existing(course_dir: Path, rels: Any) -> Optional[Path]:
    for rel in rels:
        p = course_dir / rel
        if p.exists() and p.is_file():
            return p
    return None


def _load_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("graph_layout: failed to read %s (%s).", path, exc)
        return None


def _load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    chunks.append(obj)
    except OSError as exc:
        logger.warning("graph_layout: failed to read chunkset %s (%s).", path, exc)
    return chunks


def _extract_concept_classes(graph: Any) -> Optional[Dict[str, str]]:
    """Best-effort ``{slug: class_label}`` map from a concept_graph.json.

    ``build_pedagogy_graph`` accepts this to filter scaffolding endpoints out
    of ``prerequisite_of`` / ``interferes_with``. Absent → the builder runs in
    legacy permissive mode (every concept treated as DomainConcept).
    """
    if not isinstance(graph, dict):
        return None
    nodes = graph.get("nodes")
    if not isinstance(nodes, list):
        return None
    classes: Dict[str, str] = {}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        slug = node.get("id") or node.get("slug") or node.get("concept")
        label = (
            node.get("concept_class")
            or node.get("class")
            or node.get("node_class")
        )
        if isinstance(slug, str) and isinstance(label, str):
            classes[slug] = label
    return classes or None


def ensure_graph_layout(course_dir: Path) -> Dict[str, Any]:
    """Normalize ``<course_dir>/graph/`` so training + retrieval find the graphs.

    Returns a small report dict of the actions taken:
        {
          "concept_graph_copied": bool,   # B4 copy fired
          "pedagogy_graph_emitted": bool, # B3 re-emit fired
          "pedagogy_nodes": int,          # when emitted
          "pedagogy_edges": int,
          "notes": [str, ...],            # skip reasons
        }

    Idempotent + fail-soft: never overwrites an existing artifact, never
    raises on a missing chunkset / unreadable file (logs + records a note).
    """
    course_dir = Path(course_dir)
    report: Dict[str, Any] = {
        "concept_graph_copied": False,
        "pedagogy_graph_emitted": False,
        "notes": [],
    }
    graph_dir = course_dir / "graph"

    # ---- B4: concept_graph_semantic.json into graph/ --------------------- #
    graph_semantic = graph_dir / "concept_graph_semantic.json"
    if not graph_semantic.exists():
        src = _first_existing(course_dir, _SEMANTIC_GRAPH_CANDIDATES)
        if src is not None and src != graph_semantic:
            try:
                graph_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, graph_semantic)
                report["concept_graph_copied"] = True
                logger.info(
                    "graph_layout: B4 copied concept graph %s -> %s.",
                    src, graph_semantic,
                )
            except OSError as exc:
                report["notes"].append(f"concept_graph_copy_failed: {exc}")
                logger.warning(
                    "graph_layout: B4 copy %s -> %s failed (%s).",
                    src, graph_semantic, exc,
                )
        else:
            report["notes"].append("concept_graph_semantic_source_absent")

    # ---- B3: deterministically (re-)emit pedagogy_graph.json ------------- #
    pedagogy_graph = graph_dir / "pedagogy_graph.json"
    if pedagogy_graph.exists():
        report["notes"].append("pedagogy_graph_present")
        return report

    chunk_path = _first_existing(course_dir, _CHUNK_CANDIDATES)
    if chunk_path is None:
        report["notes"].append("pedagogy_graph_skipped_no_chunkset")
        logger.warning(
            "graph_layout: B3 cannot re-emit pedagogy_graph.json for %s — no "
            "chunkset found (%s).",
            course_dir, list(_CHUNK_CANDIDATES),
        )
        return report

    chunks = _load_chunks(chunk_path)
    if not chunks:
        report["notes"].append("pedagogy_graph_skipped_empty_chunkset")
        return report

    # Objectives + concept-class map are optional inputs (the builder degrades
    # gracefully without them).
    objectives: Optional[Dict[str, Any]] = None
    obj_path = _first_existing(course_dir, ("objectives.json", "course.json"))
    if obj_path is not None:
        loaded = _load_json(obj_path)
        if isinstance(loaded, dict):
            objectives = loaded

    concept_classes: Optional[Dict[str, str]] = None
    cc_path = _first_existing(course_dir, _CONCEPT_CLASSES_CANDIDATES)
    if cc_path is not None:
        concept_classes = _extract_concept_classes(_load_json(cc_path))

    try:
        from Trainforge.pedagogy_graph_builder import build_pedagogy_graph
    except Exception as exc:  # noqa: BLE001 — import-failure guard
        report["notes"].append(f"pedagogy_builder_import_failed: {exc}")
        logger.warning(
            "graph_layout: B3 pedagogy_graph_builder import failed (%s); "
            "skipping re-emit.", exc,
        )
        return report

    # Derive a stable course_id from the course dir name so the builder's
    # chunk-ID fallback isn't the only anchor.
    course_id = course_dir.name

    try:
        graph = build_pedagogy_graph(
            chunks,
            objectives=objectives,
            course_id=course_id,
            concept_classes=concept_classes,
        )
    except Exception as exc:  # noqa: BLE001 — builder is fail-soft here
        report["notes"].append(f"pedagogy_build_failed: {exc}")
        logger.warning(
            "graph_layout: B3 build_pedagogy_graph raised (%s); skipping "
            "re-emit for %s.", exc, course_dir,
        )
        return report

    if not isinstance(graph, dict):
        report["notes"].append("pedagogy_build_non_dict")
        return report

    try:
        graph_dir.mkdir(parents=True, exist_ok=True)
        tmp = pedagogy_graph.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(graph, indent=2, sort_keys=True), encoding="utf-8"
        )
        tmp.replace(pedagogy_graph)
    except OSError as exc:
        report["notes"].append(f"pedagogy_write_failed: {exc}")
        logger.warning(
            "graph_layout: B3 failed to write %s (%s).", pedagogy_graph, exc,
        )
        return report

    report["pedagogy_graph_emitted"] = True
    report["pedagogy_nodes"] = len(graph.get("nodes") or [])
    report["pedagogy_edges"] = len(graph.get("edges") or [])
    logger.info(
        "graph_layout: B3 re-emitted %s (nodes=%d, edges=%d) from %s.",
        pedagogy_graph, report["pedagogy_nodes"], report["pedagogy_edges"],
        chunk_path,
    )
    return report


__all__ = ["ensure_graph_layout"]
