#!/usr/bin/env python3
"""Wave 76 retroactive regeneration: re-prune ``pedagogy_graph.json``.

Wave 75 Worker C's regenerator (``wave75_regen_pedagogy.py``) replaced
the stub graph with a real one but the ``prerequisite_of`` rule
over-saturated: 7032 of 8324 edges (84%) on the rdf-shacl-550 archive
came from a temporal-cooccurrence cartesian. Wave 76 Worker D refined
the rule to require strictly-later-week + at-least-one-shared-chunk +
DomainConcept-classified endpoints (Worker B's classifier). Same
DomainConcept filter applies to ``interferes_with`` so misconceptions
no longer link to pedagogical scaffolding ("key-takeaway", "rubric",
"assessment", etc.).

This script:

* Loads ``corpus/chunks.jsonl``.
* Loads objectives (Worker A's ``objectives.json`` if present, else
  Courseforge ``synthesized_objectives.json``).
* Loads ``graph/concept_graph.json`` and extracts the per-concept
  ``class`` map (Worker B's classifier output).
* Rebuilds ``pedagogy_graph.json`` with the new rules.
* Backs the prior graph up as ``pedagogy_graph.json.bak`` (refuses to
  overwrite an existing .bak unless ``--force-bak`` is passed).
* Reports edge counts per relation_type (before vs after).

Default target: rdf-shacl-550 (the archive cited in the external
review). Use ``--archive`` to point elsewhere.

Usage::

    python scripts/wave76_regen_pedagogy.py
    python scripts/wave76_regen_pedagogy.py --archive LibV2/courses/<other>
    python scripts/wave76_regen_pedagogy.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from Trainforge.pedagogy_graph_builder import (  # noqa: E402
    build_pedagogy_graph,
    load_objectives_with_fallback,
)


DEFAULT_ARCHIVE = (
    REPO_ROOT / "LibV2" / "courses" / "rdf-shacl-550-rdf-shacl-550"
)
DEFAULT_SYNTH = (
    REPO_ROOT
    / "Courseforge"
    / "exports"
    / "PROJ-RDF_SHACL_550-20260424135037"
    / "01_learning_objectives"
    / "synthesized_objectives.json"
)


def _read_chunks(path: Path) -> list:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _resolve_objectives_path(archive: Path) -> Path:
    """Prefer Worker A's objectives.json inside the archive; else None."""
    cand = archive / "objectives.json"
    if cand.exists():
        return cand
    cand = archive / "pedagogy" / "objectives.json"
    if cand.exists():
        return cand
    return Path()  # falsy


def _load_concept_classes(concept_graph_path: Path) -> Dict[str, str]:
    """Extract ``slug -> class`` from concept_graph.json's nodes.

    Worker B's classifier annotates every concept node with a ``class``
    field (DomainConcept / PedagogicalMarker / AssessmentOption /
    LowSignal / InstructionalArtifact / Misconception). The slug is the
    node ``id`` (kebab-case). Missing class fields are skipped — the
    pedagogy builder treats unclassified concepts as DomainConcept by
    default for backwards compatibility.
    """
    if not concept_graph_path.exists():
        return {}
    with open(concept_graph_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: Dict[str, str] = {}
    for n in data.get("nodes", []) or []:
        if not isinstance(n, dict):
            continue
        slug = n.get("id")
        cls = n.get("class")
        if isinstance(slug, str) and isinstance(cls, str):
            out[slug] = cls
    return out


def _pretty_summary(graph: Dict[str, Any]) -> str:
    s = graph.get("stats") or {}
    nbc = s.get("nodes_by_class") or {}
    ebr = s.get("edges_by_relation") or {}
    lines = [
        f"  total nodes: {s.get('node_count', 0)}",
        f"  total edges: {s.get('edge_count', 0)}",
        "  nodes by class:",
    ]
    for cls in sorted(nbc):
        lines.append(f"    {cls:<22} {nbc[cls]}")
    lines.append("  edges by relation_type:")
    for rel in sorted(ebr):
        lines.append(f"    {rel:<22} {ebr[rel]}")
    return "\n".join(lines)


def _diff_summary(before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """Print before/after delta per relation type."""
    b_ebr = (before.get("stats") or {}).get("edges_by_relation") or {}
    a_ebr = (after.get("stats") or {}).get("edges_by_relation") or {}
    keys = sorted(set(b_ebr.keys()) | set(a_ebr.keys()))
    lines = [
        "  Edge counts (before -> after):",
        f"    {'relation_type':<22} {'before':>8}  {'after':>8}  {'delta':>8}",
    ]
    for k in keys:
        b = b_ebr.get(k, 0)
        a = a_ebr.get(k, 0)
        lines.append(f"    {k:<22} {b:>8}  {a:>8}  {a - b:>+8}")
    b_total = (before.get("stats") or {}).get("edge_count", 0)
    a_total = (after.get("stats") or {}).get("edge_count", 0)
    lines.append(
        f"    {'TOTAL':<22} {b_total:>8}  {a_total:>8}  {a_total - b_total:>+8}"
    )
    return "\n".join(lines)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"Course archive root (default: {DEFAULT_ARCHIVE})",
    )
    parser.add_argument(
        "--synthesized",
        type=Path,
        default=DEFAULT_SYNTH,
        help=(
            "Fallback synthesized_objectives.json path used when "
            "<archive>/objectives.json is absent."
        ),
    )
    parser.add_argument(
        "--course-id",
        type=str,
        default=None,
        help="Course id attribute (default: read from archive/manifest.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary; do not write/overwrite any file.",
    )
    parser.add_argument(
        "--force-bak",
        action="store_true",
        help=(
            "Overwrite an existing pedagogy_graph.json.bak. By default "
            "the script aborts when a .bak is already present so the "
            "Wave 75 stub graph is never lost."
        ),
    )
    args = parser.parse_args(argv)

    archive: Path = args.archive
    if not archive.exists():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        return 1

    chunks_path = archive / "corpus" / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"ERROR: chunks.jsonl missing: {chunks_path}", file=sys.stderr)
        return 1

    graph_dir = archive / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)
    pedagogy_path = graph_dir / "pedagogy_graph.json"
    bak_path = graph_dir / "pedagogy_graph.json.bak"
    concept_graph_path = graph_dir / "concept_graph.json"

    # Resolve course_id from manifest if available.
    course_id = args.course_id
    if not course_id:
        manifest_path = archive / "manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                course_id = (
                    manifest.get("course_code")
                    or manifest.get("course_id")
                    or ""
                )
            except (OSError, json.JSONDecodeError):
                course_id = ""

    # Resolve objectives.
    objectives_path = _resolve_objectives_path(archive)
    objectives = load_objectives_with_fallback(
        str(objectives_path) if objectives_path else None,
        str(args.synthesized) if args.synthesized else None,
    )
    if not objectives:
        print(
            "WARNING: no objectives found (neither archive/objectives.json "
            "nor synthesized fallback). Pedagogy graph will lack TO/CO nodes.",
            file=sys.stderr,
        )

    # Wave 76: load Worker B's classifier output.
    concept_classes = _load_concept_classes(concept_graph_path)
    if concept_classes:
        cls_counts: Dict[str, int] = {}
        for c in concept_classes.values():
            cls_counts[c] = cls_counts.get(c, 0) + 1
        print(
            f"Loaded concept classes from {concept_graph_path} "
            f"({len(concept_classes)} concepts: {cls_counts})"
        )
    else:
        print(
            f"WARNING: no concept classes loaded from {concept_graph_path}. "
            "Pedagogy graph will skip the DomainConcept filter "
            "(legacy permissive mode).",
            file=sys.stderr,
        )

    chunks = _read_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    if objectives_path:
        print(f"Loaded objectives from {objectives_path}")
    elif args.synthesized:
        print(f"Loaded objectives (fallback) from {args.synthesized}")

    # Read prior graph (for diff reporting). Tolerate absence.
    before_graph: Dict[str, Any] = {}
    if pedagogy_path.exists():
        try:
            with open(pedagogy_path, "r", encoding="utf-8") as f:
                before_graph = json.load(f)
        except (OSError, json.JSONDecodeError):
            before_graph = {}

    graph = build_pedagogy_graph(
        chunks,
        objectives,
        course_id=course_id or None,
        concept_classes=concept_classes,
    )

    print()
    print("New pedagogy graph summary:")
    print(_pretty_summary(graph))
    print()

    if before_graph:
        print("Edge-count delta:")
        print(_diff_summary(before_graph, graph))
        print()

    if args.dry_run:
        print("(--dry-run; no files written)")
        return 0

    # Backup existing pedagogy_graph.json.
    if pedagogy_path.exists():
        if bak_path.exists() and not args.force_bak:
            print(
                f"ERROR: backup already exists at {bak_path}. "
                "Re-run with --force-bak to overwrite, or manually move it.",
                file=sys.stderr,
            )
            return 2
        shutil.copy2(pedagogy_path, bak_path)
        print(f"Backed up existing pedagogy_graph.json -> {bak_path}")

    with open(pedagogy_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False, sort_keys=False)
        f.write("\n")
    print(f"Wrote new pedagogy_graph.json -> {pedagogy_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
