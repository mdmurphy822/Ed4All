#!/usr/bin/env python3
"""Wave 75 retroactive regeneration: rebuild pedagogy_graph.json.

Walks an existing LibV2 course archive's ``corpus/chunks.jsonl`` plus
its planned objectives (Worker A's ``objectives.json`` if present;
else Courseforge's ``synthesized_objectives.json``) and regenerates a
real pedagogical graph in place. The original ``pedagogy_graph.json``
is preserved as ``pedagogy_graph.json.bak`` (never overwritten — the
script aborts if a ``.bak`` already exists for safety, unless
``--force-bak`` is passed).

Default target: the rdf-shacl-550 archive identified in the Wave 75
ChatGPT review as broken (1 node / 0 edges). Use ``--archive`` to
point at any LibV2 course directory.

Usage::

    python scripts/wave75_regen_pedagogy.py
    python scripts/wave75_regen_pedagogy.py --archive LibV2/courses/<other>
    python scripts/wave75_regen_pedagogy.py --dry-run    # print only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Optional

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


def _pretty_summary(graph: dict) -> str:
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
            "the script aborts when a .bak is already present so prior "
            "stub graphs are never lost."
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

    chunks = _read_chunks(chunks_path)
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")
    if objectives_path:
        print(f"Loaded objectives from {objectives_path}")
    elif args.synthesized:
        print(f"Loaded objectives (fallback) from {args.synthesized}")

    graph = build_pedagogy_graph(
        chunks,
        objectives,
        course_id=course_id or None,
    )

    print()
    print("Pedagogy graph summary:")
    print(_pretty_summary(graph))
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
