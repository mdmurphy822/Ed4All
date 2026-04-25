#!/usr/bin/env python3
"""Wave 76: retroactively prune polluted concept-graph nodes.

Reads existing ``concept_graph.json`` + ``concept_graph_semantic.json``
under a course's ``graph/`` directory, re-classifies every node with the
Wave 76 expanded rule set, and DROPS nodes whose class is in
:data:`lib.ontology.concept_classifier.DROPPABLE_CLASSES` (pedagogical
markers, assessment options, instructional artifacts, learning-objective
leaks, low-signal stopwords / fragments / HTML-entity contamination).
Edges referencing dropped nodes are also pruned. ``.bak`` snapshots are
taken before the in-place rewrite so the operation is reversible.

Defaults to the ``rdf-shacl-550-rdf-shacl-550`` archive that
ChatGPT's review flagged as 75% noise. Pass ``--course-dir`` to point
at any LibV2 course root.

This complements (and supersedes for the rdf-shacl-550 archive)
``scripts/wave75_classify_concept_graph.py``: Wave 75 stamped a
``class`` field on every node but kept the noisy ones in the graph.
Wave 76 now actually deletes them. Backups taken by Wave 76 are written
with the suffix ``.wave76.bak`` so they don't clobber the Wave 75 ones.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.ontology.concept_classifier import (  # noqa: E402
    DOMAIN_CONCEPT,
    canonicalize_alias,
    classify_concept,
    is_droppable_class,
)


DEFAULT_COURSE_DIR = (
    REPO_ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-550-rdf-shacl-550"
)

GRAPH_FILES = ("concept_graph.json", "concept_graph_semantic.json")
BACKUP_SUFFIX = ".wave76.bak"


def _bare_slug(node_id: str) -> str:
    """Strip a leading ``course_id:`` scope prefix.

    Mirrors the helper in ``wave75_classify_concept_graph.py`` so the
    classifier sees the bare slug it was designed against (under
    ``TRAINFORGE_SCOPE_CONCEPT_IDS=true`` IDs are
    ``f"{course_id}:{slug}"``).
    """
    if not isinstance(node_id, str):
        return ""
    return node_id.split(":", 1)[-1]


def _classify_node(node: Dict[str, Any]) -> str:
    """Compute the class for a single node using the Wave 76 ruleset."""
    return classify_concept(_bare_slug(node.get("id") or ""), label=node.get("label"))


def _prune_graph(
    graph_path: Path,
    *,
    dry_run: bool = False,
    backup_suffix: str = BACKUP_SUFFIX,
) -> Dict[str, Any]:
    """Prune droppable nodes + dangling edges from ``graph_path``.

    Returns a per-graph report with before/after counts and the
    breakdown of which classes the dropped nodes belonged to.
    """
    with graph_path.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes_in: List[Dict[str, Any]] = list(graph.get("nodes") or [])
    edges_in: List[Dict[str, Any]] = list(graph.get("edges") or [])

    # Stamp each node with the Wave 76 class for the report; partition
    # into kept vs dropped.
    kept_nodes: List[Dict[str, Any]] = []
    dropped_class_counts: Counter = Counter()
    dropped_examples: Dict[str, List[str]] = {}
    kept_class_counts: Counter = Counter()

    for node in nodes_in:
        klass = _classify_node(node)
        node["class"] = klass
        if is_droppable_class(klass):
            dropped_class_counts[klass] += 1
            dropped_examples.setdefault(klass, []).append(node.get("id", ""))
        else:
            kept_class_counts[klass] += 1
            kept_nodes.append(node)

    # Cap example lists so very-noisy classes don't dominate the report.
    for klass, ids in list(dropped_examples.items()):
        if len(ids) > 10:
            dropped_examples[klass] = ids[:10] + [f"... +{len(ids) - 10} more"]

    kept_ids = {n.get("id") for n in kept_nodes if n.get("id")}

    # Edge prune: drop any edge whose endpoint(s) reference a dropped
    # node. Apply alias canonicalization to endpoints first so a kept
    # alias survives even if a duplicate noisy slug was dropped.
    kept_edges: List[Dict[str, Any]] = []
    dropped_edge_count = 0
    for edge in edges_in:
        src = edge.get("source")
        tgt = edge.get("target")
        # Re-write any aliased endpoint onto its canonical form so
        # ``rdfxml`` survives as ``rdf-xml`` for example. Leave the bare
        # ID in place when no alias applies.
        if isinstance(src, str):
            edge["source"] = src.replace(
                _bare_slug(src), canonicalize_alias(_bare_slug(src))
            ) if _bare_slug(src) != canonicalize_alias(_bare_slug(src)) else src
        if isinstance(tgt, str):
            edge["target"] = tgt.replace(
                _bare_slug(tgt), canonicalize_alias(_bare_slug(tgt))
            ) if _bare_slug(tgt) != canonicalize_alias(_bare_slug(tgt)) else tgt
        if edge.get("source") in kept_ids and edge.get("target") in kept_ids:
            kept_edges.append(edge)
        else:
            dropped_edge_count += 1

    graph["nodes"] = kept_nodes
    graph["edges"] = kept_edges

    if not dry_run:
        bak = graph_path.with_suffix(graph_path.suffix + backup_suffix)
        if not bak.exists():
            shutil.copy2(graph_path, bak)
        with graph_path.open("w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return {
        "path": str(graph_path),
        "nodes_before": len(nodes_in),
        "nodes_after": len(kept_nodes),
        "edges_before": len(edges_in),
        "edges_after": len(kept_edges),
        "edges_dropped": dropped_edge_count,
        "dropped_class_counts": dict(dropped_class_counts),
        "kept_class_counts": dict(kept_class_counts),
        "dropped_examples": dropped_examples,
    }


def clean_course(
    course_dir: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run the cleanup across all graph files in ``course_dir/graph``."""
    graph_dir = course_dir / "graph"
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"Graph dir not found: {graph_dir}")

    reports: List[Dict[str, Any]] = []
    for filename in GRAPH_FILES:
        target = graph_dir / filename
        if not target.is_file():
            print(f"[skip] {target} not present", file=sys.stderr)
            continue
        report = _prune_graph(target, dry_run=dry_run)
        reports.append(report)
    return {"course_dir": str(course_dir), "graphs": reports}


def _print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== Wave 76 concept-graph cleanup: {report['course_dir']} ===")
    for graph_report in report["graphs"]:
        print(f"\n-- {graph_report['path']}")
        print(
            f"   nodes:   {graph_report['nodes_before']:>5d} -> {graph_report['nodes_after']:>5d}"
            f"  ({graph_report['nodes_before'] - graph_report['nodes_after']} dropped)"
        )
        print(
            f"   edges:   {graph_report['edges_before']:>5d} -> {graph_report['edges_after']:>5d}"
            f"  ({graph_report['edges_dropped']} dropped via dangling endpoint)"
        )

        print("   dropped node classes:")
        if graph_report["dropped_class_counts"]:
            for klass, count in sorted(
                graph_report["dropped_class_counts"].items(),
                key=lambda kv: (-kv[1], kv[0]),
            ):
                print(f"     {klass:<24s} {count}")
        else:
            print("     (none)")

        print("   kept node classes:")
        for klass, count in sorted(
            graph_report["kept_class_counts"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            print(f"     {klass:<24s} {count}")

        if graph_report["dropped_examples"]:
            print("   examples of dropped nodes:")
            for klass, ids in graph_report["dropped_examples"].items():
                print(f"     [{klass}]")
                for node_id in ids:
                    print(f"       - {node_id}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course-dir",
        type=Path,
        default=DEFAULT_COURSE_DIR,
        help=(
            "Path to a LibV2 course root (the directory containing graph/). "
            f"Default: {DEFAULT_COURSE_DIR}"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the report but do NOT write files.",
    )
    args = parser.parse_args(argv)

    report = clean_course(args.course_dir, dry_run=args.dry_run)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
