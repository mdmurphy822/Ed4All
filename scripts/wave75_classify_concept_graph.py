#!/usr/bin/env python3
"""Wave 75: retroactively classify concept-graph nodes.

Reads existing ``concept_graph.json`` + ``concept_graph_semantic.json``
under a course's ``graph/`` directory, stamps each node with a
``class`` value via :func:`lib.ontology.concept_classifier.classify_concept`,
and writes the updated graphs in place. ``.bak`` snapshots are taken
first so the operation is reversible.

Defaults to the ``rdf-shacl-550-rdf-shacl-550`` archive that
ChatGPT's review flagged. Pass ``--course-dir`` to point at any
LibV2 course root.

The script also emits a small report to stdout: counts per class,
plus the list of nodes that were previously masquerading as domain
concepts (PedagogicalMarker / LowSignal / AssessmentOption /
InstructionalArtifact / LearningObjective).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.ontology.concept_classifier import (  # noqa: E402
    DOMAIN_CONCEPT,
    classify_concept,
)


DEFAULT_COURSE_DIR = (
    REPO_ROOT
    / "LibV2"
    / "courses"
    / "rdf-shacl-550-rdf-shacl-550"
)

GRAPH_FILES = ("concept_graph.json", "concept_graph_semantic.json")


def _classify_node(node: Dict[str, Any]) -> str:
    """Compute the class for a single node.

    Strips a leading ``course_id:`` scope prefix (per
    ``TRAINFORGE_SCOPE_CONCEPT_IDS``) before classification so the
    classifier sees the bare slug it was designed against.
    """
    raw_id = node.get("id") or ""
    slug = raw_id.split(":", 1)[-1] if isinstance(raw_id, str) else ""
    return classify_concept(slug, label=node.get("label"))


def _stamp_classes(graph_path: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Add ``class`` to every node in ``graph_path``.

    Returns a per-graph report dict with class counts and the list of
    non-domain nodes that were previously bare domain concepts.
    """
    with graph_path.open("r", encoding="utf-8") as f:
        graph = json.load(f)

    nodes: List[Dict[str, Any]] = graph.get("nodes") or []
    counts: Counter = Counter()
    reclassified_non_domain: List[Dict[str, str]] = []

    for node in nodes:
        new_class = _classify_node(node)
        old_class = node.get("class")
        node["class"] = new_class
        counts[new_class] += 1
        if new_class != DOMAIN_CONCEPT and (old_class is None or old_class == DOMAIN_CONCEPT):
            reclassified_non_domain.append({
                "id": node.get("id", ""),
                "label": node.get("label", ""),
                "class": new_class,
            })

    if not dry_run:
        bak = graph_path.with_suffix(graph_path.suffix + ".bak")
        if not bak.exists():
            shutil.copy2(graph_path, bak)
        with graph_path.open("w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2, ensure_ascii=False)
            f.write("\n")

    return {
        "path": str(graph_path),
        "node_count": len(nodes),
        "class_counts": dict(counts),
        "reclassified_non_domain": reclassified_non_domain,
    }


def regen_course(course_dir: Path, dry_run: bool = False) -> Dict[str, Any]:
    """Run the retroactive classification across both graph files."""
    graph_dir = course_dir / "graph"
    if not graph_dir.is_dir():
        raise FileNotFoundError(f"Graph dir not found: {graph_dir}")

    reports: List[Dict[str, Any]] = []
    for filename in GRAPH_FILES:
        target = graph_dir / filename
        if not target.is_file():
            print(f"[skip] {target} not present", file=sys.stderr)
            continue
        report = _stamp_classes(target, dry_run=dry_run)
        reports.append(report)
    return {"course_dir": str(course_dir), "graphs": reports}


def _print_report(report: Dict[str, Any]) -> None:
    print(f"\n=== Wave 75 concept-graph classification: {report['course_dir']} ===")
    for graph_report in report["graphs"]:
        print(f"\n-- {graph_report['path']}")
        print(f"   nodes: {graph_report['node_count']}")
        print("   class counts:")
        for klass, count in sorted(
            graph_report["class_counts"].items(),
            key=lambda kv: (-kv[1], kv[0]),
        ):
            print(f"     {klass:<24s} {count}")

        non_domain = graph_report["reclassified_non_domain"]
        if non_domain:
            print(f"   nodes reclassified out of domain space: {len(non_domain)}")
            for entry in non_domain:
                print(f"     [{entry['class']}] {entry['id']} ({entry['label']})")
        else:
            print("   nodes reclassified out of domain space: 0")


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

    report = regen_course(args.course_dir, dry_run=args.dry_run)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
