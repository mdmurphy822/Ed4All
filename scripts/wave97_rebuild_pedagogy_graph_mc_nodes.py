"""Wave 97 — surgical rebuild of pedagogy_graph.json Misconception nodes.

The Wave 95 audit revealed cross-artifact-join failure for `mc_*` IDs:

* ``Trainforge/pedagogy_graph_builder.py::_mc_id`` hashes ONLY the
  misconception text (legacy algorithm).
* ``Trainforge/process_course.py::_build_misconceptions_for_graph``
  hashes ``statement|correction|bloom_level`` (canonical Wave 69 / 72 /
  95 algorithm — matches ``concept_graph_semantic.json`` and
  ``Trainforge/generators/preference_factory._misconception_id``).

Result: ``pedagogy_graph.json`` Misconception node IDs and
``concept_graph_semantic.json`` Misconception IDs (and the
``misconception_id`` field stamped on synthesized DPO pairs) lived in
different namespaces. Joins across artifacts on ``mc_*`` failed
silently — the audit caught it on rdf-shacl-551-2 (34 ped mc nodes,
34 concept mc nodes, ZERO overlap).

The right long-term fix is to update ``pedagogy_graph_builder._mc_id``
to call the canonical algorithm directly. This script is the
SURGICAL fix for already-shipped corpora: it rewrites ONLY the
Misconception node IDs (and the corresponding ``interferes_with`` edge
``source`` references) in-place, leaving every other node, every
other edge type, and the per-relation edge counts byte-identical.

Usage::

    python scripts/wave97_rebuild_pedagogy_graph_mc_nodes.py \\
        --course rdf-shacl-551-2

Run from repo root. Writes:

* ``LibV2/courses/<course>/graph/pedagogy_graph.json.pre-wave97.bak`` —
  exact copy of the pre-rebuild file (only when not already present).
* ``LibV2/courses/<course>/graph/pedagogy_graph.json`` — atomically
  rewritten (tmpfile + rename).

Verification (printed to stdout):

* before/after Misconception node count (must match the chunk-derived
  canonical count).
* before/after edge-type counts per relation_type (must be byte-equal
  except that ``interferes_with`` edge ``source`` strings are
  rewritten).
* coverage: every mc_id derivable from the course chunks under the
  canonical algorithm appears as a node in the rebuilt graph.

No commits. No global side effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def canonical_mc_id(statement: str, correction: str, bloom_level: str) -> str:
    """Wave 69 / 72 / 95 canonical misconception ID.

    Mirrors ``Trainforge/process_course.py::_build_misconceptions_for_graph``
    and ``Trainforge/generators/preference_factory._misconception_id``.
    """

    statement = (statement or "").strip()
    correction = (correction or "").strip()
    bloom_level = (bloom_level or "").strip().lower()
    if bloom_level:
        seed = f"{statement}|{correction}|{bloom_level}"
    else:
        seed = f"{statement}|{correction}"
    return "mc_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]


def legacy_mc_id(text: str) -> str:
    """The pre-Wave-97 ``pedagogy_graph_builder._mc_id`` algorithm."""

    h = hashlib.sha256((text or "").strip().lower().encode("utf-8")).hexdigest()
    return f"mc_{h[:16]}"


def stream_chunk_misconceptions(
    chunks_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Walk chunks.jsonl streaming-style; emit a {legacy_id -> entry} map.

    Entry carries statement / correction / bloom_level so the caller can
    compute the canonical ID and union concept_tags from the chunks
    that mention each misconception.
    """

    out: Dict[str, Dict[str, Any]] = {}
    if not chunks_path.exists():
        raise FileNotFoundError(f"chunks.jsonl not found at {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            mcs = chunk.get("misconceptions") or []
            if not isinstance(mcs, list):
                continue
            for m in mcs:
                if isinstance(m, dict):
                    stmt = (m.get("misconception") or m.get("text") or "").strip()
                    corr = (m.get("correction") or "").strip()
                    bloom = (m.get("bloom_level") or "").strip().lower()
                elif isinstance(m, str):
                    stmt, corr, bloom = m.strip(), "", ""
                else:
                    continue
                if not stmt:
                    continue
                lid = legacy_mc_id(stmt)
                cid = canonical_mc_id(stmt, corr, bloom)
                if lid not in out:
                    out[lid] = {
                        "legacy_id": lid,
                        "canonical_id": cid,
                        "statement": stmt,
                        "correction": corr,
                        "bloom_level": bloom,
                    }
                else:
                    # Two chunks claim the same legacy_id; canonical_id
                    # could differ if they have different (correction,
                    # bloom_level). Keep the first-seen (matches
                    # _build_misconceptions_for_graph's first-seen-wins
                    # dedup behaviour).
                    pass
    return out


def rewrite_graph(
    graph: Dict[str, Any],
    legacy_to_canonical: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Return a new graph dict with mc_* node IDs + edges.source rewritten.

    Only ``Misconception`` nodes and ``interferes_with`` edges are
    touched. Every other node + edge is preserved verbatim. Edge
    sort/order is preserved (we mutate the source field in place on a
    deep-copied edges list).
    """

    new_nodes: List[Dict[str, Any]] = []
    rewrite_count = 0
    unmapped_node_ids: List[str] = []

    for node in graph.get("nodes", []):
        if node.get("class") == "Misconception":
            old_id = node.get("id")
            new_id = legacy_to_canonical.get(old_id)
            if new_id is None:
                # Misconception node we can't trace back to a chunk —
                # keep it verbatim. Could happen for nodes anchored only
                # by an interferes_with edge whose chunk later got
                # re-chunked. The audit explicitly notes this should be
                # vanishingly rare; we surface them in the report.
                unmapped_node_ids.append(old_id)
                new_nodes.append(dict(node))
                continue
            updated = dict(node)
            updated["id"] = new_id
            new_nodes.append(updated)
            rewrite_count += 1
        else:
            new_nodes.append(node)

    new_edges: List[Dict[str, Any]] = []
    edge_rewrite_count = 0
    for edge in graph.get("edges", []):
        if edge.get("relation_type") == "interferes_with":
            updated = dict(edge)
            old_source = updated.get("source")
            new_source = legacy_to_canonical.get(old_source, old_source)
            if new_source != old_source:
                edge_rewrite_count += 1
            updated["source"] = new_source
            new_edges.append(updated)
        else:
            new_edges.append(edge)

    new_graph = dict(graph)
    new_graph["nodes"] = new_nodes
    new_graph["edges"] = new_edges

    stats = {
        "nodes_rewritten": rewrite_count,
        "edges_rewritten": edge_rewrite_count,
        "unmapped_node_count": len(unmapped_node_ids),
    }
    if unmapped_node_ids:
        stats["unmapped_node_ids"] = unmapped_node_ids  # type: ignore[assignment]
    return new_graph, stats


def edge_type_counts(graph: Dict[str, Any]) -> Dict[str, int]:
    return dict(Counter(e.get("relation_type") for e in graph.get("edges", [])))


def write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """tmpfile + rename — never leaves a half-written file on disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--course",
        required=True,
        help="Course slug (e.g. rdf-shacl-551-2). Resolves under LibV2/courses/.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repo root (defaults to script's parent's parent).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print the diff but don't write the rebuilt graph.",
    )
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    course_dir = repo / "LibV2" / "courses" / args.course
    if not course_dir.exists():
        print(f"ERROR: course dir not found: {course_dir}", file=sys.stderr)
        return 2

    chunks_path = course_dir / "corpus" / "chunks.jsonl"
    graph_path = course_dir / "graph" / "pedagogy_graph.json"
    backup_path = graph_path.with_name(graph_path.name + ".pre-wave97.bak")

    if not chunks_path.exists():
        print(f"ERROR: missing {chunks_path}", file=sys.stderr)
        return 2
    if not graph_path.exists():
        print(f"ERROR: missing {graph_path}", file=sys.stderr)
        return 2

    print(f"course:   {args.course}")
    print(f"chunks:   {chunks_path}")
    print(f"graph:    {graph_path}")
    print(f"backup:   {backup_path}")
    print()

    # Build legacy_id -> canonical_id map from chunks.
    chunk_mcs = stream_chunk_misconceptions(chunks_path)
    legacy_to_canonical: Dict[str, str] = {
        e["legacy_id"]: e["canonical_id"] for e in chunk_mcs.values()
    }
    canonical_chunk_ids: Set[str] = {e["canonical_id"] for e in chunk_mcs.values()}
    legacy_chunk_ids: Set[str] = set(legacy_to_canonical.keys())

    # Read current graph.
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    before_mc_ids: Set[str] = {
        n["id"] for n in graph.get("nodes", []) if n.get("class") == "Misconception"
    }
    before_edge_counts = edge_type_counts(graph)

    print(f"before: misconception nodes = {len(before_mc_ids)}")
    print(f"before: edge_types ({len(before_edge_counts)}) = {before_edge_counts}")
    print()
    print(f"chunk-derived canonical mc IDs: {len(canonical_chunk_ids)}")
    print(f"chunk-derived legacy mc IDs:    {len(legacy_chunk_ids)}")
    print(
        "before-graph ∩ chunk-legacy: "
        f"{len(before_mc_ids & legacy_chunk_ids)} (=should equal before count)"
    )
    print(
        "before-graph ∩ chunk-canonical: "
        f"{len(before_mc_ids & canonical_chunk_ids)} (=should be 0 — the bug)"
    )
    print()

    new_graph, rewrite_stats = rewrite_graph(graph, legacy_to_canonical)
    after_mc_ids: Set[str] = {
        n["id"] for n in new_graph.get("nodes", []) if n.get("class") == "Misconception"
    }
    after_edge_counts = edge_type_counts(new_graph)

    print(f"after:  misconception nodes = {len(after_mc_ids)}")
    print(f"after:  edge_types ({len(after_edge_counts)}) = {after_edge_counts}")
    print()
    print(f"rewrite stats: {rewrite_stats}")
    print()
    print(
        "after ∩ chunk-canonical: "
        f"{len(after_mc_ids & canonical_chunk_ids)} (target: full coverage)"
    )
    missing = canonical_chunk_ids - after_mc_ids
    extra = after_mc_ids - canonical_chunk_ids
    print(f"chunk-canonical missing from rebuilt graph: {len(missing)}")
    print(f"rebuilt-graph nodes not in chunk-canonical: {len(extra)}")
    if missing:
        print(f"  missing samples: {sorted(missing)[:5]}")
    if extra:
        print(f"  extra samples:   {sorted(extra)[:5]}")
    print()

    # Edge-type counts must match exactly across all relation types.
    if before_edge_counts != after_edge_counts:
        print(
            "ERROR: edge-type counts diverged — refusing to write.",
            file=sys.stderr,
        )
        for k in set(before_edge_counts) | set(after_edge_counts):
            if before_edge_counts.get(k) != after_edge_counts.get(k):
                print(
                    f"  {k}: {before_edge_counts.get(k)} -> {after_edge_counts.get(k)}",
                    file=sys.stderr,
                )
        return 3

    if args.dry_run:
        print("dry-run: no write")
        return 0

    # Backup once (don't clobber an existing pre-wave97 backup).
    if not backup_path.exists():
        shutil.copy2(graph_path, backup_path)
        print(f"backup written: {backup_path}")
    else:
        print(f"backup already exists, leaving in place: {backup_path}")

    write_atomic(graph_path, new_graph)
    print(f"rewrote: {graph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
