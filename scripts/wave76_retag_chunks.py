#!/usr/bin/env python3
"""Wave 76: retroactively retag chunks for coverage gaps.

Reads ``LibV2/courses/<slug>/corpus/chunks.jsonl`` and applies the
vocabulary retag + parent-outcome rollup defined in
``Trainforge/retag_outcomes.py``. Produces .bak backups, then writes
both ``chunks.jsonl`` and the materialized ``chunks.json`` so the two
files stay in sync at commit time.

Default target: rdf-shacl-550-rdf-shacl-550 (the archive flagged by
the external KG-quality review). Pass ``--course-slug <slug>`` to run
against another LibV2 archive — the same retag rules are
content-addressed, so the script is generic.

Usage::

    python -m scripts.wave76_retag_chunks \
        [--course-slug rdf-shacl-550-rdf-shacl-550] \
        [--libv2-root LibV2]

Output (always to stderr / stdout):

  * Per-CO retag count (chunks newly tagged by the vocabulary rule).
  * Parent-rollup count (chunks newly carrying the terminal parent).
  * Before/after coverage for every CO and TO id present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Make the project root importable when running as `python scripts/...`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.retag_outcomes import (  # noqa: E402  (after sys.path tweak)
    RETAG_VOCABULARIES,
    build_parent_map,
    retag_chunk_outcomes,
)


def _coverage(chunks: List[Dict[str, Any]]) -> Counter:
    counts: Counter = Counter()
    for c in chunks:
        for ref in c.get("learning_outcome_refs") or []:
            counts[ref] += 1
    return counts


def _load_chunks(jsonl_path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _write_jsonl(jsonl_path: Path, chunks: List[Dict[str, Any]]) -> None:
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(c, ensure_ascii=False))
            fh.write("\n")


def _write_json_array(json_path: Path, chunks: List[Dict[str, Any]]) -> None:
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False, indent=2)


def retag_archive(
    course_dir: Path,
) -> Dict[str, Any]:
    """Apply retag pass to a single LibV2 archive.

    Returns a report dict with before/after coverage and per-CO add
    counts so callers can ship a structured commit-message body.
    """
    corpus_dir = course_dir / "corpus"
    jsonl_path = corpus_dir / "chunks.jsonl"
    json_path = corpus_dir / "chunks.json"
    objectives_path = course_dir / "objectives.json"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"chunks.jsonl missing: {jsonl_path}")
    if not objectives_path.exists():
        raise FileNotFoundError(
            f"objectives.json missing: {objectives_path}"
        )

    objectives = json.loads(objectives_path.read_text(encoding="utf-8"))
    parent_map = build_parent_map(objectives)

    # Load + snapshot before retag.
    chunks = _load_chunks(jsonl_path)
    before = _coverage(chunks)

    # Per-CO add counts (additions only; never decrements).
    co_add_counts: Counter = Counter()
    parent_add_counts: Counter = Counter()
    chunks_changed = 0

    for chunk in chunks:
        existing = {
            r.lower() for r in (chunk.get("learning_outcome_refs") or [])
            if isinstance(r, str)
        }
        retag_chunk_outcomes(chunk, parent_map=parent_map)
        new_refs = {
            r.lower() for r in (chunk.get("learning_outcome_refs") or [])
            if isinstance(r, str)
        }
        added = new_refs - existing
        if added:
            chunks_changed += 1
        for a in added:
            if a in RETAG_VOCABULARIES:
                co_add_counts[a] += 1
            elif a.startswith("to-"):
                parent_add_counts[a] += 1
            else:
                # Component IDs added via parent-rollup chain (rare —
                # parents shouldn't pick up children) get tracked too.
                co_add_counts[a] += 1

    after = _coverage(chunks)

    # Backup .bak before writing. Keep a single .bak per file
    # (overwrite on re-run) so repeated invocations don't pile up.
    shutil.copyfile(jsonl_path, jsonl_path.with_suffix(".jsonl.bak"))
    if json_path.exists():
        shutil.copyfile(json_path, json_path.with_suffix(".json.bak"))

    _write_jsonl(jsonl_path, chunks)
    _write_json_array(json_path, chunks)

    return {
        "course_dir": str(course_dir),
        "chunks_total": len(chunks),
        "chunks_changed": chunks_changed,
        "co_add_counts": dict(co_add_counts),
        "parent_add_counts": dict(parent_add_counts),
        "before": dict(before),
        "after": dict(after),
    }


def _format_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"course_dir: {report['course_dir']}")
    lines.append(f"chunks_total: {report['chunks_total']}")
    lines.append(f"chunks_changed: {report['chunks_changed']}")
    lines.append("")
    lines.append("Per-CO retag (vocabulary rule):")
    for co_id, n in sorted(report["co_add_counts"].items()):
        lines.append(f"  {co_id}: +{n}")
    lines.append("")
    lines.append("Parent-rollup additions:")
    for to_id, n in sorted(report["parent_add_counts"].items()):
        lines.append(f"  {to_id}: +{n}")
    lines.append("")
    lines.append("Coverage before -> after (all refs):")
    keys = sorted(set(report["before"]) | set(report["after"]))
    for k in keys:
        b = report["before"].get(k, 0)
        a = report["after"].get(k, 0)
        marker = " *" if b != a else ""
        lines.append(f"  {k}: {b} -> {a}{marker}")
    return "\n".join(lines)


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Wave 76: retag LibV2 chunks for coverage gaps.",
    )
    parser.add_argument(
        "--course-slug",
        default="rdf-shacl-550-rdf-shacl-550",
        help="LibV2 course slug to retag (default: rdf-shacl-550-rdf-shacl-550).",
    )
    parser.add_argument(
        "--libv2-root",
        default=str(PROJECT_ROOT / "LibV2"),
        help="LibV2 repository root (default: <project_root>/LibV2).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON instead of plain text.",
    )
    args = parser.parse_args(argv)

    course_dir = Path(args.libv2_root) / "courses" / args.course_slug
    if not course_dir.exists():
        print(f"course not found: {course_dir}", file=sys.stderr)
        return 2

    report = retag_archive(course_dir)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
