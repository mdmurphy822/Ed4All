#!/usr/bin/env python3
"""Wave 75: retroactively regenerate objectives.json + normalize chunk refs.

ChatGPT's review of the RDF_SHACL_550 knowledge package flagged 312
unresolvable ``learning_outcome_refs`` because chunks reference both
terminal (``to-*``) AND component (``co-*``) objectives, but
``course.json`` only declared the 7 terminal outcomes — the 29
component objectives existed in
``synthesized_objectives.json`` but never propagated to the LibV2
archive.

This script fixes the existing archive in place:

1. Reads
   ``Courseforge/exports/<project>/01_learning_objectives/synthesized_objectives.json``
2. Emits ``LibV2/courses/<slug>/objectives.json`` (canonical Wave-75 shape).
3. Updates ``LibV2/courses/<slug>/course.json`` so
   ``learning_outcomes[]`` includes the COs (with ``hierarchy_level=chapter``
   + ``type=component``).
4. Walks ``LibV2/courses/<slug>/corpus/chunks.jsonl`` and rewrites it,
   normalizing any comma-delimited
   ``learning_outcome_refs`` (e.g. ``"co-01,co-02,co-03"`` → three
   refs). A ``chunks.jsonl.bak`` is written first as a safety net.
5. Prints a report: chunks scanned, comma-refs split, total refs
   before/after, broken refs remaining.

Defaults target the ``rdf-shacl-550-rdf-shacl-550`` archive flagged
in the review. Pass ``--archive-dir`` / ``--objectives-source`` to
point at any other course.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make sure the project root is importable so we can reuse the helper.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Trainforge.process_course import normalize_outcome_refs  # noqa: E402

DEFAULT_ARCHIVE = (
    REPO_ROOT / "LibV2" / "courses" / "rdf-shacl-550-rdf-shacl-550"
)
DEFAULT_OBJECTIVES = (
    REPO_ROOT
    / "Courseforge"
    / "exports"
    / "PROJ-RDF_SHACL_550-20260424135037"
    / "01_learning_objectives"
    / "synthesized_objectives.json"
)


def _id(raw: str) -> str:
    return raw.lower().strip()


def build_objectives_json(synth: Dict[str, Any], course_code: str) -> Dict[str, Any]:
    """Translate synthesized_objectives.json → Wave-75 objectives.json shape."""
    terminal_outcomes: List[Dict[str, Any]] = []
    for to in synth.get("terminal_objectives", []):
        if not isinstance(to, dict) or "id" not in to:
            continue
        entry: Dict[str, Any] = {
            "id": _id(to["id"]),
            "statement": to.get("statement") or to.get("text") or "",
        }
        if to.get("bloom_level") or to.get("bloomLevel"):
            entry["bloom_level"] = (
                to.get("bloom_level") or to.get("bloomLevel")
            )
        if to.get("bloom_verb"):
            entry["bloom_verb"] = to["bloom_verb"]
        if to.get("cognitive_domain"):
            entry["cognitive_domain"] = to["cognitive_domain"]
        if to.get("weeks"):
            entry["weeks"] = list(to["weeks"])
        terminal_outcomes.append(entry)

    component_objectives: List[Dict[str, Any]] = []
    for ch in synth.get("chapter_objectives", []):
        if isinstance(ch, dict) and "objectives" in ch:
            inner = ch.get("objectives") or []
        else:
            inner = [ch]
        for obj in inner:
            if not isinstance(obj, dict) or "id" not in obj:
                continue
            entry = {
                "id": _id(obj["id"]),
                "statement": obj.get("statement") or obj.get("text") or "",
            }
            parent = obj.get("parent_to") or obj.get("parent_terminal")
            if parent:
                entry["parent_terminal"] = _id(parent)
            if obj.get("bloom_level") or obj.get("bloomLevel"):
                entry["bloom_level"] = (
                    obj.get("bloom_level") or obj.get("bloomLevel")
                )
            if obj.get("bloom_verb"):
                entry["bloom_verb"] = obj["bloom_verb"]
            if obj.get("cognitive_domain"):
                entry["cognitive_domain"] = obj["cognitive_domain"]
            if obj.get("week") is not None:
                entry["week"] = obj["week"]
            if obj.get("source_refs"):
                entry["source_refs"] = list(obj["source_refs"])
            component_objectives.append(entry)

    return {
        "schema_version": "v1",
        "course_code": course_code,
        "terminal_outcomes": terminal_outcomes,
        "component_objectives": component_objectives,
        "objective_count": {
            "terminal": len(terminal_outcomes),
            "component": len(component_objectives),
        },
    }


def update_course_json(
    course_data: Dict[str, Any], objectives: Dict[str, Any]
) -> Dict[str, Any]:
    """Rebuild course.json::learning_outcomes from the objectives doc.

    Preserves any extra fields already on course_data. Drops any
    legacy ``learning_outcomes[]`` content and replaces it with
    terminal-first, component-second flat list (matching the
    Trainforge ``_build_course_json`` emit).
    """
    flat: List[Dict[str, Any]] = []
    for to in objectives.get("terminal_outcomes", []):
        entry = {
            "id": to["id"],
            "statement": to["statement"],
            "hierarchy_level": "terminal",
        }
        if to.get("bloom_level"):
            entry["bloom_level"] = to["bloom_level"]
        flat.append(entry)
    for co in objectives.get("component_objectives", []):
        entry = {
            "id": co["id"],
            "statement": co["statement"],
            "hierarchy_level": "chapter",
            "type": "component",
        }
        if co.get("bloom_level"):
            entry["bloom_level"] = co["bloom_level"]
        flat.append(entry)

    course_data = dict(course_data)
    course_data["learning_outcomes"] = flat
    # If course.json had a Wave-30 ``note`` indicating empty LOs, drop it.
    if "note" in course_data and not flat:
        pass
    elif "note" in course_data and flat:
        course_data.pop("note")
    return course_data


def normalize_chunks_jsonl(
    chunks_path: Path,
) -> Dict[str, Any]:
    """Rewrite chunks.jsonl normalizing comma-delimited refs.

    Returns a report dict: {scanned, comma_refs_split, before_total,
    after_total, refs_seen}.
    """
    bak_path = chunks_path.with_suffix(chunks_path.suffix + ".bak")
    if not bak_path.exists():
        shutil.copy(chunks_path, bak_path)

    scanned = 0
    comma_refs_split = 0
    before_total = 0
    after_total = 0
    refs_seen: Set[str] = set()

    out_lines: List[str] = []
    with chunks_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            chunk = json.loads(line)
            scanned += 1
            raw = chunk.get("learning_outcome_refs") or []
            before_total += len(raw)
            had_comma = any(
                isinstance(r, str) and "," in r for r in raw
            )
            normed = normalize_outcome_refs(raw)
            if had_comma:
                # Count the extra refs that the split produced.
                # before split: N entries with K comma items expand to
                # K-N additional refs (roughly). Count actual delta.
                comma_refs_split += sum(
                    len([p for p in r.split(",") if p.strip()]) - 1
                    for r in raw
                    if isinstance(r, str) and "," in r
                )
            chunk["learning_outcome_refs"] = normed
            after_total += len(normed)
            for r in normed:
                refs_seen.add(r)
            out_lines.append(json.dumps(chunk, ensure_ascii=False))

    with chunks_path.open("w", encoding="utf-8") as fh:
        for line in out_lines:
            fh.write(line + "\n")

    return {
        "scanned": scanned,
        "comma_refs_split": comma_refs_split,
        "before_total": before_total,
        "after_total": after_total,
        "refs_seen": refs_seen,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="Path to LibV2 course archive directory.",
    )
    p.add_argument(
        "--objectives-source",
        type=Path,
        default=DEFAULT_OBJECTIVES,
        help="Path to Courseforge synthesized_objectives.json.",
    )
    p.add_argument(
        "--course-code",
        default=None,
        help="Override the course_code emitted into objectives.json. "
        "Defaults to the value already on course.json.",
    )
    args = p.parse_args()

    archive_dir: Path = args.archive_dir
    obj_src: Path = args.objectives_source

    if not archive_dir.exists():
        print(f"ERROR: archive dir not found: {archive_dir}", file=sys.stderr)
        return 1
    if not obj_src.exists():
        print(f"ERROR: objectives source not found: {obj_src}", file=sys.stderr)
        return 1

    course_json_path = archive_dir / "course.json"
    chunks_path = archive_dir / "corpus" / "chunks.jsonl"

    if not course_json_path.exists():
        print(f"ERROR: missing course.json under {archive_dir}", file=sys.stderr)
        return 1
    if not chunks_path.exists():
        print(f"ERROR: missing corpus/chunks.jsonl under {archive_dir}", file=sys.stderr)
        return 1

    with course_json_path.open(encoding="utf-8") as fh:
        course_data = json.load(fh)
    course_code = args.course_code or course_data.get("course_code", "UNKNOWN")

    with obj_src.open(encoding="utf-8") as fh:
        synth = json.load(fh)

    # 1. objectives.json
    objectives = build_objectives_json(synth, course_code)
    objectives_path = archive_dir / "objectives.json"
    with objectives_path.open("w", encoding="utf-8") as fh:
        json.dump(objectives, fh, indent=2, ensure_ascii=False)
    print(
        f"[objectives.json] wrote {objectives_path} — "
        f"{objectives['objective_count']['terminal']} TO + "
        f"{objectives['objective_count']['component']} CO"
    )

    # 2. course.json (rebuild learning_outcomes)
    new_course_data = update_course_json(course_data, objectives)
    with course_json_path.open("w", encoding="utf-8") as fh:
        json.dump(new_course_data, fh, indent=2, ensure_ascii=False)
    print(
        f"[course.json] updated {course_json_path} — "
        f"now {len(new_course_data['learning_outcomes'])} LOs"
    )

    # 3. chunks.jsonl normalization
    report = normalize_chunks_jsonl(chunks_path)
    print(
        f"[chunks.jsonl] scanned {report['scanned']} chunks, "
        f"split {report['comma_refs_split']} comma-refs, "
        f"refs before={report['before_total']} after={report['after_total']}"
    )

    # 4. Resolution check.
    valid_ids: Set[str] = set()
    for to in objectives["terminal_outcomes"]:
        valid_ids.add(to["id"])
    for co in objectives["component_objectives"]:
        valid_ids.add(co["id"])

    refs_seen = report["refs_seen"]
    broken = sorted(r for r in refs_seen if r not in valid_ids)

    print(
        f"[resolution] {len(refs_seen)} distinct refs across chunks; "
        f"{len(refs_seen) - len(broken)} resolve, {len(broken)} broken"
    )
    if broken:
        print("  broken refs (NOT in objectives.json):")
        for r in broken:
            print(f"    - {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
