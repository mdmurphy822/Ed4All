#!/usr/bin/env python3
"""Wave 76 — retroactively regenerate quality_report.json + manifest
validation_errors for an existing LibV2 archive.

Why this exists: three validator bugs (see ``Wave 76: fix three
validator bugs masking real package quality``) caused
``LibV2/courses/rdf-shacl-550-rdf-shacl-550`` to be tagged with 312
broken_refs (only 1 was real), an "Unknown domain 'computer science'"
error, and an "outcomes < 10" error. This script reuses the
post-fix validators / quality scorer to overwrite the cached reports
in place so a downstream consumer (LibV2 retrieval, dataset filters)
sees the corrected truth without re-running the whole pipeline.

Targets ``rdf-shacl-550-rdf-shacl-550`` by default; point at any
other archive via ``--archive-dir``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LibV2.tools.libv2.validator import (  # noqa: E402
    validate_course_strict,
)
from Trainforge.process_course import (  # noqa: E402
    CourseProcessor,
    load_objectives,
)

DEFAULT_ARCHIVE = (
    REPO_ROOT / "LibV2" / "courses" / "rdf-shacl-550-rdf-shacl-550"
)


def _load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _build_processor_for_archive(archive: Path) -> CourseProcessor:
    """Construct a CourseProcessor stub wired only with the bits the
    quality-report path needs.

    We deliberately bypass ``__init__`` because the full constructor
    expects an .imscc input. ``_generate_quality_report`` only reads
    a small set of attributes; the rest are nullable / unused for
    after-the-fact regeneration.
    """
    cp = CourseProcessor.__new__(CourseProcessor)

    # Wire the minimal attribute surface that _generate_quality_report
    # touches via its statics + helpers.
    cp.output_dir = archive
    cp.course_code = archive.name.upper()
    cp.MIN_CHUNK_SIZE = 100
    cp.MAX_CHUNK_SIZE = 800

    # Load objectives (Wave 75 shape) for ID resolution. Fall back
    # gracefully if absent — the Wave 76 fix walks course.json too.
    objectives_path = archive / "objectives.json"
    cp.objectives = (
        load_objectives(objectives_path) if objectives_path.exists() else {}
    )

    # Empty caches — these populate during a normal run, but for a
    # retroactive regen we only need _build_valid_outcome_ids and the
    # ref-resolution paths.
    cp._boilerplate_spans = []
    cp._factual_flags = []
    cp.stats = {"total_words": 0, "total_chunks": 0}
    cp.strict_mode = False
    cp._valid_outcome_ids = cp._build_valid_outcome_ids()

    return cp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help="LibV2 course archive to regenerate (default: rdf-shacl-550)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print before/after deltas without overwriting any file.",
    )
    args = parser.parse_args()

    archive: Path = args.archive_dir
    if not archive.exists():
        print(f"ERROR: archive not found: {archive}", file=sys.stderr)
        return 2

    chunks_path = archive / "corpus" / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"ERROR: corpus/chunks.jsonl not found at {chunks_path}", file=sys.stderr)
        return 2

    chunks = _load_chunks_jsonl(chunks_path)

    # ------------------------------------------------------------------
    # 1. quality_report.json
    # ------------------------------------------------------------------
    cp = _build_processor_for_archive(archive)

    # Word counts — needed for chunk-size compliance.
    for c in chunks:
        if "word_count" not in c:
            html = c.get("html") or c.get("text") or ""
            c["word_count"] = max(len(html.split()), 1)
    cp.stats["total_words"] = sum(c.get("word_count", 0) for c in chunks)
    cp.stats["total_chunks"] = len(chunks)

    quality_report = cp._generate_quality_report(chunks)

    quality_dir = archive / "quality"
    quality_dir.mkdir(exist_ok=True)
    quality_path = quality_dir / "quality_report.json"

    # Capture before metrics for the report
    before_quality: Dict[str, Any] = {}
    if quality_path.exists():
        try:
            before_quality = json.loads(quality_path.read_text())
        except json.JSONDecodeError:
            pass

    # ------------------------------------------------------------------
    # 2. manifest.json::quality_metadata.validation_errors
    # ------------------------------------------------------------------
    manifest_path = archive / "manifest.json"
    manifest_before = json.loads(manifest_path.read_text())

    val_result = validate_course_strict(archive, REPO_ROOT)
    new_quality_metadata = {
        "validation_status": "validated" if val_result.valid else "failed",
        "last_validated": datetime.now().isoformat(),
        "validation_errors": list(val_result.errors),
        "validation_warnings": list(val_result.warnings),
    }

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    before_broken = len(before_quality.get("integrity", {}).get("broken_refs", []))
    after_broken = len(quality_report.get("integrity", {}).get("broken_refs", []))
    before_lo_cov = before_quality.get("metrics", {}).get("learning_outcome_coverage", 0.0)
    after_lo_cov = quality_report.get("metrics", {}).get("learning_outcome_coverage", 0.0)
    before_status = before_quality.get("validation", {}).get("passed", "unknown")
    after_status = quality_report.get("validation", {}).get("passed", "unknown")
    before_score = before_quality.get("overall_quality_score", 0.0)
    after_score = quality_report.get("overall_quality_score", 0.0)

    before_manifest_errors = (
        manifest_before.get("quality_metadata", {}).get("validation_errors", [])
    )

    print(f"Archive: {archive.name}")
    print(f"  quality_report.json")
    print(f"    broken_refs:               {before_broken} -> {after_broken}")
    print(f"    learning_outcome_coverage: {before_lo_cov} -> {after_lo_cov}")
    print(f"    validation.passed:         {before_status} -> {after_status}")
    print(f"    overall_quality_score:     {before_score} -> {after_score}")
    print(f"  manifest.json::validation_errors")
    for e in before_manifest_errors:
        print(f"    - (was) {e}")
    for e in val_result.errors:
        print(f"    - (now) {e}")
    if not val_result.errors:
        print("    (no validation errors)")

    if args.dry_run:
        print("\n[dry-run] no files written")
        return 0

    # Write quality_report.json
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality_report, f, indent=2)
    print(f"\nWrote {quality_path}")

    # Update manifest
    manifest_after = dict(manifest_before)
    manifest_after["quality_metadata"] = new_quality_metadata
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_after, f, indent=2)
    print(f"Wrote {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
